"""Morning EMA Cross + Gamma direction filter, ATR-fixed SL.

Filter logic (applied at entry decision, using PRIOR day's gamma — no lookahead):
  - If yesterday's qqq_gamma_sign == +1 (pos gamma): only allow LONG entries
  - If yesterday's qqq_gamma_sign == -1 (neg gamma): only allow SHORT entries
  - If no gamma data available: skip entry (no trade)

The gamma_sign for date X is computed from X's 17:15 ET settle data.
For a trade ON date X at 10 AM, we use gamma from date X-1 (yesterday's settle).
Implementation: asof-backward merge with allow_exact_matches=False AND a
tolerance of 5 days (to accommodate weekends/holidays).

Sweep is NOT done here — we run the 2 top configs from the SL sweep:
  - Risk-adjusted: EMA=60, ATR=28, SL_mult=1.25
  - Absolute $:    EMA=60, ATR=10, SL_mult=2.75

Reports: PF, MDD, total trades, yearly breakdown, IS/OOS split.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from strategy_atr_sl import build_30min_bars, NQ_PT

PARQUET = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
GAMMA_PATH = "D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet"
ET = "America/New_York"
OUT_DIR = Path(__file__).parent / "results"


def load_gamma_lookup() -> dict:
    """Returns dict: session_date (date obj) -> {prior_gamma_sign}.
    'prior' = the previous business-day gamma_sign, since gamma for date X
    isn't available until 17:15 ET on X (after our 10 AM entry on X).
    """
    g = pd.read_parquet(GAMMA_PATH)
    g["date"] = pd.to_datetime(g["date"]).dt.date
    g = g[["date", "qqq_gamma_sign"]].dropna(subset=["qqq_gamma_sign"]).sort_values("date")
    g = g.reset_index(drop=True)

    # Map: for each session date, the PRIOR available gamma_sign
    # Iterate through sorted gamma rows. For trade on date X, use the row
    # whose date is the most recent <= X-1.
    out = {}
    sorted_g = list(zip(g["date"].tolist(), g["qqq_gamma_sign"].tolist()))
    # Build a sorted array of (date, sign) tuples
    # For each unique session date that might trade, find prior gamma
    return sorted_g  # we'll do the lookup inline


def prior_gamma_sign(trade_date, gamma_pairs: list, max_stale_days: int = 5):
    """Return the most recent gamma_sign with date < trade_date, within max_stale_days."""
    # Binary search would be faster; linear is fine for our scale
    result = None
    result_date = None
    for d, s in gamma_pairs:
        if d < trade_date:
            result = s
            result_date = d
        else:
            break
    if result is None:
        return None
    if (trade_date - result_date).days > max_stale_days:
        return None
    return result


def compute_indicators(bars: pd.DataFrame, ema_n: int, atr_n: int) -> pd.DataFrame:
    b = bars.copy()
    b["ema"] = b["close"].ewm(span=ema_n, adjust=False).mean()
    prev_close = b["close"].shift(1)
    tr = pd.concat([
        b["high"] - b["low"],
        (b["high"] - prev_close).abs(),
        (b["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    b["atr"] = tr.ewm(alpha=1.0/atr_n, adjust=False).mean()
    b["hhmm"] = b.index.hour * 100 + b.index.minute
    b["dow"] = b.index.dayofweek
    b["date"] = b.index.date
    return b


def run_strategy_with_gamma_filter(b: pd.DataFrame, sl_mult: float,
                                    gamma_pairs: list,
                                    use_marti: bool = True,
                                    base_qty: int = 1, loss_qty: int = 2) -> pd.DataFrame:
    """Re-simulate with gamma-direction filter at entry."""
    trades = []
    marti_state = 0
    skipped_filter = {"no_gamma": 0, "wrong_dir_pos": 0, "wrong_dir_neg": 0}

    for date, day in b.groupby("date"):
        if day["dow"].iloc[0] >= 5: continue
        entry_bar = day[day["hhmm"] == 1000]
        if len(entry_bar) == 0: continue
        eb = entry_bar.iloc[0]
        if pd.isna(eb["ema"]) or pd.isna(eb["atr"]) or eb["atr"] <= 0:
            continue

        entry_price = float(eb["close"])
        entry_atr   = float(eb["atr"])
        # Direction from EMA cross
        if entry_price > eb["ema"]:   ema_sign = 1
        elif entry_price < eb["ema"]: ema_sign = -1
        else: continue

        # Gamma filter — uses PRIOR day's gamma (no lookahead)
        prior_g = prior_gamma_sign(date, gamma_pairs)
        if prior_g is None:
            skipped_filter["no_gamma"] += 1
            continue
        # Direction filter: only LONG on pos gamma days, only SHORT on neg gamma days
        if prior_g > 0 and ema_sign != 1:
            skipped_filter["wrong_dir_pos"] += 1
            continue
        if prior_g < 0 and ema_sign != -1:
            skipped_filter["wrong_dir_neg"] += 1
            continue

        sign = ema_sign
        qty = loss_qty if (use_marti and marti_state == 1) else base_qty
        sl_price = entry_price - sign * sl_mult * entry_atr

        post = day[(day["hhmm"] > 1000) & (day["hhmm"] <= 1430)]
        if len(post) == 0: continue

        exit_price = None; exit_reason = None; exit_time = None; bars_held = 0
        for ts, bar in post.iterrows():
            bars_held += 1
            high = float(bar["high"]); low = float(bar["low"]); close = float(bar["close"])
            if int(bar["hhmm"]) == 1430:
                exit_price = close; exit_reason = "FC"; exit_time = ts; break
            if sign > 0 and low <= sl_price:
                exit_price = sl_price; exit_reason = "SL"; exit_time = ts; break
            if sign < 0 and high >= sl_price:
                exit_price = sl_price; exit_reason = "SL"; exit_time = ts; break

        if exit_price is None: continue
        pnl_pts = sign * (exit_price - entry_price)
        pnl_usd = pnl_pts * NQ_PT * qty
        trades.append({
            "date": date, "entry_time": eb.name, "exit_time": exit_time,
            "direction": "LONG" if sign > 0 else "SHORT",
            "prior_gamma_sign": prior_g,
            "entry_price": entry_price, "exit_price": exit_price,
            "qty": qty, "bars_held": bars_held,
            "pnl_pts": pnl_pts, "pnl_$": pnl_usd, "reason": exit_reason,
        })

        if use_marti:
            last_was_loss = pnl_usd < 0
            if marti_state == 0:
                marti_state = 1 if last_was_loss else 0
            elif marti_state == 1:
                marti_state = 2
            else:
                marti_state = 1 if last_was_loss else 0

    return pd.DataFrame(trades), skipped_filter


def report(trades: pd.DataFrame, label: str):
    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")
    if len(trades) == 0:
        print("  NO TRADES"); return
    pnls = trades["pnl_$"].values
    n = len(pnls)
    w = (pnls > 0).sum(); l = (pnls < 0).sum()
    wr = w / n * 100
    gross_w = pnls[pnls > 0].sum(); gross_l = -pnls[pnls < 0].sum()
    pf = gross_w / gross_l if gross_l > 0 else 99.0
    cum = pnls.cumsum()
    mdd = float((cum - np.maximum.accumulate(cum)).min())
    avg_w = pnls[pnls > 0].mean() if w > 0 else 0
    avg_l = pnls[pnls < 0].mean() if l > 0 else 0

    print(f"  Trades:    {n}")
    print(f"  WR:        {wr:.1f}%  ({w} wins / {l} losses)")
    print(f"  Net $:     ${pnls.sum():>+,.0f}")
    print(f"  PF:        {pf:.3f}")
    print(f"  MaxDD:     ${mdd:>+,.0f}")
    print(f"  Avg win:   ${avg_w:>+,.0f}   Avg loss: ${avg_l:>+,.0f}   R:R = {avg_w/abs(avg_l):.2f}")
    print(f"  Best:      ${pnls.max():>+,.0f}   Worst: ${pnls.min():>+,.0f}")

    # IS/OOS
    trades_sorted = trades.sort_values("date").reset_index(drop=True)
    dates = sorted(trades_sorted["date"].unique())
    cutoff = dates[int(len(dates) * 0.6)]
    is_t  = trades_sorted[trades_sorted["date"] <  cutoff]
    oos_t = trades_sorted[trades_sorted["date"] >= cutoff]
    for nm, sub in [("IS", is_t), ("OOS", oos_t)]:
        if len(sub) == 0: continue
        p = sub["pnl_$"].values
        ww = p[p>0].sum(); ll = -p[p<0].sum()
        spf = ww/ll if ll > 0 else 99.0
        cum_s = p.cumsum()
        smdd = float((cum_s - np.maximum.accumulate(cum_s)).min())
        print(f"  {nm:<3} (cutoff {cutoff}): n={len(sub):>4}  net=${p.sum():>+9,.0f}  PF={spf:.3f}  MDD=${smdd:>+9,.0f}")

    # Yearly breakdown
    trades_sorted["year"] = pd.to_datetime(trades_sorted["date"]).dt.year
    print(f"\n  -- Yearly breakdown --")
    print(f"  {'year':<6} {'n':>4} {'WR':>5} {'net $':>11} {'PF':>6} {'MDD $':>11} {'longs':>5} {'shorts':>6}")
    for year, sub in trades_sorted.groupby("year"):
        p = sub["pnl_$"].values
        ww = p[p>0].sum(); ll = -p[p<0].sum()
        spf = ww/ll if ll > 0 else 99.0
        cum_s = p.cumsum(); smdd = float((cum_s - np.maximum.accumulate(cum_s)).min())
        n_long = (sub["direction"] == "LONG").sum()
        n_short = (sub["direction"] == "SHORT").sum()
        wr_y = (p > 0).mean() * 100
        print(f"  {year:<6} {len(sub):>4} {wr_y:>4.1f}% ${p.sum():>+10,.0f} {spf:>6.3f} ${smdd:>+10,.0f} {n_long:>5} {n_short:>6}")

    # Direction breakdown
    print(f"\n  -- By direction × prior gamma --")
    for d in ["LONG", "SHORT"]:
        sub = trades_sorted[trades_sorted["direction"] == d]
        if len(sub) == 0: continue
        p = sub["pnl_$"].values
        ww = p[p>0].sum(); ll = -p[p<0].sum()
        spf = ww/ll if ll > 0 else 99.0
        gs = sub["prior_gamma_sign"].iloc[0] if len(sub) > 0 else 0
        wr_d = (p > 0).mean() * 100
        print(f"    {d:<5} (prior gamma={int(gs):+d}): n={len(sub):>4} WR={wr_d:.1f}% net=${p.sum():>+9,.0f} PF={spf:.3f}")


def main():
    bars = build_30min_bars()
    gamma_pairs = load_gamma_lookup()
    print(f"Loaded {len(bars):,} 30-min bars; {len(gamma_pairs):,} gamma data points")
    print(f"Gamma date range: {gamma_pairs[0][0]} -> {gamma_pairs[-1][0]}")

    # Config 1: risk-adjusted winner
    b1 = compute_indicators(bars, ema_n=60, atr_n=28)
    trades1, skipped1 = run_strategy_with_gamma_filter(b1, sl_mult=1.25, gamma_pairs=gamma_pairs)
    report(trades1, "FILTERED — EMA=60, ATR=28, SL_MULT=1.25 (risk-adj winner)")
    print(f"  Skipped by filter: no_gamma={skipped1['no_gamma']}, wrong_dir_pos={skipped1['wrong_dir_pos']}, wrong_dir_neg={skipped1['wrong_dir_neg']}")

    # Config 2: absolute $ winner
    b2 = compute_indicators(bars, ema_n=60, atr_n=10)
    trades2, skipped2 = run_strategy_with_gamma_filter(b2, sl_mult=2.75, gamma_pairs=gamma_pairs)
    report(trades2, "FILTERED — EMA=60, ATR=10, SL_MULT=2.75 (abs $ winner)")
    print(f"  Skipped by filter: no_gamma={skipped2['no_gamma']}, wrong_dir_pos={skipped2['wrong_dir_pos']}, wrong_dir_neg={skipped2['wrong_dir_neg']}")

    OUT_DIR.mkdir(exist_ok=True)
    trades1.to_csv(OUT_DIR / "filtered_trades_risk_adj.csv", index=False)
    trades2.to_csv(OUT_DIR / "filtered_trades_abs_dollar.csv", index=False)

    # Sanity-check the prior-gamma lookup (verify no lookahead)
    print(f"\n=== LOOKAHEAD SANITY CHECK ===")
    sample = trades1.head(5)
    for _, t in sample.iterrows():
        prior_date_actual = None
        for d, s in gamma_pairs:
            if d < t["date"]:
                prior_date_actual = d
            else:
                break
        print(f"  Trade {t['date']} ({t['direction']}, gamma={int(t['prior_gamma_sign']):+d})  "
              f"-> gamma lookup used date {prior_date_actual}  "
              f"-> gap: {(t['date'] - prior_date_actual).days} day(s) BEFORE trade")


if __name__ == "__main__":
    main()
