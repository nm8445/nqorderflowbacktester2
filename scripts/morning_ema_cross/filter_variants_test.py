"""Test all 4 gamma-filter variants on both top configs.

Each variant is FULLY RE-SIMULATED (not summed from per-leg slices) so MDD
and martingale state propagation are accurate.

Filters tested:
  A. NO_FILTER         — original strategy (baseline)
  B. BLOCK_POS_LONG    — skip entries when prior gamma=+1 AND ema_sign=+1
  C. BLOCK_POS_SHORT   — skip entries when prior gamma=+1 AND ema_sign=-1
  D. BLOCK_POS_BOTH    — skip ALL entries on prior pos-gamma days

Goal: find a variant with MDD <= $30K while keeping net $ as high as possible.

Configs tested:
  - EMA=60, ATR=28, SL=1.25  (risk-adjusted winner)
  - EMA=60, ATR=10, SL=2.75  (absolute $ winner)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from strategy_atr_sl import build_30min_bars, compute_indicators, NQ_PT

GAMMA_PATH = "D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet"


def load_gamma_pairs() -> list:
    g = pd.read_parquet(GAMMA_PATH)
    g["date"] = pd.to_datetime(g["date"]).dt.date
    g = g[["date", "qqq_gamma_sign"]].dropna(subset=["qqq_gamma_sign"]).sort_values("date")
    return list(zip(g["date"].tolist(), g["qqq_gamma_sign"].tolist()))


def prior_gamma(d, pairs, max_stale=5):
    prev_d, prev_s = None, None
    for gd, gs in pairs:
        if gd < d:
            prev_d, prev_s = gd, gs
        else:
            break
    if prev_d is None or (d - prev_d).days > max_stale:
        return None
    return prev_s


def run_with_filter(b: pd.DataFrame, sl_mult: float, gamma_pairs: list,
                    filter_name: str,
                    use_marti: bool = True, base_qty: int = 1, loss_qty: int = 2):
    """filter_name in {'NO_FILTER', 'BLOCK_POS_LONG', 'BLOCK_POS_SHORT', 'BLOCK_POS_BOTH'}."""
    trades = []
    marti_state = 0
    n_skipped = 0

    for date, day in b.groupby("date"):
        if day["dow"].iloc[0] >= 5: continue
        entry_bar = day[day["hhmm"] == 1000]
        if len(entry_bar) == 0: continue
        eb = entry_bar.iloc[0]
        if pd.isna(eb["ema"]) or pd.isna(eb["atr"]) or eb["atr"] <= 0:
            continue

        entry_price = float(eb["close"])
        entry_atr   = float(eb["atr"])
        if entry_price > eb["ema"]:   sign = 1
        elif entry_price < eb["ema"]: sign = -1
        else: continue

        # Gamma filter
        prior_g = prior_gamma(date, gamma_pairs)
        if filter_name != "NO_FILTER":
            if prior_g is None:
                n_skipped += 1; continue   # require gamma data when filter is active
            if filter_name == "BLOCK_POS_LONG"  and prior_g > 0 and sign == 1:
                n_skipped += 1; continue
            if filter_name == "BLOCK_POS_SHORT" and prior_g > 0 and sign == -1:
                n_skipped += 1; continue
            if filter_name == "BLOCK_POS_BOTH"  and prior_g > 0:
                n_skipped += 1; continue

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
            "date": date, "direction": "LONG" if sign > 0 else "SHORT",
            "prior_gamma": prior_g if prior_g is not None else 0,
            "qty": qty, "pnl_$": pnl_usd, "reason": exit_reason,
        })
        if use_marti:
            last_was_loss = pnl_usd < 0
            if marti_state == 0:
                marti_state = 1 if last_was_loss else 0
            elif marti_state == 1:
                marti_state = 2
            else:
                marti_state = 1 if last_was_loss else 0

    return pd.DataFrame(trades), n_skipped


def stats(t: pd.DataFrame) -> dict:
    if len(t) == 0:
        return dict(n=0, wr=0, net=0, pf=0, mdd=0)
    p = t["pnl_$"].values
    w = p[p > 0]; l = p[p < 0]
    pf = w.sum() / abs(l.sum()) if len(l) > 0 else 99.0
    cum = p.cumsum()
    mdd = float((cum - np.maximum.accumulate(cum)).min())
    return dict(n=len(p), wr=round((p>0).sum()/len(p)*100, 1),
                net=round(float(p.sum()), 0), pf=round(pf, 3), mdd=round(mdd, 0))


def yearly(t: pd.DataFrame) -> pd.DataFrame:
    if len(t) == 0: return pd.DataFrame()
    t = t.copy()
    t["year"] = pd.to_datetime(t["date"]).dt.year
    rows = []
    for y, sub in t.groupby("year"):
        p = sub["pnl_$"].values
        wr = (p > 0).mean() * 100
        net = float(p.sum())
        gw = p[p>0].sum(); gl = -p[p<0].sum()
        pf = gw/gl if gl > 0 else 99.0
        cum = p.cumsum(); mdd = float((cum - np.maximum.accumulate(cum)).min())
        rows.append(dict(year=int(y), n=len(sub), wr=round(wr,1), net=round(net,0),
                          pf=round(pf,3), mdd=round(mdd,0)))
    return pd.DataFrame(rows)


def run_all(label: str, bars, ema_n, atr_n, sl_mult, gamma_pairs):
    b = compute_indicators(bars, ema_n, atr_n)
    print(f"\n{'='*100}")
    print(f"  {label}: EMA={ema_n}, ATR={atr_n}, SL_MULT={sl_mult}")
    print(f"{'='*100}")
    results = {}
    for filt in ["NO_FILTER", "BLOCK_POS_LONG", "BLOCK_POS_SHORT", "BLOCK_POS_BOTH"]:
        t, skipped = run_with_filter(b, sl_mult, gamma_pairs, filt)
        s = stats(t)
        results[filt] = (t, s, skipped)

    print(f"\n  {'filter':<20} {'n':>5} {'WR':>5} {'net':>11} {'PF':>6} {'MDD':>11} {'<=$30K?':>9} {'skipped':>8}")
    for filt, (t, s, sk) in results.items():
        mdd_ok = "YES" if s["mdd"] >= -30000 else "no"
        print(f"  {filt:<20} {s['n']:>5} {s['wr']:>4.1f}% ${s['net']:>+10,.0f} {s['pf']:>6.3f} ${s['mdd']:>+10,.0f} {mdd_ok:>9} {sk:>8}")

    # Yearly for each filter that meets MDD goal
    for filt, (t, s, sk) in results.items():
        if s["mdd"] < -30000 and filt != "NO_FILTER":
            continue   # only report yearly for MDD-passing or baseline
        if len(t) == 0: continue
        print(f"\n  -- YEARLY: {filt} --")
        y = yearly(t)
        print(f"  {'year':>5} {'n':>4} {'WR':>5} {'net':>11} {'PF':>6} {'MDD':>11}")
        for _, r in y.iterrows():
            print(f"  {r['year']:>5} {r['n']:>4} {r['wr']:>4.1f}% ${r['net']:>+10,.0f} {r['pf']:>6.3f} ${r['mdd']:>+10,.0f}")

    return results


def main():
    bars = build_30min_bars()
    gamma_pairs = load_gamma_pairs()
    print(f"Loaded {len(bars):,} bars, {len(gamma_pairs):,} gamma rows")

    run_all("RISK-ADJ", bars, 60, 28, 1.25, gamma_pairs)
    run_all("ABS-$",    bars, 60, 10, 2.75, gamma_pairs)


if __name__ == "__main__":
    main()
