"""
Inspect a single rough-vol config: trade-level capture, year-by-year metrics,
monthly trade distribution, and equity curve PNG.
"""
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
RESULTS_DIR = HERE / "results"
CACHE_DIR = HERE / ".cache"

import core  # noqa


# Target config: 20m N=400 Z=75 EMA=80 HZ=2.00 SL=2.0 TP=2.0 RR=1.00 + window_N8_D150 + gamma=none
CFG = dict(
    bm=20, norm=400, zlook=75, ema_len=80, hz=2.00, sl=2.0, tp=2.0,
    variant="window", N=8, D=150, gamma_mode=0, gamma_name="none",
)


def run_with_trades(b, z_vol, ema, atr, long_mask, short_mask, g_sign,
                    hz, atr_sl, atr_tp, gamma_mode):
    """Pure-Python mirror of backtest_jit that records trade details."""
    highs = b["highs"]; lows = b["lows"]; closes = b["closes"]
    mod = b["minutes_of_day"]; di = b["day_idx"]
    n = len(closes)
    ss = core.SESSION_START_MIN; se = core.SESSION_END_MIN
    max_trades = core.MAX_TRADES_PER_DAY
    is_end_ord = b["is_end_ord"]

    pos = 0
    ep = sl_p = tp_p = 0.0
    cur_day = -1
    dt = 0
    entry_idx = 0
    trades = []

    for i in range(n):
        in_session = ss <= mod[i] < se
        if pos != 0 and (not in_session) and mod[i] >= se:
            xp = closes[i]
            trades.append((entry_idx, i, pos, ep, xp, (xp-ep)*pos, "force_close"))
            pos = 0
        if not in_session:
            continue
        if di[i] != cur_day:
            cur_day = di[i]
            dt = 0
        if pos != 0:
            exited = False; xp = 0.0; reason = ""
            if pos == 1:
                if lows[i] <= sl_p:
                    xp = sl_p; exited = True; reason = "stop"
                elif highs[i] >= tp_p:
                    xp = tp_p; exited = True; reason = "target"
            else:
                if highs[i] >= sl_p:
                    xp = sl_p; exited = True; reason = "stop"
                elif lows[i] <= tp_p:
                    xp = tp_p; exited = True; reason = "target"
            if exited:
                trades.append((entry_idx, i, pos, ep, xp, (xp-ep)*pos, reason))
                pos = 0
                continue
        if pos == 0 and dt < max_trades:
            atr_v = atr[i]
            if atr_v <= 0: continue
            z = z_vol[i]; cl = closes[i]; em = ema[i]
            if z > hz:
                new_dir = 0
                if cl > em: new_dir = 1
                elif cl < em: new_dir = -1
                if new_dir != 0:
                    if new_dir == 1 and long_mask[i] == 0: continue
                    if new_dir == -1 and short_mask[i] == 0: continue
                    if gamma_mode != 0:
                        g = g_sign[i]
                        if gamma_mode == 1 and g == -1: continue
                        if gamma_mode == 2 and g == 1: continue
                    pos = new_dir; ep = cl; entry_idx = i
                    if new_dir == 1:
                        sl_p = cl - atr_sl * atr_v; tp_p = cl + atr_tp * atr_v
                    else:
                        sl_p = cl + atr_sl * atr_v; tp_p = cl - atr_tp * atr_v
                    dt += 1
    return trades


def main():
    bm = CFG["bm"]
    # Load bars cache
    with open(CACHE_DIR / f"bars_{bm}m.pkl", "rb") as f:
        b = pickle.load(f)
    # Need timestamps too — rebuild bar index from build_bars for trade timestamps
    bars_df = core.build_bars(bm)
    bar_ts = bars_df.index  # tz-aware ET
    assert len(bar_ts) == len(b["closes"]), f"bar length mismatch: {len(bar_ts)} vs {len(b['closes'])}"

    # Signals
    z_vol = core.compute_zvol(b["closes"], CFG["norm"], CFG["zlook"])
    ema = core.compute_ema(b["closes"], CFG["ema_len"])
    atr = b["atr"]

    # Orderflow masks
    with open(CACHE_DIR / f"orderflow_{bm}m.pkl", "rb") as f:
        of = pickle.load(f)
    lmask = of["window_long"][(CFG["N"], CFG["D"])]
    smask = of["window_short"][(CFG["N"], CFG["D"])]

    with open(CACHE_DIR / f"gamma_{bm}m.pkl", "rb") as f:
        gs = pickle.load(f)

    trades = run_with_trades(b, z_vol, ema, atr, lmask, smask, gs,
                             CFG["hz"], CFG["sl"], CFG["tp"], CFG["gamma_mode"])
    print(f"Captured {len(trades)} trades")

    # Build trade DF
    rows = []
    for (ent_i, ext_i, side, ep, xp, pnl_pts, reason) in trades:
        rows.append(dict(
            entry_ts=bar_ts[ent_i], exit_ts=bar_ts[ext_i],
            side="LONG" if side == 1 else "SHORT",
            entry_price=ep, exit_price=xp,
            pnl_pts=pnl_pts, pnl_dollars=pnl_pts * core.POINT_VALUE,
            reason=reason,
        ))
    td = pd.DataFrame(rows)
    td["year"] = td["entry_ts"].dt.year
    td["month"] = td["entry_ts"].dt.to_period("M").astype(str)
    td["date"] = td["entry_ts"].dt.date

    # Save trade log
    log_path = RESULTS_DIR / "inspect_20m_N400_window_N8D150_trades.csv"
    td.to_csv(log_path, index=False)
    print(f"Trade log -> {log_path}")

    # Yearly breakdown
    print("\n=== Year-by-year ===")
    print(f"{'year':>6} {'trades':>6} {'PF':>6} {'WR':>6} {'PnL$':>10} {'MDD$':>10} {'avg_w$':>8} {'avg_l$':>8}")
    cum = 0.0
    yearly = []
    for y, g in td.groupby("year"):
        p = g["pnl_dollars"].to_numpy()
        w = p[p > 0]; l = p[p < 0]
        pf = w.sum() / abs(l.sum()) if len(l) else 99.0
        wr = 100 * len(w) / len(p)
        cumy = p.cumsum()
        mdd = (cumy - np.maximum.accumulate(cumy)).min()
        avg_w = w.mean() if len(w) else 0
        avg_l = l.mean() if len(l) else 0
        print(f"{y:>6} {len(p):>6d} {pf:>6.2f} {wr:>5.1f}% {p.sum():>+10,.0f} {mdd:>+10,.0f} {avg_w:>+8,.0f} {avg_l:>+8,.0f}")
        yearly.append((y, len(p), pf, wr, p.sum(), mdd))
        cum += p.sum()

    # Monthly trade count
    print("\n=== Monthly trade count + PnL ===")
    monthly = td.groupby("month").agg(trades=("pnl_dollars", "size"),
                                       pnl=("pnl_dollars", "sum"))
    monthly["cum"] = monthly["pnl"].cumsum()
    # Print compact
    months = monthly.index.tolist()
    n_months = len(months)
    zeros = 0
    print(f"Total months in trade range: {n_months}")
    print(f"Months with <2 trades: {(monthly['trades'] < 2).sum()}")
    print(f"Months with <=5 trades: {(monthly['trades'] <= 5).sum()}")
    print(f"Avg trades/month: {monthly['trades'].mean():.1f}, median: {monthly['trades'].median():.0f}")
    print(f"Min trades/month: {monthly['trades'].min()}, max: {monthly['trades'].max()}")

    print("\nMonth-by-month (last 24):")
    print(monthly.tail(24).to_string())

    monthly.to_csv(RESULTS_DIR / "inspect_20m_N400_window_N8D150_monthly.csv")

    # Equity curve
    td_sorted = td.sort_values("entry_ts").reset_index(drop=True)
    td_sorted["cum_pnl"] = td_sorted["pnl_dollars"].cumsum()
    td_sorted["peak"] = td_sorted["cum_pnl"].cummax()
    td_sorted["dd"] = td_sorted["cum_pnl"] - td_sorted["peak"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    # Equity
    axes[0].plot(td_sorted["entry_ts"], td_sorted["cum_pnl"], color="steelblue", lw=1.2)
    axes[0].fill_between(td_sorted["entry_ts"], 0, td_sorted["cum_pnl"], alpha=0.15, color="steelblue")
    is_end = pd.Timestamp(core.IS_END).tz_localize("America/New_York")
    axes[0].axvline(is_end, color="red", ls="--", lw=1, alpha=0.7, label="IS/OOS split")
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_ylabel("Cumulative PnL ($)")
    axes[0].set_title(f"20m N=400 Z=75 EMA=80 HZ=2.00 SL=2.0 TP=2.0 (RR=1.0) | window_N8_D150 + gamma=none\n"
                       f"{len(td_sorted)} trades  PnL ${td_sorted['cum_pnl'].iloc[-1]:+,.0f}  MDD ${td_sorted['dd'].min():+,.0f}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    # Drawdown
    axes[1].fill_between(td_sorted["entry_ts"], td_sorted["dd"], 0, color="red", alpha=0.4)
    axes[1].set_ylabel("DD ($)")
    axes[1].grid(alpha=0.3)
    # Monthly trade count
    monthly_ts = pd.to_datetime(monthly.index.astype(str) + "-01")
    axes[2].bar(monthly_ts, monthly["trades"], width=20, color="darkgreen", alpha=0.7)
    axes[2].set_ylabel("Trades / month")
    axes[2].set_xlabel("Date")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out_png = RESULTS_DIR / "inspect_20m_N400_window_N8D150_curve.png"
    plt.savefig(out_png, dpi=110)
    print(f"\nEquity curve -> {out_png}")


if __name__ == "__main__":
    main()
