"""
1) Re-run v3 backtest capturing signal-bar OHLC + SL + TP per trade.
2) Filter 2026 log and write it.
3) Sweep martingale variants (streak / mult / max_doubles / fc_only / always).
"""
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
RESULTS_DIR = HERE / "results"
CACHE_DIR = HERE / ".cache"

import core  # noqa

# v3 config (locked)
CFG = dict(bm=20, norm=400, zlook=75, ema_len=80, hz=2.00, sl=2.0, tp=2.0, N=8, D=150)
ENTRY_START_MIN = 9 * 60
SKIP_START_MIN = 13 * 60
SKIP_END_MIN = 14 * 60
SESSION_END_MIN = 14 * 60 + 45


def run_with_full_log(b, z_vol, ema, atr, long_mask, short_mask, g_sign,
                       hz, atr_sl, atr_tp):
    highs = b["highs"]; lows = b["lows"]; closes = b["closes"]; opens = b["opens"]
    mod = b["minutes_of_day"]; di = b["day_idx"]
    n = len(closes)

    pos = 0
    ep = sl_p = tp_p = 0.0
    cur_day = -1
    entry_idx = 0
    entry_atr = 0.0
    entry_gsign = 0
    trades = []

    for i in range(n):
        m = mod[i]
        in_session_full = ENTRY_START_MIN <= m < SESSION_END_MIN

        if pos != 0 and m >= SESSION_END_MIN:
            xp = closes[i]
            trades.append(dict(
                entry_idx=entry_idx, exit_idx=i, side=pos, entry_price=ep, exit_price=xp,
                sl_price=sl_p, tp_price=tp_p, pnl_pts=(xp-ep)*pos, reason="force_close",
                entry_atr=entry_atr, gsign=entry_gsign,
            ))
            pos = 0
        if not in_session_full:
            continue
        if di[i] != cur_day:
            cur_day = di[i]
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
                trades.append(dict(
                    entry_idx=entry_idx, exit_idx=i, side=pos, entry_price=ep, exit_price=xp,
                    sl_price=sl_p, tp_price=tp_p, pnl_pts=(xp-ep)*pos, reason=reason,
                    entry_atr=entry_atr, gsign=entry_gsign,
                ))
                pos = 0
                continue
        in_skip = SKIP_START_MIN <= m < SKIP_END_MIN
        if pos == 0 and not in_skip:
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
                    pos = new_dir; ep = cl; entry_idx = i
                    entry_atr = atr_v; entry_gsign = int(g_sign[i])
                    if new_dir == 1:
                        sl_p = cl - atr_sl * atr_v; tp_p = cl + atr_tp * atr_v
                    else:
                        sl_p = cl + atr_sl * atr_v; tp_p = cl - atr_tp * atr_v
    return trades


def run():
    bm = CFG["bm"]
    with open(CACHE_DIR / f"bars_{bm}m.pkl", "rb") as f:
        b = pickle.load(f)
    with open(CACHE_DIR / f"orderflow_{bm}m.pkl", "rb") as f:
        of = pickle.load(f)
    with open(CACHE_DIR / f"gamma_{bm}m.pkl", "rb") as f:
        gs = pickle.load(f)
    z_vol = core.compute_zvol(b["closes"], CFG["norm"], CFG["zlook"])
    ema = core.compute_ema(b["closes"], CFG["ema_len"])
    atr = b["atr"]
    lmask = of["window_long"][(CFG["N"], CFG["D"])]
    smask = of["window_short"][(CFG["N"], CFG["D"])]
    trades = run_with_full_log(b, z_vol, ema, atr, lmask, smask, gs,
                                CFG["hz"], CFG["sl"], CFG["tp"])
    bars20 = core.build_bars(20)
    bar_ts = bars20.index
    opens = bars20["open"].to_numpy()
    highs = bars20["high"].to_numpy()
    lows = bars20["low"].to_numpy()
    closes_b = bars20["close"].to_numpy()
    rows = []
    for t in trades:
        ei = t["entry_idx"]; xi = t["exit_idx"]
        rows.append(dict(
            entry_ts=bar_ts[ei],
            exit_ts=bar_ts[xi],
            side="LONG" if t["side"] == 1 else "SHORT",
            sig_open=opens[ei], sig_high=highs[ei], sig_low=lows[ei], sig_close=closes_b[ei],
            entry_price=t["entry_price"],
            sl_price=t["sl_price"], tp_price=t["tp_price"],
            exit_price=t["exit_price"],
            entry_atr=t["entry_atr"],
            pnl_pts=t["pnl_pts"], pnl_dollars=t["pnl_pts"] * core.POINT_VALUE,
            reason=t["reason"],
        ))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "inspect_v3_FULL_log.csv", index=False)
    print(f"Full log: {len(df)} trades")
    return df


def report_2026(df):
    df2026 = df[df["entry_ts"].dt.year == 2026].copy()
    print(f"\n=== 2026 trade log ({len(df2026)} trades) ===")
    print(f"{'#':>3} {'entry_ts':>19} {'side':>5} {'sig_O':>8} {'sig_H':>8} {'sig_L':>8} {'sig_C':>8} "
          f"{'entry':>8} {'SL':>8} {'TP':>8} {'exit_ts':>19} {'exit':>8} {'reason':>11} {'PnL$':>9}")
    for i, r in enumerate(df2026.itertuples(), 1):
        print(f"{i:>3} {str(r.entry_ts)[:19]:>19} {r.side:>5} "
              f"{r.sig_open:>8.2f} {r.sig_high:>8.2f} {r.sig_low:>8.2f} {r.sig_close:>8.2f} "
              f"{r.entry_price:>8.2f} {r.sl_price:>8.2f} {r.tp_price:>8.2f} "
              f"{str(r.exit_ts)[:19]:>19} {r.exit_price:>8.2f} {r.reason:>11} "
              f"${r.pnl_dollars:>+8,.0f}")
    df2026.to_csv(RESULTS_DIR / "inspect_v3_2026_log.csv", index=False)
    print(f"\nWrote inspect_v3_2026_log.csv")
    p = df2026["pnl_dollars"].to_numpy()
    if len(p):
        w = p[p>0]; l = p[p<0]
        pf = w.sum()/abs(l.sum()) if len(l) else 99
        cum = p.cumsum()
        mdd = (cum - np.maximum.accumulate(cum)).min()
        print(f"\n2026 summary: {len(p)}t  PF {pf:.2f}  WR {100*len(w)/len(p):.1f}%  "
              f"PnL ${p.sum():+,.0f}  MDD ${mdd:+,.0f}")


def replay_with_martingale(df, streak, mult, max_doubles, fc_only):
    """Replay trades with martingale sizing.

    streak: # consecutive losses before doubling
    mult: size multiplier per qualifying loss (e.g., 1.5, 2.0)
    max_doubles: cap on consecutive doublings
    fc_only: if True, only count force_close losses toward streak (B2-style)
    Returns: sized PnL array, metrics.
    """
    qty = 1
    loss_streak = 0
    out = np.zeros(len(df), dtype=np.float64)
    qty_arr = np.zeros(len(df), dtype=np.int32)
    for i, r in enumerate(df.itertuples()):
        out[i] = r.pnl_dollars * qty
        qty_arr[i] = qty
        is_loss = r.pnl_dollars < 0
        # update streak
        if is_loss:
            if fc_only:
                if r.reason == "force_close":
                    loss_streak += 1
                else:
                    loss_streak = 0
            else:
                loss_streak += 1
        else:
            loss_streak = 0
        # next qty
        if loss_streak >= streak:
            steps = min(loss_streak - streak + 1, max_doubles)
            qty = max(1, int(round(mult ** steps)))
        else:
            qty = 1
    p = out
    w = p[p > 0]; l = p[p < 0]
    pf = w.sum() / abs(l.sum()) if len(l) else 99.0
    wr = 100 * len(w) / len(p) if len(p) else 0
    cum = p.cumsum()
    mdd = (cum - np.maximum.accumulate(cum)).min()
    mar = p.sum() / abs(mdd) if mdd < 0 else 99.0
    return dict(trades=len(p), pf=pf, wr=wr, pnl=p.sum(), mdd=mdd, mar=mar,
                max_qty=int(qty_arr.max()), avg_qty=float(qty_arr.mean()))


def martingale_sweep(df):
    df = df.sort_values("entry_ts").reset_index(drop=True)
    rows = []
    # baseline: no martingale
    base = replay_with_martingale(df, streak=999, mult=1.0, max_doubles=0, fc_only=False)
    rows.append(("BASELINE (no mart)", "-", "-", "-", "-", base))
    # sweep
    for fc_only in (False, True):
        for streak in (1, 2, 3):
            for mult in (1.5, 2.0, 2.5):
                for maxd in (1, 2, 3, 4):
                    if mult ** maxd > 16:
                        continue  # cap absurd
                    m = replay_with_martingale(df, streak=streak, mult=mult,
                                                max_doubles=maxd, fc_only=fc_only)
                    rows.append((
                        f"streak={streak} mult={mult} maxd={maxd}",
                        streak, mult, maxd,
                        "fc_only" if fc_only else "any_loss",
                        m,
                    ))
    print(f"\n=== Martingale sweep ({len(rows)} variants) ===")
    print(f"{'config':>32} {'qual':>10} {'tr':>4} {'PF':>5} {'WR':>5} {'PnL$':>11} "
          f"{'MDD$':>11} {'MAR':>5} {'maxQ':>4} {'avgQ':>5}")
    rows_with_metrics = [(label, qual, m) for label, _, _, _, qual, m in rows if qual != "-"]
    # also print baseline
    base = rows[0][5]
    print(f"{'BASELINE (no mart)':>32} {'-':>10} {base['trades']:>4} {base['pf']:>5.2f} "
          f"{base['wr']:>4.1f}% ${base['pnl']:>+9,.0f} ${base['mdd']:>+9,.0f} "
          f"{base['mar']:>5.2f} {base['max_qty']:>4} {base['avg_qty']:>5.2f}")
    # sort by MAR
    rows_sorted = sorted(rows_with_metrics, key=lambda x: x[2]["pnl"], reverse=True)
    for label, qual, m in rows_sorted[:40]:
        print(f"{label:>32} {qual:>10} {m['trades']:>4} {m['pf']:>5.2f} "
              f"{m['wr']:>4.1f}% ${m['pnl']:>+9,.0f} ${m['mdd']:>+9,.0f} "
              f"{m['mar']:>5.2f} {m['max_qty']:>4} {m['avg_qty']:>5.2f}")
    # save full
    save_rows = []
    save_rows.append(dict(label="BASELINE", qual="-", **base))
    for label, qual, m in rows_with_metrics:
        save_rows.append(dict(label=label, qual=qual, **m))
    pd.DataFrame(save_rows).to_csv(RESULTS_DIR / "martingale_sweep.csv", index=False)
    print(f"\nWrote martingale_sweep.csv")


def main():
    df = run()
    report_2026(df)
    martingale_sweep(df)


if __name__ == "__main__":
    main()
