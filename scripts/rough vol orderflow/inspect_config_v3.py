"""
v3 inspector — gamma=none, MT=999, entry session:
  09:00 <= entry_time < 13:00  (morning)
  14:00 <= entry_time < 14:45  (late afternoon)
Force-close still at 14:45. Positions opened in the morning ride through 13:00-13:59
(SL/TP still active), but no NEW entries during that hour.
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


# New session
ENTRY_START_MIN = 9 * 60          # 540 (09:00)
SKIP_START_MIN = 13 * 60          # 780 (13:00)
SKIP_END_MIN = 14 * 60            # 840 (14:00)
SESSION_END_MIN = 14 * 60 + 45    # 885 (14:45 force-close)

CONFIGS = {
    "N400_v3": dict(
        label="20m N=400 Z=75 EMA=80 HZ=2.00 SL=2.0 TP=2.0 | window_N8_D150 | g=none | MT=999 | sess 09:00-12:59 + 14:00-14:44",
        norm=400, zlook=75, ema_len=80, hz=2.00, sl=2.0, tp=2.0, N=8, D=150,
    ),
    "N100_v3": dict(
        label="20m N=100 Z=175 EMA=80 HZ=1.90 SL=2.0 TP=2.0 | window_N8_D150 | g=none | MT=999 | sess 09:00-12:59 + 14:00-14:44",
        norm=100, zlook=175, ema_len=80, hz=1.90, sl=2.0, tp=2.0, N=8, D=150,
    ),
}


def run_with_trades(b, z_vol, ema, atr, long_mask, short_mask, g_sign,
                    hz, atr_sl, atr_tp):
    highs = b["highs"]; lows = b["lows"]; closes = b["closes"]
    mod = b["minutes_of_day"]; di = b["day_idx"]
    n = len(closes)

    pos = 0
    ep = sl_p = tp_p = 0.0
    cur_day = -1
    entry_idx = 0
    entry_gsign = 0
    trades = []

    for i in range(n):
        m = mod[i]
        # In any active window (including skip — positions can exit during skip)
        in_session_full = ENTRY_START_MIN <= m < SESSION_END_MIN

        # Force close at/after session end
        if pos != 0 and m >= SESSION_END_MIN:
            xp = closes[i]
            trades.append((entry_idx, i, pos, ep, xp, (xp - ep) * pos, "force_close", entry_gsign))
            pos = 0
        # We allow position management (SL/TP) during entry windows AND skip window
        if not in_session_full:
            continue

        if di[i] != cur_day:
            cur_day = di[i]

        # Exit check (always active during 09:00-14:45)
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
                trades.append((entry_idx, i, pos, ep, xp, (xp - ep) * pos, reason, entry_gsign))
                pos = 0
                continue

        # Entry only outside the skip window
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
                    pos = new_dir; ep = cl; entry_idx = i; entry_gsign = int(g_sign[i])
                    if new_dir == 1:
                        sl_p = cl - atr_sl * atr_v; tp_p = cl + atr_tp * atr_v
                    else:
                        sl_p = cl + atr_sl * atr_v; tp_p = cl - atr_tp * atr_v
    return trades


def run_config(cfg, bar_ts):
    bm = 20
    with open(CACHE_DIR / f"bars_{bm}m.pkl", "rb") as f:
        b = pickle.load(f)
    with open(CACHE_DIR / f"orderflow_{bm}m.pkl", "rb") as f:
        of = pickle.load(f)
    with open(CACHE_DIR / f"gamma_{bm}m.pkl", "rb") as f:
        gs = pickle.load(f)
    z_vol = core.compute_zvol(b["closes"], cfg["norm"], cfg["zlook"])
    ema = core.compute_ema(b["closes"], cfg["ema_len"])
    atr = b["atr"]
    lmask = of["window_long"][(cfg["N"], cfg["D"])]
    smask = of["window_short"][(cfg["N"], cfg["D"])]
    trades = run_with_trades(b, z_vol, ema, atr, lmask, smask, gs,
                             cfg["hz"], cfg["sl"], cfg["tp"])
    rows = []
    for (ent_i, ext_i, side, ep, xp, pnl_pts, reason, gsign) in trades:
        rows.append(dict(
            entry_ts=bar_ts[ent_i], exit_ts=bar_ts[ext_i],
            side="LONG" if side == 1 else "SHORT",
            entry_price=ep, exit_price=xp,
            pnl_pts=pnl_pts, pnl_dollars=pnl_pts * core.POINT_VALUE,
            reason=reason, gsign=gsign,
        ))
    return pd.DataFrame(rows)


def summary(p):
    if len(p) == 0: return (0, 0.0, 0.0, 0.0, 0.0)
    w = p[p > 0]; l = p[p < 0]
    pf = w.sum() / abs(l.sum()) if len(l) else 99.0
    wr = 100 * len(w) / len(p)
    cum = np.cumsum(p)
    mdd = (cum - np.maximum.accumulate(cum)).min()
    return (len(p), pf, wr, p.sum(), mdd)


def report(td, label):
    is_end = pd.Timestamp(core.IS_END)
    td = td.copy()
    td["year"] = td["entry_ts"].dt.year
    td["hour"] = td["entry_ts"].dt.hour
    td["regime"] = td["gsign"].map({-1: "NEG", 0: "ZERO", 1: "POS"})
    td["in_is"] = td["entry_ts"].dt.tz_localize(None) <= is_end

    print(f"\n{'='*100}")
    print(label)
    print('=' * 100)
    n, pf, wr, pnl, mdd = summary(td["pnl_dollars"].to_numpy())
    print(f"OVERALL: {n} trades  PF {pf:.2f}  WR {wr:.1f}%  PnL ${pnl:+,.0f}  MDD ${mdd:+,.0f}")
    for k, mask in (("IS ", td["in_is"]), ("OOS", ~td["in_is"])):
        sn, spf, swr, spnl, smdd = summary(td[mask]["pnl_dollars"].to_numpy())
        print(f"  {k}: {sn} trades  PF {spf:.2f}  WR {swr:.1f}%  PnL ${spnl:+,.0f}  MDD ${smdd:+,.0f}")

    print("\n--- Year-by-year ---")
    print(f"{'year':>4} {'tr':>4} {'PF':>5} {'WR':>5} {'PnL$':>10} {'MDD$':>10}")
    for y, g in td.groupby("year"):
        sn, spf, swr, spnl, smdd = summary(g["pnl_dollars"].to_numpy())
        print(f"{y:>4} {sn:>4} {spf:>5.2f} {swr:>4.1f}% {spnl:>+10,.0f} {smdd:>+10,.0f}")

    print("\n--- Hour-by-hour (entry hour ET) ---")
    print(f"{'hr':>3} {'tr':>4} {'PF':>5} {'WR':>5} {'PnL$':>10}")
    for h, g in td.groupby("hour"):
        sn, spf, swr, spnl, _ = summary(g["pnl_dollars"].to_numpy())
        print(f"{h:>3} {sn:>4} {spf:>5.2f} {swr:>4.1f}% {spnl:>+10,.0f}")

    print("\n--- Side x Gamma regime ---")
    print(f"{'regime':>6} {'side':>5} {'tr':>4} {'PF':>5} {'WR':>5} {'avg':>6} {'PnL$':>10}")
    for reg in ["NEG", "ZERO", "POS"]:
        for side in ["LONG", "SHORT"]:
            g = td[(td["regime"] == reg) & (td["side"] == side)]
            p = g["pnl_dollars"].to_numpy()
            sn, spf, swr, spnl, _ = summary(p)
            avg = p.mean() if len(p) else 0
            print(f"{reg:>6} {side:>5} {sn:>4} {spf:>5.2f} {swr:>4.1f}% {avg:>+6,.0f} {spnl:>+10,.0f}")


def plot_curves(tds_labeled, out_path):
    fig, ax = plt.subplots(figsize=(13, 7))
    is_end = pd.Timestamp(core.IS_END).tz_localize("America/New_York")
    for label, td in tds_labeled:
        td_sorted = td.sort_values("entry_ts").reset_index(drop=True)
        td_sorted["cum"] = td_sorted["pnl_dollars"].cumsum()
        ax.plot(td_sorted["entry_ts"], td_sorted["cum"], lw=1.3, label=label)
    ax.axvline(is_end, color="red", ls="--", lw=1, alpha=0.7, label="IS/OOS")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_xlabel("Date")
    ax.set_title("Rough Vol v3 — session 09:00-12:59 + 14:00-14:44, gamma=none, MT=999")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    print(f"\nCurve -> {out_path}")


def main():
    bars20 = core.build_bars(20)
    bar_ts = bars20.index

    results = []
    for key, cfg in CONFIGS.items():
        td = run_config(cfg, bar_ts)
        report(td, cfg["label"])
        td.to_csv(RESULTS_DIR / f"inspect_v3_{key}_trades.csv", index=False)
        # short label for plot
        short_label = f"N={cfg['norm']} Z={cfg['zlook']} HZ={cfg['hz']}"
        results.append((short_label, td))

    plot_curves(results, RESULTS_DIR / "inspect_v3_curves.png")


if __name__ == "__main__":
    main()
