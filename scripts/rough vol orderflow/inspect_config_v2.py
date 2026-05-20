"""
Deep inspector v2:
- Removes MAX_TRADES_PER_DAY cap (sets to 999)
- Runs the gamma-filtered config AND the no-gamma variant so we can compare
  long/short performance per gamma regime (pos / zero / neg).
- Tags each trade with gamma regime, side, exit reason.
- Outputs: per-hour, per-gamma-regime, per-side cross-tab, worst losses,
  equity curve PNG.
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


# Configs to inspect — both unlimited daily trades
CONFIGS = {
    "N400_max999_g_none": dict(
        label="20m N=400 Z=75 EMA=80 HZ=2.00 SL=2.0 TP=2.0 | window_N8_D150 | g=none | MT=999",
        bm=20, norm=400, zlook=75, ema_len=80, hz=2.00, sl=2.0, tp=2.0,
        of_kind="window", N=8, D=150, gamma_mode=0,
        max_trades=999,
    ),
    "N100_max999_g_drop_pos": dict(
        label="20m N=100 Z=175 EMA=80 HZ=1.90 SL=2.0 TP=2.0 | window_N8_D150 | g=drop_pos | MT=999",
        bm=20, norm=100, zlook=175, ema_len=80, hz=1.90, sl=2.0, tp=2.0,
        of_kind="window", N=8, D=150, gamma_mode=2,
        max_trades=999,
    ),
    "N100_max999_g_none": dict(
        label="20m N=100 Z=175 EMA=80 HZ=1.90 SL=2.0 TP=2.0 | window_N8_D150 | g=none | MT=999",
        bm=20, norm=100, zlook=175, ema_len=80, hz=1.90, sl=2.0, tp=2.0,
        of_kind="window", N=8, D=150, gamma_mode=0,
        max_trades=999,
    ),
}


def run_with_trades(b, z_vol, ema, atr, long_mask, short_mask, g_sign,
                    hz, atr_sl, atr_tp, gamma_mode, max_trades):
    highs = b["highs"]; lows = b["lows"]; closes = b["closes"]
    mod = b["minutes_of_day"]; di = b["day_idx"]
    n = len(closes)
    ss = core.SESSION_START_MIN; se = core.SESSION_END_MIN

    pos = 0
    ep = sl_p = tp_p = 0.0
    cur_day = -1
    dt = 0
    entry_idx = 0
    entry_gsign = 0
    trades = []

    for i in range(n):
        in_session = ss <= mod[i] < se
        if pos != 0 and (not in_session) and mod[i] >= se:
            xp = closes[i]
            trades.append((entry_idx, i, pos, ep, xp, (xp - ep) * pos, "force_close", entry_gsign))
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
                trades.append((entry_idx, i, pos, ep, xp, (xp - ep) * pos, reason, entry_gsign))
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
                    pos = new_dir; ep = cl; entry_idx = i; entry_gsign = int(g_sign[i])
                    if new_dir == 1:
                        sl_p = cl - atr_sl * atr_v; tp_p = cl + atr_tp * atr_v
                    else:
                        sl_p = cl + atr_sl * atr_v; tp_p = cl - atr_tp * atr_v
                    dt += 1
    return trades


def run_config(cfg, bar_ts):
    bm = cfg["bm"]
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
                             cfg["hz"], cfg["sl"], cfg["tp"],
                             cfg["gamma_mode"], cfg["max_trades"])
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


def report(td, label):
    is_end = pd.Timestamp(core.IS_END)
    td["year"] = td["entry_ts"].dt.year
    td["hour"] = td["entry_ts"].dt.hour
    td["regime"] = td["gsign"].map({-1: "NEG", 0: "ZERO", 1: "POS"})
    td["in_is"] = td["entry_ts"].dt.tz_localize(None) <= is_end

    def summary(p):
        if len(p) == 0: return (0, 0.0, 0.0, 0.0)
        w = p[p > 0]; l = p[p < 0]
        pf = w.sum() / abs(l.sum()) if len(l) else 99.0
        wr = 100 * len(w) / len(p)
        return (len(p), pf, wr, p.sum())

    n, pf, wr, pnl = summary(td["pnl_dollars"].to_numpy())
    cum = td["pnl_dollars"].cumsum().to_numpy()
    mdd = (cum - np.maximum.accumulate(cum)).min()
    print(f"\n{'='*100}")
    print(f"{label}")
    print(f"{'='*100}")
    print(f"OVERALL: {n} trades  PF {pf:.2f}  WR {wr:.1f}%  PnL ${pnl:+,.0f}  MDD ${mdd:+,.0f}")

    # IS vs OOS
    for k, mask in (("IS ", td["in_is"]), ("OOS", ~td["in_is"])):
        p = td[mask]["pnl_dollars"].to_numpy()
        sn, spf, swr, spnl = summary(p)
        cum = p.cumsum() if len(p) else np.array([0])
        smdd = (cum - np.maximum.accumulate(cum)).min()
        print(f"  {k}: {sn} trades  PF {spf:.2f}  WR {swr:.1f}%  PnL ${spnl:+,.0f}  MDD ${smdd:+,.0f}")

    print("\n--- Year-by-year ---")
    print(f"{'year':>4} {'tr':>4} {'PF':>5} {'WR':>5} {'PnL$':>10}")
    for y, g in td.groupby("year"):
        sn, spf, swr, spnl = summary(g["pnl_dollars"].to_numpy())
        print(f"{y:>4} {sn:>4} {spf:>5.2f} {swr:>4.1f}% {spnl:>+10,.0f}")

    print("\n--- Hour-by-hour ---")
    print(f"{'hr':>3} {'tr':>4} {'PF':>5} {'WR':>5} {'PnL$':>10}  {'long%':>6}")
    for h, g in td.groupby("hour"):
        sn, spf, swr, spnl = summary(g["pnl_dollars"].to_numpy())
        lp = 100 * (g["side"] == "LONG").sum() / len(g)
        print(f"{h:>3} {sn:>4} {spf:>5.2f} {swr:>4.1f}% {spnl:>+10,.0f}  {lp:>5.1f}%")

    print("\n--- Side x Gamma regime (most important) ---")
    print(f"{'regime':>6} {'side':>5} {'tr':>4} {'PF':>5} {'WR':>5} {'avg':>6} {'PnL$':>10}")
    for reg in ["NEG", "ZERO", "POS"]:
        for side in ["LONG", "SHORT"]:
            g = td[(td["regime"] == reg) & (td["side"] == side)]
            p = g["pnl_dollars"].to_numpy()
            sn, spf, swr, spnl = summary(p)
            avg = p.mean() if len(p) else 0
            print(f"{reg:>6} {side:>5} {sn:>4} {spf:>5.2f} {swr:>4.1f}% {avg:>+6,.0f} {spnl:>+10,.0f}")

    print("\n--- Exit reason ---")
    print(f"{'reason':>12} {'tr':>4} {'WR':>5} {'avg':>7} {'PnL$':>10}")
    for r, g in td.groupby("reason"):
        p = g["pnl_dollars"].to_numpy()
        sn, spf, swr, spnl = summary(p)
        avg = p.mean()
        print(f"{r:>12} {sn:>4} {swr:>4.1f}% {avg:>+7,.0f} {spnl:>+10,.0f}")

    print("\n--- Worst 10 losses ---")
    worst = td.nsmallest(10, "pnl_dollars")[["entry_ts", "side", "reason", "regime", "pnl_dollars"]]
    for _, r in worst.iterrows():
        print(f"  {r['entry_ts']}  {r['side']:>5}  {r['reason']:>11}  g={r['regime']:>4}  ${r['pnl_dollars']:+,.0f}")


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
    ax.set_title("Rough Vol configs — unlimited daily trades")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    print(f"\nCurve -> {out_path}")


def main():
    bars20 = core.build_bars(20)
    bar_ts = bars20.index

    results = {}
    for key, cfg in CONFIGS.items():
        td = run_config(cfg, bar_ts)
        results[key] = (cfg["label"], td)
        report(td, cfg["label"])
        td.to_csv(RESULTS_DIR / f"inspect_v2_{key}_trades.csv", index=False)

    plot_curves(
        [(label, td) for label, td in results.values()],
        RESULTS_DIR / "inspect_v2_curves.png",
    )


if __name__ == "__main__":
    main()
