"""Does anything beat the ACTUAL live B2 config?

Holds the live entry stack completely fixed (B2 variant, pinbar X=0.75,
|abs_delta_w15|>=70, strict shorts, conf_half_w5>=75, chained dedupe, hours 9-14,
drop POS-gamma shorts) and sweeps only the EXIT parameters that live/combined/
b2_engine.py hard-codes:

    YMULT     yellow ratchet distance  (live 2.50 x ATR_14_20min)
    TPMULT    fixed green TP           (live 2.00 x ATR_at_entry)
    MFE_K     MFE-guard arming trigger (live 0.80 x TP distance)
    MFE_LOCK  locked giveback fraction (live 0.45 x peak MFE)

Same 20-min bar builder and same exit loop order as the locked backtest, so the
(2.50, 2.00, 0.80, 0.45) cell must reproduce the live baseline exactly.

Reason for running this: the older ratchet_sl_fixed_tp_sweep.txt (pre-MFE-lock,
data only to 2025-11) ranked tp_mult 2.50 above 2.00, but live is locked at 2.00.
This re-tests on the full 2020-12 -> 2026-06 span WITH the MFE lock in place.

Output -> tradelogs/naked_break/live_b2_exit_param_sweep.txt
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from sweep_ratchet_sl_fixed_tp import filter_pre_dedupe          # noqa: E402
from test_pure_ratchet_exits import build_20min_bars, FORCE_CLOSE_TIME  # noqa: E402
import lock_v2_k08_lock045_mart_fc_filtered as LOCK              # noqa: E402

PARQUETS = HERE / "parquets"
OUT = HERE.parent / "tradelogs" / "naked_break" / "live_b2_exit_param_sweep.txt"

LIVE = (2.50, 2.00, 0.80, 0.45)      # YMULT, TPMULT, MFE_K, MFE_LOCK
IS_END = pd.Timestamp("2024-12-31").date()

YMULTS = [1.5, 2.0, 2.5, 3.0, 3.5]
TPMULTS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
# (mfe_k, mfe_lock); mfe_k = None -> MFE guard disabled (pure ratchet)
MFES = [(None, None), (0.6, 0.45), (0.8, 0.30), (0.8, 0.45), (0.8, 0.60), (1.0, 0.45)]


def simulate_exit(direction, entry_ts, entry_price, bars20, ymult, tpmult, mfe_k, mfe_lock):
    """Verbatim simulate_exit_v2 from the locked backtest, with the four
    hard-coded multipliers lifted into arguments."""
    sign = 1 if direction == "LONG" else -1
    bars_idx = bars20.index
    start = bars_idx.searchsorted(entry_ts, side="right")
    if start >= len(bars_idx):
        return None
    ent_date = entry_ts.date()
    end = start
    while end < len(bars_idx) and bars_idx[end].date() == ent_date:
        end += 1
    if end == start:
        return None
    init_idx = start - 1
    if init_idx < 0 or np.isnan(bars20["atr_y"].iloc[init_idx]):
        return None
    init_atr_y = float(bars20["atr_y"].iloc[init_idx])
    yellow_val = entry_price - sign * ymult * init_atr_y
    prev_yellow = yellow_val
    o = bars20["open"].values[start:end]; h = bars20["high"].values[start:end]
    l = bars20["low"].values[start:end];  c = bars20["close"].values[start:end]
    ay = bars20["atr_y"].values[start:end]; ts_arr = bars_idx[start:end]
    green_val = entry_price + sign * tpmult * init_atr_y
    tp_dist = abs(green_val - entry_price)
    mfe_so_far = 0.0
    mae = 0.0
    for i in range(end - start):
        bar_close_ts = ts_arr[i] + pd.Timedelta(minutes=20)
        cur_mfe = (h[i] - entry_price) if sign > 0 else (entry_price - l[i])
        if cur_mfe > mfe_so_far:
            mfe_so_far = cur_mfe
        cur_mae = (l[i] - entry_price) if sign > 0 else (entry_price - h[i])
        if cur_mae < mae:
            mae = cur_mae
        if not np.isnan(ay[i]):
            raw_yellow = c[i] - sign * ymult * ay[i]
            yellow_val = max(prev_yellow, raw_yellow) if sign > 0 else min(prev_yellow, raw_yellow)
        if mfe_k is not None and mfe_so_far >= mfe_k * tp_dist:
            mfe_stop = entry_price + sign * mfe_lock * mfe_so_far
            stop_level = max(yellow_val, mfe_stop) if sign > 0 else min(yellow_val, mfe_stop)
        else:
            stop_level = yellow_val
        if sign > 0 and h[i] >= green_val:
            return (sign * (green_val - entry_price), "TP_FIXED", bar_close_ts, mae)
        if sign < 0 and l[i] <= green_val:
            return (sign * (green_val - entry_price), "TP_FIXED", bar_close_ts, mae)
        if sign > 0 and c[i] <= stop_level and c[i] < o[i]:
            return (c[i] - entry_price, "SL_TRAIL", bar_close_ts, mae)
        if sign < 0 and c[i] >= stop_level and c[i] > o[i]:
            return (entry_price - c[i], "SL_TRAIL", bar_close_ts, mae)
        if ts_arr[i].time() >= FORCE_CLOSE_TIME:
            return (sign * (c[i] - entry_price), "FORCE_CLOSE", bar_close_ts, mae)
        prev_yellow = yellow_val
    return (sign * (c[-1] - entry_price), "EOD", ts_arr[-1] + pd.Timedelta(minutes=20), mae)


def run(cands, bars20, ymult, tpmult, mfe_k, mfe_lock):
    """Chained dedupe by realised exit time (same rule as the locked backtest)."""
    rows = []
    last_exit = pd.Timestamp(0, tz="America/New_York")
    for _, row in cands.iterrows():
        if row["entry_time_et"] <= last_exit:
            continue
        ex = simulate_exit(row["direction"], row["entry_time_et"],
                           float(row["entry_price"]), bars20, ymult, tpmult, mfe_k, mfe_lock)
        if ex is None:
            continue
        pnl, reason, exit_ts, mae = ex
        rows.append(dict(date=row["entry_time_et"].date(), direction=row["direction"],
                         pnl=pnl, reason=reason, mae=mae,
                         entry_hour=row["entry_time_et"].hour))
        last_exit = exit_ts
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["period"] = np.where(df["date"] <= IS_END, "IS", "OOS")
    df["year"] = pd.to_datetime(df["date"]).dt.year
    return df


def stats(df):
    if df.empty:
        return dict(n=0, net=0.0, pf=0.0, wr=0.0, sharpe=0.0, mdd=0.0,
                    worst=0.0, worst_mae=0.0, tp=0, sl=0, fc=0)
    p = df["pnl"].values
    w, lo = p[p > 0], p[p < 0]
    daily = pd.Series(p, index=pd.to_datetime(df["date"].values)).groupby(level=0).sum()
    eq = np.cumsum(p)
    return dict(n=len(p), net=p.sum(),
                pf=(w.sum() / abs(lo.sum())) if lo.size and lo.sum() != 0 else float("inf"),
                wr=(p > 0).mean() * 100,
                sharpe=(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0,
                mdd=(eq - np.maximum.accumulate(eq)).min(),
                worst=p.min(), worst_mae=df["mae"].min(),
                tp=int((df["reason"] == "TP_FIXED").sum()),
                sl=int((df["reason"] == "SL_TRAIL").sum()),
                fc=int((df["reason"].isin(["FORCE_CLOSE", "EOD"])).sum()))


def main():
    t0 = _time.time()
    bars20 = build_20min_bars()
    parts = []
    for f in ("entry_signal_trades.parquet", "entry_signal_trades_oos.parquet"):
        c = filter_pre_dedupe(pd.read_parquet(PARQUETS / f))
        c = LOCK.attach_gamma_to_candidates(c)
        c, _ = LOCK.filter_candidates(c)
        parts.append(c)
    cands = pd.concat(parts, ignore_index=True).sort_values("entry_time_et").reset_index(drop=True)
    print(f"{len(cands)} post-filter entry candidates")

    rows = []
    for ym in YMULTS:
        for tm in TPMULTS:
            for mk, ml in MFES:
                df = run(cands, bars20, ym, tm, mk, ml)
                s = stats(df)
                i = stats(df[df["period"] == "IS"]); o = stats(df[df["period"] == "OOS"])
                yrs = df.groupby("year")["pnl"].sum()
                rows.append(dict(
                    ymult=ym, tpmult=tm, mfe=("off" if mk is None else f"{mk}/{ml}"),
                    label=f"y{ym:g} tp{tm:g} mfe{'off' if mk is None else f'{mk:g}/{ml:g}'}",
                    is_live=(ym, tm, mk, ml) == LIVE,
                    **{k: s[k] for k in ("n", "net", "pf", "wr", "sharpe", "mdd",
                                         "worst", "worst_mae", "tp", "sl", "fc")},
                    is_net=i["net"], is_pf=i["pf"], is_sharpe=i["sharpe"],
                    oos_net=o["net"], oos_pf=o["pf"], oos_sharpe=o["sharpe"],
                    min_net=min(i["net"], o["net"]), min_pf=min(i["pf"], o["pf"]),
                    min_sharpe=min(i["sharpe"], o["sharpe"]),
                    yrs_pos=int((yrs > 0).sum()), yrs_tot=int(len(yrs))))
        print(f"  ymult {ym} done ({_time.time()-t0:.0f}s)")

    g = pd.DataFrame(rows)
    g.to_csv(OUT.parent / "live_b2_exit_param_grid.csv", index=False)
    live = g[g.is_live].iloc[0]

    def table(sub, title, sort, n=None):
        L = [title,
             f"  {'#':>3} {'config':<24} {'n':>4} {'net':>8} {'$MNQ':>9} {'pf':>5} {'wr':>6} "
             f"{'shrp':>5} {'mdd':>8} {'is_net':>8} {'is_shrp':>7} {'oos_net':>8} "
             f"{'oos_shrp':>8} {'minShrp':>7} {'yrs+':>5} {'wMAE':>7}"]
        s = sub.sort_values(sort, ascending=False)
        if n:
            s = s.head(n)
        for j, (_, r) in enumerate(s.iterrows(), 1):
            tag = " <<< LIVE" if r.is_live else ""
            L.append(f"  {j:>3} {r.label:<24} {r.n:>4} {r.net:>+8.1f} {r.net*2:>+9,.0f} "
                     f"{r.pf:>5.2f} {r.wr:>5.1f}% {r.sharpe:>5.2f} {r['mdd']:>+8.1f} "
                     f"{r.is_net:>+8.1f} {r.is_sharpe:>7.2f} {r.oos_net:>+8.1f} "
                     f"{r.oos_sharpe:>8.2f} {r.min_sharpe:>7.2f} "
                     f"{r.yrs_pos:>2}/{r.yrs_tot:<2} {r.worst_mae:>+7.1f}{tag}")
        return L

    L = ["=" * 175,
         "LIVE B2 EXIT-PARAMETER SWEEP — entry stack frozen, only YMULT/TPMULT/MFE vary",
         "=" * 175, "",
         f"  candidates {len(cands)}   |   {len(g)} cells   |   "
         f"IS <= {IS_END}, OOS after   |   1 NQ contract, no costs", "",
         f"  LIVE CELL (y2.5 tp2.0 mfe0.8/0.45): n={live.n}  net={live.net:+.1f} pts "
         f"(${live.net*2:+,.0f} MNQ)  pf={live.pf:.3f}  wr={live.wr:.1f}%  "
         f"sharpe={live.sharpe:.2f}  mdd={live['mdd']:+.1f}",
         f"    -> reproduces the locked backtest's +4930.7 pts / 60.3% WR: "
         f"{'YES' if abs(live.net-4930.7) < 1 else 'NO — parity broken, do not trust this run'}",
         ""]
    beat_all = g[(g.net > live.net) & (g.pf > live.pf) & (g.sharpe > live.sharpe) &
                 (g['mdd'] > live['mdd']) & (g.is_net > 0) & (g.oos_net > 0)]
    L += [f"  cells beating LIVE on net: {int((g.net > live.net).sum())}   "
          f"on Sharpe: {int((g.sharpe > live.sharpe).sum())}   "
          f"on MDD: {int((g['mdd'] > live['mdd']).sum())}   "
          f"on PF: {int((g.pf > live.pf).sum())}",
          f"  cells DOMINATING live on all four (+IS>0, OOS>0): {len(beat_all)}", ""]
    if len(beat_all):
        L += table(beat_all, "DOMINATING CELLS (sorted by min IS/OOS Sharpe):", "min_sharpe")
        L.append("")
    L += table(g, "TOP 20 by min(IS,OOS) Sharpe:", "min_sharpe", 20)
    L.append("")
    L += table(g, "TOP 20 by net:", "net", 20)
    L.append("")
    L += table(g, "TOP 15 by smallest MDD:", "mdd", 15)
    L.append("")
    L.append("MARGINALS (median net / median Sharpe over cells holding that value):")
    for dim in ("ymult", "tpmult", "mfe"):
        m = g.groupby(dim).agg(net=("net", "median"), shrp=("sharpe", "median"))
        L.append(f"  {dim:<8} " + "   ".join(
            f"{k}: {r.net:+.0f}/{r.shrp:.2f}" for k, r in m.iterrows()))
    L.append("")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {OUT}\ntotal {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
