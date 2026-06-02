"""Quick-scalp ORB sweep for MFF eval pass.

Rules:
  - ORB window: 09:30:00 to 09:59:59 ET (30-min)
  - Entry: short if price breaks ORB_low by 1 tick (= ORB_low - 0.25)
           long  if price breaks ORB_high by 1 tick (= ORB_high + 0.25)
  - One trade per day (first break wins)
  - SL/TP fixed in ticks (TP small, SL large = high WR scalp)
  - Force-close at 15:55 ET if neither hit

Sweep:
  - SL: 100 to 300 ticks, step 10  (= 25 to 75 pts, $500-$1,500 on 1 NQ)
  - TP: 10 to 30 ticks, step 1     (= 2.5 to 7.5 pts, $50-$150 on 1 NQ)
  - 441 cells total

Goal: find highest WR cell that beats both IS+OOS, with realistic exposure.
"""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
import pandas as pd

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ET = "America/New_York"

TICK_SIZE = 0.25
DOLLARS_PER_PT = 5.0 / TICK_SIZE   # 20
SLIP_TICKS_RT = 1   # 1 tick slippage round trip
COMM_RT = 5.0       # $5 round-trip commission

ORB_START_HHMM = 930
ORB_END_HHMM   = 1000
SESSION_END_HHMM = 1555

SL_GRID_TICKS = list(range(100, 301, 10))   # 21 values
TP_GRID_TICKS = list(range(10, 31, 1))      # 21 values

N_WORKERS = 6
_BARS = None


def load_bars():
    df = pd.read_parquet(VOL_PARQUET)
    df["bar_open_time"] = pd.to_datetime(df["bar_open_time"]).dt.tz_convert(ET)
    agg = df.groupby("bar_open_time", as_index=False).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
    )
    agg["close_et"] = agg["bar_open_time"] + pd.Timedelta(minutes=5)
    agg["hhmm"] = agg["close_et"].dt.hour * 100 + agg["close_et"].dt.minute
    agg["session_date"] = agg["close_et"].dt.normalize().dt.tz_localize(None)
    agg = agg.dropna(subset=["open","high","low","close"]).copy()
    agg = agg[agg["session_date"] >= pd.Timestamp("2021-01-01")].copy()
    return agg


def run_strategy(bars, sl_ticks: int, tp_ticks: int):
    sl_pts = sl_ticks * TICK_SIZE
    tp_pts = tp_ticks * TICK_SIZE
    trades = []

    for sd, day in bars.groupby("session_date"):
        orb = day[(day["hhmm"] > ORB_START_HHMM) & (day["hhmm"] <= ORB_END_HHMM)]
        if len(orb) == 0: continue
        orb_high = float(orb["high"].max())
        orb_low  = float(orb["low"].min())

        post = day[(day["hhmm"] > ORB_END_HHMM) & (day["hhmm"] <= SESSION_END_HHMM)].reset_index(drop=True)
        if len(post) == 0: continue

        # Find first break (long if high >= orb_high + 1 tick, short if low <= orb_low - 1 tick)
        long_trigger  = orb_high + TICK_SIZE
        short_trigger = orb_low  - TICK_SIZE

        entry_dir = None  # +1 long, -1 short
        entry_price = None
        entry_idx = None

        for i, bar in post.iterrows():
            hits_long  = bar["high"] >= long_trigger
            hits_short = bar["low"]  <= short_trigger
            if hits_long and hits_short:
                # Same bar, ambiguous — use open to determine which hit first
                # If opens above orb_high → likely long fires immediately
                # If opens below orb_low → likely short
                # Otherwise: use closer trigger to open
                if bar["open"] >= long_trigger:
                    entry_dir, entry_price = +1, long_trigger
                elif bar["open"] <= short_trigger:
                    entry_dir, entry_price = -1, short_trigger
                else:
                    # Pick whichever trigger is closer to the open
                    d_long  = long_trigger  - bar["open"]
                    d_short = bar["open"] - short_trigger
                    if d_long < d_short:
                        entry_dir, entry_price = +1, long_trigger
                    else:
                        entry_dir, entry_price = -1, short_trigger
            elif hits_long:
                entry_dir, entry_price = +1, long_trigger
            elif hits_short:
                entry_dir, entry_price = -1, short_trigger
            if entry_dir is not None:
                entry_idx = i
                break

        if entry_dir is None: continue

        # SL/TP levels
        if entry_dir == +1:
            sl_level = entry_price - sl_pts
            tp_level = entry_price + tp_pts
        else:
            sl_level = entry_price + sl_pts
            tp_level = entry_price - tp_pts

        # Walk subsequent bars to find exit
        exit_price = None; reason = None
        for j in range(entry_idx, len(post)):
            bar = post.iloc[j]
            if entry_dir == +1:
                hit_tp = bar["high"] >= tp_level
                hit_sl = bar["low"]  <= sl_level
            else:
                hit_tp = bar["low"]  <= tp_level
                hit_sl = bar["high"] >= sl_level

            # If both in same bar, assume SL fills first (conservative)
            if hit_sl and hit_tp:
                exit_price, reason = sl_level, "SL_TP"; break
            if hit_sl: exit_price, reason = sl_level, "SL"; break
            if hit_tp: exit_price, reason = tp_level, "TP"; break
            if int(bar["hhmm"]) >= SESSION_END_HHMM:
                exit_price, reason = float(bar["close"]), "EOD"; break
        if exit_price is None:
            last = post.iloc[-1]
            exit_price, reason = float(last["close"]), "EOD_LAST"

        # P&L
        net_pts = entry_dir * (exit_price - entry_price) - SLIP_TICKS_RT * TICK_SIZE
        net_dollars = net_pts * DOLLARS_PER_PT - COMM_RT
        trades.append({
            "session_date": sd, "dir": "L" if entry_dir == +1 else "S",
            "entry": entry_price, "exit": exit_price, "reason": reason,
            "net_pts": net_pts, "net_dollars": net_dollars,
        })

    return pd.DataFrame(trades)


def stats_block(pnls):
    n = len(pnls)
    if n == 0: return dict(n=0, wr=0, net=0, pf=0, mdd=0, avg=0)
    w = pnls[pnls > 0]; l = pnls[pnls < 0]
    pf = w.sum() / abs(l.sum()) if len(l) > 0 else 99.0
    cum = pnls.cumsum()
    mdd = float((cum - np.maximum.accumulate(cum)).min())
    return dict(n=n, wr=round(len(w)/n*100, 1), net=round(float(pnls.sum()), 0),
                pf=round(pf, 3), mdd=round(mdd, 0), avg=round(float(pnls.mean()), 1))


def _init_worker(bars):
    global _BARS
    _BARS = bars


def _run_cell(args):
    sl, tp = args
    trades = run_strategy(_BARS, sl, tp)
    if len(trades) == 0:
        return {"sl_ticks": sl, "tp_ticks": tp, "n_all": 0}
    trades = trades.sort_values("session_date").reset_index(drop=True)
    pnls = trades["net_dollars"].values
    dates = sorted(trades["session_date"].unique())
    cutoff = dates[int(len(dates) * 0.6)] if len(dates) > 1 else dates[-1]
    is_mask  = trades["session_date"] <  cutoff
    oos_mask = trades["session_date"] >= cutoff
    s_all = stats_block(pnls)
    s_is  = stats_block(trades.loc[is_mask,  "net_dollars"].values)
    s_oos = stats_block(trades.loc[oos_mask, "net_dollars"].values)
    return {
        "sl_ticks": sl, "tp_ticks": tp,
        "n_all": s_all["n"], "wr_all": s_all["wr"], "all_net": s_all["net"],
        "all_PF": s_all["pf"], "all_mdd": s_all["mdd"], "all_avg": s_all["avg"],
        "is_wr": s_is["wr"], "is_net": s_is["net"], "is_PF": s_is["pf"], "is_mdd": s_is["mdd"],
        "oos_wr": s_oos["wr"], "oos_net": s_oos["net"], "oos_PF": s_oos["pf"], "oos_mdd": s_oos["mdd"],
    }


def main():
    print("Loading bars...")
    t0 = time.time()
    bars = load_bars()
    print(f"  {len(bars):,} bars  ({time.time()-t0:.1f}s)")

    configs = [(sl, tp) for sl in SL_GRID_TICKS for tp in TP_GRID_TICKS]
    print(f"\nSweeping {len(configs)} cells on {N_WORKERS} workers...")
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker, initargs=(bars,)) as ex:
        rows = list(ex.map(_run_cell, configs))
    print(f"Done in {time.time()-t1:.1f}s")

    df = pd.DataFrame(rows)
    df["beat_both"] = (df["is_net"] > 0) & (df["oos_net"] > 0)
    df["wr_min"] = df[["is_wr", "oos_wr"]].min(axis=1)
    df.to_csv(OUT_DIR / "quick_scalp_orb_sweep.csv", index=False)

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)

    cols = ["sl_ticks","tp_ticks","n_all","wr_all","all_net","all_PF","all_mdd","is_wr","oos_wr","is_net","oos_net"]

    print(f"\n=== TOP 20 by ALL WR (must beat both IS+OOS net) ===")
    print(df[df["beat_both"]].sort_values("wr_all", ascending=False).head(20)[cols].to_string(index=False))

    print(f"\n=== TOP 20 by MIN(IS_WR, OOS_WR) - most robust high-WR ===")
    print(df[df["beat_both"]].sort_values("wr_min", ascending=False).head(20)[cols].to_string(index=False))

    print(f"\n=== TOP 10 by ALL NET $ (must beat both IS+OOS) ===")
    print(df[df["beat_both"]].sort_values("all_net", ascending=False).head(10)[cols].to_string(index=False))

    # Specific MFF angle: $200 in one go = $200 per trade if 1 NQ
    print(f"\n=== HEATMAP: ALL WR % by (sl x tp) ===")
    pivot_wr = df.pivot(index="sl_ticks", columns="tp_ticks", values="wr_all")
    print(pivot_wr.round(1).to_string())

    # Find the highest WR config and report its trade-by-trade for MFF context
    best = df[df["beat_both"]].sort_values("wr_min", ascending=False).iloc[0]
    print(f"\n=== BEST robust HIGH-WR config ===")
    print(f"  sl={int(best['sl_ticks'])} ticks (${int(best['sl_ticks'])*0.25*20:.0f} risk on 1 NQ)")
    print(f"  tp={int(best['tp_ticks'])} ticks (${int(best['tp_ticks'])*0.25*20:.0f} profit per win on 1 NQ)")
    print(f"  WR: all={best['wr_all']}%  is={best['is_wr']}%  oos={best['oos_wr']}%")
    print(f"  Net: all=${best['all_net']:,.0f}  is=${best['is_net']:,.0f}  oos=${best['oos_net']:,.0f}")
    print(f"  MDD: ${best['all_mdd']:,.0f}")

    # For MFF $200 pass: how many trades / how many contracts needed
    tp_dollars_1c = int(best['tp_ticks']) * 0.25 * 20
    print(f"\n  To pass MFF $200 target:")
    print(f"    1 NQ: 1 winning trade = ${tp_dollars_1c:.0f}")
    if tp_dollars_1c >= 200:
        print(f"    1 NQ alone hits $200 in 1 trade. Eval pass prob = {best['wr_all']}% in one day")
    else:
        n_wins_needed = int(np.ceil(200 / tp_dollars_1c))
        print(f"    Need {n_wins_needed} wins to hit $200 on 1 NQ")
        # Approx pass prob: WR^n if all-or-nothing
        from math import comb
        wr = best['wr_all'] / 100
        # P(at least n_wins in N attempts before sl wipes)
        # Simpler: with sl much larger than tp, can absorb losses
        # E.g., sl=200 ticks = $1K, account DD = $2K, so 2 sl losses = blow
        # For 200/200 sl: blow after 2 losses (assuming acc starts at 0)
        sl_dollars = int(best['sl_ticks']) * 0.25 * 20
        max_losses_before_blow = int(2000 / sl_dollars)  # rough
        print(f"    SL hits/account DD limit: {max_losses_before_blow}")
        print(f"    Need {n_wins_needed} wins without {max_losses_before_blow} losses")

    # Show MNQ math too
    tp_dollars_1mnq = tp_dollars_1c / 10
    sl_dollars_1mnq = int(best['sl_ticks']) * 0.25 * 20 / 10
    print(f"\n  At 1 MNQ (1/10 size):")
    print(f"    Per win: ${tp_dollars_1mnq:.0f}")
    print(f"    Per SL:  ${sl_dollars_1mnq:.0f}")


if __name__ == "__main__":
    main()
