"""
Overfit Test Framework — Fabio ORB (locked config).

Locked: N=4 confirmation + skip 9:30 + delta>=300, TP=4R, EOD@14:00 ET, long-only.

Runs all 5 tests from docs/OVERFIT_TEST_FRAMEWORK.md:
  1. Parameter Stability  — 3D neighborhood around (N, delta, TP)
  2. Walk-Forward         — non-overlapping 6-mo windows
  3. MC Order Shuffle     — 10,000 permutations of trade order
  4. Bootstrap CI         — 10,000 resamples with replacement
  5. Direction Permutation— random long/short per trade, 1,000 perms (strongest)
"""
from __future__ import annotations
import warnings; warnings.simplefilter("ignore")
from pathlib import Path
import numpy as np, pandas as pd

VOL_PARQUET = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
OUT_DIR = Path("D:/trading_pythonbacktest_data/fabio orb")
ET = "America/New_York"
ORB_START_HHMM, ORB_END_HHMM, TRADE_END_HHMM = 830, 900, 1400
SKIP_BUCKET = 930

TICK_SIZE, TICK_VALUE = 0.25, 5.0
DPP = TICK_VALUE / TICK_SIZE
SLIP_PTS = 1 * TICK_SIZE
COMM = 5.0

# LOCKED PARAMETERS
LOCKED_N = 4
LOCKED_DTHR = 300.0
LOCKED_TP = 4.0


# ============================== bar loading ==============================

def load_days():
    df = pd.read_parquet(VOL_PARQUET)
    agg = df.groupby("bar_open_time", as_index=False).agg(
        open=("open","first"), high=("high","max"), low=("low","min"),
        close=("close","last"), buy_vol=("buy_vol","sum"), sell_vol=("sell_vol","sum"),
    ).dropna(subset=["open","high","low","close"])
    agg["close_et"] = agg["bar_open_time"] + pd.Timedelta(minutes=5)
    agg["hhmm"] = agg["close_et"].dt.hour*100 + agg["close_et"].dt.minute
    agg = agg[(agg["hhmm"] > ORB_START_HHMM) & (agg["hhmm"] <= TRADE_END_HHMM)].copy()
    agg["delta"] = agg["buy_vol"] - agg["sell_vol"]
    agg["session_date"] = agg["close_et"].dt.normalize()
    days = {}
    for sd, sub in agg.groupby("session_date"):
        sub = sub.sort_values("close_et").reset_index(drop=True)
        in_orb = sub[(sub["hhmm"] > ORB_START_HHMM) & (sub["hhmm"] <= ORB_END_HHMM)]
        if in_orb.empty: continue
        post = sub[(sub["hhmm"] > ORB_END_HHMM) & (sub["hhmm"] <= TRADE_END_HHMM)]
        if post.empty: continue
        days[pd.Timestamp(sd).tz_localize(None)] = {
            "orb_high": float(in_orb["high"].max()),
            "orb_low":  float(in_orb["low"].min()),
            "hhmm":  post["hhmm"].to_numpy(),
            "high":  post["high"].to_numpy(dtype=np.float64),
            "low":   post["low"].to_numpy(dtype=np.float64),
            "close": post["close"].to_numpy(dtype=np.float64),
            "delta": post["delta"].to_numpy(dtype=np.float64),
            "close_et": post["close_et"].to_numpy(),
        }
    return days


# ============================== strategy run ==============================

def find_entry(day, n_confirm=LOCKED_N, delta_thr=LOCKED_DTHR):
    oh = day["orb_high"]; ol = day["orb_low"]
    hhmm = day["hhmm"]; close = day["close"]; delta = day["delta"]
    etime = day["close_et"]; n = len(hhmm)
    for i in range(n_confirm - 1, n):
        if hhmm[i] > TRADE_END_HHMM: break
        if hhmm[i] == SKIP_BUCKET: continue
        if not all(close[i-k] > oh for k in range(n_confirm)): continue
        if delta[i] < delta_thr: continue
        ep = close[i]
        if ol >= ep: continue
        return i, ep, ol, ep + LOCKED_TP*(ep-ol), etime[i]
    return None


def simulate_trade(day, entry_idx, entry_price, sl, tp, etime0, direction=1):
    """Direction: +1 long, -1 short (with mirrored SL/TP)."""
    hhmm, high, low, close, etime = day["hhmm"], day["high"], day["low"], day["close"], day["close_et"]
    if direction == 1:
        long_sl = sl; long_tp = tp
    else:
        # short: mirror SL/TP around entry
        risk = entry_price - sl
        long_sl = entry_price + risk     # short SL (above entry)
        long_tp = entry_price - LOCKED_TP * risk  # short TP (below entry)

    for j in range(entry_idx + 1, len(hhmm)):
        if hhmm[j] >= TRADE_END_HHMM:
            xp = close[j]; reason = "EOD"; xt = etime[j]; break
        if direction == 1:
            hit_sl = low[j] <= long_sl; hit_tp = high[j] >= long_tp
        else:
            hit_sl = high[j] >= long_sl; hit_tp = low[j] <= long_tp
        if hit_sl and hit_tp:
            xp = long_sl; reason = "SL_TP"; xt = etime[j]; break
        if hit_sl:
            xp = long_sl; reason = "SL"; xt = etime[j]; break
        if hit_tp:
            xp = long_tp; reason = "TP"; xt = etime[j]; break
    else:
        xp = close[-1]; reason = "EOD_LAST"; xt = etime[-1]

    raw = (xp - entry_price) * direction
    gross = raw * DPP
    net = gross - 2 * SLIP_PTS * DPP - COMM
    return {
        "entry_time": pd.Timestamp(etime0),
        "exit_time": pd.Timestamp(xt),
        "entry": entry_price, "exit": xp, "sl": long_sl, "tp": long_tp,
        "direction": direction,
        "risk_pts": entry_price - sl,  # always positive (long-defined)
        "raw_pts": raw, "gross_dollars": gross, "net_dollars": net,
        "reason": reason,
    }


def run_strategy(days, n_confirm=LOCKED_N, delta_thr=LOCKED_DTHR):
    trades = []
    entries_for_perm = []   # for the direction permutation test
    for d in sorted(days.keys()):
        ent = find_entry(days[d], n_confirm=n_confirm, delta_thr=delta_thr)
        if ent is None: continue
        i, ep, sl, tp, etime0 = ent
        t = simulate_trade(days[d], i, ep, sl, tp, etime0, direction=1)
        trades.append(t)
        # also pre-compute short-direction outcome for the same entry
        ts = simulate_trade(days[d], i, ep, sl, tp, etime0, direction=-1)
        entries_for_perm.append((t["net_dollars"], ts["net_dollars"]))
    return trades, entries_for_perm


# ============================== stats helpers ==============================

def perf(pnls: np.ndarray) -> dict:
    n = len(pnls)
    if n == 0: return {"n":0,"net":0,"PF":float("nan"),"win_rate":float("nan"),"maxdd":0}
    wins = int((pnls > 0).sum())
    wd = pnls[pnls>0].sum(); ld = -pnls[pnls<0].sum()
    pf = wd/ld if ld > 0 else float("inf")
    eq = np.cumsum(pnls)
    dd = (eq - np.maximum.accumulate(eq)).min()
    return {"n":n, "net":float(pnls.sum()), "PF":float(pf),
            "win_rate":wins/n, "maxdd":float(dd)}


# ============================== Test 1: Parameter Stability ==============================

def test_1_parameter_stability(days):
    print("\n" + "="*78)
    print("TEST 1: Parameter Stability  (3D neighborhood around locked N=4, delta=300, TP=4.0)")
    print("="*78)
    Ns = [3, 4, 5]
    Ds = [275, 300, 325]
    TPs = [3.5, 4.0, 4.5]
    cells = []
    for N in Ns:
        for D in Ds:
            for TP in TPs:
                # set TP via closure trick — easier: temporarily override LOCKED_TP via the find_entry's calc
                trades = []
                for d in sorted(days.keys()):
                    day = days[d]
                    ent = find_entry(day, n_confirm=N, delta_thr=D)
                    if ent is None: continue
                    i, ep, sl, _tp, etime0 = ent
                    tp = ep + TP * (ep - sl)
                    t = simulate_trade(day, i, ep, sl, tp, etime0, direction=1)
                    trades.append(t)
                pnls = np.array([t["net_dollars"] for t in trades])
                s = perf(pnls)
                marker = " *LOCKED*" if (N==LOCKED_N and D==LOCKED_DTHR and TP==LOCKED_TP) else ""
                cells.append({
                    "N":N, "delta":D, "TP":TP,
                    "n":s["n"], "win_rate":s["win_rate"], "PF":s["PF"],
                    "net":s["net"], "maxdd":s["maxdd"], "tag":marker.strip()
                })
    df = pd.DataFrame(cells)
    df["win_rate"] = (df["win_rate"]*100).round(1)
    df["PF"] = df["PF"].round(3)
    df["net"] = df["net"].round(0)
    df["maxdd"] = df["maxdd"].round(0)
    print(df.to_string(index=False))
    # Pass criterion: all 26 neighbors profitable
    n_profitable = (df["net"] > 0).sum()
    print(f"\n  Profitable cells: {n_profitable}/{len(df)}")
    locked = df[(df["N"]==LOCKED_N)&(df["delta"]==LOCKED_DTHR)&(df["TP"]==LOCKED_TP)].iloc[0]
    others = df[~((df["N"]==LOCKED_N)&(df["delta"]==LOCKED_DTHR)&(df["TP"]==LOCKED_TP))]
    print(f"  Locked net: ${locked['net']:,.0f}.   Neighbors median: ${others['net'].median():,.0f}   neighbors p25: ${others['net'].quantile(0.25):,.0f}")
    passed = n_profitable >= len(df) - 1  # tolerate 1 fail
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  (need >= {len(df)-1} of {len(df)} profitable)")
    return passed


# ============================== Test 2: Walk-Forward ==============================

def test_2_walk_forward(trades):
    print("\n" + "="*78)
    print("TEST 2: Walk-Forward  (non-overlapping 6-month windows)")
    print("="*78)
    df = pd.DataFrame(trades)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["date"] = df["entry_time"].dt.tz_convert(ET).dt.date
    # 6-month chunks
    start = pd.Timestamp(df["date"].min())
    end   = pd.Timestamp(df["date"].max())
    cur = start
    rows = []
    while cur < end:
        w_end = cur + pd.DateOffset(months=6)
        s = df[(df["date"] >= cur.date()) & (df["date"] < w_end.date())]
        if len(s) == 0:
            cur = w_end; continue
        pnls = s["net_dollars"].to_numpy()
        m = perf(pnls)
        rows.append({
            "window_start": cur.date(),
            "window_end":   min(w_end.date(), end.date()),
            "n": m["n"], "win_rate": m["win_rate"],
            "PF": m["PF"], "net": m["net"], "maxdd": m["maxdd"],
        })
        cur = w_end
    w = pd.DataFrame(rows)
    w["win_rate"] = (w["win_rate"]*100).round(1)
    w["PF"] = w["PF"].round(3)
    w["net"] = w["net"].round(0)
    w["maxdd"] = w["maxdd"].round(0)
    print(w.to_string(index=False))
    n_profit = (w["net"] > 0).sum()
    print(f"\n  Profitable windows: {n_profit}/{len(w)}")
    biggest_loss = w["net"].min()
    biggest_win = w["net"].max()
    print(f"  Worst window: ${biggest_loss:,.0f}   Best window: ${biggest_win:,.0f}   Ratio: {abs(biggest_loss)/biggest_win:.2f}")
    passed = n_profit >= 0.75 * len(w) and biggest_loss > -10000
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  (need >=75% profitable + no catastrophic <-$10k window)")
    return passed


# ============================== Test 3: MC Shuffle ==============================

def test_3_mc_shuffle(trades, n_sims=10000, rng_seed=42):
    print("\n" + "="*78)
    print(f"TEST 3: MC Order Shuffling  ({n_sims:,} permutations)")
    print("="*78)
    pnls = np.array([t["net_dollars"] for t in trades])
    n = len(pnls)
    real_eq = np.cumsum(pnls)
    real_dd = (real_eq - np.maximum.accumulate(real_eq)).min()
    print(f"  Real MaxDD: ${real_dd:,.0f}   (n={n} trades)")

    rng = np.random.default_rng(rng_seed)
    dds = np.empty(n_sims)
    for k in range(n_sims):
        perm = rng.permutation(pnls)
        eq = np.cumsum(perm)
        dds[k] = (eq - np.maximum.accumulate(eq)).min()
    # pct of shuffles whose DD is WORSE (more negative) than real
    worse = (dds < real_dd).sum()
    pct_better = 100 * worse / n_sims
    p5, p50, p95 = np.percentile(dds, [5, 50, 95])
    print(f"  Shuffled DD distribution:")
    print(f"    p5:   ${p5:,.0f}    median: ${p50:,.0f}    p95: ${p95:,.0f}")
    print(f"  Real DD is better than {pct_better:.1f}% of random orderings")
    passed = pct_better >= 50
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  (need real DD better than >=50% of shuffles)")
    return passed


# ============================== Test 4: Bootstrap ==============================

def test_4_bootstrap(trades, n_resamples=10000, rng_seed=43):
    print("\n" + "="*78)
    print(f"TEST 4: Bootstrap Confidence Intervals  ({n_resamples:,} resamples)")
    print("="*78)
    pnls = np.array([t["net_dollars"] for t in trades])
    n = len(pnls)
    real_total = pnls.sum()
    print(f"  Real total: ${real_total:,.0f}   Real avg: ${pnls.mean():.0f}   n={n}")

    rng = np.random.default_rng(rng_seed)
    sums = np.empty(n_resamples)
    pfs = np.empty(n_resamples)
    for k in range(n_resamples):
        sample = rng.choice(pnls, size=n, replace=True)
        sums[k] = sample.sum()
        wd = sample[sample>0].sum(); ld = -sample[sample<0].sum()
        pfs[k] = wd/ld if ld > 0 else float("inf")
    p_lose = 100 * (sums < 0).sum() / n_resamples
    sum_ci = np.percentile(sums, [2.5, 50, 97.5])
    pf_ci  = np.percentile(pfs, [2.5, 50, 97.5])
    print(f"  Total PnL  95% CI: [${sum_ci[0]:,.0f},  ${sum_ci[2]:,.0f}]   median: ${sum_ci[1]:,.0f}")
    print(f"  PF         95% CI: [{pf_ci[0]:.3f},  {pf_ci[2]:.3f}]   median: {pf_ci[1]:.3f}")
    print(f"  P(losing strategy): {p_lose:.2f}%")
    passed = (p_lose < 1.0) and (sum_ci[0] > 0) and (pf_ci[0] >= 1.0)
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  (need P(loss)<1%, 95% PnL CI lower >0, PF CI lower >=1.0)")
    return passed


# ============================== Test 5: Direction Permutation ==============================

def test_5_direction_permutation(entries_for_perm, real_total, n_perms=1000, rng_seed=44):
    print("\n" + "="*78)
    print(f"TEST 5: Direction Permutation  ({n_perms:,} random long/short assignments)")
    print("="*78)
    longs  = np.array([e[0] for e in entries_for_perm])
    shorts = np.array([e[1] for e in entries_for_perm])
    n = len(longs)
    print(f"  Real (all long): ${real_total:,.0f}   n={n}")
    print(f"  Hypothetical (all short): ${shorts.sum():,.0f}")

    rng = np.random.default_rng(rng_seed)
    sums = np.empty(n_perms)
    for k in range(n_perms):
        coin = rng.integers(0, 2, size=n)  # 0=short, 1=long
        sums[k] = np.where(coin==1, longs, shorts).sum()
    pct_real_beats = 100 * (sums < real_total).sum() / n_perms
    p_med = np.median(sums); p25, p75 = np.percentile(sums, [25, 75]); p5, p95 = np.percentile(sums, [5, 95])
    print(f"  Perm distribution: 5%={p5:,.0f}  25%={p25:,.0f}  median={p_med:,.0f}  75%={p75:,.0f}  95%={p95:,.0f}")
    print(f"  Real PnL beats {pct_real_beats:.1f}% of random direction assignments")
    p_value = (n_perms - (sums < real_total).sum()) / n_perms   # one-tailed
    print(f"  One-tailed p-value: {p_value:.4f}")
    passed = pct_real_beats >= 99.0
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}  (need real PnL in top 1% — p<0.01)")
    return passed


# ============================== main ==============================

def main():
    print("Loading bars...", flush=True)
    days = load_days()
    print(f"  {len(days)} session days")

    print("\nRunning locked strategy (Mode A)...")
    trades, entries_for_perm = run_strategy(days)
    pnls = np.array([t["net_dollars"] for t in trades])
    perf_summary = perf(pnls)
    print(f"  Trades: {perf_summary['n']}   Win rate: {100*perf_summary['win_rate']:.1f}%   "
          f"Net: ${perf_summary['net']:,.0f}   PF: {perf_summary['PF']:.3f}   MaxDD: ${perf_summary['maxdd']:,.0f}")

    results = {}
    results["1_parameter_stability"]   = test_1_parameter_stability(days)
    results["2_walk_forward"]          = test_2_walk_forward(trades)
    results["3_mc_shuffle"]            = test_3_mc_shuffle(trades)
    results["4_bootstrap"]             = test_4_bootstrap(trades)
    results["5_direction_permutation"] = test_5_direction_permutation(entries_for_perm, perf_summary["net"])

    print("\n" + "="*78)
    print("SUMMARY")
    print("="*78)
    for k, v in results.items():
        print(f"  {k:30}  {'PASS' if v else 'FAIL'}")
    n_pass = sum(results.values())
    print(f"\n  {n_pass}/5 tests passed")


if __name__ == "__main__":
    main()
