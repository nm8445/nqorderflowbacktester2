"""The 5%ers 100K MC with news-blackout filter applied.

The 5%ers rule: no entry within ±2 min of red folder news.

Synthetic USD red folder calendar (recurring high-impact events):
  - FOMC announcements:   ~8/yr at 14:00 ET (4th Wednesday of FOMC months)
  - FOMC Minutes:         ~8/yr at 14:00 ET (3 weeks after FOMC)
  - ISM Manufacturing:    12/yr at 10:00 ET (1st business day of month)
  - ISM Services:         12/yr at 10:00 ET (3rd business day)
  - JOLTS:                12/yr at 10:00 ET (~8th business day)
  - Consumer Confidence:  12/yr at 10:00 ET (~18th business day)
  - Pending Home Sales:   12/yr at 10:00 ET (~19th business day)
  - New Home Sales:       12/yr at 10:00 ET (~17th business day)

8:30 ET events (NFP, CPI, PPI, Retail Sales, GDP, jobless claims, PCE)
are NOT in this list because no strategy enters before 9:00 AM ET.
They'd only affect held-positions' MAE, not entry blocking.

For each strategy's trades, drop any entry within ±2 min of any event.
Then rerun the 5%ers 100K MC and compare to unfiltered.
"""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import calendar
import bisect
import time
from datetime import datetime, time as dtime, timedelta
import pandas as pd
import numpy as np
import zoneinfo

ROOT = Path(__file__).resolve().parents[2]
TRADES_CSV = ROOT / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
B2_TRADES  = ROOT / "scripts" / "overnight range strat" / "tradelogs" / "robust_configs" / "locked_v2_k08_lock045_mart_fc_filtered_trades.csv"
FAB_TRADES = Path("D:/trading_pythonbacktest_data/fabio orb/trades_final_modeA.csv")
ET = zoneinfo.ZoneInfo("America/New_York")

START_DATE = datetime(2020, 11, 1, tzinfo=ET)
END_DATE   = datetime(2026, 6, 1, tzinfo=ET)
BLACKOUT_MIN = 2

NQ_PT = 20.0
ACCOUNT = 100_000
TRADER_SPLIT = 0.80
REFRESH_FEE = 95.0

PHASE1_TARGET_PCT = 0.08
PHASE2_TARGET_PCT = 0.05
MIN_PROFITABLE_DAYS_P1 = 3
DAILY_CAP_PCT = 0.05
MAX_DD_PCT = 0.10

HORIZON_DAYS = 252
N_SIMS = 5000

BASE_SIZING_100K = {"OD": 2.50, "B2": 0.83, "RV": 2.50, "FB": 3.33}
SCALES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
N_WORKERS = 6

_DAILY = None


# ============================== news calendar ==============================

def nth_business_day(year: int, month: int, n: int):
    cal = calendar.Calendar()
    bdays = [d for d in cal.itermonthdates(year, month) if d.month == month and d.weekday() < 5]
    return bdays[n - 1] if 0 < n <= len(bdays) else None


def nth_weekday(year: int, month: int, weekday: int, n: int):
    cal = calendar.Calendar()
    days = [d for d in cal.itermonthdates(year, month) if d.month == month and d.weekday() == weekday]
    return days[n - 1] if 0 < n <= len(days) else None


def generate_event_calendar() -> list[tuple[str, datetime]]:
    events = []
    fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]  # 8 FOMC meetings/yr
    for year in range(2020, 2027):
        # FOMC announcements — 4th Wednesday of FOMC months at 14:00 ET
        for m in fomc_months:
            d = nth_weekday(year, m, 2, 4)
            if d:
                ts = datetime(d.year, d.month, d.day, 14, 0, tzinfo=ET)
                if START_DATE <= ts <= END_DATE: events.append(("FOMC", ts))
            # FOMC Minutes ~3 weeks after FOMC announcement, 14:00 ET
            if d:
                minutes_date = d + timedelta(days=21)
                ts2 = datetime(minutes_date.year, minutes_date.month, minutes_date.day, 14, 0, tzinfo=ET)
                if START_DATE <= ts2 <= END_DATE: events.append(("FOMC_Minutes", ts2))
        # Monthly 10:00 ET events
        for m in range(1, 13):
            for bd, name in [(1, "ISM_Mfg"), (3, "ISM_Svc"), (8, "JOLTS"),
                              (17, "NewHomeSales"), (18, "ConsConf"), (19, "PendingHomes")]:
                d = nth_business_day(year, m, bd)
                if d:
                    ts = datetime(d.year, d.month, d.day, 10, 0, tzinfo=ET)
                    if START_DATE <= ts <= END_DATE: events.append((name, ts))
    events.sort(key=lambda x: x[1])
    return events


def filter_trades_by_news(trades: pd.DataFrame, events: list[tuple[str, datetime]],
                          ts_col: str = "entry_ts") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop trades within ±BLACKOUT_MIN of any event. Returns (kept, blocked)."""
    # Use int64 nanoseconds for fast comparison (avoids tz/dtype issues)
    event_ns = np.array([pd.Timestamp(ev[1]).value for ev in events], dtype=np.int64)
    blackout_ns = BLACKOUT_MIN * 60 * 1_000_000_000

    trade_ns = trades[ts_col].view("int64").values if hasattr(trades[ts_col], "view") \
               else np.array([pd.Timestamp(t).value for t in trades[ts_col]], dtype=np.int64)

    keep_mask = np.ones(len(trades), dtype=bool)
    for i, ts_n in enumerate(trade_ns):
        idx = np.searchsorted(event_ns, ts_n)
        # Check neighbors
        for j in [idx - 1, idx]:
            if 0 <= j < len(event_ns):
                if abs(ts_n - event_ns[j]) <= blackout_ns:
                    keep_mask[i] = False
                    break
    kept = trades[keep_mask].copy()
    blocked = trades[~keep_mask].copy()
    return kept, blocked


# ============================== data prep ==============================

def load_and_filter_trade_data(events: list) -> tuple[dict, dict]:
    """Load trades, apply news filter, return (daily_unfiltered, daily_filtered)."""
    df = pd.read_csv(TRADES_CSV)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True).dt.tz_convert(ET)
    df["exit_ts"]  = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert(ET)
    df["date"]     = df["exit_ts"].dt.date

    b2 = pd.read_csv(B2_TRADES)
    b2["entry_ts"] = pd.to_datetime(b2["entry_ts"], utc=True).dt.tz_convert(ET)
    b2["mae_$_1nq"] = (-b2["scaled_pnl"]).clip(lower=0) * NQ_PT

    fb = pd.read_csv(FAB_TRADES)
    fb = fb[fb["mode"] == "A"].copy()
    fb["entry_ts"] = pd.to_datetime(fb["entry_time"], utc=True).dt.tz_convert(ET)
    fb["mae_$_1nq"] = fb["mae_pts"] * NQ_PT

    b2_key = b2.set_index("entry_ts")["mae_$_1nq"].to_dict()
    fb_key = fb.set_index("entry_ts")["mae_$_1nq"].to_dict()

    def _mae(r):
        if r["strat"] == "B2": return b2_key.get(r["entry_ts"], max(0, -r["pnl_$"]))
        if r["strat"] == "FB": return fb_key.get(r["entry_ts"], max(0, -r["pnl_$"]))
        return max(0, -r["pnl_$"])
    df["mae_$_1nq"] = df.apply(_mae, axis=1)
    df["pnl_$_1mnq"] = df["pnl_$"] * 0.1
    df["mae_$_1mnq"] = df["mae_$_1nq"] * 0.1

    print(f"  Loaded {len(df)} total trades")
    counts_before = df["strat"].value_counts().to_dict()
    print(f"  By strat: {counts_before}")

    # Apply news filter
    df_kept, df_blocked = filter_trades_by_news(df, events, "entry_ts")
    n_blocked = len(df_blocked)
    print(f"  News filter: blocked {n_blocked} / {len(df)} entries ({n_blocked/len(df)*100:.1f}%)")
    counts_after = df_kept["strat"].value_counts().to_dict()
    print(f"  After filter by strat: {counts_after}")
    print(f"  Blocked by strat:")
    for s in ["OD", "B2", "RV", "FB"]:
        before = counts_before.get(s, 0)
        after = counts_after.get(s, 0)
        print(f"    {s}: {before - after} blocked / {before} ({(before-after)/max(before,1)*100:.1f}%)")

    return _aggregate_daily(df), _aggregate_daily(df_kept)


def _aggregate_daily(df: pd.DataFrame) -> dict:
    agg = df.groupby(["date", "strat"]).agg(
        pnl=("pnl_$_1mnq", "sum"),
        mae=("mae_$_1mnq", "max"),
    ).reset_index()
    out = {}
    for d, g in agg.groupby("date"):
        per_strat = {s: (0.0, 0.0) for s in ["OD", "B2", "RV", "FB"]}
        for _, r in g.iterrows():
            per_strat[r["strat"]] = (float(r["pnl"]), float(r["mae"]))
        out[d] = per_strat
    return out


def daily_arrays(daily: dict) -> dict:
    dates = sorted(daily.keys())
    n = len(dates)
    out = {f"{s}_{k}": np.zeros(n) for s in ["OD", "B2", "RV", "FB"] for k in ["pnl", "mae"]}
    for i, d in enumerate(dates):
        per_strat = daily[d]
        for s in ["OD", "B2", "RV", "FB"]:
            out[f"{s}_pnl"][i] = per_strat[s][0]
            out[f"{s}_mae"][i] = per_strat[s][1]
    return out


# ============================== MC core (same as before) ==============================

def _init_worker(daily_arr):
    global _DAILY
    _DAILY = daily_arr


def _day_stats(idx: int, mnq: dict):
    od_pnl = _DAILY["OD_pnl"][idx] * mnq["OD"]; od_mae = _DAILY["OD_mae"][idx] * mnq["OD"]
    b2_pnl = _DAILY["B2_pnl"][idx] * mnq["B2"]; b2_mae = _DAILY["B2_mae"][idx] * mnq["B2"]
    rv_pnl = _DAILY["RV_pnl"][idx] * mnq["RV"]; rv_mae = _DAILY["RV_mae"][idx] * mnq["RV"]
    fb_pnl = _DAILY["FB_pnl"][idx] * mnq["FB"]; fb_mae = _DAILY["FB_mae"][idx] * mnq["FB"]
    day_total = od_pnl + b2_pnl + rv_pnl + fb_pnl
    day_peak_unrealized = max(od_mae, b2_mae + rv_mae + fb_mae)
    return day_total, day_peak_unrealized, day_total > 0


def _run_challenge_sim(rng, mnq):
    n_pool = len(_DAILY["OD_pnl"])
    daily_cap = ACCOUNT * DAILY_CAP_PCT
    max_dd_cap = ACCOUNT * MAX_DD_PCT
    floor = ACCOUNT - max_dd_cap

    bal = ACCOUNT; profitable_days = 0; days_p1 = 0
    p1_passed = False; fail_reason = None
    for d in range(HORIZON_DAYS):
        idx = rng.integers(0, n_pool)
        day_total, day_peak_unr, is_profit = _day_stats(idx, mnq)
        days_p1 = d + 1
        if day_peak_unr > daily_cap: fail_reason = "daily_unrealized"; break
        bal += day_total
        if bal < floor: fail_reason = "max_dd"; break
        if is_profit: profitable_days += 1
        if bal >= ACCOUNT * (1 + PHASE1_TARGET_PCT) and profitable_days >= MIN_PROFITABLE_DAYS_P1:
            p1_passed = True; break
    if not p1_passed:
        return {"p1_pass": False, "p2_pass": False, "p1_days": days_p1, "p2_days": 0,
                "total_days": days_p1, "fail_reason": fail_reason or "p1_timeout"}

    bal = ACCOUNT; days_p2 = 0; p2_passed = False; fail_reason = None
    for d in range(HORIZON_DAYS):
        idx = rng.integers(0, n_pool)
        day_total, day_peak_unr, _ = _day_stats(idx, mnq)
        days_p2 = d + 1
        if day_peak_unr > daily_cap: fail_reason = "daily_unrealized"; break
        bal += day_total
        if bal < floor: fail_reason = "max_dd"; break
        if bal >= ACCOUNT * (1 + PHASE2_TARGET_PCT): p2_passed = True; break
    return {"p1_pass": True, "p2_pass": p2_passed, "p1_days": days_p1, "p2_days": days_p2,
            "total_days": days_p1 + days_p2,
            "fail_reason": fail_reason if not p2_passed else None}


def _run_funded_sim(rng, mnq):
    n_pool = len(_DAILY["OD_pnl"])
    daily_cap = ACCOUNT * DAILY_CAP_PCT
    max_dd_cap = ACCOUNT * MAX_DD_PCT
    floor = ACCOUNT - max_dd_cap

    bal = ACCOUNT; peak = ACCOUNT; busted = False; bust_reason = None
    days_lived = 0; n_payouts = 0; total_withdrawn = 0.0
    days_to_first_payout = None
    for d in range(HORIZON_DAYS):
        idx = rng.integers(0, n_pool)
        day_total, day_peak_unr, _ = _day_stats(idx, mnq)
        days_lived = d + 1
        if day_peak_unr > daily_cap: busted = True; bust_reason = "daily_unrealized"; break
        bal += day_total
        peak = max(peak, bal)
        if bal < floor: busted = True; bust_reason = "max_dd"; break
        if bal > ACCOUNT:
            profit = bal - ACCOUNT
            n_payouts += 1
            total_withdrawn += profit * TRADER_SPLIT
            bal = ACCOUNT
            if days_to_first_payout is None: days_to_first_payout = d + 1
    return {"busted": busted, "bust_reason": bust_reason, "days_lived": days_lived,
            "n_payouts": n_payouts, "trader_received": total_withdrawn,
            "days_to_first_payout": days_to_first_payout}


def _worker_run(args):
    scale, mode = args
    mnq = {s: round(BASE_SIZING_100K[s] * scale, 3) for s in ["OD", "B2", "RV", "FB"]}
    seed = 2026 + int(scale * 1000) + (777 if mode == "funded" else 0)
    rng = np.random.default_rng(seed)
    if mode == "challenge":
        sims = [_run_challenge_sim(rng, mnq) for _ in range(N_SIMS)]
        p1_pass = np.mean([s["p1_pass"] for s in sims])
        p2_pass = np.mean([s["p2_pass"] for s in sims])
        total_pass = [s["total_days"] for s in sims if s["p2_pass"]]
        fails = [s["fail_reason"] for s in sims if s["fail_reason"]]
        n_daily = sum(1 for r in fails if r == "daily_unrealized")
        n_dd = sum(1 for r in fails if r == "max_dd")
        n_to = sum(1 for r in fails if r and "timeout" in r)
        return {"mode": "challenge", "scale": scale,
                "OD_mnq": mnq["OD"], "B2_mnq": mnq["B2"], "RV_mnq": mnq["RV"], "FB_mnq": mnq["FB"],
                "p1_pass%": round(p1_pass * 100, 2),
                "overall%": round(p2_pass * 100, 2),
                "by_daily%": round(n_daily / N_SIMS * 100, 2),
                "by_dd%": round(n_dd / N_SIMS * 100, 2),
                "by_timeout%": round(n_to / N_SIMS * 100, 2),
                "median_days": int(np.median(total_pass)) if total_pass else None,
                "p25_days": int(np.percentile(total_pass, 25)) if total_pass else None,
                "p75_days": int(np.percentile(total_pass, 75)) if total_pass else None}
    else:
        sims = [_run_funded_sim(rng, mnq) for _ in range(N_SIMS)]
        bust_rate = np.mean([s["busted"] for s in sims])
        mean_life = np.mean([s["days_lived"] for s in sims])
        accts_per_yr = HORIZON_DAYS / mean_life if mean_life > 0 else 0
        annual_trader = np.mean([s["trader_received"] for s in sims]) * accts_per_yr
        annual_payouts = np.mean([s["n_payouts"] for s in sims]) * accts_per_yr
        annual_fees = accts_per_yr * REFRESH_FEE if bust_rate > 0.01 else 0
        annual_net = annual_trader - annual_fees
        all_payouts = [s["trader_received"] / max(1, s["n_payouts"]) for s in sims if s["n_payouts"] > 0]
        fails = [s["bust_reason"] for s in sims if s["busted"] and s["bust_reason"]]
        n_daily = sum(1 for r in fails if r == "daily_unrealized")
        n_dd = sum(1 for r in fails if r == "max_dd")
        return {"mode": "funded", "scale": scale,
                "OD_mnq": mnq["OD"], "B2_mnq": mnq["B2"], "RV_mnq": mnq["RV"], "FB_mnq": mnq["FB"],
                "bust%": round(bust_rate * 100, 2),
                "by_daily%": round(n_daily / N_SIMS * 100, 2),
                "by_dd%": round(n_dd / N_SIMS * 100, 2),
                "avg_life_d": round(mean_life, 1),
                "payouts/yr": round(annual_payouts, 1),
                "trader_$/yr": round(annual_trader, 0),
                "NET_$/yr": round(annual_net, 0),
                "NET_$/mo": round(annual_net / 12, 0),
                "avg_payout_$": round(np.mean(all_payouts), 0) if all_payouts else 0}


def run_mc(daily_arr, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = [(s, "challenge") for s in SCALES] + [(s, "funded") for s in SCALES]
    print(f"\n[{time.strftime('%H:%M:%S')}] [{label}] Dispatching {len(configs)} configs...")
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init_worker, initargs=(daily_arr,)) as ex:
        results = list(ex.map(_worker_run, configs))
    print(f"[{time.strftime('%H:%M:%S')}] [{label}] Done in {time.time()-t1:.1f}s")
    challenge_df = pd.DataFrame([r for r in results if r["mode"] == "challenge"]).sort_values("scale")
    funded_df    = pd.DataFrame([r for r in results if r["mode"] == "funded"]).sort_values("scale")
    return challenge_df, funded_df


def main():
    print(f"[{time.strftime('%H:%M:%S')}] Building synthetic USD red folder calendar...")
    events = generate_event_calendar()
    print(f"  {len(events)} events from {events[0][1]} to {events[-1][1]}")
    name_counts = {}
    for n, _ in events: name_counts[n] = name_counts.get(n, 0) + 1
    print(f"  Breakdown: {name_counts}")

    print(f"\n[{time.strftime('%H:%M:%S')}] Loading + filtering trade data...")
    daily_unfilt, daily_filt = load_and_filter_trade_data(events)
    print(f"  Days with trades — unfilt: {len(daily_unfilt)}, filt: {len(daily_filt)}")

    daily_arr_unfilt = daily_arrays(daily_unfilt)
    daily_arr_filt   = daily_arrays(daily_filt)

    # Run both
    chal_unfilt, fund_unfilt = run_mc(daily_arr_unfilt, "UNFILTERED")
    chal_filt,   fund_filt   = run_mc(daily_arr_filt,   "NEWS-FILTERED")

    out_dir = ROOT / "live" / "combined deployment plan"
    chal_filt.to_csv(out_dir / "the5ers_100k_challenge_newsfilt.csv", index=False)
    fund_filt.to_csv(out_dir / "the5ers_100k_funded_newsfilt.csv", index=False)

    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print(f"\n{'='*100}")
    print(f"COMPARISON — Challenge (overall pass%, median days)")
    print(f"{'='*100}")
    comp_chal = chal_unfilt[["scale", "overall%", "median_days", "by_daily%", "by_dd%"]].copy()
    comp_chal.columns = ["scale", "u_overall%", "u_median_days", "u_by_daily%", "u_by_dd%"]
    comp_chal["f_overall%"]    = chal_filt["overall%"].values
    comp_chal["f_median_days"] = chal_filt["median_days"].values
    comp_chal["f_by_daily%"]   = chal_filt["by_daily%"].values
    comp_chal["f_by_dd%"]      = chal_filt["by_dd%"].values
    comp_chal["d_overall"]      = (comp_chal["f_overall%"] - comp_chal["u_overall%"]).round(2)
    print(comp_chal.to_string(index=False))

    print(f"\n{'='*100}")
    print(f"COMPARISON — Funded (NET $/yr, bust%)")
    print(f"{'='*100}")
    comp_fund = fund_unfilt[["scale", "NET_$/yr", "bust%", "by_daily%", "by_dd%", "payouts/yr"]].copy()
    comp_fund.columns = ["scale", "u_NET_$/yr", "u_bust%", "u_by_daily%", "u_by_dd%", "u_payouts/yr"]
    comp_fund["f_NET_$/yr"]   = fund_filt["NET_$/yr"].values
    comp_fund["f_bust%"]      = fund_filt["bust%"].values
    comp_fund["f_by_daily%"]  = fund_filt["by_daily%"].values
    comp_fund["f_by_dd%"]     = fund_filt["by_dd%"].values
    comp_fund["f_payouts/yr"] = fund_filt["payouts/yr"].values
    comp_fund["d__NET_$"]      = (comp_fund["f_NET_$/yr"] - comp_fund["u_NET_$/yr"]).round(0)
    comp_fund["d__bust%"]      = (comp_fund["f_bust%"] - comp_fund["u_bust%"]).round(2)
    print(comp_fund.to_string(index=False))


if __name__ == "__main__":
    main()
