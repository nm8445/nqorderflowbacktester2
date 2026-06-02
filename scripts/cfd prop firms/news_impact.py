"""FundingPips news-rule impact: exclude trades that open/close within +-5 min of a red-folder
USD event, then re-run the futures 50k eval and FundingPips funded EV vs the no-news baseline.

Red-folder calendar (rule-based, the day-session-relevant ones; 8:30 events are pre-session so they
don't touch the 9am+ day strats, and OD is exempt — opened >5h before, 0% exposure):
  10:00 ET : 1st business day (ISM Mfg), 3rd business day (ISM Services), last Tuesday (Cons Conf)
  14:00 ET : FOMC (~8/yr; modeled as a mid-month Wednesday in the 8 FOMC months)
Compliant behavior modeled = AVOID: any RV/B2/FB trade whose fill OR exit is in a +-5min window is
DROPPED (conservative — removes the whole trade). OD never qualifies.

Run:  python "scripts/cfd prop firms/news_impact.py"
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import calendar
import numpy as np, pandas as pd

CSV = Path(__file__).resolve().parents[2] / "scripts" / "montecarlo" / "results" / "combined_4way_with_mae_1min.csv"
WIN = 5  # minutes
FOMC_MONTHS = {1, 3, 5, 6, 7, 9, 11, 12}


def business_days(y, m):
    return [d for d in range(1, calendar.monthrange(y, m)[1] + 1)
            if calendar.weekday(y, m, d) < 5]


def events_for(d):
    """Return list of event minute-of-day (ET) for a given date d."""
    bd = business_days(d.year, d.month)
    ev = []
    if d.day == bd[0]:           ev.append(10 * 60)        # ISM Mfg 10:00
    if len(bd) >= 3 and d.day == bd[2]: ev.append(10 * 60) # ISM Svc 10:00
    if len(bd) >= 6 and d.day == bd[5]: ev.append(10 * 60) # JOLTS ~6th bus day 10:00
    # last Tuesday (Consumer Confidence) 10:00
    last_tue = max(day for day in range(1, calendar.monthrange(d.year, d.month)[1] + 1)
                   if calendar.weekday(d.year, d.month, day) == 1)
    if d.day == last_tue:        ev.append(10 * 60)
    # UMich sentiment: 2nd Friday (prelim) + last Friday (final), 10:00
    fris = [day for day in range(1, calendar.monthrange(d.year, d.month)[1] + 1)
            if calendar.weekday(d.year, d.month, day) == 4]
    if len(fris) >= 2 and d.day in (fris[1], fris[-1]): ev.append(10 * 60)
    # FOMC ~ 3rd Wednesday of FOMC months, 14:00
    weds = [day for day in range(1, calendar.monthrange(d.year, d.month)[1] + 1)
            if calendar.weekday(d.year, d.month, day) == 2]
    if d.month in FOMC_MONTHS and len(weds) >= 3 and d.day == weds[2]:
        ev.append(14 * 60)
    return ev


def in_window(ts, ev_mins):
    mod = ts.hour * 60 + ts.minute
    return any(abs(mod - e) <= WIN for e in ev_mins)


def load(news_filter):
    df = pd.read_csv(CSV)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("America/New_York")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    df["d"] = df["ts"].dt.date
    if news_filter:
        keep = []
        flagged = 0
        for _, r in df.iterrows():
            if r["strat"] == "OD":
                keep.append(True); continue
            ev = events_for(r["ts"])
            ev_x = events_for(r["exit_ts"])
            hit = (in_window(r["ts"], ev) or in_window(r["exit_ts"], ev_x))
            keep.append(not hit); flagged += hit
        df["keep"] = keep
        print(f"  news filter: dropped {flagged}/{len(df)} trades ({100*flagged/len(df):.1f}%) "
              f"[RV {((df.strat=='RV')&(~df.keep)).sum()}, B2 {((df.strat=='B2')&(~df.keep)).sum()}, "
              f"FB {((df.strat=='FB')&(~df.keep)).sum()}]")
        df = df[df["keep"]]
    return df


def day_packs(df, strats):
    df = df[df.strat.isin(strats)].sort_values("ts")
    return [list(zip(g["pnl_1c"], g["mae_1c"])) for _, g in df.groupby("d", sort=True)]


def fut(packs, mnq, rng):
    s = mnq/10.; bal=50000.; peak=50000.; floor=48000.; n=len(packs)
    for d in range(504):
        tr=packs[rng.integers(0,n)]; real=0.; bust=False
        for p,m in tr:
            if bal+real+(m*s-2.*mnq)<floor: bust=True;break
            real+=p*s-2.*mnq
        if bust: return (0,d+1)
        bal+=real
        if bal-50000>=3000: return (1,d+1)
        if bal>peak: peak=bal
        floor=min(50000.,max(48000.,peak-2000.))
    return (0,504)


def funded(df, sizing, rng):
    by={}
    for s,mnq in sizing.items():
        sc=mnq/10.
        for _,r in df[df.strat==s].iterrows(): by.setdefault(r["d"],[]).append((r["pnl_1c"]*sc, r["mae_1c"]*sc, mnq))
    packs=[by[k] for k in sorted(by)]; n=len(packs)
    def one(rng):
        bal=100000.;dic=0;cash=0.
        for d in range(252):
            base=bal;dll=0.05*base;real=0.;bust=False
            for p,m,mnq in packs[rng.integers(0,n)]:
                flo=(-m) if m<0 else 0.
                if flo>=2000. or bal+real-flo<=90000. or bal+real-flo<=base-dll: bust=True;break
                real+=p-mnq*4.
            if bust: return cash,1
            bal+=real;dic+=1
            if dic>=10:
                pr=bal-100000.
                if pr>=200.: cash+=pr*0.8;bal=100000.
                dic=0
        return cash,0
    r=[one(rng) for _ in range(12000)]
    return np.mean([x[0] for x in r]), np.mean([x[1] for x in r])


def main():
    for tag, nf in [("BASELINE (no news rule)", False), ("NEWS-COMPLIANT (+-5min excluded)", True)]:
        print(f"\n=== {tag} ===")
        df = load(nf)
        packs = day_packs(df, ["OD","RV","B2","FB"])
        print("  Futures 50k (4-way, marti off, floating, trailing-then-lock):")
        for mnq in [1, 2]:
            rng=np.random.default_rng(11+mnq)
            res=[fut(packs,mnq,rng) for _ in range(12000)]
            pa=np.mean([r[0] for r in res]); dd=[r[1] for r in res if r[0]==1]
            print(f"    {mnq} MNQ: pass {100*pa:.1f}%  med {int(np.median(dd))}d")
        rng=np.random.default_rng(5)
        ev,blow=funded(df, {"OD":1,"RV":1,"B2":2,"FB":2} if False else {"RV":1,"OD":2,"B2":2,"FB":2}, rng)
        print(f"  FundingPips funded (RV@1, OD/B2/FB@2): E[$wd]=${ev:.0f}/yr  blow={100*blow:.1f}%")


if __name__ == "__main__":
    main()
