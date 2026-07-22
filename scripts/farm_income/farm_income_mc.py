"""Farm income MC: how many $1500-1:1 evals to buy for $100k/yr, funded structures A vs B.

Measured inputs (this session):
  - 4-way combined in farm 1:1 config: WR = 54.0% (scripts/propfirm_milking/farm_config_wr.py)
  - eval pass MC calibrated EXACT to the user's framework (50%->44.6%, 48%->40.9%)

Funded rules (user, 2026-06-22), all in PROFIT terms (profit = balance - 50,000):
  - $50k acct, $2k trailing EOD DD; floor LOCKS at profit 0 (= $50,000) once profit hits +$3,000.
    (user's mental model: at +$3k DD=$3k; withdraw $1.5k -> profit +$1.5k, DD $1.5k, floor stays.)
  - Gamble = $1,500 1:1 at WR, 1 trade/day. Must reach +$3,000 before the floor to "go funded-live".
  - Payout = withdraw HALF of profit after 5 winning days (>=$150). 90% profit split. Min withdrawal $500.
  - Structure A (aggressive): reach +3k -> withdraw $1.5k -> RE-GAMBLE $1.5k back to +3k -> withdraw ...
    a losing gamble blows the account.
  - Structure B (milk-to-death): reach +3k -> withdraw $1.5k -> then 1-MNQ MILK 5 winning days ->
    withdraw half -> repeat; DD halves each payout until withdrawal < $500 (stop) or a milk cycle busts.

Milk is NOT assumed: it samples the REAL 4-way combined daily P&L at 1 MNQ (no martingale) day by
day -> a winning day is >=$150, accumulate 5, red/crash days can blow the shrinking DD. So milk
profit, milk DURATION (days), and milk BUST risk all emerge from the strategy, not from constants.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rv_1min_pass_opt"))
import eval_mc  # calibrated eval pass-rate MC

WR = 0.54
EVAL_FEE = 100.0
SPLIT = 0.90
MIN_WD = 500.0
WIN_DAY = 150.0           # firm rule: a "winning day" is >= $150
WINS_NEEDED = 5           # 5 winning days to withdraw
MAX_WD = 2                # CAP: 3rd withdrawal -> promotion to live (avoided). 2 payouts/funded max.
TRADING_DAYS_YR = 251     # NYSE/bank-holiday-excluded trading days/yr (avg 2021-25); firm-day basis

# Real milking P&L = 4-way combined, DYNAMIC exits, 1 MNQ, NO martingale (pnl_1c is 1 NQ -> /10).
# Sampled per day to drive milk profit, milk duration, AND milk blow risk (no assumptions).
_MILK_CSV = Path(__file__).resolve().parents[1] / "futurespropmc" / "results" / "combined_4way_with_mae_1min.csv"


def load_daily_1mnq() -> np.ndarray:
    """Daily P&L at 1 MNQ, aggregated to FIRM trading days: OD enters 19:00 ET (>=17:00) so it
    settles in the NEXT session -> merge into the next business day (else OD Sunday-evening entries
    get their own date and over-count trading days ~288/yr vs the true ~251)."""
    df = pd.read_csv(_MILK_CSV)
    ts = pd.to_datetime(df["ts"], utc=True, format="mixed").dt.tz_convert("America/New_York")
    base = ts.dt.normalize()
    firm_date = base.where(ts.dt.hour < 17, base + pd.tseries.offsets.BDay(1)).dt.date
    daily = df.assign(fd=firm_date).groupby("fd")["pnl_1c"].sum().to_numpy() / 10.0
    return daily


def load_daily_1mnq_mae() -> np.ndarray:
    """Per FIRM-day (net_$, worst_intraday_low_$) at 1 MNQ. Intraday low = the running cumulative
    equity's worst point within the firm-day: each trade's floating dip = cum_realized_before +
    trade_mae (trades ordered by ts; overlapping holds approximated as sequential). Lets milk BLOW on
    an intraday dip through the floor even if the day closes green (the net-only load_daily_1mnq missed
    this). Both columns are 1-MNQ ($ /10 from the 1-NQ csv)."""
    df = pd.read_csv(_MILK_CSV)
    ts = pd.to_datetime(df["ts"], utc=True, format="mixed").dt.tz_convert("America/New_York")
    base = ts.dt.normalize()
    firm_date = base.where(ts.dt.hour < 17, base + pd.tseries.offsets.BDay(1)).dt.date
    df = df.assign(fd=firm_date, _ts=ts).sort_values("_ts")
    out = []
    for _, g in df.groupby("fd"):
        cum = 0.0; low = 0.0
        for pnl, mae in zip(g["pnl_1c"].to_numpy(), g["mae_1c"].to_numpy()):
            low = min(low, cum + mae)      # deepest floating dip while this trade is open
            cum += pnl                     # realize the trade
            low = min(low, cum)            # ...and at its realized close
        out.append((cum / 10.0, low / 10.0))
    return np.array(out)


def _gamble_to_3k(p, rng, locked=False):
    """$1500 1:1, 1 trade/day, up to +3000. Floor trails min(0,peak-2000) then locks at 0
    (locked=True forces floor 0 for re-gambles). Returns (reached_profit|None, days, winning_days)."""
    peak = max(p, 0.0); days = 0; wins = 0
    for _ in range(500):
        days += 1
        if rng.random() < WR:
            p += 1500.0; wins += 1          # a $1500 win is a winning day (>=150)
        else:
            p -= 1500.0
        peak = max(peak, p)
        floor = 0.0 if locked else min(0.0, peak - 2000.0)
        if p <= floor + 1e-9:
            return None, days, wins
        if p >= 3000.0:
            return 3000.0, days, wins
    return None, days, wins


def _milk(p, need_wins, daily, rng, floor=0.0):
    """Milk 1-MNQ 4-way combined: sample REAL daily P&L until `need_wins` winning days (>=$150).
    Blow if a red/crash day drops profit to the floor. Returns (profit|None, days)."""
    days = 0; wins = 0; n = daily.size
    while wins < need_wins:
        days += 1
        pnl = daily[rng.integers(n)]
        p += pnl
        if p <= floor + 1e-9:
            return None, days
        if pnl >= WIN_DAY:
            wins += 1
    return p, days


def funded_lifecycle(structure, daily, rng):
    """Total $ the USER pockets (after split) + total trading days, for one funded account.
    Milk profit & duration & blow risk all come from the REAL 1-MNQ daily P&L."""
    cash = 0.0; tot_days = 0
    p, gd, gwins = _gamble_to_3k(0.0, rng, locked=False)   # initial reach-3k
    tot_days += gd
    if p is None:
        return 0.0, tot_days
    nwd = 0
    while nwd < MAX_WD:
        if nwd > 0:  # 2nd+ payout: re-acquire +3000
            if structure == "A":
                p2, gd2, gwins = _gamble_to_3k(p, rng, locked=True)
                tot_days += gd2
                if p2 is None:
                    return cash, tot_days
                p = p2
            else:        # B: no re-gamble, milk from current profit
                gwins = 0
        # need 5 winning days; gamble wins (this round) count, milk the rest at 1 MNQ
        need = max(0, WINS_NEEDED - gwins)
        if need > 0:
            p, md = _milk(p, need, daily, rng, floor=0.0)
            tot_days += md
            if p is None:
                return cash, tot_days
        w = p / 2.0
        if w < MIN_WD:
            return cash, tot_days
        cash += w * SPLIT; p -= w; nwd += 1
    return cash, tot_days


def funded_stats(structure, daily, n=120_000, seed=1):
    rng = np.random.default_rng(seed)
    cash = np.empty(n); days = np.empty(n)
    for i in range(n):
        cash[i], days[i] = funded_lifecycle(structure, daily, rng)
    paid = cash > 0
    return cash.mean(), paid.mean(), days[paid].mean() if paid.any() else 0.0, np.percentile(days, 80)


def batch_year(structure, daily, n_lanes, n_firms, p_pass, n_cycles, rng):
    """One simulated YEAR of the PAIRED structure: n_lanes decorrelated lanes, each lane = n_firms
    accounts taking IDENTICAL signals (perfectly correlated -> the lane's eval+funded is simulated
    ONCE and counted n_firms times). Variance comes from the n_lanes, not n_lanes*n_firms."""
    annual = 0.0
    for _ in range(n_cycles):
        for _ in range(n_lanes):
            passed = rng.random() < p_pass
            ext = funded_lifecycle(structure, daily, rng)[0] if passed else 0.0
            annual += n_firms * ext - n_firms * EVAL_FEE   # fee paid on every eval account
    return annual


def paired_stats(structure, daily, n_lanes, n_firms, p_pass, cycles_yr, n_years=8000, seed=11):
    rng = np.random.default_rng(seed)
    nc = int(round(cycles_yr))
    v = np.array([batch_year(structure, daily, n_lanes, n_firms, p_pass, nc, rng) for _ in range(n_years)])
    return dict(mean=v.mean(), p10=np.percentile(v, 10), p50=np.percentile(v, 50),
                p90=np.percentile(v, 90), ploss=(v < 0).mean(),
                cv=v.std() / v.mean() if v.mean() else float("nan"))


_EVAL_R_CSV = Path(__file__).resolve().parent / "combined_eval_R.csv"


def load_eval_R() -> np.ndarray:
    """Real per-trade R-multiple of the farm-config 1:1 combined (TP=+1, SL=-1, FC=partial)."""
    return pd.read_csv(_EVAL_R_CSV)["R"].to_numpy()


def eval_sim(Rd, daily, rng):
    """Strategy-driven eval: gamble the REAL 1:1 R (1 trade/day, risk ~$1500 with NQ/MNQ sizing
    granularity) to +$3000, then 1-MNQ MILK top-up to the consistency target
    (true_target = max(3000, 2*biggest_day)). Floor trails $2k, locks at +$100. Returns (pass?, days)."""
    profit = 0.0; peak = 0.0; biggest = 0.0; days = 0
    while profit < 3000.0:                       # gamble to nominal target
        days += 1
        risk = 1500.0 + rng.uniform(-60.0, 60.0)  # mixed NQ+MNQ sizing -> ~$1440-$1560
        dp = Rd[rng.integers(Rd.size)] * risk
        profit += dp
        peak = max(peak, profit)
        if dp > biggest:
            biggest = dp
        if profit <= min(100.0, peak - 2000.0) + 1e-9:
            return False, days
    true_target = max(3000.0, 2.0 * biggest)     # consistency-adjusted
    floor = min(100.0, peak - 2000.0)
    while profit < true_target:                  # 1-MNQ milk the consistency gap
        days += 1
        profit += daily[rng.integers(daily.size)]
        if profit <= floor + 1e-9:
            return False, days
    return True, days


def eval_pass_rate(n=200_000, seed=7):
    Rd = load_eval_R(); daily = load_daily_1mnq()
    rng = np.random.default_rng(seed)
    out = [eval_sim(Rd, daily, rng) for _ in range(n)]
    passed = np.array([o[0] for o in out]); days = np.array([o[1] for o in out])
    return {"pass": passed.mean(), "median_days": float(np.median(days[passed])),
            "mean_days": float(days[passed].mean())}


def main():
    daily = load_daily_1mnq()
    print(f"milk daily P&L (1 MNQ, no-marti 4way): n={daily.size}  mean=${daily.mean():.0f}  "
          f"green>=$150={ (daily>=WIN_DAY).mean()*100:.0f}%  red={ (daily<0).mean()*100:.0f}%  "
          f"p5=${np.percentile(daily,5):.0f} p95=${np.percentile(daily,95):.0f}\n")
    ev = eval_pass_rate()
    p_pass = ev["pass"]; days_pass = ev["median_days"]
    print(f"WR={WR*100:.0f}%  eval pass={p_pass*100:.1f}%  median days-to-pass={days_pass:.0f}\n")

    for structure in ("A", "B"):
        per_funded, p_anypay, days_paid, days_p80 = funded_stats(structure, daily)
        per_eval = p_pass * per_funded - EVAL_FEE
        # cycle = pass eval + extract ALL fundeds (run in parallel) ~ eval days + p80 of funded life
        cycle_days = days_pass + days_p80
        cycles_yr = TRADING_DAYS_YR / cycle_days
        evals_for_100k = 100_000.0 / (per_eval * cycles_yr) if per_eval > 0 else float("inf")
        print(f"=== Structure {structure} ===")
        print(f"  E[$/funded] (user take) = ${per_funded:,.0f}   P(>=1 payout)={p_anypay*100:.0f}%")
        print(f"  funded life: mean(paid)={days_paid:.0f} td   p80(all)={days_p80:.0f} td")
        print(f"  $/eval = pass*funded - fee = {p_pass:.3f}*${per_funded:,.0f} - ${EVAL_FEE:.0f} = ${per_eval:,.0f}")
        print(f"  cycle ~{cycle_days:.0f} td -> {cycles_yr:.1f} cycles/yr")
        print(f"  EVALS FOR $100k/yr  ~= {evals_for_100k:,.0f}   (spend ${evals_for_100k*EVAL_FEE:,.0f}/cycle)")
        print(f"  annual @ that count = ${per_eval*evals_for_100k*cycles_yr:,.0f}")
        # PAIRED structure: 10 decorrelated lanes x 3 firms = 30 evals/cycle (user's plan)
        ps = paired_stats(structure, daily, n_lanes=10, n_firms=3, p_pass=p_pass, cycles_yr=cycles_yr)
        print(f"  PAIRED 10 lanes x 3 firms (30 evals/cycle): mean ${ps['mean']:,.0f}/yr  "
              f"p10 ${ps['p10']:,.0f} / p50 ${ps['p50']:,.0f} / p90 ${ps['p90']:,.0f}  "
              f"P(losing yr)={ps['ploss']*100:.0f}%  CV={ps['cv']:.2f}")
        # for $100k EXPECTED via lanes-of-3:
        lanes_100k = evals_for_100k / 3.0
        print(f"  -> for $100k EXPECTED: ~{lanes_100k:.0f} lanes x 3 firms = ~{lanes_100k*3:.0f} evals/cycle\n")


if __name__ == "__main__":
    main()
