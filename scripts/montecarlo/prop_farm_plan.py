"""Prop-farm capital plan: can a copier farm bootstrapped from <$5k reach $100k/yr NET take-home?

Models the full lifecycle on the real 4-way combined day-packs (combined_4way_with_mae_1min.csv,
values at 1 NQ = 10 MNQ; scale = mnq/10):
  CHALLENGE (2-step CFD, RPTI-exempt) -> FUNDED (RPTI + daily + max-loss, biweekly 80% payout)
  -> blow -> rebuy.  Plus a cheap futures-eval lane (50k, $2k trailing-then-lock, no RPTI).

Three outputs:
  (A) per-account funded economics by size (net $/yr, blow%, payouts) -- swept MNQ
  (B) challenge pass-rate + expected $ cost per funded account
  (C) bootstrap ramp from a starting budget w/ reinvestment + a parallel-account cap:
      cumulative NET take-home trajectory, months to $100k run-rate, steady-state farm size.

Assumptions stated in FIRMS (correct me and re-run). Run: python scripts/montecarlo/prop_farm_plan.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from pathlib import Path

CSV = Path(__file__).resolve().parent / "results" / "combined_4way_with_mae_1min.csv"
COST = 4.0          # $/MNQ round-trip (commission + slippage)
MIN_PAYOUT = 200.0
TD_YEAR = 252


@dataclass
class Firm:
    name: str
    size: float
    cost: float          # eval fee ($)
    p1: float            # step-1 target (frac of size)
    p2: float            # step-2 target (0 => 1-step)
    dll: float           # daily loss frac (0 => none)
    rpti: float          # funded RPTI $ (0 => none, e.g. futures)
    split: float
    cycle: int           # td between payouts
    cap: int             # max payouts then dead (0 => unlimited)
    trailing: float      # trailing DD $ for futures (0 => static max-loss)
    maxloss: float       # static max-loss frac (CFD); ignored if trailing>0
    lock: float          # trailing lock balance (futures); 0 => no lock


FIRMS = {
    # FundedNext Stellar 2-step, RPTI mid 2.5%, 80%/biweekly, no payout cap
    "FN200": Firm("FN200k", 200_000, 1200, .08, .05, .05, 5000, .80, 10, 0, 0, .10, 0),
    "FN100": Firm("FN100k", 100_000,  600, .08, .05, .05, 2500, .80, 10, 0, 0, .10, 0),
    # Futures 50k eval->funded: $2k trailing-then-lock @50k, no RPTI, no daily, 90% payout
    "FUT50": Firm("Fut50k",  50_000,  150, .06, .00, .00,    0, .90, 10, 0, 2000, 0, 50_000),
}


def packs():
    df = pd.read_csv(CSV).sort_values("ts")
    return [list(zip(g["pnl_1c"].astype(float), (-g["mae_1c"]).astype(float)))  # (pnl, |float|) @1NQ
            for _, g in df.groupby("date", sort=True)]


# ---------- challenge (RPTI-exempt) ----------
def sim_challenge(P, f: Firm, mnq, rng, max_days=240):
    s = mnq / 10.; n = len(P); total = 0
    floor = f.size * (1 - f.maxloss) if f.trailing == 0 else f.size - f.trailing
    for tgt in ([f.p1, f.p2] if f.p2 > 0 else [f.p1]):
        bal = f.size; goal = f.size * (1 + tgt); peak = f.size; done = False
        for d in range(max_days):
            total += 1; base = bal; dll = base * (1 - f.dll) if f.dll else -1e18
            real = 0.; bust = False
            for p, m in P[rng.integers(0, n)]:
                flo = m * s
                eq = bal + real - flo
                if eq <= floor or eq <= dll: bust = True; break
                real += p * s - mnq * COST
            if bust: return False, total
            bal += real
            if f.trailing:
                if bal > peak: peak = bal
                floor = min(f.lock, peak - f.trailing) if f.lock else peak - f.trailing
            if bal >= goal: done = True; break
        if not done: return False, total
    return True, total


# ---------- funded ----------
def sim_funded(P, f: Firm, mnq, rng, horizon=TD_YEAR):
    s = mnq / 10.; n = len(P); bal = f.size; dic = 0; cash = 0.; pays = 0
    peak = f.size
    floor = f.size * (1 - f.maxloss) if f.trailing == 0 else f.size - f.trailing
    for d in range(horizon):
        base = bal; dll = base * (1 - f.dll) if f.dll else -1e18; real = 0.; bust = False
        for p, m in P[rng.integers(0, n)]:
            flo = m * s
            if f.rpti and flo >= f.rpti: bust = True; break
            eq = bal + real - flo
            if eq <= floor or eq <= dll: bust = True; break
            real += p * s - mnq * COST
        if bust: return cash, pays, True, d + 1
        bal += real; dic += 1
        if f.trailing:
            if bal > peak: peak = bal
            floor = min(f.lock, peak - f.trailing) if f.lock else peak - f.trailing
        if dic >= f.cycle:
            profit = bal - f.size
            if profit >= MIN_PAYOUT:
                cash += profit * f.split; pays += 1; bal = f.size
                if f.trailing: peak = f.size; floor = f.size - f.trailing if not f.lock else min(f.lock, f.size - f.trailing)
            dic = 0
            if f.cap and pays >= f.cap: return cash, pays, False, d + 1
    return cash, pays, False, horizon


def per_account_table(P):
    print("=== (A) per-account FUNDED economics (1yr, swept MNQ) ===")
    print(f"{'firm':>7} {'mnq':>4} {'net$/yr':>9} {'payouts':>8} {'blow%':>7} {'$/payout':>9}")
    best = {}
    for key, f in FIRMS.items():
        grid = [0.5, 1, 1.5, 2, 3] if key != "FUT50" else [1, 2, 3]
        rows = []
        for mnq in grid:
            rng = np.random.default_rng(11 + int(mnq * 100))
            r = [sim_funded(P, f, mnq, rng) for _ in range(8000)]
            cash = np.mean([x[0] for x in r]); pays = np.mean([x[1] for x in r])
            blow = np.mean([x[2] for x in r])
            ppay = (cash / pays) if pays else 0
            rows.append((mnq, cash, pays, blow, ppay))
            print(f"{f.name:>7} {mnq:>4} {cash:>9.0f} {pays:>8.1f} {blow*100:>6.1f}% {ppay:>9.0f}")
        # pick MNQ maximizing net$/yr with blow<=55%
        ok = [r for r in rows if r[3] <= 0.55] or rows
        best[key] = max(ok, key=lambda r: r[1])
        print(f"        -> best {f.name}: {best[key][0]} MNQ  ${best[key][1]:.0f}/yr  blow {best[key][3]*100:.0f}%\n")
    return best


def challenge_table(P, best):
    print("=== (B) challenge pass-rate + cost per funded account ===")
    print(f"{'firm':>7} {'mnq':>4} {'pass%':>7} {'med days':>9} {'$/funded':>9}")
    out = {}
    for key, f in FIRMS.items():
        mnq = best[key][0]
        rng = np.random.default_rng(7)
        r = [sim_challenge(P, f, mnq, rng) for _ in range(8000)]
        pa = np.mean([x[0] for x in r]); days = [x[1] for x in r if x[0]]
        med = int(np.median(days)) if days else 999
        cpf = f.cost / pa if pa else 9e9
        out[key] = (pa, med, cpf)
        print(f"{f.name:>7} {mnq:>4} {pa*100:>6.1f}% {med:>9} {cpf:>9.0f}")
    print()
    return out


def ramp(P, best, chal, firm_key, budget, cap_accts, months=18, sims=1200):
    """Bootstrap ramp: reinvest payouts into evals up to cap_accts live; track cumulative NET."""
    f = FIRMS[firm_key]; mnq = best[firm_key][0]
    n = len(P); td = months * 21
    pass_p, _, _ = chal[firm_key]
    # precompute per-account challenge duration sampler
    rng0 = np.random.default_rng(123)
    net_paths = np.zeros((sims, months))
    live_end = []; takehome_year = []
    for sweep in range(sims):
        rng = np.random.default_rng(1000 + sweep)
        cash = budget; spent = budget * 0  # spent tracked separately from take-home
        eval_spend = 0.; withdrawn = 0.
        accts = []   # each: dict(state, days_left/funded fields)
        for day in range(td):
            pk = P[rng.integers(0, n)]
            # --- buy evals (start of day) up to cap & cash ---
            live = sum(1 for a in accts if a["st"] != "dead")
            while cash >= f.cost and live < cap_accts:
                cash -= f.cost; eval_spend += f.cost
                # decide pass/fail + duration upfront (challenge modeled as a delay)
                passed = rng.random() < pass_p
                dur = int(max(5, rng.normal(chal[firm_key][1], chal[firm_key][1] * 0.4)))
                accts.append({"st": "chal", "t": dur, "passed": passed,
                              "bal": f.size, "dic": 0, "pays": 0, "peak": f.size,
                              "floor": f.size * (1 - f.maxloss) if f.trailing == 0 else f.size - f.trailing})
                live += 1
            # --- advance accounts on today's pack ---
            s = mnq / 10.
            for a in accts:
                if a["st"] == "chal":
                    a["t"] -= 1
                    if a["t"] <= 0:
                        a["st"] = "funded" if a["passed"] else "dead"
                    continue
                if a["st"] != "funded":
                    continue
                base = a["bal"]; dll = base * (1 - f.dll) if f.dll else -1e18; real = 0.; bust = False
                for p, m in pk:
                    flo = m * s
                    if f.rpti and flo >= f.rpti: bust = True; break
                    eq = a["bal"] + real - flo
                    if eq <= a["floor"] or eq <= dll: bust = True; break
                    real += p * s - mnq * COST
                if bust:
                    a["st"] = "dead"; continue
                a["bal"] += real; a["dic"] += 1
                if f.trailing:
                    if a["bal"] > a["peak"]: a["peak"] = a["bal"]
                    a["floor"] = min(f.lock, a["peak"] - f.trailing) if f.lock else a["peak"] - f.trailing
                if a["dic"] >= f.cycle:
                    profit = a["bal"] - f.size
                    if profit >= MIN_PAYOUT:
                        pay = profit * f.split
                        # split payout: feed eval pipeline first, rest is take-home
                        cash += pay; withdrawn += pay; a["bal"] = f.size; a["pays"] += 1
                        if f.trailing:
                            a["peak"] = f.size; a["floor"] = f.size - f.trailing if not f.lock else min(f.lock, f.size - f.trailing)
                    a["dic"] = 0
                    if f.cap and a["pays"] >= f.cap: a["st"] = "dead"
            if (day + 1) % 21 == 0:
                net_paths[sweep, (day + 1) // 21 - 1] = withdrawn - eval_spend
        live_end.append(sum(1 for a in accts if a["st"] == "funded"))
        # trailing 12-mo run-rate at end
        takehome_year.append(net_paths[sweep, -1] - (net_paths[sweep, max(0, months - 12) - 1] if months > 12 else 0))
    med = np.median(net_paths, axis=0)
    p25 = np.percentile(net_paths, 25, axis=0); p75 = np.percentile(net_paths, 75, axis=0)
    print(f"=== (C) bootstrap ramp: {f.name}  start ${budget:,.0f}  cap {cap_accts} accts  {mnq} MNQ ===")
    print(f"{'month':>6} {'net p25':>9} {'net med':>9} {'net p75':>9}")
    for mo in range(months):
        print(f"{mo+1:>6} {p25[mo]:>9.0f} {med[mo]:>9.0f} {p75[mo]:>9.0f}")
    ss = np.median(live_end)
    print(f"\nsteady-state funded accounts (median end): {ss:.0f}")
    print(f"trailing-12mo NET take-home (median): ${np.median(takehome_year):,.0f}")
    # months to $100k cumulative net
    hit = next((mo + 1 for mo in range(months) if med[mo] >= 100_000), None)
    print(f"months to $100k cumulative NET (median path): {hit if hit else '>'+str(months)}")
    print(f"per-funded-account net $/yr: ${best[firm_key][1]:,.0f}  ->  accounts needed for $100k/yr: "
          f"{100_000/ max(best[firm_key][1],1):.1f}\n")


def main():
    pd.set_option("display.width", 200)
    P = packs()
    print(f"{len(P)} trading days in 4-way log.  cost ${COST}/MNQ RT, min payout ${MIN_PAYOUT:.0f}.\n")
    best = per_account_table(P)
    chal = challenge_table(P, best)
    # core CFD workhorse ramp from <$5k
    ramp(P, best, chal, "FN200", budget=4800, cap_accts=20, months=18)
    ramp(P, best, chal, "FUT50", budget=4800, cap_accts=20, months=18)


if __name__ == "__main__":
    main()
