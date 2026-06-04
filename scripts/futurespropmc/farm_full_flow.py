"""Full flow, correct config:
  challenge passed via $1,500 gamble (folded into acquisition cost) -> funded account gambles at
  $2k to build a buffer -> de-risk to 1 MNQ at ANY profit -> milk the 4-way (COPY-TRADED together,
  shared daily pack = correlated) -> payout at +$3k, withdraw to +$2k -> reinvest into new accounts.

Gamble draws independent (de-correlated); milk pack shared (correlated). RV ATR-150 filtered, -$2k
loss cap. Acquisition: $165 challenge / 45% pass = ~$367/funded => ~5 funded per $2k, ~5-day lag
(the $1,500 gamble passes in ~4d). Run: python scripts/montecarlo/farm_full_flow.py
"""
from __future__ import annotations
import numpy as np
from farm_firing_order import gamble_pool, milk_packs


def newg():
    return dict(st="gamble", bal=50000., peak=50000., floor=48000., locked=False, paid=0, wdays=0)


def run(G, P, rng, days=252, cap=30, start=10, per_2k=9, lag=5):
    # per_2k=9: $2k of $100 challenges (=20) x 45.2% pass = ~9 funded; lag ~5d ($1.5k gamble passes ~4d)
    ng, npk = len(G), len(P)
    accts = [newg() for _ in range(start)]; pending = []
    cash = 0.; wd = 0.; ev = 0.; npay = 0; n_ever = start; n_paid = 0
    for day in range(days):
        still = []
        for rd, c in pending:
            if rd <= day:
                add = min(c, max(0, cap - len(accts))); accts += [newg() for _ in range(add)]; n_ever += add
            else: still.append((rd, c))
        pending = still
        pack = P[rng.integers(0, npk)]; alive = []
        for a in accts:
            if a["st"] == "gamble":                       # independent gamble (de-correlated)
                pr, ma = G[rng.integers(0, ng)]
                if ma >= 1.0 or pr <= -1.0: continue       # blow (-$2k cap)
                a["bal"] += pr * 2000. - 30.
                if a["bal"] > a["peak"]: a["peak"] = a["bal"]
                if not a["locked"]:
                    a["floor"] = min(50000., a["peak"] - 2000.)
                    if a["floor"] >= 50000.: a["locked"] = True; a["floor"] = 50000.
                if a["bal"] > 50000.: a["st"] = "milk"     # de-risk at ANY profit
                alive.append(a)
            else:                                          # milk 1 MNQ, SHARED pack (copy-traded = correlated)
                d0 = a["bal"]; dead = False
                for pn, fl in pack:
                    if a["bal"] - fl <= a["floor"]: dead = True; break
                    a["bal"] += pn - 2.
                if dead: continue
                if a["bal"] > a["peak"]: a["peak"] = a["bal"]
                if not a["locked"]:
                    a["floor"] = min(50000., a["peak"] - 2000.)
                    if a["floor"] >= 50000.: a["locked"] = True; a["floor"] = 50000.
                if a["bal"] - d0 >= 50.: a["wdays"] += 1   # winning day
                profit = a["bal"] - 50000.
                if a["wdays"] >= 5 and profit > 0.:        # 5 winning days -> withdraw 50% profit, 80% split
                    withdraw = 0.5 * profit
                    take = 0.8 * withdraw                  # = 0.4*profit take-home; leave 50% + DD in acct
                    wd += take; cash += take; a["bal"] -= withdraw; a["wdays"] = 0; npay += 1
                    if a["paid"] == 0: n_paid += 1
                    a["paid"] += 1
                alive.append(a)
        accts = alive
        inflight = len(accts) + sum(c for _, c in pending)
        while cash >= 2000. and inflight < cap:
            cash -= 2000.; ev += 2000.; pending.append((day + lag, per_2k)); inflight += per_2k
    milking = sum(1 for a in accts if a["st"] == "milk")
    return wd - ev, wd, ev, npay, n_ever, n_paid, milking


def main():
    G = gamble_pool(); P = milk_packs(); N = 3000
    rng = np.random.default_rng(11)
    nets, wds, evs, pays, evers, paids, mlks = ([] for _ in range(7))
    for _ in range(N):
        net, wd, ev, np_, ne, npd, mk = run(G, P, rng)
        nets.append(net); wds.append(wd); evs.append(ev); pays.append(np_)
        evers.append(ne); paids.append(npd); mlks.append(mk)
    nets = np.array(nets); pays = np.array(pays); paids = np.array(paids); evers = np.array(evers)
    print("FULL FLOW (challenge $1.5k gamble -> funded $2k gamble -> de-risk 1MNQ -> milk -> reinvest)\n")
    print(f"  NET take-home: mean ${nets.mean():,.0f}/yr  median ${np.median(nets):,.0f}  "
          f"p25 ${np.percentile(nets,25):,.0f}  p10 ${np.percentile(nets,10):,.0f}  P(<=0) {np.mean(nets<=0)*100:.0f}%")
    print(f"  gross withdrawn ${np.mean(wds):,.0f}/yr | eval/challenge spend ${np.mean(evs):,.0f}/yr\n")
    print(f"  PAYOUTS: ~{pays.mean():.0f}/yr total, ~${np.mean(wds)/max(pays.mean(),1):,.0f} each")
    print(f"  funded accounts cycled through the year: ~{evers.mean():.0f}")
    print(f"  of those, {paids.mean():.0f} paid at least once ({100*paids.mean()/evers.mean():.0f}%); "
          f"the rest blew (gamble or milk) before paying")
    print(f"  accounts milking at year-end: ~{np.mean(mlks):.0f}")


if __name__ == "__main__":
    main()
