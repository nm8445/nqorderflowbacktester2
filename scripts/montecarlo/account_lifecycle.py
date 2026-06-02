"""Per-funded-account lifecycle: gamble at $2k -> de-risk at any profit -> milk 1 MNQ ONLY (no more
gambling) -> payout when 5 winning days (>= $150 each), withdraw 50% of profit (capped $2k), keep 80%,
leave 50%+DD -> max 4 payouts then done. How fast/big are payouts? Run:
    python scripts/montecarlo/account_lifecycle.py
"""
from __future__ import annotations
import numpy as np
from farm_firing_order import gamble_pool, milk_packs

WIN_DAY, MAXPAY, SPLIT, MAXW = 150., 4, 0.80, 2000.


def lifecycle(G, M, rng):
    ng, nm = len(G), len(M)
    # ---- gamble to buffer ----
    bal = 50000.; peak = 50000.; floor = 48000.; locked = False; gd = 0
    der = False
    for _ in range(60):
        gd += 1
        pr, ma = G[rng.integers(0, ng)]
        if ma >= 1.0 or pr <= -1.0:
            return dict(out="blow_gamble")
        bal += pr * 2000. - 30.
        if bal > peak: peak = bal
        if not locked:
            floor = min(50000., peak - 2000.)
            if floor >= 50000.: locked = True; floor = 50000.
        if bal > 50000.: der = True; break
    if not der:
        return dict(out="no_buffer")
    buffer = bal - 50000.
    # ---- milk 1 MNQ ----
    wd = 0; pays = 0; take = 0.; first = None; md = 0; pay_days = []
    for _ in range(504):
        md += 1
        d0 = bal; dead = False
        for pn, fl in M[rng.integers(0, nm)]:
            if bal - fl <= floor: dead = True; break
            bal += pn - 2.
        if dead:
            return dict(out="blow_milk", buffer=buffer, pays=pays, take=take, first=first, pay_days=pay_days)
        if bal > peak: peak = bal
        if not locked:
            floor = min(50000., peak - 2000.)
            if floor >= 50000.: locked = True; floor = 50000.
        if bal - d0 >= WIN_DAY: wd += 1
        profit = bal - 50000.
        if wd >= 5 and profit > 0.:
            w = min(0.5 * profit, MAXW); t = SPLIT * w
            take += t; bal -= w; wd = 0; pays += 1; pay_days.append(gd + md)
            if first is None: first = (gd + md, t)
            if pays >= MAXPAY:
                return dict(out="capped", buffer=buffer, pays=pays, take=take, first=first, pay_days=pay_days)
    return dict(out="survive", buffer=buffer, pays=pays, take=take, first=first, pay_days=pay_days)


def main():
    G = gamble_pool(); M = milk_packs()
    # winning-day frequency at 1 MNQ
    rng = np.random.default_rng(1)
    dd = [sum(p - 2. for p, _ in M[rng.integers(0, len(M))]) for _ in range(200000)]
    dd = np.array(dd)
    print(f"Milk day P&L @1 MNQ: mean ${dd.mean():.0f}, P(day>=$150)={100*np.mean(dd>=150)*1:.0f}%, "
          f"P(day>0)={100*np.mean(dd>0):.0f}%  -> ~{5/np.mean(dd>=150):.0f} days to get 5 winning days\n")

    rng = np.random.default_rng(7); res = [lifecycle(G, M, rng) for _ in range(60000)]
    der = [r for r in res if r["out"] not in ("blow_gamble", "no_buffer")]
    paid = [r for r in der if r["pays"] >= 1]
    firsts = [r["first"] for r in paid]
    print(f"Reach buffer & de-risk: {len(der)/len(res)*100:.0f}%   (then milk 1 MNQ only, no re-gamble)")
    print(f"Of de-risked accts, {len(paid)/len(der)*100:.0f}% get >=1 payout\n")
    if firsts:
        fd = np.array([f[0] for f in firsts]); fa = np.array([f[1] for f in firsts])
        print(f"FIRST payout: median day {int(np.median(fd))} (from funding), "
              f"avg take-home ${fa.mean():,.0f} (median ${np.median(fa):,.0f})")
    allp = np.array([r["pays"] for r in der])
    tk = np.array([r["take"] for r in der])
    print(f"payouts/de-risked acct: mean {allp.mean():.2f} (capped at 4)")
    print(f"total take-home / de-risked acct: mean ${tk.mean():,.0f}")
    print(f"total take-home / funded acct (incl gamble-blows): mean ${np.mean([r.get('take',0.) for r in res]):,.0f}")
    # spacing between payouts
    gaps = []
    for r in der:
        pd_ = r["pay_days"]
        gaps += list(np.diff(pd_)) if len(pd_) > 1 else []
    if gaps:
        print(f"days between payouts (after the first): median {int(np.median(gaps))}")
    print(f"\nReinvest pace: first payout ~${np.median(fa):,.0f} at ~day {int(np.median(fd))} -> "
          f"buys ~{int(np.median(fa)//100)} new $100 challenges right away.")


if __name__ == "__main__":
    main()
