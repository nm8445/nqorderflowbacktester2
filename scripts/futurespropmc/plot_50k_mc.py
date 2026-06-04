"""Monte Carlo outcome distribution PNG — 50k futures eval, live 4-way combined.

Rule: $50k, +$3k target, $2k trailing-then-lock floor (min(50000,max(48000,peak-2000))),
floating-blowable, marti OFF, 1 MNQ. Data: combined_4way_with_mae_1min.csv.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "scripts" / "montecarlo" / "results" / "combined_4way_with_mae_1min.csv"
OUT = ROOT / "scripts" / "montecarlo" / "results" / "futures_50k_mc_distribution.png"
START, TARGET, DD, LOCK, COST, MNQ = 50000., 3000., 2000., 50000., 2.0, 1
N_SIMS, N_PATHS, CAP = 20000, 300, 504


def packs():
    df = pd.read_csv(CSV).sort_values("ts")
    return [list(zip(g["pnl_1c"], g["mae_1c"])) for _, g in df.groupby("date", sort=True)]


def sim(P, rng, record=False):
    s = MNQ/10.; bal=START; peak=START; floor=START-DD; n=len(P); path=[START]
    for d in range(CAP):
        tr = P[rng.integers(0, n)]; real=0.; bust=False
        for p, m in tr:
            if bal+real+(m*s-COST*MNQ) < floor: bust=True; break
            real += p*s-COST*MNQ
        if bust:
            if record: path.append(bal+real)
            return "bust", d+1, path
        bal += real
        if record: path.append(bal)
        if bal-START >= TARGET: return "pass", d+1, path
        if bal>peak: peak=bal
        floor = min(LOCK, max(START-DD, peak-DD))
    return "timeout", CAP, path


def main():
    P = packs()
    rng = np.random.default_rng(7)
    outs=[]; dys=[]
    for _ in range(N_SIMS):
        o, dd, _ = sim(P, rng)
        outs.append(o); dys.append(dd)
    p_pass = np.mean([o=="pass" for o in outs]); p_bust=np.mean([o=="bust" for o in outs])
    p_to = np.mean([o=="timeout" for o in outs])
    pass_days = [d for o,d in zip(outs,dys) if o=="pass"]

    # sample paths for the fan
    rng2 = np.random.default_rng(99)
    paths=[]
    for _ in range(N_PATHS):
        o,dd,pth = sim(P, rng2, record=True); paths.append((o,pth))

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios":[1.6,1]})
    # Panel 1: equity fan
    for o,pth in paths:
        c = "#2ca02c" if o=="pass" else ("#d62728" if o=="bust" else "#888888")
        ax[0].plot(pth, color=c, alpha=0.25, lw=0.8)
    ax[0].axhline(START+TARGET, color="green", ls="--", lw=1.5, label="+$3k target ($53k)")
    ax[0].axhline(START-DD, color="red", ls="--", lw=1.5, label="initial floor ($48k)")
    ax[0].axhline(START, color="black", ls=":", lw=1, alpha=0.6)
    ax[0].set_title(f"50k futures eval — {N_PATHS} MC equity paths (4-way, 1 MNQ, marti off)")
    ax[0].set_xlabel("trading day"); ax[0].set_ylabel("balance ($)")
    ax[0].set_xlim(0, 180); ax[0].legend(loc="upper left", fontsize=9)
    ax[0].grid(alpha=0.2)

    # Panel 2: days-to-pass hist + outcome box
    ax[1].hist(pass_days, bins=40, color="#2ca02c", alpha=0.75, edgecolor="white")
    ax[1].axvline(np.median(pass_days), color="black", ls="--", lw=1.5,
                  label=f"median {int(np.median(pass_days))}d")
    ax[1].set_title("Days-to-pass (passing runs)")
    ax[1].set_xlabel("trading days to reach +$3k"); ax[1].set_ylabel("frequency")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.2)
    txt = (f"Outcomes ({N_SIMS:,} sims, 1 MNQ):\n"
           f"  PASS:   {p_pass*100:4.1f}%\n  BUST:   {p_bust*100:4.1f}%\n  timeout:{p_to*100:4.1f}%\n"
           f"  median pass: {int(np.median(pass_days))} td\n"
           f"  p25/p75: {int(np.percentile(pass_days,25))}/{int(np.percentile(pass_days,75))} td")
    ax[1].text(0.97, 0.97, txt, transform=ax[1].transAxes, ha="right", va="top",
               fontsize=10, family="monospace", bbox=dict(boxstyle="round", fc="#f5f5f5", ec="gray"))
    fig.suptitle("Monte Carlo prop-firm outcome distribution — 4-way combined (OD+RV+B2+FB)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"PASS {p_pass:.1%} BUST {p_bust:.1%} timeout {p_to:.1%} | median pass {int(np.median(pass_days))}d")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
