"""Combined 4-strategy deployment: RV + B2 + OD + Fabio ORB.

Conflict resolution:
  - Each strategy maintains its OWN independent stop/exit.
  - Trades may overlap in time (different strategies firing).
  - No hedging: if two strategies want OPPOSITE directions at the same time,
    the SECOND entry is blocked.
  - Same direction: both fire (each with own qty).

Output:
  - results/combined_4way_trades.csv     — all kept trades, all 4 strats
  - results/combined_4way_curve.png      — combined equity + per-strat lines
  - results/combined_4way_overlap.csv    — pairwise overlap stats
  - prints overlap matrix + summary stats
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
ROOT = HERE.parent.parent
NQ_PT = 20.0


def _to_et(s):
    s = pd.to_datetime(s, utc=True)
    return s.dt.tz_convert("America/New_York") if hasattr(s, "dt") else s.tz_convert("America/New_York")


def load_rv() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "inspect_v3_FULL_log.csv")
    df["entry_ts"] = _to_et(df["entry_ts"])
    df["exit_ts"]  = _to_et(df["exit_ts"])
    df["direction"] = df["side"]
    df["pnl_$"] = df["pnl_dollars"]
    df["strat"] = "RV"
    return df[["entry_ts", "exit_ts", "direction", "pnl_$", "strat"]].copy()


def load_b2() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "scripts" / "overnight range strat" / "tradelogs" / "robust_configs"
                      / "locked_v2_k08_lock045_mart_fc_filtered_trades.csv")
    df["entry_ts"] = _to_et(df["entry_ts"])
    df["exit_ts"]  = _to_et(df["exit_ts"])
    df["pnl_$"] = df["scaled_pnl"] * NQ_PT
    df["strat"] = "B2"
    return df[["entry_ts", "exit_ts", "direction", "pnl_$", "strat"]].copy()


def load_od() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "live" / "overnight drift" / "trades.csv")
    df["entry_ts"] = _to_et(df["entry_time"])
    df["exit_ts"]  = _to_et(df["exit_time"])
    df["direction"] = "LONG"
    df["pnl_$"] = df["pnl_dollars"]
    df["strat"] = "OD"
    return df[["entry_ts", "exit_ts", "direction", "pnl_$", "strat"]].copy()


def load_fabio() -> pd.DataFrame:
    df = pd.read_csv("D:/trading_pythonbacktest_data/fabio orb/trades_final_modeA.csv")
    df["entry_ts"] = _to_et(df["entry_time"])
    df["exit_ts"]  = _to_et(df["exit_time"])
    df["direction"] = "LONG"   # long-only
    df["pnl_$"] = df["net_dollars"]   # net of slippage/commissions
    df["strat"] = "FB"
    return df[["entry_ts", "exit_ts", "direction", "pnl_$", "strat"]].copy()


def conflict_resolve(intraday: pd.DataFrame):
    """Walk all trades chronologically. Block any entry where another open
    position has the opposite direction (no-hedging rule). Same-direction
    overlaps are allowed.

    Returns (kept_df, n_blocked, blocked_by_strat_dict).
    """
    intraday = intraday.sort_values("entry_ts").reset_index(drop=True)
    kept = []
    open_positions = []   # list of (exit_ts, direction, strat)
    blocked_by = {}
    for _, r in intraday.iterrows():
        open_positions = [p for p in open_positions if p[0] > r["entry_ts"]]
        if any(p[1] != r["direction"] for p in open_positions):
            blocked_by[r["strat"]] = blocked_by.get(r["strat"], 0) + 1
            continue
        kept.append(r)
        open_positions.append((r["exit_ts"], r["direction"], r["strat"]))
    return pd.DataFrame(kept), sum(blocked_by.values()), blocked_by


def overlap_matrix(df: pd.DataFrame):
    """Compute pairwise overlap stats between strategies. For each pair (A,B),
    count: # of A-trades whose [entry_ts, exit_ts] window overlaps with at
    least one B-trade window (same-direction or opposite).
    """
    strats = sorted(df["strat"].unique())
    n = len(strats)
    overlap_cnt = np.zeros((n, n), dtype=int)
    same_dir = np.zeros((n, n), dtype=int)
    opp_dir = np.zeros((n, n), dtype=int)
    for i, sa in enumerate(strats):
        a = df[df["strat"] == sa].reset_index(drop=True)
        for j, sb in enumerate(strats):
            if i == j: continue
            b = df[df["strat"] == sb].reset_index(drop=True)
            for _, ra in a.iterrows():
                # Any B-trade where exit > ra.entry AND entry < ra.exit
                mask = (b["exit_ts"] > ra["entry_ts"]) & (b["entry_ts"] < ra["exit_ts"])
                if mask.any():
                    overlap_cnt[i, j] += 1
                    sub = b[mask]
                    if (sub["direction"] == ra["direction"]).any():
                        same_dir[i, j] += 1
                    if (sub["direction"] != ra["direction"]).any():
                        opp_dir[i, j] += 1
    return strats, overlap_cnt, same_dir, opp_dir


def main():
    rv = load_rv(); b2 = load_b2(); od = load_od(); fb = load_fabio()
    print(f"RV   trades: {len(rv):>5}   PnL ${rv['pnl_$'].sum():+,.0f}")
    print(f"B2   trades: {len(b2):>5}   PnL ${b2['pnl_$'].sum():+,.0f}")
    print(f"OD   trades: {len(od):>5}   PnL ${od['pnl_$'].sum():+,.0f}")
    print(f"FB   trades: {len(fb):>5}   PnL ${fb['pnl_$'].sum():+,.0f}")

    all_trades = pd.concat([rv, b2, od, fb], ignore_index=True)
    kept, n_blocked, blocked_by = conflict_resolve(all_trades)
    print(f"\nConflict resolution: {len(all_trades)} -> {len(kept)}  ({n_blocked} blocked by opposing-direction overlap)")
    print(f"  Blocked by strat: {blocked_by}")

    kept = kept.sort_values("exit_ts").reset_index(drop=True)
    kept["cum"] = kept["pnl_$"].cumsum()
    kept["peak"] = kept["cum"].cummax()
    kept["dd"] = kept["cum"] - kept["peak"]

    # Per-strat stats (post-conflict)
    print(f"\n{'Strat':<6} {'n':>5} {'WR':>6} {'PF':>6} {'PnL':>12} {'MDD':>11} {'MAR':>6}")
    print("-" * 65)
    for s in ["RV", "B2", "OD", "FB"]:
        sub = kept[kept["strat"] == s].sort_values("exit_ts").copy()
        if sub.empty: continue
        pnl = sub["pnl_$"]
        wr = (pnl > 0).mean() * 100
        pf = pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum()) if (pnl < 0).any() else float("inf")
        cum = pnl.cumsum()
        mdd = (cum - cum.cummax()).min()
        mar = pnl.sum() / abs(mdd) if mdd != 0 else float("nan")
        print(f"{s:<6} {len(sub):>5} {wr:>5.1f}% {pf:>5.2f}  ${pnl.sum():>10,.0f}  ${mdd:>+9,.0f}  {mar:>5.2f}")

    total = kept["pnl_$"].sum()
    mdd = kept["dd"].min()
    print("-" * 65)
    print(f"{'TOTAL':<6} {len(kept):>5} {(kept['pnl_$']>0).mean()*100:>5.1f}% "
          f"{kept[kept['pnl_$']>0]['pnl_$'].sum()/abs(kept[kept['pnl_$']<0]['pnl_$'].sum()):>5.2f}  "
          f"${total:>10,.0f}  ${mdd:>+9,.0f}  {total/abs(mdd):>5.2f}")

    # Daily Sharpe
    daily = kept.groupby(kept["exit_ts"].dt.date)["pnl_$"].sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    sortino = daily.mean() / daily[daily < 0].std() * np.sqrt(252) if (daily < 0).any() else 0
    print(f"\nCombined daily Sharpe (ann): {sharpe:.2f}")
    print(f"Combined daily Sortino (ann): {sortino:.2f}")

    # ===== Overlap matrix =====
    print("\n=== OVERLAP MATRIX (counts A-trades that overlap any B-trade in time) ===")
    strats, overlap_cnt, same_dir, opp_dir = overlap_matrix(kept)
    print(f"{'A\\B':<6}", *[f"{s:>10}" for s in strats])
    for i, sa in enumerate(strats):
        n_a = (kept["strat"] == sa).sum()
        row = [sa]
        for j, sb in enumerate(strats):
            if i == j:
                row.append(f"{'—':>10}")
            else:
                pct = 100 * overlap_cnt[i, j] / n_a if n_a else 0
                row.append(f"{overlap_cnt[i,j]:>4} ({pct:>3.0f}%)")
        print(*row)
    print("\n=== SAME-direction overlaps ===")
    print(f"{'A\\B':<6}", *[f"{s:>10}" for s in strats])
    for i, sa in enumerate(strats):
        n_a = (kept["strat"] == sa).sum()
        row = [sa]
        for j, sb in enumerate(strats):
            if i == j: row.append(f"{'—':>10}")
            else:
                pct = 100 * same_dir[i, j] / n_a if n_a else 0
                row.append(f"{same_dir[i,j]:>4} ({pct:>3.0f}%)")
        print(*row)
    print("\n=== OPPOSITE-direction overlaps (these are no-hedge blockers) ===")
    print(f"{'A\\B':<6}", *[f"{s:>10}" for s in strats])
    for i, sa in enumerate(strats):
        n_a = (kept["strat"] == sa).sum()
        row = [sa]
        for j, sb in enumerate(strats):
            if i == j: row.append(f"{'—':>10}")
            else:
                pct = 100 * opp_dir[i, j] / n_a if n_a else 0
                row.append(f"{opp_dir[i,j]:>4} ({pct:>3.0f}%)")
        print(*row)

    # Save trades + overlap to CSV
    RESULTS_DIR.mkdir(exist_ok=True)
    kept.to_csv(RESULTS_DIR / "combined_4way_trades.csv", index=False)
    pd.DataFrame(overlap_cnt, index=strats, columns=strats).to_csv(
        RESULTS_DIR / "combined_4way_overlap.csv")

    # ===== Buy-and-hold reference series from OD log (densest clean NQ price stream) =====
    od_raw = pd.read_csv(ROOT / "live" / "overnight drift" / "trades.csv")
    od_raw["t"] = _to_et(od_raw["entry_time"])
    od_raw = od_raw.sort_values("t").reset_index(drop=True)
    last_exit_t  = _to_et(od_raw["exit_time"].iloc[-1:])
    last_exit_px = float(od_raw["exit_price"].iloc[-1])
    bh = pd.DataFrame({
        "t":  list(od_raw["t"])           + [last_exit_t.iloc[0]],
        "px": list(od_raw["entry_price"]) + [last_exit_px],
    }).sort_values("t").reset_index(drop=True)
    px0 = float(bh["px"].iloc[0])
    bh["bh_$"] = (bh["px"] - px0) * NQ_PT
    bh["bh_peak"] = bh["bh_$"].cummax()
    bh["bh_dd"]   = bh["bh_$"] - bh["bh_peak"]
    bh_total = bh["bh_$"].iloc[-1]
    bh_mdd   = bh["bh_dd"].min()

    # ===== Chart =====
    fig, axes = plt.subplots(2, 1, figsize=(22, 13), sharex=True,
                              gridspec_kw={"height_ratios": [3.2, 1]})
    colors = {"RV": "steelblue", "B2": "seagreen", "OD": "darkorange", "FB": "crimson"}
    for s in ["RV", "B2", "OD", "FB"]:
        sub = kept[kept["strat"] == s].sort_values("exit_ts").copy()
        if sub.empty: continue
        sub["c"] = sub["pnl_$"].cumsum()
        label = {"RV": "Rough Vol v3 (no mart)",
                 "B2": "B2 OHI/OLO (FC-only mart)",
                 "OD": "Overnight Drift (locked)",
                 "FB": "Fabio ORB (locked)"}[s]
        axes[0].plot(sub["exit_ts"], sub["c"], lw=1.6, color=colors[s],
                     label=f"{label:<32} ${sub['pnl_$'].sum():+,.0f}")
    axes[0].plot(kept["exit_ts"], kept["cum"], lw=3.0, color="black",
                 label=f"{'COMBINED':<32} ${total:+,.0f}")
    axes[0].plot(bh["t"], bh["bh_$"], lw=2.0, color="purple", ls="--",
                 label=f"{'NQ buy & hold (1 contract)':<32} ${bh_total:+,.0f}")
    axes[0].axvline(pd.Timestamp("2024-12-31").tz_localize("America/New_York"),
                     color="red", ls="--", lw=1.5, alpha=0.7, label="IS/OOS split")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("Cumulative PnL ($, NQ basis)", fontsize=14)
    axes[0].set_title(f"4-strategy combined (RV + B2 + OD + Fabio ORB) vs NQ buy & hold (1 contract)\n"
                       f"Combined ${total:+,.0f}  MDD ${mdd:+,.0f}  MAR {total/abs(mdd):.1f}  Sharpe {sharpe:.2f}    "
                       f"|    B&H ${bh_total:+,.0f}  MDD ${bh_mdd:+,.0f}  MAR {bh_total/abs(bh_mdd):.2f}",
                       fontsize=14)
    axes[0].legend(loc="upper left", fontsize=11)
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(kept["exit_ts"], kept["dd"], 0,
                          color="crimson", alpha=0.55,
                          label=f"Combined DD (MDD ${mdd:+,.0f})")
    axes[1].fill_between(bh["t"], bh["bh_dd"], 0,
                          color="purple", alpha=0.25,
                          label=f"B&H DD (MDD ${bh_mdd:+,.0f})")
    axes[1].set_ylabel("Drawdown ($)", fontsize=12)
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].legend(loc="lower left", fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / "combined_4way_curve.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nCurve -> {out}")


if __name__ == "__main__":
    main()
