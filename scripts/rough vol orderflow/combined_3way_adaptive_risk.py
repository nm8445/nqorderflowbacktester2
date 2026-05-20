"""
Adaptive-risk overlay on the 3-strategy combined deployment.

Per-strategy rolling PnL/MDD (Calmar-like) computed from PAST trades only
is mapped to a per-trade size multiplier via four rules:

  tiered_floor : Calmar > 3 -> 100%, 1-3 -> 75%, 0-1 -> 50%, <0 -> 33%
  continuous   : size = clip(Calmar / 3.0, 0.33, 1.0)
  binary_pause : Calmar < 0 -> skip the trade entirely; else 100%
  half_risk    : Calmar < 0 -> 50%; else 100%

Window sweep: N in {50, 100, 150}.

Risk scaling is applied per-strategy on its OWN trade sequence (each strat
sees only its own rolling stats). Conflict resolution between RV and B2 is
then re-run on the size-adjusted trades.

No look-ahead: window for trade i uses trades [i-N, i-1] (strictly past).
First N trades use 100% (warm-up).
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
OUT_DIR = RESULTS_DIR / "adaptive_risk"
OUT_DIR.mkdir(exist_ok=True)

NQ_PT = 20.0


# ---------- Load source trade logs (same as combined_3way.py) ----------
def load_rv():
    rv = pd.read_csv(RESULTS_DIR / "inspect_v3_FULL_log.csv")
    rv["entry_ts"] = pd.to_datetime(rv["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    rv["exit_ts"] = pd.to_datetime(rv["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    rv["direction"] = rv["side"]
    rv["pnl_$"] = rv["pnl_dollars"]
    rv["strat"] = "RV"
    return rv[["entry_ts", "exit_ts", "direction", "pnl_$", "strat"]].copy()


def load_b2():
    b2 = pd.read_csv(ROOT / "scripts" / "overnight range strat" / "tradelogs" / "robust_configs"
                      / "locked_v2_k08_lock045_mart_fc_filtered_trades.csv")
    b2["entry_ts"] = pd.to_datetime(b2["entry_ts"], utc=True).dt.tz_convert("America/New_York")
    b2["exit_ts"] = pd.to_datetime(b2["exit_ts"], utc=True).dt.tz_convert("America/New_York")
    b2["pnl_$"] = b2["scaled_pnl"] * NQ_PT
    b2["strat"] = "B2"
    return b2[["entry_ts", "exit_ts", "direction", "pnl_$", "strat"]].copy()


def load_od():
    od = pd.read_csv(ROOT / "live" / "overnight drift" / "trades.csv")
    od["entry_ts"] = pd.to_datetime(od["entry_time"], utc=True).dt.tz_convert("America/New_York")
    od["exit_ts"] = pd.to_datetime(od["exit_time"], utc=True).dt.tz_convert("America/New_York")
    od["direction"] = "LONG"
    od["pnl_$"] = od["pnl_dollars"]
    od["strat"] = "OD"
    return od[["entry_ts", "exit_ts", "direction", "pnl_$", "strat"]].copy()


# ---------- Rolling Calmar (PnL / |MDD|) over the past N trades ----------
def rolling_calmar(pnl: np.ndarray, n: int) -> np.ndarray:
    """For each index i, Calmar over pnl[i-n:i]. NaN for i < n."""
    out = np.full(len(pnl), np.nan)
    if len(pnl) <= n:
        return out
    for i in range(n, len(pnl)):
        window = pnl[i - n:i]
        s = window.sum()
        cum = window.cumsum()
        mdd = (cum - np.maximum.accumulate(cum)).min()
        if mdd >= 0:                    # never drew down
            out[i] = 99.0 if s > 0 else 0.0
        else:
            out[i] = s / abs(mdd)
    return out


# ---------- Size rules ----------
def rule_tiered_floor(c):
    if np.isnan(c): return 1.0          # warm-up
    if c > 3.0:   return 1.0
    if c > 1.0:   return 0.75
    if c > 0.0:   return 0.50
    return 0.33


def rule_continuous(c, target=3.0):
    if np.isnan(c): return 1.0
    return float(np.clip(c / target, 0.33, 1.0))


def rule_binary_pause(c):
    if np.isnan(c): return 1.0
    return 0.0 if c < 0 else 1.0


def rule_half_risk(c):
    if np.isnan(c): return 1.0
    return 0.5 if c < 0 else 1.0


RULES = {
    "tiered_floor": rule_tiered_floor,
    "continuous":   rule_continuous,
    "binary_pause": rule_binary_pause,
    "half_risk":    rule_half_risk,
}


def apply_rule_to_strat(trades: pd.DataFrame, n: int, rule_name: str) -> pd.DataFrame:
    """trades is one strategy's log sorted by exit_ts. Returns a copy with
    scaled pnl_$ and the size column for inspection."""
    t = trades.sort_values("exit_ts").reset_index(drop=True).copy()
    pnl = t["pnl_$"].to_numpy()
    cal = rolling_calmar(pnl, n)
    rule_fn = RULES[rule_name]
    sizes = np.array([rule_fn(c) for c in cal])
    t["rolling_calmar"] = cal
    t["size"] = sizes
    t["pnl_$_scaled"] = pnl * sizes
    # For binary_pause we keep size=0 rows but they contribute 0 pnl and
    # should not occupy the conflict-resolution slot. Drop them.
    if rule_name == "binary_pause":
        t = t[t["size"] > 0].reset_index(drop=True)
    return t


# ---------- Conflict resolution between RV and B2 (same rule as combined_3way) ----------
def conflict_resolve_rv_b2(rv: pd.DataFrame, b2: pd.DataFrame) -> pd.DataFrame:
    intraday = pd.concat([rv, b2], ignore_index=True).sort_values("entry_ts").reset_index(drop=True)
    kept = []
    open_positions = []
    for _, r in intraday.iterrows():
        open_positions = [p for p in open_positions if p[0] > r["entry_ts"]]
        if any(p[1] != r["direction"] for p in open_positions):
            continue
        kept.append(r)
        open_positions.append((r["exit_ts"], r["direction"]))
    return pd.DataFrame(kept) if kept else intraday.iloc[0:0].copy()


# ---------- Metrics ----------
def metrics(pnl: np.ndarray, exit_ts: pd.Series) -> dict:
    p = np.asarray(pnl)
    if len(p) == 0:
        return dict(trades=0, pnl=0, mdd=0, pf=0, wr=0, sharpe=0, mar=0)
    w = p[p > 0]; l = p[p < 0]
    pf = w.sum() / abs(l.sum()) if len(l) else 99.0
    wr = 100 * len(w) / len(p)
    cum = p.cumsum()
    mdd = (cum - np.maximum.accumulate(cum)).min()
    mar = p.sum() / abs(mdd) if mdd < 0 else 99.0
    # daily-bucketed Sharpe
    df = pd.DataFrame({"pnl": p, "date": exit_ts.dt.date.values})
    daily = df.groupby("date")["pnl"].sum()
    sh = daily.mean() / daily.std(ddof=1) * np.sqrt(252) if daily.std(ddof=1) > 0 else 0.0
    return dict(trades=len(p), pnl=p.sum(), mdd=mdd, pf=pf, wr=wr, sharpe=sh, mar=mar)


# ---------- Driver ----------
def main():
    rv_raw = load_rv()
    b2_raw = load_b2()
    od_raw = load_od()
    print(f"RV: {len(rv_raw)}  B2: {len(b2_raw)}  OD: {len(od_raw)}")

    # Baseline: original conflict-resolved combined
    rv_kept_base = conflict_resolve_rv_b2(rv_raw, b2_raw)
    combined_base = pd.concat([rv_kept_base, od_raw], ignore_index=True).sort_values("exit_ts").reset_index(drop=True)
    base_m = metrics(combined_base["pnl_$"].to_numpy(), combined_base["exit_ts"])
    print(f"\nBASELINE (no scaling):  trades {base_m['trades']}  PnL ${base_m['pnl']:+,.0f}  "
          f"MDD ${base_m['mdd']:+,.0f}  PF {base_m['pf']:.2f}  WR {base_m['wr']:.1f}%  "
          f"Sharpe {base_m['sharpe']:.2f}  MAR {base_m['mar']:.2f}")

    windows = [50, 100, 150]
    rules = list(RULES.keys())

    rows = []
    rows.append({"window": "—", "rule": "baseline", **base_m})

    for n in windows:
        for r in rules:
            rv_s = apply_rule_to_strat(rv_raw, n, r)
            b2_s = apply_rule_to_strat(b2_raw, n, r)
            od_s = apply_rule_to_strat(od_raw, n, r)

            # For conflict resolution, use the scaled pnl as the trade's pnl
            rv_for_cr = rv_s.copy()
            rv_for_cr["pnl_$"] = rv_for_cr["pnl_$_scaled"]
            b2_for_cr = b2_s.copy()
            b2_for_cr["pnl_$"] = b2_for_cr["pnl_$_scaled"]

            intraday_kept = conflict_resolve_rv_b2(rv_for_cr, b2_for_cr)
            od_scaled = od_s.copy()
            od_scaled["pnl_$"] = od_scaled["pnl_$_scaled"]
            combo = pd.concat([intraday_kept, od_scaled], ignore_index=True).sort_values("exit_ts").reset_index(drop=True)
            m = metrics(combo["pnl_$"].to_numpy(), combo["exit_ts"])
            rows.append({"window": n, "rule": r, **m})

            # Save trades for the n=100 set (the user's reference window)
            if n == 100:
                combo.to_csv(OUT_DIR / f"trades_N{n}_{r}.csv", index=False)

    df = pd.DataFrame(rows)
    df["pnl"]  = df["pnl"].round(0)
    df["mdd"]  = df["mdd"].round(0)
    df["pf"]   = df["pf"].round(2)
    df["wr"]   = df["wr"].round(1)
    df["sharpe"] = df["sharpe"].round(2)
    df["mar"]    = df["mar"].round(2)
    df.to_csv(OUT_DIR / "summary_grid.csv", index=False)

    print("\n" + "=" * 100)
    print(f"{'window':>6} {'rule':>14} {'trades':>7} {'PnL $':>12} {'MDD $':>12} "
          f"{'PF':>5} {'WR%':>6} {'Sharpe':>7} {'MAR':>6} {'d_PnL':>9} {'d_MDD':>9} {'d_MAR':>7}")
    print("-" * 100)
    for _, r in df.iterrows():
        d_pnl = r["pnl"] - base_m["pnl"]
        d_mdd = r["mdd"] - base_m["mdd"]
        d_mar = r["mar"] - base_m["mar"]
        print(f"{str(r['window']):>6} {r['rule']:>14} {int(r['trades']):>7} "
              f"${r['pnl']:>+10,.0f} ${r['mdd']:>+10,.0f} "
              f"{r['pf']:>5.2f} {r['wr']:>5.1f}% {r['sharpe']:>7.2f} {r['mar']:>6.2f} "
              f"{d_pnl:>+9,.0f} {d_mdd:>+9,.0f} {d_mar:>+7.2f}")

    # --- Equity curves for N=100 across rules ---
    fig, ax = plt.subplots(figsize=(14, 7))
    rv_kept = conflict_resolve_rv_b2(rv_raw, b2_raw)
    base_combo = pd.concat([rv_kept, od_raw], ignore_index=True).sort_values("exit_ts").reset_index(drop=True)
    base_combo["cum"] = base_combo["pnl_$"].cumsum()
    ax.plot(base_combo["exit_ts"], base_combo["cum"], lw=2.0, color="black",
            label=f"BASELINE  ${base_m['pnl']:+,.0f}  MDD ${base_m['mdd']:+,.0f}  MAR {base_m['mar']:.1f}")
    colors = {"tiered_floor": "steelblue", "continuous": "seagreen",
              "binary_pause": "darkorange", "half_risk": "crimson"}
    for r in rules:
        rv_s = apply_rule_to_strat(rv_raw, 100, r)
        b2_s = apply_rule_to_strat(b2_raw, 100, r)
        od_s = apply_rule_to_strat(od_raw, 100, r)
        rv_for_cr = rv_s.copy(); rv_for_cr["pnl_$"] = rv_for_cr["pnl_$_scaled"]
        b2_for_cr = b2_s.copy(); b2_for_cr["pnl_$"] = b2_for_cr["pnl_$_scaled"]
        intraday_kept = conflict_resolve_rv_b2(rv_for_cr, b2_for_cr)
        od_s2 = od_s.copy(); od_s2["pnl_$"] = od_s2["pnl_$_scaled"]
        c = pd.concat([intraday_kept, od_s2], ignore_index=True).sort_values("exit_ts").reset_index(drop=True)
        c["cum"] = c["pnl_$"].cumsum()
        m = metrics(c["pnl_$"].to_numpy(), c["exit_ts"])
        ax.plot(c["exit_ts"], c["cum"], lw=1.2, color=colors[r], alpha=0.85,
                label=f"{r:>13s}  ${m['pnl']:+,.0f}  MDD ${m['mdd']:+,.0f}  MAR {m['mar']:.1f}")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("3-strategy combined: adaptive risk overlays (rolling Calmar window N=100)")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = OUT_DIR / "adaptive_risk_N100_curves.png"
    plt.savefig(out_png, dpi=110)
    print(f"\nCurve -> {out_png}")
    print(f"Summary CSV -> {OUT_DIR / 'summary_grid.csv'}")


if __name__ == "__main__":
    main()
