"""
Does prior overnight drift predict the next overnight drift?

For each night, compute:
  drift = close(08:00 ET next morning) - close(19:00 ET entry)

Then test:
  - sign autocorrelation (did up-nights cluster?)
  - lagged drift regression (does magnitude carry?)
  - up-after-up vs up-after-down probability
  - rolling regime test (does autocorrelation change over time?)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overnight_drift_strategy import build_full_20min_series  # noqa: E402

PARQUET_PATH = "D:/trading_pythonbacktest_data/markettick_1min_bars.parquet"
PICKLE_FOLDER = "D:/trading_pythonbacktest_data/timebars_5min"


def build_overnight_pairs(bars: pd.DataFrame) -> pd.DataFrame:
    """For each trading session, get the 19:00 ET close (entry) and 08:00 ET close (exit)."""
    et = bars.index
    df = bars.copy()
    df["time"] = et.time
    df["date"] = et.date

    entries = df[df.index.time == pd.Timestamp("19:00").time()][["close"]].copy()
    entries = entries.rename(columns={"close": "entry_close"})
    entries["entry_ts"] = entries.index

    exits = df[df.index.time == pd.Timestamp("08:00").time()][["close"]].copy()
    exits = exits.rename(columns={"close": "exit_close"})
    exits["exit_ts"] = exits.index

    # Pair each entry with the NEXT exit
    rows = []
    exit_iter = iter(exits.iterrows())
    exits_list = list(exits.iterrows())
    exit_idx = 0
    for ent_ts, ent_row in entries.iterrows():
        # find first exit AFTER this entry within next 24h
        while exit_idx < len(exits_list) and exits_list[exit_idx][0] <= ent_ts:
            exit_idx += 1
        if exit_idx >= len(exits_list):
            break
        ex_ts, ex_row = exits_list[exit_idx]
        if (ex_ts - ent_ts).total_seconds() > 24 * 3600:
            continue
        rows.append({
            "entry_ts": ent_ts,
            "exit_ts": ex_ts,
            "entry_close": ent_row["entry_close"],
            "exit_close": ex_row["exit_close"],
            "drift": ex_row["exit_close"] - ent_row["entry_close"],
        })
    return pd.DataFrame(rows).sort_values("entry_ts").reset_index(drop=True)


def main():
    print("Building 20-min bars...")
    bars = build_full_20min_series(PARQUET_PATH, PICKLE_FOLDER)
    print(f"Loaded {len(bars):,} bars\n")

    pairs = build_overnight_pairs(bars)
    print(f"Constructed {len(pairs)} overnight drift events\n")

    d = pairs["drift"].copy()
    pairs["prev_drift"] = d.shift(1)
    pairs["prev2_drift"] = d.shift(2)
    pairs = pairs.dropna(subset=["prev_drift"]).reset_index(drop=True)

    print("=== Overnight drift summary ===")
    print(f"Mean drift:    {d.mean():+.2f} pts  (={d.mean()*20:+.0f} $/contract)")
    print(f"Median drift:  {d.median():+.2f} pts")
    print(f"Std drift:     {d.std():.2f} pts")
    print(f"P(drift > 0):  {(d > 0).mean()*100:.1f}%")
    print(f"P(drift < 0):  {(d < 0).mean()*100:.1f}%")
    print()

    # --- Pearson autocorrelation ---
    print("=== Lagged autocorrelation (Pearson) ===")
    for lag in [1, 2, 3, 5, 10]:
        ac = d.autocorr(lag=lag)
        n = len(d) - lag
        # 95% CI for null (no autocorr): ~1.96/sqrt(n)
        ci95 = 1.96 / np.sqrt(n)
        sig = "***" if abs(ac) > ci95 else "   "
        print(f"  lag={lag:>2}: r={ac:+.4f}  (95% CI ±{ci95:.4f})  {sig}")
    print()

    # --- Sign autocorrelation: P(up | prev up) etc ---
    print("=== Conditional sign probabilities ===")
    pairs["sign"] = np.sign(pairs["drift"])
    pairs["prev_sign"] = np.sign(pairs["prev_drift"])

    base_up = (pairs["sign"] > 0).mean()
    base_dn = (pairs["sign"] < 0).mean()
    print(f"Unconditional P(up): {base_up*100:.1f}%   P(down): {base_dn*100:.1f}%")

    up_after_up = pairs[pairs["prev_sign"] > 0]["sign"].apply(lambda x: x > 0).mean()
    up_after_dn = pairs[pairs["prev_sign"] < 0]["sign"].apply(lambda x: x > 0).mean()
    dn_after_up = pairs[pairs["prev_sign"] > 0]["sign"].apply(lambda x: x < 0).mean()
    dn_after_dn = pairs[pairs["prev_sign"] < 0]["sign"].apply(lambda x: x < 0).mean()

    n_up_prior = (pairs["prev_sign"] > 0).sum()
    n_dn_prior = (pairs["prev_sign"] < 0).sum()

    print(f"\nP(up tonight | up last night):    {up_after_up*100:5.1f}%   "
          f"vs base {base_up*100:.1f}%   "
          f"(edge {(up_after_up - base_up)*100:+.1f}pp, n={n_up_prior})")
    print(f"P(up tonight | down last night):  {up_after_dn*100:5.1f}%   "
          f"vs base {base_up*100:.1f}%   "
          f"(edge {(up_after_dn - base_up)*100:+.1f}pp, n={n_dn_prior})")
    print(f"P(down tonight | up last night):  {dn_after_up*100:5.1f}%")
    print(f"P(down tonight | down last night): {dn_after_dn*100:5.1f}%")

    # Chi-square test on the 2x2 contingency
    from scipy.stats import chi2_contingency
    tbl = np.array([
        [(pairs[(pairs["prev_sign"] > 0) & (pairs["sign"] > 0)]).shape[0],
         (pairs[(pairs["prev_sign"] > 0) & (pairs["sign"] < 0)]).shape[0]],
        [(pairs[(pairs["prev_sign"] < 0) & (pairs["sign"] > 0)]).shape[0],
         (pairs[(pairs["prev_sign"] < 0) & (pairs["sign"] < 0)]).shape[0]],
    ])
    chi2, p_val, _, _ = chi2_contingency(tbl)
    print(f"\nChi-square test (2x2 contingency): chi2={chi2:.2f}, p={p_val:.4f}")
    if p_val < 0.05:
        print("  -> Statistically significant association (p < 0.05)")
    else:
        print("  -> NOT statistically significant (p >= 0.05) — looks like independence")

    # --- Conditional means: drift size after up vs down ---
    print("\n=== Conditional drift magnitudes ===")
    mean_after_up = pairs[pairs["prev_sign"] > 0]["drift"].mean()
    mean_after_dn = pairs[pairs["prev_sign"] < 0]["drift"].mean()
    median_after_up = pairs[pairs["prev_sign"] > 0]["drift"].median()
    median_after_dn = pairs[pairs["prev_sign"] < 0]["drift"].median()
    print(f"Mean drift after up:    {mean_after_up:+.2f} pts (median {median_after_up:+.2f})")
    print(f"Mean drift after down:  {mean_after_dn:+.2f} pts (median {median_after_dn:+.2f})")
    print(f"Difference (up-down):   {mean_after_up - mean_after_dn:+.2f} pts")

    # --- Per-year breakdown ---
    print("\n=== Per-year sign-autocorr (P(up|up) - P(up|down)) ===")
    pairs["year"] = pairs["entry_ts"].dt.year
    for yr, grp in pairs.groupby("year"):
        if len(grp) < 30: continue
        p_uu = grp[grp["prev_sign"] > 0]["sign"].apply(lambda x: x > 0).mean()
        p_ud = grp[grp["prev_sign"] < 0]["sign"].apply(lambda x: x > 0).mean()
        n = len(grp)
        diff = p_uu - p_ud
        print(f"  {int(yr)}: P(up|up)={p_uu*100:5.1f}%  P(up|down)={p_ud*100:5.1f}%  diff={diff*100:+.1f}pp  (n={n})")

    out = Path(__file__).parent / "drift_autocorr_results.csv"
    pairs.to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
