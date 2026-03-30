"""
Expectancy analysis for the three final setups.
Uses directional-correct outcomes — reversal setups trade AGAINST absorption direction.

KEY SEMANTIC NOTE:
  sell_absorbed (Setup A, B) = bullish bar, BUT traded as SHORT reversal at extreme.
    trade_pts = signal_price - end_price     (positive = price fell = SHORT WIN)
    mae_normalized ≈ max downside excursion  (= MAX PROFIT POTENTIAL for short, NOT adverse)
    Adverse excursion (max upward move against short) is NOT captured by the labeler.

  buy_absorbed (Setup C) = bearish bar, BUT traded as LONG reversal at extreme.
    trade_pts = end_price - signal_price     (positive = price rose = LONG WIN)
    mae_normalized ≈ max upside excursion    (= MAX PROFIT POTENTIAL for long, NOT adverse)
    Adverse excursion (max downward move against long) is NOT captured by the labeler.

Point values computed directly from signal_price and end_price (exact, no scaling).
mae estimates use per-signal denom (scale = |actual_pts| / |move_normalized|).

Saves to output/diagnostics/expectancy_report.txt
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
from nqbt.analysis.signal_diagnostics import signal_clustering, prior_absorption_count

CSV_PATH = PROJECT_ROOT / "output" / "absorption_signals_labeled.csv"
OUT_DIR  = PROJECT_ROOT / "output" / "diagnostics"
OUT_PATH = OUT_DIR / "expectancy_report.txt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ET = "America/New_York"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
for col in ["sustained", "sustained_loose", "sustained_strict",
            "reversal_rejection_confirmed", "node_proven_20t", "at_vwap_band"]:
    df[col] = df[col].astype(bool)

sig_et      = pd.to_datetime(df["signal_time"], utc=True).dt.tz_convert(ET)
df["_date"] = sig_et.dt.date

print("Computing cluster_position and prior_absorption_count...")
df_c, _ = signal_clustering(df)
df["cluster_position"] = df_c["cluster_position"]
df_p, _ = prior_absorption_count(df)
df["prior_absorption_count"] = df_p["prior_absorption_count"]
print("Done.\n")

# ── Per-signal scale: denom = |actual_pts| / |move_normalized| ───────────────
# For sell_absorbed: actual_pts (absorption direction) = end_price - signal_price
# For buy_absorbed:  actual_pts (absorption direction) = signal_price - end_price
# denom_i = actual_pts_i / move_normalized_i  (= session_realized_vol * signal_price)
nz_mask = df["move_normalized"].abs() > 1e-6
sa_mask = (df["absorption_side"] == "sell_absorbed") & nz_mask
ba_mask = (df["absorption_side"] == "buy_absorbed")  & nz_mask
df["_denom"] = np.nan
df.loc[sa_mask, "_denom"] = (df.loc[sa_mask, "end_price"] - df.loc[sa_mask, "signal_price"]) / df.loc[sa_mask, "move_normalized"]
df.loc[ba_mask, "_denom"] = (df.loc[ba_mask, "signal_price"] - df.loc[ba_mask, "end_price"]) / df.loc[ba_mask, "move_normalized"]
# median denom for reference
median_denom = float(df["_denom"].median())

# ── Setup masks ───────────────────────────────────────────────────────────────
mask_A = (
    (df["price_location"] == "outside_vah") &
    (df["absorption_side"] == "sell_absorbed") &
    ((df["vwap_band_tag"] == "+2s") | df["at_vwap_band"]) &
    df["reversal_rejection_confirmed"] &
    df["node_proven_20t"] &
    df["cluster_position"].isin(["solo", "first"]) &
    df["overnight_regime"].isin(["directional", "non_directional"])
)
mask_B = (
    (df["price_location"] == "outside_vah") &
    (df["absorption_side"] == "sell_absorbed") &
    df["cluster_position"].isin(["solo", "first"]) &
    (df["prior_absorption_count"] == 0) &
    (df["overnight_regime"] == "non_directional") &
    (~mask_A)
)
mask_C = (
    (df["price_location"] == "outside_val") &
    (df["absorption_side"] == "buy_absorbed") &
    (df["overnight_regime"] == "directional") &
    df["node_proven_20t"] &
    (~df["reversal_rejection_confirmed"]) &
    df["cluster_position"].isin(["solo", "first"])
)

A = df[mask_A].copy()
B = df[mask_B].copy()
C = df[mask_C].copy()

# ── Exact NQ point moves in trade direction ───────────────────────────────────
# Setup A/B SHORT: profit = signal_price - end_price  (positive = price fell)
# Setup C   LONG:  profit = end_price - signal_price  (positive = price rose)
for sub in [A, B]:
    sub["trade_pts"] = sub["signal_price"] - sub["end_price"]

C["trade_pts"] = C["end_price"] - C["signal_price"]

# mae in points: mae_normalized × per-signal denom (profit-direction excursion)
# This is NOT the adverse excursion — it's the max favorable move reached.
for sub in [A, B, C]:
    sub["mae_pts"] = sub["mae_normalized"] * sub["_denom"].abs()

# ── Helpers ───────────────────────────────────────────────────────────────────

def expectancy_block(sub, label):
    """Full expectancy analysis for one setup. Uses exact NQ pts from price cols."""
    lines = []
    n = len(sub)
    if n == 0:
        lines.append(f"  {label}: no signals")
        return lines

    wins   = sub[sub["trade_pts"] > 0]
    losses = sub[sub["trade_pts"] <= 0]
    nw, nl = len(wins), len(losses)
    wr = nw / n

    avg_win_pts  = wins["trade_pts"].mean()         if nw else 0.0
    avg_loss_pts = losses["trade_pts"].abs().mean() if nl else 0.0

    rr  = avg_win_pts / avg_loss_pts if avg_loss_pts > 0 else float("nan")
    bw  = avg_loss_pts / (avg_win_pts + avg_loss_pts) if (avg_win_pts + avg_loss_pts) > 0 else float("nan")
    exp = wr * avg_win_pts - (1 - wr) * avg_loss_pts

    # mae in pts: max favorable excursion (profit potential, NOT adverse stop distance)
    mae_w_pts = wins["mae_pts"].mean()   if nw else float("nan")
    mae_l_pts = losses["mae_pts"].mean() if nl else float("nan")

    lines.append(f"  n={n}  winners={nw} ({wr*100:.1f}%)  losers={nl} ({(1-wr)*100:.1f}%)")
    lines.append(f"")
    lines.append(f"  WINNERS (price moved in trade direction, net at bar+20):")
    lines.append(f"    count:                    {nw}")
    if nw:
        lines.append(f"    avg win:                  {avg_win_pts:+.2f} NQ pts  (${avg_win_pts*20:+.0f})")
        lines.append(f"    median win:               {wins['trade_pts'].median():+.2f} NQ pts")
        lines.append(f"    max win:                  {wins['trade_pts'].max():+.2f} NQ pts")
        lines.append(f"    avg mae_pts (max profit   {mae_w_pts:.2f} pts — how far it went in trade dir)")
    lines.append(f"")
    lines.append(f"  LOSERS (price moved against trade, net at bar+20):")
    lines.append(f"    count:                    {nl}")
    if nl:
        lines.append(f"    avg loss:                 {avg_loss_pts:.2f} NQ pts  (${avg_loss_pts*20:.0f})")
        lines.append(f"    median loss:              {losses['trade_pts'].abs().median():.2f} NQ pts")
        lines.append(f"    max loss:                 {losses['trade_pts'].abs().max():.2f} NQ pts")
        lines.append(f"    avg mae_pts (favorable    {mae_l_pts:.2f} pts — even losers moved this far favored)")
    lines.append(f"    NOTE: Adverse excursion (path against trade) not in labeler data.")
    lines.append(f"          Actual R:R depends on stop placement — see expectancy grid below.")
    lines.append(f"")
    lines.append(f"  RAW EXPECTANCY (no stop — hold full bar+20 period):")
    lines.append(f"    avg win:           {avg_win_pts:+.2f} pts   (${avg_win_pts*20:+.0f})")
    lines.append(f"    avg loss:          {avg_loss_pts:.2f} pts   (${avg_loss_pts*20:.0f})")
    lines.append(f"    R:R ratio:         {rr:.2f}:1")
    lines.append(f"    Breakeven win%:    {bw*100:.1f}%")
    lines.append(f"    Actual win%:       {wr*100:.1f}%")
    lines.append(f"    E per trade:       {exp:+.2f} pts  ({'POSITIVE' if exp > 0 else 'NEGATIVE'})")
    lines.append(f"    E per trade ($):   ~${exp*20:+.0f}")
    lines.append(f"")
    return lines


def stop_scenario_block(sub, label):
    """
    Stop scenario analysis using actual NQ point outcomes.
    NOTE: the labeler does NOT capture max adverse excursion for reversal setups.
    mae_pts = max FAVORABLE excursion (profit direction), NOT max adverse.
    This analysis shows:
      1. Distribution of losing trade sizes (proxy: how bad can it get by bar+20)
      2. % of losers that ended within each stop threshold (lower bound on stop-outs)
    Actual intrabar path may trigger stops on trades that recovered by bar+20.
    """
    lines = []
    n = len(sub)
    if n == 0:
        return lines

    wins   = sub[sub["trade_pts"] > 0]
    losses = sub[sub["trade_pts"] <= 0]
    nl     = len(losses)

    lines.append(f"  STOP SCENARIO ANALYSIS")
    lines.append(f"  NOTE: No intrabar path data. Loss size = end-of-period adverse move (bar+20).")
    lines.append(f"  Actual stop frequency is HIGHER — trades may go adversely before recovering.")
    lines.append(f"")
    lines.append(f"  Loss distribution (NQ pts, at bar+20):")

    if nl:
        lp = losses["trade_pts"].abs()
        lines.append(f"    mean   {lp.mean():>7.2f} pts   median {lp.median():>7.2f} pts")
        lines.append(f"    p75    {lp.quantile(0.75):>7.2f} pts   p90    {lp.quantile(0.90):>7.2f} pts")
        lines.append(f"    max    {lp.max():>7.2f} pts")
    lines.append(f"")

    # Stop thresholds in NQ pts
    stops = [5, 10, 15, 20, 25, 30]
    lines.append(f"  {'stop (pts)':>12}  {'losers_within':>14}  {'losers_exceeded':>16}  {'winners_at_risk':>16}")
    lines.append(f"  " + "-" * 64)

    for sp in stops:
        if nl:
            l_within  = (losses["trade_pts"].abs() <= sp).sum()
            l_exceed  = nl - l_within
            # winners where max favorable (mae_pts) < stop = they barely moved in our favor
            # not directly stop-at-risk, but shows if stop would have been hit on small winners
            w_at_risk = (wins["mae_pts"] < sp).sum() if len(wins) else 0
        else:
            l_within = l_exceed = w_at_risk = 0

        def fmt(v, tot): return f"{v}/{tot} ({v/tot*100:.0f}%)" if tot else "n/a"
        lines.append(
            f"  {sp:>12}  {fmt(l_within, nl):>14}  {fmt(l_exceed, nl):>16}  {fmt(w_at_risk, len(wins)):>16}"
        )

    lines.append(f"")
    lines.append(f"  Interpretation of 'winners_at_risk': these winners had max favorable excursion")
    lines.append(f"  < stop threshold — they wouldn't have been stopped, but are thin winners.")
    lines.append(f"")
    return lines


def expectancy_by_stop_rr(sub, label):
    """
    Expectancy grid at different stop levels in NQ pts.
    Simulates: losers capped at stop_pts, winners take full trade_pts.
    Optimistic — no path-based stop-outs on eventual winners.
    """
    lines = []
    n = len(sub)
    if n == 0:
        return lines

    pts = sub["trade_pts"].values

    stop_levels_pts = [5, 8, 10, 12, 15, 20, 25, 30, 50]

    lines.append(f"  EXPECTANCY GRID — varying stop (NQ pts, optimistic: no path stops)")
    lines.append(f"")
    lines.append(f"  {'stop_pts':>9}  {'win%':>6}  {'avg_win':>9}  {'avg_loss':>9}  {'R:R':>6}  {'E/trade':>9}  {'E($)':>8}")
    lines.append("  " + "-" * 68)

    for sp in stop_levels_pts:
        outcomes = np.where(pts > 0, pts, np.where(pts < -sp, -sp, pts))
        w_mask  = outcomes > 0
        l_mask  = outcomes <= 0
        nw = w_mask.sum()
        nl = l_mask.sum()
        wr  = nw / n if n else 0
        aw  = outcomes[w_mask].mean()       if nw else 0.0
        al  = abs(outcomes[l_mask]).mean()  if nl else 0.0
        rr  = aw / al if al > 0 else float("nan")
        exp = wr * aw - (1-wr) * al
        usd = exp * 20

        def fv(v): return f"{v:.2f}" if not np.isnan(v) else " n/a"
        lines.append(
            f"  {sp:>9}  {wr*100:>5.1f}%  {aw:>+9.2f}  {al:>9.2f}  {fv(rr):>6}  "
            f"{exp:>+9.2f}  {usd:>+8.0f}"
        )

    lines.append(f"")
    return lines


# ── Output ────────────────────────────────────────────────────────────────────
lines_out = []
def p(s=""): lines_out.append(s)

p("=" * 80)
p("EXPECTANCY REPORT — FINAL SETUP SUITE")
p("Training: 2025-03-17 to 2025-11-30  |  183 trading days")
p("=" * 80)
p()
p("CRITICAL SEMANTIC NOTE:")
p("  All three setups are REVERSAL trades — they trade AGAINST the absorption direction.")
p("  sell_absorbed (Setup A, B) = bullish absorption bar → traded SHORT at upper extreme")
p("  buy_absorbed  (Setup C)    = bearish absorption bar → traded LONG at lower extreme")
p()
p("  The labeler's move_normalized and sustained metric measure ABSORPTION DIRECTION.")
p("  For reversal trades: move_normalized < 0 = trade WON (price moved in trade direction).")
p()
p("  mae_normalized for reversal setups = max price movement in TRADE DIRECTION (profit")
p("  potential reached), NOT max adverse excursion. The labeler does not capture the")
p("  adverse excursion for reversal trades — that would require a separate labeling pass.")
p()
p(f"  NQ contract: $20/pt, 1 pt = 4 ticks (0.25 pt/tick)")
p(f"  Median denom (session_vol * signal_price): {median_denom:.1f} pts  [for reference only]")
p()

for setup_label, sub in [
    ("SETUP A — HIGH CONVICTION SHORT (sell_absorbed, outside_vah, +2s/at_band, rejection+proven)", A),
    ("SETUP B — STANDARD SHORT (sell_absorbed, outside_vah, non_dir, no prior abs)", B),
    ("SETUP C — HIGH CONVICTION LONG (buy_absorbed, outside_val, directional, proven)", C),
]:
    p("=" * 80)
    p(setup_label)
    p("=" * 80)
    p()
    for l in expectancy_block(sub, setup_label):
        p(l)

    p("  --- Stop Scenarios ---")
    for l in stop_scenario_block(sub, setup_label):
        p(l)

    p("  --- Expectancy Grid (simulated stop at various levels) ---")
    for l in expectancy_by_stop_rr(sub, setup_label):
        p(l)

# ── Combined / cross-setup ────────────────────────────────────────────────────
p("=" * 80)
p("CROSS-SETUP EXPECTANCY SUMMARY")
p("=" * 80)
p()

p(f"  {'setup':<14}  {'n':>5}  {'win%':>6}  {'avg_win':>9}  {'avg_loss':>9}  {'R:R':>6}  {'E/trade':>9}  {'E/day':>9}")
p("  " + "-" * 75)

for label, sub in [("A  HC Short", A), ("B  Std Short", B), ("C  HC Long", C)]:
    n = len(sub)
    if n == 0:
        p(f"  {label:<14}  {n:>5}  (no signals)")
        continue
    wins   = sub[sub["trade_pts"] > 0]
    losses = sub[sub["trade_pts"] <= 0]
    wr  = len(wins) / n
    aw  = wins["trade_pts"].mean()         if len(wins)   else 0.0
    al  = losses["trade_pts"].abs().mean() if len(losses) else 0.0
    rr  = aw / al if al > 0 else float("nan")
    exp = wr * aw - (1 - wr) * al
    avg_per_day = n / 183
    e_per_day   = exp * avg_per_day
    def fv(v): return f"{v:.2f}" if not np.isnan(v) else "  n/a"
    p(f"  {label:<14}  {n:>5}  {wr*100:>5.1f}%  {aw:>+9.2f}  {al:>9.2f}  {fv(rr):>6}  "
      f"{exp:>+9.2f}  {e_per_day:>+9.3f}")

p()
p("  Units: NQ pts. E/day = E/trade * avg_signals_per_day.")
p()

all_trades = pd.concat([A, B, C], ignore_index=True)
wr_all  = (all_trades["trade_pts"] > 0).mean()
aw_all  = all_trades[all_trades["trade_pts"] > 0]["trade_pts"].mean()
al_all  = all_trades[all_trades["trade_pts"] <= 0]["trade_pts"].abs().mean()
exp_all = wr_all * aw_all - (1 - wr_all) * al_all
p(f"  Combined portfolio (A+B+C as one pool, n={len(all_trades)}):")
p(f"    Overall win%:   {wr_all*100:.1f}%")
p(f"    Avg win:        {aw_all:+.2f} pts  (${aw_all*20:+.0f})")
p(f"    Avg loss:       {al_all:.2f} pts  (${al_all*20:.0f})")
p(f"    R:R:            {aw_all/al_all:.2f}:1")
p(f"    E per trade:    {exp_all:+.2f} pts  ({'POSITIVE' if exp_all > 0 else 'NEGATIVE'})")
p(f"    E per trade $:  ~${exp_all*20:+.0f}")
p(f"    E per day:      {exp_all * len(all_trades)/183:+.3f} pts/day")
p()

# ── Breakeven analysis ────────────────────────────────────────────────────────
p("=" * 80)
p("BREAKEVEN WIN RATE ANALYSIS BY SETUP")
p("=" * 80)
p()
p("  How often do we need to be right to break even at the observed avg win/loss?")
p()
for label, sub in [("Setup A", A), ("Setup B", B), ("Setup C", C)]:
    n = len(sub)
    if n == 0:
        continue
    wins  = sub[sub["trade_pts"] > 0]
    losses= sub[sub["trade_pts"] <= 0]
    aw = wins["trade_pts"].mean()         if len(wins)   else 0.0
    al = losses["trade_pts"].abs().mean() if len(losses) else 0.0
    wr = len(wins) / n
    bw = al / (aw + al) if (aw + al) > 0 else float("nan")
    margin = wr - bw
    verdict = "POSITIVE" if margin > 0 else "NEGATIVE"
    p(f"  {label}: win% = {wr*100:.1f}%  |  breakeven = {bw*100:.1f}%  |  "
      f"margin = {margin*100:+.1f}pp  |  {verdict} EXPECTANCY (no stop)")
    p(f"    avg win {aw:.2f}pts vs avg loss {al:.2f}pts (no stop)")
    p(f"    => Need {bw*100:.1f}% wins to break even. Currently at {wr*100:.1f}%.")
    p()

# ── Save ──────────────────────────────────────────────────────────────────────
full_text = "\n".join(lines_out)
print(full_text)
OUT_PATH.write_text(full_text, encoding="utf-8")
print(f"\nSaved to {OUT_PATH}")
