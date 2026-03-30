"""
Large volume trade checker for a specific session date.

Shows all trades >= MIN_SIZE on a given CME session day, grouped by price level
to reveal where aggressive size was consistently hitting bid or ask.
"""

import pandas as pd
from nqbt.data.loader import fetch_and_load
from nqbt.data.normalizer import normalize

# --- Config ---
SESSION_DATE = "2026-03-11"
MIN_SIZE     = 50
# --------------

# CME session opens at 5pm CT the prior calendar day (23:00 UTC),
# so fetch one day before through one day after to capture the full session.
fetch_start = str(pd.Timestamp(SESSION_DATE) - pd.Timedelta(days=1))[:10]
fetch_end   = str(pd.Timestamp(SESSION_DATE) + pd.Timedelta(days=1))[:10]

print(f"Fetching data for session {SESSION_DATE} (fetching {fetch_start} to {fetch_end})...")
df_raw = fetch_and_load(fetch_start, fetch_end)
ticks  = normalize(df_raw)

# Filter to target session and minimum size
session = ticks[
    (ticks["session_date"] == pd.Timestamp(SESSION_DATE).date()) &
    (ticks["size"] >= MIN_SIZE)
].copy()

print(f"\nTotal large trades (>= {MIN_SIZE} contracts): {len(session)}\n")

if session.empty:
    print("No trades found.")
else:
    # --- 1. Raw large prints in time order ---
    display = session[["price", "size", "aggressor"]].copy()
    display.index = session.index.tz_convert("America/New_York").strftime("%Y-%m-%d %I:%M:%S %p ET")
    display.index.name = "time (CT)"

    print("=" * 65)
    print("LARGE PRINTS — time order")
    print("=" * 65)
    print(display.to_string())

    # --- 2. Grouped by price level ---
    # Shows which prices attracted repeated large aggression
    grouped = (
        session.groupby(["price", "aggressor"])
        .agg(
            hits=("size", "count"),
            total_volume=("size", "sum"),
            max_single=("size", "max"),
        )
        .reset_index()
        .sort_values("total_volume", ascending=False)
    )

    print(f"\n{'=' * 65}")
    print("BY PRICE LEVEL — sorted by total volume")
    print("=" * 65)
    print(grouped.to_string(index=False))

    # --- 3. Delta at each price level ---
    # Net aggression: positive = net buying, negative = net selling
    pivot = session.pivot_table(
        index="price",
        columns="aggressor",
        values="size",
        aggfunc="sum",
        fill_value=0,
    )
    if "buy" not in pivot.columns:
        pivot["buy"] = 0
    if "sell" not in pivot.columns:
        pivot["sell"] = 0
    pivot["delta"] = pivot["buy"] - pivot["sell"]
    pivot["total"] = pivot["buy"] + pivot["sell"]
    pivot = pivot.sort_values("total", ascending=False)

    print(f"\n{'=' * 65}")
    print("DELTA BY PRICE LEVEL (buy vol - sell vol)")
    print("=" * 65)
    print(pivot[["buy", "sell", "delta", "total"]].to_string())
