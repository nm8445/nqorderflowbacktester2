"""
Quick test: fetch one day of NQ MBP-1 data and print a summary.

Run from the project root:
    python scripts/fetch_sample.py

Make sure .env exists with your DATABENTO_API_KEY set.
"""

import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from nqbt.data.loader import fetch_and_load
from nqbt.data.schema import Action

# --- Adjust these as needed ---
START = "2025-04-07"
END   = "2025-04-09"   # exclusive, so this fetches just Nov 1
# ------------------------------

print(f"Fetching NQ MBP-1 for {START} ...")
df = fetch_and_load(START, END)

print(f"\nLoaded {len(df):,} events")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nAction distribution:")
print(df["action"].value_counts())

trades = df[df["action"] == Action.TRADE]
print(f"\nTrade events: {len(trades):,}")
if len(trades):
    print(f"Price range: {trades['price'].min():.2f} - {trades['price'].max():.2f}")
    print(f"Total volume: {trades['size'].sum():,} contracts")
