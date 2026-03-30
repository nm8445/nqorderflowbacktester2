import pandas as pd
from nqbt.data.loader import fetch_and_load
from nqbt.data.normalizer import normalize

df = fetch_and_load("2026-03-10", "2026-03-12")
ticks = normalize(df)

start = pd.Timestamp("2026-03-11 06:00:00", tz="America/New_York")
end   = pd.Timestamp("2026-03-11 11:00:00", tz="America/New_York")
window = ticks[(ticks.index >= start) & (ticks.index <= end)]

display = window[["price", "size", "aggressor"]].copy()
display.index = window.index.tz_convert("America/New_York").strftime("%I:%M:%S %p ET")
print(display.to_string())
