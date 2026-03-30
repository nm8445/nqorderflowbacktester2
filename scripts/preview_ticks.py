from nqbt.data.loader import fetch_and_load
from nqbt.data.normalizer import normalize

df = fetch_and_load("2025-04-07", "2025-04-09")
ticks = normalize(df)
print(ticks.dtypes)
print()
print(ticks.head(20))
