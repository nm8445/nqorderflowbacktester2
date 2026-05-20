"""
Estimate the Hurst exponent (H) of NQ VOLATILITY from 5 years of 15-min data.
Multiple methods:
  1. Variogram (Gatheral et al. 2018) — the gold standard for rough vol
  2. R/S (Rescaled Range)
  3. DFA (Detrended Fluctuation Analysis)
  4. Multi-scale moment scaling on raw returns

Key: we estimate H of the LOG VOLATILITY process, not the price itself.
Gatheral et al. (2018) found H ~ 0.10 across assets.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")
ET = "America/New_York"


def load_combined():
    """Load combined 5-year 15-min bars."""
    df_mt = pd.read_parquet(MARKETTICK_PARQUET)
    frames = []
    for f in sorted(TIMEBARS_DIR.glob("timebars_5min_202*.pkl")):
        with open(f, "rb") as fh:
            bars = pickle.load(fh)
        if not bars: continue
        rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
                 "low": b["low"], "close": b["close"]} for b in bars]
        df5 = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df5["group"] = df5.index.floor("15min")
        agg = df5.groupby("group").agg(open=("open", "first"), high=("high", "max"),
                                        low=("low", "min"), close=("close", "last"))
        agg.index += pd.Timedelta(minutes=15)
        frames.append(agg)
    df_pkl = pd.concat(frames).sort_index()
    df_pkl = df_pkl[~df_pkl.index.duplicated(keep="first")]
    cutoff = df_mt.index[-1]
    df_pkl_new = df_pkl[df_pkl.index > cutoff]
    df = pd.concat([df_mt[["open", "high", "low", "close"]], df_pkl_new[["open", "high", "low", "close"]]]).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    print(f"Loaded {len(df):,} bars ({df.index[0]} to {df.index[-1]})")
    return df


def compute_log_volatility(df, window=30):
    """
    Compute realized log-volatility series.
    Use rolling std of log returns as volatility proxy, then take log.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    rvol = log_ret.rolling(window).std().dropna()
    log_vol = np.log(rvol).replace([np.inf, -np.inf], np.nan).dropna()
    return log_ret.values, log_vol.values


def method_variogram(log_vol, max_lag=100):
    """
    Gatheral et al. (2018) variogram method.
    E[|X(t+lag) - X(t)|^2] ~ C * lag^(2H)
    log(variogram) vs log(lag) has slope 2H.
    """
    print("\n--- Method 1: Variogram (Gatheral et al. 2018) ---")
    lags = np.arange(1, max_lag + 1)
    variogram = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        diffs = log_vol[lag:] - log_vol[:-lag]
        variogram[i] = np.mean(diffs ** 2)

    log_lags = np.log(lags)
    log_var = np.log(variogram)

    # Fit on short lags (1-20) where the power law is cleanest
    short = min(20, max_lag)
    slope_short, intercept, r, p, se = stats.linregress(log_lags[:short], log_var[:short])
    H_short = slope_short / 2

    slope_all, _, r_all, _, se_all = stats.linregress(log_lags, log_var)
    H_all = slope_all / 2

    print(f"  Short lags (1-{short}): H = {H_short:.4f}  (slope={slope_short:.4f}, R^2={r**2:.4f})")
    print(f"  All lags (1-{max_lag}):  H = {H_all:.4f}  (slope={slope_all:.4f}, R^2={r_all**2:.4f})")

    for start, end in [(1, 10), (1, 30), (1, 50), (5, 50)]:
        sl, _, r2, _, _ = stats.linregress(log_lags[start-1:end], log_var[start-1:end])
        print(f"  Lags {start}-{end}: H = {sl/2:.4f}  (R^2={r2**2:.4f})")

    return H_short


def method_rs(series, min_n=10, max_n=None):
    """R/S (Rescaled Range) analysis. H = slope of log(R/S) vs log(n)."""
    print("\n--- Method 2: R/S (Rescaled Range) ---")
    N = len(series)
    if max_n is None:
        max_n = N // 4

    ns = []
    n = min_n
    while n <= max_n:
        ns.append(n)
        n = int(n * 1.5)
    ns = np.array(ns)

    rs_values = []
    for n in ns:
        num_blocks = N // n
        if num_blocks < 2:
            continue
        rs_block = []
        for b in range(num_blocks):
            block = series[b * n:(b + 1) * n]
            mean_b = np.mean(block)
            devs = np.cumsum(block - mean_b)
            R = np.max(devs) - np.min(devs)
            S = np.std(block, ddof=1)
            if S > 0:
                rs_block.append(R / S)
        if rs_block:
            rs_values.append((n, np.mean(rs_block)))

    if len(rs_values) < 3:
        print("  Not enough data points")
        return 0.5

    ns_used = np.array([x[0] for x in rs_values])
    rs_used = np.array([x[1] for x in rs_values])

    slope, intercept, r, p, se = stats.linregress(np.log(ns_used), np.log(rs_used))
    print(f"  H = {slope:.4f}  (R^2={r**2:.4f}, {len(rs_values)} points, n={ns_used[0]}-{ns_used[-1]})")
    return slope


def method_dfa(series, min_n=10, max_n=None, order=1):
    """DFA (Detrended Fluctuation Analysis). H = slope of log(F(n)) vs log(n)."""
    print("\n--- Method 3: DFA (Detrended Fluctuation Analysis) ---")
    N = len(series)
    if max_n is None:
        max_n = N // 4

    Y = np.cumsum(series - np.mean(series))

    ns = []
    n = min_n
    while n <= max_n:
        ns.append(int(n))
        n *= 1.3
    ns = sorted(set(ns))

    fluctuations = []
    ns_used = []

    for n in ns:
        num_blocks = N // n
        if num_blocks < 2:
            continue
        f2 = []
        for b in range(num_blocks):
            segment = Y[b * n:(b + 1) * n]
            x = np.arange(n)
            coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(coeffs, x)
            f2.append(np.mean((segment - trend) ** 2))
        # Also from end
        for b in range(num_blocks):
            segment = Y[N - (b + 1) * n:N - b * n]
            x = np.arange(n)
            coeffs = np.polyfit(x, segment, order)
            trend = np.polyval(coeffs, x)
            f2.append(np.mean((segment - trend) ** 2))

        F_n = np.sqrt(np.mean(f2))
        fluctuations.append(F_n)
        ns_used.append(n)

    ns_used = np.array(ns_used)
    fluctuations = np.array(fluctuations)

    slope, intercept, r, p, se = stats.linregress(np.log(ns_used), np.log(fluctuations))
    print(f"  H = {slope:.4f}  (R^2={r**2:.4f}, {len(ns_used)} points, n={ns_used[0]}-{ns_used[-1]})")
    return slope


def method_abspower(log_ret_vals):
    """
    Multi-scale absolute moment scaling on raw returns.
    E[|r_agg|^q] at different aggregation scales reveals vol roughness.
    """
    print("\n--- Method 4: Multi-scale moment scaling (returns) ---")
    scales = [1, 2, 4, 8, 16, 32, 64, 128]
    powers = [0.5, 1.0, 1.5, 2.0]

    for q in powers:
        log_scales = []
        log_moments = []
        for s in scales:
            n_blocks = len(log_ret_vals) // s
            if n_blocks < 50:
                continue
            agg_ret = np.array([log_ret_vals[i*s:(i+1)*s].sum() for i in range(n_blocks)])
            moment = np.mean(np.abs(agg_ret) ** q)
            if moment > 0:
                log_scales.append(np.log(s))
                log_moments.append(np.log(moment))

        if len(log_scales) >= 3:
            slope, _, r, _, _ = stats.linregress(log_scales, log_moments)
            H_est = slope / q
            print(f"  q={q:.1f}: H = {H_est:.4f}  (slope={slope:.4f}, R^2={r**2:.4f})")


def estimate_by_year(df, window=30):
    """Estimate H year-by-year to see regime changes."""
    print("\n--- Yearly H estimates (variogram, lags 1-20) ---")
    print(f"{'Year':<6} {'H_vol':>8} {'H_price':>8} {'N bars':>8}")
    print("-" * 35)

    df_et = df.copy()
    df_et.index = pd.DatetimeIndex([
        t.tz_convert(ET) if hasattr(t, "tz_convert") and t.tzinfo
        else pd.Timestamp(t).tz_localize("UTC").tz_convert(ET)
        for t in df_et.index
    ])

    for year in sorted(set(df_et.index.year)):
        mask = df_et.index.year == year
        df_year = df_et[mask]
        if len(df_year) < 500:
            continue

        # H of volatility
        log_ret = np.log(df_year["close"] / df_year["close"].shift(1)).dropna()
        rvol = log_ret.rolling(window).std().dropna()
        log_vol = np.log(rvol).replace([np.inf, -np.inf], np.nan).dropna().values

        # H of price returns
        price_ret = log_ret.values

        lags = np.arange(1, 21)

        # Variogram on log-vol
        if len(log_vol) >= 200:
            var_vol = np.zeros(len(lags))
            for i, lag in enumerate(lags):
                diffs = log_vol[lag:] - log_vol[:-lag]
                var_vol[i] = np.mean(diffs ** 2)
            slope_v, _, _, _, _ = stats.linregress(np.log(lags), np.log(var_vol))
            H_vol = slope_v / 2
        else:
            H_vol = np.nan

        # Variogram on cumulative returns (price)
        cum_ret = np.cumsum(price_ret)
        var_price = np.zeros(len(lags))
        for i, lag in enumerate(lags):
            diffs = cum_ret[lag:] - cum_ret[:-lag]
            var_price[i] = np.mean(diffs ** 2)
        slope_p, _, _, _, _ = stats.linregress(np.log(lags), np.log(var_price))
        H_price = slope_p / 2

        print(f"{year:<6} {H_vol:>8.4f} {H_price:>8.4f} {len(df_year):>8,}")


def main():
    df = load_combined()

    print("\nComputing log volatility (window=30)...")
    log_ret_vals, log_vol = compute_log_volatility(df, window=30)
    print(f"Log-vol series length: {len(log_vol):,}")

    # Run all methods on VOLATILITY
    H1 = method_variogram(log_vol, max_lag=100)
    H2 = method_rs(log_vol)
    H3 = method_dfa(log_vol)
    method_abspower(log_ret_vals)

    print("\n" + "=" * 60)
    print("SUMMARY — H of NQ LOG-VOLATILITY")
    print("=" * 60)
    print(f"  Variogram (lags 1-20):  H = {H1:.4f}")
    print(f"  R/S analysis:           H = {H2:.4f}")
    print(f"  DFA:                    H = {H3:.4f}")
    print(f"  Average:                H = {(H1 + H2 + H3) / 3:.4f}")
    print()
    print("  H < 0.5  -> rough / anti-persistent")
    print("  H = 0.5  -> Brownian motion")
    print("  H > 0.5  -> smooth / persistent")
    print()
    print("  Gatheral et al. (2018) found H ~ 0.10 across assets")
    print(f"  Current locked config uses H = 0.10")
    print(f"  NQGRAY config uses H = 0.40")
    print()

    # Year-by-year
    estimate_by_year(df, window=30)

    # Sensitivity to vol window
    print("\n--- Sensitivity to vol estimation window ---")
    for w in [10, 20, 30, 50, 100]:
        _, lv = compute_log_volatility(df, window=w)
        lags = np.arange(1, 21)
        var = np.zeros(len(lags))
        for i, lag in enumerate(lags):
            diffs = lv[lag:] - lv[:-lag]
            var[i] = np.mean(diffs ** 2)
        slope, _, r, _, _ = stats.linregress(np.log(lags), np.log(var))
        print(f"  window={w:>3}: H = {slope/2:.4f}  (R^2={r**2:.4f})")


if __name__ == "__main__":
    main()
