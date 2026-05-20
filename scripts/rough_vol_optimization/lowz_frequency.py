"""How often does z_vol <= -1.0 fire with different params."""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")
ET = "America/New_York"
SS = "09:30"; SE = "15:30"


def build():
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
    df.index = pd.DatetimeIndex([
        t.tz_convert(ET) if hasattr(t, "tz_convert") and t.tzinfo
        else pd.Timestamp(t).tz_localize("UTC").tz_convert(ET)
        for t in df.index
    ])
    return df


def compute_zvol(closes, h, kernel_len, norm_len, z_lookback):
    n = len(closes)
    ret = np.log(closes / closes.shift(1)); ret.iloc[0] = 0.0
    mean_ret = ret.rolling(norm_len).mean()
    std_ret = ret.rolling(norm_len).std(ddof=0)
    shock = np.nan_to_num(np.where(std_ret > 0, (ret - mean_ret) / std_ret, 0.0), nan=0.0)
    ks = np.arange(1, kernel_len + 1, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = ks ** (h - 0.5) - (ks - 1) ** (h - 0.5)
    kernel[0] = 1.0
    xH = np.convolve(shock, kernel, mode="full")[:n]
    v0 = 0.0001; eta = 1.0
    v_model = np.minimum(v0 * np.exp(eta * xH), v0 * 1e4)
    v_s = pd.Series(v_model, index=closes.index)
    v_m = v_s.rolling(z_lookback).mean()
    v_d = v_s.rolling(z_lookback).std(ddof=0)
    z_vol = np.nan_to_num(np.where(v_d > 0, (v_s - v_m) / v_d, 0.0), nan=0.0)
    return pd.Series(z_vol, index=closes.index)


def main():
    print("Loading...", flush=True)
    df = build()
    closes = df["close"]
    n = len(df)

    # Session mask
    session_mask = (df.index.strftime("%H:%M") >= SS) & (df.index.strftime("%H:%M") < SE)
    total_session_bars = session_mask.sum()
    trading_days = len(set(idx.date() for idx in df[session_mask].index))

    print(f"Data: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Session bars: {total_session_bars:,}  Trading days: {trading_days:,}")

    # User's params: H=0.40, kernel=80, norm=201, zlook=23
    print("\n--- User params: H=0.40, Kernel=80, Norm=201, ZLook=23 ---")
    z1 = compute_zvol(closes, 0.40, 80, 201, 23)
    z1_sess = z1[session_mask]

    low_z_bars = (z1_sess <= -1.0).sum()
    low_z_days = len(set(idx.date() for idx in z1_sess[z1_sess <= -1.0].index))
    print(f"z_vol <= -1.0:  {low_z_bars:,} bars ({100*low_z_bars/total_session_bars:.2f}% of session bars)")
    print(f"                {low_z_days} unique days ({100*low_z_days/trading_days:.1f}% of trading days)")
    print(f"                ~{low_z_bars/trading_days:.1f} bars/day on average when it fires")

    # Also show HIGH_Z for comparison
    high_z_bars = (z1_sess >= 1.5).sum()
    high_z_days = len(set(idx.date() for idx in z1_sess[z1_sess >= 1.5].index))
    print(f"\nz_vol >= 1.5:   {high_z_bars:,} bars ({100*high_z_bars/total_session_bars:.2f}% of session bars)")
    print(f"                {high_z_days} unique days ({100*high_z_days/trading_days:.1f}% of trading days)")

    # Compare to original params: H=0.40, kernel=80, norm=250, zlook=100
    print("\n--- Original params: H=0.40, Kernel=80, Norm=250, ZLook=100 ---")
    z2 = compute_zvol(closes, 0.40, 80, 250, 100)
    z2_sess = z2[session_mask]

    low_z_bars2 = (z2_sess <= -1.0).sum()
    low_z_days2 = len(set(idx.date() for idx in z2_sess[z2_sess <= -1.0].index))
    print(f"z_vol <= -1.0:  {low_z_bars2:,} bars ({100*low_z_bars2/total_session_bars:.2f}% of session bars)")
    print(f"                {low_z_days2} unique days ({100*low_z_days2/trading_days:.1f}% of trading days)")

    high_z_bars2 = (z2_sess >= 1.5).sum()
    high_z_days2 = len(set(idx.date() for idx in z2_sess[z2_sess >= 1.5].index))
    print(f"\nz_vol >= 1.5:   {high_z_bars2:,} bars ({100*high_z_bars2/total_session_bars:.2f}% of session bars)")
    print(f"                {high_z_days2} unique days ({100*high_z_days2/trading_days:.1f}% of trading days)")

    # Distribution of z_vol for user params
    print("\n--- z_vol distribution (User params, session bars) ---")
    for thr in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        if thr < 0:
            count = (z1_sess <= thr).sum()
            print(f"  z <= {thr:+.1f}: {count:>6,} bars ({100*count/total_session_bars:.2f}%)")
        else:
            count = (z1_sess >= thr).sum()
            print(f"  z >= {thr:+.1f}: {count:>6,} bars ({100*count/total_session_bars:.2f}%)")


if __name__ == "__main__":
    main()
