"""
Shared core for rough vol + orderflow strategy.
- build_bars: aggregate 5-min timebars to 15m or 20m (tz-aware ET)
- compute_zvol: rough vol z-score (numpy)
- compute_ema, compute_atr: feature arrays
- backtest_jit: numba-fast tight-loop backtest (phase 1 base + optional orderflow + gamma)
- build_orderflow_features: per-bar lower/upper half windowed and multi-level absorption
- load_gamma_signs: qqq_gamma_sign per session date
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit

# Paths
TIMEBARS_DIRS = [
    Path("D:/trading_pythonbacktest_data/timebars_5min_5yr"),  # 2020-12 -> 2025-04
    Path("D:/trading_pythonbacktest_data/timebars_5min"),       # 2025-03 -> current
]
VOLUMETRIC_PATH = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
MENTHORQ_PATH = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")

# Strategy constants
H = 0.40
KERNEL_LEN = 80
ETA = 1.0
V0 = 0.0001
ATR_LEN = 14
ET = "America/New_York"
POINT_VALUE = 20.0
TICK_SIZE = 0.25

# Session 05:45 - 14:45 ET (inclusive of 05:45, exclusive of 14:45; force-close at >= 14:45)
SESSION_START_MIN = 5 * 60 + 45   # 345
SESSION_END_MIN   = 14 * 60 + 45  # 885
MAX_TRADES_PER_DAY = 5

# IS/OOS split
IS_END = pd.Timestamp("2024-12-31").date()


def make_kernel():
    ks = np.arange(1, KERNEL_LEN + 1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = ks ** (H - 0.5) - (ks - 1) ** (H - 0.5)
    k[0] = 1.0
    return k


KERNEL = make_kernel()


def build_bars(bar_minutes: int) -> pd.DataFrame:
    """Aggregate 5-min pkl timebars to N-min bars; index = bar close time, tz=ET."""
    files_by_date = {}
    for d in TIMEBARS_DIRS:
        for f in sorted(d.glob("timebars_5min_202*.pkl")):
            files_by_date[f.stem] = f  # later folder overrides earlier on same date
    frames = []
    for stem in sorted(files_by_date.keys()):
        f = files_by_date[stem]
        with open(f, "rb") as fh:
            bars = pickle.load(fh)
        if not bars:
            continue
        rows = [{"timestamp": b["open_time"], "open": b["open"], "high": b["high"],
                 "low": b["low"], "close": b["close"]} for b in bars]
        df5 = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df5["group"] = df5.index.floor(f"{bar_minutes}min")
        agg = df5.groupby("group").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        )
        agg.index += pd.Timedelta(minutes=bar_minutes)
        frames.append(agg)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.DatetimeIndex([
        t.tz_convert(ET) if hasattr(t, "tz_convert") and t.tzinfo
        else pd.Timestamp(t).tz_localize("UTC").tz_convert(ET)
        for t in df.index
    ])
    return df


def extract_arrays(df: pd.DataFrame):
    """Return numpy arrays + meta for backtest."""
    highs = df["high"].to_numpy(np.float64)
    lows = df["low"].to_numpy(np.float64)
    closes = df["close"].to_numpy(np.float64)
    opens = df["open"].to_numpy(np.float64)
    idx_et = df.index
    minutes_of_day = (idx_et.hour * 60 + idx_et.minute).to_numpy(np.int32)
    # day_idx = ordinal of the (session-style) date
    # use the ET date of the bar close
    dates = idx_et.date
    day_idx = np.array([d.toordinal() for d in dates], dtype=np.int64)
    is_end_ord = IS_END.toordinal()
    return dict(
        opens=opens, highs=highs, lows=lows, closes=closes,
        minutes_of_day=minutes_of_day, day_idx=day_idx,
        is_end_ord=is_end_ord, idx_et=idx_et,
    )


def compute_atr(highs, lows, closes, atr_len=ATR_LEN):
    n = len(closes)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        a = highs[i] - lows[i]
        b = abs(highs[i] - closes[i - 1])
        c = abs(lows[i] - closes[i - 1])
        tr[i] = max(a, b, c)
    atr = np.zeros(n, dtype=np.float64)
    atr[0] = tr[0]
    if n > 1:
        for i in range(1, min(atr_len, n)):
            atr[i] = tr[: i + 1].mean()
        for i in range(atr_len, n):
            atr[i] = (atr[i - 1] * (atr_len - 1) + tr[i]) / atr_len
    return atr


def compute_zvol(closes, norm_len, zlook):
    """Rough vol z-score. Returns numpy array, NaN-safe."""
    n = len(closes)
    ret = np.zeros(n, dtype=np.float64)
    ret[1:] = np.log(closes[1:] / closes[:-1])
    s = pd.Series(ret)
    mean_ret = s.rolling(norm_len).mean().to_numpy()
    std_ret = s.rolling(norm_len).std(ddof=0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        shock = np.where(std_ret > 0, (ret - mean_ret) / std_ret, 0.0)
    shock = np.nan_to_num(shock, nan=0.0, posinf=0.0, neginf=0.0)
    xH = np.convolve(shock, KERNEL, mode="full")[:n]
    v_model = np.minimum(V0 * np.exp(ETA * xH), V0 * 1e4)
    vs = pd.Series(v_model)
    vm = vs.rolling(zlook).mean().to_numpy()
    vd = vs.rolling(zlook).std(ddof=0).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(vd > 0, (v_model - vm) / vd, 0.0)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def compute_ema(closes, ema_len):
    return pd.Series(closes).ewm(span=ema_len, adjust=False).mean().to_numpy()


# Numba backtest signatures
# orderflow_mask: int8 array (1=allow, 0=skip). If None equivalent, pass all-ones.
# gamma_signs: int8 array per bar, gamma sign of prev-trading-day (-1/0/+1)
# gamma_mode: 0=off, 1=drop_neg (skip when sign==-1), 2=drop_pos (skip when sign==+1)

@njit(cache=True, fastmath=True)
def backtest_jit(highs, lows, closes,
                 z_vol, ema, atr,
                 minutes_of_day, day_idx,
                 long_mask, short_mask, gamma_signs,
                 high_z, atr_sl, atr_tp,
                 ss_min, se_min, max_trades,
                 gamma_mode, is_end_ord):
    n = closes.shape[0]
    pnls = np.zeros(n, dtype=np.float64)
    in_is = np.zeros(n, dtype=np.int8)
    n_trades = 0

    pos = 0
    ep = 0.0
    sl_p = 0.0
    tp_p = 0.0
    cur_day = np.int64(-1)
    dt = 0

    for i in range(n):
        mod = minutes_of_day[i]
        in_session = (mod >= ss_min) and (mod < se_min)

        # Force close at/after session end
        if pos != 0 and (not in_session) and mod >= se_min:
            pnl = (closes[i] - ep) * pos
            pnls[n_trades] = pnl
            in_is[n_trades] = 1 if day_idx[i] <= is_end_ord else 0
            n_trades += 1
            pos = 0

        if not in_session:
            continue

        if day_idx[i] != cur_day:
            cur_day = day_idx[i]
            dt = 0

        # Exit check (using bar i's high/low against SL/TP set on a prior bar)
        if pos != 0:
            exited = False
            xp = 0.0
            if pos == 1:
                if lows[i] <= sl_p:
                    xp = sl_p
                    exited = True
                elif highs[i] >= tp_p:
                    xp = tp_p
                    exited = True
            else:
                if highs[i] >= sl_p:
                    xp = sl_p
                    exited = True
                elif lows[i] <= tp_p:
                    xp = tp_p
                    exited = True
            if exited:
                pnl = (xp - ep) * pos
                pnls[n_trades] = pnl
                in_is[n_trades] = 1 if day_idx[i] <= is_end_ord else 0
                n_trades += 1
                pos = 0
                continue

        # Entry
        if pos == 0 and dt < max_trades:
            atr_v = atr[i]
            if atr_v <= 0.0:
                continue
            z = z_vol[i]
            cl = closes[i]
            em = ema[i]
            if z > high_z:
                new_dir = 0
                if cl > em:
                    new_dir = 1
                elif cl < em:
                    new_dir = -1
                if new_dir != 0:
                    # Orderflow gate (direction-specific)
                    if new_dir == 1 and long_mask[i] == 0:
                        continue
                    if new_dir == -1 and short_mask[i] == 0:
                        continue
                    # Gamma gate
                    if gamma_mode != 0:
                        g = gamma_signs[i]
                        if gamma_mode == 1 and g == -1:
                            continue
                        if gamma_mode == 2 and g == 1:
                            continue
                    pos = new_dir
                    ep = cl
                    if new_dir == 1:
                        sl_p = cl - atr_sl * atr_v
                        tp_p = cl + atr_tp * atr_v
                    else:
                        sl_p = cl + atr_sl * atr_v
                        tp_p = cl - atr_tp * atr_v
                    dt += 1

    return pnls[:n_trades], in_is[:n_trades]


def calc_metrics(pnls_pts, in_is, point_value=POINT_VALUE):
    """Return dict with IS/OOS/total metrics."""
    pnls_dollars = pnls_pts * point_value
    is_mask = in_is.astype(bool)
    res = {}
    for name, mask in (("is", is_mask), ("oos", ~is_mask), ("total", np.ones_like(is_mask, dtype=bool))):
        p = pnls_dollars[mask]
        if len(p) == 0:
            res[name] = dict(trades=0, pf=0.0, wr=0.0, pnl=0.0, mdd=0.0)
            continue
        w = p[p > 0]
        l = p[p < 0]
        pf = float(w.sum() / abs(l.sum())) if len(l) else 99.0
        wr = 100.0 * len(w) / len(p)
        cum = p.cumsum()
        mdd = float((cum - np.maximum.accumulate(cum)).min())
        res[name] = dict(trades=int(len(p)), pf=pf, wr=wr, pnl=float(p.sum()), mdd=mdd)
    return res


# ---------- Orderflow features (phase 2) ----------

def build_orderflow_features(bars: pd.DataFrame, bar_minutes: int,
                              window_n_ticks_list, window_d_list,
                              per_level_d_list, k_list):
    """
    For each signal bar in `bars`, precompute orderflow filter masks for:
      - windowed-absorption: best N-tick contiguous run in lower (long) / upper (short) half
        with cumulative |delta| >= D, with absorbed-side sign (LONG: sum_delta <= -D,
        SHORT: sum_delta >= +D).
      - multi-level: count of tick-levels in lower (long) / upper (short) half whose
        per-level absorbed-side |delta| >= per_level_d, with count >= K.
    Returns:
      dict keyed by (variant, *params) -> dict with 'long_mask' and 'short_mask' int8 arrays
      aligned to bars index.
    """
    vol = pd.read_parquet(VOLUMETRIC_PATH)
    # vol uses 5-min bars; aggregate to bar_minutes using bar_open_time.floor(bar_minutes)
    vol["bar_open_time"] = pd.to_datetime(vol["bar_open_time"], utc=False)
    if vol["bar_open_time"].dt.tz is None:
        vol["bar_open_time"] = vol["bar_open_time"].dt.tz_localize(ET)
    else:
        vol["bar_open_time"] = vol["bar_open_time"].dt.tz_convert(ET)
    vol["group"] = vol["bar_open_time"].dt.floor(f"{bar_minutes}min")
    # The "close time" we use for signal index is group + bar_minutes
    vol["bar_close_time"] = vol["group"] + pd.Timedelta(minutes=bar_minutes)
    grp = vol.groupby(["bar_close_time", "level_price"], as_index=False).agg(
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
    )
    grp["delta"] = grp["buy_vol"] - grp["sell_vol"]

    # Index of bars
    bar_close_times = bars.index
    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    n = len(bars)

    # Build a dict bar_close_time -> sorted (level_price, delta) array
    print(f"  grouping orderflow by bar... {len(grp)} (level,bar) rows")
    grp = grp.sort_values(["bar_close_time", "level_price"])
    bar_groups = dict(tuple(grp.groupby("bar_close_time")[["level_price", "delta"]]))
    print(f"  {len(bar_groups)} unique bars with orderflow data")

    # Precompute features for each bar
    # For each bar: get levels in lower half (price < mid) and upper half (price > mid)
    # Compute, for each (N, D) windowed: best lower-half N-run sum_delta (most negative); upper-half N-run (most positive)
    # Compute, for each (per_level_d, K) multi: count of lower-half levels with delta <= -per_level_d; upper-half levels with delta >= +per_level_d

    # Storage: arrays per bar
    # window keys: (N, D) -> long_mask, short_mask
    # multi keys: (per_level_d, K) -> long_mask, short_mask

    window_keys = [(N, D) for N in window_n_ticks_list for D in window_d_list]
    multi_keys = [(pd_, K) for pd_ in per_level_d_list for K in k_list]

    window_long = {k: np.zeros(n, dtype=np.int8) for k in window_keys}
    window_short = {k: np.zeros(n, dtype=np.int8) for k in window_keys}
    multi_long = {k: np.zeros(n, dtype=np.int8) for k in multi_keys}
    multi_short = {k: np.zeros(n, dtype=np.int8) for k in multi_keys}

    tick = TICK_SIZE

    for i, t in enumerate(bar_close_times):
        if t not in bar_groups:
            continue
        sub = bar_groups[t]
        prices = sub["level_price"].to_numpy()
        deltas = sub["delta"].to_numpy()
        mid = (highs[i] + lows[i]) / 2.0

        lower_idx = prices < mid
        upper_idx = prices > mid
        lower_deltas = deltas[lower_idx]
        upper_deltas = deltas[upper_idx]
        lower_prices = prices[lower_idx]
        upper_prices = prices[upper_idx]

        # --- windowed: contiguous N-tick run with absorbed-side sign threshold ---
        # We need contiguous price runs. Build a dense array indexed by tick offset.
        if len(lower_deltas) > 0:
            lo_min = lower_prices.min()
            lo_max = lower_prices.max()
            slots = int(round((lo_max - lo_min) / tick)) + 1
            arr_lo = np.zeros(slots, dtype=np.float64)
            for j in range(len(lower_prices)):
                slot = int(round((lower_prices[j] - lo_min) / tick))
                if 0 <= slot < slots:
                    arr_lo[slot] = lower_deltas[j]
            # Cumulative running sum for each window size N
            for N in window_n_ticks_list:
                if slots < N:
                    continue
                # rolling sum
                csum = np.cumsum(arr_lo)
                csum_pad = np.concatenate(([0.0], csum))
                windows = csum_pad[N:] - csum_pad[:-N]
                # LONG: most negative window (absorbed selling)
                min_w = windows.min()
                for D in window_d_list:
                    if min_w <= -D:
                        window_long[(N, D)][i] = 1
        if len(upper_deltas) > 0:
            up_min = upper_prices.min()
            up_max = upper_prices.max()
            slots = int(round((up_max - up_min) / tick)) + 1
            arr_up = np.zeros(slots, dtype=np.float64)
            for j in range(len(upper_prices)):
                slot = int(round((upper_prices[j] - up_min) / tick))
                if 0 <= slot < slots:
                    arr_up[slot] = upper_deltas[j]
            for N in window_n_ticks_list:
                if slots < N:
                    continue
                csum = np.cumsum(arr_up)
                csum_pad = np.concatenate(([0.0], csum))
                windows = csum_pad[N:] - csum_pad[:-N]
                max_w = windows.max()
                for D in window_d_list:
                    if max_w >= D:
                        window_short[(N, D)][i] = 1

        # --- multi-level: count of levels exceeding per-level threshold ---
        # LONG: count of lower-half levels with delta <= -per_level_d
        # SHORT: count of upper-half levels with delta >= +per_level_d
        for pd_v in per_level_d_list:
            cnt_lo = int((lower_deltas <= -pd_v).sum())
            cnt_up = int((upper_deltas >= pd_v).sum())
            for K in k_list:
                if cnt_lo >= K:
                    multi_long[(pd_v, K)][i] = 1
                if cnt_up >= K:
                    multi_short[(pd_v, K)][i] = 1

    return dict(window_long=window_long, window_short=window_short,
                multi_long=multi_long, multi_short=multi_short)


def load_gamma_signs(idx_et: pd.DatetimeIndex):
    """Return int8 array of qqq_gamma_sign per bar (prev trading day's sign, -1/0/+1)."""
    g = pd.read_parquet(MENTHORQ_PATH)[["date", "qqq_gamma_sign"]].copy()
    g["date"] = pd.to_datetime(g["date"]).dt.date
    g = g.sort_values("date").reset_index(drop=True)
    # prev-day sign: shift by 1
    g["prev_sign"] = g["qqq_gamma_sign"].shift(1).fillna(0).astype(np.int8)
    sign_map = dict(zip(g["date"], g["prev_sign"]))
    bar_dates = idx_et.date
    return np.array([sign_map.get(d, 0) for d in bar_dates], dtype=np.int8)


def warmup_jit():
    """Trigger numba JIT compile so workers don't each pay it."""
    n = 200
    h = np.ones(n, dtype=np.float64) * 100.0
    l = np.ones(n, dtype=np.float64) * 99.0
    c = np.ones(n, dtype=np.float64) * 100.0
    z = np.zeros(n, dtype=np.float64)
    e = np.ones(n, dtype=np.float64) * 100.0
    a = np.ones(n, dtype=np.float64) * 0.5
    mod = np.zeros(n, dtype=np.int32)
    di = np.zeros(n, dtype=np.int64)
    of = np.ones(n, dtype=np.int8)
    gs = np.zeros(n, dtype=np.int8)
    backtest_jit(h, l, c, z, e, a, mod, di, of, of, gs,
                 1.5, 2.0, 1.0, 345, 885, 5, 0, 100000)
