"""
Optimize ATR-based stops and targets for rough vol strategy.

NO REGIME EXITS - ATR stops only + EOD force close at 4:59pm.
Tests ATR_SL and ATR_TP from 0.5 to 2.5 in 0.25 increments.
Splits data in half: in-sample and out-of-sample.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count
from itertools import product

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("D:/trading_pythonbacktest_data")
TIMEBAR_CACHE_DIR = DATA_DIR / "timebars_15min_from_ticks"

# Fixed parameters
H = 0.10
ETA = 1.0
V0 = 0.0001
KERNEL_LEN = 100
NORM_LEN = 200
Z_LOOKBACK = 100
HIGH_Z = 1.0
lowZ = -1.0
EMA_SLOW = 50
ATR_LEN = 14

# Variable parameters (to optimize)
ATR_SL_RANGE = np.arange(0.5, 2.75, 0.25)
ATR_TP_RANGE = np.arange(0.5, 2.75, 0.25)

MARTINGALE_ENABLED = True
MARTINGALE_MULTIPLIER = 2.0
MAX_MARTINGALE_LEVEL = 2
SESSION_START = "04:15"
SESSION_END = "15:30"
EOD_CLOSE_TIME = "15:30"  # Force close all positions at end of session
POINT_VALUE = 20.0
TICK_SIZE = 0.25
ET = "America/New_York"

# Limit CPU usage
MAX_WORKERS = min(8, max(1, cpu_count() - 2))


def load_15min_bars() -> pd.DataFrame:
    """Load tick-based 15-min bars from cache."""
    timebar_files = sorted(TIMEBAR_CACHE_DIR.glob("*.pkl"))
    all_bars = []
    for f in timebar_files:
        with open(f, 'rb') as fp:
            time_bars = pickle.load(fp)
        for bar in time_bars:
            all_bars.append({
                'timestamp': bar['open_time'],
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar.get('volume', 0)
            })
    df = pd.DataFrame(all_bars)
    df = df.set_index('timestamp').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(ET)
    return df


def compute_atr(high, low, close, period=14):
    """Calculate ATR."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rough_vol_regime(bars: pd.DataFrame) -> pd.DataFrame:
    """Add rough volatility regime indicators."""
    df = bars.copy()
    closes = df["close"]
    ret = np.log(closes / closes.shift(1)).fillna(0)
    mean_ret = ret.rolling(NORM_LEN).mean()
    std_ret = ret.rolling(NORM_LEN).std()
    shock = ((ret - mean_ret) / std_ret.replace(0, np.nan)).fillna(0)
    kernel = np.array([
        (k + 1) ** (H - 0.5) - k ** (H - 0.5)
        for k in range(KERNEL_LEN)
    ])
    shock_arr = shock.values
    n = len(shock_arr)
    xH = np.zeros(n)
    for i in range(KERNEL_LEN, n):
        for k in range(KERNEL_LEN):
            xH[i] += kernel[k] * shock_arr[i - k]
    df["xH"] = xH
    df["v_model"] = V0 * np.exp(ETA * df["xH"])
    df["v_model"] = df["v_model"].clip(upper=V0 * 1e4)
    v_mean = df["v_model"].rolling(Z_LOOKBACK).mean()
    v_std = df["v_model"].rolling(Z_LOOKBACK).std()
    df["z_vol"] = (df["v_model"] - v_mean) / v_std.replace(0, np.nan)
    df["high_vol"] = df["z_vol"] > HIGH_Z
    return df


def add_ema_signals(df: pd.DataFrame, atr_sl: float, atr_tp: float) -> pd.DataFrame:
    """Add EMA and ATR-based signals."""
    closes = df["close"]
    df["ema"] = closes.ewm(span=EMA_SLOW, adjust=False).mean()
    df["atr"] = compute_atr(df["high"], df["low"], closes, ATR_LEN)
    is_high_vol = df["z_vol"] > HIGH_Z
    df["long_signal"] = is_high_vol & (closes > df["ema"])
    df["short_signal"] = is_high_vol & (closes < df["ema"])
    df["long_sl"] = closes - atr_sl * df["atr"]
    df["long_tp"] = closes + atr_tp * df["atr"]
    df["short_sl"] = closes + atr_sl * df["atr"]
    df["short_tp"] = closes - atr_tp * df["atr"]
    return df


def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Run backtest with ATR exits only + EOD force close."""
    trades = []
    position = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    entry_time = None
    position_size = 1.0
    martingale_level = 0
    daily_pnl = 0.0
    daily_trades = 0
    current_date = None

    for ts, row in df.iterrows():
        bar_date = ts.date()
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl = 0.0
            daily_trades = 0

        bar_time_str = ts.strftime("%H:%M")
        in_session = SESSION_START <= bar_time_str <= SESSION_END
        # Force close on first bar outside session
        is_eod = not in_session and position != 0
        close = row["close"]
        high = row["high"]
        low = row["low"]

        # Force close positions at EOD
        if position != 0 and is_eod:
            exit_price = close
            exit_reason = "EOD"

            if position == 1:
                pnl = (exit_price - entry_price) * POINT_VALUE * position_size
            else:
                pnl = (entry_price - exit_price) * POINT_VALUE * position_size

            daily_pnl += pnl
            trades.append({
                "entry_time": entry_time,
                "exit_time": ts,
                "direction": "LONG" if position == 1 else "SHORT",
                "contracts": position_size,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": round(pnl, 2),
                "exit_reason": exit_reason,
                "martingale_level": martingale_level,
            })

            if MARTINGALE_ENABLED:
                if pnl <= 0:
                    if martingale_level < MAX_MARTINGALE_LEVEL:
                        martingale_level += 1
                else:
                    martingale_level = 0

            position = 0

        # Check ATR exits for open positions
        if position == 1:
            hit_sl = low <= sl_price
            hit_tp = high >= tp_price

            if hit_sl or hit_tp:
                exit_price = sl_price if hit_sl else tp_price
                exit_reason = "SL" if hit_sl else "TP"
                pnl = (exit_price - entry_price) * POINT_VALUE * position_size
                daily_pnl += pnl
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "direction": "LONG",
                    "contracts": position_size,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": round(pnl, 2),
                    "exit_reason": exit_reason,
                    "martingale_level": martingale_level,
                })
                if MARTINGALE_ENABLED:
                    if pnl <= 0:
                        if martingale_level < MAX_MARTINGALE_LEVEL:
                            martingale_level += 1
                    else:
                        martingale_level = 0
                position = 0

        elif position == -1:
            hit_sl = high >= sl_price
            hit_tp = low <= tp_price

            if hit_sl or hit_tp:
                exit_price = sl_price if hit_sl else tp_price
                exit_reason = "SL" if hit_sl else "TP"
                pnl = (entry_price - exit_price) * POINT_VALUE * position_size
                daily_pnl += pnl
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "direction": "SHORT",
                    "contracts": position_size,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": round(pnl, 2),
                    "exit_reason": exit_reason,
                    "martingale_level": martingale_level,
                })
                if MARTINGALE_ENABLED:
                    if pnl <= 0:
                        if martingale_level < MAX_MARTINGALE_LEVEL:
                            martingale_level += 1
                    else:
                        martingale_level = 0
                position = 0

        # Entry logic
        can_enter = (
            position == 0
            and in_session
            and not is_eod  # Don't enter at EOD
            and not pd.isna(row.get("z_vol"))
            and not pd.isna(row.get("atr"))
        )

        if can_enter:
            position_size = MARTINGALE_MULTIPLIER ** martingale_level
            if row["long_signal"]:
                position = 1
                entry_price = close
                sl_price = row["long_sl"]
                tp_price = row["long_tp"]
                entry_time = ts
                daily_trades += 1
            elif row["short_signal"]:
                position = -1
                entry_price = close
                sl_price = row["short_sl"]
                tp_price = row["short_tp"]
                entry_time = ts
                daily_trades += 1

    return pd.DataFrame(trades)


def calculate_metrics(trades: pd.DataFrame) -> dict:
    """Calculate performance metrics."""
    if trades.empty:
        return {
            'total_pnl': 0,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_pnl': 0,
            'max_dd': 0
        }

    total_pnl = trades['pnl'].sum()
    total_trades = len(trades)
    winners = trades[trades['pnl'] > 0]
    losers = trades[trades['pnl'] <= 0]
    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

    gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

    trades['cumulative'] = trades['pnl'].cumsum()
    trades['peak'] = trades['cumulative'].cummax()
    trades['drawdown'] = trades['cumulative'] - trades['peak']
    max_dd = abs(trades['drawdown'].min())

    return {
        'total_pnl': round(total_pnl, 2),
        'total_trades': total_trades,
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'avg_pnl': round(avg_pnl, 2),
        'max_dd': round(max_dd, 2)
    }


def test_params(args):
    """Test single parameter combination on both in-sample and out-of-sample."""
    atr_sl, atr_tp, signal_df_is, signal_df_oos = args

    # In-sample backtest
    trades_is = backtest(signal_df_is)
    metrics_is = calculate_metrics(trades_is)

    # Out-of-sample backtest
    trades_oos = backtest(signal_df_oos)
    metrics_oos = calculate_metrics(trades_oos)

    return {
        'atr_sl': atr_sl,
        'atr_tp': atr_tp,
        'is_pnl': metrics_is['total_pnl'],
        'is_trades': metrics_is['total_trades'],
        'is_win_rate': metrics_is['win_rate'],
        'is_pf': metrics_is['profit_factor'],
        'is_avg_pnl': metrics_is['avg_pnl'],
        'is_max_dd': metrics_is['max_dd'],
        'oos_pnl': metrics_oos['total_pnl'],
        'oos_trades': metrics_oos['total_trades'],
        'oos_win_rate': metrics_oos['win_rate'],
        'oos_pf': metrics_oos['profit_factor'],
        'oos_avg_pnl': metrics_oos['avg_pnl'],
        'oos_max_dd': metrics_oos['max_dd'],
    }


if __name__ == "__main__":
    print("="*80)
    print("ATR STOP/TARGET OPTIMIZATION (NO REGIME EXITS + EOD CLOSE)")
    print("="*80)
    print(f"ATR_SL range: {ATR_SL_RANGE[0]} to {ATR_SL_RANGE[-1]}")
    print(f"ATR_TP range: {ATR_TP_RANGE[0]} to {ATR_TP_RANGE[-1]}")
    print(f"Total combinations: {len(ATR_SL_RANGE) * len(ATR_TP_RANGE)}")
    print(f"Using {MAX_WORKERS} workers")
    print()

    # Load and prepare data
    print("Loading 15-min bars...")
    bars = load_15min_bars()
    print(f"Loaded {len(bars):,} bars")

    print("Computing rough vol regime...")
    bars_with_vol = compute_rough_vol_regime(bars)

    # Split data in half by date
    mid_date = bars_with_vol.index[len(bars_with_vol) // 2]
    bars_is = bars_with_vol.loc[:mid_date]
    bars_oos = bars_with_vol.loc[mid_date:]

    print(f"\nIn-sample: {bars_is.index[0].date()} to {bars_is.index[-1].date()} ({len(bars_is):,} bars)")
    print(f"Out-of-sample: {bars_oos.index[0].date()} to {bars_oos.index[-1].date()} ({len(bars_oos):,} bars)")
    print()

    # Prepare all parameter combinations with pre-computed signals
    print("Preparing parameter combinations...")
    param_combinations = []
    for atr_sl in ATR_SL_RANGE:
        for atr_tp in ATR_TP_RANGE:
            # Pre-compute signals for this combination
            signal_df_is = add_ema_signals(bars_is.copy(), atr_sl, atr_tp)
            signal_df_oos = add_ema_signals(bars_oos.copy(), atr_sl, atr_tp)
            param_combinations.append((atr_sl, atr_tp, signal_df_is, signal_df_oos))

    print(f"Testing {len(param_combinations)} combinations...")

    # Run optimization in parallel
    with Pool(MAX_WORKERS) as pool:
        results = []
        for i, result in enumerate(pool.imap(test_params, param_combinations), 1):
            results.append(result)
            if i % 20 == 0:
                print(f"Progress: {i}/{len(param_combinations)} ({i/len(param_combinations)*100:.1f}%)")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by out-of-sample profit factor
    results_df = results_df.sort_values('oos_pf', ascending=False)

    # Save to CSV
    output_file = Path("results/csv/atr_optimization_no_regime_exits.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    print()
    print("="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_file}")
    print()
    print("TOP 20 RESULTS BY OUT-OF-SAMPLE PROFIT FACTOR:")
    print("="*80)

    # Display top 20
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(results_df.head(20).to_string(index=False))
