"""
Test rough volatility regime filter with EMA entries.

Phase 1: Simple EMA trend + rough vol regime filter
Phase 2: Replace EMA with your orderflow logic (next test)

Config: H=0.57 (from Hurst estimation), 1.5 ATR SL/TP
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("D:/trading_pythonbacktest_data")
TIMEBAR_CACHE_DIR = DATA_DIR / "timebars_5min"

# Rough vol parameters (from Hurst estimation)
H = 0.57          # NQ Hurst exponent (from estimate_hurst_exponent.py)
ETA = 1.5         # vol of vol
V0 = 0.04         # baseline variance
KERNEL_LEN = 80   # rough driver memory
NORM_LEN = 100    # return normalization lookback
Z_LOOKBACK = 100  # vol z-score lookback
HIGH_Z = 1.1      # vol regime threshold (z-score)

# Entry parameters

EMA_SLOW = 200    # slow EMA
ATR_LEN = 14      # ATR period
ATR_SL = 2.0      # stop loss multiplier (5x ATR - emergency only)
ATR_TP = 5.0      # take profit multiplier (5x ATR - emergency only)

# Session / risk
SESSION_START = "04:15"  # 6am ET start
SESSION_END = "15:30"    # 4:55pm ET end (before close)
MAX_TRADES = 999         # no limit on trades per day
MAX_LOSS = -999999.0     # no max daily loss
POINT_VALUE = 20.0       # NQ = $20/point
TICK_SIZE = 0.25

ET = "America/New_York"


# ── STEP 1: LOAD 5-MIN BARS ───────────────────────────────────────────────────

def load_5min_bars() -> pd.DataFrame:
    """Load existing 5-min time bars from cache."""
    timebar_files = sorted(TIMEBAR_CACHE_DIR.glob("*.pkl"))

    if not timebar_files:
        raise FileNotFoundError(f"No 5-min bars in {TIMEBAR_CACHE_DIR}")

    print(f"Loading {len(timebar_files)} 5-min bar cache files...")
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

    # Convert to ET for session filtering
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(ET)

    print(f"  Loaded {len(df):,} 5-min bars from {df.index[0]} to {df.index[-1]}")
    return df


# ── STEP 2: COMPUTE ROUGH VOL SIGNAL ──────────────────────────────────────────

def compute_atr(high, low, close, period=14):
    """ATR for stops/targets."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rough_vol_regime(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Add rough volatility regime filter to bars.

    Returns bars with:
      - v_model: rough variance estimate
      - z_vol: vol z-score
      - high_vol: bool, True when z_vol > HIGH_Z (regime filter)
    """
    df = bars.copy()
    closes = df["close"]

    # 1. Log returns + standardized shocks
    ret = np.log(closes / closes.shift(1)).fillna(0)
    mean_ret = ret.rolling(NORM_LEN).mean()
    std_ret = ret.rolling(NORM_LEN).std()
    shock = ((ret - mean_ret) / std_ret.replace(0, np.nan)).fillna(0)

    # 2. Power-law kernel weights (fractional Brownian motion)
    kernel = np.array([
        (k + 1) ** (H - 0.5) - k ** (H - 0.5)
        for k in range(KERNEL_LEN)
    ])

    # 3. Rough driver xH = weighted sum of lagged shocks
    shock_arr = shock.values
    n = len(shock_arr)
    xH = np.zeros(n)

    for i in range(KERNEL_LEN, n):
        for k in range(KERNEL_LEN):
            xH[i] += kernel[k] * shock_arr[i - k]

    df["xH"] = xH

    # 4. Rough variance process: v(t) = V0 * exp(ETA * xH)
    df["v_model"] = V0 * np.exp(ETA * df["xH"])
    df["v_model"] = df["v_model"].clip(upper=V0 * 1e4)  # prevent explosion

    # 5. Vol z-score (regime detection)
    v_mean = df["v_model"].rolling(Z_LOOKBACK).mean()
    v_std = df["v_model"].rolling(Z_LOOKBACK).std()
    df["z_vol"] = (df["v_model"] - v_mean) / v_std.replace(0, np.nan)

    # 6. High vol regime flag
    df["high_vol"] = df["z_vol"] > HIGH_Z

    return df


# ── STEP 3: ADD EMA ENTRY LOGIC ───────────────────────────────────────────────

def add_ema_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple EMA trend filter + rough vol regime filter.

    Entry rules:
      - LONG: price > EMA AND high_vol regime
      - SHORT: price < EMA AND high_vol regime
    """
    closes = df["close"]

    # Single EMA (trend filter)
    df["ema"] = closes.ewm(span=EMA_SLOW, adjust=False).mean()

    # ATR
    df["atr"] = compute_atr(df["high"], df["low"], closes, ATR_LEN)

    # 7. Entry signals
    is_high_vol        = df["z_vol"] > HIGH_Z
    df["long_signal"]  = is_high_vol & (closes > df["ema"])
    df["short_signal"] = is_high_vol & (closes < df["ema"])

    # 8. SL / TP prices
    df["long_sl"]  = closes - ATR_SL * df["atr"]
    df["long_tp"]  = closes + ATR_TP * df["atr"]
    df["short_sl"] = closes + ATR_SL * df["atr"]
    df["short_tp"] = closes - ATR_TP * df["atr"]

    return df


# ── STEP 4: BACKTEST ──────────────────────────────────────────────────────────

def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Run backtest with session/risk filters."""
    trades = []
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    entry_time = None

    daily_pnl = 0.0
    daily_trades = 0
    current_date = None

    for ts, row in df.iterrows():
        # Reset daily counters
        bar_date = ts.date()
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl = 0.0
            daily_trades = 0

        # Session filter (RTH only)
        bar_time_str = ts.strftime("%H:%M")
        in_session = SESSION_START <= bar_time_str <= SESSION_END

        close = row["close"]
        high = row["high"]
        low = row["low"]

        # ── Manage open position ──────────────────────────────────────────────
        if position == 1:  # long
            # Check emergency exits (5x ATR)
            hit_sl = low <= sl_price
            hit_tp = high >= tp_price

            # Check regime exit (primary exit)
            regime_switched = not row["high_vol"]

            if hit_sl or hit_tp or regime_switched:
                if regime_switched:
                    exit_price = close
                    exit_reason = "REGIME"
                else:
                    exit_price = sl_price if hit_sl else tp_price
                    exit_reason = "SL" if hit_sl else "TP"

                pnl = (exit_price - entry_price) * POINT_VALUE
                daily_pnl += pnl
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "direction": "LONG",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": round(pnl, 2),
                    "exit_reason": exit_reason,
                })
                position = 0

        elif position == -1:  # short
            # Check emergency exits (5x ATR)
            hit_sl = high >= sl_price
            hit_tp = low <= tp_price

            # Check regime exit (primary exit)
            regime_switched = not row["high_vol"]

            if hit_sl or hit_tp or regime_switched:
                if regime_switched:
                    exit_price = close
                    exit_reason = "REGIME"
                else:
                    exit_price = sl_price if hit_sl else tp_price
                    exit_reason = "SL" if hit_sl else "TP"

                pnl = (entry_price - exit_price) * POINT_VALUE
                daily_pnl += pnl
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "direction": "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": round(pnl, 2),
                    "exit_reason": exit_reason,
                })
                position = 0

        # ── Check for entry ───────────────────────────────────────────────────
        can_enter = (
            position == 0
            and in_session
            and daily_pnl > MAX_LOSS
            and daily_trades < MAX_TRADES
            and not pd.isna(row.get("z_vol"))
            and not pd.isna(row.get("atr"))
        )

        if can_enter:
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


# ── STEP 5: PERFORMANCE REPORT ────────────────────────────────────────────────

def report(trades: pd.DataFrame):
    """Print performance metrics."""
    if trades.empty:
        print("\nNo trades generated.")
        print("  Possible reasons:")
        print("  - Vol regime filter too strict (try lowering HIGH_Z)")
        print("  - EMA crossovers rare in this period")
        print("  - Session filter too narrow")
        return

    total = len(trades)
    winners = trades[trades["pnl"] > 0]
    losers = trades[trades["pnl"] <= 0]
    win_rate = len(winners) / total * 100
    total_pnl = trades["pnl"].sum()
    avg_win = winners["pnl"].mean() if len(winners) else 0
    avg_loss = losers["pnl"].mean() if len(losers) else 0
    profit_factor = winners["pnl"].sum() / abs(losers["pnl"].sum()) if len(losers) else float("inf")

    # Drawdown
    cumulative = trades["pnl"].cumsum()
    rolling_max = cumulative.cummax()
    drawdown = cumulative - rolling_max
    max_dd = drawdown.min()

    # By exit reason
    regime_trades = trades[trades["exit_reason"] == "REGIME"]
    sl_trades = trades[trades["exit_reason"] == "SL"]
    tp_trades = trades[trades["exit_reason"] == "TP"]

    regime_winners = regime_trades[regime_trades["pnl"] > 0]
    regime_win_rate = len(regime_winners) / len(regime_trades) * 100 if len(regime_trades) > 0 else 0

    print("\n" + "="*80)
    print("ROUGH VOL + EMA BACKTEST RESULTS (REGIME EXIT)")
    print("="*80)
    print(f"Config: H={H}, ETA={ETA}, HIGH_Z={HIGH_Z}")
    print(f"Entry: Price > EMA({EMA_SLOW}) + high vol regime (z_vol > {HIGH_Z})")
    print(f"Primary Exit: Regime switch (high vol -> not high vol)")
    print(f"Emergency Exit: {ATR_SL}x ATR SL / {ATR_TP}x ATR TP")
    print(f"Session: {SESSION_START}-{SESSION_END} ET")
    print("="*80)
    print(f"Total trades:      {total}")
    print(f"Win rate:          {win_rate:.1f}%")
    print(f"Total P&L:         ${total_pnl:,.2f}")
    print(f"Avg winner:        ${avg_win:,.2f}")
    print(f"Avg loser:         ${avg_loss:,.2f}")
    print(f"Profit factor:     {profit_factor:.2f}")
    print(f"Max drawdown:      ${max_dd:,.2f}")
    print()
    print(f"EXIT BREAKDOWN:")
    print(f"  REGIME exits:    {len(regime_trades)} ({len(regime_trades)/total*100:.1f}%) - Win rate: {regime_win_rate:.1f}%")
    print(f"  TP exits:        {len(tp_trades)} ({len(tp_trades)/total*100:.1f}%)")
    print(f"  SL exits:        {len(sl_trades)} ({len(sl_trades)/total*100:.1f}%)")
    print("="*80)

    # Save trades
    out = Path("rough_vol_ema_trades.csv")
    trades.to_csv(out, index=False)
    print(f"\nTrades saved to {out}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*80)
    print("TESTING ROUGH VOL REGIME FILTER WITH EMA ENTRIES")
    print("="*80)
    print()

    print("Step 1: Loading 5-min bars from cache...")
    bars = load_5min_bars()
    print()

    print("Step 2: Computing rough vol regime (H=0.57)...")
    bars_with_vol = compute_rough_vol_regime(bars)
    print(f"  High vol bars: {bars_with_vol['high_vol'].sum():,} / {len(bars_with_vol):,} ({bars_with_vol['high_vol'].mean()*100:.1f}%)")
    print()

    print("Step 3: Adding EMA entry signals...")
    signal_df = add_ema_signals(bars_with_vol)
    long_sigs = signal_df["long_signal"].sum()
    short_sigs = signal_df["short_signal"].sum()
    print(f"  Long signals: {long_sigs:,}")
    print(f"  Short signals: {short_sigs:,}")
    print()

    print("Step 4: Running backtest...")
    trade_log = backtest(signal_df)
    print()

    print("Step 5: Performance report...")
    report(trade_log)
    print()

    print("Next: Test with your orderflow logic instead of EMA")
    print("  - Modify signal logic to use VAL/VAH absorption entries")
    print("  - Keep rough vol as regime filter (only trade in high_vol)")
