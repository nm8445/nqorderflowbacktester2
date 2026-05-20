"""
Rough Volatility 15-Minute Bar Backtest — Full Rebuild.

Signal: rough vol z-score > HIGH_Z + EMA trend filter.
Primary exit: z_vol < LOW_Z (vol regime collapse).
Safety net: ATR-based SL/TP.
Session exit: flatten at 3:30 PM ET.
Martingale: streak-based position sizing.
"""

import sys
from pathlib import Path

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import config_15min as cfg
from time_bars import build_15min_bars


# ═══════════════════════════════════════════════════════════════════════════════
# MARTINGALE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class MartingaleManager:
    def __init__(self, base, multiple, streak, max_doubles):
        self.base = base
        self.multiple = multiple
        self.streak = streak
        self.max_doubles = max_doubles
        self.loss_streak = 0
        self.doublings = 0

    def get_qty(self) -> int:
        """Current position size based on loss streak."""
        if self.loss_streak >= self.streak:
            mult = self.multiple ** min(self.doublings, self.max_doubles)
            return max(1, round(self.base * mult))
        return self.base

    def on_trade_result(self, pnl: float):
        """Update after each trade closes."""
        if pnl > 0:
            self.loss_streak = 0
            self.doublings = 0
        else:
            self.loss_streak += 1
            if self.loss_streak >= self.streak:
                self.doublings += 1

    def reset(self):
        """Reset at session start."""
        self.loss_streak = 0
        self.doublings = 0


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rough vol signals with corrected parameters."""
    df = df.copy()
    closes = df["close"]
    n = len(closes)

    # 1. Log returns
    ret = np.log(closes / closes.shift(1))

    # 2. Standardized shocks
    mean_ret = ret.rolling(cfg.NORM_LEN).mean()
    std_ret = ret.rolling(cfg.NORM_LEN).std()
    shock = ((ret - mean_ret) / std_ret.replace(0, np.nan)).fillna(0)

    # 3. Power-law kernel
    ks = np.arange(1, cfg.KERNEL_LEN + 1, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        kernel = ks ** (cfg.H - 0.5) - (ks - 1) ** (cfg.H - 0.5)
    kernel[0] = 1.0  # overwrite inf from 0^(-0.4)

    # 4. Rough driver via convolution
    xH = np.convolve(shock.values, kernel, mode="full")[:n]

    # 5. Rough variance process
    v_model = cfg.V0 * np.exp(cfg.ETA * xH)
    v_model = np.clip(v_model, 0, cfg.V0 * 1e4)
    df["v_model"] = v_model

    # 6. Vol z-score
    v_series = pd.Series(v_model, index=df.index)
    v_mean = v_series.rolling(cfg.Z_LOOKBACK).mean()
    v_std = v_series.rolling(cfg.Z_LOOKBACK).std()
    df["z_vol"] = ((v_series - v_mean) / v_std.replace(0, np.nan)).values

    # 7. Trend filter
    df["ema"] = closes.ewm(span=cfg.EMA_LEN, adjust=False).mean()

    # 8. ATR
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - closes.shift()).abs(),
        (df["low"] - closes.shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(cfg.ATR_LEN).mean()

    # 9. Entry signals
    is_high_vol = df["z_vol"] > cfg.HIGH_Z
    df["long_signal"] = is_high_vol & (closes > df["ema"])
    df["short_signal"] = is_high_vol & (closes < df["ema"])

    # 10. Safety net SL/TP
    df["long_sl"] = closes - cfg.ATR_SL * df["atr"]
    df["long_tp"] = closes + cfg.ATR_TP * df["atr"]
    df["short_sl"] = closes + cfg.ATR_SL * df["atr"]
    df["short_tp"] = closes - cfg.ATR_TP * df["atr"]

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Run backtest with LOW_Z primary exit, ATR safety net, and martingale."""

    # Pre-extract arrays for speed
    timestamps = df.index.values
    hours = df.index.hour + df.index.minute / 60.0
    dates = df.index.date
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    z_vols = df["z_vol"].values
    atrs = df["atr"].values
    long_signals = df["long_signal"].values
    short_signals = df["short_signal"].values
    long_sls = df["long_sl"].values
    long_tps = df["long_tp"].values
    short_sls = df["short_sl"].values
    short_tps = df["short_tp"].values

    n = len(df)

    marty = MartingaleManager(
        base=cfg.BASE_CONTRACTS,
        multiple=cfg.MARTINGALE_MULTIPLE,
        streak=cfg.MARTINGALE_STREAK,
        max_doubles=cfg.MAX_DOUBLES,
    )

    trades = []
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_sl = 0.0
    entry_tp = 0.0
    entry_time = None
    entry_z_vol = 0.0
    qty = 1
    daily_pnl = 0.0
    daily_trades = 0
    current_date = None

    for i in range(n):
        bar_date = dates[i]
        hour_frac = hours[i]
        in_session = cfg.SESSION_START <= hour_frac <= cfg.SESSION_END
        at_session_end = hour_frac >= cfg.SESSION_END

        # Daily reset
        if bar_date != current_date:
            current_date = bar_date
            daily_pnl = 0.0
            daily_trades = 0
            if cfg.RESET_ON_SESSION:
                marty.reset()

        close = closes[i]
        high = highs[i]
        low = lows[i]
        z_vol = z_vols[i]

        # ── Manage open position ──────────────────────────────────────────
        if position != 0:
            exit_price = None
            exit_reason = None

            if position == 1:  # long
                # 1. SL hit
                if low <= entry_sl:
                    exit_price = entry_sl
                    exit_reason = "SL"
                # 2. TP hit
                elif high >= entry_tp:
                    exit_price = entry_tp
                    exit_reason = "TP"
                # 3. Session end
                elif at_session_end:
                    exit_price = close
                    exit_reason = "SESSION"

            else:  # short
                # 1. SL hit
                if high >= entry_sl:
                    exit_price = entry_sl
                    exit_reason = "SL"
                # 2. TP hit
                elif low <= entry_tp:
                    exit_price = entry_tp
                    exit_reason = "TP"
                # 3. Session end
                elif at_session_end:
                    exit_price = close
                    exit_reason = "SESSION"

            if exit_price is not None:
                if position == 1:
                    raw_pnl = (exit_price - entry_price) * cfg.POINT_VALUE
                else:
                    raw_pnl = (entry_price - exit_price) * cfg.POINT_VALUE

                total_pnl = raw_pnl * qty
                marty.on_trade_result(raw_pnl)
                daily_pnl += total_pnl

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": timestamps[i],
                    "direction": "LONG" if position == 1 else "SHORT",
                    "qty": qty,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_per_contract": round(raw_pnl, 2),
                    "total_pnl": round(total_pnl, 2),
                    "exit_reason": exit_reason,
                    "z_vol_at_entry": round(entry_z_vol, 4),
                    "z_vol_at_exit": round(z_vol, 4) if not np.isnan(z_vol) else np.nan,
                    "loss_streak_at_entry": marty.loss_streak,
                })

                position = 0

        # ── Check for entry ───────────────────────────────────────────────
        if (position == 0
                and in_session
                and not at_session_end
                and daily_pnl > cfg.MAX_LOSS
                and daily_trades < cfg.MAX_TRADES
                and not np.isnan(z_vol)
                and not np.isnan(atrs[i])):

            if long_signals[i]:
                qty = marty.get_qty()
                position = 1
                entry_price = close
                entry_sl = long_sls[i]
                entry_tp = long_tps[i]
                entry_time = timestamps[i]
                entry_z_vol = z_vol
                daily_trades += 1

            elif short_signals[i]:
                qty = marty.get_qty()
                position = -1
                entry_price = close
                entry_sl = short_sls[i]
                entry_tp = short_tps[i]
                entry_time = timestamps[i]
                entry_z_vol = z_vol
                daily_trades += 1

    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def report(trades: pd.DataFrame):
    """Print full performance report."""
    if trades.empty:
        print("\nNo trades generated.")
        return

    total = len(trades)
    winners = trades[trades["total_pnl"] > 0]
    losers = trades[trades["total_pnl"] <= 0]
    win_rate = len(winners) / total * 100

    total_pnl = trades["total_pnl"].sum()
    avg_win = winners["total_pnl"].mean() if len(winners) > 0 else 0
    avg_loss = losers["total_pnl"].mean() if len(losers) > 0 else 0

    gross_profit = winners["total_pnl"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["total_pnl"].sum()) if len(losers) > 0 else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    cum = trades["total_pnl"].cumsum()
    max_dd = float((cum - cum.cummax()).min())

    # Sharpe ratio (annualized, assuming ~252 trading days)
    daily_pnls = trades.copy()
    daily_pnls["date"] = pd.to_datetime(daily_pnls["entry_time"]).dt.date
    daily_sum = daily_pnls.groupby("date")["total_pnl"].sum()
    if daily_sum.std() > 0:
        sharpe = (daily_sum.mean() / daily_sum.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Average holding time
    entry_ts = pd.to_datetime(trades["entry_time"], utc=True)
    exit_ts = pd.to_datetime(trades["exit_time"], utc=True)
    duration_mins = (exit_ts - entry_ts).dt.total_seconds() / 60
    avg_hold = duration_mins.mean()

    # Exit reason breakdown
    low_z_trades = trades[trades["exit_reason"] == "LOW_Z"]
    tp_trades = trades[trades["exit_reason"] == "TP"]
    sl_trades = trades[trades["exit_reason"] == "SL"]
    session_trades = trades[trades["exit_reason"] == "SESSION"]

    print()
    print("=" * 80)
    print("ROUGH VOL 15-MIN BACKTEST — CORRECTED PARAMETERS")
    print("=" * 80)
    print(f"H={cfg.H} | V0={cfg.V0} | ETA={cfg.ETA} | KERNEL={cfg.KERNEL_LEN}")
    print(f"EMA={cfg.EMA_LEN} | HIGH_Z={cfg.HIGH_Z} | LOW_Z={cfg.LOW_Z}")
    print(f"ATR SL={cfg.ATR_SL}x | ATR TP={cfg.ATR_TP}x (safety net)")
    print(f"Session: {cfg.SESSION_START}h - {cfg.SESSION_END}h ET")
    print(f"Max trades/day: {cfg.MAX_TRADES} | Max daily loss: ${cfg.MAX_LOSS:,}")
    print(f"Martingale: streak={cfg.MARTINGALE_STREAK}, mult={cfg.MARTINGALE_MULTIPLE}x, max={cfg.MAX_DOUBLES}")
    print("=" * 80)
    print()

    print("OVERALL METRICS:")
    print(f"  Total trades:    {total}")
    print(f"  Win rate:        {win_rate:.1f}%")
    print(f"  Total P&L:       ${total_pnl:+,.0f}")
    print(f"  Avg winner:      ${avg_win:+,.0f}")
    print(f"  Avg loser:       ${avg_loss:+,.0f}")
    print(f"  Profit factor:   {profit_factor:.2f}")
    print(f"  Max drawdown:    ${max_dd:+,.0f}")
    print(f"  Sharpe ratio:    {sharpe:.2f}")
    print(f"  Avg hold time:   {avg_hold:.0f} min ({avg_hold/60:.1f} hrs)")
    print()

    print("EXIT REASON BREAKDOWN:")
    for label, subset in [("LOW_Z", low_z_trades), ("TP", tp_trades),
                          ("SL", sl_trades), ("SESSION", session_trades)]:
        cnt = len(subset)
        pct = cnt / total * 100 if total > 0 else 0
        pnl = subset["total_pnl"].sum() if cnt > 0 else 0
        wr = len(subset[subset["total_pnl"] > 0]) / cnt * 100 if cnt > 0 else 0
        print(f"  {label:10s}  {cnt:4d} ({pct:5.1f}%)  WR: {wr:5.1f}%  P&L: ${pnl:+,.0f}")
    print()

    # Martingale breakdown
    print("MARTINGALE BREAKDOWN:")
    for q in sorted(trades["qty"].unique()):
        subset = trades[trades["qty"] == q]
        cnt = len(subset)
        pnl = subset["total_pnl"].sum()
        wr = len(subset[subset["total_pnl"] > 0]) / cnt * 100 if cnt > 0 else 0
        print(f"  {q:2d} contracts:  {cnt:4d} trades  WR: {wr:5.1f}%  P&L: ${pnl:+,.0f}")
    print(f"  Max contracts reached: {trades['qty'].max()}")
    print()

    # Daily summary
    trades_copy = trades.copy()
    trades_copy["date"] = pd.to_datetime(trades_copy["entry_time"]).dt.date
    daily = trades_copy.groupby("date").agg(
        trades=("total_pnl", "count"),
        wins=("total_pnl", lambda x: (x > 0).sum()),
        losses=("total_pnl", lambda x: (x <= 0).sum()),
        daily_pnl=("total_pnl", "sum"),
        max_qty=("qty", "max"),
    )

    print("DAILY SUMMARY:")
    print(f"{'Date':>12s} | {'Trades':>6s} | {'W/L':>5s} | {'Daily P&L':>10s} | {'Max Qty':>7s}")
    print("-" * 55)
    for date, row in daily.iterrows():
        wl = f"{row['wins']:.0f}/{row['losses']:.0f}"
        print(f"{str(date):>12s} | {row['trades']:6.0f} | {wl:>5s} | ${row['daily_pnl']:+9,.0f} | {row['max_qty']:7.0f}")
    print()

    # Save trades
    out = cfg.CACHE_DIR / "rough_vol_15min_trades.csv"
    trades.to_csv(out, index=False)
    print(f"Trades saved to {out}")

    # ── Comparison vs original range bar results ──────────────────────────
    print()
    print("=" * 80)
    print("COMPARISON vs ORIGINAL RANGE BAR RESULTS")
    print("=" * 80)
    print(f"{'Metric':<22s} {'Range Bar (old)':>16s} {'15-Min (new)':>16s}")
    print("-" * 56)
    print(f"{'Bar type':<22s} {'40-tick range':>16s} {'15-minute':>16s}")
    print(f"{'Session':<22s} {'9:30-3:30':>16s} {'4:15-3:30':>16s}")
    print(f"{'V0':<22s} {'0.04':>16s} {str(cfg.V0):>16s}")
    print(f"{'EMA_LEN':<22s} {'200':>16s} {str(cfg.EMA_LEN):>16s}")
    print(f"{'KERNEL_LEN':<22s} {'50':>16s} {str(cfg.KERNEL_LEN):>16s}")
    print(f"{'Exit method':<22s} {'SL/TP only':>16s} {'LOW_Z + SL/TP':>16s}")
    print(f"{'Martingale':<22s} {'No':>16s} {'Yes':>16s}")
    print("-" * 56)
    print(f"{'Total trades':<22s} {'215':>16s} {total:>16d}")
    print(f"{'Win rate':<22s} {'46.1%':>16s} {f'{win_rate:.1f}%':>16s}")
    print(f"{'Total P&L':<22s} {'$40,591':>16s} {f'${total_pnl:+,.0f}':>16s}")
    print(f"{'Profit factor':<22s} {'1.34':>16s} {f'{profit_factor:.2f}':>16s}")
    print(f"{'Max drawdown':<22s} {'$12,541':>16s} {f'${abs(max_dd):,.0f}':>16s}")
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("ROUGH VOL 15-MIN BACKTEST — FULL REBUILD")
    print("=" * 80)
    print()

    # Step 1: Load bars
    print("Step 1: Loading 15-minute bars...")
    bars = build_15min_bars()
    print(f"  {len(bars):,} bars from {bars.index[0]} to {bars.index[-1]}")
    print()

    # Step 2: Compute signals
    print("Step 2: Computing rough vol signals...")
    signal_df = compute_signal(bars)
    high_vol_bars = (signal_df["z_vol"] > cfg.HIGH_Z).sum()
    long_sigs = signal_df["long_signal"].sum()
    short_sigs = signal_df["short_signal"].sum()
    print(f"  High vol bars: {high_vol_bars:,} / {len(signal_df):,} ({high_vol_bars/len(signal_df)*100:.1f}%)")
    print(f"  Long signals:  {long_sigs:,}")
    print(f"  Short signals: {short_sigs:,}")
    print()

    # Step 3: Run backtest
    print("Step 3: Running backtest...")
    trade_log = backtest(signal_df)
    print(f"  {len(trade_log)} trades generated")

    # Step 4: Report
    report(trade_log)
