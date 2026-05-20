"""
Noise-band intraday momentum strategy backtest.

Based on Zarattini/Aziz/Barbon 2024 ("Beat the Market") with Quantitativo
modifications for NQ futures (90-day lookback, 3% vol target, 8x leverage cap).

Runs both Quantitativo and original-paper configs side-by-side.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

# ─── paths ───────────────────────────────────────────────────────────────────
ONEMN_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")
SIGMA_CACHE_DIR = Path("D:/trading_pythonbacktest_data/noise_band_sigma_cache")
RESULTS_DIR = Path("C:/trading/nqorderflowbacktester/results/noise_band")

ET = "America/New_York"
NQ_POINT_VALUE = 20.0    # $20 per index point
NQ_TICK_SIZE = 0.25       # 0.25 index points per tick
NQ_TICK_VALUE = 5.0       # $5 per tick

# Transaction costs per contract per side
COMMISSION_PER_SIDE = 0.85
EXCHANGE_FEE_PER_SIDE = 1.40
SLIPPAGE_TICKS_PER_SIDE = 0.25  # in ticks (0.25 * $5 = $1.25 per side)


@dataclass
class Config:
    """Strategy configuration — all tuneable parameters in one place."""
    name: str = "quantitativo"
    lookback_days: int = 90
    vol_target: float = 0.03       # daily portfolio vol target
    leverage_cap: float = 8.0
    vol_window: int = 14           # days for realized vol
    session_start: str = "09:30"
    session_end: str = "16:45"      # include post-close for exits
    first_entry_time: str = "10:00"
    last_entry_time: str = "15:30"  # no new entries after this
    entry_interval_min: int = 30    # check every 30 minutes
    force_close_time: str = "16:45"
    starting_equity: float = 100_000.0
    max_trades_per_day: int = 99    # effectively unlimited


@dataclass
class Trade:
    """Single completed trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str              # "long" or "short"
    entry_price: float
    exit_price: float
    contracts: int
    bars_held: int
    gross_pnl: float
    net_pnl: float
    exit_reason: str            # "vwap_stop", "band_stop", "eod_close"
    signal_time: pd.Timestamp   # the half-hour mark that triggered entry


def load_1min_bars() -> pd.DataFrame:
    """Load 1-min bars, convert to ET, return OHLCV DataFrame."""
    if not ONEMN_PARQUET.exists():
        raise FileNotFoundError(
            f"{ONEMN_PARQUET} not found. Run build_1min_from_markettick.py first."
        )
    df = pd.read_parquet(ONEMN_PARQUET)
    # Convert index to ET
    new_idx = []
    for t in df.index:
        if hasattr(t, "tz_convert") and t.tzinfo:
            new_idx.append(t.tz_convert(ET))
        else:
            new_idx.append(pd.Timestamp(t).tz_localize("UTC").tz_convert(ET))
    df.index = pd.DatetimeIndex(new_idx)
    return df


def get_session_bars(df: pd.DataFrame, sess_start: str, sess_end: str) -> pd.DataFrame:
    """Filter to RTH session bars only."""
    hm = df.index.strftime("%H:%M")
    mask = (hm >= sess_start) & (hm <= sess_end)
    return df[mask].copy()


def compute_daily_ohlc(session_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily OHLCV from session 1-min bars."""
    session_df = session_df.copy()
    session_df["date"] = session_df.index.date
    daily = session_df.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    daily.index = pd.DatetimeIndex([pd.Timestamp(d) for d in daily.index])
    return daily


def precompute_daily_returns(
    session_df: pd.DataFrame,
    trading_dates: list,
    bars_by_date: dict,
) -> dict:
    """
    Pre-compute per-day, per-minute returns once upfront.

    Returns dict: date -> list of (hm_str, return_float) tuples.
    """
    daily_rets = {}
    for d in trading_dates:
        day_bars = bars_by_date.get(d)
        if day_bars is None or len(day_bars) < 10:
            continue
        day_open = day_bars.iloc[0]["open"]
        if day_open <= 0:
            continue
        closes = day_bars["close"].values
        hms = day_bars.index.strftime("%H:%M")
        rets = (closes - day_open) / day_open
        daily_rets[d] = list(zip(hms, rets))
    return daily_rets


def compute_sigma_vector(
    daily_rets: dict,
    lookback_days: int,
    trading_dates: list,
    today_idx: int,
) -> dict[str, float]:
    """
    Compute σ(t) for each minute-of-day from historical data.

    For each of the last `lookback_days` trading days, compute:
        r(d, t) = (price_at_minute_t_on_day_d - open_on_day_d) / open_on_day_d
    Then σ(t) = std(r(:, t)) across those days (population std).

    Returns dict mapping "HH:MM" -> σ value.
    """
    start_idx = max(0, today_idx - lookback_days)
    history_dates = trading_dates[start_idx:today_idx]
    if len(history_dates) < max(10, lookback_days // 2):
        return {}

    minute_returns: dict[str, list[float]] = {}
    for d in history_dates:
        day_data = daily_rets.get(d)
        if day_data is None:
            continue
        for hm, r in day_data:
            if hm not in minute_returns:
                minute_returns[hm] = []
            minute_returns[hm].append(r)

    sigma = {}
    for hm, returns in minute_returns.items():
        arr = np.array(returns)
        if len(arr) < 10:
            continue
        mu = arr.mean()
        sigma[hm] = np.sqrt(np.mean((arr - mu) ** 2))

    return sigma


def build_sigma_cache(
    session_df: pd.DataFrame,
    bars_by_date: dict,
    trading_dates: list,
    lookback_days: int,
) -> dict:
    """
    Compute sigma vectors for ALL trading days and cache to disk.

    Returns dict: date -> {hm: sigma_value}.
    On subsequent runs with the same lookback, loads from pickle.
    """
    import pickle

    SIGMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = SIGMA_CACHE_DIR / f"sigma_lookback_{lookback_days}.pkl"

    if cache_file.exists():
        print(f"  Loading cached sigma vectors from {cache_file.name}...", flush=True)
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        # Check if cache covers our date range
        cached_dates = set(cached.keys())
        needed_dates = set(trading_dates[lookback_days:])
        missing = needed_dates - cached_dates
        if not missing:
            print(f"  Cache hit: {len(cached)} days", flush=True)
            return cached
        print(f"  Cache partial: {len(missing)} days missing, rebuilding...", flush=True)

    print(f"  Computing sigma vectors (lookback={lookback_days}) for {len(trading_dates)} days...", flush=True)
    daily_rets = precompute_daily_returns(session_df, trading_dates, bars_by_date)

    all_sigma = {}
    for day_num, today in enumerate(trading_dates):
        if day_num < lookback_days:
            continue
        sigma = compute_sigma_vector(daily_rets, lookback_days, trading_dates, day_num)
        if sigma:
            all_sigma[today] = sigma

        if (day_num + 1) % 200 == 0:
            print(f"    sigma: {day_num+1}/{len(trading_dates)}", flush=True)

    with open(cache_file, "wb") as f:
        pickle.dump(all_sigma, f)
    print(f"  Cached {len(all_sigma)} sigma vectors to {cache_file.name}", flush=True)

    return all_sigma


def compute_entry_times(first_entry: str, last_entry: str, interval_min: int) -> list[str]:
    """Generate the list of half-hour entry check times as HH:MM strings."""
    times = []
    h, m = map(int, first_entry.split(":"))
    end_h, end_m = map(int, last_entry.split(":"))
    while (h, m) <= (end_h, end_m):
        times.append(f"{h:02d}:{m:02d}")
        m += interval_min
        if m >= 60:
            h += m // 60
            m = m % 60
    return times


def compute_realized_vol(daily_df: pd.DataFrame, today_idx: int, window: int) -> float:
    """Annualized daily realized vol from last `window` daily log returns."""
    if today_idx < window + 1:
        return 0.15  # default fallback
    start = max(0, today_idx - window)
    subset = daily_df.iloc[start:today_idx]
    log_rets = np.log(subset["close"] / subset["close"].shift(1)).dropna()
    if len(log_rets) < 5:
        return 0.15
    return log_rets.std() * np.sqrt(252)


def compute_position_size(
    equity: float,
    vol_target: float,
    realized_vol: float,
    leverage_cap: float,
    current_price: float,
) -> int:
    """
    Position size in contracts.

    size = (equity * vol_target) / (realized_vol * point_value * price)
    Capped so notional <= leverage_cap * equity.
    """
    if realized_vol <= 0 or current_price <= 0:
        return 1
    # Daily vol target approach
    daily_realized = realized_vol / np.sqrt(252)
    contracts_vol = (equity * vol_target) / (daily_realized * current_price * NQ_POINT_VALUE)

    # Leverage cap
    notional_per_contract = current_price * NQ_POINT_VALUE
    max_contracts_lev = (equity * leverage_cap) / notional_per_contract

    contracts = int(min(contracts_vol, max_contracts_lev))
    return max(1, contracts)


def transaction_cost(contracts: int) -> float:
    """Round-trip transaction cost for given number of contracts."""
    per_contract = (COMMISSION_PER_SIDE + EXCHANGE_FEE_PER_SIDE + SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_VALUE) * 2
    return contracts * per_contract


def run_backtest(session_df: pd.DataFrame, daily_df: pd.DataFrame, cfg: Config) -> list[Trade]:
    """
    Run the noise-band backtest.

    Args:
        session_df: 1-min OHLCV bars filtered to session hours, sorted by time.
        daily_df: Daily OHLCV for position sizing vol calc.
        cfg: Strategy configuration.

    Returns:
        List of completed Trade objects.
    """
    trading_dates = sorted(set(session_df.index.date))
    daily_dates = sorted(set(daily_df.index.date))
    daily_date_to_idx = {d: i for i, d in enumerate(daily_dates)}
    entry_times = set(compute_entry_times(cfg.first_entry_time, cfg.last_entry_time, cfg.entry_interval_min))

    trades: list[Trade] = []
    equity = cfg.starting_equity

    # Pre-index session bars by date for fast lookup
    bars_by_date: dict = {}
    for d in trading_dates:
        mask = session_df.index.date == d
        bars_by_date[d] = session_df[mask]

    # Build or load cached sigma vectors
    sigma_cache = build_sigma_cache(session_df, bars_by_date, trading_dates, cfg.lookback_days)

    print(f"[{cfg.name}] {len(trading_dates)} trading days, "
          f"lookback={cfg.lookback_days}, vol_target={cfg.vol_target}, "
          f"lev_cap={cfg.leverage_cap}", flush=True)

    for day_num, today in enumerate(trading_dates):
        if day_num < cfg.lookback_days:
            continue  # warm-up period

        today_bars = bars_by_date[today]
        if len(today_bars) < 30:
            continue  # skip thin days

        today_open = today_bars.iloc[0]["open"]
        # Yesterday's close: last bar of previous trading day
        prev_day_idx = day_num - 1
        while prev_day_idx >= 0 and trading_dates[prev_day_idx] not in bars_by_date:
            prev_day_idx -= 1
        if prev_day_idx < 0:
            continue
        prev_day = trading_dates[prev_day_idx]
        yesterday_close = bars_by_date[prev_day].iloc[-1]["close"]

        # Lookup cached sigma vector
        sigma = sigma_cache.get(today)
        if not sigma:
            continue

        # Band anchors
        upper_anchor = max(today_open, yesterday_close)
        lower_anchor = min(today_open, yesterday_close)
        last_sigma = max(sigma.values()) if sigma else 0.001

        # Realized vol for position sizing
        daily_idx = daily_date_to_idx.get(today)
        if daily_idx is None:
            continue
        realized_vol = compute_realized_vol(daily_df, daily_idx, cfg.vol_window)

        # Intraday state
        in_position = False
        direction = ""
        entry_price = 0.0
        entry_time = None
        signal_time = None
        contracts = 0
        trailing_stop = 0.0
        bars_held = 0
        trades_today = 0

        # VWAP accumulators
        vwap_cum_pv = 0.0  # cumulative price*volume
        vwap_cum_v = 0.0   # cumulative volume

        for bar_i in range(len(today_bars)):
            bar = today_bars.iloc[bar_i]
            bar_ts = today_bars.index[bar_i]
            hm = bar_ts.strftime("%H:%M")

            # Update VWAP
            typical_price = (bar["high"] + bar["low"] + bar["close"]) / 3.0
            bar_vol = bar["volume"] if bar["volume"] > 0 else 1
            vwap_cum_pv += typical_price * bar_vol
            vwap_cum_v += bar_vol
            vwap = vwap_cum_pv / vwap_cum_v if vwap_cum_v > 0 else bar["close"]

            # Get current sigma and bands
            sig = sigma.get(hm)
            if sig is None or sig <= 0:
                sig = last_sigma  # fallback to last known sigma
            upper_band = upper_anchor * (1.0 + sig)
            lower_band = lower_anchor * (1.0 - sig)

            # ── EXIT LOGIC (checked every bar) ──
            if in_position:
                bars_held += 1

                # Force close at end of day
                if hm >= cfg.force_close_time:
                    exit_price = bar["close"]
                    # Apply slippage unfavorably
                    if direction == "long":
                        exit_price -= SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        gross = (exit_price - entry_price) * NQ_POINT_VALUE * contracts
                    else:
                        exit_price += SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        gross = (entry_price - exit_price) * NQ_POINT_VALUE * contracts
                    cost = (COMMISSION_PER_SIDE + EXCHANGE_FEE_PER_SIDE) * 2 * contracts
                    trades.append(Trade(
                        entry_time=entry_time, exit_time=bar_ts, direction=direction,
                        entry_price=entry_price, exit_price=exit_price,
                        contracts=contracts, bars_held=bars_held,
                        gross_pnl=round(gross, 2), net_pnl=round(gross - cost, 2),
                        exit_reason="eod_close", signal_time=signal_time,
                    ))
                    equity += gross - cost
                    in_position = False
                    continue

                # Compute current trailing stop
                if direction == "long":
                    # Stop = max(upper_band, vwap) — and can only go UP
                    new_stop = max(upper_band, vwap)
                    trailing_stop = max(trailing_stop, new_stop)
                    # Check if bar's low breaches stop
                    if bar["low"] <= trailing_stop:
                        # Fill at stop price, or open if gap past
                        if bar["open"] <= trailing_stop:
                            exit_price = bar["open"]  # gapped past
                        else:
                            exit_price = trailing_stop
                        # Add slippage (unfavorable)
                        exit_price -= SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        gross = (exit_price - entry_price) * NQ_POINT_VALUE * contracts
                        cost_no_slip = (COMMISSION_PER_SIDE + EXCHANGE_FEE_PER_SIDE) * 2 * contracts
                        # Entry slippage already in entry_price, exit slippage in exit_price
                        # So total cost is just commissions + exchange fees (slippage baked into prices)
                        exit_reason = "vwap_stop" if vwap >= upper_band else "band_stop"
                        trades.append(Trade(
                            entry_time=entry_time, exit_time=bar_ts, direction=direction,
                            entry_price=entry_price, exit_price=exit_price,
                            contracts=contracts, bars_held=bars_held,
                            gross_pnl=round((exit_price - entry_price) * NQ_POINT_VALUE * contracts, 2),
                            net_pnl=round((exit_price - entry_price) * NQ_POINT_VALUE * contracts - cost_no_slip, 2),
                            exit_reason=exit_reason, signal_time=signal_time,
                        ))
                        equity += (exit_price - entry_price) * NQ_POINT_VALUE * contracts - cost_no_slip
                        in_position = False
                        continue

                else:  # short
                    new_stop = min(lower_band, vwap)
                    trailing_stop = min(trailing_stop, new_stop)
                    if bar["high"] >= trailing_stop:
                        if bar["open"] >= trailing_stop:
                            exit_price = bar["open"]
                        else:
                            exit_price = trailing_stop
                        exit_price += SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        cost_no_slip = (COMMISSION_PER_SIDE + EXCHANGE_FEE_PER_SIDE) * 2 * contracts
                        exit_reason = "vwap_stop" if vwap <= lower_band else "band_stop"
                        trades.append(Trade(
                            entry_time=entry_time, exit_time=bar_ts, direction=direction,
                            entry_price=entry_price, exit_price=exit_price,
                            contracts=contracts, bars_held=bars_held,
                            gross_pnl=round((entry_price - exit_price) * NQ_POINT_VALUE * contracts, 2),
                            net_pnl=round((entry_price - exit_price) * NQ_POINT_VALUE * contracts - cost_no_slip, 2),
                            exit_reason=exit_reason, signal_time=signal_time,
                        ))
                        equity += (entry_price - exit_price) * NQ_POINT_VALUE * contracts - cost_no_slip
                        in_position = False
                        continue

            # ── ENTRY LOGIC (only at half-hour marks) ──
            if not in_position and hm in entry_times and trades_today < cfg.max_trades_per_day:
                price = bar["close"]
                if price > upper_band:
                    # Long signal — enter at next bar's open
                    if bar_i + 1 < len(today_bars):
                        next_bar = today_bars.iloc[bar_i + 1]
                        ep = next_bar["open"] + SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        contracts = compute_position_size(
                            equity, cfg.vol_target, realized_vol, cfg.leverage_cap, ep
                        )
                        in_position = True
                        direction = "long"
                        entry_price = ep
                        entry_time = today_bars.index[bar_i + 1]
                        signal_time = bar_ts
                        bars_held = 0
                        trades_today += 1
                        # Initial trailing stop
                        trailing_stop = max(upper_band, vwap)

                elif price < lower_band:
                    # Short signal — enter at next bar's open
                    if bar_i + 1 < len(today_bars):
                        next_bar = today_bars.iloc[bar_i + 1]
                        ep = next_bar["open"] - SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        contracts = compute_position_size(
                            equity, cfg.vol_target, realized_vol, cfg.leverage_cap, ep
                        )
                        in_position = True
                        direction = "short"
                        entry_price = ep
                        entry_time = today_bars.index[bar_i + 1]
                        signal_time = bar_ts
                        bars_held = 0
                        trades_today += 1
                        trailing_stop = min(lower_band, vwap)

        # Safety: if still in position at end of data for this day, force close
        if in_position:
            last_bar = today_bars.iloc[-1]
            exit_price = last_bar["close"]
            if direction == "long":
                exit_price -= SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                gross = (exit_price - entry_price) * NQ_POINT_VALUE * contracts
            else:
                exit_price += SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                gross = (entry_price - exit_price) * NQ_POINT_VALUE * contracts
            cost = (COMMISSION_PER_SIDE + EXCHANGE_FEE_PER_SIDE) * 2 * contracts
            trades.append(Trade(
                entry_time=entry_time, exit_time=today_bars.index[-1], direction=direction,
                entry_price=entry_price, exit_price=exit_price,
                contracts=contracts, bars_held=bars_held,
                gross_pnl=round(gross, 2), net_pnl=round(gross - cost, 2),
                exit_reason="eod_close", signal_time=signal_time,
            ))
            equity += gross - cost
            in_position = False

        # Progress
        if (day_num + 1) % 200 == 0:
            print(f"  Day {day_num+1}/{len(trading_dates)}  trades={len(trades)}  "
                  f"equity=${equity:,.0f}", flush=True)

    print(f"[{cfg.name}] Done: {len(trades)} trades, final equity=${equity:,.0f}", flush=True)
    return trades


def compute_stats(trades: list[Trade], starting_equity: float) -> dict:
    """Compute summary statistics from trade list."""
    if not trades:
        return {"error": "no trades"}

    pnls = np.array([t.net_pnl for t in trades])
    gross_pnls = np.array([t.gross_pnl for t in trades])
    durations = np.array([t.bars_held for t in trades])

    # Equity curve
    equity_curve = starting_equity + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    max_dd = drawdowns.min()

    # Returns per trade in bps (relative to entry notional)
    bps_returns = []
    for t in trades:
        notional = t.entry_price * NQ_POINT_VALUE * t.contracts
        if notional > 0:
            bps_returns.append(t.net_pnl / notional * 10000)
    bps_returns = np.array(bps_returns)

    # Daily PnL for Sharpe/Sortino
    trade_dates = [t.exit_time.date() for t in trades]
    daily_pnl = {}
    for t in trades:
        d = t.exit_time.date()
        daily_pnl[d] = daily_pnl.get(d, 0) + t.net_pnl
    daily_pnl_arr = np.array(list(daily_pnl.values()))

    # Use actual trading days for annualization
    all_dates = sorted(daily_pnl.keys())
    n_years = (all_dates[-1] - all_dates[0]).days / 365.25 if len(all_dates) > 1 else 1
    trading_days_per_year = len(all_dates) / n_years if n_years > 0 else 252

    daily_mean = daily_pnl_arr.mean()
    daily_std = daily_pnl_arr.std(ddof=1) if len(daily_pnl_arr) > 1 else 1
    downside = daily_pnl_arr[daily_pnl_arr < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 1

    total_return = (equity_curve[-1] - starting_equity) / starting_equity
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.1)) - 1
    sharpe = (daily_mean / daily_std) * np.sqrt(trading_days_per_year) if daily_std > 0 else 0
    sortino = (daily_mean / downside_std) * np.sqrt(trading_days_per_year) if downside_std > 0 else 0
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0
    payoff = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return {
        "total_trades": len(trades),
        "trades_per_year": len(trades) / max(n_years, 0.1),
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(ann_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_drawdown_pct": round(drawdowns[drawdowns < 0].mean() * 100, 2) if (drawdowns < 0).any() else 0,
        "calmar": round(calmar, 3),
        "win_rate_pct": round(win_rate * 100, 1),
        "payoff_ratio": round(payoff, 3),
        "avg_return_bps": round(bps_returns.mean(), 2) if len(bps_returns) > 0 else 0,
        "avg_duration_min": round(durations.mean(), 1),
        "total_pnl": round(pnls.sum(), 2),
        "avg_pnl": round(pnls.mean(), 2),
        "avg_win": round(wins.mean(), 2) if len(wins) > 0 else 0,
        "avg_loss": round(losses.mean(), 2) if len(losses) > 0 else 0,
        "profit_factor": round(wins.sum() / abs(losses.sum()), 3) if len(losses) > 0 and losses.sum() != 0 else 0,
        "final_equity": round(starting_equity + pnls.sum(), 2),
        "exit_reasons": exit_reasons,
        "n_years": round(n_years, 2),
    }


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    """Convert trade list to DataFrame for CSV export."""
    rows = []
    for t in trades:
        rows.append({
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "contracts": t.contracts,
            "bars_held": t.bars_held,
            "gross_pnl": t.gross_pnl,
            "net_pnl": t.net_pnl,
            "exit_reason": t.exit_reason,
            "signal_time": t.signal_time,
        })
    return pd.DataFrame(rows)


def build_equity_curve(trades: list[Trade], starting_equity: float) -> pd.Series:
    """Build equity curve indexed by exit_time."""
    if not trades:
        return pd.Series(dtype=float)
    times = [trades[0].entry_time] + [t.exit_time for t in trades]
    values = [starting_equity]
    eq = starting_equity
    for t in trades:
        eq += t.net_pnl
        values.append(eq)
    return pd.Series(values, index=times)


def build_monthly_returns(trades: list[Trade], starting_equity: float) -> pd.DataFrame:
    """Build monthly returns matrix (years x months) from trades."""
    if not trades:
        return pd.DataFrame()

    # Daily PnL
    daily = {}
    for t in trades:
        d = t.exit_time.date()
        daily[d] = daily.get(d, 0) + t.net_pnl

    # Build equity series by date
    dates = sorted(daily.keys())
    eq = starting_equity
    monthly = {}
    for d in dates:
        yr, mo = d.year, d.month
        key = (yr, mo)
        if key not in monthly:
            monthly[key] = {"start_eq": eq, "pnl": 0}
        monthly[key]["pnl"] += daily[d]
        eq += daily[d]

    # Convert to return %
    rows = {}
    for (yr, mo), v in monthly.items():
        if yr not in rows:
            rows[yr] = {}
        ret = v["pnl"] / v["start_eq"] * 100 if v["start_eq"] > 0 else 0
        rows[yr][mo] = round(ret, 2)

    df = pd.DataFrame(rows).T
    df.columns = [f"{m:02d}" for m in df.columns]
    df = df.reindex(columns=[f"{m:02d}" for m in range(1, 13)])
    df.index.name = "Year"
    return df


def generate_html_report(
    results: dict[str, dict],
    trades_dict: dict[str, list[Trade]],
    starting_equity: float,
) -> str:
    """Generate interactive HTML report with equity curves, drawdown, monthly heatmap."""

    # Build data for each config
    configs_data = {}
    for name, tlist in trades_dict.items():
        stats = results[name]
        if not tlist or "error" in stats:
            configs_data[name] = {
                "eq_times": [], "eq_values": [], "dd_pct": [],
                "monthly": {}, "stats": stats,
            }
            continue

        eq_curve = build_equity_curve(tlist, starting_equity)
        monthly = build_monthly_returns(tlist, starting_equity)

        # Drawdown series
        eq_arr = eq_curve.values
        peak = np.maximum.accumulate(eq_arr)
        dd_pct = (eq_arr - peak) / peak * 100

        configs_data[name] = {
            "eq_times": [str(t) for t in eq_curve.index],
            "eq_values": [round(v, 2) for v in eq_curve.values],
            "dd_pct": [round(v, 4) for v in dd_pct],
            "monthly": monthly.to_dict() if not monthly.empty else {},
            "stats": stats,
        }

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Noise Band Strategy — Backtest Results</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0; margin: 20px; }
  h1 { color: #00d4ff; margin-bottom: 5px; }
  h2 { color: #aaa; font-size: 14px; margin-top: 0; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }
  .stats-card { background: #1a1a2e; border-radius: 8px; padding: 16px; border: 1px solid #333; }
  .stats-card h3 { color: #00d4ff; margin: 0 0 12px 0; font-size: 16px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  td, th { padding: 4px 8px; text-align: right; }
  th { color: #888; text-align: left; }
  .positive { color: #00e676; }
  .negative { color: #ff5252; }
  .chart-container { background: #1a1a2e; border-radius: 8px; padding: 10px; margin: 15px 0; border: 1px solid #333; }
  .heatmap-container { background: #1a1a2e; border-radius: 8px; padding: 16px; margin: 15px 0; border: 1px solid #333; }
  .heatmap-container h3 { color: #00d4ff; margin: 0 0 10px 0; }
  .heatmap { border-collapse: collapse; font-size: 12px; }
  .heatmap td, .heatmap th { padding: 6px 10px; text-align: center; border: 1px solid #333; }
  .heatmap th { background: #252540; color: #aaa; }
</style>
</head><body>
<h1>Noise Band Intraday Momentum</h1>
<h2>Zarattini/Aziz/Barbon 2024 + Quantitativo NQ modifications</h2>

<div class="stats-grid">
"""

    # Stats cards
    for name, data in configs_data.items():
        s = data["stats"]
        if "error" in s:
            html += f'<div class="stats-card"><h3>{name.upper()}</h3><p>No trades generated</p></div>'
            continue
        pnl_class = "positive" if s["total_pnl"] > 0 else "negative"
        html += f"""<div class="stats-card">
<h3>{name.upper()}</h3>
<table>
<tr><th>Total Trades</th><td>{s['total_trades']}</td><th>Trades/Year</th><td>{s['trades_per_year']:.0f}</td></tr>
<tr><th>Total Return</th><td class="{pnl_class}">{s['total_return_pct']:+.1f}%</td><th>Ann. Return</th><td class="{pnl_class}">{s['annualized_return_pct']:+.1f}%</td></tr>
<tr><th>Sharpe</th><td>{s['sharpe']:.3f}</td><th>Sortino</th><td>{s['sortino']:.3f}</td></tr>
<tr><th>Max DD</th><td class="negative">{s['max_drawdown_pct']:.1f}%</td><th>Calmar</th><td>{s['calmar']:.3f}</td></tr>
<tr><th>Win Rate</th><td>{s['win_rate_pct']:.1f}%</td><th>Payoff</th><td>{s['payoff_ratio']:.2f}</td></tr>
<tr><th>Profit Factor</th><td>{s['profit_factor']:.3f}</td><th>Avg bps/trade</th><td>{s['avg_return_bps']:+.1f}</td></tr>
<tr><th>Avg Duration</th><td>{s['avg_duration_min']:.0f} min</td><th>Total PnL</th><td class="{pnl_class}">${s['total_pnl']:+,.0f}</td></tr>
<tr><th>Avg Win</th><td class="positive">${s['avg_win']:+,.0f}</td><th>Avg Loss</th><td class="negative">${s['avg_loss']:+,.0f}</td></tr>
</table>
<p style="font-size:11px;color:#666;margin:8px 0 0;">Exit reasons: {', '.join(f'{k}: {v}' for k,v in s['exit_reasons'].items())}</p>
</div>"""

    html += "</div>\n"

    # Equity curve chart
    html += '<div class="chart-container"><div id="equity-chart"></div></div>\n'
    html += '<div class="chart-container"><div id="dd-chart"></div></div>\n'

    # Monthly heatmaps
    for name, data in configs_data.items():
        monthly = data.get("monthly_df")

    html += "<script>\n"

    # Equity traces
    eq_traces = []
    dd_traces = []
    colors = {"quantitativo": "#00d4ff", "original_paper": "#ff9800"}
    for name, data in configs_data.items():
        color = colors.get(name, "#ffffff")
        eq_traces.append(f"""{{
            x: {json.dumps(data['eq_times'])},
            y: {json.dumps(data['eq_values'])},
            name: '{name}',
            type: 'scatter',
            mode: 'lines',
            line: {{color: '{color}', width: 1.5}}
        }}""")
        dd_traces.append(f"""{{
            x: {json.dumps(data['eq_times'])},
            y: {json.dumps(data['dd_pct'])},
            name: '{name}',
            type: 'scatter',
            fill: 'tozeroy',
            line: {{color: '{color}', width: 1}},
            fillcolor: '{color}22'
        }}""")

    html += f"""
Plotly.newPlot('equity-chart', [{','.join(eq_traces)}], {{
    title: {{text: 'Equity Curve', font: {{color: '#aaa', size: 14}}}},
    paper_bgcolor: '#1a1a2e', plot_bgcolor: '#1a1a2e',
    xaxis: {{color: '#666', gridcolor: '#333'}},
    yaxis: {{color: '#666', gridcolor: '#333', tickprefix: '$'}},
    legend: {{font: {{color: '#aaa'}}}},
    margin: {{l: 80, r: 30, t: 40, b: 40}}
}});
Plotly.newPlot('dd-chart', [{','.join(dd_traces)}], {{
    title: {{text: 'Drawdown (%)', font: {{color: '#aaa', size: 14}}}},
    paper_bgcolor: '#1a1a2e', plot_bgcolor: '#1a1a2e',
    xaxis: {{color: '#666', gridcolor: '#333'}},
    yaxis: {{color: '#666', gridcolor: '#333', ticksuffix: '%'}},
    legend: {{font: {{color: '#aaa'}}}},
    margin: {{l: 80, r: 30, t: 40, b: 40}}
}});
"""
    html += "</script>\n"

    # Monthly heatmaps as HTML tables
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for name, tlist in trades_dict.items():
        monthly_df = build_monthly_returns(tlist, starting_equity)
        if monthly_df.empty:
            continue
        html += f'<div class="heatmap-container"><h3>{name.upper()} — Monthly Returns (%)</h3>\n'
        html += '<table class="heatmap"><tr><th>Year</th>'
        for mn in month_names:
            html += f"<th>{mn}</th>"
        html += "<th>Total</th></tr>\n"
        for yr in sorted(monthly_df.index):
            html += f"<tr><th>{yr}</th>"
            yr_total = 0
            for m in range(1, 13):
                col = f"{m:02d}"
                val = monthly_df.loc[yr, col] if col in monthly_df.columns and not pd.isna(monthly_df.loc[yr].get(col)) else None
                if val is not None:
                    yr_total += val
                    if val > 0:
                        bg = f"rgba(0, 230, 118, {min(abs(val)/10, 0.6):.2f})"
                    elif val < 0:
                        bg = f"rgba(255, 82, 82, {min(abs(val)/10, 0.6):.2f})"
                    else:
                        bg = "transparent"
                    html += f'<td style="background:{bg}">{val:+.1f}</td>'
                else:
                    html += '<td style="color:#333">—</td>'
            # Year total
            if yr_total > 0:
                bg = f"rgba(0, 230, 118, 0.3)"
            elif yr_total < 0:
                bg = f"rgba(255, 82, 82, 0.3)"
            else:
                bg = "transparent"
            html += f'<td style="background:{bg};font-weight:bold">{yr_total:+.1f}</td>'
            html += "</tr>\n"
        html += "</table></div>\n"

    html += "</body></html>"
    return html


def main():
    print("Loading 1-min bars...", flush=True)
    df = load_1min_bars()
    print(f"  {len(df):,} bars, {df.index[0]} to {df.index[-1]}", flush=True)

    # Filter to extended session (9:30-16:45 for holding positions past close)
    session_df = get_session_bars(df, "09:30", "16:45")
    print(f"  Session bars (incl. extended): {len(session_df):,}", flush=True)

    # Daily OHLCV from RTH only (9:30-16:00) for vol calc and sigma
    rth_df = get_session_bars(df, "09:30", "16:00")
    daily_df = compute_daily_ohlc(rth_df)
    print(f"  Trading days: {len(daily_df)}", flush=True)

    # Run both configs
    configs = {
        "quantitativo": Config(
            name="quantitativo",
            lookback_days=90,
            vol_target=0.03,
            leverage_cap=8.0,
        ),
        "original_paper": Config(
            name="original_paper",
            lookback_days=14,
            vol_target=0.02,
            leverage_cap=4.0,
        ),
    }

    all_trades = {}
    all_stats = {}

    for name, cfg in configs.items():
        print(f"\n{'='*60}\nRunning {name}...\n{'='*60}", flush=True)
        trades = run_backtest(session_df, daily_df, cfg)
        stats = compute_stats(trades, cfg.starting_equity)
        all_trades[name] = trades
        all_stats[name] = stats

        # Print summary
        print(f"\n--- {name.upper()} Summary ---")
        for k, v in stats.items():
            if k != "exit_reasons":
                print(f"  {k}: {v}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Trade logs
    for name, trades in all_trades.items():
        csv_path = RESULTS_DIR / f"trades_{name}.csv"
        trades_to_df(trades).to_csv(csv_path, index=False)
        print(f"Saved {csv_path}", flush=True)

    # HTML report
    html = generate_html_report(all_stats, all_trades, configs["quantitativo"].starting_equity)
    html_path = RESULTS_DIR / "noise_band_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"Saved {html_path}", flush=True)

    # Side-by-side comparison
    print(f"\n{'='*70}")
    print(f"{'METRIC':<25} {'QUANTITATIVO':>20} {'ORIGINAL PAPER':>20}")
    print(f"{'='*70}")
    for key in ["total_trades", "trades_per_year", "total_return_pct", "annualized_return_pct",
                 "sharpe", "sortino", "max_drawdown_pct", "calmar", "win_rate_pct",
                 "payoff_ratio", "profit_factor", "avg_return_bps", "avg_duration_min",
                 "total_pnl", "avg_win", "avg_loss"]:
        v1 = all_stats["quantitativo"].get(key, "N/A")
        v2 = all_stats["original_paper"].get(key, "N/A")
        if isinstance(v1, float):
            v1_str = f"{v1:>20,.2f}"
        else:
            v1_str = f"{str(v1):>20}"
        if isinstance(v2, float):
            v2_str = f"{v2:>20,.2f}"
        else:
            v2_str = f"{str(v2):>20}"
        print(f"{key:<25} {v1_str} {v2_str}")


if __name__ == "__main__":
    main()
