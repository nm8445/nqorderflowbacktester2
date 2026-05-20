"""
Test different trailing stop modes for the noise-band strategy.
Modes: baseline, vwap_only, buffer k=0.5/1.0/1.5, time-based switch to VWAP after 15 min.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

ONEMN_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
SIGMA_CACHE_DIR = Path("D:/trading_pythonbacktest_data/noise_band_sigma_cache")
ET = "America/New_York"
NQ_POINT_VALUE = 20.0
NQ_TICK_SIZE = 0.25
COMMISSION_PER_SIDE = 0.85
EXCHANGE_FEE_PER_SIDE = 1.40
SLIPPAGE_TICKS_PER_SIDE = 0.25


def load_bars():
    df = pd.read_parquet(ONEMN_PARQUET)
    new_idx = []
    for t in df.index:
        if hasattr(t, "tz_convert") and t.tzinfo:
            new_idx.append(t.tz_convert(ET))
        else:
            new_idx.append(pd.Timestamp(t).tz_localize("UTC").tz_convert(ET))
    df.index = pd.DatetimeIndex(new_idx)
    return df


def compute_realized_vol(daily_df, today_idx, window=14):
    if today_idx < window + 1:
        return 0.15
    start = max(0, today_idx - window)
    subset = daily_df.iloc[start:today_idx]
    log_rets = np.log(subset["close"] / subset["close"].shift(1)).dropna()
    if len(log_rets) < 5:
        return 0.15
    return log_rets.std() * np.sqrt(252)


def compute_position_size(equity, vol_target, realized_vol, leverage_cap, price):
    if realized_vol <= 0 or price <= 0:
        return 1
    daily_realized = realized_vol / np.sqrt(252)
    contracts_vol = (equity * vol_target) / (daily_realized * price * NQ_POINT_VALUE)
    max_lev = (equity * leverage_cap) / (price * NQ_POINT_VALUE)
    return max(1, int(min(contracts_vol, max_lev)))


MODES = {
    "baseline":         {"band_buffer_k": 0.0, "time_switch_min": 0,  "use_band": True},
    "vwap_only":        {"band_buffer_k": 0.0, "time_switch_min": 0,  "use_band": False},
    "buffer_k0.5":      {"band_buffer_k": 0.5, "time_switch_min": 0,  "use_band": True},
    "buffer_k1.0":      {"band_buffer_k": 1.0, "time_switch_min": 0,  "use_band": True},
    "buffer_k1.5":      {"band_buffer_k": 1.5, "time_switch_min": 0,  "use_band": True},
    "time15_then_vwap": {"band_buffer_k": 0.0, "time_switch_min": 15, "use_band": True},
}


def run_mode(mode_name, mode_cfg, session_df, bars_by_date, trading_dates,
             sigma_cache, daily_df, daily_date_to_idx, daily_dates):
    use_band = mode_cfg["use_band"]
    buffer_k = mode_cfg["band_buffer_k"]
    time_switch = mode_cfg["time_switch_min"]

    entry_times = set()
    h, m = 10, 0
    while (h, m) <= (15, 30):
        entry_times.add(f"{h:02d}:{m:02d}")
        m += 30
        if m >= 60:
            h += m // 60
            m = m % 60

    equity = 100000.0
    trades = []

    for day_num, today in enumerate(trading_dates):
        if day_num < 90:
            continue

        today_bars = bars_by_date.get(today)
        if today_bars is None or len(today_bars) < 30:
            continue

        today_open = today_bars.iloc[0]["open"]
        prev_day_idx = day_num - 1
        while prev_day_idx >= 0 and trading_dates[prev_day_idx] not in bars_by_date:
            prev_day_idx -= 1
        if prev_day_idx < 0:
            continue
        yesterday_close = bars_by_date[trading_dates[prev_day_idx]].iloc[-1]["close"]

        sigma = sigma_cache.get(today)
        if not sigma:
            continue

        upper_anchor = max(today_open, yesterday_close)
        lower_anchor = min(today_open, yesterday_close)
        last_sigma = max(sigma.values())

        daily_idx = daily_date_to_idx.get(today)
        if daily_idx is None:
            continue
        realized_vol = compute_realized_vol(daily_df, daily_idx)

        in_pos = False
        direction = ""
        entry_price = 0.0
        contracts = 0
        trailing_stop = 0.0
        bars_held = 0
        vwap_pv = 0.0
        vwap_v = 0.0

        for bar_i in range(len(today_bars)):
            bar = today_bars.iloc[bar_i]
            bar_ts = today_bars.index[bar_i]
            hm_str = bar_ts.strftime("%H:%M")

            tp = (bar["high"] + bar["low"] + bar["close"]) / 3.0
            bv = bar["volume"] if bar["volume"] > 0 else 1
            vwap_pv += tp * bv
            vwap_v += bv
            vwap = vwap_pv / vwap_v if vwap_v > 0 else bar["close"]

            sig = sigma.get(hm_str)
            if sig is None or sig <= 0:
                sig = last_sigma
            upper_band = upper_anchor * (1.0 + sig)
            lower_band = lower_anchor * (1.0 - sig)

            if in_pos:
                bars_held += 1

                # Force close
                if hm_str >= "16:45":
                    ep_out = bar["close"]
                    if direction == "long":
                        ep_out -= SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        gross = (ep_out - entry_price) * NQ_POINT_VALUE * contracts
                    else:
                        ep_out += SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        gross = (entry_price - ep_out) * NQ_POINT_VALUE * contracts
                    cost = (COMMISSION_PER_SIDE + EXCHANGE_FEE_PER_SIDE) * 2 * contracts
                    trades.append({"pnl": gross - cost, "exit": "eod_close", "dir": direction})
                    equity += gross - cost
                    in_pos = False
                    continue

                # Determine whether band is active
                use_band_now = use_band
                if time_switch > 0 and bars_held > time_switch:
                    use_band_now = False

                if direction == "long":
                    if use_band_now:
                        band_stop_level = upper_band - buffer_k * sig * upper_anchor
                        new_stop = max(band_stop_level, vwap)
                    else:
                        new_stop = vwap
                    trailing_stop = max(trailing_stop, new_stop)

                    if bar["low"] <= trailing_stop:
                        if bar["open"] <= trailing_stop:
                            ep_out = bar["open"]
                        else:
                            ep_out = trailing_stop
                        ep_out -= SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        gross = (ep_out - entry_price) * NQ_POINT_VALUE * contracts
                        cost = (COMMISSION_PER_SIDE + EXCHANGE_FEE_PER_SIDE) * 2 * contracts
                        if not use_band_now:
                            ex_reason = "vwap_stop"
                        elif vwap >= band_stop_level:
                            ex_reason = "vwap_stop"
                        else:
                            ex_reason = "band_stop"
                        trades.append({"pnl": gross - cost, "exit": ex_reason, "dir": direction})
                        equity += gross - cost
                        in_pos = False
                        continue
                else:  # short
                    if use_band_now:
                        band_stop_level = lower_band + buffer_k * sig * lower_anchor
                        new_stop = min(band_stop_level, vwap)
                    else:
                        new_stop = vwap
                    trailing_stop = min(trailing_stop, new_stop)

                    if bar["high"] >= trailing_stop:
                        if bar["open"] >= trailing_stop:
                            ep_out = bar["open"]
                        else:
                            ep_out = trailing_stop
                        ep_out += SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        gross = (entry_price - ep_out) * NQ_POINT_VALUE * contracts
                        cost = (COMMISSION_PER_SIDE + EXCHANGE_FEE_PER_SIDE) * 2 * contracts
                        if not use_band_now:
                            ex_reason = "vwap_stop"
                        elif vwap <= band_stop_level:
                            ex_reason = "vwap_stop"
                        else:
                            ex_reason = "band_stop"
                        trades.append({"pnl": gross - cost, "exit": ex_reason, "dir": direction})
                        equity += gross - cost
                        in_pos = False
                        continue

            # Entry
            if not in_pos and hm_str in entry_times:
                price = bar["close"]
                if price > upper_band and bar_i + 1 < len(today_bars):
                    nb = today_bars.iloc[bar_i + 1]
                    ep = nb["open"] + SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                    contracts = compute_position_size(equity, 0.03, realized_vol, 8.0, ep)
                    in_pos = True
                    direction = "long"
                    entry_price = ep
                    bars_held = 0
                    if use_band:
                        band_sl = upper_band - buffer_k * sig * upper_anchor
                        trailing_stop = max(band_sl, vwap)
                    else:
                        trailing_stop = vwap
                elif price < lower_band and bar_i + 1 < len(today_bars):
                    nb = today_bars.iloc[bar_i + 1]
                    ep = nb["open"] - SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                    contracts = compute_position_size(equity, 0.03, realized_vol, 8.0, ep)
                    in_pos = True
                    direction = "short"
                    entry_price = ep
                    bars_held = 0
                    if use_band:
                        band_sl = lower_band + buffer_k * sig * lower_anchor
                        trailing_stop = min(band_sl, vwap)
                    else:
                        trailing_stop = vwap

        # Safety close
        if in_pos:
            last_bar = today_bars.iloc[-1]
            ep_out = last_bar["close"]
            if direction == "long":
                ep_out -= SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                gross = (ep_out - entry_price) * NQ_POINT_VALUE * contracts
            else:
                ep_out += SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                gross = (entry_price - ep_out) * NQ_POINT_VALUE * contracts
            cost = (COMMISSION_PER_SIDE + EXCHANGE_FEE_PER_SIDE) * 2 * contracts
            trades.append({"pnl": gross - cost, "exit": "eod_close", "dir": direction})
            equity += gross - cost
            in_pos = False

    # Stats
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else 0
    wr = 100 * len(wins) / len(pnls) if len(pnls) > 0 else 0
    cum = pnls.cumsum()
    peak = np.maximum.accumulate(100000 + cum)
    dd = ((100000 + cum) - peak) / peak * 100

    exits = {}
    for t in trades:
        ex = t["exit"]
        if ex not in exits:
            exits[ex] = []
        exits[ex].append(t["pnl"])

    return {
        "trades": len(pnls),
        "wr": wr,
        "pf": pf,
        "total_pnl": pnls.sum(),
        "max_dd": dd.min(),
        "avg_win": wins.mean() if len(wins) > 0 else 0,
        "avg_loss": losses.mean() if len(losses) > 0 else 0,
        "final_eq": equity,
        "exits": exits,
    }


def main():
    print("Loading 1-min bars...", flush=True)
    df = load_bars()
    hm = df.index.strftime("%H:%M")
    session_df = df[(hm >= "09:30") & (hm <= "16:45")].copy()

    trading_dates = sorted(set(session_df.index.date))
    bars_by_date = {}
    for d in trading_dates:
        bars_by_date[d] = session_df[session_df.index.date == d]

    rth = df[(df.index.strftime("%H:%M") >= "09:30") & (df.index.strftime("%H:%M") <= "16:00")]
    rth_copy = rth.copy()
    rth_copy["date"] = rth_copy.index.date
    daily_df = rth_copy.groupby("date").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    )
    daily_df.index = pd.DatetimeIndex([pd.Timestamp(d) for d in daily_df.index])
    daily_dates = sorted(set(daily_df.index.date))
    daily_date_to_idx = {d: i for i, d in enumerate(daily_dates)}

    print("Loading sigma cache...", flush=True)
    with open(SIGMA_CACHE_DIR / "sigma_lookback_90.pkl", "rb") as f:
        sigma_cache = pickle.load(f)

    print(f"Running {len(MODES)} stop modes...\n", flush=True)

    results = {}
    for mode_name, mode_cfg in MODES.items():
        print(f"  {mode_name}...", flush=True)
        results[mode_name] = run_mode(
            mode_name, mode_cfg, session_df, bars_by_date, trading_dates,
            sigma_cache, daily_df, daily_date_to_idx, daily_dates,
        )

    # Comparison table
    print(f"\n{'='*110}")
    print(f"{'Mode':<20} {'Trades':>7} {'WR%':>6} {'PF':>6} {'Total PnL':>12} "
          f"{'MaxDD%':>8} {'Final Eq':>12} {'Avg Win':>9} {'Avg Loss':>9}")
    print(f"{'='*110}")
    for mode_name in MODES:
        r = results[mode_name]
        print(f"{mode_name:<20} {r['trades']:>7} {r['wr']:>5.1f}% {r['pf']:>5.2f} "
              f"${r['total_pnl']:>+11,.0f} {r['max_dd']:>7.1f}% "
              f"${r['final_eq']:>11,.0f} ${r['avg_win']:>+8,.0f} ${r['avg_loss']:>+8,.0f}")

    # Exit breakdown per mode
    for mode_name in MODES:
        r = results[mode_name]
        print(f"\n--- {mode_name} exit breakdown ---")
        for ex in sorted(r["exits"].keys()):
            arr = np.array(r["exits"][ex])
            ewr = 100 * (arr > 0).sum() / len(arr) if len(arr) > 0 else 0
            print(f"  {ex:>12}: {len(arr):>5} ({100*len(arr)/r['trades']:>5.1f}%)  "
                  f"WR: {ewr:>5.1f}%  Avg: ${arr.mean():>+8,.0f}  Total: ${arr.sum():>+12,.0f}")


if __name__ == "__main__":
    main()
