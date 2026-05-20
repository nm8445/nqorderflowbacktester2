"""Compare buffer k=0.5 with fixed 1 contract vs dynamic vol-target sizing."""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict

ONEMN_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
SIGMA_CACHE_DIR = Path("D:/trading_pythonbacktest_data/noise_band_sigma_cache")
ET = "America/New_York"
NQ_POINT_VALUE = 20.0
NQ_TICK_SIZE = 0.25
COMMISSION_PER_SIDE = 0.85
EXCHANGE_FEE_PER_SIDE = 1.40
SLIPPAGE_TICKS_PER_SIDE = 0.25
BUFFER_K = 0.5


def load_data():
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
    return log_rets.std() * np.sqrt(252) if len(log_rets) >= 5 else 0.15


def compute_position_size(equity, realized_vol, price):
    if realized_vol <= 0 or price <= 0:
        return 1
    daily_realized = realized_vol / np.sqrt(252)
    contracts_vol = (equity * 0.03) / (daily_realized * price * NQ_POINT_VALUE)
    max_lev = (equity * 8.0) / (price * NQ_POINT_VALUE)
    return max(1, int(min(contracts_vol, max_lev)))


def run_backtest(bars_by_date, trading_dates, sigma_cache, daily_df, daily_date_to_idx,
                 fixed_qty=None):
    """Run buffer k=0.5 backtest. fixed_qty=None for dynamic, int for fixed."""
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
            hm_str = today_bars.index[bar_i].strftime("%H:%M")

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

                if hm_str >= "16:45":
                    ep_out = bar["close"]
                    if direction == "long":
                        ep_out -= SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        gross = (ep_out - entry_price) * NQ_POINT_VALUE * contracts
                    else:
                        ep_out += SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        gross = (entry_price - ep_out) * NQ_POINT_VALUE * contracts
                    cost = (COMMISSION_PER_SIDE + EXCHANGE_FEE_PER_SIDE) * 2 * contracts
                    trades.append({"pnl": gross - cost, "exit": "eod_close",
                                   "dir": direction, "date": str(today), "bars": bars_held,
                                   "contracts": contracts})
                    equity += gross - cost
                    in_pos = False
                    continue

                if direction == "long":
                    band_sl = upper_band - BUFFER_K * sig * upper_anchor
                    new_stop = max(band_sl, vwap)
                    trailing_stop = max(trailing_stop, new_stop)
                    if bar["low"] <= trailing_stop:
                        ep_out = bar["open"] if bar["open"] <= trailing_stop else trailing_stop
                        ep_out -= SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        gross = (ep_out - entry_price) * NQ_POINT_VALUE * contracts
                        cost = (COMMISSION_PER_SIDE + EXCHANGE_FEE_PER_SIDE) * 2 * contracts
                        ex = "vwap_stop" if vwap >= band_sl else "band_stop"
                        trades.append({"pnl": gross - cost, "exit": ex,
                                       "dir": direction, "date": str(today), "bars": bars_held,
                                       "contracts": contracts})
                        equity += gross - cost
                        in_pos = False
                        continue
                else:
                    band_sl = lower_band + BUFFER_K * sig * lower_anchor
                    new_stop = min(band_sl, vwap)
                    trailing_stop = min(trailing_stop, new_stop)
                    if bar["high"] >= trailing_stop:
                        ep_out = bar["open"] if bar["open"] >= trailing_stop else trailing_stop
                        ep_out += SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        gross = (entry_price - ep_out) * NQ_POINT_VALUE * contracts
                        cost = (COMMISSION_PER_SIDE + EXCHANGE_FEE_PER_SIDE) * 2 * contracts
                        ex = "vwap_stop" if vwap <= band_sl else "band_stop"
                        trades.append({"pnl": gross - cost, "exit": ex,
                                       "dir": direction, "date": str(today), "bars": bars_held,
                                       "contracts": contracts})
                        equity += gross - cost
                        in_pos = False
                        continue

            # Entry
            if not in_pos and hm_str in entry_times:
                price = bar["close"]
                entered = False
                if price > upper_band and bar_i + 1 < len(today_bars):
                    nb = today_bars.iloc[bar_i + 1]
                    ep = nb["open"] + SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                    if fixed_qty is not None:
                        contracts = fixed_qty
                    else:
                        contracts = compute_position_size(equity, realized_vol, ep)
                    in_pos = True
                    direction = "long"
                    entry_price = ep
                    bars_held = 0
                    band_sl = upper_band - BUFFER_K * sig * upper_anchor
                    trailing_stop = max(band_sl, vwap)
                elif price < lower_band and bar_i + 1 < len(today_bars):
                    nb = today_bars.iloc[bar_i + 1]
                    ep = nb["open"] - SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                    if fixed_qty is not None:
                        contracts = fixed_qty
                    else:
                        contracts = compute_position_size(equity, realized_vol, ep)
                    in_pos = True
                    direction = "short"
                    entry_price = ep
                    bars_held = 0
                    band_sl = lower_band + BUFFER_K * sig * lower_anchor
                    trailing_stop = min(band_sl, vwap)

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
            trades.append({"pnl": gross - cost, "exit": "eod_close",
                           "dir": direction, "date": str(today), "bars": bars_held,
                           "contracts": contracts})
            equity += gross - cost

    return trades, equity


def print_stats(label, trades, starting_eq=100000):
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 else 0
    wr = 100 * len(wins) / len(pnls)
    cum = pnls.cumsum()
    eq_curve = starting_eq + cum
    peak = np.maximum.accumulate(eq_curve)
    dd = ((eq_curve - peak) / peak * 100).min()

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"Trades:    {len(pnls)}")
    print(f"WR:        {wr:.1f}%")
    print(f"PF:        {pf:.3f}")
    print(f"Total PnL: ${pnls.sum():+,.0f}")
    print(f"MaxDD:     {dd:.1f}%")
    print(f"Final Eq:  ${starting_eq + pnls.sum():,.0f}")
    print(f"Avg Win:   ${wins.mean():+,.0f}")
    print(f"Avg Loss:  ${losses.mean():+,.0f}")
    print(f"Avg PnL:   ${pnls.mean():+,.0f}")

    # Position size stats
    contract_sizes = [t["contracts"] for t in trades]
    print(f"\nPosition sizing:")
    print(f"  Min contracts:  {min(contract_sizes)}")
    print(f"  Max contracts:  {max(contract_sizes)}")
    print(f"  Mean contracts: {np.mean(contract_sizes):.1f}")
    print(f"  Median:         {np.median(contract_sizes):.0f}")

    # Exit breakdown
    exits = defaultdict(list)
    for t in trades:
        exits[t["exit"]].append(t["pnl"])
    print(f"\nExit breakdown:")
    for ex in sorted(exits.keys()):
        arr = np.array(exits[ex])
        ewr = 100 * (arr > 0).sum() / len(arr)
        print(f"  {ex:>12}: {len(arr):>5} ({100*len(arr)/len(pnls):>5.1f}%)  "
              f"WR: {ewr:>5.1f}%  Avg: ${arr.mean():>+8,.0f}  Total: ${arr.sum():>+12,.0f}")

    # Year by year
    print(f"\n{'Year':>6} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'PnL':>12}")
    print("-" * 50)
    yearly = defaultdict(list)
    for t in trades:
        yearly[t["date"][:4]].append(t["pnl"])
    for yr in sorted(yearly.keys()):
        arr = np.array(yearly[yr])
        w = arr[arr > 0]
        l = arr[arr < 0]
        yr_pf = w.sum() / abs(l.sum()) if len(l) > 0 else 0
        yr_wr = 100 * len(w) / len(arr)
        print(f"{yr:>6} | {len(arr):>6} | {yr_wr:>5.1f} | {yr_pf:>5.2f} | ${arr.sum():>+11,.0f}")

    # Long vs Short
    longs = [t for t in trades if t["dir"] == "long"]
    shorts = [t for t in trades if t["dir"] == "short"]
    l_pnl = np.array([t["pnl"] for t in longs])
    s_pnl = np.array([t["pnl"] for t in shorts])
    print(f"\nLong vs Short:")
    print(f"  Long:  {len(longs):>5}  WR: {100*(l_pnl>0).sum()/len(l_pnl):.1f}%  "
          f"Avg: ${l_pnl.mean():>+,.0f}  Total: ${l_pnl.sum():>+,.0f}")
    print(f"  Short: {len(shorts):>5}  WR: {100*(s_pnl>0).sum()/len(s_pnl):.1f}%  "
          f"Avg: ${s_pnl.mean():>+,.0f}  Total: ${s_pnl.sum():>+,.0f}")

    # Duration
    w_bars = np.array([t["bars"] for t in trades if t["pnl"] > 0])
    l_bars = np.array([t["bars"] for t in trades if t["pnl"] <= 0])
    print(f"\nDuration:")
    print(f"  Winner median: {np.median(w_bars):.0f} min  mean: {w_bars.mean():.0f} min")
    print(f"  Loser median:  {np.median(l_bars):.0f} min  mean: {l_bars.mean():.0f} min")


def main():
    print("Loading data...", flush=True)
    df = load_data()
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

    # Dynamic sizing
    print("\nRunning DYNAMIC sizing...", flush=True)
    trades_dyn, eq_dyn = run_backtest(
        bars_by_date, trading_dates, sigma_cache, daily_df, daily_date_to_idx,
        fixed_qty=None,
    )
    print_stats("BUFFER k=0.5 — DYNAMIC VOL-TARGET SIZING", trades_dyn)

    # Fixed 1 contract
    print("\nRunning FIXED 1 contract...", flush=True)
    trades_fix, eq_fix = run_backtest(
        bars_by_date, trading_dates, sigma_cache, daily_df, daily_date_to_idx,
        fixed_qty=1,
    )
    print_stats("BUFFER k=0.5 — FIXED 1 NQ CONTRACT", trades_fix)


if __name__ == "__main__":
    main()
