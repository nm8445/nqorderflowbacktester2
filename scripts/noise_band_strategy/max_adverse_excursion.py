"""
Max Adverse Excursion (MAE) analysis for noise-band buffer k=0.5.
Tracks the worst unrealized loss per trade (bar-by-bar) before exit.
Reports in both NQ points and MNQ dollars (1-10 contracts).
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

ONEMN_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
SIGMA_CACHE_DIR = Path("D:/trading_pythonbacktest_data/noise_band_sigma_cache")
ET = "America/New_York"
NQ_TICK_SIZE = 0.25
SLIPPAGE_TICKS_PER_SIDE = 0.25
BUFFER_K = 0.5
MNQ_POINT_VALUE = 2.0


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


def run_mae_analysis():
    print("Loading 1-min bars...", flush=True)
    df = load_data()
    hm = df.index.strftime("%H:%M")
    session_df = df[(hm >= "09:30") & (hm <= "16:45")].copy()

    trading_dates = sorted(set(session_df.index.date))
    bars_by_date = {}
    for d in trading_dates:
        bars_by_date[d] = session_df[session_df.index.date == d]

    print("Loading sigma cache...", flush=True)
    with open(SIGMA_CACHE_DIR / "sigma_lookback_90.pkl", "rb") as f:
        sigma_cache = pickle.load(f)

    entry_times = set()
    h, m = 10, 0
    while (h, m) <= (15, 30):
        entry_times.add(f"{h:02d}:{m:02d}")
        m += 30
        if m >= 60:
            h += m // 60
            m = m % 60

    trades = []  # each: {mae_points, pnl_points, direction, date, bars_held}

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

        in_pos = False
        direction = ""
        entry_price = 0.0
        trailing_stop = 0.0
        bars_held = 0
        worst_unrealized = 0.0  # most negative unrealized PnL in points
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

                # Track MAE using bar extremes
                if direction == "long":
                    unrealized_worst = bar["low"] - entry_price  # negative when losing
                else:
                    unrealized_worst = entry_price - bar["high"]  # negative when losing
                worst_unrealized = min(worst_unrealized, unrealized_worst)

                # Exit: EOD close
                if hm_str >= "16:45":
                    ep_out = bar["close"]
                    if direction == "long":
                        ep_out -= SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        pnl_pts = ep_out - entry_price
                    else:
                        ep_out += SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        pnl_pts = entry_price - ep_out
                    trades.append({
                        "mae_points": worst_unrealized,
                        "pnl_points": pnl_pts,
                        "direction": direction,
                        "date": str(today),
                        "bars_held": bars_held,
                    })
                    in_pos = False
                    continue

                # Exit: trailing stop
                if direction == "long":
                    band_sl = upper_band - BUFFER_K * sig * upper_anchor
                    new_stop = max(band_sl, vwap)
                    trailing_stop = max(trailing_stop, new_stop)
                    if bar["low"] <= trailing_stop:
                        ep_out = bar["open"] if bar["open"] <= trailing_stop else trailing_stop
                        ep_out -= SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        pnl_pts = ep_out - entry_price
                        trades.append({
                            "mae_points": worst_unrealized,
                            "pnl_points": pnl_pts,
                            "direction": direction,
                            "date": str(today),
                            "bars_held": bars_held,
                        })
                        in_pos = False
                        continue
                else:
                    band_sl = lower_band + BUFFER_K * sig * lower_anchor
                    new_stop = min(band_sl, vwap)
                    trailing_stop = min(trailing_stop, new_stop)
                    if bar["high"] >= trailing_stop:
                        ep_out = bar["open"] if bar["open"] >= trailing_stop else trailing_stop
                        ep_out += SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                        pnl_pts = entry_price - ep_out
                        trades.append({
                            "mae_points": worst_unrealized,
                            "pnl_points": pnl_pts,
                            "direction": direction,
                            "date": str(today),
                            "bars_held": bars_held,
                        })
                        in_pos = False
                        continue

            # Entry
            if not in_pos and hm_str in entry_times:
                price = bar["close"]
                if price > upper_band and bar_i + 1 < len(today_bars):
                    nb = today_bars.iloc[bar_i + 1]
                    entry_price = nb["open"] + SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                    in_pos = True
                    direction = "long"
                    bars_held = 0
                    worst_unrealized = 0.0
                    band_sl = upper_band - BUFFER_K * sig * upper_anchor
                    trailing_stop = max(band_sl, vwap)
                elif price < lower_band and bar_i + 1 < len(today_bars):
                    nb = today_bars.iloc[bar_i + 1]
                    entry_price = nb["open"] - SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                    in_pos = True
                    direction = "short"
                    bars_held = 0
                    worst_unrealized = 0.0
                    band_sl = lower_band + BUFFER_K * sig * lower_anchor
                    trailing_stop = min(band_sl, vwap)

        # Safety close
        if in_pos:
            last_bar = today_bars.iloc[-1]
            ep_out = last_bar["close"]
            if direction == "long":
                ep_out -= SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                pnl_pts = ep_out - entry_price
            else:
                ep_out += SLIPPAGE_TICKS_PER_SIDE * NQ_TICK_SIZE
                pnl_pts = entry_price - ep_out
            trades.append({
                "mae_points": worst_unrealized,
                "pnl_points": pnl_pts,
                "direction": direction,
                "date": str(today),
                "bars_held": bars_held,
            })

    return trades


def main():
    trades = run_mae_analysis()

    mae_pts = np.array([t["mae_points"] for t in trades])
    pnl_pts = np.array([t["pnl_points"] for t in trades])

    print(f"\n{'='*80}")
    print(f"MAX ADVERSE EXCURSION — noise-band buffer k=0.5, {len(trades)} trades")
    print(f"{'='*80}")

    print(f"\n--- MAE in NQ points (negative = against you) ---")
    print(f"  Worst MAE:     {mae_pts.min():+.2f} pts")
    print(f"  Mean MAE:      {mae_pts.mean():+.2f} pts")
    print(f"  Median MAE:    {np.median(mae_pts):+.2f} pts")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  P{p:>2}:            {np.percentile(mae_pts, p):+.2f} pts")

    # Winners vs losers
    win_mask = pnl_pts > 0
    lose_mask = pnl_pts <= 0
    w_mae = mae_pts[win_mask]
    l_mae = mae_pts[lose_mask]

    print(f"\n--- Winners ({win_mask.sum()}) ---")
    print(f"  Worst MAE:     {w_mae.min():+.2f} pts")
    print(f"  Mean MAE:      {w_mae.mean():+.2f} pts")
    print(f"  Median MAE:    {np.median(w_mae):+.2f} pts")

    print(f"\n--- Losers ({lose_mask.sum()}) ---")
    print(f"  Worst MAE:     {l_mae.min():+.2f} pts")
    print(f"  Mean MAE:      {l_mae.mean():+.2f} pts")
    print(f"  Median MAE:    {np.median(l_mae):+.2f} pts")

    # MNQ dollar impact for 1-10 contracts
    print(f"\n{'='*80}")
    print(f"MAE IN MNQ DOLLARS ($2/pt)")
    print(f"{'='*80}")
    print(f"\n{'MNQ':>4} | {'Worst MAE':>10} | {'Mean MAE':>10} | {'Median MAE':>10} | "
          f"{'P95 MAE':>10} | {'P99 MAE':>10}")
    print("-" * 75)
    for nc in range(1, 11):
        mae_dollars = mae_pts * MNQ_POINT_VALUE * nc
        print(f"{nc:>4} | ${mae_dollars.min():>+9,.0f} | ${mae_dollars.mean():>+9,.0f} | "
              f"${np.median(mae_dollars):>+9,.0f} | "
              f"${np.percentile(mae_dollars, 5):>+9,.0f} | "
              f"${np.percentile(mae_dollars, 1):>+9,.0f}")

    # How many trades breach $2k DD threshold per contract count
    print(f"\n{'='*80}")
    print(f"TRADES BREACHING LUCID $2k DD (unrealized loss > $2,000)")
    print(f"{'='*80}")
    print(f"\n{'MNQ':>4} | {'Breach Count':>12} | {'Breach %':>8} | {'Worst $':>10}")
    print("-" * 50)
    for nc in range(1, 11):
        mae_dollars = mae_pts * MNQ_POINT_VALUE * nc
        breach = mae_dollars < -2000
        print(f"{nc:>4} | {breach.sum():>12} | {100*breach.sum()/len(mae_dollars):>7.1f}% | "
              f"${mae_dollars.min():>+9,.0f}")

    # Scatter: MAE vs final PnL (top 20 worst)
    print(f"\n{'='*80}")
    print(f"20 WORST MAE TRADES (NQ points)")
    print(f"{'='*80}")
    sorted_idx = np.argsort(mae_pts)
    print(f"{'#':>3} | {'MAE pts':>8} | {'PnL pts':>8} | {'Dir':>5} | {'Bars':>4} | {'Date':>10}")
    print("-" * 55)
    for i in range(min(20, len(sorted_idx))):
        t = trades[sorted_idx[i]]
        print(f"{i+1:>3} | {t['mae_points']:>+8.2f} | {t['pnl_points']:>+8.2f} | "
              f"{t['direction']:>5} | {t['bars_held']:>4} | {t['date']}")


if __name__ == "__main__":
    main()
