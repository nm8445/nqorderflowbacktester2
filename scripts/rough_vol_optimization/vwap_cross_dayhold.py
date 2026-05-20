"""
Pure VWAP cross day-hold strategy.

Entry LONG: previous 15-min close was BELOW vwap, current candle closes ABOVE vwap
Entry SHORT: previous 15-min close was ABOVE vwap, current candle closes BELOW vwap
Exit: close below VWAP closes long (and opens short). Force close 16:30.
No SL, no TP. Latest entry 16:00. Session 09:30+.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

MARKETTICK_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_15min_bars.parquet")
TIMEBARS_DIR = Path("D:/trading_pythonbacktest_data/timebars_5min")
VWAP_CACHE = Path("D:/trading_pythonbacktest_data/vwap_cache_5yr")
ET = "America/New_York"; POINT_VALUE = 20.0

FORCE_CLOSE = "16:30"
LATEST_ENTRY = "16:00"
SESSION_START = "09:30"


def build_combined_bars():
    print("Loading 15-min bars...", flush=True)
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


def load_vwap_for_date(date_obj):
    cache_file = VWAP_CACHE / f"{date_obj}.pkl"
    if not cache_file.exists():
        return None
    with open(cache_file, "rb") as f:
        df = pickle.load(f)
    df["group"] = df.index.floor("15min")
    vwap_15 = df.groupby("group")["vwap"].last()
    return vwap_15.to_dict()


def get_vwap(vwap_dict, idx):
    if vwap_dict is None:
        return None
    bar_utc = idx.tz_convert("UTC").floor("15min")
    val = vwap_dict.get(bar_utc)
    if val is None:
        val = vwap_dict.get(bar_utc - pd.Timedelta(minutes=15))
    return val


def main():
    df = build_combined_bars()
    n = len(df)

    print("Loading VWAP...", flush=True)
    all_dates = sorted(set(idx.date() for idx in df.index))
    vwap_by_date = {}
    for d in all_dates:
        v = load_vwap_for_date(d)
        if v is not None:
            vwap_by_date[d] = v
    print(f"  {len(vwap_by_date)} dates loaded", flush=True)

    trading_days = sorted(set(idx.date() for idx in df.index))
    split_idx = int(len(trading_days) * 0.60)
    is_cutoff = str(trading_days[split_idx])

    # ── Version 1: Flip on every cross ──
    print("\n" + "=" * 80)
    print("V1: Flip on every VWAP cross (close below = close long & open short)")
    print("=" * 80)

    trades = []
    pos = 0; ep = 0; d = ""; cd = None
    prev_close = None; prev_vwap = None

    for i in range(n):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M")
        cl = row["close"]

        vwap_dict = vwap_by_date.get(td)
        vwap_val = get_vwap(vwap_dict, idx)

        # Force close
        if pos != 0 and hm >= FORCE_CLOSE:
            pnl = ((cl - ep) if d == "long" else (ep - cl)) * POINT_VALUE
            trades.append({"pnl": round(pnl, 2), "date": str(td), "dir": d, "reason": "CLOSE"})
            pos = 0
            prev_close = cl; prev_vwap = vwap_val
            continue

        if hm >= FORCE_CLOSE or hm < SESSION_START:
            prev_close = cl; prev_vwap = vwap_val
            continue

        if td != cd:
            cd = td; prev_close = None; prev_vwap = None
            prev_close = cl; prev_vwap = vwap_val
            continue

        if vwap_val is None or prev_close is None or prev_vwap is None:
            prev_close = cl; prev_vwap = vwap_val
            continue

        # Cross detection: prev close vs prev vwap -> current close vs current vwap
        was_below = prev_close < prev_vwap
        was_above = prev_close > prev_vwap
        now_above = cl > vwap_val
        now_below = cl < vwap_val

        cross_up = was_below and now_above
        cross_down = was_above and now_below

        if cross_up:
            if pos == -1:
                # Close short
                pnl = (ep - cl) * POINT_VALUE
                trades.append({"pnl": round(pnl, 2), "date": str(td), "dir": "short", "reason": "FLIP"})
                pos = 0
            if pos == 0 and hm <= LATEST_ENTRY:
                pos = 1; d = "long"; ep = cl
        elif cross_down:
            if pos == 1:
                # Close long
                pnl = (cl - ep) * POINT_VALUE
                trades.append({"pnl": round(pnl, 2), "date": str(td), "dir": "long", "reason": "FLIP"})
                pos = 0
            if pos == 0 and hm <= LATEST_ENTRY:
                pos = -1; d = "short"; ep = cl

        prev_close = cl; prev_vwap = vwap_val

    print_results(trades, is_cutoff, "V1 Flip")

    # ── Version 2: First cross only, hold to close ──
    print("\n" + "=" * 80)
    print("V2: First VWAP cross of the day only, hold to 16:30 (no flip)")
    print("=" * 80)

    trades2 = []
    pos = 0; ep = 0; d = ""; cd = None
    prev_close = None; prev_vwap = None

    for i in range(n):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M")
        cl = row["close"]

        vwap_dict = vwap_by_date.get(td)
        vwap_val = get_vwap(vwap_dict, idx)

        if pos != 0 and hm >= FORCE_CLOSE:
            pnl = ((cl - ep) if d == "long" else (ep - cl)) * POINT_VALUE
            trades2.append({"pnl": round(pnl, 2), "date": str(td), "dir": d})
            pos = 0
            prev_close = cl; prev_vwap = vwap_val
            continue

        if hm >= FORCE_CLOSE or hm < SESSION_START:
            prev_close = cl; prev_vwap = vwap_val
            continue

        if td != cd:
            cd = td; prev_close = None; prev_vwap = None
            prev_close = cl; prev_vwap = vwap_val
            continue

        if vwap_val is None or prev_close is None or prev_vwap is None:
            prev_close = cl; prev_vwap = vwap_val
            continue

        if pos == 0 and hm <= LATEST_ENTRY:
            was_below = prev_close < prev_vwap
            was_above = prev_close > prev_vwap
            now_above = cl > vwap_val
            now_below = cl < vwap_val

            if was_below and now_above:
                pos = 1; d = "long"; ep = cl
            elif was_above and now_below:
                pos = -1; d = "short"; ep = cl

        prev_close = cl; prev_vwap = vwap_val

    print_results(trades2, is_cutoff, "V2 First cross")

    # ── Version 3: First cross + close on opposite cross ──
    print("\n" + "=" * 80)
    print("V3: Enter on first cross, EXIT on opposite cross (no flip to other side)")
    print("=" * 80)

    trades3 = []
    pos = 0; ep = 0; d = ""; cd = None
    prev_close = None; prev_vwap = None

    for i in range(n):
        row = df.iloc[i]; idx = df.index[i]; td = idx.date()
        hm = idx.strftime("%H:%M")
        cl = row["close"]

        vwap_dict = vwap_by_date.get(td)
        vwap_val = get_vwap(vwap_dict, idx)

        if pos != 0 and hm >= FORCE_CLOSE:
            pnl = ((cl - ep) if d == "long" else (ep - cl)) * POINT_VALUE
            trades3.append({"pnl": round(pnl, 2), "date": str(td), "dir": d, "reason": "CLOSE"})
            pos = 0
            prev_close = cl; prev_vwap = vwap_val
            continue

        if hm >= FORCE_CLOSE or hm < SESSION_START:
            prev_close = cl; prev_vwap = vwap_val
            continue

        if td != cd:
            cd = td; prev_close = None; prev_vwap = None
            prev_close = cl; prev_vwap = vwap_val
            continue

        if vwap_val is None or prev_close is None or prev_vwap is None:
            prev_close = cl; prev_vwap = vwap_val
            continue

        was_below = prev_close < prev_vwap
        was_above = prev_close > prev_vwap
        now_above = cl > vwap_val
        now_below = cl < vwap_val
        cross_up = was_below and now_above
        cross_down = was_above and now_below

        if pos == 1 and cross_down:
            pnl = (cl - ep) * POINT_VALUE
            trades3.append({"pnl": round(pnl, 2), "date": str(td), "dir": "long", "reason": "EXIT"})
            pos = 0
        elif pos == -1 and cross_up:
            pnl = (ep - cl) * POINT_VALUE
            trades3.append({"pnl": round(pnl, 2), "date": str(td), "dir": "short", "reason": "EXIT"})
            pos = 0

        if pos == 0 and hm <= LATEST_ENTRY:
            if cross_up:
                pos = 1; d = "long"; ep = cl
            elif cross_down:
                pos = -1; d = "short"; ep = cl

        prev_close = cl; prev_vwap = vwap_val

    print_results(trades3, is_cutoff, "V3 Exit on cross")


def print_results(trades, is_cutoff, label):
    if not trades:
        print("  No trades")
        return
    pnls = np.array([t["pnl"] for t in trades])
    wins = pnls[pnls > 0]; losses = pnls[pnls < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) else 0
    wr = 100 * len(wins) / len(pnls)
    cum = pnls.cumsum()
    dd = (cum - np.maximum.accumulate(cum)).min()
    is_t = [t for t in trades if t["date"] < is_cutoff]
    oos_t = [t for t in trades if t["date"] >= is_cutoff]
    is_pnls = np.array([t["pnl"] for t in is_t]) if is_t else np.array([0.0])
    oos_pnls = np.array([t["pnl"] for t in oos_t]) if oos_t else np.array([0.0])
    is_pf = is_pnls[is_pnls > 0].sum() / abs(is_pnls[is_pnls < 0].sum()) if (is_pnls < 0).any() else 0
    oos_pf = oos_pnls[oos_pnls > 0].sum() / abs(oos_pnls[oos_pnls < 0].sum()) if (oos_pnls < 0).any() else 0

    longs = sum(1 for t in trades if t["dir"] == "long")
    shorts = sum(1 for t in trades if t["dir"] == "short")

    print(f"  Trades: {len(trades)}  Longs: {longs}  Shorts: {shorts}")
    print(f"  WR: {wr:.1f}%  PF: {pf:.2f}  PnL: ${pnls.sum():+,.0f}  MaxDD: ${dd:,.0f}")
    print(f"  IS PF: {is_pf:.2f}  OOS PF: {oos_pf:.2f}  Avg PF: {(is_pf + oos_pf) / 2:.2f}")
    print(f"  Avg trade: ${pnls.mean():+,.0f}  Avg win: ${wins.mean():+,.0f}  Avg loss: ${losses.mean():+,.0f}")

    yearly = defaultdict(list)
    for t in trades:
        yearly[t["date"][:4]].append(t)
    print(f"  {'Year':>6} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'PnL':>11}")
    print(f"  " + "-" * 50)
    for yr in sorted(yearly.keys()):
        yr_t = yearly[yr]
        yr_p = np.array([t["pnl"] for t in yr_t])
        yr_pf = yr_p[yr_p > 0].sum() / abs(yr_p[yr_p < 0].sum()) if (yr_p < 0).any() else 0
        yr_wr = 100 * sum(1 for p in yr_p if p > 0) / len(yr_p)
        print(f"  {yr:>6} | {len(yr_t):>6} | {yr_wr:>5.1f} | {yr_pf:>5.2f} | ${yr_p.sum():>+10,.0f}")


if __name__ == "__main__":
    main()
