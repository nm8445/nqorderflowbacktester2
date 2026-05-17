"""Per-trade log for April 2026 using the locked filtered B2 entry +
fixed 20pt TP / 20pt SL exits. Output format matches b2_trade_log_dec2024.txt.

Entry: B2 X=0.75 N=15 D=70 strict, BAND_K=0.25 + conf_N=5 D=75 HALF
Filters: hours 9-14 ET only, drop POS-gamma SHORT entries (prior-day EOD gamma)
Exit:    fixed TP = entry + 20 pts, fixed SL = entry - 20 pts (mirrored for SHORT)
         Force close at 16:00 ET if neither hit.
Dedupe:  chained Mode 1 on actual exit times.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sweep_ratchet_sl_fixed_tp import filter_pre_dedupe

PARQUET_DIR  = Path(__file__).parent / "parquets"
TRADELOG_DIR = Path(__file__).parent.parent / "tradelogs"
MQ           = Path("D:/trading_pythonbacktest_data/menthorq_levels_nq.parquet")
VOL5M        = Path("D:/trading_pythonbacktest_data/volumetric_5min_1tpl.parquet")
OUT_TXT      = TRADELOG_DIR / "b2_trade_log_apr2026_locked20.txt"

ALLOWED_HOURS = {9, 10, 11, 12, 13, 14}
DROP_POS_SHORT = True
TP_PTS = 20.0
SL_PTS = 20.0
FORCE_CLOSE = dt.time(16, 0)


def load_5m_bars_for_dates(dates):
    df = pd.read_parquet(VOL5M, columns=["session_date", "bar_open_time",
                                          "open", "high", "low", "close"])
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    df = df[df["session_date"].isin(dates)]
    bot = pd.to_datetime(df["bar_open_time"], utc=True).dt.tz_convert("America/New_York")
    df["bar_open_time"] = bot
    rth_mask = ((bot.dt.time >= dt.time(9, 30)) & (bot.dt.time <= dt.time(17, 0)))
    df = df[rth_mask & df["open"].notna()].copy()
    df = df.drop_duplicates(subset=["session_date", "bar_open_time"]).reset_index(drop=True)
    df = df.sort_values(["session_date", "bar_open_time"])
    return df


def label_level(mq_row, level_price, tol=0.5):
    if mq_row is None or (hasattr(mq_row, "empty") and mq_row.empty):
        return "no-mq-data"
    candidates = []
    for col in mq_row.index:
        if not col.endswith("_nq"):
            continue
        v = mq_row[col]
        if pd.isna(v):
            continue
        if abs(v - level_price) <= tol:
            candidates.append((col, v))
    if candidates:
        return candidates[0][0].replace("_nq", "")
    best = None; best_d = 1e9
    for col in mq_row.index:
        if not col.endswith("_nq"):
            continue
        v = mq_row[col]
        if pd.isna(v):
            continue
        d = abs(v - level_price)
        if d < best_d:
            best_d = d; best = col
    return f"{best.replace('_nq','')}_~{best_d:.1f}pt_off" if best else "unknown"


def attach_gamma(cands, mq, prior_mq):
    cands = cands.copy()
    cands["entry_date"] = pd.to_datetime(cands["entry_time_et"]).dt.date
    col = "qqq_gamma_sign" if "qqq_gamma_sign" in mq.columns else None
    if col is None:
        cands["gamma_sign"] = np.nan
        return cands
    g = {}
    for d in cands["entry_date"].unique():
        p = prior_mq(d)
        g[d] = mq.loc[p, col] if p in mq.index else np.nan
    cands["gamma_sign"] = cands["entry_date"].map(g)
    return cands


def find_exit_fixed(direction, entry_ts, entry_price, day_bars, tp_pts, sl_pts):
    """Walk forward in 5-min bars to find TP/SL hit or force-close. Returns dict."""
    sign = 1 if direction == "LONG" else -1
    tp_price = entry_price + sign * tp_pts
    sl_price = entry_price - sign * sl_pts
    after = day_bars[day_bars["bar_open_time"] > entry_ts].reset_index(drop=True)
    if after.empty:
        return {"outcome": "no-data", "tp_hit_time": None, "sl_hit_time": None,
                "result_pts": 0.0, "exit_time": entry_ts,
                "tp_price": tp_price, "sl_price": sl_price}
    for i in range(len(after)):
        b = after.iloc[i]
        bt = b["bar_open_time"]
        # Force-close at 16:00 if reached (use prior close)
        if bt.time() >= FORCE_CLOSE:
            close_px = float(after.iloc[max(i - 1, 0)]["close"]) if i > 0 else float(b["open"])
            return {"outcome": "held->close", "tp_hit_time": None, "sl_hit_time": None,
                    "result_pts": sign * (close_px - entry_price),
                    "exit_time": bt, "tp_price": tp_price, "sl_price": sl_price}
        hi = float(b["high"]); lo = float(b["low"])
        if sign > 0:
            tp_hit = hi >= tp_price
            sl_hit = lo <= sl_price
        else:
            tp_hit = lo <= tp_price
            sl_hit = hi >= sl_price
        if tp_hit and sl_hit:
            return {"outcome": "SL hit", "tp_hit_time": None, "sl_hit_time": bt,
                    "result_pts": -sl_pts, "exit_time": bt,
                    "tp_price": tp_price, "sl_price": sl_price}
        if tp_hit:
            return {"outcome": "TP hit", "tp_hit_time": bt, "sl_hit_time": None,
                    "result_pts": +tp_pts, "exit_time": bt,
                    "tp_price": tp_price, "sl_price": sl_price}
        if sl_hit:
            return {"outcome": "SL hit", "tp_hit_time": None, "sl_hit_time": bt,
                    "result_pts": -sl_pts, "exit_time": bt,
                    "tp_price": tp_price, "sl_price": sl_price}
    last = after.iloc[-1]
    return {"outcome": "held->close", "tp_hit_time": None, "sl_hit_time": None,
            "result_pts": sign * (float(last["close"]) - entry_price),
            "exit_time": last["bar_open_time"], "tp_price": tp_price, "sl_price": sl_price}


def fmt_time(t):
    if t is None:
        return "-"
    try:
        if pd.isna(t):
            return "-"
    except (TypeError, ValueError):
        pass
    if hasattr(t, "strftime"):
        return t.strftime("%m-%d %H:%M")
    return str(t)


def main():
    print("loading OOS trades parquet...")
    oos = pd.read_parquet(PARQUET_DIR / "entry_signal_trades_oos.parquet")
    oos["date"] = pd.to_datetime(oos["date"])
    apr = oos[(oos["date"] >= "2026-04-01") & (oos["date"] < "2026-05-01")].copy()
    print(f"  April 2026 raw rows: {len(apr)}")
    if apr.empty:
        print("  no rows for April 2026 — exiting.")
        return

    print("applying B2 X=0.75 N=15 D=70 strict + conf_N=5 D=75 HALF + chained Mode 1...")
    cands = filter_pre_dedupe(apr)
    print(f"  after entry filter + conf: {len(cands)}")

    print("loading MenthorQ and attaching prior-day gamma_sign...")
    mq = pd.read_parquet(MQ)
    mq["date"] = pd.to_datetime(mq["date"]).dt.date
    mq = mq.set_index("date").sort_index()
    mq_dates_sorted = sorted(mq.index.tolist())
    def prior_mq_day(session_date):
        prev = None
        for md in mq_dates_sorted:
            if md < session_date: prev = md
            else: break
        return prev
    cands = attach_gamma(cands, mq, prior_mq_day)

    print(f"applying hour 9-14 filter + drop POS-gamma SHORT...")
    cands["entry_hour"] = pd.to_datetime(cands["entry_time_et"]).dt.hour
    keep = cands["entry_hour"].isin(ALLOWED_HOURS)
    if DROP_POS_SHORT:
        keep &= ~((cands["gamma_sign"] == 1) & (cands["direction"] == "SHORT"))
    cands = cands[keep].sort_values("entry_time_et").reset_index(drop=True)
    print(f"  kept after filters: {len(cands)}")

    if cands.empty:
        print("  no trades after filters for April 2026.")
        return

    print("loading 5-min bars for April 2026 trading days...")
    days = sorted(set(pd.to_datetime(cands["entry_time_et"]).dt.date.tolist()))
    bars = load_5m_bars_for_dates(days)
    bars_by_day = {d: g.reset_index(drop=True) for d, g in bars.groupby("session_date")}

    # Compute exit per trade and chained-Mode-1 dedupe by actual exit time.
    print("simulating fixed 20pt TP / 20pt SL exits + chained dedupe...")
    last_exit = pd.Timestamp(0, tz="America/New_York")
    kept_rows = []
    for _, t in cands.iterrows():
        ent_t = pd.Timestamp(t["entry_time_et"])
        d = ent_t.date()
        day_bars = bars_by_day.get(d)
        if day_bars is None:
            continue
        if ent_t <= last_exit:
            continue
        ex = find_exit_fixed(t["direction"], ent_t, float(t["entry_price"]),
                             day_bars, TP_PTS, SL_PTS)
        kept_rows.append({**t.to_dict(), **ex})
        last_exit = pd.Timestamp(ex["exit_time"])

    trades = pd.DataFrame(kept_rows)
    print(f"  final trades after chained dedupe: {len(trades)}")
    if trades.empty:
        return

    # Build per-trade log
    L = []
    width = 280
    L.append(f"loading trades parquet...")
    L.append(f"  Apr 2026 raw rows: {len(apr)}")
    L.append(f"  After B2 X=0.75 N=15 D=70 strict (BAND_K=0.25) + conf_N=5 D=75 HALF + filters (hours 9-14 ET, drop POS+SHORT)")
    L.append(f"  + chained Mode 1 by 20pt TP/20pt SL exit times: {len(trades)} trades")
    L.append("")
    L.append("=" * width)
    header = (f"{'Date':<10} {'OHI':>8} {'OLO':>8} | "
              f"{'Signal Time':<11} {'SigO':>9} {'SigH':>9} {'SigL':>9} {'SigC':>9} | "
              f"{'Conf Time':<11} {'ConfO':>9} {'ConfH':>9} {'ConfL':>9} {'ConfC':>9} | "
              f"{'Entry Time':<11} {'Entry':>9} {'Dir':>5} | "
              f"{'Level':<12} {'LvlPrice':>9} | "
              f"{'TP @':>9} {'SL @':>9} | "
              f"{'TP Hit':<11} {'SL Hit':<11} {'Outcome':<14} {'Result(pts)':>11}")
    L.append(header)
    L.append("-" * width)

    total_pnl = 0.0
    n_tp = n_sl = n_held = 0
    for _, t in trades.iterrows():
        ent_t = pd.Timestamp(t["entry_time_et"])
        sig_t = pd.Timestamp(t["signal_time"])
        d = ent_t.date()
        conf_t = sig_t + pd.Timedelta(minutes=5)
        day_bars = bars_by_day.get(d)
        conf_row = day_bars[day_bars["bar_open_time"] == conf_t] if day_bars is not None else None
        if conf_row is not None and not conf_row.empty:
            conf_o = float(conf_row["open"].iloc[0])
            conf_h = float(conf_row["high"].iloc[0])
            conf_l = float(conf_row["low"].iloc[0])
            conf_c = float(conf_row["close"].iloc[0])
        else:
            conf_o = conf_h = conf_l = conf_c = float("nan")

        prior_d = prior_mq_day(d)
        mq_row = mq.loc[prior_d] if prior_d in mq.index else None
        near_level = t.get("near_level", np.nan)
        level_name = label_level(mq_row, near_level) if mq_row is not None and not pd.isna(near_level) else "no-level"

        if t["outcome"] == "TP hit":
            n_tp += 1
        elif t["outcome"] == "SL hit":
            n_sl += 1
        else:
            n_held += 1
        total_pnl += t["result_pts"]

        L.append(
            f"{d!s:<10} {t['ohi']:>8.2f} {t['olo']:>8.2f} | "
            f"{sig_t.strftime('%m-%d %H:%M'):<11} "
            f"{t['signal_open']:>9.2f} {t['signal_high']:>9.2f} {t['signal_low']:>9.2f} {t['signal_close']:>9.2f} | "
            f"{fmt_time(conf_t):<11} "
            f"{conf_o:>9.2f} {conf_h:>9.2f} {conf_l:>9.2f} {conf_c:>9.2f} | "
            f"{ent_t.strftime('%m-%d %H:%M'):<11} {float(t['entry_price']):>9.2f} {t['direction']:>5} | "
            f"{level_name:<12} {float(near_level):>9.2f} | "
            f"{t['tp_price']:>9.2f} {t['sl_price']:>9.2f} | "
            f"{fmt_time(t['tp_hit_time']):<11} {fmt_time(t['sl_hit_time']):<11} "
            f"{t['outcome']:<14} {t['result_pts']:>+11.2f}"
        )

    L.append("-" * width)
    L.append(f"Apr 2026 total: n={len(trades)}  "
             f"TP={n_tp}  SL={n_sl}  held->close={n_held}  "
             f"realized PnL = {total_pnl:+.2f} pts "
             f"(${total_pnl*20:+.0f} per NQ, ${total_pnl*2:+.0f} per MNQ)")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT_TXT}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
