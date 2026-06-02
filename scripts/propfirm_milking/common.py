"""Shared infrastructure for prop-firm-milking fixed-ATR-bracket optimization.

All four strategies (OD/RV/B2/Fabio) get their native exits replaced with a pure
ATR bracket:   SL = entry - sign*sl_mult*ATR   TP = entry + sign*tp_mult*ATR
detected by tick/minute first-touch (SL-first tie-break), sized so each trade
risks ~$1k via MNQ micros (capped 40), with commission + slippage costs.

This module is data-source + math only. Strategy-specific entry feeds live in
entries.py; the sweep driver lives in sweep_exits.py.

Decisions (locked with user):
  - MNQ micro: $2/point. NQ tick 0.25 = $0.50/tick/MNQ.
  - Risk target ~$1000/trade: n = round(1000 / (sl_pts * $2)), clamp [1, 40].
  - Costs: $2/RT/MNQ commission + 2 ticks/side slippage (=$2/MNQ RT) -> $4/MNQ RT.
  - IS/OOS: chronological, oldest 60% in-sample, newest 40% out-of-sample.
  - Native-timeframe ATR (Wilder/RMA via ewm), length swept. 20-min for OD/RV/B2,
    5-min for Fabio.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

# ============ Paths ============
ONE_MIN_PARQUET = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
REPO = Path(__file__).resolve().parents[2]
FOURWAY_TRADES = REPO / "scripts" / "rough vol orderflow" / "results" / "combined_4way_trades.csv"
FUNDED_MAE_CSV = REPO / "live" / "combined deployment plan" / "combined_trades_with_mae.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

ET = "America/New_York"

# ============ Instrument / costs ============
NQ_POINT_VALUE = 20.0          # $/pt per 1 NQ
MNQ_POINT_VALUE = 2.0          # $/pt per 1 MNQ
TICK = 0.25
TICK_VALUE_MNQ = 0.50          # $/tick per 1 MNQ
RISK_TARGET = 1000.0           # target $ risk per trade
MAX_MNQ = 40                   # firm cap
COMMISSION_RT_PER_MNQ = 2.0    # $/round-turn/MNQ
SLIP_TICKS_PER_SIDE = 2        # 2 ticks/side
SLIP_RT_PER_MNQ = SLIP_TICKS_PER_SIDE * TICK_VALUE_MNQ * 2  # $2/MNQ round-turn
COST_RT_PER_MNQ = COMMISSION_RT_PER_MNQ + SLIP_RT_PER_MNQ   # $4/MNQ round-turn

IS_FRAC = 0.60

# ============ ATR-length grid + bracket grid (defaults) ============
ATR_LEN_GRID = [7, 10, 14, 20, 28]
SL_MULT_GRID = [round(x, 2) for x in np.arange(0.5, 3.001, 0.25)]   # 0.50..3.00
TP_MULT_GRID = [round(x, 2) for x in np.arange(0.75, 4.001, 0.25)]  # 0.75..4.00


# ============ Bars ============
_BARS_1M_CACHE = None


def load_1min_bars() -> pd.DataFrame:
    """Load the 1-min bar parquet, return ET-localized index, OHLC[+volume]."""
    global _BARS_1M_CACHE
    if _BARS_1M_CACHE is not None:
        return _BARS_1M_CACHE
    df = pd.read_parquet(ONE_MIN_PARQUET)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    df = df[["open", "high", "low", "close"]].sort_index()
    _BARS_1M_CACHE = df
    return df


def build_bars(df1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample 1-min ET bars to `minutes`-min bars. Index = bar CLOSE time (right edge).
    Adds `open_time` (left edge = close - minutes)."""
    rule = f"{minutes}min"
    bars = df1m.resample(rule, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna(subset=["open"])
    bars["open_time"] = bars.index - pd.Timedelta(minutes=minutes)
    return bars


def wilder_atr(bars: pd.DataFrame, n: int) -> pd.Series:
    """Wilder/RMA ATR via ewm(alpha=1/n). Matches the user's existing sweep template."""
    prev_close = bars["close"].shift(1)
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


# ============ Sizing / costs ============
def mnq_size(sl_pts: float) -> int:
    """Contracts to risk ~$1k given stop distance in points; clamp [1, MAX_MNQ]."""
    if sl_pts <= 0:
        return 1
    n = int(round(RISK_TARGET / (sl_pts * MNQ_POINT_VALUE)))
    return int(np.clip(n, 1, MAX_MNQ))


def trade_cost(n: int) -> float:
    return n * COST_RT_PER_MNQ


# ============ Fixed-bracket first-touch ============
@dataclass
class EntryPack:
    """Precomputed per-entry data — built ONCE, reused across the sl/tp/atr_len sweep."""
    strat: str
    fill_time: pd.Timestamp
    date: object               # python date (calendar day, ET) for IS/OOS + daily grouping
    sign: int                  # +1 long, -1 short
    entry_price: float
    atr_by_len: dict           # {atr_len: atr_value_at_entry}
    fwd_high: np.ndarray       # 1-min highs from fill -> force-close (exclusive of fill bar)
    fwd_low: np.ndarray
    fc_close: float            # close at force-close (fallback exit price)


def first_touch(pack: EntryPack, sl_dist: float, tp_dist: float):
    """Resolve first-touch outcome for one entry given SL/TP distances (in points).

    Returns (exit_price, reason, is_win). reason in {'TP','SL','FC'}.
    SL-first on same-bar ambiguity (conservative). `dist` are positive point distances.
    """
    sign = pack.sign
    entry = pack.entry_price
    sl_level = entry - sign * sl_dist
    tp_level = entry + sign * tp_dist
    hi = pack.fwd_high
    lo = pack.fwd_low
    n = hi.shape[0]
    if n == 0:
        return pack.fc_close, "FC", (sign * (pack.fc_close - entry) > 0)

    if sign > 0:
        sl_hits = lo <= sl_level
        tp_hits = hi >= tp_level
    else:
        sl_hits = hi >= sl_level
        tp_hits = lo <= tp_level

    sl_any = sl_hits.any()
    tp_any = tp_hits.any()
    sl_i = int(np.argmax(sl_hits)) if sl_any else n + 1
    tp_i = int(np.argmax(tp_hits)) if tp_any else n + 1

    if not sl_any and not tp_any:
        return pack.fc_close, "FC", (sign * (pack.fc_close - entry) > 0)
    # SL-first tie-break: SL wins ties (sl_i <= tp_i)
    if sl_i <= tp_i:
        return sl_level, "SL", False
    return tp_level, "TP", True


# ============ Sweep evaluation over an entry list ============
def eval_bracket(packs: list[EntryPack], atr_len: int, sl_mult: float, tp_mult: float) -> pd.DataFrame:
    """Apply a fixed ATR bracket to every entry pack. Returns per-trade DataFrame."""
    rows = []
    for p in packs:
        atr = p.atr_by_len.get(atr_len)
        if atr is None or not np.isfinite(atr) or atr <= 0:
            continue
        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr
        exit_price, reason, is_win = first_touch(p, sl_dist, tp_dist)
        n = mnq_size(sl_dist)
        pnl_pts = p.sign * (exit_price - p.entry_price)
        pnl_usd = pnl_pts * MNQ_POINT_VALUE * n - trade_cost(n)
        rows.append((p.date, p.fill_time, p.sign, reason, is_win, n,
                     sl_dist, pnl_pts, pnl_usd))
    cols = ["date", "fill_time", "sign", "reason", "win", "mnq",
            "sl_dist_pts", "pnl_pts", "pnl_$"]
    return pd.DataFrame(rows, columns=cols)


# ============ IS/OOS + stats ============
def is_oos_cutoff(dates) -> object:
    ud = sorted(set(dates))
    if len(ud) <= 1:
        return ud[-1] if ud else None
    return ud[int(len(ud) * IS_FRAC)]


def stats_block(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return dict(n=0, wr=0.0, net=0.0, pf=0.0, mdd=0.0,
                    tp=0, sl=0, fc=0, tagged_frac=0.0, mean_risk=0.0, mean_mnq=0.0)
    pnl = trades["pnl_$"].values
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 99.0
    cum = np.cumsum(pnl)
    mdd = float((cum - np.maximum.accumulate(cum)).min()) if n else 0.0
    rc = trades["reason"].value_counts()
    n_tp = int(rc.get("TP", 0)); n_sl = int(rc.get("SL", 0)); n_fc = int(rc.get("FC", 0))
    mean_risk = float((trades["sl_dist_pts"] * MNQ_POINT_VALUE * trades["mnq"]).mean())
    return dict(
        n=n, wr=round(float((pnl > 0).mean()) * 100, 1),
        net=round(float(pnl.sum()), 0), pf=round(float(pf), 3), mdd=round(mdd, 0),
        tp=n_tp, sl=n_sl, fc=n_fc,
        tagged_frac=round((n_tp + n_sl) / n, 3),
        mean_risk=round(mean_risk, 0),
        mean_mnq=round(float(trades["mnq"].mean()), 1),
    )
