"""
Overnight Drift - Reconstructed Bands v4 (Martingale + BE)

Python port of the Pine Script v5 strategy. 20-minute NQ bars, anchored to
midnight ET so 19:00 ET (entry) and 08:00 ET (force close) land on bar
boundaries.

Pine semantics carried over:
- process_orders_on_close=true: strategy.entry / strategy.close fill at the
  bar's close. strategy.exit (BE stop) is a real stop that triggers intrabar.
- ATR uses RMA (Pine ta.atr).
- Yellow band uses drift_floor: max(prev_yellow + drift, close - k*ATR).
- Green target tightens with bars_in_trade: red + base - decay*bars + k*ATR.
- BE arms once when prev_close > prev_yellow AND current low <= prev_yellow,
  with bars_in_trade >= 1. From the next bar onward, a stop sits at entry.
- Martingale: state 0 -> base; state 1 (after a loss) -> loss size; state 2
  (forced base after the follow-up trade) -> base again, regardless of result.

NQ point value: $20 / point per contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

NQ_POINT_VALUE = 20.0  # $/point/contract


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class StrategyParams:
    # Schedule (ET local times)
    entry_hour: int = 19
    entry_minute: int = 0
    forced_hour: int = 8
    forced_minute: int = 0

    # Yellow (stop) — DEFAULTS UPDATED 2026-05-25 to new locked production config
    yellow_atr_len: int = 14
    yellow_atr_mult: float = 1.40           # was 1.42; new locked = 1.40
    yellow_drift: float = 0.0               # was 1.0; locked uses pure_ratchet
    yellow_mode: str = "pure_ratchet"       # was drift_floor; locked = pure_ratchet
    # bearish_drift: like pure_ratchet but yellow can drift DOWN by bearish_drift_pts
    # on bars where close < open. Cannot drop below raw_atr.
    bearish_drift_pts: float = 0.0

    # Green (target) — UPDATED to new locked production config
    green_atr_len: int = 14                 # was 13; locked = 14
    green_atr_mult: float = 1.50            # was 2.60; new locked = 1.50
    green_base: float = 160.0               # was 107.6; new locked = 160.0
    green_decay: float = 2.00               # was 1.31; new locked = 2.00

    # Red baseline
    red_intercept: float = 0.0
    red_drift: float = 0.45

    # Breakeven rule
    use_be: bool = False                    # was True; locked = OFF

    # TP fill mode: False = bar close (Pine process_orders_on_close); True = at green_val (intrabar)
    tp_intrabar_fill: bool = False

    # Suppress yellow (SL) for the first N bars after entry (still allows TP and force-close).
    # 0 = original locked behavior. 30 = new locked (2026-05-25 sweep-validated).
    yellow_suppress_bars: int = 30          # was 0; new locked = 30

    # Martingale
    use_martingale: bool = True
    base_qty: int = 1
    loss_qty: int = 2


# ---------------------------------------------------------------------------
# ATR (Pine RMA)
# ---------------------------------------------------------------------------


def rma_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Pine ta.atr equivalent: RMA of true range with seed = SMA over first `length` bars."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = pd.Series(np.nan, index=close.index, dtype=float)
    if len(tr) < length:
        return atr
    seed = tr.iloc[:length].mean()
    atr.iloc[length - 1] = seed
    alpha = 1.0 / length
    prev = seed
    tr_vals = tr.values
    out = atr.values
    for i in range(length, len(tr)):
        cur = tr_vals[i]
        if np.isnan(cur):
            out[i] = prev
            continue
        prev = (1 - alpha) * prev + alpha * cur
        out[i] = prev
    return atr


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: int
    reason: str
    bars_held: int
    be_arm_time: pd.Timestamp | None = None
    be_arm_bar: int | None = None  # bars-from-entry when BE was armed
    pnl_points: float = field(init=False)
    pnl_dollars: float = field(init=False)

    def __post_init__(self) -> None:
        self.pnl_points = self.exit_price - self.entry_price
        self.pnl_dollars = self.pnl_points * NQ_POINT_VALUE * self.qty


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


def run_backtest(bars: pd.DataFrame, params: StrategyParams = StrategyParams()) -> list[Trade]:
    """
    `bars` must be a DataFrame indexed by tz-aware ET timestamp with columns
    ['open', 'high', 'low', 'close']. Index must be 20-min bar starts.
    """
    if bars.index.tz is None:
        raise ValueError("bars index must be tz-aware (America/New_York)")

    atr_y = rma_atr(bars["high"], bars["low"], bars["close"], params.yellow_atr_len)
    atr_g = rma_atr(bars["high"], bars["low"], bars["close"], params.green_atr_len)

    entry_t = time(params.entry_hour, params.entry_minute)
    force_t = time(params.forced_hour, params.forced_minute)

    open_a = bars["open"].values
    high_a = bars["high"].values
    low_a = bars["low"].values
    close_a = bars["close"].values
    atr_y_a = atr_y.values
    atr_g_a = atr_g.values
    idx = bars.index

    trades: list[Trade] = []

    in_pos = False
    entry_price = np.nan
    entry_idx = -1
    entry_qty = 0
    yellow_val = np.nan
    prev_yellow = np.nan
    prev_close = np.nan
    be_active = False  # set to True when the BE stop is armed; takes effect next bar
    be_arm_time: pd.Timestamp | None = None
    be_arm_bar: int | None = None

    # Martingale state machine
    marti_state = 0
    next_qty = params.base_qty

    for i in range(len(bars)):
        ts = idx[i]
        local_t = ts.time()
        o = open_a[i]
        h = high_a[i]
        l = low_a[i]
        c = close_a[i]
        ay = atr_y_a[i]
        ag = atr_g_a[i]

        # ---------- ENTRY ----------
        if (
            not in_pos
            and local_t == entry_t
            and not np.isnan(ay)
            and not np.isnan(c)
        ):
            if params.use_martingale:
                if marti_state == 0:
                    next_qty = params.base_qty
                elif marti_state == 1:
                    next_qty = params.loss_qty
                else:
                    next_qty = params.base_qty
            else:
                next_qty = params.base_qty

            in_pos = True
            entry_price = c
            entry_idx = i
            entry_qty = int(next_qty)
            yellow_val = c - params.yellow_atr_mult * ay
            prev_yellow = np.nan
            prev_close = c
            be_active = False
            be_arm_time = None
            be_arm_bar = None
            continue

        # ---------- IN-TRADE LOGIC ----------
        if in_pos:
            bars_in_trade = i - entry_idx

            # Bands for this bar
            raw_yellow = c - params.yellow_atr_mult * ay if not np.isnan(ay) else np.nan
            if params.yellow_mode == "pure_ratchet":
                cand = max(prev_yellow if not np.isnan(prev_yellow) else raw_yellow, raw_yellow) \
                    if not np.isnan(raw_yellow) else np.nan
                yellow_val = cand
            elif params.yellow_mode == "drift_floor":
                base_floor = (prev_yellow + params.yellow_drift) if not np.isnan(prev_yellow) else (raw_yellow if not np.isnan(raw_yellow) else np.nan)
                if not np.isnan(raw_yellow) and not np.isnan(base_floor):
                    yellow_val = max(base_floor, raw_yellow)
                else:
                    yellow_val = raw_yellow if not np.isnan(raw_yellow) else base_floor
            elif params.yellow_mode == "bearish_drift":
                # Like pure_ratchet but allow downward drift on bearish candles.
                # Never below raw_yellow (so still respects ATR-based floor on big moves up).
                if np.isnan(prev_yellow):
                    yellow_val = raw_yellow
                else:
                    is_bearish = c < o
                    if is_bearish:
                        drifted = prev_yellow - params.bearish_drift_pts
                        # don't drift below raw_yellow if price rallied (raw_yellow tracks close-k*ATR)
                        yellow_val = max(drifted, raw_yellow) if not np.isnan(raw_yellow) else drifted
                    else:
                        yellow_val = max(prev_yellow, raw_yellow) if not np.isnan(raw_yellow) else prev_yellow
            else:  # raw_atr
                yellow_val = raw_yellow

            red_val = entry_price + params.red_intercept + params.red_drift * bars_in_trade
            green_val = (
                red_val
                + params.green_base
                - params.green_decay * bars_in_trade
                + (params.green_atr_mult * ag if not np.isnan(ag) else 0.0)
            )

            exited = False
            exit_price = np.nan
            exit_reason = ""

            # 1. BE stop -- only if armed BEFORE this bar (be_active was set on a prior bar
            #    and the order is live this bar). A sell-stop at entry_price fills:
            #      - at the bar's open if the bar gaps THROUGH the stop (open <= entry)
            #      - at entry_price if the bar opens above and trades down through it
            if params.use_be and be_active:
                if o <= entry_price:
                    exit_price = o  # gap-through: stop fires at the open
                    exit_reason = "BE Stop"
                    exited = True
                elif l <= entry_price:
                    exit_price = entry_price  # intrabar touch at the stop level
                    exit_reason = "BE Stop"
                    exited = True

            # 2. TP green (Pine: fills at bar close; intrabar mode: fills at green_val)
            if not exited and not np.isnan(green_val) and h >= green_val:
                exit_price = green_val if params.tp_intrabar_fill else c
                exit_reason = "TP Green"
                exited = True

            # 3. SL yellow on close (bearish close at/below yellow)
            # Suppressed for the first `yellow_suppress_bars` bars after entry.
            if (not exited and not np.isnan(yellow_val) and c <= yellow_val and c < o
                    and bars_in_trade >= params.yellow_suppress_bars):
                exit_price = c
                exit_reason = "SL Yellow"
                exited = True

            # 4. Force close at scheduled time
            if not exited and local_t == force_t:
                exit_price = c
                exit_reason = "Force Close"
                exited = True

            if exited:
                trades.append(
                    Trade(
                        entry_time=idx[entry_idx],
                        exit_time=ts,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        qty=entry_qty,
                        reason=exit_reason,
                        bars_held=bars_in_trade,
                        be_arm_time=be_arm_time,
                        be_arm_bar=be_arm_bar,
                    )
                )
                # Update martingale state
                last_was_loss = (exit_price - entry_price) < 0
                if marti_state == 0:
                    marti_state = 1 if last_was_loss else 0
                elif marti_state == 1:
                    marti_state = 2
                else:
                    marti_state = 1 if last_was_loss else 0

                in_pos = False
                entry_price = np.nan
                entry_idx = -1
                entry_qty = 0
                yellow_val = np.nan
                prev_yellow = np.nan
                prev_close = np.nan
                be_active = False
                be_arm_time = None
                be_arm_bar = None
                continue

            # ---------- BE ARMING ----------
            # Arm BE if previous bar's close was above prev yellow and current low touched it.
            # Only arms once. Activates from NEXT bar (Pine: order placed during this bar
            # becomes live; with process_orders_on_close, the actual stop-trigger is on
            # subsequent bars).
            if (
                params.use_be
                and not be_active
                and bars_in_trade >= 1
                and not np.isnan(prev_yellow)
                and prev_close > prev_yellow
                and l <= prev_yellow
            ):
                be_active = True
                be_arm_time = ts
                be_arm_bar = bars_in_trade

            # Roll prev_* for next bar
            prev_yellow = yellow_val
            prev_close = c

    return trades


# ---------------------------------------------------------------------------
# Bar building
# ---------------------------------------------------------------------------


def _resample_to_20min(bars: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    """Resample finer-grained bars to 20-minute bars anchored to midnight ET."""
    if bars.index.tz is None:
        bars = bars.tz_localize("UTC")
    et = bars.tz_convert(tz)
    agg = et.resample("20min", origin="start_day", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    return agg.dropna()


def load_1min_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[["open", "high", "low", "close"]].copy()


def load_5min_pickles(folder: str, after_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load daily 5-min pickle bars and stack into a single DataFrame (UTC)."""
    folder_p = Path(folder)
    pkls = sorted(folder_p.glob("timebars_5min_*.pkl"))
    rows: list[dict] = []
    import pickle

    for p in pkls:
        date_part = p.stem.replace("timebars_5min_", "").replace("_", "-")
        ts_date = pd.Timestamp(date_part)
        if after_date is not None and ts_date <= after_date:
            continue
        with open(p, "rb") as fh:
            day = pickle.load(fh)
        rows.extend(day)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"]).set_index(
            pd.DatetimeIndex([], tz="UTC")
        )
    df = pd.DataFrame(rows)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.set_index("open_time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["open", "high", "low", "close"]]


def build_full_20min_series(
    parquet_path: str,
    pickle_folder: str,
) -> pd.DataFrame:
    """Combine 1-min parquet (long history) + 5-min daily pickles (recent) -> 20-min ET bars."""
    df1m = load_1min_parquet(parquet_path)
    cutoff = df1m.index.max()
    df5m_extra = load_5min_pickles(pickle_folder, after_date=cutoff.tz_convert(None).normalize())

    bars20_a = _resample_to_20min(df1m)
    if not df5m_extra.empty:
        bars20_b = _resample_to_20min(df5m_extra)
        bars20 = pd.concat([bars20_a, bars20_b])
        bars20 = bars20[~bars20.index.duplicated(keep="last")].sort_index()
    else:
        bars20 = bars20_a
    return bars20


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def trades_to_df(trades: Iterable[Trade]) -> pd.DataFrame:
    rows = [
        {
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "qty": t.qty,
            "reason": t.reason,
            "bars_held": t.bars_held,
            "be_arm_time": t.be_arm_time,
            "be_arm_bar": t.be_arm_bar,
            "pnl_points": t.pnl_points,
            "pnl_dollars": t.pnl_dollars,
        }
        for t in trades
    ]
    return pd.DataFrame(rows)


def yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["year"] = df["entry_time"].dt.year
    grp = df.groupby("year")
    out = pd.DataFrame(
        {
            "trades": grp.size(),
            "wins": grp.apply(lambda g: (g["pnl_dollars"] > 0).sum(), include_groups=False),
            "losses": grp.apply(lambda g: (g["pnl_dollars"] < 0).sum(), include_groups=False),
            "win_rate": grp.apply(
                lambda g: (g["pnl_dollars"] > 0).mean() * 100, include_groups=False
            ),
            "gross_pnl_$": grp["pnl_dollars"].sum(),
            "avg_pnl_$": grp["pnl_dollars"].mean(),
            "best_$": grp["pnl_dollars"].max(),
            "worst_$": grp["pnl_dollars"].min(),
            "avg_pts": grp["pnl_points"].mean(),
        }
    )
    out["profit_factor"] = grp.apply(
        lambda g: (
            g.loc[g["pnl_dollars"] > 0, "pnl_dollars"].sum()
            / abs(g.loc[g["pnl_dollars"] < 0, "pnl_dollars"].sum())
            if (g["pnl_dollars"] < 0).any()
            else np.inf
        ),
        include_groups=False,
    )
    return out
