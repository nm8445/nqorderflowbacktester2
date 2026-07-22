"""1-min RV backtest for the pass-rate study.

Session: 19:00 ET (Asia open) -> 16:00 ET next day.  Hard 16:00 ET force-close (flatten).
Bracket: SL = TP = k*ATR (1:1), sized so risk = $1500.  PnL recorded in R units and in $:
  TP -> +1R (+$1500), SL -> -1R (-$1500), force-close -> fractional R = move/(k*ATR).
Per trade also records mae_R / mae_$ = worst floating excursion (>= -1R; capped at the SL),
computed from the 1-min bars over the hold -> feeds the eval-MC floating-blow check.

Roll-seam / session-open suppression: no entry on the first in-session bar after a >1min gap.
Conservative intrabar fill: if SL and TP both inside a bar, SL fills first.
Features are precomputed (features.compute_features); orderflow flags optional (data.build_orderflow_flags).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time

import numpy as np
import pandas as pd

RISK_DOLLARS = 1500.0
SESSION_START = time(19, 0)   # 19:00 ET
SESSION_END = time(16, 0)     # 16:00 ET (force-close)


def _in_session(tmin: np.ndarray) -> np.ndarray:
    """tmin = minutes-since-midnight ET of the bar CLOSE. Session = [19:00, 24:00) U [00:00, 16:00)."""
    return (tmin >= 19 * 60) | (tmin < 16 * 60)


def _force_close_window(tmin: np.ndarray) -> np.ndarray:
    """[16:00, 19:00) ET — flatten any open position on the first such bar."""
    return (tmin >= 16 * 60) & (tmin < 19 * 60)


@dataclass
class BTParams:
    high_z: float = 2.0
    k: float = 2.0           # ATR multiple, SL = TP = k*ATR
    atr_max: float = 150.0
    use_orderflow: bool = True
    # direction rule for the entry (edge-existence probe):
    #   ema_cont  : long if close>ema  (current live behavior)   ema_revert: long if close<ema
    #   mom_cont  : long if close>open (bar momentum)            mom_fade  : long if close<open
    direction_mode: str = "ema_cont"


def run(bars: pd.DataFrame, feat: dict[str, np.ndarray], params: BTParams,
        long_ok: np.ndarray | None = None, short_ok: np.ndarray | None = None) -> pd.DataFrame:
    """bars: UTC-indexed OHLC (+ 'et', 'session_break'). feat: z_vol/ema/atr arrays.
    long_ok/short_ok: per-bar bool arrays (orderflow). Returns a trades DataFrame."""
    close = bars["close"].to_numpy(np.float64)
    open_ = bars["open"].to_numpy(np.float64)
    high = bars["high"].to_numpy(np.float64)
    low = bars["low"].to_numpy(np.float64)
    et = bars["et"]
    tmin = (et.dt.hour * 60 + et.dt.minute).to_numpy()
    sess_break = bars["session_break"].to_numpy()
    et_date = et.dt.normalize()  # ET calendar day (session bucket handled in the MC via session_date)

    z = feat["z_vol"]; ema = feat["ema"]; atr = feat["atr"]
    insess = _in_session(tmin)
    fc = _force_close_window(tmin)
    n = close.size
    if long_ok is None:
        long_ok = np.ones(n, bool)
    if short_ok is None:
        short_ok = np.ones(n, bool)

    idx = bars.index
    trades = []
    pos = 0            # 0 flat, +1 long, -1 short
    entry_px = entry_i = 0
    sl = tp = 0.0
    worst = 0.0        # worst adverse price excursion (points, positive number)
    k = params.k

    for i in range(n):
        if pos != 0:
            # --- exits ---
            # force-close first (hard 16:00 flatten)
            if fc[i]:
                move = (close[i] - entry_px) * pos
                _emit(trades, idx, entry_i, i, pos, "force_close", move, worst, k, atr[entry_i])
                pos = 0
            else:
                a = atr[entry_i]
                if pos == 1:
                    adv = entry_px - low[i]
                    if adv > worst:
                        worst = adv
                    if low[i] <= sl:
                        _emit(trades, idx, entry_i, i, pos, "stop", -(k * a), worst, k, a)
                        pos = 0
                    elif high[i] >= tp:
                        _emit(trades, idx, entry_i, i, pos, "target", k * a, max(worst, 0.0), k, a)
                        pos = 0
                else:
                    adv = high[i] - entry_px
                    if adv > worst:
                        worst = adv
                    if high[i] >= sl:
                        _emit(trades, idx, entry_i, i, pos, "stop", -(k * a), worst, k, a)
                        pos = 0
                    elif low[i] <= tp:
                        _emit(trades, idx, entry_i, i, pos, "target", k * a, max(worst, 0.0), k, a)
                        pos = 0

        if pos == 0:
            # --- entry ---
            if (insess[i] and not sess_break[i] and not fc[i]
                    and np.isfinite(z[i]) and z[i] > params.high_z
                    and atr[i] > 0 and atr[i] <= params.atr_max
                    and np.isfinite(ema[i])):
                mode = params.direction_mode
                if mode == "ema_cont":
                    cand = 1 if close[i] > ema[i] else (-1 if close[i] < ema[i] else 0)
                elif mode == "ema_revert":
                    cand = 1 if close[i] < ema[i] else (-1 if close[i] > ema[i] else 0)
                elif mode == "mom_cont":
                    cand = 1 if close[i] > open_[i] else (-1 if close[i] < open_[i] else 0)
                elif mode == "mom_fade":
                    cand = 1 if close[i] < open_[i] else (-1 if close[i] > open_[i] else 0)
                else:
                    cand = 0
                if cand == 1 and (not params.use_orderflow or long_ok[i]):
                    pos = 1
                elif cand == -1 and (not params.use_orderflow or short_ok[i]):
                    pos = -1
                if pos != 0:
                    entry_px = close[i]; entry_i = i; worst = 0.0
                    a = atr[i]
                    sl = entry_px - pos * k * a
                    tp = entry_px + pos * k * a

    df = pd.DataFrame(trades)
    return df


def _emit(trades, idx, ei, xi, pos, reason, move_pts, worst_pts, k, atr_at_entry):
    risk_pts = k * atr_at_entry
    r = move_pts / risk_pts if risk_pts > 0 else 0.0
    r = max(r, -1.0)                       # SL caps the loss at -1R
    mae_r = -min(worst_pts / risk_pts, 1.0) if risk_pts > 0 else 0.0
    trades.append({
        "entry_ts": idx[ei], "exit_ts": idx[xi],
        "direction": "LONG" if pos == 1 else "SHORT",
        "reason": reason,
        "r": r, "pnl_$": r * RISK_DOLLARS,
        "mae_r": mae_r, "mae_$": mae_r * RISK_DOLLARS,
        "atr": atr_at_entry,
    })
