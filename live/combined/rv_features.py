"""Rough Vol feature state machine — stateful, O(1)-per-bar updates.

Maintains rolling windows for:
  - log returns (ret_t = ln(close_t / close_{t-1}))
  - return z-score (shock_t)
  - rough vol kernel convolution (xH_t)
  - model vol (v_model_t)
  - vol z-score (z_vol_t)
  - EMA close (Wilder/standard)
  - ATR (Wilder)

Locked params per live config:
  H=0.4, KERNEL_LEN=80, ETA=1.0, V0=0.0001
  NORM_LEN=400, Z_LOOKBACK=75, EMA_LEN=80, ATR_LEN=14
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class RVConfig:
    H: float = 0.4
    KERNEL_LEN: int = 80
    ETA: float = 1.0
    V0: float = 0.0001
    NORM_LEN: int = 400
    Z_LOOKBACK: int = 75
    EMA_LEN: int = 80
    ATR_LEN: int = 14


def make_kernel(kernel_len: int = 80, H: float = 0.4) -> np.ndarray:
    """Fractional Brownian kernel. kernel[k] = k^(H-0.5) - (k-1)^(H-0.5), kernel[0]=1.0"""
    ks = np.arange(1, kernel_len + 1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = ks ** (H - 0.5) - (ks - 1) ** (H - 0.5)
    k[0] = 1.0
    return k


class RVFeatures:
    """Rolling state for rough-vol features. Update per bar close."""

    def __init__(self, cfg: RVConfig = RVConfig()):
        self.cfg = cfg
        self.kernel = make_kernel(cfg.KERNEL_LEN, cfg.H)

        # History buffers (most recent at the end)
        self._closes: deque[float] = deque(maxlen=max(cfg.NORM_LEN, cfg.KERNEL_LEN, cfg.EMA_LEN) + 10)
        self._highs: deque[float] = deque(maxlen=cfg.ATR_LEN + 5)
        self._lows: deque[float] = deque(maxlen=cfg.ATR_LEN + 5)
        self._rets: deque[float] = deque(maxlen=cfg.NORM_LEN + cfg.KERNEL_LEN + 10)
        self._shocks: deque[float] = deque(maxlen=cfg.KERNEL_LEN + 10)
        self._v_models: deque[float] = deque(maxlen=cfg.Z_LOOKBACK + 10)

        # Recursive states
        self._ema: float | None = None
        self._atr: float | None = None
        self._tr_buf: deque[float] = deque(maxlen=cfg.ATR_LEN)  # for the SMA seed
        self._n_bars = 0

        # Cached current values (None until enough bars)
        self.z_vol: float | None = None
        self.ema: float | None = None
        self.atr: float | None = None
        self.close: float | None = None

    @property
    def ready(self) -> bool:
        """True once we have enough bars for all features to be defined."""
        return self._n_bars >= max(self.cfg.NORM_LEN, self.cfg.KERNEL_LEN, self.cfg.Z_LOOKBACK)

    def n_bars(self) -> int:
        return self._n_bars

    def update(self, open_p: float, high: float, low: float, close: float) -> None:
        """Update all features with a new closed bar. Call once per 20-min bar close."""
        cfg = self.cfg

        # ---- Returns ----
        if len(self._closes) > 0:
            prev_close = self._closes[-1]
            if prev_close > 0 and close > 0:
                ret = math.log(close / prev_close)
            else:
                ret = 0.0
        else:
            ret = 0.0
        self._rets.append(ret)
        self._closes.append(close)
        self._highs.append(high)
        self._lows.append(low)

        # ---- Shock (return z-score over NORM_LEN window) ----
        if len(self._rets) >= cfg.NORM_LEN:
            window = np.fromiter(list(self._rets)[-cfg.NORM_LEN:], dtype=np.float64, count=cfg.NORM_LEN)
            m = window.mean()
            s = window.std(ddof=0)
            shock = (ret - m) / s if s > 0 else 0.0
        else:
            shock = 0.0
        self._shocks.append(shock)

        # ---- xH (causal kernel convolution at current bar) ----
        # xH_t = sum_{k=0..KERNEL_LEN-1} shock_{t-k} * kernel[k]
        if len(self._shocks) >= cfg.KERNEL_LEN:
            shock_window = np.fromiter(list(self._shocks)[-cfg.KERNEL_LEN:][::-1],
                                        dtype=np.float64, count=cfg.KERNEL_LEN)
            xH = float(np.dot(shock_window, self.kernel))
        else:
            xH = 0.0

        # ---- v_model ----
        v_model = min(cfg.V0 * math.exp(cfg.ETA * xH), cfg.V0 * 1e4)
        self._v_models.append(v_model)

        # ---- z_vol over Z_LOOKBACK window ----
        if len(self._v_models) >= cfg.Z_LOOKBACK:
            vm = np.fromiter(list(self._v_models)[-cfg.Z_LOOKBACK:], dtype=np.float64, count=cfg.Z_LOOKBACK)
            mu = vm.mean()
            sd = vm.std(ddof=0)
            z = (v_model - mu) / sd if sd > 0 else 0.0
        else:
            z = 0.0
        self.z_vol = z

        # ---- EMA (Pandas-equivalent ewm with adjust=False) ----
        alpha = 2.0 / (cfg.EMA_LEN + 1)
        if self._ema is None:
            self._ema = close
        else:
            self._ema = alpha * close + (1 - alpha) * self._ema
        self.ema = self._ema

        # ---- ATR (Wilder/RMA) ----
        # TR = max(high-low, |high-prev_close|, |low-prev_close|)
        if len(self._closes) >= 2:
            prev_close = self._closes[-2]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        else:
            tr = high - low

        if self._atr is None:
            # Seed phase: collect TR values for SMA seed
            self._tr_buf.append(tr)
            if len(self._tr_buf) >= cfg.ATR_LEN:
                self._atr = sum(self._tr_buf) / cfg.ATR_LEN
        else:
            # RMA: ATR_t = ((ATR_{t-1} * (n-1)) + TR_t) / n
            self._atr = (self._atr * (cfg.ATR_LEN - 1) + tr) / cfg.ATR_LEN
        self.atr = self._atr

        self.close = close
        self._n_bars += 1

    def snapshot(self) -> dict:
        """Return current feature values as a dict (for logging/debug)."""
        return {
            "n_bars": self._n_bars,
            "ready": self.ready,
            "close": self.close,
            "z_vol": self.z_vol,
            "ema": self.ema,
            "atr": self.atr,
        }
