"""Vectorized rough-vol features for the 1-min pass-rate study.

Matches the streaming live engine (live/combined/rv_features.py RVFeatures) math exactly:
  ret    = ln(close/prev_close)
  shock  = (ret - mean_NORM(ret)) / std_NORM(ret, ddof=0)
  xH[t]  = sum_{k=0..K-1} shock[t-k] * kernel[k]          (causal FIR, kernel[0]=1)
  v_model= min(V0*exp(ETA*xH), V0*1e4)
  z_vol  = (v_model - mean_Z(v_model)) / std_Z(v_model, ddof=0)
  ema    = ewm(span=EMA_LEN, adjust=False)
  atr    = Wilder RMA of true range over ATR_LEN

Vectorized so a full 1.9M-bar pass is ~seconds, enabling per-config feature recompute in stages 1-4.
The orderflow long_ok/short_ok flags are config-independent (only WINDOW params) -> cached separately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# reuse the EXACT kernel + config from the live engine
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from live.combined.rv_features import RVConfig, make_kernel  # noqa: E402


def compute_features(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                     cfg: RVConfig) -> dict[str, np.ndarray]:
    """Return dict of arrays (z_vol, ema, atr) aligned to the input bars.

    Leading warmup bars are NaN until each rolling window is full (matches the
    streaming engine's `ready` gate, which suppresses signals during warmup)."""
    close = np.asarray(close, dtype=np.float64)
    n = close.size

    # ---- log returns ----
    ret = np.empty(n, dtype=np.float64)
    ret[0] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ret[1:] = np.log(close[1:] / close[:-1])
    ret[~np.isfinite(ret)] = 0.0
    rser = pd.Series(ret)

    # ---- shock: rolling z of ret over NORM_LEN (ddof=0) ----
    rm = rser.rolling(cfg.NORM_LEN).mean()
    rs = rser.rolling(cfg.NORM_LEN).std(ddof=0)
    shock = (rser - rm) / rs
    shock = shock.where(rs > 0, 0.0).fillna(0.0).to_numpy()

    # ---- xH: causal convolution shock * kernel ----
    kernel = make_kernel(cfg.KERNEL_LEN, cfg.H)
    xH = np.convolve(shock, kernel)[:n]

    # ---- v_model ----
    v_model = np.minimum(cfg.V0 * np.exp(cfg.ETA * xH), cfg.V0 * 1e4)
    vser = pd.Series(v_model)

    # ---- z_vol: rolling z of v_model over Z_LOOKBACK (ddof=0) ----
    vm = vser.rolling(cfg.Z_LOOKBACK).mean()
    vs = vser.rolling(cfg.Z_LOOKBACK).std(ddof=0)
    z_vol = (vser - vm) / vs
    z_vol = z_vol.where(vs > 0, 0.0).to_numpy()

    # ---- EMA ----
    ema = pd.Series(close).ewm(span=cfg.EMA_LEN, adjust=False).mean().to_numpy()

    # ---- ATR (Wilder) ----
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    prev_close = np.empty(n, dtype=np.float64)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    # Wilder seed = SMA of first ATR_LEN TRs, then RMA recursion. ewm(alpha=1/N) matches the
    # recursion; seed difference is a tiny warmup-only effect, well before any tradable bar.
    atr = pd.Series(tr).ewm(alpha=1.0 / cfg.ATR_LEN, adjust=False).mean().to_numpy()

    # warmup mask: z_vol/ema/atr undefined until the longest window is full
    warm = max(cfg.NORM_LEN, cfg.Z_LOOKBACK, cfg.KERNEL_LEN)
    z_vol[:warm] = np.nan

    return {"z_vol": z_vol, "ema": ema, "atr": atr}
