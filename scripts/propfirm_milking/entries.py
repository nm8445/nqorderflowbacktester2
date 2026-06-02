"""Per-strategy locked-entry feeds for the Phase-1 bracket sweep.

Phase 1 keeps each strategy's ENTRY logic LOCKED — we read the actual entry
signals from the existing 4-way combined trade log and re-bracket them. (Phase 2
re-sweeps entry params via the full engines.) For each logged entry we
reconstruct, from the 1-min parquet:
    fill_time, entry_price, native-timeframe ATR-at-entry (per atr_len),
    the forward 1-min high/low path to force-close, and the force-close price.

Per-strat conventions (verified against the live engines):
  OD  : entry_ts = 19:00 ET 20-min bar OPEN; fill at that bar's CLOSE (19:20);
        20-min ATR; force-close 08:00 ET next day.
  RV  : entry_ts = 20-min bar CLOSE; 20-min ATR; force-close 14:45 ET same day.
  B2  : entry_ts = 5-min conf-bar close; 20-min ATR (asof); force-close 16:00 ET.
  FB  : entry_ts = 5-min bar close; 5-min ATR; force-close 14:00 ET same day.

Caveat (documented): re-bracketing the locked entries independently ignores the
flat-gating shift that new (faster) exits would cause for the multi-entry strats
(RV/B2) — slightly conservative on signal count. Phase 2 recovers this.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from scripts.propfirm_milking.common import (
    ET, ATR_LEN_GRID, FOURWAY_TRADES, load_1min_bars, build_bars, wilder_atr, EntryPack,
)

# strat -> (native timeframe minutes, force-close spec)
# force-close spec: ("same", hours, minutes) or ("next", hours, minutes)
STRAT_TF = {"OD": 20, "RV": 20, "B2": 20, "FB": 5, "MEC": 30}
STRAT_FC = {
    "OD": ("next", 8, 0),
    "RV": ("same", 14, 45),
    "B2": ("same", 16, 0),
    "FB": ("same", 14, 0),
    "MEC": ("same", 14, 30),   # Morning EMA Cross: entry 10:00 ET, force-close 14:30 ET
}
# OD fills at the 20-min bar CLOSE, so its fill is 20 min after the logged bar-open ts.
STRAT_FILL_SHIFT_MIN = {"OD": 20, "RV": 0, "B2": 0, "FB": 0, "MEC": 0}

# Morning EMA Cross: long if 10:00 ET 30-min close > EMA(60), short if <. (winner from
# scripts/morning_ema_cross prior sweep). Entry generated from bars (not in the 4-way log).
MEC_EMA_SPAN = 60


def _fc_time(fill_time: pd.Timestamp, strat: str) -> pd.Timestamp:
    mode, hh, mm = STRAT_FC[strat]
    base = fill_time
    if mode == "next":
        base = base + pd.Timedelta(days=1)
    day = pd.Timestamp(base.tz_convert(ET).date(), tz=ET)
    return day + pd.Timedelta(hours=hh, minutes=mm)


def _asof_values(sorted_idx_ns: np.ndarray, values: np.ndarray, query_ns: np.ndarray) -> np.ndarray:
    """For each query time, return value at the last index <= query (NaN if none)."""
    pos = np.searchsorted(sorted_idx_ns, query_ns, side="right") - 1
    out = np.full(query_ns.shape, np.nan)
    ok = pos >= 0
    out[ok] = values[pos[ok]]
    return out


def load_4way_entries() -> pd.DataFrame:
    df = pd.read_csv(FOURWAY_TRADES)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, format="mixed").dt.tz_convert(ET)
    df = df[["entry_ts", "direction", "strat"]].copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)
    return df


def _log_fills_signs(strat: str):
    """Locked entries from the 4-way log: returns (fills[Timestamp], signs[int])."""
    ent = load_4way_entries()
    ent = ent[ent["strat"] == strat]
    if len(ent) == 0:
        return [], []
    shift = STRAT_FILL_SHIFT_MIN[strat]
    fills = list(ent["entry_ts"] + pd.Timedelta(minutes=shift))
    signs = [1 if d == "LONG" else -1 for d in ent["direction"]]
    return fills, signs


def _mec_fills_signs(df1m: pd.DataFrame):
    """Morning EMA Cross entries, generated from 30-min bars (one/weekday at 10:00 ET)."""
    nb = build_bars(df1m, 30)
    ema = nb["close"].ewm(span=MEC_EMA_SPAN, adjust=False).mean()
    hhmm = nb.index.hour * 100 + nb.index.minute
    dow = nb.index.dayofweek
    mask = (hhmm == 1000) & (dow < 5)
    eb = nb[mask]
    em = ema[mask]
    sign = np.where(eb["close"].values > em.values, 1,
                    np.where(eb["close"].values < em.values, -1, 0))
    keep = sign != 0
    return list(eb.index[keep]), [int(x) for x in sign[keep]]


def packs_from_fills(strat: str, fills: list, signs: list,
                     atr_lens=ATR_LEN_GRID, verbose=True) -> list[EntryPack]:
    """Build precomputed EntryPacks from a custom (fills, signs) entry list.
    Uses STRAT_TF[strat] for the native ATR timeframe and STRAT_FC[strat] for force-close."""
    df1m = load_1min_bars()
    idx_ns = df1m.index.values.astype("int64")
    hi = df1m["high"].values
    lo = df1m["low"].values
    cl = df1m["close"].values

    tf = STRAT_TF[strat]
    nb = build_bars(df1m, tf)
    nb_idx_ns = nb.index.values.astype("int64")
    atr_series = {n: wilder_atr(nb, n).values for n in atr_lens}

    if not fills:
        return []

    fill_ns = np.array([t.value for t in fills], dtype=np.int64)
    entry_price = _asof_values(idx_ns, cl, fill_ns)              # 1-min close asof fill
    atr_at = {n: _asof_values(nb_idx_ns, atr_series[n], fill_ns) for n in atr_lens}

    packs: list[EntryPack] = []
    n_skipped = 0
    for k in range(len(fills)):
        ft = fills[k]
        ep = entry_price[k]
        if not np.isfinite(ep):
            n_skipped += 1
            continue
        sign = signs[k]
        fc = _fc_time(ft, strat)
        ft_ns = np.int64(ft.value)
        fc_ns = np.int64(fc.value)
        start = int(np.searchsorted(idx_ns, ft_ns, side="right"))   # first bar AFTER fill
        end = int(np.searchsorted(idx_ns, fc_ns, side="right"))     # one past last bar <= fc
        if end <= start:
            n_skipped += 1
            continue
        atr_by_len = {n: atr_at[n][k] for n in atr_lens}
        packs.append(EntryPack(
            strat=strat, fill_time=ft, date=ft.tz_convert(ET).date(),
            sign=sign, entry_price=float(ep), atr_by_len=atr_by_len,
            fwd_high=hi[start:end], fwd_low=lo[start:end], fc_close=float(cl[end - 1]),
        ))
    if verbose:
        print(f"  [{strat}] entries={len(fills)} packs={len(packs)} skipped={n_skipped}")
    return packs


def build_entry_packs(strat: str, atr_lens=ATR_LEN_GRID, verbose=True) -> list[EntryPack]:
    """Build EntryPacks for one strategy from its standard entry source
    (generated for MEC, locked 4-way log for OD/RV/B2/FB)."""
    df1m = load_1min_bars()
    if strat == "MEC":
        fills, signs = _mec_fills_signs(df1m)
    else:
        fills, signs = _log_fills_signs(strat)
    return packs_from_fills(strat, fills, signs, atr_lens, verbose)


if __name__ == "__main__":
    # Sanity: build packs for each strat, report counts + a sample bracket outcome
    from scripts.propfirm_milking.common import eval_bracket, stats_block
    for s in ["RV", "B2", "FB", "MEC"]:
        packs = build_entry_packs(s)
        if not packs:
            print(f"{s}: NO PACKS")
            continue
        tr = eval_bracket(packs, atr_len=14, sl_mult=1.5, tp_mult=2.0)
        st = stats_block(tr)
        print(f"{s}: n={st['n']} wr={st['wr']}% net=${st['net']:,.0f} "
              f"tp={st['tp']} sl={st['sl']} fc={st['fc']} tagged={st['tagged_frac']} "
              f"mean_risk=${st['mean_risk']:,.0f} mean_mnq={st['mean_mnq']}")
