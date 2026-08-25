"""NAKED overnight-range break on 20-min bars — TP x SL sweep.

Question this answers: does B2 need the gamma levels (+ pinbar + orderflow
absorption + confirmation) at all, or is the edge just "overnight range broke on
the 20-min chart, go with it"?

STRATEGY (no gamma, no orderflow, no pinbar):
  - Overnight range OHI/OLO from 18:00 ET (D-1) -> 09:30 ET (D), frozen at the open.
    Computed from 1-min bars so the range is exact.
  - Entry: the first 20-min RTH bar that CLOSES beyond the range (N_CONFIRM
    consecutive closes, default 1).  close > OHI -> LONG, close < OLO -> SHORT.
    Entry price = that bar's close (== next bar's open), management starts on the
    NEXT 20-min bar.  Flat overnight, max 1 trade/day by default.
  - Exit: Fabio-ORB-style GIVEBACK TRAILING YELLOW, but on 20-min ATR(14)
    (live FB config values for giveback/scale_body/max_gb/min_gap; k swept):
        raw_yellow = close - sign*k*ATR
        after an ADVERSE candle -> give back min(gap*frac, max_gb*ATR) of the
            ratchet, frac = giveback * min(1, adverse_body/ATR)
        else ratchet (never loosens)
        then tighten to at least raw_yellow + min_gap*ATR
        then clamp so it is never LOOSER than the floor (see floor_mode)
    Trigger = ADVERSE 20-min CLOSE beyond yellow (engine-driven, like FB/OD/B2).
  - Per-bar exit order: hard disaster SL (intrabar, stage 2 only) -> TP (intrabar)
    -> yellow (adverse close) -> force close at the 15:50 bar.

SWEEP
  tp_mode  'atr'      : TP = entry +/- tp * ATR_at_entry
           'r_candle' : TP = entry +/- tp * R,  R = |entry - break bar's low/high|
  k        yellow ATR multiple
  floor    'level'  = OHI/OLO (the FB "never looser than ORB_Low" analog)
           'candle' = break bar's low (LONG) / high (SHORT)
           'atrX'   = entry -/+ X * ATR_at_entry
           'none'   = pure trail, no floor
  giveback on / off

  n_confirm  1 / 2 / 3 consecutive closes beyond the range before entering

Stage 2 overlays the top configs with a hard intrabar disaster stop; stage 3
varies the entry window / trades-per-day; stage 4 re-runs the top configs on the
standard :00/:20/:40 20-min grid (instead of the 09:30-anchored one) as a
shift-robustness check; finally the winner is run against fade / always-long /
always-short / 25 coin-flip-direction controls, which is the real significance
gate (the exit engine alone makes money on random directions).

Outputs -> scripts/overnight range strat/tradelogs/naked_break/
No commissions/slippage modelled (matches the other strat scripts in this repo).
"""
from __future__ import annotations

import pickle
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"
NQ_1MIN = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
OUT_DIR = Path(__file__).parent.parent / "tradelogs" / "naked_break"
CACHE_DIR = Path("C:/Users/njchi/AppData/Local/Temp/claude/C--trading-nqorderflowbacktester/"
                 "e26b2d1a-c511-4b4b-a44e-b5d1dfed6d07/scratchpad")

ATR_LEN = 14                 # 20-min ATR(14), Pine RMA
RTH_OPEN = dtime(9, 30)
FORCE_CLOSE_OPEN_MIN = 15 * 60 + 50   # force close on the bar OPENING at 15:50 ET
IS_END = pd.Timestamp("2024-12-31").date()

# FB live-config yellow shape constants (only k is swept)
YELLOW_GIVEBACK = 0.3
YELLOW_SCALE_BODY = True
YELLOW_MAX_GB = 0.5
YELLOW_MIN_GAP = 0.3

N_WORKERS = 6

# B2 locked-config baseline for comparison (locked_v2_k08_lock045_mart_fc_filtered_trades.csv,
# 1 contract / no martingale, 2020-12-07 -> 2026-05-01)
B2_BASELINE = dict(n=713, net=4930.7, wr=60.31)


@dataclass(frozen=True)
class Cfg:
    tp_mode: str = "atr"       # 'atr' | 'r_candle'
    tp: float = 2.0
    k: float = 1.5
    floor: str = "level"       # 'level' | 'candle' | 'atr1.5' | 'atr2.5' | 'none'
    giveback: bool = True
    n_confirm: int = 1
    hard_sl: float = 0.0       # 0 = off; else intrabar stop at hard_sl * ATR_at_entry
    max_trades: int = 1
    hours: tuple = ()          # () = all RTH hours; else allowed entry-close hours
    # direction controls (significance testing): trade WITH the break, against it,
    # always one way, or coin-flip. Entry bars and exit engine stay identical.
    dir_mode: str = "with"     # 'with' | 'fade' | 'long' | 'short' | 'random'
    seed: int = 0

    def label(self) -> str:
        gb = "gb" if self.giveback else "nogb"
        hs = f" hsl{self.hard_sl:g}" if self.hard_sl else ""
        return f"N{self.n_confirm} {self.tp_mode}{self.tp:g} k{self.k:g} {self.floor} {gb}{hs}"


# ---------------------------------------------------------------- data prep
def rma_atr(high, low, close, length):
    """Pine-style RMA ATR."""
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    out = np.full(len(tr), np.nan)
    if len(tr) < length:
        return pd.Series(out, index=close.index)
    tr_v = tr.values
    prev = np.nanmean(tr_v[:length])
    out[length - 1] = prev
    alpha = 1.0 / length
    for i in range(length, len(tr_v)):
        cur = tr_v[i]
        if not np.isfinite(cur):
            out[i] = prev
            continue
        prev = (1 - alpha) * prev + alpha * cur
        out[i] = prev
    return pd.Series(out, index=close.index)


def _load_1min() -> pd.DataFrame:
    df = pd.read_parquet(NQ_1MIN)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(ET)
    df = df.sort_index()
    # markettick 1-min bars are RIGHT-labelled (ts = bar close) -> shift to OPEN times
    df.index = df.index - pd.Timedelta(minutes=1)
    return df[["open", "high", "low", "close"]]


def build_days(anchor: str = "0930") -> list[dict]:
    """Per-session numpy bundles: 20-min RTH bars + that session's overnight range."""
    m1 = _load_1min()

    # --- overnight range per session (18:00 D-1 -> 09:30 D), from 1-min bars ---
    t = m1.index
    on_mask = (t.hour >= 18) | (t.time < RTH_OPEN)
    sess = np.where(t.hour >= 18, (t + pd.Timedelta(days=1)).date, t.date)
    on = pd.DataFrame({"sess": sess[on_mask],
                       "high": m1["high"].values[on_mask],
                       "low": m1["low"].values[on_mask]})
    rng = on.groupby("sess").agg(ohi=("high", "max"), olo=("low", "min"))

    # --- 20-min bars over the full ETH series (ATR carries across sessions) ---
    if anchor == "0930":
        origin = m1.index[0].normalize() + pd.Timedelta(hours=9, minutes=30)
    else:
        origin = "start_day"          # standard :00/:20/:40 grid
    r = m1.resample("20min", label="left", closed="left", origin=origin)
    b = pd.DataFrame({"open": r["open"].first(), "high": r["high"].max(),
                      "low": r["low"].min(), "close": r["close"].last()}).dropna()
    b["atr"] = rma_atr(b["high"], b["low"], b["close"], ATR_LEN)

    # --- RTH slice, grouped per calendar day ---
    idx = b.index
    open_min = idx.hour * 60 + idx.minute
    rth = (open_min >= 9 * 60 + 30) & (open_min < 16 * 60)
    br = b[rth]
    bmin = open_min[rth]
    dates = br.index.date

    days = []
    for d, grp_idx in pd.Series(np.arange(len(br)), index=dates).groupby(level=0):
        if d not in rng.index:
            continue
        sl = grp_idx.values
        if len(sl) < 3:
            continue
        days.append(dict(
            date=d,
            o=br["open"].values[sl], h=br["high"].values[sl],
            l=br["low"].values[sl], c=br["close"].values[sl],
            atr=br["atr"].values[sl], omin=bmin[sl].values,
            ts=br.index[sl],
            ohi=float(rng.loc[d, "ohi"]), olo=float(rng.loc[d, "olo"]),
        ))
    return days


# ---------------------------------------------------------------- exit engine
def _next_yellow(prev_yellow, adverse_body, close, atr, sign, floor, k, giveback):
    """One bar of the FB giveback-trailing-yellow update, mirrored for shorts."""
    raw = close - sign * k * atr
    if giveback and adverse_body > 0.0 and prev_yellow == prev_yellow:
        gap = max(0.0, sign * (prev_yellow - raw))
        frac = YELLOW_GIVEBACK * (min(1.0, adverse_body / atr) if YELLOW_SCALE_BODY else 1.0)
        cand = prev_yellow - sign * min(gap * frac, YELLOW_MAX_GB * atr)
    else:
        base = prev_yellow if prev_yellow == prev_yellow else raw
        cand = base if sign * (base - raw) > 0 else raw          # ratchet
    if giveback:
        tight = raw + sign * YELLOW_MIN_GAP * atr                # keep it hittable
        cand = cand if sign * (cand - tight) > 0 else tight
    if floor is not None:
        cand = cand if sign * (cand - floor) > 0 else floor      # never looser than floor
    return cand


def sim_day(day: dict, cfg: Cfg, rng=None) -> list[dict]:
    o, h, l, c = day["o"], day["h"], day["l"], day["c"]
    atr, omin, ts = day["atr"], day["omin"], day["ts"]
    ohi, olo = day["ohi"], day["olo"]
    n = len(c)
    trades: list[dict] = []

    up = dn = 0
    armed_long = armed_short = True
    free_from = 0
    n_taken = 0

    for i in range(n):
        # break-state machine on closes
        if c[i] > ohi:
            up += 1; dn = 0
        elif c[i] < olo:
            dn += 1; up = 0
        else:
            up = dn = 0
            armed_long = armed_short = True      # re-arm once price closes back inside

        if i < free_from or n_taken >= cfg.max_trades or i >= n - 1:
            continue
        if cfg.hours and ts[i].hour + (1 if ts[i].minute + 20 >= 60 else 0) not in cfg.hours:
            # gate on the entry bar's CLOSE hour (bar opens at ts[i], closes +20min)
            continue
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue

        if up >= cfg.n_confirm and armed_long:
            sign = 1
        elif dn >= cfg.n_confirm and armed_short:
            sign = -1
        else:
            continue
        broke = sign

        if cfg.dir_mode == "fade":
            sign = -sign
        elif cfg.dir_mode == "long":
            sign = 1
        elif cfg.dir_mode == "short":
            sign = -1
        elif cfg.dir_mode == "random":
            sign = 1 if rng.random() < 0.5 else -1

        tr = _sim_exit(day, i, sign, cfg)
        trades.append(tr)
        n_taken += 1
        free_from = tr["exit_i"] + 1
        if broke > 0:          # arm off the BREAK side, not the (possibly overridden) trade side
            armed_long = False
        else:
            armed_short = False

    return trades


def _sim_exit(day: dict, i0: int, sign: int, cfg: Cfg) -> dict:
    o, h, l, c = day["o"], day["h"], day["l"], day["c"]
    atr, omin, ts = day["atr"], day["omin"], day["ts"]
    n = len(c)

    entry = float(c[i0])
    atr0 = float(atr[i0])
    level = day["ohi"] if sign > 0 else day["olo"]
    candle = float(l[i0]) if sign > 0 else float(h[i0])

    if cfg.tp_mode == "atr":
        tp = entry + sign * cfg.tp * atr0
    else:                                    # r_candle
        R = max(abs(entry - candle), 0.25 * atr0)   # guard degenerate 0-risk candles
        tp = entry + sign * cfg.tp * R

    if cfg.floor == "level":
        floor = level
    elif cfg.floor == "candle":
        floor = candle
    elif cfg.floor.startswith("atr"):
        floor = entry - sign * float(cfg.floor[3:]) * atr0
    else:
        floor = None

    hard = entry - sign * cfg.hard_sl * atr0 if cfg.hard_sl else None

    yellow = floor if floor is not None else entry - sign * cfg.k * atr0
    prev_yellow = float("nan")
    adverse_body = sign * (float(o[i0]) - float(c[i0]))
    mae = 0.0
    mfe = 0.0
    exit_i = n - 1
    exit_px = float(c[-1])
    reason = "EOD"

    for j in range(i0 + 1, n):
        a = float(atr[j])
        if np.isfinite(a) and a > 0:
            yellow = _next_yellow(prev_yellow, adverse_body, float(c[j]), a,
                                  sign, floor, cfg.k, cfg.giveback)
        exc_lo = sign * (float(l[j]) - entry) if sign > 0 else sign * (float(h[j]) - entry)
        exc_hi = sign * (float(h[j]) - entry) if sign > 0 else sign * (float(l[j]) - entry)
        mae = min(mae, exc_lo)
        mfe = max(mfe, exc_hi)

        if hard is not None and ((sign > 0 and l[j] <= hard) or (sign < 0 and h[j] >= hard)):
            exit_i, exit_px, reason = j, hard, "HARD_SL"
            break
        if (sign > 0 and h[j] >= tp) or (sign < 0 and l[j] <= tp):
            exit_i, exit_px, reason = j, tp, "TP"
            break
        adverse_close = (c[j] < o[j]) if sign > 0 else (c[j] > o[j])
        if adverse_close and ((sign > 0 and c[j] <= yellow) or (sign < 0 and c[j] >= yellow)):
            exit_i, exit_px, reason = j, float(c[j]), "SL_YELLOW"
            break
        if omin[j] >= FORCE_CLOSE_OPEN_MIN:
            exit_i, exit_px, reason = j, float(c[j]), "FORCE_CLOSE"
            break

        prev_yellow = yellow
        adverse_body = sign * (float(o[j]) - float(c[j]))

    return dict(
        date=day["date"], direction="LONG" if sign > 0 else "SHORT",
        entry_ts=ts[i0] + pd.Timedelta(minutes=20), exit_ts=ts[exit_i] + pd.Timedelta(minutes=20),
        entry_price=entry, exit_price=exit_px, reason=reason,
        pnl=sign * (exit_px - entry), bars_held=exit_i - i0,
        mae=mae, mfe=mfe, init_atr=atr0, exit_i=exit_i,
        entry_hour=(ts[i0] + pd.Timedelta(minutes=20)).hour,
        on_range=day["ohi"] - day["olo"],
    )


# ---------------------------------------------------------------- stats
def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(n=0, net=0.0, pf=0.0, wr=0.0, sharpe=0.0, mdd=0.0,
                    avg_win=0.0, avg_loss=0.0, worst=0.0, worst_mae=0.0,
                    tp=0, sl=0, fc=0, hsl=0)
    p = df["pnl"].values
    w, lo = p[p > 0], p[p < 0]
    daily = pd.Series(p, index=pd.to_datetime(df["date"].values)).groupby(level=0).sum()
    eq = np.cumsum(p)
    return dict(
        n=len(p), net=p.sum(),
        pf=(w.sum() / abs(lo.sum())) if lo.size and lo.sum() != 0 else float("inf"),
        wr=(p > 0).mean() * 100,
        sharpe=(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0,
        mdd=(eq - np.maximum.accumulate(eq)).min(),
        avg_win=w.mean() if w.size else 0.0, avg_loss=lo.mean() if lo.size else 0.0,
        worst=p.min(), worst_mae=df["mae"].min(),
        tp=int((df["reason"] == "TP").sum()),
        sl=int((df["reason"] == "SL_YELLOW").sum()),
        fc=int((df["reason"].isin(["FORCE_CLOSE", "EOD"])).sum()),
        hsl=int((df["reason"] == "HARD_SL").sum()),
    )


def run_cfg_df(days: list[dict], cfg: Cfg) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed) if cfg.dir_mode == "random" else None
    rows = []
    for d in days:
        rows.extend(sim_day(d, cfg, rng))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["period"] = np.where(df["date"] <= IS_END, "IS", "OOS")
    df["year"] = pd.to_datetime(df["date"]).dt.year
    return df


def summarise(cfg: Cfg, df: pd.DataFrame) -> dict:
    a = stats(df)
    i = stats(df[df["period"] == "IS"]) if not df.empty else stats(df)
    o = stats(df[df["period"] == "OOS"]) if not df.empty else stats(df)
    yrs = df.groupby("year")["pnl"].sum() if not df.empty else pd.Series(dtype=float)
    return dict(label=cfg.label(), tp_mode=cfg.tp_mode, tp=cfg.tp, k=cfg.k,
                floor=cfg.floor, giveback=cfg.giveback, hard_sl=cfg.hard_sl,
                n_confirm=cfg.n_confirm,
                n=a["n"], net=a["net"], pf=a["pf"], wr=a["wr"], sharpe=a["sharpe"],
                mdd=a["mdd"], worst=a["worst"], worst_mae=a["worst_mae"],
                tp_n=a["tp"], sl_n=a["sl"], fc_n=a["fc"], hsl_n=a["hsl"],
                is_n=i["n"], is_net=i["net"], is_pf=i["pf"], is_sharpe=i["sharpe"],
                oos_n=o["n"], oos_net=o["net"], oos_pf=o["pf"], oos_sharpe=o["sharpe"],
                min_net=min(i["net"], o["net"]), min_pf=min(i["pf"], o["pf"]),
                yrs_pos=int((yrs > 0).sum()), yrs_tot=int(len(yrs)))


# ---------------------------------------------------------------- workers
_DAYS: dict[str, list[dict]] = {}


def _get_days(anchor: str) -> list[dict]:
    if anchor not in _DAYS:
        with open(CACHE_DIR / f"nb_days_{anchor}.pkl", "rb") as fh:
            _DAYS[anchor] = pickle.load(fh)
    return _DAYS[anchor]


def _work(args) -> dict:
    cfg, anchor = args
    return summarise(cfg, run_cfg_df(_get_days(anchor), cfg))


# ---------------------------------------------------------------- report bits
def grid_table(rows: list[dict], sort_key: str, top: int | None = None,
               title: str = "") -> list[str]:
    rs = sorted(rows, key=lambda r: -r[sort_key])
    if top:
        rs = rs[:top]
    L = []
    if title:
        L.append(title)
    L.append(f"  {'#':>3} {'config':<34} {'n':>5} {'net_pts':>9} {'$MNQ':>9} {'pf':>5} "
             f"{'wr':>6} {'shrp':>5} {'mdd':>8} {'is_net':>8} {'is_pf':>5} "
             f"{'oos_net':>8} {'oos_pf':>5} {'yrs+':>5} {'worst':>7} {'wMAE':>7} "
             f"{'tp/sl/fc':>12}")
    for i, r in enumerate(rs, 1):
        L.append(f"  {i:>3} {r['label']:<34} {r['n']:>5} {r['net']:>+9.1f} "
                 f"{r['net']*2:>+9,.0f} {r['pf']:>5.2f} {r['wr']:>5.1f}% {r['sharpe']:>5.2f} "
                 f"{r['mdd']:>+8.1f} {r['is_net']:>+8.1f} {r['is_pf']:>5.2f} "
                 f"{r['oos_net']:>+8.1f} {r['oos_pf']:>5.2f} "
                 f"{r['yrs_pos']:>2}/{r['yrs_tot']:<2} {r['worst']:>+7.1f} "
                 f"{r['worst_mae']:>+7.1f} {r['tp_n']}/{r['sl_n']}/{r['fc_n']:<4}")
    return L


# Cells worth a closer look after the stage-1 sweep (see the report's top tables):
#   A best DD/MAE profile — OHI/OLO floor keeps the tail tight (prop-account friendly)
#   B highest raw net
#   C best min(IS,OOS) among the high-net cells; also the most grid-shift stable
#   D highest Sharpe / 78% WR scalp — best shape for eval pass rate
CANDIDATES = [
    ("A  level-floor (tight tail)", Cfg(n_confirm=2, tp_mode="atr", tp=4.0, k=2.5,
                                        floor="level", giveback=True)),
    ("B  max net", Cfg(n_confirm=2, tp_mode="atr", tp=4.0, k=3.0,
                       floor="none", giveback=False)),
    ("C  N3 balanced", Cfg(n_confirm=3, tp_mode="atr", tp=4.0, k=2.5,
                           floor="none", giveback=True)),
    ("D  1R scalp, 78% WR", Cfg(n_confirm=2, tp_mode="r_candle", tp=1.0, k=2.5,
                                floor="none", giveback=False)),
    # E is the pick: centre of the broad N3 + OHI/OLO-floor plateau. Every cell with
    # tp 3-4 x k 1.5-3 in that corner lands +6.4k..+8.1k pts / PF 1.19-1.24 / 6-of-7
    # positive years, and the floor pins worst-MAE at -364 pts for all of them.
    ("E  N3 + level floor (PICK)", Cfg(n_confirm=3, tp_mode="atr", tp=4.0, k=2.5,
                                       floor="level", giveback=True)),
]


def confirm():
    """Deep-dive the CANDIDATES: both bar grids, direction controls, year/hour/MAE detail."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    grids = {}
    for anchor in ("0930", "mid"):
        d = build_days(anchor)
        with open(CACHE_DIR / f"nb_days_{anchor}.pkl", "wb") as fh:
            pickle.dump(d, fh)
        grids[anchor] = d

    L = ["=" * 150,
         "NAKED 20-MIN RANGE BREAK — CANDIDATE CONFIRMATION",
         "=" * 150, ""]
    for name, cfg in CANDIDATES:
        L.append("-" * 150)
        L.append(f"{name}   —   {cfg.label()}")
        L.append("-" * 150)
        L.append(f"  {'grid':<26} {'n':>5} {'net_pts':>9} {'$MNQ':>9} {'pf':>5} {'wr':>6} "
                 f"{'shrp':>5} {'mdd_pts':>8} {'$MNQ_dd':>9} {'is_net':>9} {'oos_net':>9} "
                 f"{'worst':>7} {'wMAE':>7}")
        dfs = {}
        for anchor, tag in (("0930", "09:30-anchored"), ("mid", ":00/:20/:40")):
            df = run_cfg_df(grids[anchor], cfg)
            dfs[anchor] = df
            s = stats(df)
            i = stats(df[df["period"] == "IS"]); o = stats(df[df["period"] == "OOS"])
            L.append(f"  {tag:<26} {s['n']:>5} {s['net']:>+9.1f} {s['net']*2:>+9,.0f} "
                     f"{s['pf']:>5.2f} {s['wr']:>5.1f}% {s['sharpe']:>5.2f} "
                     f"{s['mdd']:>+8.1f} {s['mdd']*2:>+9,.0f} {i['net']:>+9.1f} "
                     f"{o['net']:>+9.1f} {s['worst']:>+7.1f} {s['worst_mae']:>+7.1f}")
        df = dfs["0930"]
        s = stats(df)

        # direction controls
        ctrl = [replace(cfg, dir_mode=m) for m in ("fade", "long", "short")] + \
               [replace(cfg, dir_mode="random", seed=sd) for sd in range(25)]
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            rc = list(ex.map(_work, [(c, "0930") for c in ctrl], chunksize=2))
        rnd = np.array([r["net"] for r in rc[3:]])
        mu, sd_ = rnd.mean(), rnd.std(ddof=1)
        z = (s["net"] - mu) / sd_ if sd_ > 0 else float("nan")
        L.append(f"  controls: fade {rc[0]['net']:+.0f} (mirror {-rc[0]['net']/s['net']:.2f}) | "
                 f"always-long {rc[1]['net']:+.0f} | always-short {rc[2]['net']:+.0f} | "
                 f"coin-flip {mu:+.0f} +/- {sd_:.0f}  ==>  {z:.2f} sd "
                 f"{'OK' if z >= 2 else 'WEAK'}")
        yr = df.groupby("year")["pnl"].sum()
        L.append("  per-year net: " + "  ".join(f"{y}:{v:+.0f}" for y, v in yr.items()))
        hr = df.groupby("entry_hour")["pnl"].agg(["count", "sum"])
        L.append("  by entry hour: " + "  ".join(
            f"{h}:{r['sum']:+.0f}({int(r['count'])})" for h, r in hr.iterrows()))
        for d_ in ("LONG", "SHORT"):
            sub = df[df["direction"] == d_]
            ss = stats(sub)
            L.append(f"  {d_:<6} n={ss['n']:<5} net={ss['net']:+.0f} pf={ss['pf']:.2f} "
                     f"wr={ss['wr']:.1f}% worst={ss['worst']:+.0f}")
        L.append("  MAE pctiles (pts): " + "  ".join(
            f"p{q}={np.percentile(df['mae'], 100-q):+.0f}" for q in (99, 95, 90, 75, 50)))
        L.append(f"  exit mix: TP={s['tp']} SL_YELLOW={s['sl']} FC/EOD={s['fc']}   "
                 f"avg win {s['avg_win']:+.1f} / avg loss {s['avg_loss']:+.1f}")
        L.append("")
        df.drop(columns=["exit_i"]).to_csv(
            OUT_DIR / f"cand_{name.split()[0]}_trades.csv", index=False)

    out = OUT_DIR / "candidate_confirmation.txt"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")


def main():
    t0 = _time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("building 20-min bars + overnight ranges (09:30 anchor)...")
    days = build_days("0930")
    with open(CACHE_DIR / "nb_days_0930.pkl", "wb") as fh:
        pickle.dump(days, fh)
    print(f"  {len(days)} sessions  {days[0]['date']} -> {days[-1]['date']}")
    print("building the :00/:20/:40 grid variant...")
    days_mid = build_days("mid")
    with open(CACHE_DIR / "nb_days_mid.pkl", "wb") as fh:
        pickle.dump(days_mid, fh)

    # ---------------- stage 1: TP x SL x N_CONFIRM grid ----------------
    grid: list[Cfg] = []
    for nc in (1, 2, 3):
        for k in (1.0, 1.5, 2.0, 2.5, 3.0):
            for floor in ("level", "candle", "atr1.5", "atr2.5", "none"):
                for gb in (True, False):
                    for tp in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
                        grid.append(Cfg(tp_mode="atr", tp=tp, k=k, floor=floor,
                                        giveback=gb, n_confirm=nc))
                    for tp in (1.0, 1.5, 2.0, 3.0, 4.0):
                        grid.append(Cfg(tp_mode="r_candle", tp=tp, k=k, floor=floor,
                                        giveback=gb, n_confirm=nc))
    print(f"stage 1: {len(grid)} configs on {N_WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        rows = list(ex.map(_work, [(c, "0930") for c in grid], chunksize=8))
    pd.DataFrame(rows).to_csv(OUT_DIR / "sweep_grid.csv", index=False)
    print(f"  done in {_time.time()-t0:.0f}s")

    by_label = {c.label(): c for c in grid}
    robust = [r for r in rows if r["is_net"] > 0 and r["oos_net"] > 0]

    L: list[str] = []
    L.append("=" * 190)
    L.append("NAKED 20-MIN OVERNIGHT-RANGE BREAK  —  no gamma / no orderflow / no pinbar")
    L.append("=" * 190)
    L.append("")
    L.append("ENTRY  N_CONFIRM consecutive 20-min RTH closes beyond the 18:00->09:30 ET range")
    L.append("       (N swept 1/2/3), entry at that bar's close, max 1 trade/day, flat")
    L.append("       overnight, all RTH hours.")
    L.append("EXIT   FB live-config giveback trailing yellow on 20-min ATR(14)")
    L.append(f"       (giveback={YELLOW_GIVEBACK} scale_body={YELLOW_SCALE_BODY} "
             f"max_gb={YELLOW_MAX_GB} min_gap={YELLOW_MIN_GAP}), adverse 20-min close through")
    L.append("       yellow; TP intrabar; force close on the 15:50 bar.")
    L.append(f"DATA   {len(days)} sessions {days[0]['date']} -> {days[-1]['date']}  "
             f"(IS <= {IS_END}, OOS after).  1 NQ contract, no costs.")
    L.append("")
    L.append(f"B2 LOCKED BASELINE (same span, 1 contract, no mart): n={B2_BASELINE['n']}  "
             f"net={B2_BASELINE['net']:+.1f} pts (${B2_BASELINE['net']*2:+,.0f} MNQ)  "
             f"wr={B2_BASELINE['wr']:.1f}%")
    L.append("")
    L.append("=" * 190)
    L.append(f"STAGE 1 — {len(grid)} cells.  {len(robust)} have IS>0 AND OOS>0.")
    L.append("=" * 190)
    L.append("")
    L += grid_table(rows, "net", 30, "TOP 30 by total net:")
    L.append("")
    L += grid_table(robust, "min_net", 30, "TOP 30 by min(IS net, OOS net)  [robustness rank]:")
    L.append("")
    L += grid_table(rows, "sharpe", 15, "TOP 15 by Sharpe:")
    L.append("")

    # marginals — is the surface smooth or a needle?
    g = pd.DataFrame(rows)
    L.append("MARGINALS (median net pts over every cell holding that value):")
    for dim in ("n_confirm", "tp_mode", "tp", "k", "floor", "giveback"):
        med = g.groupby(dim)["net"].median().sort_values(ascending=False)
        L.append(f"  {dim:<10} " + "   ".join(f"{k}={v:+.0f}" for k, v in med.items()))
    L.append("")

    # ---------------- stage 2: hard disaster SL overlay ----------------
    top = [by_label[r["label"]] for r in sorted(robust, key=lambda r: -r["min_net"])[:5]] \
        if robust else [by_label[r["label"]] for r in sorted(rows, key=lambda r: -r["net"])[:5]]
    s2 = [replace(c, hard_sl=hs) for c in top for hs in (0.0, 1.5, 2.0, 2.5, 3.0)]
    print(f"stage 2: {len(s2)} hard-SL overlays...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        rows2 = list(ex.map(_work, [(c, "0930") for c in s2], chunksize=4))
    L.append("=" * 190)
    L.append("STAGE 2 — hard intrabar disaster stop on the top-5 (hsl0 = none). "
             "Same-bar ties resolve to the stop (pessimistic).")
    L.append("=" * 190)
    L.append("")
    L += grid_table(rows2, "min_net", None)
    L.append("")

    # ---------------- stage 3: N_CONFIRM ----------------
    best_cfg = by_label[sorted(robust or rows, key=lambda r: -(r.get("min_net") or r["net"]))[0]["label"]]
    s3 = [best_cfg,
          replace(best_cfg, max_trades=3),
          replace(best_cfg, hours=(9, 10, 11, 12, 13, 14)),
          replace(best_cfg, hours=(9, 10, 11, 12, 13)),
          replace(best_cfg, hours=(9, 10, 13, 14, 15))]
    print(f"stage 3: {len(s3)} entry-rule variants...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        rows3 = list(ex.map(_work, [(c, "0930") for c in s3], chunksize=1))
    for r, c in zip(rows3, s3):
        hl = "".join(str(h) for h in c.hours) if c.hours else "allh"
        r["label"] = f"{c.label()} mx{c.max_trades} h{hl}"
    L.append("=" * 190)
    L.append(f"STAGE 3 — entry-rule variants on the winner ({best_cfg.label()})")
    L.append("=" * 190)
    L.append("")
    L += grid_table(rows3, "min_net", None)
    L.append("")

    # ---------------- stage 4: grid-anchor robustness ----------------
    print("stage 4: bar-grid shift robustness...")
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        rows4 = list(ex.map(_work, [(c, "mid") for c in top], chunksize=1))
    L.append("=" * 190)
    L.append("STAGE 4 — same top-5 on the standard :00/:20/:40 20-min grid "
             "(a 10-min shift; results should barely move if the edge is real)")
    L.append("=" * 190)
    L.append("")
    L += grid_table(rows4, "min_net", None)
    L.append("")

    # ---------------- winner detail ----------------
    wdf = run_cfg_df(days, best_cfg)
    wdf.drop(columns=["exit_i"]).to_csv(OUT_DIR / "best_config_trades.csv", index=False)
    a = stats(wdf)
    L.append("=" * 190)
    L.append(f"WINNER DETAIL — {best_cfg.label()}")
    L.append("=" * 190)
    L.append(f"  n={a['n']}  net={a['net']:+.1f} pts (${a['net']*2:+,.0f} MNQ / "
             f"${a['net']*20:+,.0f} NQ)  pf={a['pf']:.2f}  wr={a['wr']:.1f}%  "
             f"sharpe={a['sharpe']:.2f}  mdd={a['mdd']:+.1f} pts (${a['mdd']*2:+,.0f} MNQ)")
    L.append(f"  avg win {a['avg_win']:+.1f}  avg loss {a['avg_loss']:+.1f}  "
             f"worst trade {a['worst']:+.1f}  worst intrabar MAE {a['worst_mae']:+.1f} pts "
             f"(${a['worst_mae']*2:+,.0f} MNQ)")
    L.append("")
    L.append(f"  {'year':<6} {'n':>4} {'net':>9} {'$MNQ':>9} {'pf':>5} {'wr':>6} {'mdd':>8}")
    for y, sub in wdf.groupby("year"):
        s = stats(sub)
        L.append(f"  {y:<6} {s['n']:>4} {s['net']:>+9.1f} {s['net']*2:>+9,.0f} "
                 f"{s['pf']:>5.2f} {s['wr']:>5.1f}% {s['mdd']:>+8.1f}")
    L.append("")
    L.append(f"  {'entry_hr':<9} {'n':>4} {'net':>9} {'pf':>5} {'wr':>6}")
    for hr, sub in wdf.groupby("entry_hour"):
        s = stats(sub)
        L.append(f"  {hr:<9} {s['n']:>4} {s['net']:>+9.1f} {s['pf']:>5.2f} {s['wr']:>5.1f}%")
    L.append("")
    L.append(f"  {'direction':<10} {'n':>4} {'net':>9} {'pf':>5} {'wr':>6} {'worst':>8}")
    for dr, sub in wdf.groupby("direction"):
        s = stats(sub)
        L.append(f"  {dr:<10} {s['n']:>4} {s['net']:>+9.1f} {s['pf']:>5.2f} "
                 f"{s['wr']:>5.1f}% {s['worst']:>+8.1f}")
    L.append("")
    L.append(f"  {'reason':<12} {'n':>4} {'net':>9} {'avg':>7}")
    for rr, sub in wdf.groupby("reason"):
        L.append(f"  {rr:<12} {len(sub):>4} {sub['pnl'].sum():>+9.1f} {sub['pnl'].mean():>+7.2f}")
    L.append("")
    L.append("  MAE tail (worst intrabar excursion, pts):  " +
             "  ".join(f"p{q}={np.percentile(wdf['mae'], 100-q):+.0f}" for q in (99, 95, 90, 75, 50)))
    L.append("")

    # ---------------- direction controls: is the BREAK the edge? ----------------
    print("direction controls...")
    ctrl = [replace(best_cfg, dir_mode=m) for m in ("fade", "long", "short")] + \
           [replace(best_cfg, dir_mode="random", seed=s) for s in range(25)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        rowsc = list(ex.map(_work, [(c, "0930") for c in ctrl], chunksize=2))
    L.append("=" * 190)
    L.append("DIRECTION CONTROLS — identical entry bars and exit engine, direction replaced.")
    L.append("This is the significance gate: the with-break result has to clear the")
    L.append("coin-flip distribution, and fading should be its mirror image.")
    L.append("=" * 190)
    L.append("")
    L.append(f"  {'variant':<24} {'n':>5} {'net_pts':>9} {'$MNQ':>9} {'pf':>5} {'wr':>6} "
             f"{'shrp':>5} {'is_net':>9} {'oos_net':>9}")
    L.append(f"  {'WITH the break':<24} {a['n']:>5} {a['net']:>+9.1f} {a['net']*2:>+9,.0f} "
             f"{a['pf']:>5.2f} {a['wr']:>5.1f}% {a['sharpe']:>5.2f} "
             f"{stats(wdf[wdf['period']=='IS'])['net']:>+9.1f} "
             f"{stats(wdf[wdf['period']=='OOS'])['net']:>+9.1f}")
    for r, c in zip(rowsc[:3], ctrl[:3]):
        L.append(f"  {c.dir_mode.upper():<24} {r['n']:>5} {r['net']:>+9.1f} {r['net']*2:>+9,.0f} "
                 f"{r['pf']:>5.2f} {r['wr']:>5.1f}% {r['sharpe']:>5.2f} "
                 f"{r['is_net']:>+9.1f} {r['oos_net']:>+9.1f}")
    rnd = np.array([r["net"] for r in rowsc[3:]])
    mu, sd = rnd.mean(), rnd.std(ddof=1)
    z = (a["net"] - mu) / sd if sd > 0 else float("nan")
    L.append(f"  {'RANDOM (25 seeds) mean':<24} {'':>5} {mu:>+9.1f} {mu*2:>+9,.0f}"
             f"   sd {sd:.0f} pts   range {rnd.min():+.0f} .. {rnd.max():+.0f}")
    L.append("")
    L.append(f"  ==> with-break beats the coin-flip mean by {a['net']-mu:+.0f} pts "
             f"= {z:.2f} sd.  {'SIGNIFICANT' if z >= 2 else 'NOT convincing (<2 sd)'}")
    L.append(f"      fade net {rowsc[0]['net']:+.1f} vs with-break {a['net']:+.1f} "
             f"(mirror ratio {-rowsc[0]['net']/a['net']:.2f}; ~1.0 means the direction is the edge)")
    L.append("")
    L.append("=" * 190)
    L.append(f"VS B2 LOCKED (n={B2_BASELINE['n']}, {B2_BASELINE['net']:+.1f} pts, "
             f"{B2_BASELINE['wr']:.1f}% WR)")
    L.append("=" * 190)
    L.append(f"  naked break : n={a['n']:<5} {a['net']:+.1f} pts  wr {a['wr']:.1f}%  "
             f"pf {a['pf']:.2f}   ->  {a['net']/B2_BASELINE['net']:.2f}x B2's net "
             f"on {a['n']/B2_BASELINE['n']:.2f}x the trades")
    L.append(f"  per-trade   : naked {a['net']/max(a['n'],1):+.2f} pts  vs  "
             f"B2 {B2_BASELINE['net']/B2_BASELINE['n']:+.2f} pts")
    L.append("")

    out = OUT_DIR / "naked_20m_range_break_sweep.txt"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")
    print(f"     {OUT_DIR/'sweep_grid.csv'}")
    print(f"     {OUT_DIR/'best_config_trades.csv'}")
    print(f"total {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    if "--confirm" in sys.argv:
        confirm()
    else:
        main()
