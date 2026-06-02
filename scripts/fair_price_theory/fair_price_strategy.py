"""Fair Price Theory — NQ mean-reversion-to-open intraday strategy (1-min bars).

FAIR PRICE = 09:30 cash open (open of the 09:30 bar).  On CPI days = the pre-8:30 price
(close of the 08:29 bar).  Price tends to revert to fair price after deviating.

Brackets (RR ~1.52), by the ENTRY candle's high-low range:
   range >= 25 pt  ->  SL 50 / TP 76
   range <  25 pt  ->  SL 25 / TP 38
Entry candle filter: body (|close-open|) >= 80% of high-low range ("mostly body, barely any wicks").
Reversion entries additionally require range >= 25 pt (=> reversion always uses the 50/76 bracket).

CONTINUATION (09:30-09:40):  above fair -> long on a qualifying BULLISH candle;
                             below fair -> short on a qualifying BEARISH candle.  (no BOS, no space rule)
REVERSION   (09:40-11:00):   above fair -> short toward fair;  below fair -> long toward fair.
   * space rule: TP must not overshoot fair (entry must be >= TP_pts away from fair).
   * candle entries: any qualifying candle of the right color.
   * BOS entries: arm when price closes through the most-recent unbroken pivot (low if above fair /
     high if below fair); then take the first qualifying candle (incl. the breaking one).  Broken
     pivots are ERASED.  Pivots: left=right=3, confirmed with a 3-bar delay (no lookahead),
     seeded from the hour before the fair-price reference, reset every day.

Fills: at the OPEN of the bar AFTER the qualifying candle.  Exits: first-touch on later bars,
SL-first on an inside-both bar; open trades force-closed at 16:00 ET.

Modes compared: 'candle' (cont + candle reversion), 'bos' (cont + BOS reversion),
'combined' (cont + either).  Outputs raw edge stats + per-mode trade logs (results/).

Run:  python scripts/fair_price_theory/fair_price_strategy.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from datetime import time
from pathlib import Path

PARQUET = Path("D:/trading_pythonbacktest_data/markettick_1min_bars.parquet")
OUT = Path(__file__).resolve().parent / "results"
ET = "America/New_York"
NQ_PT = 20.0                       # $/pt at 1 NQ (stats only)
BODY = 0.80                        # body / range
REV_MIN = 25.0                     # reversion min candle range (pt)
BRK = 25.0                         # bracket size threshold (pt)
SL_BIG, TP_BIG, SL_SM, TP_SM = 50., 76., 25., 38.
PIV = 3                            # pivot left/right
TIE_SL_FIRST = True

# BLS CPI release dates (8:30 ET) 2020-12..2026-05 — VERIFY/replace if needed (CPI-only news rule).
CPI = {  # year: [ (m,d), ... ]
 2020: [(12,10)],
 2021: [(1,13),(2,10),(3,10),(4,13),(5,12),(6,10),(7,13),(8,11),(9,14),(10,13),(11,10),(12,10)],
 2022: [(1,12),(2,10),(3,10),(4,12),(5,11),(6,10),(7,13),(8,10),(9,13),(10,13),(11,10),(12,13)],
 2023: [(1,12),(2,14),(3,14),(4,12),(5,10),(6,13),(7,12),(8,10),(9,13),(10,12),(11,14),(12,12)],
 2024: [(1,11),(2,13),(3,12),(4,10),(5,15),(6,12),(7,11),(8,14),(9,11),(10,10),(11,13),(12,11)],
 2025: [(1,15),(2,12),(3,12),(4,10),(5,13),(6,11),(7,15),(8,12),(9,11),(10,15),(11,13),(12,10)],
 2026: [(1,14),(2,11),(3,11),(4,10),(5,12)],
}
CPI_DATES = {pd.Timestamp(y, m, d).date() for y, ds in CPI.items() for m, d in ds}


def load():
    d = pd.read_parquet(PARQUET, columns=["open", "high", "low", "close"])
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    d.index = d.index.tz_convert(ET)
    # The parquet is RIGHT-labeled (timestamp = bar CLOSE): verified via the 17:00 ET maintenance
    # halt (last stamp 17:00, resume 18:01) and the cash-open volume surge landing on stamp 09:31.
    # Shift -1 min to LEFT-labeled (timestamp = bar OPEN) so it matches TradingView and the engine's
    # logic (fair price = open of the 09:30 bar = true 9:30 cash open).
    d.index = d.index - pd.Timedelta(minutes=1)
    return d.sort_index()


def qualifies(o, c, h, l, want_bull):
    rng = h - l
    if rng <= 0:
        return False, 0.0
    body = abs(c - o)
    if body < BODY * rng:
        return False, rng
    if want_bull and not (c > o):
        return False, rng
    if (not want_bull) and not (c < o):
        return False, rng
    return True, rng


def bracket(rng):
    return (SL_BIG, TP_BIG) if rng >= BRK else (SL_SM, TP_SM)


def day_signals(g: pd.DataFrame):
    """Return list of signal dicts for one session date (chronological)."""
    cpi = g.index[0].date() in CPI_DATES
    o = g["open"].values; h = g["high"].values; l = g["low"].values; c = g["close"].values
    t = g.index
    mins = t.hour * 60 + t.minute
    # fair price
    if cpi:
        idx829 = np.where(mins == 8 * 60 + 29)[0]
        if len(idx829) == 0:
            return [], cpi
        fair = c[idx829[0]]; pre_start = np.where(mins == 7 * 60 + 30)[0]
    else:
        idx930 = np.where(mins == 9 * 60 + 30)[0]
        if len(idx930) == 0:
            return [], cpi
        fair = o[idx930[0]]; pre_start = np.where(mins == 8 * 60 + 30)[0]
    ps = pre_start[0] if len(pre_start) else 0
    n = len(g)
    cont_lo, cont_hi, rev_hi = 9 * 60 + 30, 9 * 60 + 40, 11 * 60
    # pivot state
    piv_lows = []   # [idx, price, broken]
    piv_highs = []
    bos_short_arm = bos_long_arm = False
    sigs = []
    seen_fill = set()
    for i in range(ps, n - 1):           # need i+1 for fill; stop entries at 11:00 anyway
        m = mins[i]
        # confirm pivot centered at i-PIV (known now, no lookahead)
        cdt = i - PIV
        if cdt - PIV >= 0:
            wl, wr = slice(cdt - PIV, cdt), slice(cdt + 1, cdt + PIV + 1)
            if h[cdt] > h[wl].max() and h[cdt] > h[wr].max():
                piv_highs.append([cdt, h[cdt], False])
            if l[cdt] < l[wl].min() and l[cdt] < l[wr].min():
                piv_lows.append([cdt, l[cdt], False])
        above = c[i] > fair
        in_cont = cont_lo <= m < cont_hi
        in_rev = cont_hi <= m < rev_hi
        # --- break detection (erase + arm) ---
        mrl = next((p for p in reversed(piv_lows) if not p[2]), None)
        mrh = next((p for p in reversed(piv_highs) if not p[2]), None)
        if mrl and c[i] < mrl[1]:
            mrl[2] = True
            if in_rev and above:
                bos_short_arm = True
        if mrh and c[i] > mrh[1]:
            mrh[2] = True
            if in_rev and (not above):
                bos_long_arm = True
        # entries fill on i+1 open
        fo = o[i + 1]
        # --- CONTINUATION ---
        if in_cont:
            if above:
                ok, rng = qualifies(o[i], c[i], h[i], l[i], want_bull=True)
                if ok:
                    sl, tp = bracket(rng)
                    sigs.append(dict(fill=i + 1, dir=1, sl=sl, tp=tp, kind="CONT", cand=True, bos=False,
                                     fair=fair, rng=rng, above=above))
            else:
                ok, rng = qualifies(o[i], c[i], h[i], l[i], want_bull=False)
                if ok:
                    sl, tp = bracket(rng)
                    sigs.append(dict(fill=i + 1, dir=-1, sl=sl, tp=tp, kind="CONT", cand=True, bos=False,
                                     fair=fair, rng=rng, above=above))
        # --- REVERSION ---
        if in_rev:
            if above:   # want SHORT toward fair
                ok, rng = qualifies(o[i], c[i], h[i], l[i], want_bull=False)
                if ok and rng >= REV_MIN:
                    sl, tp = bracket(rng)
                    space = (fo - tp) >= fair          # TP (entry-tp) must not go below fair
                    if space:
                        is_cand = True
                        is_bos = bos_short_arm
                        sigs.append(dict(fill=i + 1, dir=-1, sl=sl, tp=tp, kind="REV",
                                         cand=is_cand, bos=is_bos, fair=fair, rng=rng, above=above))
                        if is_bos:
                            bos_short_arm = False
            else:       # want LONG toward fair
                ok, rng = qualifies(o[i], c[i], h[i], l[i], want_bull=True)
                if ok and rng >= REV_MIN:
                    sl, tp = bracket(rng)
                    space = (fo + tp) <= fair
                    if space:
                        is_bos = bos_long_arm
                        sigs.append(dict(fill=i + 1, dir=1, sl=sl, tp=tp, kind="REV",
                                         cand=True, bos=is_bos, fair=fair, rng=rng, above=above))
                        if is_bos:
                            bos_long_arm = False
    # attach arrays for evaluation
    return sigs, (o, h, l, c, t, mins)


def eval_trade(arrs, fill, d, sl_pts, tp_pts):
    o, h, l, c, t, mins = arrs
    n = len(o)
    entry = o[fill]
    if d == 1:
        sl, tp = entry - sl_pts, entry + tp_pts
    else:
        sl, tp = entry + sl_pts, entry - tp_pts
    # force-close index = last bar with minute <= 16:00 that day
    fc = n - 1
    for j in range(fill, n):
        if mins[j] > 16 * 60:
            fc = j - 1; break
    fc = max(fc, fill)
    for j in range(fill, fc + 1):
        hi, lo = h[j], l[j]
        if d == 1:
            hit_sl, hit_tp = lo <= sl, hi >= tp
        else:
            hit_sl, hit_tp = hi >= sl, lo <= tp
        if hit_sl and hit_tp:
            return (-sl_pts if TIE_SL_FIRST else tp_pts), j, ("SL" if TIE_SL_FIRST else "TP")
        if hit_sl:
            return -sl_pts, j, "SL"
        if hit_tp:
            return tp_pts, j, "TP"
    exitp = c[fc]
    return (exitp - entry) * d, fc, "FC"


def run(mode):
    """mode in {'candle','bos','combined'}; returns DataFrame of serial single-account trades."""
    df = load()
    df["date"] = df.index.date
    rows = []
    for date, g in df.groupby("date", sort=True):
        sigs, arrs = day_signals(g)
        if not arrs:
            continue
        # filter signals per mode
        sel = []
        for s in sigs:
            if s["kind"] == "CONT":
                sel.append(s); continue            # continuation in every mode
            if mode == "cont_only":
                continue
            take = (mode == "candle" and s["cand"]) or (mode == "bos" and s["bos"]) \
                or (mode == "combined" and (s["cand"] or s["bos"]))
            if take:
                sel.append(s)
        # single-account serial: take if flat (entry fill > last exit idx)
        last_exit = -1
        for s in sorted(sel, key=lambda x: x["fill"]):
            if s["fill"] <= last_exit:
                continue
            pnl_pts, jx, reason = eval_trade(arrs, s["fill"], s["dir"], s["sl"], s["tp"])
            last_exit = jx
            rows.append(dict(date=date, fill_idx=s["fill"], dir=s["dir"], kind=s["kind"],
                             bos=s["bos"], sl=s["sl"], tp=s["tp"], pnl_pts=round(pnl_pts, 2),
                             reason=reason))
    return pd.DataFrame(rows)


def stats(name, t: pd.DataFrame):
    if t.empty:
        print(f"{name:>9}: no trades"); return
    wins = (t.pnl_pts > 0).sum(); n = len(t)
    wr = wins / n
    pnl_usd = t.pnl_pts * NQ_PT
    exp = pnl_usd.mean()
    gross_w = pnl_usd[pnl_usd > 0].sum(); gross_l = -pnl_usd[pnl_usd < 0].sum()
    pf = gross_w / gross_l if gross_l else np.inf
    rmix = t.reason.value_counts(normalize=True)
    days = t.date.nunique()
    print(f"{name:>9}: n={n:4d}  wr={wr*100:4.1f}%  exp=${exp:6.1f}/1NQ  PF={pf:4.2f}  "
          f"net=${pnl_usd.sum():8.0f}  TP/SL/FC={rmix.get('TP',0)*100:4.0f}/"
          f"{rmix.get('SL',0)*100:4.0f}/{rmix.get('FC',0)*100:4.0f}%  trd-days={days}  "
          f"({n/max(days,1):.2f}/day)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Fair Price Theory — single-account serial (1 position at a time), 1 NQ stats\n")
    print(f"{'mode':>9}  edge summary")
    for mode in ("candle", "bos", "combined"):
        t = run(mode)
        t.to_csv(OUT / f"fpt_{mode}_trades.csv", index=False)
        stats(mode, t)
        # split continuation vs reversion
        stats(f"{mode[:3]}/cont", t[t.kind == "CONT"])
        stats(f"{mode[:3]}/rev", t[t.kind == "REV"])
        print()
    print(f"trade logs -> {OUT}")


if __name__ == "__main__":
    main()
