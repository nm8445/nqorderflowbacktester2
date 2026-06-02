"""Trace the 2026-04-07 BOS long: bars, body/range, confirmed pivots (3-bar delay), break, fire."""
from __future__ import annotations
import numpy as np
from fair_price_strategy import load, BODY, REV_MIN, PIV

DATE = "2026-04-07"


def main():
    df = load()
    g = df[df.index.date == np.datetime64(DATE).astype("datetime64[D]").astype(object)]
    o = g["open"].values; h = g["high"].values; l = g["low"].values; c = g["close"].values
    t = g.index; mins = t.hour * 60 + t.minute
    fair = o[np.where(mins == 9 * 60 + 30)[0][0]]
    print(f"{DATE}  fair (09:30 open) = {fair:.2f}   (below fair => reversion LONG, break of pivot HIGH)\n")
    ps = np.where(mins == 8 * 60 + 30)[0][0]
    piv_highs = []  # [center_idx, price, broken]
    print(f"{'time':>6} {'open':>9} {'high':>9} {'low':>9} {'close':>9} {'rng':>5} {'body%':>6} "
          f"{'dir':>4} {'qual?':>5} {'pivHIGH@confirm':>16} {'MRU pivHigh':>12} {'break?':>7} {'arm':>4}")
    bos_long_arm = False
    for i in range(ps, len(g) - 1):
        m = mins[i]
        # confirm pivot centered at i-PIV
        newp = ""
        cdt = i - PIV
        if cdt - PIV >= 0:
            wl, wr = slice(cdt - PIV, cdt), slice(cdt + 1, cdt + PIV + 1)
            if h[cdt] > h[wl].max() and h[cdt] > h[wr].max():
                piv_highs.append([cdt, h[cdt], False])
                newp = f"{t[cdt].strftime('%H:%M')}={h[cdt]:.2f}"
        mrh = next((p for p in reversed(piv_highs) if not p[2]), None)
        rng = h[i] - l[i]; body = abs(c[i] - o[i]); ratio = (body / rng) if rng else 0
        bull = c[i] > o[i]
        below = c[i] < fair
        in_rev = (9 * 60 + 40) <= m < (11 * 60)
        brk = ""
        if mrh and c[i] > mrh[1]:
            mrh[2] = True
            brk = f"brk {mrh[1]:.2f}"
            if in_rev and below:
                bos_long_arm = True
        qual = bull and ratio >= BODY and rng >= REV_MIN
        fired = ""
        if in_rev and below and bos_long_arm and qual:
            # space check
            sl, tp = 50., 76.
            fo = o[i + 1]
            if fo + tp <= fair:
                fired = f"  <== FIRE LONG fill {t[i+1].strftime('%H:%M')} @ {fo:.2f}"
                bos_long_arm = False
        if m < 10 * 60 + 25 or m > 10 * 60 + 40:  # focus window
            continue
        print(f"{t[i].strftime('%H:%M'):>6} {o[i]:>9.2f} {h[i]:>9.2f} {l[i]:>9.2f} {c[i]:>9.2f} "
              f"{rng:>5.1f} {ratio*100:>5.0f}% {'BULL' if bull else 'BEAR':>4} {str(qual):>5} "
              f"{newp:>16} {(f'{mrh[1]:.2f}' if mrh else '-'):>12} {brk:>7} {str(bos_long_arm):>4}{fired}")


if __name__ == "__main__":
    main()
