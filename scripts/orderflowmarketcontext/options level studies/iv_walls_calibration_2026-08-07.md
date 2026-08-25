# IV Walls — expected-move bands as reversal levels (2026-08-07)

## Question

Can option-implied vol + historical vol build "walls" that NQ rarely **closes** outside of,
and do touches of those walls act as **reversal** points?

## Construction

```
anchor(D)   = NQ close at 17:00 ET on day D
EM(D)       = ATM_IV(D) * spot(D) * sqrt(1/252)      # MenthorQ 1D expected move
wall_k      = anchor(D) +/- k * EM(D)                # evaluated on session D+1
```

ATM IV / EM read from `menthorq_levels_nq.parquet` (`ndx_iv`, `ndx_em` — NDX-native, no ratio
scaling). The chain settles 17:15 ET on D, so applying it to D+1 is lookahead-free; the frame is
shifted one session explicitly. NQ sessions from `markettick_1min_bars.parquet` (right-labeled,
shifted −1 min), RTH 09:30→17:00 ET.

HV comparator: identical formula with ATM IV replaced by trailing 20-day close-to-close vol.

**Sample**: 1,281 sessions, 2021-02-11 → 2026-06-18. Mean ATM IV 0.213, mean HV20 0.214,
**median IV/HV = 0.99**, mean 1-day EM = 227 NQ points.

## 1. The walls are real as a containment statistic

| k | wall ± pts | P(close outside), IV band | P(close outside), HV band | P(touch) |
|---|--:|--:|--:|--:|
| 0.75 | 171 | 44.7% | 44.3% | 74.0% |
| **1.00** | **227** | **31.5%** | 32.6% | 54.9% |
| 1.25 | 284 | 21.0% | 20.7% | 37.0% |
| **1.50** | **341** | **13.6%** | 12.8% | 25.1% |
| 2.00 | 455 | 6.9% | 5.3% | 12.2% |
| 2.50 | 568 | 3.0% | 1.8% | 5.3% |

So "rarely closes outside" = **k ≈ 1.5 (13.6%)** or k ≈ 2.0 (6.9%). Calibrate empirically — a
normal assumption would put k=1.5 at 13.4% and k=2.0 at 4.6%, so the upper tail is fatter than
Gaussian and gets worse the further out you go.

**The IV band is not better than the HV band.** Close-out rates are within ~1pt everywhere, and
at k≥1.75 the HV band is actually *tighter-fitting* (5.3% vs 6.9% at k=2.0). With median IV/HV
= 0.99 there is no usable variance risk premium at this horizon, so option pricing buys nothing
over the realized-vol bands already built in `overnight_band_gamma_2026-05-04.md`.

## 2. Touching a wall does NOT produce rejection

| k | n touch | closed back inside | up-touch rejected | down-touch rejected |
|---|--:|--:|--:|--:|
| 1.00 | 703 | 42.5% | 39.3% | 48.1% |
| 1.50 | 321 | 45.8% | 42.7% | 49.2% |
| 2.00 | 156 | 43.6% | 32.8% | 50.5% |

Once price reaches the upper wall it closes **beyond** it ~60% of the time, and that gets *more*
extreme further out (67% at k=2.0). This is continuation, not rejection.

## 3. Fading the touch loses money — significantly

Baseline anchor→close drift: **+11.4 pts/day**.

| k | side | n | P(win) | mean pts | t |
|---|---|--:|--:|--:|--:|
| 1.00 | short upper | 354 | 39.3% | **−28.3** | **−3.13** |
| 1.00 | long lower | 364 | 48.1% | −17.6 | −1.76 |
| 1.50 | short upper | 143 | 42.7% | −32.7 | −2.47 |
| 1.50 | long lower | 181 | 49.2% | −3.6 | −0.24 |

Every fade cohort is negative; the short side is significantly so.

## 4. Regime conditioning does not rescue it

| cohort | side | n | P(win) | mean pts | t |
|---|---|--:|--:|--:|--:|
| pos-gamma | short | 59 | 40.7% | −23.5 | −1.40 |
| pos-gamma | long | 67 | 50.7% | +0.5 | 0.03 |
| neg-gamma | short | 295 | 39.0% | −29.2 | −2.83 |
| IV/HV high | short | 164 | 47.0% | −4.2 | −0.30 |
| IV/HV high | long | 138 | 51.4% | +9.7 | 0.64 |
| **IV/HV low** | **short** | 190 | 32.6% | **−49.0** | **−4.27** |
| IV/HV low | long | 226 | 46.0% | −34.2 | −2.62 |

Even pos-gamma — the regime that measurably compresses intraday range
(`hvl_0dte_meanreversion`: |ret| 0.66% vs 0.98%, p<0.00005) — does not make the fade work.

## 5. The inverse IS the edge

The fade numbers mirror exactly into a continuation trade (enter at wall touch, exit 17:00):

| cohort | side | n | P(win) | mean pts | t |
|---|---|--:|--:|--:|--:|
| all, k=1.0 | **long upper touch** | 354 | **60.7%** | **+28.3** | **+3.13** |
| all, k=1.0 | short lower touch | 364 | 51.9% | +17.6 | +1.76 |
| **IV/HV low** | **long upper touch** | 190 | **67.4%** | **+49.0** | **+4.27** |
| IV/HV low | short lower touch | 226 | 54.0% | +34.2 | +2.62 |

**IV/HV < 1 (implied cheap vs realized = vol expanding) is the amplifier for wall breaks.**

## Verdict

The premise is inverted by the data: EM walls are a good *containment* statistic and a poor
*reversal* signal. Touch → continuation, ~60/40, and the strongest cell is a breakout in the
low-IV/HV regime. This reproduces the existing band-break result (+37.4 pts, 65.3% on the
HV-band long break) rather than improving on it — **IV adds no incremental information over
the realized-vol bands already in production.**

Use the walls for: stop/target placement, sizing, and as continuation targets. Not for fading.

Script: `scripts/iv_walls_calibration.py`.
Related: `overnight_band_gamma_2026-05-04.md`, `band_break_with_qqq_levels_2026-05-04.md`,
`hvl_0dte_meanreversion_2026-05-04.md`.

## Caveats

- Exit is always the 17:00 close; no intrabar stop/target path is modelled, so these are
  signal-quality numbers, not a tradeable P&L.
- Entry is assumed at the wall price on a touch (daily high/low), which is optimistic on fills.
- IV/HV compares a 1-day ATM IV against a 20-day realized window — horizons are mismatched, so
  read IV/HV as a vol-regime tag, not a clean variance-premium estimate.
- No overfit-framework pass, no direction-permutation control. The continuation cells reproduce
  an independently-measured prior result, which is the main reason to believe them.
