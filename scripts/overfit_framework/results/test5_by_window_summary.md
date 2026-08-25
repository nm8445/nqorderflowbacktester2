# Direction permutation BY WINDOW (2026-08-09)

`scripts/overfit_framework/test_5_by_window.py`, 1000 perms, IS/OOS split 2024-01-01.
Entry bars and exit engine held fixed; only the DIRECTION decision is replaced by a coin flip and
the exits re-run. Real should sit in the top 1% of the permutation distribution.

| strat | window | real $ | perm median | beats | p | verdict |
|---|---|--:|--:|--:|--:|---|
| FB | IS | 52,250 | 22,565 | 87.3% | 0.127 | FAIL |
| **FB** | **OOS** | **78,965** | 8,220 | **99.7%** | **0.003** | **PASS** |
| **RV** | **IS** | **96,131** | −3,256 | **99.9%** | **0.001** | **PASS** |
| RV | OOS | 58,615 | 9,238 | 95.4% | 0.046 | weak |
| OD | IS | 27,205 | 9,435 | 74.9% | 0.251 | FAIL |
| **OD** | **OOS** | **106,715** | 14,250 | **99.3%** | **0.007** | **PASS** |

**Every strategy passes in exactly one window and fails in the other.** No leg has a direction
edge that is significant in both halves.

Note the permutation medians are all positive — the exit engine makes money on random directions,
which is precisely why raw t-vs-zero on trade PnL overstates significance.

## The OD/FB passes are confounded with market drift

OD and FB are **long-only**, so "real" = all-long and the test is really "did being long beat
coin-flipping long/short". That is close to guaranteed in a bull market:

| window | NQ drift | FB all-long | FB all-short | long−short spread |
|---|--:|--:|--:|--:|
| IS (2020-12→2023-12) | **+10.7%/yr** | 52,250 | −4,305 | 56,555 |
| OOS (2024-01→2026-06) | **+26.8%/yr** | 78,965 | −61,870 | **140,835** |

NQ's annualised drift **2.5x'd** from IS to OOS, and FB's long−short spread 2.5x'd with it. The
OOS "PASS" for both long-only legs is substantially a regime artifact, not evidence that entry
TIMING is predictive. The correct control for a long-only strategy is a **timing-randomised long
null** (same day, random entry bar, same exit engine) — not a direction permutation. Not yet run.

OD is the starker case: it enters at 19:00 ET *every* day, so there is no entry selection at all.
Its direction test is purely "was overnight drift positive", and in 2020–2023 it was not
significant (p=0.251).

## RV is the one clean read, and it weakened

RV is bidirectional (479 short / 384 long in the baseline), so direction permutation is a valid
test for it. It goes the *opposite* way to the long-only legs: strong in-sample (p=0.001),
**weak out-of-sample (p=0.046)**. Its direction edge decayed rather than strengthened.

## Not covered

- **B2 was not re-run by window.** It is bidirectional so the test would be meaningful; its
  existing full-sample result was already marginal (p=0.013, FAIL at the 0.01 bar).
- **LIVE cannot support this test.** A counterfactual short requires re-running the engine on the
  same session, which the forward paper log has no way to produce — and at n=115 it would be
  powerless regardless (see `project_oos_live_significance`).

## Bottom line

The earlier combined t=4.57 (p<0.0001) on OOS+live per-trade PnL does **not** survive as evidence
of directional entry edge. What is actually established: the trade distributions are positive, the
exit engines are good, and the long-only legs benefited from a strong bull regime. Only RV showed
a genuine direction edge, and only in-sample.
