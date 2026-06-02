# Personal Account Scaling — Live 4-Way Combined (OD+RV+B2+FB)

How much of YOUR OWN capital to put behind the strategy per MNQ, and what it returns. No prop
rules — pure risk-of-ruin / position sizing. Monte-Carlo bootstrap of the 4-way daily P&L
(`combined_4way_trades.csv`, 1 NQ = 10 MNQ), 20k simulated years.
Script: `scripts/montecarlo/personal_account_scaling.py`.

## Strategy stats (1 NQ basis)
- ~$451/day mean | worst day −$14,104 | historical max drawdown −$28,528 (over 5.6 yrs).
- Costs $4/MNQ round-turn; overnight margin assumed ~$1,700/MNQ.

## Safe capital + annual return by size
Safe capital = **2× the p99 yearly max drawdown** (so even a 1-in-100 bad year only draws ~50% of
equity and you keep trading). Return is FLAT ~60%/yr at every size — it scales linearly.

| MNQ | E[annual $] | p99 yearly max-DD | **SAFE capital** | **annual return** |
|-----|-------------|-------------------|------------------|-------------------|
| 1   | $8,928      | $7,484            | ~$15,000         | ~60% |
| 2   | $17,722     | $14,704           | ~$29,400         | ~60% |
| 3   | $26,722     | $22,029           | ~$44,100         | ~61% |
| 5   | $44,901     | $37,078           | ~$74,200         | ~61% |
| 10  | $89,689     | $73,814           | ~$147,600        | ~61% |
| 20  | $177,761    | $148,958          | ~$297,900        | ~60% |
| 30  | $267,798    | $227,615          | ~$455,200        | ~59% |

## Rule of thumb
- **~$15k of capital per MNQ → ~$8.9k/yr (~60% return).** Scale linearly: want $45k/yr? Run ~5 MNQ
  on a ~$74k account.
- Risk tolerance knob: 1× p99 DD = aggressive (full-send, deep drawdowns), 2× = balanced
  (recommended), 3× = conservative (~$22k/MNQ, shallow drawdowns, ~40% return).

## Caveats
- Uses the 4-way log with OD martingale ON (conservative). Marti is now OFF in live -> ~10–15% less
  capital needed for the same safety.
- Bootstrap IID-resamples days (breaks autocorrelation); tail-day clustering makes p99 DD worse than
  the benign historical path, which is the safe assumption.
- Assumes the edge persists. 60%/yr is a leveraged-futures return — sized for survival, not comfort;
  expect 50%-equity drawdowns in a bad year at the 2× sizing.
