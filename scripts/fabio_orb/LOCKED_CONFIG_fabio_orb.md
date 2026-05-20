# LOCKED CONFIG — Fabio ORB (Live)

**Strategy ID:** `fabio_orb_v1`
**Locked on:** 2026-05-17
**Backtest source:** `scripts/fabio_orb/run_final_config.py` (Mode A)
**Performance baseline (5.4 years, 2020-12 → 2026-05, NQ 5-min):**
- 709 trades, 53.7% wins, **$157,965 net** (gross of fees: $182,330), **PF 1.347**, MaxDD -$20,240
- IS (60%): $44,180 / PF 1.16  ·  OOS (40%): $113,785 / PF 1.61

---

## Instrument & Timeframe
- **Symbol:** NQ (continuous front-month)
- **Bar size:** 5-minute time bars (ET clock)
- **Session reference:** ET (America/New_York)

## Inputs (locked)
| Param | Value |
|---|---|
| `ORB_Start_H_NY` | 8 |
| `ORB_Start_M_NY` | 30 |
| `ORB_Dur_Min` | 30 |
| `Trade_End_H_NY` | 14 |
| `Trade_End_M_NY` | 0 |
| `TP_RR_Ratio` | 4.0 |
| `Num_Contracts` | 1 |
| `DeltaThreshold` | 300  (contracts; `buy_vol − sell_vol`) |
| `UseCumulativeDelta` | false |
| `N_ConfirmCloses` | 4 |
| `SkipBucket_HHMM` | 930  (skip entries on the bar that closes at 09:30 ET) |

## Direction
**Long-only.**

## Opening Range
Built from bars whose **close time** is in (08:30, 09:00] ET (6 bars: closes at 08:35, 08:40, …, 09:00).
- `ORB_High` = max(high) over those bars
- `ORB_Low`  = min(low) over those bars

## Entry Rules (all must be true)
1. Bar close time is in (09:00, 14:00] ET.
2. **N=4 consecutive closes** above `ORB_High` (current bar plus 3 prior).
3. **Skip** any entry whose entry bar closes at exactly **09:30 ET** (`hhmm == 930`).
4. **Delta filter** on the entry bar only: `buy_vol − sell_vol ≥ 300` contracts.
5. `ORB_Low < Close` (the standard sanity check — entry above the stop).

If all five hold, enter LONG at the close of the current bar.

## Exit Rules
- **SL:** sell at `ORB_Low` (static, set at entry, never moved).
- **TP:** sell at `Entry + 4.0 × (Entry − ORB_Low)` (limit). In practice almost never hit (~0.3% of trades).
- **EOD:** if still long at the bar that closes at or after 14:00 ET, exit at that bar's close.

If both SL and TP are touched on the same bar, conservatively assume SL fills first.

## Costs assumed in backtest (for reference)
- 1 tick slippage per side (round-trip = 2 ticks = 0.5 pt = $10)
- $5 round-trip commission per contract

## Data inputs needed live
- 5-min bars with: `open, high, low, close, buy_vol, sell_vol`
- `buy_vol`/`sell_vol` must be aggressor-classified (NOT L2-depth — Lee-Ready or feed-level trade tape)

## Pseudocode (live evaluation, called each closed bar after 09:00)
```python
def fabio_orb_signal(bars_today_so_far) -> bool:
    """Return True to enter long at the close of the most recent bar.

    bars_today_so_far: list of dicts ordered by close_time, each with
        keys: close_time_et, hhmm, open, high, low, close, buy_vol, sell_vol
        (only bars from today's session, ET).
    """
    # 1) Build ORB from 08:30 → 09:00 ET window
    orb_bars = [b for b in bars_today_so_far if ORB_START_HHMM < b["hhmm"] <= ORB_END_HHMM]
    if not orb_bars:
        return False
    orb_high = max(b["high"] for b in orb_bars)
    orb_low  = min(b["low"]  for b in orb_bars)

    # 2) Identify candidate entry bar (the most recent closed bar)
    post_orb = [b for b in bars_today_so_far if ORB_END_HHMM < b["hhmm"] <= TRADE_END_HHMM]
    if len(post_orb) < N_CONFIRM:
        return False
    entry_bar = post_orb[-1]

    # 3) Skip 09:30 bucket
    if entry_bar["hhmm"] == SKIP_BUCKET_HHMM:
        return False

    # 4) Require N=4 consecutive closes above ORB_High (entry bar + 3 prior)
    confirm = post_orb[-N_CONFIRM:]
    if not all(b["close"] > orb_high for b in confirm):
        return False

    # 5) Delta filter on entry bar only
    if entry_bar["buy_vol"] - entry_bar["sell_vol"] < DELTA_THRESHOLD:
        return False

    # 6) Sanity check
    if orb_low >= entry_bar["close"]:
        return False

    return True   # caller enters long at entry_bar["close"]; SL=orb_low; TP=entry + 4*(entry-orb_low)
```

## Failure modes / things to monitor live
- **Wide ORB days**: avg risk = 115 pts ($2,300). On 1-contract NQ, a single SL = -$2.3k. Plan sizing accordingly.
- **EOD-drift dependence**: 72% of trades exit at 14:00 EOD — the alpha is the slow drift into the close, NOT the 4R TP. If the post-2022 drift regime breaks, this strategy will go flat (it already did in 2022: PF 1.00, $105 for the year).
- **Skipped 9:30 bucket**: 9 trades historically per year had `hhmm==930`. They were structurally bad (-$6.5k aggregate) but it's a structural quirk; keep an eye on whether the regime changes.
- **Aggressor classification accuracy**: live `buy_vol`/`sell_vol` must come from a real trade-tape feed (e.g. Databento `mbp-1`), not L2-depth aggregation. Wrong delta will break the entry filter.
