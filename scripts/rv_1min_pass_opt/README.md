# RV 1-min pass-rate optimization

Self-contained study: a **separate** rough-vol config on the **1-min** timeframe, optimized for
**futures prop-firm pass rate** (decorrelated bulk evals, copies=1), NOT live PF. 1:1 ATR bracket
sized to $1,500 risk; session 19:00 ET (Asia) → 16:00 ET with a hard 16:00 force-close. Eval rules =
the live farm `futures_50` plan (target $3k, $2k trailing-then-lock DD, 50% consistency).

Everything for this study lives in THIS folder.

## Layout
```
features.py    vectorized rough-vol features (matches live RVFeatures to 1e-8)
data.py        load_bars() + build_orderflow_flags() (config-independent flag cache)
backtest.py    1-min entry/exit sim -> trades (r, pnl_$, mae_$); session + hard 16:00 FC + seam suppress
eval_mc.py     decorrelated futures_50 eval Monte Carlo (true_target, floor lock, floating blow, day cap)
stage0.py      harness validation + throughput + signals/day-vs-HIGH_Z
stage1.py      NORM_LEN x Z_LOOKBACK lookback sweep
cache/         orderflow flag caches (orderflow_flags_n{N}_d{D}.parquet)
results/       per-stage CSV outputs
```
Underlying data (shared, on D:): `markettick_1min_bars.parquet`, `volumetric_1min_1tpl.parquet`
(both backfilled to 2026-06-19; Jun15-19 patched to liquid NQU6).

## Optimization order (lock each stage; OOS 40% touched once at the end)
0. harness + 60/40 split (IS 2020-12-01→2024-04-05, OOS →2026-06-19)   [DONE]
1. NORM_LEN x Z_LOOKBACK                                                [DONE — no edge: ~51% gross win ceiling]
2. kernel H, KERNEL_LEN
3. entry gate HIGH_Z + EMA_LEN
4. bracket k (=ATR mult) + ATR_LEN
5. orderflow WINDOW_N_TICKS, WINDOW_D
6. ATR_MAX + session + MAX_TRADES/day
7. full pass-rate MC on IS
8. OOS validation

## Key findings so far
- Throughput is ample: ~6 trades/day after orderflow (not the constraint).
- Orderflow WINDOW_D must drop 150 -> ~40 for 1-min (volume scales ~20x with bar size).
- COST is binding: $1500 / tiny 1-min ATR = many MNQ; small k is cost-crippled.
- **Stage 1: the rough-vol lookbacks contain no directional edge on 1-min — every cell ~50-51%
  gross win rate (a coinflip), net-negative after cost.** The signal predicts vol, not direction.
