"""Smoke test: build bars, compute signals, run one backtest config end-to-end."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from core import (build_bars, extract_arrays, compute_atr, compute_zvol, compute_ema,
                  backtest_jit, calc_metrics, warmup_jit,
                  SESSION_START_MIN, SESSION_END_MIN, MAX_TRADES_PER_DAY, IS_END)


def main():
    print("warmup JIT...")
    t0 = time.time()
    warmup_jit()
    print(f"  done in {time.time()-t0:.1f}s")

    for bar_minutes in (15, 20):
        print(f"\n=== {bar_minutes}-min bars ===")
        t0 = time.time()
        df = build_bars(bar_minutes)
        print(f"build_bars: {len(df)} bars in {time.time()-t0:.1f}s")
        print(f"  range: {df.index[0]} -> {df.index[-1]}")

        arr = extract_arrays(df)
        t0 = time.time()
        atr = compute_atr(arr["highs"], arr["lows"], arr["closes"])
        print(f"compute_atr: {time.time()-t0:.2f}s")

        # Pick one config
        norm, zlook, ema_len, high_z, atr_sl, atr_tp = 300, 100, 40, 1.5, 2.0, 1.2

        t0 = time.time()
        z_vol = compute_zvol(arr["closes"], norm, zlook)
        print(f"compute_zvol: {time.time()-t0:.2f}s")
        ema = compute_ema(arr["closes"], ema_len)

        of_mask = np.ones(len(df), dtype=np.int8)  # all pass
        g_sign = np.zeros(len(df), dtype=np.int8)

        t0 = time.time()
        pnls, in_is = backtest_jit(
            arr["highs"], arr["lows"], arr["closes"],
            z_vol, ema, atr,
            arr["minutes_of_day"], arr["day_idx"],
            of_mask, g_sign,
            high_z, atr_sl, atr_tp,
            SESSION_START_MIN, SESSION_END_MIN, MAX_TRADES_PER_DAY,
            0, arr["is_end_ord"],
        )
        bt_time = time.time() - t0
        print(f"backtest_jit: {bt_time*1000:.1f}ms, {len(pnls)} trades")

        metrics = calc_metrics(pnls, in_is)
        for k in ("is", "oos", "total"):
            m = metrics[k]
            print(f"  {k.upper():>5}: trades={m['trades']:>5}  PF={m['pf']:.2f}  "
                  f"WR={m['wr']:.1f}%  PnL=${m['pnl']:+,.0f}  MDD=${m['mdd']:,.0f}")


if __name__ == "__main__":
    main()
