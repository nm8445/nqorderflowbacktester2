# Logs

Script execution logs and cache build outputs.

## Log Types

### Backtest Logs
- `backtest_*.log` - Backtest execution logs
- `atr_backtest_*.log` - ATR optimization logs

### Cache Build Logs
- `build_*.log` - Signal cache build logs
- `rebuild_*.log` - Cache rebuild logs

## Usage

Logs are automatically generated when running:
- Cache building scripts
- Long-running backtests
- Optimization scripts

View logs:
```bash
tail -f logs/build_inverted.log
cat logs/backtest_normal.log
```

## Cleanup

Old logs can be safely deleted. Scripts will create new logs as needed.
