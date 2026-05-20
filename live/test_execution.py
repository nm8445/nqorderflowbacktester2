"""
Test execution mechanics with auto-generated signals.

Runs live trader in test mode:
- Connects to live feed
- Builds bars in real-time
- After 20 bars, auto-sends a BUY signal to NT8
- Tests stop/target/breakeven without waiting for real orderflow setups

Run:
    python live/test_execution.py [--paper] [--bars N]

Flags:
    --paper    Paper mode - logs signals but doesn't send to NT8
    --bars N   Send test signal after N bars (default: 20)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from live.live_trader import LiveTrader
import logging

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test NQ order execution")
    parser.add_argument("--paper", action="store_true", help="Paper mode - no orders sent")
    parser.add_argument("--bars", type=int, default=20, help="Send signal after N bars")
    args = parser.parse_args()

    trader = LiveTrader(paper=args.paper)

    # Enable test mode
    trader.engine.enable_test_mode(signal_after_bars=args.bars)

    log.info("="*60)
    log.info("EXECUTION TEST MODE")
    log.info(f"Will auto-send BUY signal after {args.bars} bars close")
    log.info("Make sure NT8 Add-On is running with Auto Execute + Test Mode enabled")
    log.info("="*60)

    trader.start()


if __name__ == "__main__":
    main()
