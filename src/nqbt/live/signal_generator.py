"""
Live signal generator for NQ Order Flow Strategy.
Connects to Databento live stream and sends signals to NT8.
"""

import os
import asyncio
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import pandas as pd
from dotenv import load_dotenv

import databento as db

# Import strategy components
from nqbt.analysis.range_bars import RangeBar
from nqbt.analysis.range_bars_streaming import StreamingRangeBarBuilder
from nqbt.analysis.volume_profile import VolumeProfile

# Load environment
load_dotenv()

# Constants
TICK_SIZE = 0.25
ET = "America/New_York"
NT8_URL = os.getenv("NT8_URL", "http://localhost:8080/signal")  # NT8 endpoint
DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY")

# Strategy parameters
RANGE_TICKS = 40
TICKS_PER_LEVEL = 5
MIN_DELTA = 30
PROXIMITY_TICKS = 100  # 25 points = 100 ticks
PROFILE_REFRESH_MINUTES = 1
MIN_CLUSTER_DELTA = 20


class LiveSignalGenerator:
    """Generates live trading signals and sends to NT8."""

    def __init__(self):
        self.client = db.Live(key=DATABENTO_API_KEY)

        # State tracking
        self.tick_buffer = []
        self.bar_builder = StreamingRangeBarBuilder(range_ticks=RANGE_TICKS, ticks_per_level=TICKS_PER_LEVEL)
        self.profiles = {}
        self.last_profile_time = None
        self.session_start = None
        self.current_session_date = None

        # Signal tracking
        self.last_signal_time = None
        self.in_position = False

        # Session times (in ET)
        self.trading_start_hour = 9
        self.trading_start_minute = 30
        self.trading_end_hour = 11
        self.trading_end_minute = 0

        print("="*80)
        print("NQ ORDER FLOW - LIVE SIGNAL GENERATOR")
        print("="*80)
        print("Data Source: NQ.c.0 (continuous front month)")
        print("Execution: MNQ (Micro NQ)")
        print(f"NT8 Endpoint: {NT8_URL}")
        print(f"Trading Hours: {self.trading_start_hour}:{self.trading_start_minute:02d} - {self.trading_end_hour}:{self.trading_end_minute:02d} ET")
        print("="*80)
        print()

    def is_trading_hours(self, timestamp):
        """Check if timestamp is within trading hours (9:30-11:00 ET)."""
        et_time = pd.Timestamp(timestamp).tz_convert(ET)

        # Check if weekday
        if et_time.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Check time range
        trading_start = et_time.replace(hour=self.trading_start_hour, minute=self.trading_start_minute, second=0, microsecond=0)
        trading_end = et_time.replace(hour=self.trading_end_hour, minute=self.trading_end_minute, second=0, microsecond=0)

        return trading_start <= et_time < trading_end

    def get_session_date(self, timestamp):
        """Get trading session date (6pm previous day = new session)."""
        et_time = pd.Timestamp(timestamp).tz_convert(ET)

        # If before 6pm, use previous day
        if et_time.hour < 18:
            return et_time.date()
        else:
            return (et_time + timedelta(days=1)).date()

    def update_volume_profile(self, current_time):
        """Update volume profile if refresh interval passed."""
        if not self.tick_buffer:
            return

        # Initialize session start if needed
        if self.session_start is None:
            session_date = self.get_session_date(current_time)
            prev_day = session_date - timedelta(days=1)
            self.session_start = pd.Timestamp(f"{prev_day} 18:00:00", tz=ET).tz_convert("UTC")
            self.current_session_date = session_date

        # Check if new session
        current_session = self.get_session_date(current_time)
        if current_session != self.current_session_date:
            # New session - reset profiles
            print(f"\n[{current_time}] New session started: {current_session}")
            self.profiles = {}
            self.session_start = pd.Timestamp(f"{current_session - timedelta(days=1)} 18:00:00", tz=ET).tz_convert("UTC")
            self.current_session_date = current_session
            self.last_profile_time = None

        # Check if time to refresh profile
        should_refresh = False
        if self.last_profile_time is None:
            should_refresh = True
        else:
            time_diff = (current_time - self.last_profile_time).total_seconds()
            if time_diff >= PROFILE_REFRESH_MINUTES * 60:
                should_refresh = True

        if should_refresh:
            # Build profile from session start to current time
            session_ticks = [t for t in self.tick_buffer if t.index >= self.session_start]

            if len(session_ticks) > 0:
                df_ticks = pd.DataFrame([{
                    'ts_event': t.index,
                    'price': t['price'],
                    'bid_px_00': t['bid_px_00'],
                    'ask_px_00': t['ask_px_00'],
                    'bid_sz_00': t['bid_sz_00'],
                    'ask_sz_00': t['ask_sz_00'],
                    'bid_ct_00': t.get('bid_ct_00', 1),
                    'ask_ct_00': t.get('ask_ct_00', 1),
                } for t in session_ticks])

                df_ticks.set_index('ts_event', inplace=True)

                profile = VolumeProfile.build(df_ticks, ticks_per_level=10, value_area_pct=0.68)
                self.profiles[current_time] = profile
                self.last_profile_time = current_time

                print(f"[{current_time}] Profile updated - VAL: {profile.val:.2f}, POC: {profile.poc:.2f}, VAH: {profile.vah:.2f}")

    def analyze_bar_characteristics(self, bar: RangeBar):
        """Analyze absorption levels in a bar."""
        closed_bearish = bar.close < bar.open
        closed_bullish = bar.close > bar.open

        if not closed_bearish and not closed_bullish:
            return {"absorption_levels": []}

        absorption_levels = []
        for price, lv in bar.levels.items():
            delta = lv.delta
            if closed_bearish and delta >= MIN_DELTA:
                absorption_levels.append((price, delta))
            elif closed_bullish and delta <= -MIN_DELTA:
                absorption_levels.append((price, delta))

        return {"absorption_levels": absorption_levels}

    def detect_signal(self):
        """Detect absorption signal from recent bars."""
        bars = self.bar_builder.get_completed_bars()

        if len(bars) < 2:
            return None

        signal_bar = bars[-2]
        confirm_bar = bars[-1]

        if not signal_bar.closed or not confirm_bar.closed:
            return None

        signal_chars = self.analyze_bar_characteristics(signal_bar)
        signal_bearish = signal_bar.close < signal_bar.open
        signal_bullish = signal_bar.close > signal_bar.open
        confirm_bearish = confirm_bar.close < confirm_bar.open
        confirm_bullish = confirm_bar.close > confirm_bar.open

        has_absorption = len(signal_chars["absorption_levels"]) > 0

        if signal_bearish and confirm_bearish and has_absorption:
            return {
                "signal_bar": signal_bar,
                "confirm_bar": confirm_bar,
                "signal_type": "buyer_absorbed",
                "direction": "bearish",
                "confirm_time": confirm_bar.close_time,
                "absorption_levels": signal_chars["absorption_levels"]
            }

        if signal_bullish and confirm_bullish and has_absorption:
            return {
                "signal_bar": signal_bar,
                "confirm_bar": confirm_bar,
                "signal_type": "seller_absorbed",
                "direction": "bullish",
                "confirm_time": confirm_bar.close_time,
                "absorption_levels": signal_chars["absorption_levels"]
            }

        return None

    def check_entry_conditions(self, signal):
        """Check if signal meets entry conditions (VAL/VAH proximity and cluster bias)."""
        if not self.profiles:
            return None

        # Get most recent profile
        latest_profile = self.profiles[max(self.profiles.keys())]

        val_price = latest_profile.val
        vah_price = latest_profile.vah
        proximity_points = PROXIMITY_TICKS * TICK_SIZE

        # Determine reference price for proximity check
        if signal["signal_type"] == "seller_absorbed":
            signal_price_ref = signal["signal_bar"].high
        else:
            signal_price_ref = signal["signal_bar"].low

        near_val = abs(signal_price_ref - val_price) <= proximity_points
        near_vah = abs(signal_price_ref - vah_price) <= proximity_points

        if not near_val and not near_vah:
            return None

        # Check cluster bias
        if near_val:
            val_delta = sum(
                lv.delta for price, lv in latest_profile.levels.items()
                if abs(price - val_price) <= proximity_points
            )
            val_bias = "bullish" if val_delta >= MIN_CLUSTER_DELTA else ("bearish" if val_delta <= -MIN_CLUSTER_DELTA else "neutral")
        else:
            val_bias = None

        if near_vah:
            vah_delta = sum(
                lv.delta for price, lv in latest_profile.levels.items()
                if abs(price - vah_price) <= proximity_points
            )
            vah_bias = "bullish" if vah_delta >= MIN_CLUSTER_DELTA else ("bearish" if vah_delta <= -MIN_CLUSTER_DELTA else "neutral")
        else:
            vah_bias = None

        # Determine entry type
        entry_type = None
        level_type = None

        if signal["signal_type"] == "seller_absorbed":
            if near_val and val_bias == "bullish":
                entry_type = "BUY"
                level_type = "VAL"
            elif near_vah and vah_bias == "bullish":
                entry_type = "BUY"
                level_type = "VAH"
        elif signal["signal_type"] == "buyer_absorbed":
            if near_val and val_bias == "bearish":
                entry_type = "SELL"
                level_type = "VAL"
            elif near_vah and vah_bias == "bearish":
                entry_type = "SELL"
                level_type = "VAH"

        if entry_type:
            return {
                "entry_type": entry_type,
                "level_type": level_type,
                "signal": signal,
                "profile": latest_profile
            }

        return None

    def send_signal_to_nt8(self, entry_signal):
        """Send entry signal to NT8 via HTTP."""
        signal = entry_signal["signal"]
        confirm_bar = signal["confirm_bar"]
        signal_bar = signal["signal_bar"]

        entry_price = confirm_bar.close
        stop_distance = 10 * TICK_SIZE

        if entry_signal["entry_type"] == "BUY":
            stop_price = signal_bar.low - stop_distance
        else:
            stop_price = signal_bar.high + stop_distance

        payload = {
            "timestamp": str(confirm_bar.close_time),
            "instrument": "MNQ",  # Execute on Micro NQ
            "action": entry_signal["entry_type"],
            "entry_price": entry_price,
            "stop_price": stop_price,
            "level": entry_signal["level_type"],
            "val": entry_signal["profile"].val,
            "poc": entry_signal["profile"].poc,
            "vah": entry_signal["profile"].vah,
            "signal_type": signal["signal_type"],
            "data_source": "NQ"  # Signal generated from NQ order flow
        }

        try:
            response = requests.post(NT8_URL, json=payload, timeout=5)
            if response.status_code == 200:
                print(f"[SIGNAL SENT] {entry_signal['entry_type']} @ {entry_price:.2f}, Stop: {stop_price:.2f}")
                return True
            else:
                print(f"[ERROR] NT8 returned status {response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to send signal to NT8: {e}")
            return False

    async def process_live_data(self):
        """Main loop to process live data and generate signals."""
        print("Connecting to Databento live stream...")
        print("Waiting for market data...")
        print()

        # Subscribe to NQ continuous front month
        async for record in self.client.subscribe(
            dataset="GLBX.MDP3",
            schema="mbp-1",
            stype_in="continuous",
            symbols=["NQ.c.0"]
        ):
            # Convert to normalized format
            ts_event = pd.Timestamp(record.ts_event, tz="UTC")

            # Check if trading hours
            if not self.is_trading_hours(ts_event):
                continue

            # Add to tick buffer
            tick_data = {
                'ts_event': ts_event,
                'price': (record.bid_px_00 + record.ask_px_00) / 2 / 1e9,
                'bid_px_00': record.bid_px_00 / 1e9,
                'ask_px_00': record.ask_px_00 / 1e9,
                'bid_sz_00': record.bid_sz_00,
                'ask_sz_00': record.ask_sz_00,
                'bid_ct_00': getattr(record, 'bid_ct_00', 1),
                'ask_ct_00': getattr(record, 'ask_ct_00', 1),
            }
            tick_data['index'] = ts_event

            self.tick_buffer.append(tick_data)

            # Keep buffer manageable (last 1 hour of ticks)
            if len(self.tick_buffer) > 100000:
                self.tick_buffer = self.tick_buffer[-50000:]

            # Update volume profile
            self.update_volume_profile(ts_event)

            # Add tick to bar builder
            completed_bar = self.bar_builder.add_tick(tick_data)

            if completed_bar:
                print(f"[{ts_event}] Bar completed: O:{completed_bar.open:.2f} H:{completed_bar.high:.2f} L:{completed_bar.low:.2f} C:{completed_bar.close:.2f}")

                # Check for new signal
                signal = self.detect_signal()

                if signal and not self.in_position:
                    # Check entry conditions
                    entry_signal = self.check_entry_conditions(signal)

                    if entry_signal:
                        # Send to NT8
                        success = self.send_signal_to_nt8(entry_signal)
                        if success:
                            self.in_position = True
                            self.last_signal_time = ts_event

    def run(self):
        """Run the live signal generator."""
        try:
            asyncio.run(self.process_live_data())
        except KeyboardInterrupt:
            print("\nShutting down signal generator...")
        except Exception as e:
            print(f"Error: {e}")
            raise


if __name__ == "__main__":
    generator = LiveSignalGenerator()
    generator.run()
