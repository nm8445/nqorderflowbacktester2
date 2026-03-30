"""
Custom backtest engine foundation.

Designed to replay MBP-1 events tick-by-tick, allowing strategies to react
to individual order book updates and trade prints.

Current state: skeleton with the core event loop and position tracking.
Build out Strategy subclasses as your orderflow logic develops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


@dataclass
class Quote:
    """Top-of-book snapshot at a point in time."""
    ts: pd.Timestamp
    bid: float
    ask: float
    bid_size: int
    ask_size: int

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class Trade:
    """A single trade print."""
    ts: pd.Timestamp
    price: float
    size: int
    aggressor_side: str  # 'B' buy aggressor, 'A' sell aggressor, 'N' unknown


@dataclass
class Fill:
    """A simulated fill for a strategy order."""
    ts: pd.Timestamp
    price: float
    size: int
    side: str  # 'B' or 'A'


@dataclass
class Position:
    """Track running PnL and position size."""
    size: int = 0          # positive = long, negative = short
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    fills: list[Fill] = field(default_factory=list)

    def apply_fill(self, fill: Fill) -> None:
        qty = fill.size if fill.side == "B" else -fill.size

        if self.size == 0:
            self.size = qty
            self.avg_price = fill.price
        elif (self.size > 0 and qty > 0) or (self.size < 0 and qty < 0):
            # Adding to position
            total = self.size + qty
            self.avg_price = (self.avg_price * self.size + fill.price * qty) / total
            self.size = total
        else:
            # Reducing or flipping
            close_qty = min(abs(self.size), abs(qty))
            pnl_per_unit = (fill.price - self.avg_price) * (1 if self.size > 0 else -1)
            self.realized_pnl += pnl_per_unit * close_qty

            net = self.size + qty
            if net == 0:
                self.size = 0
                self.avg_price = 0.0
            else:
                self.size = net
                self.avg_price = fill.price  # new position at fill price

        self.fills.append(fill)

    def unrealized_pnl(self, current_price: float) -> float:
        if self.size == 0:
            return 0.0
        return (current_price - self.avg_price) * self.size

    def total_pnl(self, current_price: float) -> float:
        return self.realized_pnl + self.unrealized_pnl(current_price)


class Strategy(Protocol):
    """
    Interface for orderflow strategies.

    Implement on_quote and/or on_trade, then return a list of orders to submit.
    An 'order' here is a simple dict: {"side": "B"/"A", "size": int, "price": float or None}
    (price=None means market order — filled at current ask/bid immediately).
    """

    def on_quote(self, quote: Quote, position: Position) -> list[dict]:
        ...

    def on_trade(self, trade: Trade, quote: Quote, position: Position) -> list[dict]:
        ...


class BacktestEngine:
    """
    Replays MBP-1 events and calls strategy callbacks on each tick.

    Usage
    -----
    engine = BacktestEngine(df, strategy=MyStrategy(), slippage_ticks=1)
    result = engine.run()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        tick_size: float = 0.25,      # NQ tick = 0.25 points
        tick_value: float = 5.0,      # NQ tick value = $5
        slippage_ticks: int = 1,
    ):
        self.df = df
        self.strategy = strategy
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.slippage_ticks = slippage_ticks
        self.position = Position()
        self._current_quote: Quote | None = None

    def _apply_slippage(self, side: str, price: float) -> float:
        offset = self.slippage_ticks * self.tick_size
        return price + offset if side == "B" else price - offset

    def _execute_orders(self, orders: list[dict], quote: Quote, ts: pd.Timestamp) -> None:
        for order in orders:
            side = order["side"]
            size = order["size"]
            raw_price = order.get("price") or (quote.ask if side == "B" else quote.bid)
            fill_price = self._apply_slippage(side, raw_price)
            fill = Fill(ts=ts, price=fill_price, size=size, side=side)
            self.position.apply_fill(fill)

    def run(self) -> pd.DataFrame:
        """
        Replay all events and return a DataFrame of per-event position snapshots.
        """
        records = []
        quote = Quote(ts=pd.NaT, bid=0.0, ask=0.0, bid_size=0, ask_size=0)

        for ts, row in self.df.iterrows():
            action = row.get("action", "")
            side = row.get("side", "N")
            price = row.get("price", 0.0)
            size = int(row.get("size", 0))

            # Update top-of-book quote from every row
            bid = row.get("bid_px_00", quote.bid)
            ask = row.get("ask_px_00", quote.ask)
            bid_sz = int(row.get("bid_sz_00", quote.bid_size))
            ask_sz = int(row.get("ask_sz_00", quote.ask_size))

            quote = Quote(ts=ts, bid=bid, ask=ask, bid_size=bid_sz, ask_size=ask_sz)

            orders = []

            if action == "T":
                trade = Trade(ts=ts, price=price, size=size, aggressor_side=side)
                orders = self.strategy.on_trade(trade, quote, self.position)
            else:
                orders = self.strategy.on_quote(quote, self.position)

            if orders:
                self._execute_orders(orders, quote, ts)

            records.append({
                "ts": ts,
                "bid": quote.bid,
                "ask": quote.ask,
                "position": self.position.size,
                "avg_price": self.position.avg_price,
                "realized_pnl": self.position.realized_pnl,
                "unrealized_pnl": self.position.unrealized_pnl(quote.mid),
                "total_pnl": self.position.total_pnl(quote.mid),
            })

        result = pd.DataFrame(records).set_index("ts")
        # Convert PnL from points to dollars
        for col in ("realized_pnl", "unrealized_pnl", "total_pnl"):
            result[col] = result[col] / self.tick_size * self.tick_value

        return result
