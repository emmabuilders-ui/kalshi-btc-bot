"""Kalshi market, quote and order types. Prices are in dollars (0.01 - 0.99)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Market:
    ticker: str
    open_time: int          # epoch seconds
    close_time: int         # epoch seconds (settlement time)
    strike: float           # opening reference price
    result: str | None = None   # "yes" / "no" once known

    def is_open(self, t: int) -> bool:
        return self.open_time <= t < self.close_time

    def seconds_remaining(self, t: int) -> int:
        return self.close_time - t


Level = tuple[float, float]   # (price, size)


@dataclass(frozen=True)
class Quote:
    """Snapshot of one market's YES book. Levels are best-first."""

    t: int
    ticker: str
    yes_bid: float | None
    yes_ask: float | None
    yes_bid_size: float | None = None
    yes_ask_size: float | None = None
    last_price: float | None = None
    ask_levels: tuple[Level, ...] = ()
    bid_levels: tuple[Level, ...] = ()

    def asks(self) -> tuple[Level, ...]:
        if self.ask_levels:
            return self.ask_levels
        if self.yes_ask is None:
            return ()
        return ((self.yes_ask, math.inf if self.yes_ask_size is None else self.yes_ask_size),)

    def bids(self) -> tuple[Level, ...]:
        if self.bid_levels:
            return self.bid_levels
        if self.yes_bid is None:
            return ()
        return ((self.yes_bid, math.inf if self.yes_bid_size is None else self.yes_bid_size),)

    def price(self, which: str) -> float | None:
        if which == "ask":
            return self.yes_ask
        if which == "bid":
            return self.yes_bid
        if which == "last":
            return self.last_price
        if which == "mid":
            if self.yes_bid is None or self.yes_ask is None:
                return None
            return (self.yes_bid + self.yes_ask) / 2
        raise ValueError(which)


@dataclass
class Order:
    client_order_id: str
    ticker: str
    action: str             # "buy" | "sell" (always the YES side)
    quantity: int
    limit_price: float | None
    submitted_t: int
    position_id: str
    filled: int = 0
    status: str = "pending"  # pending | filled | partial | cancelled

    @property
    def remaining(self) -> int:
        return self.quantity - self.filled


@dataclass(frozen=True)
class Fill:
    order: Order
    t: int
    quantity: int
    price: float
    fee: float
    source: str             # "book" | "last_trade"


@dataclass(frozen=True)
class Cancel:
    order: Order
    t: int
    reason: str


def kalshi_fee(price: float, quantity: int, rate: float) -> float:
    """Kalshi trading fee: ceil(rate * C * P * (1 - P)) rounded up to the next cent."""
    raw = rate * quantity * price * (1.0 - price)
    return math.ceil(round(raw * 100, 6)) / 100.0


def walk_book(levels, quantity: int, limit: float | None, side: str) -> tuple[int, float]:
    """Fill up to `quantity` across levels within `limit`. Returns (filled, avg_price)."""
    filled = 0
    cost = 0.0
    for px, size in levels:
        if limit is not None and ((side == "buy" and px > limit + 1e-9)
                                  or (side == "sell" and px < limit - 1e-9)):
            break
        take = int(min(quantity - filled, size))
        if take <= 0:
            continue
        filled += take
        cost += take * px
        if filled >= quantity:
            break
    return filled, (cost / filled if filled else 0.0)
