"""Order execution. PaperBroker simulates fills; LiveBroker is locked.

PaperBroker has two pricing modes:
* "executable": buys walk the YES asks, sells walk the YES bids, with size limits.
* "last": fills everything at the last trade price (matches the Oct 26 exploration).

Orders submitted at second t fill against the quote at t + latency_s, so a
decision never trades on the quote it was made from.
"""

from __future__ import annotations

from .config import Config, live_trading_allowed, LIVE_TRADING_ENV_VAR, LIVE_TRADING_ENV_VALUE
from .models import Cancel, Fill, Order, Quote, kalshi_fee, walk_book


class DuplicateOrderError(Exception):
    pass


class PaperBroker:
    def __init__(self, cfg: Config, pricing: str = "executable"):
        if pricing not in ("executable", "last"):
            raise ValueError(pricing)
        self.cfg = cfg
        self.pricing = pricing
        self._seen_ids: set[str] = set()
        self.pending: list[Order] = []

    def submit(self, order: Order) -> Order:
        if order.client_order_id in self._seen_ids:
            raise DuplicateOrderError(order.client_order_id)
        self._seen_ids.add(order.client_order_id)
        self.pending.append(order)
        return order

    def cancel_ticker(self, ticker: str, t: int, reason: str) -> list[Cancel]:
        out = []
        for o in [o for o in self.pending if o.ticker == ticker]:
            o.status = "cancelled"
            self.pending.remove(o)
            out.append(Cancel(o, t, reason))
        return out

    def on_tick(self, t: int, quote: Quote | None) -> list[Fill | Cancel]:
        events: list[Fill | Cancel] = []
        tc = self.cfg.trading
        for o in list(self.pending):
            if t < o.submitted_t + tc.latency_s:
                continue
            usable = (quote is not None and quote.ticker == o.ticker
                      and t - quote.t <= tc.quote_max_age_s)
            if o.action == "buy":
                # Entries are immediate-or-cancel: fill what is available now, drop the rest.
                filled, px, src = self._match(o, quote) if usable else (0, 0.0, "")
                if filled and (filled == o.remaining or tc.allow_partial_fills):
                    events.append(self._fill(o, t, filled, px, src))
                if o.remaining > 0:
                    o.status = "cancelled" if o.filled == 0 else "partial"
                    events.append(Cancel(o, t, "no_liquidity_within_limit" if usable else "no_quote"))
                self.pending.remove(o)
            else:
                # Exits keep working each second until fully filled or the market closes.
                if not usable:
                    continue
                filled, px, src = self._match(o, quote)
                if filled:
                    events.append(self._fill(o, t, filled, px, src))
                if o.remaining == 0:
                    self.pending.remove(o)
        return events

    def _match(self, o: Order, q: Quote) -> tuple[int, float, str]:
        if self.pricing == "last":
            if q.last_price is None:
                return 0, 0.0, ""
            if o.action == "buy" and o.limit_price is not None and q.last_price > o.limit_price + 1e-9:
                return 0, 0.0, ""
            return o.remaining, q.last_price, "last_trade"
        levels = q.asks() if o.action == "buy" else q.bids()
        filled, px = walk_book(levels, o.remaining, o.limit_price, o.action)
        return filled, px, "book"

    def _fill(self, o: Order, t: int, qty: int, px: float, src: str) -> Fill:
        o.filled += qty
        o.status = "filled" if o.remaining == 0 else "partial"
        return Fill(o, t, qty, px, kalshi_fee(px, qty, self.cfg.trading.fee_rate), src)


class LiveTradingLocked(RuntimeError):
    pass


class LiveBroker:
    """Placeholder for real Kalshi order routing. It refuses to start unless live
    trading is explicitly enabled, and even then real routing is not implemented yet:
    it must be built and reviewed against Kalshi's authenticated API first."""

    def __init__(self, cfg: Config):
        if not live_trading_allowed(cfg):
            raise LiveTradingLocked(
                "Live trading is OFF. It requires trading.live_trading_enabled = true in the "
                f"config AND {LIVE_TRADING_ENV_VAR}={LIVE_TRADING_ENV_VALUE} in the environment.")
        raise NotImplementedError(
            "Live Kalshi order routing is intentionally not implemented in this version. "
            "Run paper trading first; live routing will be added and reviewed separately.")
