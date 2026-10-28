"""Strategy engine: turns indicator snapshots + Kalshi quotes into orders and positions.

One engine runs one exit method against one broker. The backtester runs several
engines side by side on the same data; paper trading does the same live.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .config import Config, ExitMethod
from .indicators import EntrySnapshot, ExitBandSnapshot, TrailingStop
from .models import Cancel, Fill, Market, Order, Quote

log = logging.getLogger("kalshi_btc_bot")


class EventLog:
    """Append-only JSON-lines log of every signal, order, fill, exit and settlement."""

    def __init__(self, path: str | Path | None = None, echo: bool = False):
        self.path = Path(path) if path else None
        self.echo = echo
        self.events: list[dict] = []
        self._fh = open(self.path, "a") if self.path else None

    def write(self, event: str, **data) -> None:
        rec = {"event": event, **data}
        self.events.append(rec)
        if self._fh:
            self._fh.write(json.dumps(rec, default=str) + "\n")
            self._fh.flush()
        if self.echo:
            log.info("%s %s", event, json.dumps(data, default=str))

    def close(self) -> None:
        if self._fh:
            self._fh.close()


@dataclass
class Tick:
    t: int
    price: float
    entry: EntrySnapshot
    market: Market | None
    quote: Quote | None


@dataclass
class Position:
    position_id: str
    ticker: str
    method: str
    strike: float
    close_time: int
    signal_t: int
    signal_yes_price: float
    signal_btc: float
    qty: int = 0                 # contracts currently held
    bought: int = 0
    cost: float = 0.0            # sum(price * qty) paid
    proceeds: float = 0.0        # sum(price * qty) received from sells
    fees: float = 0.0
    entry_t: int | None = None
    entry_price: float | None = None
    exit_signal_t: int | None = None
    exit_t: int | None = None
    exit_price: float | None = None
    sold: int = 0
    settled_qty: int = 0
    settlement: str | None = None
    payout: float = 0.0
    status: str = "pending_entry"   # pending_entry | open | exiting | closed | missed
    exit_reason: str | None = None
    stop_activated_t: int | None = None
    stop: TrailingStop | None = field(default=None, repr=False)

    @property
    def gross_pnl(self) -> float:
        return self.proceeds + self.payout - self.cost

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.fees

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("stop")
        d["gross_pnl"] = round(self.gross_pnl, 4)
        d["net_pnl"] = round(self.net_pnl, 4)
        return d


class StrategyEngine:
    def __init__(self, cfg: Config, method: ExitMethod, broker, events: EventLog, label: str = ""):
        self.cfg = cfg
        self.method = method
        self.broker = broker
        self.events = events
        self.label = label or method.name
        self.positions: dict[str, Position] = {}
        self.entries_by_ticker: Counter[str] = Counter()
        self._orders: dict[str, Position] = {}

    # ------------------------------------------------------------------ helpers
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.status in ("pending_entry", "open", "exiting")]

    def closed_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.status == "closed"]

    def _log(self, event: str, **data) -> None:
        self.events.write(event, engine=self.label, method=self.method.name, **data)

    # ---------------------------------------------------------------- main loop
    def on_tick(self, tick: Tick, exit_snap: ExitBandSnapshot | None) -> None:
        for ev in self.broker.on_tick(tick.t, tick.quote):
            self._on_broker_event(ev, tick)
        self._manage_exits(tick, exit_snap)
        self._check_entry(tick)

    def _on_broker_event(self, ev: Fill | Cancel, tick: Tick) -> None:
        pos = self._orders[ev.order.client_order_id]
        if isinstance(ev, Fill):
            pos.fees += ev.fee
            if ev.order.action == "buy":
                pos.qty += ev.quantity
                pos.bought += ev.quantity
                pos.cost += ev.quantity * ev.price
                pos.entry_t = ev.t
                pos.entry_price = pos.cost / pos.bought
                pos.status = "open"
                pos.stop = TrailingStop(self.method) if self.method.kind != "hold" else None
                self._log("entry_fill", position_id=pos.position_id, ticker=pos.ticker, t=ev.t,
                          qty=ev.quantity, price=round(ev.price, 4), fee=ev.fee, source=ev.source,
                          seconds_remaining=pos.close_time - ev.t)
            else:
                pos.qty -= ev.quantity
                pos.sold += ev.quantity
                pos.proceeds += ev.quantity * ev.price
                pos.exit_t = ev.t
                pos.exit_price = pos.proceeds / pos.sold
                self._log("exit_fill", position_id=pos.position_id, ticker=pos.ticker, t=ev.t,
                          qty=ev.quantity, price=round(ev.price, 4), fee=ev.fee, source=ev.source,
                          seconds_remaining=pos.close_time - ev.t)
                if pos.qty == 0:
                    pos.status = "closed"
                    pos.exit_reason = "trailing_stop"
        else:
            self._log("order_cancel", position_id=pos.position_id, ticker=pos.ticker, t=ev.t,
                      action=ev.order.action, filled=ev.order.filled, reason=ev.reason)
            if ev.order.action == "buy" and pos.bought == 0:
                pos.status = "missed"

    def _manage_exits(self, tick: Tick, snap: ExitBandSnapshot | None) -> None:
        if snap is None:
            return
        for pos in self.open_positions():
            if pos.status != "open" or pos.stop is None or tick.t >= pos.close_time:
                continue
            was_active = pos.stop.activated
            fire = pos.stop.update(snap)
            if pos.stop.activated and not was_active:
                pos.stop_activated_t = tick.t
                self._log("trailing_stop_activated", position_id=pos.position_id, t=tick.t,
                          btc=tick.price, exit_upper=_r(snap.upper), exit_middle=_r(snap.middle),
                          seconds_remaining=pos.close_time - tick.t)
            if fire:
                pos.status = "exiting"
                pos.exit_signal_t = tick.t
                self._log("exit_signal", position_id=pos.position_id, t=tick.t, btc=tick.price,
                          exit_middle=_r(snap.middle), seconds_below=pos.stop.seconds_below,
                          seconds_remaining=pos.close_time - tick.t)
                self.broker.submit(Order(f"{pos.position_id}-sell", pos.ticker, "sell", pos.qty,
                                         None, tick.t, pos.position_id))
                self._orders[f"{pos.position_id}-sell"] = pos

    def _check_entry(self, tick: Tick) -> None:
        e = tick.entry
        if not (e.golden_cross and e.recent_lower_breach):
            return
        ecfg, tcfg = self.cfg.entry, self.cfg.trading
        m, q = tick.market, tick.quote
        base = dict(t=tick.t, btc=tick.price, z_breach_t=e.last_lower_breach_t,
                    secs_since_breach=tick.t - e.last_lower_breach_t,
                    indicators=e.as_log())
        reject = None
        yes_px = None
        if m is None or not m.is_open(tick.t):
            reject = "no_open_market"
        elif m.seconds_remaining(tick.t) < ecfg.min_seconds_remaining:
            reject = "contract_expiring"
        elif q is None or q.ticker != m.ticker or tick.t - q.t > tcfg.quote_max_age_s:
            reject = "no_fresh_quote"
        else:
            yes_px = q.price(ecfg.yes_price_field)
            if yes_px is None:
                reject = f"no_{ecfg.yes_price_field}_price"
            elif not (ecfg.yes_price_min <= yes_px <= ecfg.yes_price_max):
                reject = "yes_price_out_of_range"
            elif self.entries_by_ticker[m.ticker] >= tcfg.max_entries_per_contract:
                reject = "duplicate_contract_entry"
            elif sum(p.close_time > tick.t for p in self.open_positions()) >= tcfg.max_open_positions:
                # Positions in already-closed markets are only awaiting settlement, so they
                # do not block entries in the next contract.
                reject = "max_open_positions"
        ctx = dict(ticker=m.ticker if m else None, yes_price=yes_px,
                   seconds_remaining=m.seconds_remaining(tick.t) if m else None,
                   strike=m.strike if m else None)
        if reject:
            self._log("signal_rejected", reason=reject, **base, **ctx)
            return

        self.entries_by_ticker[m.ticker] += 1
        pid = f"{m.ticker}-{tick.t}-{self.label}"
        pos = Position(pid, m.ticker, self.method.name, m.strike, m.close_time, tick.t, yes_px, tick.price)
        self.positions[pid] = pos
        limit = min(0.99, round(yes_px + tcfg.entry_slippage, 2))
        order = Order(f"{pid}-buy", m.ticker, "buy", tcfg.quantity, limit, tick.t, pid)
        self.broker.submit(order)
        self._orders[order.client_order_id] = pos
        self._log("entry_signal", position_id=pid, qty=tcfg.quantity, limit=limit, **base, **ctx)

    # --------------------------------------------------------------- settlement
    def settle(self, ticker: str, result: str, t: int, settle_value: float | None = None) -> None:
        for ev in self.broker.cancel_ticker(ticker, t, "market_closed"):
            self._on_broker_event(ev, None)
        for pos in self.positions.values():
            if pos.ticker != ticker or pos.status in ("closed", "missed"):
                continue
            pos.settlement = result
            pos.settled_qty = pos.qty
            pos.payout = pos.qty * (1.0 if result == "yes" else 0.0)
            pos.qty = 0
            if pos.status == "pending_entry":
                pos.status = "missed"
                continue
            if pos.exit_reason is None:
                if pos.status == "exiting":
                    pos.exit_reason = "exit_unfilled_then_settlement"
                else:
                    pos.exit_reason = "settlement" if pos.sold == 0 else "partial_exit_then_settlement"
            pos.status = "closed"
            self._log("settlement", position_id=pos.position_id, ticker=ticker, t=t, result=result,
                      settle_value=settle_value, strike=pos.strike, settled_qty=pos.settled_qty,
                      payout=pos.payout, gross_pnl=round(pos.gross_pnl, 4),
                      net_pnl=round(pos.net_pnl, 4))


def _r(v):
    return None if v is None else round(v, 4)
