"""Paper trading: live BTC + live Kalshi data, simulated fills, no orders sent.

Runs every exit method side by side (each with its own PaperBroker) so the
paper session doubles as a forward test of the four exits. It also records the
raw data in the backtester's CSV format, so a paper session can be replayed.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

from .broker import PaperBroker
from .config import Config
from .data import settle_result
from .engine import EventLog, StrategyEngine, Tick
from .indicators import IndicatorHub
from .live_data import KalshiMarketData, _px, make_feed
from .models import Market

log = logging.getLogger("kalshi_btc_bot")


class Recorder:
    def __init__(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        self._btc = self._open(out / "btc.csv", ["timestamp", "price"])
        self._q = self._open(out / "quotes.csv", ["timestamp", "ticker", "yes_bid", "yes_ask",
                                                  "yes_bid_size", "yes_ask_size", "last_price"])
        self._m = self._open(out / "markets.csv", ["ticker", "open_time", "close_time", "strike", "result"])
        self._seen: set[str] = set()

    @staticmethod
    def _open(path: Path, header):
        new = not path.exists()
        fh = open(path, "a", newline="")
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        return fh, w

    def btc(self, t, price):
        self._btc[1].writerow([t, price])

    def quote(self, q):
        self._q[1].writerow([q.t, q.ticker, q.yes_bid, q.yes_ask, q.yes_bid_size, q.yes_ask_size,
                             q.last_price])

    def market(self, m: Market):
        if m.result and m.ticker not in self._seen:
            self._seen.add(m.ticker)
            self._m[1].writerow([m.ticker, m.open_time, m.close_time, m.strike, m.result])

    def flush(self):
        for fh, _ in (self._btc, self._q, self._m):
            fh.flush()


def run_paper(cfg: Config, out_dir: str | Path, feed_name: str = "composite",
              methods: list[str] | None = None, duration_s: int | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    events = EventLog(out / "events.jsonl", echo=True)
    rec = Recorder(out / "recorded_data")
    feed = make_feed(feed_name)
    kalshi = KalshiMarketData(cfg.kalshi)
    hub = IndicatorHub(cfg.entry, cfg.exit_methods)
    chosen = [m for m in cfg.exit_methods if methods is None or m.name in methods]
    engines = {m.name: StrategyEngine(cfg, m, PaperBroker(cfg, "executable"), events, f"paper/{m.name}")
               for m in chosen}
    events.write("paper_start", feed=feed_name, methods=[m.name for m in chosen],
                 quantity=cfg.trading.quantity, live_trading=False,
                 note="Paper trading only; no orders are sent to Kalshi.")

    market: Market | None = None
    awaiting: dict[str, Market] = {}      # closed markets waiting for an official result
    spot_hist: dict[int, float] = {}     # for the spot-proxy settlement estimate
    last_trade: dict[str, float | None] = {}
    last_t: int | None = None
    price: float | None = None
    started = time.time()
    next_status = 0.0
    try:
        while duration_s is None or time.time() - started < duration_s:
            time.sleep(max(0.0, 1.0 - (time.time() % 1.0)))
            now = int(time.time())
            try:
                price = feed.get_price()
            except Exception as e:
                log.warning("BTC feed error (%s); reusing last price", e)
                if price is None:
                    continue
            # Roll the Kalshi market when the current one closes.
            if market is None or now >= market.close_time:
                if market is not None:
                    awaiting[market.ticker] = market
                try:
                    market = next((m for m in kalshi.open_markets() if m.is_open(now)), None)
                    if market:
                        events.write("market_open", ticker=market.ticker, strike=market.strike,
                                     close_time=market.close_time)
                except Exception as e:
                    log.warning("Kalshi market list error: %s", e)
                    market = None
            quote = None
            if market is not None:
                try:
                    if now % 5 == 0 or last_trade.get(market.ticker) is None:
                        _, raw = kalshi.market(market.ticker)
                        last_trade[market.ticker] = _px(raw, "last_price")
                    quote = kalshi.quote(now, market.ticker, last_trade.get(market.ticker))
                    rec.quote(quote)
                except Exception as e:
                    log.warning("Kalshi quote error: %s", e)
            # Feed every whole second (forward-fill if the loop fell behind).
            start_t = now if last_t is None else last_t + 1
            for t in range(start_t, now + 1):
                spot_hist[t] = price
                rec.btc(t, price)
                entry_snap, exit_snaps = hub.update(t, price)
                tick = Tick(t, price, entry_snap, market if market and market.is_open(t) else None, quote)
                for name, eng in engines.items():
                    eng.on_tick(tick, exit_snaps.get(name))
            last_t = now
            for old_t in [k for k in spot_hist if k < now - 7200]:
                del spot_hist[old_t]
            _settle_closed(awaiting, kalshi, engines, events, rec, spot_hist, cfg, now)
            if time.time() >= next_status:
                next_status = time.time() + 15
                e = entry_snap
                log.info("t=%s btc=%.2f z=%s ma50=%s ma250=%s market=%s secs_left=%s yes_ask=%s open=%s",
                         now, price, _f(e.z), _f(e.ma50), _f(e.ma250),
                         market.ticker if market else None,
                         market.seconds_remaining(now) if market else None,
                         quote.yes_ask if quote else None,
                         {n: len(g.open_positions()) for n, g in engines.items()})
                rec.flush()
    except KeyboardInterrupt:
        log.info("stopping paper session")
    finally:
        rec.flush()
        summary = {n: {"closed": len(g.closed_positions()),
                       "gross_pnl": round(sum(p.gross_pnl for p in g.closed_positions()), 2),
                       "net_pnl": round(sum(p.net_pnl for p in g.closed_positions()), 2),
                       "open": len(g.open_positions())} for n, g in engines.items()}
        events.write("paper_stop", summary=summary)
        events.close()
    return summary


def _settle_closed(awaiting, kalshi, engines, events, rec, spot_hist, cfg, now):
    for ticker, m in list(awaiting.items()):
        if now % 10 and now < m.close_time + 3600:
            continue
        official = None
        try:
            official, _ = kalshi.market(ticker)
        except Exception as e:
            log.warning("Kalshi settlement lookup failed for %s: %s", ticker, e)
        w = cfg.settlement.window_s
        vals = [spot_hist[t] for t in range(m.close_time - w, m.close_time) if t in spot_hist]
        est = sum(vals) / len(vals) if len(vals) == w else None
        est_result = settle_result(est, m.strike, cfg.settlement) if est is not None else None
        result = official.result if official else None
        if result is None and now < m.close_time + 3600:
            continue      # wait up to an hour for Kalshi's official result
        result = result or est_result
        events.write("market_settled", ticker=ticker, official_result=official.result if official else None,
                     estimated_result=est_result, estimated_value=est, index="spot_proxy",
                     result_used=result)
        if official:
            rec.market(official)
        if result:
            for eng in engines.values():
                eng.settle(ticker, result, m.close_time, est)
        del awaiting[ticker]


def _f(v):
    return None if v is None else round(v, 3)
