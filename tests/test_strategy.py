import os

import numpy as np
import pytest

from kalshi_btc_bot.backtest import run_backtest
from kalshi_btc_bot.broker import DuplicateOrderError, LiveBroker, LiveTradingLocked, PaperBroker
from kalshi_btc_bot.config import (LIVE_TRADING_ENV_VALUE, LIVE_TRADING_ENV_VAR, Config,
                                   EntryConfig, ExitMethod, TradingConfig)
from kalshi_btc_bot.data import MarketData, load_dir, settle_result
from kalshi_btc_bot.engine import EventLog, StrategyEngine, Tick
from kalshi_btc_bot.indicators import (EntryIndicator, ExitBandSnapshot, ExitBands, RollingStats,
                                       TrailingStop)
from kalshi_btc_bot.live_data import parse_market, parse_orderbook
from kalshi_btc_bot.models import Market, Order, Quote, kalshi_fee
from kalshi_btc_bot.synthetic import generate


# ---------------------------------------------------------------- indicators
def test_rolling_stats_matches_numpy():
    rng = np.random.default_rng(1)
    x = 100_000 + np.cumsum(rng.normal(0, 5, 5000))
    rs = RollingStats(250)
    for i, v in enumerate(x):
        rs.push(v)
        if i >= 249:
            w = x[i - 249:i + 1]
            assert rs.mean() == pytest.approx(w.mean(), abs=1e-6)
            assert rs.std() == pytest.approx(w.std(), abs=1e-6)


def test_entry_bands_zscore_and_separate_breaches():
    ind = EntryIndicator(EntryConfig())
    rng = np.random.default_rng(2)
    prices = list(100_000 + rng.normal(0, 2, 300))
    for t, p in enumerate(prices):
        snap = ind.update(t, p)
    w = np.array(prices[-250:])
    assert snap.ma250 == pytest.approx(w.mean())
    assert snap.upper == pytest.approx(w.mean() + 2 * w.std())
    assert snap.z == pytest.approx((prices[-1] - w.mean()) / w.std())
    prev_upper = ind.last_upper_breach_t
    low = ind.update(300, snap.ma250 - 50)     # far below the lower band
    assert low.lower_breach and not low.upper_breach
    assert ind.last_lower_breach_t == 300 and ind.last_upper_breach_t == prev_upper
    hi = ind.update(301, low.ma250 + 80)
    assert hi.upper_breach and ind.last_upper_breach_t == 301
    assert ind.last_lower_breach_t == 300       # an upper breach does not touch lower tracking


def _dip_then_rally(breach_to_cross_gap):
    """Flat, a sharp dip (lower breach), a hold for the requested gap, then a rally (cross)."""
    p = [100_000.0 + (0.5 if i % 2 else -0.5) for i in range(600)]
    p += [99_960.0] * 5                         # dip -> lower breach
    p += [99_990.0] * breach_to_cross_gap
    p += [100_040.0] * 200                      # rally -> MA50 crosses above MA250
    return p


def _first_signal(prices):
    ind = EntryIndicator(EntryConfig())
    for t, px in enumerate(prices):
        s = ind.update(t, px)
        if s.golden_cross:
            return s
    return None


def test_golden_cross_within_breach_window_signals():
    s = _first_signal(_dip_then_rally(10))
    assert s is not None and s.recent_lower_breach
    assert s.t - s.last_lower_breach_t <= 120


def test_breach_expires_after_120_seconds():
    prices = _dip_then_rally(10)
    ind = EntryIndicator(EntryConfig(breach_window_s=5))
    crosses = [ind.update(t, px) for t, px in enumerate(prices)]
    gc = [s for s in crosses if s.golden_cross]
    assert gc and not any(s.recent_lower_breach for s in gc)


def test_golden_cross_definition():
    ind = EntryIndicator(EntryConfig(fast_ma=2, slow_ma=4, long_ma=4))
    out = [ind.update(t, p) for t, p in enumerate([10, 10, 10, 10, 9, 9, 12, 12])]
    assert [s.t for s in out if s.golden_cross] == [6]


# ------------------------------------------------------------- trailing stop
M5 = ExitMethod("bb20x5s_60s", candle_seconds=5, lookback=20, confirm_seconds=60)


def _snap(t, price, closed=True, upper=110.0, middle=100.0):
    return ExitBandSnapshot(t, price, upper, middle, 90.0, closed, price if closed else None)


def test_touching_upper_band_activates_but_does_not_exit():
    ts = TrailingStop(M5)
    assert not ts.update(_snap(0, 111))
    assert ts.activated
    assert not any(ts.update(_snap(5 * i, 105)) for i in range(1, 50))


def test_no_exit_before_activation():
    ts = TrailingStop(M5)
    assert not any(ts.update(_snap(5 * i, 95)) for i in range(50))
    assert not ts.activated


def test_exit_after_12_consecutive_5s_closes_below_middle_and_reset():
    ts = TrailingStop(M5)
    ts.update(_snap(0, 111))
    for i in range(11):
        assert not ts.update(_snap(5 * (i + 1), 99))
    assert not ts.update(_snap(60, 100))    # close AT the middle resets the timer
    assert ts.seconds_below == 0
    results = [ts.update(_snap(65 + 5 * i, 99)) for i in range(12)]
    assert results == [False] * 11 + [True]


def test_one_minute_candles_exit_on_first_close_below():
    ts = TrailingStop(ExitMethod("m1", candle_seconds=60, lookback=20, confirm_seconds=60))
    ts.update(_snap(0, 111, closed=False))
    assert ts.update(_snap(59, 99))


def test_exit_bands_on_5s_candles():
    eb = ExitBands(M5)
    closes = []
    for t in range(200):
        snap = eb.update(t, 100 + t)
        if snap.candle_closed:
            closes.append(100 + t)
    c = np.array(closes[-20:])
    assert snap.middle == pytest.approx(c.mean())
    assert snap.upper == pytest.approx(c.mean() + 2 * c.std())
    assert closes[0] == 104                 # first candle covers t=0..4


# -------------------------------------------------------------- broker/engine
def test_fee_formula():
    assert kalshi_fee(0.50, 100, 0.07) == 1.75
    assert kalshi_fee(0.30, 1, 0.07) == 0.02


def test_duplicate_order_rejected():
    b = PaperBroker(Config())
    b.submit(Order("x", "T", "buy", 1, 0.5, 0, "p"))
    with pytest.raises(DuplicateOrderError):
        b.submit(Order("x", "T", "buy", 1, 0.5, 0, "p"))


def test_buy_walks_book_and_respects_limit():
    b = PaperBroker(Config())
    b.submit(Order("o", "T", "buy", 100, 0.52, 0, "p"))
    q = Quote(1, "T", 0.48, 0.50, ask_levels=((0.50, 40), (0.52, 30), (0.55, 500)))
    fills = [e for e in b.on_tick(1, q) if hasattr(e, "price")]
    assert fills[0].quantity == 70
    assert fills[0].price == pytest.approx((0.50 * 40 + 0.52 * 30) / 70)


def test_live_trading_locked_by_default(monkeypatch):
    monkeypatch.delenv(LIVE_TRADING_ENV_VAR, raising=False)
    with pytest.raises(LiveTradingLocked):
        LiveBroker(Config())
    cfg = Config(trading=TradingConfig(live_trading_enabled=True))
    with pytest.raises(LiveTradingLocked):          # flag alone is not enough
        LiveBroker(cfg)
    monkeypatch.setenv(LIVE_TRADING_ENV_VAR, LIVE_TRADING_ENV_VALUE)
    with pytest.raises(NotImplementedError):        # and routing is not built yet
        LiveBroker(cfg)


def _engine_run(prices, quote_px=0.5, cfg=None):
    cfg = cfg or Config()
    m = Market("T1", 0, len(prices) + 10, 100_000.0)
    eng = StrategyEngine(cfg, cfg.exit_method("bb20x5s_60s"), PaperBroker(cfg), EventLog())
    ind = EntryIndicator(cfg.entry)
    eb = ExitBands(cfg.exit_method("bb20x5s_60s"))
    for t, p in enumerate(prices):
        q = Quote(t, "T1", quote_px - 0.01, quote_px, 500, 500, quote_px)
        eng.on_tick(Tick(t, p, ind.update(t, p), m, q), eb.update(t, p))
    return eng, m


def test_engine_enters_once_per_contract_with_configured_quantity():
    prices = _dip_then_rally(10) + _dip_then_rally(10)
    cfg = Config(trading=TradingConfig(quantity=7))
    eng, _ = _engine_run(prices, cfg=cfg)
    assert len(eng.positions) == 1
    p = next(iter(eng.positions.values()))
    assert p.bought == 7
    assert any(e["reason"] == "duplicate_contract_entry" for e in eng.events.events
               if e["event"] == "signal_rejected")


def test_engine_price_filter():
    eng, _ = _engine_run(_dip_then_rally(10), quote_px=0.75)
    assert not eng.positions
    assert eng.events.events[-1]["reason"] == "yes_price_out_of_range"


def test_settlement_pays_one_dollar_on_yes():
    eng, m = _engine_run(_dip_then_rally(10))
    eng.settle("T1", "yes", m.close_time)
    p = next(iter(eng.positions.values()))
    assert p.status == "closed"
    assert p.payout == pytest.approx(p.settled_qty * 1.0)


def test_settlement_rule():
    cfg = Config().settlement
    assert settle_result(100.0, 100.0, cfg) == "yes"
    assert settle_result(99.99, 100.0, cfg) == "no"


# ------------------------------------------------------------------- parsers
def test_parse_kalshi_market_and_orderbook():
    m = parse_market({"ticker": "KXBTC15M-X", "open_time": "2026-10-28T01:30:00Z",
                      "close_time": "2026-10-28T01:45:00Z", "floor_strike": 101234.5})
    assert m.close_time - m.open_time == 900 and m.strike == 101234.5
    q = parse_orderbook(1, "KXBTC15M-X", {"orderbook": {"yes": [[40, 10], [45, 5]],
                                                       "no": [[50, 7], [52, 3]]}}, 0.46)
    assert q.yes_bid == 0.45 and q.yes_bid_size == 5
    assert q.yes_ask == pytest.approx(0.48) and q.yes_ask_size == 3
    q2 = parse_orderbook(1, "X", {"orderbook_fp": {"yes_dollars": [["0.4500", "5"]],
                                                   "no_dollars": [["0.5200", "3"]]}}, None)
    assert q2.yes_bid == 0.45 and q2.yes_ask == pytest.approx(0.48)


# ------------------------------------------------------------------ backtest
def test_backtest_entries_identical_across_exit_methods(tmp_path):
    generate(tmp_path, hours=6, seed=3)
    res = run_backtest(load_dir(tmp_path), Config())
    assert res.entries_identical
    rows = res.summary_rows()
    by = {(r["exit_method"], r["pricing"]): r for r in rows}
    assert by[("hold_to_settlement", "executable")]["early_exits"] == 0
    n = {r["entries"] for r in rows if r["pricing"] == "executable"}
    assert len(n) == 1 and n.pop() > 0


def test_recorder_parse_book_best_levels():
    from kalshi_btc_bot.recorder import parse_book
    j = {"orderbook_fp": {"yes_dollars": [["0.46", "10"], ["0.50", "5"]],
                          "no_dollars": [["0.45", "3"], ["0.49", "7"]]}}
    b = parse_book(j)
    assert b["yes_bid"] == 0.50 and b["yes_ask"] == 0.51
    assert b["yes_bids"][0] == [0.50, 5.0] and b["no_bids"][0] == [0.49, 7.0]
    old = parse_book({"orderbook": {"yes": [[46, 10]], "no": [[49, 7]]}})
    assert old["yes_bid"] == 0.46 and old["yes_ask"] == 0.51
