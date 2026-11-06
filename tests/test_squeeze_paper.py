import json
from datetime import datetime, timezone

import numpy as np

from kalshi_btc_bot.dashboard import history_events
from kalshi_btc_bot.squeeze_paper import (SqueezeMakerPaper, SqueezePaper, SqueezeWatch, WARMUP, apply_trades,
                                          bid_price, maker_fee, squeeze_bars)

S = 1_800_000_000          # multiple of 3600


def squeezed(k, bars=40, jitter=0.02):
    """k-minute bars with a $10 intrabar range but closes that hardly move: Bollinger inside Keltner."""
    size = 60 * k
    x = np.full(bars * size, 1000.0)
    x += jitter * np.sin(np.arange(len(x)) / size)           # tiny drift of the closes
    x[100::size] = 1005.0
    x[200::size] = 995.0
    return x


def test_squeeze_detection_on_and_off():
    b = squeeze_bars(squeezed(5), 5)
    assert b["on"][-1] and b["on"][25:].all()
    trend = 1000 + np.arange(40 * 300) * 0.01                # steady climb: wide Bollinger, small range
    assert not squeeze_bars(trend, 5)["on"][-1]
    assert squeeze_bars(np.full(600, 1000.0), 5) is None     # too few bars


def test_watch_fires_once_per_episode_and_picks_side():
    w = SqueezeWatch(ks=(5,))
    sec = squeezed(5)
    boundary = S
    w.refresh(5, boundary, sec)
    st = w.state[5]
    assert st["on"] and st["bar"] == S // 300 - 1
    t = S + 59                                                # last second of the first minute of the next bar
    assert w.check(t, 1000.0) == []                           # inside the band
    up = w.check(t, st["up"] + 1)
    assert len(up) == 1 and up[0]["side"] == "yes"
    assert w.check(t + 60, st["up"] + 2) == []                # same episode: only the first break counts
    w2 = SqueezeWatch(ks=(5,))
    w2.refresh(5, boundary, sec)
    assert w2.check(t, st["dn"] - 1)[0]["side"] == "no"
    w3 = SqueezeWatch(ks=(5,))
    w3.refresh(5, boundary, sec)
    assert w3.check(t + 300, st["up"] + 1) == []              # state is for an older bar: ignored


def test_bid_price_and_fee():
    assert bid_price(0.70, 0.75) == 0.74                      # 1c under the ask
    assert bid_price(0.74, 0.75) == 0.74                      # 1c spread: join the best bid
    assert bid_price(None, 0.75) == 0.74
    assert maker_fee(20, 0.75) == 0.07 and maker_fee(20, 0.75, 0.0) == 0.0


def _order(side="yes", bid=0.75, queue=0.0):
    return {"side": side, "bid": bid, "queue_ahead": queue, "contracts": 20, "filled": 0, "posted_iso": "2026-11-05T10:00:00"}


def _tr(tid, ts, yes_price, n, taker):
    return {"trade_id": tid, "created_time": f"2026-11-05T{ts}Z", "yes_price_dollars": f"{yes_price:.4f}",
            "count_fp": f"{n:.2f}", "taker_side": taker}


def test_yes_bid_fills_from_yes_sellers_only():
    o = _order("yes", 0.75)
    got = apply_trades(o, [_tr("a", "09:59:59", 0.70, 50, "no"),    # before posting
                           _tr("b", "10:00:05", 0.74, 5, "yes"),    # a YES buyer: not ours
                           _tr("c", "10:00:06", 0.76, 5, "no"),     # above our bid
                           _tr("d", "10:00:07", 0.75, 6, "no"),     # at our bid
                           _tr("e", "10:00:08", 0.74, 3, "no")])    # through us
    assert got == 9 and o["filled"] == 9


def test_no_bid_fills_from_no_sellers_and_prices_on_the_no_side():
    # NO bid at 0.75 = YES price 0.25. A taker who bought YES hits NO bids.
    o = _order("no", 0.75)
    got = apply_trades(o, [_tr("a", "10:00:05", 0.24, 4, "no"),     # a NO buyer: not ours
                           _tr("b", "10:00:06", 0.20, 4, "yes"),    # NO price .80 above our bid: not filled
                           _tr("c", "10:00:07", 0.25, 5, "yes"),    # NO price .75 = our bid
                           _tr("d", "10:00:08", 0.27, 2, "yes")])   # NO price .73 below: through us
    assert got == 7 and o["filled"] == 7


def test_queue_ahead_goes_first():
    o = _order("no", 0.77, queue=10)
    assert apply_trades(o, [_tr("a", "10:00:05", 0.23, 6, "yes")]) == 0 and o["queue_ahead"] == 4
    assert apply_trades(o, [_tr("b", "10:00:06", 0.23, 7, "yes")]) == 3


class FakeKalshi:
    key = None

    def __init__(self, yes_bids, no_bids, trades=None, result="yes", ticker="KXBTC15M-TEST-00"):
        self.books = {"yes_bids": yes_bids, "no_bids": no_bids}
        self.trades, self.res, self.ticker = trades or [], result, ticker

    def open_market(self, now):
        o = int(now) // 900 * 900
        return {"ticker": self.ticker, "strike": 1000.0, "open_ts": o, "close_ts": o + 900}

    def orderbook(self, ticker):
        return self.books

    def get(self, path, **params):
        return {"trades": self.trades, "cursor": ""}

    def result(self, ticker):
        return self.res


def _prep(bot, move):
    """A flat, squeezed market for two days, then a close `move` dollars away at the end of the first minute."""
    sec = squeezed(5, bars=(WARMUP + 8000) // 300 + 1)
    start = S - WARMUP
    for i in range(WARMUP + 8000):
        bot.series.brti[start + i] = float(sec[i])
    for t in range(S, S + 60):
        bot.series.brti[t] = 1000.0
    bot.series.brti[S + 59] = 1000.0 + move
    bot._fresh = False
    bot.wall = lambda: S + 62
    for t in range(S, S + 62):
        bot.process(t, "brti")


def test_taker_buys_breakout_side_once_per_market_and_settles(tmp_path):
    bot = SqueezePaper(tmp_path, max_lag=60)
    bot.k = FakeKalshi([[0.70, 500.0]], [[0.25, 500.0]], result="yes")
    _prep(bot, +3.0)
    sig = [l for l in (tmp_path / "signals.csv").read_text().splitlines()[1:]]
    assert sum("BUY YES" in l for l in sig) == 1
    assert any("already traded" in l for l in sig)              # 15- and 60-min squeezes fired on the same minute
    assert len(bot.unsettled) == 1
    tr = bot.unsettled[0]
    assert tr["side"] == "YES" and tr["contracts"] == 20 and abs(tr["avg_price"] - 0.75) < 1e-9
    assert (tmp_path / "open_trades.json").exists()
    bot.settle(S + 1000)
    rows = (tmp_path / "trades.csv").read_text().splitlines()
    assert len(rows) == 2 and ",yes," in rows[1] and not bot.unsettled
    assert json.loads((tmp_path / "open_trades.json").read_text()) == []


def test_taker_skips_when_ask_outside_60_89(tmp_path):
    bot = SqueezePaper(tmp_path, max_lag=60)
    bot.k = FakeKalshi([[0.70, 500.0]], [[0.05, 500.0]])        # YES ask 0.95
    _prep(bot, +3.0)
    assert not bot.unsettled
    assert "outside 0.60-0.89" in (tmp_path / "signals.csv").read_text()


def test_taker_down_break_buys_no(tmp_path):
    bot = SqueezePaper(tmp_path, max_lag=60)
    bot.k = FakeKalshi([[0.22, 500.0]], [[0.70, 500.0]], result="no")   # NO ask = 1 - 0.22 = 0.78
    _prep(bot, -3.0)
    assert len(bot.unsettled) == 1 and bot.unsettled[0]["side"] == "NO" and abs(bot.unsettled[0]["avg_price"] - 0.78) < 1e-9


def test_first_start_does_not_trade_a_squeeze_that_was_already_running(tmp_path):
    bot = SqueezePaper(tmp_path, max_lag=60)
    bot.k = FakeKalshi([[0.70, 500.0]], [[0.25, 500.0]])
    sec = squeezed(5, bars=(WARMUP + 8000) // 300 + 1)
    for i in range(WARMUP + 8000):
        bot.series.brti[S - WARMUP + i] = float(sec[i])
    bot.series.brti[S + 59] = 1003.0
    bot.wall = lambda: S + 62
    for t in range(S, S + 62):
        bot.process(t, "brti")
    assert not bot.unsettled and not bot.pending


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def test_maker_rests_bid_fills_and_settles(tmp_path):
    bot = SqueezeMakerPaper(tmp_path, max_lag=60)
    trades = [{"trade_id": "x1", "created_time": iso(S + 65) + ".500000Z", "yes_price_dollars": "0.7400",
               "count_fp": "30.00", "taker_side": "no"}]
    bot.k = FakeKalshi([[0.70, 500.0]], [[0.25, 500.0]], trades=trades, result="yes")
    _prep(bot, +3.0)
    assert len(bot.orders) == 1
    o = next(iter(bot.orders.values()))
    assert o["side"] == "yes" and abs(o["bid"] - 0.74) < 1e-9 and o["queue_ahead"] == 0
    bot.settle(S + 70)                                          # polls the public trades
    assert not bot.orders and len(bot.unsettled) == 1
    tr = bot.unsettled[0]
    assert tr["contracts"] == 20 and abs(tr["avg_price"] - 0.74) < 1e-9 and tr["fee"] == maker_fee(20, 0.74)
    sig = (tmp_path / "signals.csv").read_text()
    assert "BID YES @ 0.74" in sig and "BUY YES (limit fill)" in sig
    bot.settle(S + 1000)
    assert len((tmp_path / "trades.csv").read_text().splitlines()) == 2


def test_maker_unfilled_bid_is_cancelled_at_close_and_scored(tmp_path):
    bot = SqueezeMakerPaper(tmp_path, max_lag=60)
    bot.k = FakeKalshi([[0.22, 500.0]], [[0.70, 500.0]], trades=[], result="no")   # NO breakout, nobody sells to us
    _prep(bot, -3.0)
    o = next(iter(bot.orders.values()))
    assert o["side"] == "no"
    bot.settle(o["close_ts"] + 1)
    assert not bot.orders and not bot.unsettled and len(bot.missed) == 1
    bot._last_settle = 0
    bot.settle(o["close_ts"] + 120)
    row = (tmp_path / "missed.csv").read_text().splitlines()[1]
    assert ",no," in row and row.rstrip().endswith("1")        # the bid we missed was a winner


def test_dashboard_history_includes_squeeze_buys(tmp_path):
    sq = tmp_path / "squeeze"
    sq.mkdir()
    (sq / "signals.csv").write_text("time_utc,ticker,side,level,level_price,secs_left,brti,squeeze_bars,lag_s,ask,ask_size,book,decision\n"
                                    "2026-11-05 22:00:01,KXBTC15M-X-00,YES,15-min squeeze,100.0,800,100.0,3,2,0.75,50,[],BUY YES\n")
    (sq / "trades.csv").write_text("time_utc,ticker,side,level,secs_left,lag_s,contracts,avg_price,fee,cost,result,won,pnl\n"
                                   "2026-11-05 22:00:01,KXBTC15M-X-00,YES,15-min squeeze,800,2,20,0.75,0.26,15.26,yes,1,4.74\n")
    none = tmp_path / "none"
    h = history_events({"range": none, "breakdown": none, "fastpush": none, "squeeze": sq})
    assert [b["bot"] for b in h["buys"]] == ["Squeeze"] and h["buys"][0]["status"] == "won" and h["buys"][0]["pnl"] == 4.74
