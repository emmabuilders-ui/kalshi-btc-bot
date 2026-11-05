from kalshi_btc_bot.maker_paper import MakerPaper, apply_trades, bid_price, maker_fee


def test_bid_price_and_fee():
    assert bid_price(0.74, 0.78) == (0.75, True)
    assert bid_price(0.76, 0.78) == (0.77, True)
    assert bid_price(0.77, 0.78) == (0.77, False)
    assert maker_fee(20, 0.75) == 0.07          # 1.75% x 20 x .75 x .25 = 0.0656 -> 0.07
    assert maker_fee(20, 0.75, 0.0) == 0.0


def _order(bid=0.75, queue=0.0):
    return {"bid": bid, "queue_ahead": queue, "contracts": 20, "filled": 0, "posted_iso": "2026-11-05T10:00:00"}


def _tr(tid, ts, price, n, taker="no"):
    return {"trade_id": tid, "created_time": f"2026-11-05T{ts}Z", "yes_price_dollars": f"{price:.4f}",
            "count_fp": f"{n:.2f}", "taker_side": taker}


def test_fills_only_from_yes_sellers_at_or_below_bid_after_posting():
    o = _order()
    got = apply_trades(o, [_tr("a", "09:59:59", 0.70, 50),            # before posting
                           _tr("b", "10:00:05", 0.74, 5, "yes"),     # a YES buyer: not ours
                           _tr("c", "10:00:06", 0.76, 5),            # above our bid
                           _tr("d", "10:00:07", 0.75, 6),            # at our bid, nobody ahead
                           _tr("e", "10:00:08", 0.74, 3)])           # through us
    assert got == 9 and o["filled"] == 9
    assert apply_trades(o, [_tr("e", "10:00:08", 0.74, 3)]) == 0     # same trade again: ignored
    assert apply_trades(o, [_tr("f", "10:00:09", 0.73, 100)]) == 11 and o["filled"] == 20


def test_queue_ahead_goes_first():
    o = _order(0.77, queue=10)
    assert apply_trades(o, [_tr("a", "10:00:05", 0.77, 6)]) == 0 and o["queue_ahead"] == 4
    assert apply_trades(o, [_tr("b", "10:00:06", 0.77, 7)]) == 3 and o["queue_ahead"] == 0


class FakeKalshi:
    key = "k"

    def __init__(self, markets, books, trades, result="yes"):
        self.markets, self.books, self.trades, self.res = markets, books, trades, result

    def get(self, path, **params):
        if path == "/markets/trades":
            return {"trades": self.trades, "cursor": ""}
        return {"markets": self.markets}

    def orderbook(self, ticker):
        return self.books.get(ticker, {})

    def result(self, ticker):
        return self.res


def _bot(tmp_path, trades, result="yes"):
    markets = [{"ticker": "B", "status": "active", "floor_strike": 1000, "cap_strike": 1024.99,
                "yes_bid_dollars": 0.74, "yes_ask_dollars": 0.78}]
    books = {"B": {"yes_bids": [[0.74, 30.0]], "no_bids": [[0.22, 100.0]], "yes_bid": 0.74, "yes_ask": 0.78}}
    bot = MakerPaper(tmp_path, None, None)
    bot.k = FakeKalshi(markets, books, trades, result)
    close = 3600 * 1000
    bot.brti = {s: 1010.0 for s in range(close - 3600, close)}
    return bot, close


def test_posts_bid_fills_partially_and_settles(tmp_path):
    from datetime import datetime, timezone
    close = 3600 * 1000
    iso = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    trades = [{"trade_id": "x", "created_time": iso(close - 590) + ".1Z", "yes_price_dollars": "0.7500",
               "count_fp": "8.00", "taker_side": "no"}]
    bot, close = _bot(tmp_path, trades)
    bot.check(close - 10 * 60)
    o = bot.orders["B"]
    assert o["bid"] == 0.75 and o["improved"] and o["queue_ahead"] == 0 and o["taker_avg"] == 0.78
    bot.poll_orders(close - 580)
    assert bot.orders["B"]["filled"] == 8
    bot.check(close - 9 * 60)                    # still trend-in: keeps resting, no second bid
    assert list(bot.orders) == ["B"] and not bot.held
    bot.check(close - 3 * 60)                    # cancel at 3 min left: 8 filled are held
    assert not bot.orders and bot.held["B"]["contracts"] == 8 and bot.held["B"]["avg_price"] == 0.75
    bot.settle(close + 61)
    rows = (tmp_path / "trades.csv").read_text().splitlines()
    assert len(rows) == 2 and ",yes,1," in rows[1]
    ev = [r.split(",")[1] for r in (tmp_path / "orders.csv").read_text().splitlines()[1:]]
    assert ev == ["post", "fill", "done"]


def test_unfilled_bid_goes_to_missed_and_cancels_when_trend_fails(tmp_path):
    bot, close = _bot(tmp_path, [], result="no")
    bot.check(close - 10 * 60)
    assert "B" in bot.orders
    bot.brti = {s: 1030.0 for s in range(close - 3600, close)}    # BRTI left the band
    bot.check(close - 9 * 60)
    assert not bot.orders and not bot.held and "B" in bot.missed
    bot.settle(close + 61)
    m = (tmp_path / "missed.csv").read_text().splitlines()
    assert len(m) == 2 and m[1].endswith(",no,0") and not bot.missed
