import csv
import json

import numpy as np

from kalshi_btc_bot import coin_break_paper as cb
from kalshi_btc_bot.coin_break_paper import CoinBreakPaper, break_threshold, gate_limit, next_state

CLOSE = 1_800_000_900          # a 15-minute close
K = 100.0


def test_threshold_scales_with_the_coin_price():
    assert abs(break_threshold(75000.0) - 5.0) < 1e-9          # $5 on BTC at 75,000
    assert abs(break_threshold(2500.0) - 5.0 * 2500 / 75000) < 1e-9


def test_break_needs_a_threshold_on_each_side_of_the_strike():
    thr = break_threshold(K)
    prev, brk = next_state(None, K - 3 * thr, K)
    assert (prev, brk) == ("no", None)                          # first reading: just the starting side
    assert next_state("no", K + thr / 2, K) == ("no", None)     # across the strike but not far enough
    assert next_state("no", K - 2 * thr, K) == ("no", None)     # same side again
    assert next_state("no", K + 2 * thr, K) == ("yes", "yes")   # a real break upward
    assert next_state("yes", K - 2 * thr, K) == ("no", "no")    # and downward
    assert next_state(None, K + 2 * thr, K) == ("yes", None)    # no break without a side to leave


def test_gate_limit_is_the_highest_price_the_chance_still_covers():
    levels = [[0.60, 10], [0.70, 10], [0.90, 10]]
    lim = gate_limit(0.80, levels)
    assert lim == 0.70                                          # 0.70 + fee 0.0147 <= 0.80, 0.90 is not
    assert gate_limit(0.50, levels) == 0.0


class Stub:
    def __init__(self, no_bids, yes_bids=None):
        self.book = {"no_bids": no_bids, "yes_bids": yes_bids or []}
        self.results = {}

    def orderbook(self, ticker):
        return self.book

    def result(self, ticker):
        return self.results.get(ticker)


def make(tmp_path, monkeypatch, stub, wall_lag=1):
    bot = CoinBreakPaper(tmp_path, coins=("ETH",), contracts=20)
    coin = bot.coins["ETH"]
    coin.client = stub
    monkeypatch.setattr(cb.time, "time", lambda: float(CLOSE - 130 + wall_lag))
    # quiet index: 99.9 until 130 s left, then 100.1 (a break upward with the average still to come)
    for t in range(CLOSE - 900, CLOSE):
        coin.series[t] = (99.9 if CLOSE - t > 130 else 100.1) + 0.0003 * np.sin(t)
    coin.session = cb.Session("KXETH15M-TEST", K, CLOSE - 900, CLOSE, cursor=CLOSE - 210)
    return bot, coin


def run_seconds(bot, coin, upto):
    for t in range(coin.session.cursor, upto + 1):
        bot.step(coin, t)
    coin.session.cursor = upto + 1


def test_break_with_a_gate_that_holds_buys_at_the_ask(tmp_path, monkeypatch):
    bot, coin = make(tmp_path, monkeypatch, Stub(no_bids=[[0.40, 50]]))     # YES ask 60c
    run_seconds(bot, coin, CLOSE - 125)
    assert len(coin.unsettled) == 1
    tr = coin.unsettled[0]
    assert tr["side"] == "YES" and tr["contracts"] == 20 and tr["avg_price"] == 0.60 and tr["secs_left"] == 130
    rows = list(csv.DictReader((tmp_path / "ETH" / "signals.csv").open()))
    assert rows[0]["decision"] == "BUY YES" and float(rows[0]["model"]) > 0.7
    run_seconds(bot, coin, CLOSE - 1)
    assert len(coin.unsettled) == 1                                          # one trade per contract


def test_gate_that_fails_skips(tmp_path, monkeypatch):
    bot, coin = make(tmp_path, monkeypatch, Stub(no_bids=[[0.001, 50]]))     # YES ask 99.9c: out of range
    run_seconds(bot, coin, CLOSE - 100)
    assert not coin.unsettled
    rows = list(csv.DictReader((tmp_path / "ETH" / "signals.csv").open()))
    assert rows[0]["decision"].startswith("skip: YES ask")


def test_late_decision_is_skipped(tmp_path, monkeypatch):
    bot, coin = make(tmp_path, monkeypatch, Stub(no_bids=[[0.40, 50]]), wall_lag=30)
    run_seconds(bot, coin, CLOSE - 100)
    assert not coin.unsettled
    rows = list(csv.DictReader((tmp_path / "ETH" / "signals.csv").open()))
    assert "after the break" in rows[0]["decision"]


def test_break_outside_the_window_is_ignored(tmp_path, monkeypatch):
    bot, coin = make(tmp_path, monkeypatch, Stub(no_bids=[[0.40, 50]]))
    for t in range(CLOSE - 900, CLOSE):                                      # the break comes with 40 s left
        coin.series[t] = 99.9 if CLOSE - t > 40 else 100.1
    run_seconds(bot, coin, CLOSE - 1)
    assert not coin.unsettled and not (tmp_path / "ETH" / "signals.csv").exists()


def test_settle_writes_trades_and_clears_open(tmp_path, monkeypatch):
    stub = Stub(no_bids=[[0.40, 50]])
    bot, coin = make(tmp_path, monkeypatch, stub)
    run_seconds(bot, coin, CLOSE - 125)
    stub.results["KXETH15M-TEST"] = "yes"
    bot.settle(coin, CLOSE + 60)
    assert not coin.unsettled and json.loads((tmp_path / "ETH" / "open_trades.json").read_text()) == []
    row = next(csv.DictReader((tmp_path / "ETH" / "trades.csv").open()))
    assert row["won"] == "1" and float(row["pnl"]) > 0
    assert json.loads((tmp_path / "ETH" / "status.json").read_text())["settled_trades"] == 1


def test_prices_and_sessions_are_logged_for_the_coin_page(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from kalshi_btc_bot.dashboard_coin import coin_data
    bot, coin = make(tmp_path, monkeypatch, Stub(no_bids=[[0.40, 50]]))
    now = CLOSE - 130
    bot._log_prices(coin, now)
    bot._append(coin, "sessions.csv", {"ticker": "KXETH15M-TEST", "open_utc": cb._iso(CLOSE - 900),
                                       "close_utc": cb._iso(CLOSE), "strike": K})
    coin.last_flush = 0
    bot._log_prices(coin, now + 100)              # a second flush must not duplicate seconds
    h = datetime.fromtimestamp(CLOSE - 10, timezone.utc).strftime("%Y-%m-%d/%H")
    assert (tmp_path / "ETH" / "prices" / f"{h}.csv.gz").exists()
    d = coin_data(tmp_path / "ETH", "ETH", now=CLOSE)
    assert len(d["prices"]) == len({p[0] for p in d["prices"]}) and d["prices"][-1][0] == CLOSE - 1
    assert d["sessions"][0]["strike"] == K and d["perf"]["trades"] == 0
