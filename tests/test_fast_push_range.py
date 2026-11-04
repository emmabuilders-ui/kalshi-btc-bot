import numpy as np

from kalshi_btc_bot.fast_push import FPSession
from kalshi_btc_bot.fast_push_range import FastPushRangePaper, band_label, band_of


def _mk(lo, hi, tk, st=None):
    m = {"ticker": tk, "status": "active"}
    if lo is not None:
        m["floor_strike"] = lo
    if hi is not None:
        m["cap_strike"] = hi
    if st:
        m["strike_type"] = st
    return m


MARKETS = [_mk(None, 954.99, "T-LOW", "less"), _mk(955, 979.99, "A"), _mk(980, 1004.99, "B"), _mk(1005, 1029.99, "C"),
           _mk(1030, None, "T-HIGH", "greater")]


def test_band_lookup_and_labels():
    assert band_of(MARKETS, 1000.0)["ticker"] == "B" and band_of(MARKETS, 980.0)["ticker"] == "B"
    assert band_of(MARKETS, 979.995)["ticker"] == "A"
    assert band_of(MARKETS, 940.0)["ticker"] == "T-LOW" and band_of(MARKETS, 2000.0)["ticker"] == "T-HIGH"
    assert band_label(MARKETS[2]) == "980-1,005" and band_label(MARKETS[0]) == "below 955"
    assert band_label(MARKETS[4]) == "above 1,030"


class FakeKalshi:
    key = None

    def __init__(self, books, result="no"):
        self.books, self.res = books, result
        self.calls = []

    def get(self, path, **params):
        return {"markets": MARKETS}

    def orderbook(self, ticker):
        self.calls.append(ticker)
        return self.books.get(ticker, {"yes_bids": [], "no_bids": []})

    def result(self, ticker):
        return self.res


def _bot(tmp_path, books, result="no", noise=0.5):
    bot = FastPushRangePaper(tmp_path, max_lag=None)
    bot.k = FakeKalshi(books, result)
    o = 3600 * 1000 + 2700  # a session that closes on the hour (xx:45 -> xx:00)
    rng = np.random.default_rng(1)
    for t in range(o - 7000, o + 700):
        bot.series.brti[t] = 1000.0 + rng.normal(0, noise)
    return bot, o


def _push_down(bot, o, to=970.0):
    base = bot.series.brti[o + 699]
    for k, sec in enumerate(range(700, 781)):  # sharp drop to `to` at 780 s (120 s left); 15 s earlier ~44% of the way
        bot.series.brti[o + sec] = base + (to - base) * (k / 80) ** 4
    for sec in range(700, 781):
        bot.step(o + sec, bot.series.brti[o + sec])


def test_push_across_band_edge_buys_yes_new_band_and_no_old_band(tmp_path):
    # BRTI falls $5 through 975 from band B (980-1005) into band A (955-980): YES on A at 0.88, NO on B at 0.90
    books = {"A": {"yes_bids": [[0.80, 500.0]], "no_bids": [[0.12, 500.0]]},
             "B": {"yes_bids": [[0.10, 500.0]], "no_bids": [[0.85, 500.0]]}}
    bot, o = _bot(tmp_path, books, result="yes")
    bot.session = FPSession("S", 1000.0, o, o + 900, 1000.0, {"sup": ("no", 975.0)})
    _push_down(bot, o, 970.0)
    sides = {(t["ticker"], t["side"]) for t in bot.unsettled}
    assert sides == {("A", "YES"), ("B", "NO")}
    yes = next(t for t in bot.unsettled if t["side"] == "YES")
    assert yes["contracts"] == 20 and yes["avg_price"] == 0.88 and yes["floor"] == 955
    assert "judge: into 955-980, out of 980-1,005" in (tmp_path / "breaks.csv").read_text()
    bot.settle(o + 900 + 61)
    rows = (tmp_path / "trades.csv").read_text().splitlines()
    assert len(rows) == 3 and sum(",yes,1," in r for r in rows) == 1 and sum(",yes,0," in r for r in rows) == 1


def test_push_inside_one_band_is_skipped(tmp_path):
    bot, o = _bot(tmp_path, {})
    bot.session = FPSession("S", 1000.0, o, o + 900, 1000.0, {"sup": ("no", 1012.0)})
    for t in range(o - 7000, o + 700):
        bot.series.brti[t] = 1015.0 + (bot.series.brti[t] - 1000.0)
    _push_down(bot, o, 1007.0)   # 1015 -> 1007 stays inside band C (1005-1030)
    assert not bot.unsettled and not bot.k.calls
    assert "push stayed inside 1,005-1,030" in (tmp_path / "breaks.csv").read_text()


def test_not_last_quarter_of_hour_is_skipped(tmp_path):
    bot, o = _bot(tmp_path, {})
    o2 = o - 900  # session closing at xx:45
    bot.session = FPSession("S", 1000.0, o2, o2 + 900, 1000.0, {"sup": ("no", 975.0)})
    base = bot.series.brti[o2 + 699]
    for k, sec in enumerate(range(700, 781)):
        bot.series.brti[o2 + sec] = base + (970.0 - base) * (k / 80) ** 4
        bot.step(o2 + sec, bot.series.brti[o2 + sec])
    assert not bot.unsettled
    assert "not the last 15 minutes of the hour" in (tmp_path / "breaks.csv").read_text()


def test_bad_price_or_repeat_band_skips(tmp_path):
    # YES ask on A is 0.62 (not allowed); NO ask on B is 0.90 (allowed) -> only the NO trade
    books = {"A": {"yes_bids": [[0.5, 500.0]], "no_bids": [[0.38, 500.0]]},
             "B": {"yes_bids": [[0.10, 500.0]], "no_bids": [[0.85, 500.0]]}}
    bot, o = _bot(tmp_path, books)
    bot.session = FPSession("S", 1000.0, o, o + 900, 1000.0, {"sup": ("no", 975.0)})
    _push_down(bot, o, 970.0)
    assert [t["side"] for t in bot.unsettled] == ["NO"]
    assert "not under 50c or 75-96c" in (tmp_path / "signals.csv").read_text()
    # a second qualifying break in the same session must not trade band B again
    bot.session.done.clear()
    bot.session.levels["sup2"] = ("no", 973.0)
    _push_down(bot, o, 968.0)
    assert len(bot.unsettled) == 1
    assert "band already traded this hour" in (tmp_path / "signals.csv").read_text()


def test_dashboard_fastpush_range_events(tmp_path):
    from kalshi_btc_bot.dashboard import fastpush_range_events, history_events, pnl_points, _rows
    d = tmp_path / "fastpush-range"
    d.mkdir()
    (d / "signals.csv").write_text(
        "break_utc,ticker,floor,cap,side,level,secs_left,brti,push,lag_s,ask,ask_size,book,book_utc,decision\n"
        "2026-11-04 15:58:00,A,955,979.99,YES,sup,119,970,5.1,1.0,0.88,500,[],2026-11-04 15:58:01,BUY YES\n"
        "2026-11-04 15:58:00,B,980,1004.99,NO,sup,119,970,5.1,1.0,0.90,500,[],2026-11-04 15:58:01,BUY NO\n"
        "2026-11-04 16:58:00,C,1005,1029.99,YES,sup,119,1010,5.1,1.0,0.60,500,[],2026-11-04 16:58:01,skip: YES ask 0.60 not under 50c or 75-96c\n")
    (d / "trades.csv").write_text(
        "time_utc,ticker,side,floor,cap,level,secs_left,push,lag_s,contracts,avg_price,fee,cost,result,won,pnl\n"
        "2026-11-04 15:58:01,A,YES,955,979.99,sup,119,5.1,1.0,20,0.88,0.15,17.75,yes,1,2.25\n")
    ev = fastpush_range_events(d)
    assert [(e["bot"], e["side"], e["status"]) for e in ev] == [("FP range", "YES", "won"), ("FP range", "NO", "open")]
    assert ev[0]["lo"] == 955 and ev[0]["hi"] == 979.99 and ev[0]["close"] % 3600 == 0 and ev[0]["pnl"] == 2.25
    assert ev[1]["price"] == 0.90 and ev[1]["close"] == ev[0]["close"]
    assert fastpush_range_events(None) == [] and fastpush_range_events(tmp_path / "missing") == []
    for k in ("range", "breakdown", "fastpush"):
        (tmp_path / k).mkdir()
    h = history_events({"range": tmp_path / "range", "breakdown": tmp_path / "breakdown",
                        "fastpush": tmp_path / "fastpush", "fastpush_range": d})
    assert len(h["range"]) == 2 and h["range"][0]["bot"] == "FP range"
    assert pnl_points(_rows(d / "trades.csv"), "FP range")[0]["pnl"] == 2.25
