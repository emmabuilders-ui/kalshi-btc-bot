"""The dashboard's web handler: shared answer cache and gzip for the big JSON answers."""
import gzip
import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from kalshi_btc_bot.dashboard import Dashboard, make_handler


class FakeDash:
    """Just enough of Dashboard for the handler: counts how often the logs are really read."""
    cached = Dashboard.cached

    def __init__(self):
        self.logs = {"breakdown": Path("/nonexistent")}
        self._memo, self._memo_lock = {}, threading.Lock()
        self.reads = 0
        self.levels = self

    def since(self, t):
        return []

    def data(self):
        self.reads += 1
        return {"now": time.time(), "prices": [[1, 2.0]] * 5000, "reads": self.reads}

    def history(self):
        return {"prices": []}


def _serve(dash):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(dash, "tok"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/tok/"


def _get(url, gz):
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"} if gz else {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.headers.get("Content-Encoding"), r.read()


def test_data_json_is_shared_between_tabs_and_gzipped():
    dash = FakeDash()
    srv, base = _serve(dash)
    try:
        enc, body = _get(base + "data.json", gz=True)
        assert enc == "gzip"
        first = json.loads(gzip.decompress(body))
        enc2, body2 = _get(base + "data.json", gz=False)        # a second tab, within the same second
        assert enc2 is None
        assert json.loads(body2)["reads"] == first["reads"] == 1   # the logs were read once, not twice
        assert b"<!doctype html>" in gzip.decompress(_get(base, gz=True)[1])
    finally:
        srv.shutdown()


def test_small_answers_are_not_compressed():
    dash = FakeDash()
    srv, base = _serve(dash)
    try:
        enc, body = _get(base + "levels.json?since=0", gz=True)
        assert enc is None and body == b"[]"
    finally:
        srv.shutdown()


def test_coinbreak_bot_is_in_the_performance_table(tmp_path):
    """The ETH/SOL/XRP break bot keeps one folder per coin; each coin gets its own row."""
    import csv
    for coin, won in (("ETH", "1"), ("SOL", "0")):
        d = tmp_path / coin
        d.mkdir()
        with (d / "trades.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_utc", "coin", "ticker", "side", "contracts", "avg_price", "fee", "cost", "won", "pnl"])
            w.writerow(["2026-11-12 18:00:00", coin, "T1", "YES", 20, 0.55, 0.35, 11.35, won, 8.65 if won == "1" else -11.35])

    class Fake:
        logs = {"coinbreak": tmp_path}
    rows = {r["bot"]: r for r in Dashboard.perf(Fake())}
    assert rows["Break + gate ETH (15-min)"]["trades"] == 1 and rows["Break + gate ETH (15-min)"]["wins"] == 1
    assert rows["Break + gate SOL (15-min)"]["wins"] == 0
    assert rows["Break + gate XRP (15-min)"]["trades"] == 0          # no folder yet: an empty row, not an error


def test_lockin_page_and_data_are_served(tmp_path):
    """The lock-in bot has its own page at /lockin/ with stats from its trades.csv."""
    import csv
    d = tmp_path / "lockin"
    d.mkdir()
    cols = ["time_utc", "ticker", "side", "level", "secs_left", "model_prob", "gap_pts", "brti", "locked_avg", "strike",
            "contracts", "avg_price", "fee", "cost", "result", "won", "pnl"]
    with (d / "trades.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow(["2026-11-13 18:14:20", "T1", "YES", "lock-in at 40 s", 40, 0.95, 7.1, 1004, 1004, 1000, 20, 0.85, 0.18, 17.18, "yes", 1, 2.82])
        w.writerow(["2026-11-13 18:29:30", "T2", "NO", "lock-in at 30 s", 30, 0.05, 6.0, 996, 996, 1000, 20, 0.80, 0.23, 16.23, "yes", 0, -16.23])
    dash = FakeDash()
    dash.logs["lockin"] = d
    srv, base = _serve(dash)
    try:
        assert b"Lock-in bot" in _get(base + "lockin/", gz=False)[1]
        j = json.loads(_get(base + "lockin/data.json", gz=False)[1])
        assert j["found"] and j["all"]["trades"] == 2 and j["all"]["wins"] == 1
        assert abs(j["all"]["avg_price"] - 82.5) < 1e-9 and abs(j["all"]["win_pct"] - 50.0) < 1e-9
        assert j["by_sec"]["40"]["trades"] == 1 and j["by_sec"]["10"]["trades"] == 0    # JSON turns the int keys into strings
        assert j["by_side"]["NO"]["trades"] == 1 and j["worse1c"]["ev"] < j["all"]["ev"]
        from kalshi_btc_bot.dashboard import perf_row, _rows
        row = perf_row("Lock-in", _rows(d / "trades.csv"))
        assert row["trades"] == 2 and row["avg_price"] == 82.5 and row["win_pct"] == 50.0
    finally:
        srv.shutdown()
