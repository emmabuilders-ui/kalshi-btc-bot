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


def test_coinbreak_pages_are_served_and_empty_logs_are_fine(tmp_path):
    dash = FakeDash()
    dash.logs["coinbreak"] = tmp_path                      # no coin folders yet
    srv, base = _serve(dash)
    try:
        html = urllib.request.urlopen(base + "coinbreak/SOL/").read().decode()
        assert "SOL break bot" in html and "__COIN__" not in html
        j = json.loads(urllib.request.urlopen(base + "coinbreak/SOL/data.json").read())
        assert j["coin"] == "SOL" and j["prices"] == [] and j["perf"]["trades"] == 0
        assert urllib.request.urlopen(base + "coinbreak/ETH").url.endswith("/coinbreak/ETH/")
        try:
            urllib.request.urlopen(base + "coinbreak/DOGE/")
            raise AssertionError("unknown coin should 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        srv.shutdown()


def test_models_page_has_a_results_row_per_bot(tmp_path):
    import csv
    from kalshi_btc_bot.dashboard_models import PAGEM, models_data
    d = tmp_path / "mg"
    d.mkdir()
    with (d / "trades.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_utc", "ticker", "side", "contracts", "avg_price", "cost", "won", "pnl"])
        w.writerow(["2026-11-12 18:00:00", "T1", "NO", 20, 0.80, 16.4, 1, 3.6])
    out = models_data({"model_gap": d})
    assert out["perf"][0]["bot"] == "Direct ML" and out["perf"][0]["trades"] == 1 and out["perf"][0]["win_pct"] == 100.0
    assert "id=\"perf\"" in PAGEM and "ftWatch" in PAGEM
