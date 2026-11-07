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
