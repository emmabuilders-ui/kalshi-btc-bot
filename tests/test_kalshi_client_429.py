"""KalshiClient.get: a 429 is retried with a growing wait (Retry-After wins), then the result is returned."""
from kalshi_btc_bot import late_breakdown as lb


class Resp:
    def __init__(self, code, headers=None):
        self.status_code, self.headers = code, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"{self.status_code}")

    def json(self):
        return {"ok": True}


class Session:
    def __init__(self, codes):
        self.codes, self.calls = list(codes), 0

    def get(self, *a, **k):
        self.calls += 1
        c = self.codes.pop(0)
        return Resp(c, {"Retry-After": "7"} if c == 429 and self.calls == 1 else {})


def _client(codes, monkeypatch):
    c = lb.KalshiClient.__new__(lb.KalshiClient)
    c.s, c.timeout, c.key, c.key_id = Session(codes), 5, None, None
    c._sign = lambda *a: {}
    naps = []
    monkeypatch.setattr(lb.time, "sleep", naps.append)
    monkeypatch.setattr(lb.random, "uniform", lambda a, b: 0.0)
    return c, naps


def test_429_then_success(monkeypatch):
    c, naps = _client([429, 429, 200], monkeypatch)
    assert c.get("/x", retries=5, wait=3.0) == {"ok": True}
    assert naps == [7.0, 6.0]        # Retry-After honoured first, then wait * (k + 1)
    assert c.s.calls == 3


def test_gives_up_after_retries(monkeypatch):
    c, naps = _client([429] * 3, monkeypatch)
    try:
        c.get("/x", retries=3, wait=1.0)
    except RuntimeError as e:
        assert str(e) == "429"
    else:
        raise AssertionError("expected the last 429 to be raised")
    assert len(naps) == 3
