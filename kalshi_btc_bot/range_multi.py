"""Hourly Bitcoin range (Kalshi KXBTC) PAPER trader split into 5 separate accounts.
Read-only: no orders are ever sent. Runs as its own service, separate from the other bots.

Michael's plan, 2026-11-03 (backtest Dec 2025 - Oct 2026, Sep-Oct = honest test):

* Same market set-up as the range bot: the 14 bands nearest the BRTI price 30 minutes before
  the hourly close, checked every minute from 30 minutes left, held to settlement.
* Only quotes with a YES spread of 5c or less are used (as in the backtest).
* Each account takes at most ONE trade per hour (the first band that qualifies) and keeps its own
  results. "Rebuilt" = the finish-in-range model with its move table rebuilt on Dec-Aug 2026 BRTI
  (data/range_cdf_rebuilt.json); "current" = the range bot's model (data/range_cdf.json).

  1. YES 90-99c, rebuilt model >= ask + 2c ............ 80 contracts
  2. NO, current model, 75-89c +10c / 90-99c +2c ...... 40 contracts (the range bot's NO rule)
  3. YES 75-89c, rebuilt model >= ask ................. 56 contracts
  4. NO 90-99c, rebuilt model >= ask + 2c ............. 12 contracts
  5. YES 90-99c, rebuilt model >= ask + 2c, doubling:  spend $7, double after each win, back to $7
     after a loss, never more than $224 (own $350 balance)

  Sizes above are scale 1.0 (about $3,000 of capital). The service runs at --scale 0.333
  (about $1,000): 27 / 13 / 19 / 4 contracts and $2.33 / $74.67 / $117 for account 5.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from importlib import resources

from .late_breakdown import kalshi_fee
from .range_paper import (N_BANDS, RangeModel, RangePaper, _f, _iso, band_mid, event_ticker, fill)

log = logging.getLogger("kalshi_btc_bot")

WINDOW = 30            # minutes left when checks start
MAX_SPREAD = 0.05      # backtest used quotes with YES ask - bid <= 5c
LO, MID, HI = 0.745, 0.895, 1.0   # 75c, 90c, 100c on a 1c grid

ACCOUNTS = [
    {"id": 1, "name": "YES 90-99c rebuilt +2c", "side": "YES", "model": "rebuilt",
     "groups": [(MID, HI, 0.02)], "contracts": 80},
    {"id": 2, "name": "NO current bot rule", "side": "NO", "model": "current",
     "groups": [(LO, MID, 0.10), (MID, HI, 0.02)], "contracts": 40},
    {"id": 3, "name": "YES 75-89c rebuilt >= ask", "side": "YES", "model": "rebuilt",
     "groups": [(LO, MID, 0.0)], "contracts": 56},
    {"id": 4, "name": "NO 90-99c rebuilt +2c", "side": "NO", "model": "rebuilt",
     "groups": [(MID, HI, 0.02)], "contracts": 12},
    {"id": 5, "name": "YES 90-99c rebuilt +2c doubling", "side": "YES", "model": "rebuilt",
     "groups": [(MID, HI, 0.02)], "base": 7.0, "cap": 224.0, "bank": 350.0},
]


def rebuilt_model() -> RangeModel:
    # the rebuilt table was fitted on real 60-s settlements with x = ln(settle/P)/(s100*sqrt(T-0.5)): keep 30 s
    return RangeModel(str(resources.files("kalshi_btc_bot").joinpath("data/range_cdf_rebuilt.json")), shrink_minutes=0.5)


def rule_limit(groups, model_p: float, ask: float | None) -> float | None:
    """Highest price this account may pay, or None if the ask does not qualify."""
    if ask is None or not 0 < ask < 1:
        return None
    for lo, hi, gap in groups:
        if lo <= ask < hi:
            if model_p + 1e-9 < ask + gap:
                return None
            # pay up to the model minus the gap, staying inside the same price group
            return round(min(model_p - gap, hi - 0.005, 0.99), 4)
    return None


def next_stake(stake: float, won: bool, base: float, cap: float) -> float:
    return min(stake * 2, cap) if won else base


class RangeMulti(RangePaper):
    def __init__(self, out, key_id, key_path, push_logs=False, scale: float = 1 / 3):
        self.scale = scale
        self.accts = []
        for a in ACCOUNTS:
            a = dict(a)
            if "contracts" in a:
                a["contracts"] = max(1, round(a["contracts"] * scale))
            else:
                for k in ("base", "cap", "bank"):
                    a[k] = round(a[k] * scale, 2)
            self.accts.append(a)
        super().__init__(out, key_id, key_path, push_logs)
        self.models = {"current": self.model, "rebuilt": rebuilt_model()}

    # ---------------------------------------------------------------- state
    def _load_open(self):
        p, s = self.out / "open_positions.json", self.out / "state.json"
        self.held = {}
        if p.exists():
            for tr in json.loads(p.read_text()):
                self.held[f'{tr["account"]}:{tr["ticker"]}'] = tr
        self.state = json.loads(s.read_text()) if s.exists() else {}
        self.state.setdefault("traded_hours", [])
        self.state.setdefault("stake5", None)
        self.state.setdefault("cash5", None)

    def _save_open(self):
        (self.out / "open_positions.json").write_text(json.dumps(list(self.held.values()), indent=1))
        self.state["traded_hours"] = self.state["traded_hours"][-500:]
        (self.out / "state.json").write_text(json.dumps(self.state, indent=1))

    def _acct5(self):
        a = next(x for x in self.accts if "base" in x)
        if self.state["stake5"] is None:
            self.state["stake5"], self.state["cash5"] = a["base"], a["bank"]
        return a

    def _status(self, now: float, note: str = ""):
        import csv
        rows = list(csv.DictReader((self.out / "trades.csv").open())) if (self.out / "trades.csv").exists() else []
        by = {}
        for a in self.accts:
            r = [x for x in rows if x["account"] == str(a["id"])]
            by[f'{a["id"]} {a["name"]}'] = {
                "size": a.get("contracts", f'${a.get("base")} doubling, cap ${a.get("cap")}'),
                "trades": len(r), "wins": sum(x["won"] == "1" for x in r),
                "pnl_dollars": round(sum(float(x["pnl"]) for x in r), 2),
                "open": sum(1 for t in self.held.values() if t["account"] == a["id"])}
        a5 = self._acct5()
        by[f'{a5["id"]} {a5["name"]}'].update(next_stake_dollars=self.state["stake5"], cash=round(self.state["cash5"], 2))
        total = round(sum(v["pnl_dollars"] for v in by.values()), 2)
        (self.out / "status.json").write_text(json.dumps({
            "updated_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
            "scale": round(self.scale, 3), "total_pnl_dollars": total, "accounts": by, "note": note}, indent=1))

    # ---------------------------------------------------------------- the rule
    def check(self, t: int):
        close = (t // 3600 + 1) * 3600
        left = (close - t) // 60
        if left > WINDOW or left < 1:
            return
        todo = [a for a in self.accts if f'{a["id"]}:{close}' not in self.state["traded_hours"]]
        if not todo:
            return
        price = self.price_at(t)
        closes = self.minute_closes(t)
        v = self.model.vol(closes) if len(closes) else None
        if price is None or v is None:
            log.warning("%s: not enough BRTI history yet", _iso(t))
            return
        s100, vb = v
        anchor = self.anchor.setdefault(close, self.price_at(close - 1800) or price)
        ev = event_ticker(close)
        markets = self.k.get("/markets", event_ticker=ev, limit=1000).get("markets", [])
        markets = [m for m in markets if m.get("status") in ("active", "open")]
        if not markets:
            log.warning("no open markets for %s", ev)
            return
        markets = sorted(markets, key=lambda m: abs(band_mid(m) - anchor))[:N_BANDS]
        quotes = []
        for m in markets:
            lo, hi = _f(m.get("floor_strike")), _f(m.get("cap_strike"))
            if m.get("strike_type") == "greater":
                hi = None
            elif m.get("strike_type") == "less":
                lo = None
            yb, ya = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
            if yb is None or ya is None or ya - yb > MAX_SPREAD + 1e-9:
                continue
            p = {k: mdl.p_inside(price, lo, hi, left, s100, vb) for k, mdl in self.models.items()}
            quotes.append((m, p, ya, round(1 - yb, 4)))
        for a in todo:
            best = None
            for m, p, ya, na in quotes:
                if f'{a["id"]}:{m["ticker"]}' in self.held:
                    continue
                mp = p[a["model"]] if a["side"] == "YES" else 1 - p[a["model"]]
                ask = ya if a["side"] == "YES" else na
                lim = rule_limit(a["groups"], mp, ask)
                if lim is not None and (best is None or mp - ask > best[1] - best[2]):
                    best = (m, mp, ask, lim)
            if best:
                self.enter(a, t, close, left, *best, price, s100, vb)

    def enter(self, a, t, close, left, m, mp, ask, limit, price, s100, vb):
        tk = m["ticker"]
        try:
            book = self.k.orderbook(tk)
        except Exception as e:
            log.warning("orderbook %s failed: %s", tk, e)
            book = {}
        other = book.get("no_bids" if a["side"] == "YES" else "yes_bids") or []
        if "base" in a:
            self._acct5()
            spend = min(self.state["stake5"], self.state["cash5"])
            want = int(spend // ask) if ask else 0
        else:
            want = a["contracts"]
        n, avg = fill(other, want, limit) if want > 0 else (0, None)
        self._append("signals.csv", {
            "time_utc": _iso(t), "account": a["id"], "ticker": tk, "side": a["side"], "mins_left": left,
            "floor": m.get("floor_strike"), "cap": m.get("cap_strike"), "brti": round(price, 2),
            "model": round(mp, 4), "ask_snapshot": ask, "book_best": round(1 - other[0][0], 4) if other else "",
            "limit": limit, "wanted": want, "filled": n, "avg_price": round(avg, 4) if avg else "",
            "vol_1m": round(s100, 6), "vol_state": vb, "book": json.dumps(other[:6])})
        if n == 0:
            log.info("%s acct %d %s %s: signal at %.2f but no fill under %.2f", _iso(t), a["id"], a["side"], tk, ask, limit)
            return
        fee = kalshi_fee(n, avg)
        cost = round(n * avg + fee, 2)
        if "base" in a:
            self.state["cash5"] = round(self.state["cash5"] - cost, 2)
        self.held[f'{a["id"]}:{tk}'] = {
            "time_utc": _iso(t), "account": a["id"], "ticker": tk, "side": a["side"], "mins_left": left,
            "contracts": n, "wanted": want, "avg_price": round(avg, 4), "fee": fee, "cost": cost,
            "model": round(mp, 4), "close_ts": close}
        self.state["traded_hours"].append(f'{a["id"]}:{close}')
        self._save_open()
        log.info("PAPER BUY acct %d: %d %s %s @ %.4f (model %.3f, %d min left)", a["id"], n, a["side"], tk, avg, mp, left)

    def settle(self, now: int):
        if now - self._last_settle < 60:
            return
        self._last_settle = now
        changed = False
        for key, tr in sorted(self.held.items(), key=lambda kv: kv[1]["close_ts"]):
            if now < tr["close_ts"] + 60:
                continue
            try:
                res = self.k.result(tr["ticker"])
            except Exception as e:
                log.warning("result %s failed: %s", tr["ticker"], e)
                continue
            if res is None:
                continue
            won = int(res == tr["side"].lower())
            pnl = round(tr["contracts"] * (won - tr["avg_price"]) - tr["fee"], 2)
            if tr["account"] == self._acct5()["id"]:
                a5 = self._acct5()
                self.state["cash5"] = round(self.state["cash5"] + tr["cost"] + pnl, 2)
                self.state["stake5"] = next_stake(self.state["stake5"], bool(won), a5["base"], a5["cap"])
            row = {k: v for k, v in tr.items() if k != "close_ts"}
            row.update(result=res, won=won, pnl=pnl)
            self._append("trades.csv", row)
            del self.held[key]
            changed = True
            log.info("SETTLED acct %d %s %s %s pnl %.2f", tr["account"], tr["side"], tr["ticker"], res, pnl)
        if changed:
            self._save_open()
        self._status(now)


def from_env(out: str, push_logs: bool, scale: float) -> RangeMulti:
    import os
    return RangeMulti(out, os.environ.get("KALSHI_KEY_ID"), os.environ.get("KALSHI_KEY_FILE"), push_logs, scale)
