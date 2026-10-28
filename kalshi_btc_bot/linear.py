"""Futures / ETF backtest mode.

Runs the same entry signal (lower-band breach, then MA50/MA250 golden cross within
120 s) and the same four exits on an instrument whose P&L moves one-for-one with
price: micro futures (MNQ, MES, MBT) or ETFs (TQQQ, SQQQ, ...).

Differences from the Kalshi mode:

* There is no $1/$0 settlement, so every trade also has a hard stop (default
  3 x SD250 below entry) and a time stop (default 900 s, matching the 15-minute
  contract). "hold_to_settlement" becomes "hold until the time stop".
* There is no YES-price filter. After an entry, new entries are locked out for
  `lockout_s` (default = max_hold_s), so every exit method trades identical entries.
* Optional short side: an upper-band breach followed by a death cross, with the
  trailing stop mirrored (activates at the lower exit band, exits after 60 s above middle).
* Market-hours instruments (ETFs) only enter during regular hours and are flattened
  before the close. Data gaps longer than `max_ffill_s` reset the indicators and
  flatten open trades, so overnight and weekend gaps never become flat fake prices.

Commission and tick values in the presets are typical retail numbers. Check your broker.
"""

from __future__ import annotations

import csv
import json
import tomllib
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import Config, ExitMethod
from .data import to_epoch_seconds
from .indicators import ExitBandSnapshot, IndicatorHub, TrailingStop

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Instrument:
    symbol: str
    multiplier: float            # $ P&L per 1.0 move in price, per contract/share
    tick_size: float
    commission_per_side: float   # $ per contract/share per side (exchange + broker)
    default_quantity: int
    session: str                 # "24h" (futures, crypto) or "rth" (9:30-16:00 ET)
    half_spread_ticks: float = 0.5   # used only when the data has no bid/ask
    description: str = ""


PRESETS: dict[str, Instrument] = {
    "MNQ": Instrument("MNQ", 2.0, 0.25, 0.62, 1, "24h", description="Micro E-mini Nasdaq-100, $2/pt"),
    "MES": Instrument("MES", 5.0, 0.25, 0.62, 1, "24h", description="Micro E-mini S&P 500, $5/pt"),
    "MBT": Instrument("MBT", 0.1, 5.0, 2.50, 1, "24h", description="CME Micro Bitcoin, 0.1 BTC"),
    "TQQQ": Instrument("TQQQ", 1.0, 0.01, 0.0, 100, "rth", description="3x Nasdaq-100 ETF"),
    "SQQQ": Instrument("SQQQ", 1.0, 0.01, 0.0, 100, "rth", description="-3x Nasdaq-100 ETF"),
    "UPRO": Instrument("UPRO", 1.0, 0.01, 0.0, 100, "rth", description="3x S&P 500 ETF"),
    "SOXL": Instrument("SOXL", 1.0, 0.01, 0.0, 100, "rth", description="3x semiconductor ETF"),
}


@dataclass(frozen=True)
class LinearConfig:
    quantity: int | None = None      # None -> instrument default
    max_hold_s: int = 900            # time stop
    hard_stop_sd: float = 3.0        # stop distance = k x SD250 at the signal
    hard_stop_points: float | None = None   # fixed stop distance in price units (overrides SD)
    lockout_s: int | None = None     # None -> max_hold_s
    allow_short: bool = False
    latency_s: int = 1
    max_ffill_s: int = 5
    no_entry_last_s: int = 900       # rth only: no entries in the last 15 minutes
    flatten_before_close_s: int = 60 # rth only


def load_linear_config(path) -> tuple[LinearConfig, dict]:
    if path is None:
        return LinearConfig(), {}
    data = tomllib.loads(Path(path).read_text())
    return replace(LinearConfig(), **data.get("linear", {})), data.get("instrument", {})


def resolve_instrument(symbol: str, overrides: dict | None = None) -> Instrument:
    base = PRESETS.get(symbol.upper())
    if base is None:
        if not overrides:
            raise KeyError(f"no preset for {symbol}; presets: {sorted(PRESETS)} "
                           "(or define [instrument] in the config)")
        return Instrument(symbol=symbol, **overrides)
    return replace(base, **(overrides or {}))


# ------------------------------------------------------------------------- data
@dataclass
class Segment:
    t0: int
    price: np.ndarray
    bid: np.ndarray | None
    ask: np.ndarray | None


def load_price_segments(path, max_ffill_s: int = 5) -> list[Segment]:
    """Load price.csv (timestamp, price[, bid, ask]); split at gaps > max_ffill_s."""
    df = pd.read_csv(path)
    df["t"] = to_epoch_seconds(df["timestamp"])
    if "price" not in df.columns:
        df["price"] = (df["bid"] + df["ask"]) / 2
    df = df.sort_values("t").groupby("t").last()
    has_book = {"bid", "ask"} <= set(df.columns)
    gaps = np.flatnonzero(np.diff(df.index.to_numpy()) > max_ffill_s + 1)
    bounds = np.split(np.arange(len(df)), gaps + 1)
    segs = []
    for b in bounds:
        part = df.iloc[b]
        full = pd.RangeIndex(part.index[0], part.index[-1] + 1)
        part = part.reindex(full).ffill()
        segs.append(Segment(int(full[0]), part["price"].to_numpy(float),
                            part["bid"].to_numpy(float) if has_book else None,
                            part["ask"].to_numpy(float) if has_book else None))
    return segs


def rth_session(t: int) -> tuple[int, int] | None:
    """(open, close) epoch seconds of the regular session containing t, else None."""
    d = datetime.fromtimestamp(t, ET)
    if d.weekday() >= 5:
        return None
    o = int(datetime.combine(d.date(), dtime(9, 30), ET).timestamp())
    c = int(datetime.combine(d.date(), dtime(16, 0), ET).timestamp())
    return (o, c) if o <= t < c else None


# ----------------------------------------------------------------------- trades
@dataclass
class LinearTrade:
    method: str
    side: str                    # "long" | "short"
    signal_t: int
    signal_price: float
    qty: int
    stop_distance: float
    deadline_t: int              # time stop / session flatten, whichever is first
    entry_t: int | None = None
    entry_price: float | None = None
    stop_price: float | None = None
    exit_signal_t: int | None = None
    exit_reason: str | None = None
    exit_t: int | None = None
    exit_price: float | None = None
    stop_activated_t: int | None = None
    gross_pnl: float = 0.0
    commission: float = 0.0
    trail: TrailingStop | None = field(default=None, repr=False)

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.commission

    @property
    def is_open(self) -> bool:
        return self.exit_t is None

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("trail")
        d["net_pnl"] = round(self.net_pnl, 4)
        d["gross_pnl"] = round(self.gross_pnl, 4)
        return d


class LinearBacktester:
    def __init__(self, cfg: Config, lcfg: LinearConfig, inst: Instrument):
        self.cfg, self.lcfg, self.inst = cfg, lcfg, inst
        self.qty = lcfg.quantity or inst.default_quantity
        self.lockout = lcfg.max_hold_s if lcfg.lockout_s is None else lcfg.lockout_s
        self.trades: list[LinearTrade] = []
        self.events: list[dict] = []
        self.rejected: dict[str, int] = {}

    # Executable prices: buy at the ask, sell at the bid.
    def _px(self, seg: Segment, i: int, action: str) -> float:
        if seg.bid is not None and not np.isnan(seg.bid[i]) and not np.isnan(seg.ask[i]):
            return float(seg.ask[i] if action == "buy" else seg.bid[i])
        half = self.inst.half_spread_ticks * self.inst.tick_size
        return float(seg.price[i] + (half if action == "buy" else -half))

    def run(self, segments: list[Segment]) -> "LinearBacktester":
        for seg in segments:
            self._run_segment(seg)
        return self

    def _run_segment(self, seg: Segment) -> None:
        hub = IndicatorHub(self.cfg.entry, self.cfg.exit_methods)
        last_upper_t: int | None = None
        lockout_until = -1
        active: list[LinearTrade] = []
        n = len(seg.price)
        for i in range(n):
            t = seg.t0 + i
            price = float(seg.price[i])
            e, snaps = hub.update(t, price)
            if e.upper_breach:
                last_upper_t = t
            for tr in list(active):
                self._step(tr, seg, i, t, price, snaps.get(tr.method))
                if not tr.is_open:
                    active.remove(tr)
            # ---- entries
            long_sig = e.golden_cross and e.recent_lower_breach
            short_sig = (self.lcfg.allow_short and e.death_cross and last_upper_t is not None
                         and 0 <= t - last_upper_t <= self.cfg.entry.breach_window_s)
            if not (long_sig or short_sig):
                continue
            side = "long" if long_sig else "short"
            reason = None
            deadline = t + self.lcfg.max_hold_s
            if t < lockout_until:
                reason = "lockout"
            elif i + self.lcfg.latency_s >= n:
                reason = "end_of_data"
            elif self.inst.session == "rth":
                sess = rth_session(t)
                if sess is None:
                    reason = "outside_regular_hours"
                elif t >= sess[1] - self.lcfg.no_entry_last_s:
                    reason = "too_close_to_market_close"
                else:
                    deadline = min(deadline, sess[1] - self.lcfg.flatten_before_close_s)
            if reason:
                self.rejected[reason] = self.rejected.get(reason, 0) + 1
                self.events.append({"event": "signal_rejected", "t": t, "side": side, "reason": reason})
                continue
            lockout_until = t + self.lockout
            dist = (self.lcfg.hard_stop_points if self.lcfg.hard_stop_points is not None
                    else self.lcfg.hard_stop_sd * (e.sd250 or 0.0))
            self.events.append({"event": "entry_signal", "t": t, "side": side, "price": price,
                                "stop_distance": round(dist, 6), "indicators": e.as_log()})
            for m in self.cfg.exit_methods:
                tr = LinearTrade(m.name, side, t, price, self.qty, dist, deadline,
                                 trail=TrailingStop(m, side) if m.kind != "hold" else None)
                self.trades.append(tr)
                active.append(tr)
        # Data ends or a gap starts: flatten at the last available price.
        for tr in active:
            if tr.entry_t is None:
                tr.exit_t, tr.exit_reason = seg.t0 + n - 1, "not_filled_data_gap"
                continue
            self._fill_exit(tr, seg, n - 1, seg.t0 + n - 1, tr.exit_reason or "data_gap_or_end")

    def _step(self, tr: LinearTrade, seg: Segment, i: int, t: int, price: float,
              snap: ExitBandSnapshot | None) -> None:
        lat = self.lcfg.latency_s
        if tr.entry_t is None:
            if t >= tr.signal_t + lat:
                tr.entry_t = t
                tr.entry_price = self._px(seg, i, "buy" if tr.side == "long" else "sell")
                sgn = 1 if tr.side == "long" else -1
                tr.stop_price = tr.entry_price - sgn * tr.stop_distance if tr.stop_distance > 0 else None
            return
        if tr.exit_signal_t is not None:
            if t >= tr.exit_signal_t + lat:
                self._fill_exit(tr, seg, i, t, tr.exit_reason)
            return
        reason = None
        if tr.stop_price is not None and (price <= tr.stop_price if tr.side == "long"
                                          else price >= tr.stop_price):
            reason = "hard_stop"
        elif t >= tr.deadline_t:
            reason = "time_stop"
        elif tr.trail is not None and snap is not None:
            was = tr.trail.activated
            fire = tr.trail.update(snap)
            if tr.trail.activated and not was:
                tr.stop_activated_t = t
            if fire:
                reason = "trailing_stop"
        if reason:
            tr.exit_signal_t, tr.exit_reason = t, reason

    def _fill_exit(self, tr: LinearTrade, seg: Segment, i: int, t: int, reason: str) -> None:
        tr.exit_t = t
        tr.exit_reason = reason
        tr.exit_price = self._px(seg, i, "sell" if tr.side == "long" else "buy")
        sgn = 1 if tr.side == "long" else -1
        tr.gross_pnl = sgn * (tr.exit_price - tr.entry_price) * tr.qty * self.inst.multiplier
        tr.commission = 2 * self.inst.commission_per_side * tr.qty

    # ------------------------------------------------------------------ report
    def summary_rows(self) -> list[dict]:
        rows = []
        for m in self.cfg.exit_methods:
            ts = [t for t in self.trades if t.method == m.name and t.entry_t is not None]
            reasons: dict[str, int] = {}
            for t in ts:
                reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
            gross = sum(t.gross_pnl for t in ts)
            comm = sum(t.commission for t in ts)
            rows.append({
                "exit_method": m.name if m.kind != "hold" else "hold_until_time_stop",
                "trades": len(ts), "longs": sum(t.side == "long" for t in ts),
                "shorts": sum(t.side == "short" for t in ts),
                "winners": sum(t.net_pnl > 0 for t in ts),
                "avg_hold_s": round(float(np.mean([t.exit_t - t.entry_t for t in ts])), 1) if ts else 0,
                "gross_pnl": round(gross, 2), "commission": round(comm, 2),
                "net_pnl": round(gross - comm, 2),
                "worst_trade": round(min((t.net_pnl for t in ts), default=0.0), 2),
                "exit_reasons": json.dumps(reasons, sort_keys=True),
            })
        return rows

    def entries_identical(self) -> bool:
        sets = {}
        for t in self.trades:
            sets.setdefault(t.method, []).append((t.signal_t, t.side, t.entry_price))
        vals = list(sets.values())
        return all(v == vals[0] for v in vals)

    def write_report(self, out_dir, data_note: str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = self.summary_rows()
        with open(out / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        trades = [t.summary() for t in self.trades]
        if trades:
            with open(out / "trades.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(trades[0]))
                w.writeheader()
                w.writerows(trades)
        with open(out / "events.jsonl", "w") as f:
            for e in self.events:
                f.write(json.dumps(e, default=str) + "\n")
        i, l = self.inst, self.lcfg
        stop = (f"{l.hard_stop_points} points" if l.hard_stop_points is not None
                else f"{l.hard_stop_sd} x SD250")
        lines = [f"# {i.symbol} exit method comparison", "",
                 f"**Data:** {data_note}", "",
                 f"- Instrument: {i.symbol} ({i.description}), ${i.multiplier:g} per point, "
                 f"{self.qty} per trade, ${i.commission_per_side:g} commission per side",
                 f"- Hard stop {stop}; time stop {l.max_hold_s} s; shorts "
                 f"{'on' if l.allow_short else 'off'}; session {i.session}",
                 f"- Entries identical across exit methods: **{self.entries_identical()}**",
                 f"- Rejected signals: {json.dumps(self.rejected, sort_keys=True) or 'none'}", "",
                 "Prices are executable: buy at the ask, sell at the bid, 1 second after each decision.", "",
                 "| Exit method | Trades | Winners | Avg hold (s) | Gross P&L | Commission | Net P&L | Worst trade | Exit reasons |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in rows:
            lines.append(f"| {r['exit_method']} | {r['trades']} | {r['winners']} | {r['avg_hold_s']} | "
                         f"${r['gross_pnl']:,.2f} | ${r['commission']:,.2f} | ${r['net_pnl']:,.2f} | "
                         f"${r['worst_trade']:,.2f} | {r['exit_reasons']} |")
        (out / "report.md").write_text("\n".join(lines) + "\n")
        return out / "report.md"
