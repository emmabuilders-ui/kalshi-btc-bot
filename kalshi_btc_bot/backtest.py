"""Backtester: runs every exit method on identical entry signals.

One IndicatorHub feeds one StrategyEngine per (exit method, pricing mode). Every
engine uses the same entry rule and the same broker model, so their entries are
identical; only the exit differs. The run checks that and reports any divergence.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .broker import PaperBroker
from .config import Config
from .data import MarketData, settle_result
from .engine import EventLog, StrategyEngine, Tick
from .indicators import IndicatorHub

PRICING_MODES = ("executable", "last")


@dataclass
class BacktestResult:
    engines: dict[tuple[str, str], StrategyEngine]   # (method, pricing) -> engine
    events: EventLog
    settlements: list[dict]
    entries_identical: bool
    notes: list[str]

    def summary_rows(self) -> list[dict]:
        rows = []
        for (method, pricing), eng in self.engines.items():
            closed = eng.closed_positions()
            gross = sum(p.gross_pnl for p in closed)
            fees = sum(p.fees for p in closed)
            rows.append({
                "exit_method": method, "pricing": pricing, "entries": len(closed),
                "missed_entries": sum(p.status == "missed" for p in eng.positions.values()),
                "unsettled": len(eng.open_positions()),
                "early_exits": sum(p.sold > 0 for p in closed),
                "winners": sum(p.gross_pnl > 0 for p in closed),
                "contracts": sum(p.bought for p in closed),
                "gross_pnl": round(gross, 2), "fees": round(fees, 2),
                "net_pnl": round(gross - fees, 2),
            })
        return rows

    def trades(self) -> list[dict]:
        out = []
        for (method, pricing), eng in self.engines.items():
            for p in eng.positions.values():
                d = p.summary()
                d["pricing"] = pricing
                out.append(d)
        return out


def run_backtest(data: MarketData, cfg: Config, pricing_modes=PRICING_MODES,
                 event_log_path: str | Path | None = None) -> BacktestResult:
    events = EventLog(event_log_path)
    hub = IndicatorHub(cfg.entry, cfg.exit_methods)
    engines = {(m.name, pm): StrategyEngine(cfg, m, PaperBroker(cfg, pm), events, f"{m.name}/{pm}")
               for pm in pricing_modes for m in cfg.exit_methods}
    notes: list[str] = []
    if not data.has_brti:
        notes.append("No BRTI column: settlement estimated from spot 60s average (spot != BRTI).")
    if data.filled_seconds:
        notes.append(f"{data.filled_seconds} missing BTC seconds were forward-filled.")

    markets = data.markets
    mi = 0                       # index of the first market whose close_time is in the future
    qi = 0
    latest: dict[str, object] = {}
    unsettled = list(markets)
    settlements: list[dict] = []

    for i, price in enumerate(data.price):
        t = data.t0 + i
        # 1. Settle every market that has closed by now.
        while unsettled and unsettled[0].close_time <= t:
            m = unsettled.pop(0)
            settlements.append(_settle(m, data, cfg, engines, events))
        # 2. Advance quotes to t.
        while qi < len(data.quotes) and data.quotes[qi].t <= t:
            latest[data.quotes[qi].ticker] = data.quotes[qi]
            qi += 1
        while mi < len(markets) and markets[mi].close_time <= t:
            mi += 1
        market = markets[mi] if mi < len(markets) and markets[mi].is_open(t) else None
        quote = latest.get(market.ticker) if market else None
        # 3. Indicators and engines.
        entry_snap, exit_snaps = hub.update(t, float(price))
        tick = Tick(t, float(price), entry_snap, market, quote)
        for (name, _), eng in engines.items():
            eng.on_tick(tick, exit_snaps.get(name))
    for m in unsettled:
        if m.close_time == data.t_end + 1:
            settlements.append(_settle(m, data, cfg, engines, events))
        elif any(p.ticker == m.ticker for e in engines.values() for p in e.open_positions()):
            notes.append(f"{m.ticker} closes after the data ends; its positions are unsettled.")

    entry_sets = {}
    for (name, pm), eng in engines.items():
        entry_sets.setdefault(pm, []).append(
            sorted((p.ticker, p.signal_t, p.bought, round(p.cost, 6)) for p in eng.positions.values()))
    identical = all(all(s == sets[0] for s in sets) for sets in entry_sets.values())
    if not identical:
        notes.append("Entries differ between exit methods (check max_open_positions / "
                     "max_entries_per_contract).")
    events.close()
    return BacktestResult(engines, events, settlements, identical, notes)


def _settle(m, data, cfg, engines, events) -> dict:
    est = data.settlement_value(m, cfg.settlement.window_s)
    est_result = settle_result(est, m.strike, cfg.settlement) if est is not None else None
    result = m.result or est_result
    rec = {"ticker": m.ticker, "close_time": m.close_time, "strike": m.strike,
           "official_result": m.result, "estimated_value": est, "estimated_result": est_result,
           "result_used": result, "index": "brti" if data.has_brti else "spot_proxy"}
    events.write("market_settled", **rec)
    if result is not None:
        for eng in engines.values():
            eng.settle(m.ticker, result, m.close_time, est)
    return rec


def write_report(res: BacktestResult, out_dir: str | Path, title: str, data_note: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = res.summary_rows()
    trades = res.trades()
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if trades:
        with open(out / "trades.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(trades[0]))
            w.writeheader()
            w.writerows(trades)
    with open(out / "settlements.json", "w") as f:
        json.dump(res.settlements, f, indent=1)

    lines = [f"# {title}", "", f"**Data:** {data_note}", ""]
    lines += [f"- {n}" for n in res.notes]
    lines += [f"- Entries identical across exit methods: **{res.entries_identical}**", ""]
    for pm in PRICING_MODES:
        sub = [r for r in rows if r["pricing"] == pm]
        if not sub:
            continue
        label = ("Executable prices (buy at YES ask, sell at YES bid, book size limits, fees)"
                 if pm == "executable" else "Last-trade prices (same method as the Oct 26 exploration)")
        lines += [f"## {label}", "",
                  "| Exit method | Entries | Early exits | Winners | Gross P&L | Fees | Net P&L |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for r in sub:
            lines.append(f"| {r['exit_method']} | {r['entries']} | {r['early_exits']} | {r['winners']} | "
                         f"${r['gross_pnl']:,.2f} | ${r['fees']:,.2f} | ${r['net_pnl']:,.2f} |")
        lines.append("")
    lines += ["Per-trade detail is in trades.csv, every signal/order/fill/settlement in events.jsonl."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return out / "report.md"
