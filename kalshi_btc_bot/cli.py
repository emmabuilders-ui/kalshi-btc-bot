"""Command line: synth | backtest | paper | live."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="kalshi_btc_bot")
    ap.add_argument("--config", help="TOML config file (defaults are the spec values)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="generate SYNTHETIC test data")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--hours", type=float, default=24)
    s.add_argument("--seed", type=int, default=7)

    b = sub.add_parser("backtest", help="compare the four exit methods on historical data")
    b.add_argument("--data", required=True, help="folder with btc.csv, markets.csv, quotes.csv")
    b.add_argument("--out", default="results/backtest")
    b.add_argument("--synthetic", action="store_true", help="label the report as synthetic data")

    p = sub.add_parser("paper", help="paper trade on live data (no orders are sent)")
    p.add_argument("--out", default="runs/paper")
    p.add_argument("--feed", default="composite", choices=["composite", "coinbase", "kraken", "bitstamp"])
    p.add_argument("--methods", nargs="*", help="exit methods to run (default: all)")
    p.add_argument("--minutes", type=float, help="stop after this many minutes")

    sl = sub.add_parser("synth-linear", help="generate SYNTHETIC futures/ETF test data")
    sl.add_argument("--symbol", default="MNQ")
    sl.add_argument("--price", type=float, default=21_000.0, help="starting price")
    sl.add_argument("--out", default="data/synthetic_linear")
    sl.add_argument("--hours", type=float, default=48)
    sl.add_argument("--seed", type=int, default=7)

    bl = sub.add_parser("backtest-linear", help="backtest on futures or ETFs (MNQ, MES, MBT, TQQQ, ...)")
    bl.add_argument("--data", required=True, help="price.csv, or a folder containing it")
    bl.add_argument("--symbol", required=True, help="preset symbol, or any name with [instrument] in --linear-config")
    bl.add_argument("--linear-config", help="TOML with [linear] and [instrument] sections")
    bl.add_argument("--quantity", type=int)
    bl.add_argument("--allow-short", action="store_true")
    bl.add_argument("--out", default="results/linear")

    sub.add_parser("live", help="live trading (locked)")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(a.config)

    if a.cmd == "synth":
        from .synthetic import generate
        print(f"SYNTHETIC data written to {generate(a.out, a.hours, a.seed)}")
    elif a.cmd == "backtest":
        from pathlib import Path
        from .backtest import run_backtest, write_report
        from .data import load_dir
        synthetic = a.synthetic or (Path(a.data) / "SYNTHETIC_DATA_README.txt").exists()
        data = load_dir(a.data)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        res = run_backtest(data, cfg, event_log_path=Path(a.out) / "events.jsonl")
        note = ("SYNTHETIC data (pipeline test only; P&L is meaningless)" if synthetic
                else f"historical data from {a.data}")
        report = write_report(res, a.out, "Exit method comparison", note)
        print(report.read_text())
    elif a.cmd == "synth-linear":
        from .linear import resolve_instrument
        from .synthetic import generate_linear
        inst = resolve_instrument(a.symbol)
        print(f"SYNTHETIC data written to "
              f"{generate_linear(a.out, inst.symbol, a.price, inst.tick_size, a.hours, a.seed)}")
    elif a.cmd == "backtest-linear":
        from dataclasses import replace
        from pathlib import Path
        from .linear import LinearBacktester, load_linear_config, load_price_segments, resolve_instrument
        lcfg, inst_over = load_linear_config(a.linear_config)
        if a.quantity:
            lcfg = replace(lcfg, quantity=a.quantity)
        if a.allow_short:
            lcfg = replace(lcfg, allow_short=True)
        inst = resolve_instrument(a.symbol, inst_over)
        src = Path(a.data)
        csv_path = src / "price.csv" if src.is_dir() else src
        synthetic = (csv_path.parent / "SYNTHETIC_DATA_README.txt").exists()
        bt = LinearBacktester(cfg, lcfg, inst).run(load_price_segments(csv_path, lcfg.max_ffill_s))
        note = ("SYNTHETIC data (pipeline test only; P&L is meaningless)" if synthetic
                else f"historical data from {csv_path}")
        print(bt.write_report(a.out, note).read_text())
    elif a.cmd == "paper":
        from .paper import run_paper
        run_paper(cfg, a.out, a.feed, a.methods, int(a.minutes * 60) if a.minutes else None)
    elif a.cmd == "live":
        from .broker import LiveBroker, LiveTradingLocked
        try:
            LiveBroker(cfg)
        except (LiveTradingLocked, NotImplementedError) as e:
            print(e, file=sys.stderr)
            sys.exit(2)
