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

    r = sub.add_parser("record", help="record Kalshi order books and BTC prices every second (read-only)")
    r.add_argument("--out", default="data/recordings")
    r.add_argument("--interval", type=float, default=1.0)
    r.add_argument("--depth", type=int, default=10)
    r.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    lb = sub.add_parser("paper-breakdown", help="paper trade the late-breakdown NO rule (no orders are sent)")
    lb.add_argument("--out", default="runs/breakdown")
    lb.add_argument("--stake", type=float, default=20.0, help="dollars per paper trade")
    lb.add_argument("--max-price", type=float, default=0.96, help="highest NO price to pay")
    lb.add_argument("--push-logs", action="store_true", help="git commit and push --out every hour")
    lb.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    rp = sub.add_parser("paper-range", help="paper trade the hourly Bitcoin range rule (no orders are sent)")
    rp.add_argument("--out", default="runs/range")
    rp.add_argument("--contracts", type=int, default=20, help="contracts per paper trade")
    rp.add_argument("--push-logs", action="store_true", help="git commit and push --out every hour")
    rp.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    fp = sub.add_parser("paper-fastpush", help="paper trade fast-push level breaks on the 15-min markets (no orders are sent)")
    fp.add_argument("--out", default="runs/fastpush")
    fp.add_argument("--contracts", type=int, default=20, help="contracts per paper trade")
    fp.add_argument("--push-logs", action="store_true", help="git commit and push --out every hour")
    fp.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    t = sub.add_parser("touch", help="chance price touches a target within N bars (live Coinbase prices)")
    t.add_argument("coin", help="BTC, ETH or SOL")
    t.add_argument("tf", choices=["1m", "5m", "15m", "1h", "4h"])
    t.add_argument("bars", type=int, help="window length in bars (1-24)")
    t.add_argument("target", nargs="?", help="target as a price (85000) or a move (+1%%, -0.5%%); omit for a ladder")

    sub.add_parser("live", help="live trading (locked)")

    argv = list(sys.argv[1:] if argv is None else argv)
    if "touch" in argv:  # let "-1%" or "-0.5%" be a target, not an option
        for i, x in enumerate(argv):
            if i > argv.index("touch") and x[:2] != "--" and len(x) > 1 and x[0] == "-" and (x[1].isdigit() or x[1] == "."):
                argv.insert(i, "--")
                break
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
    elif a.cmd == "paper":
        from .paper import run_paper
        run_paper(cfg, a.out, a.feed, a.methods, int(a.minutes * 60) if a.minutes else None)
    elif a.cmd == "record":
        from .recorder import Recorder
        n = Recorder(a.out, a.interval, a.depth).run(a.minutes * 60 if a.minutes else None)
        print(f"{n} snapshots written to {a.out}")
    elif a.cmd == "paper-breakdown":
        from .late_breakdown import from_env
        from_env(a.out, a.stake, a.max_price, a.push_logs).run(a.minutes * 60 if a.minutes else None)
    elif a.cmd == "paper-range":
        from .range_paper import from_env as range_env
        range_env(a.out, a.push_logs, a.contracts).run(a.minutes * 60 if a.minutes else None)
    elif a.cmd == "paper-fastpush":
        from .fast_push import from_env as fp_env
        fp_env(a.out, a.push_logs, a.contracts).run(a.minutes * 60 if a.minutes else None)
    elif a.cmd == "touch":
        from .touch_prob import run_cli
        run_cli(a.coin, a.tf, a.bars, a.target)
    elif a.cmd == "live":
        from .broker import LiveBroker, LiveTradingLocked
        try:
            LiveBroker(cfg)
        except (LiveTradingLocked, NotImplementedError) as e:
            print(e, file=sys.stderr)
            sys.exit(2)
