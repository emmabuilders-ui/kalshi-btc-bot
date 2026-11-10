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

    rm = sub.add_parser("paper-range5", help="paper trade the 5-account hourly range plan (no orders are sent)")
    rm.add_argument("--out", default="runs/range5")
    rm.add_argument("--scale", type=float, default=1 / 3, help="1.0 = sizes for about $3,000; default 1/3 (about $1,000)")
    rm.add_argument("--push-logs", action="store_true", help="git commit and push --out every hour")
    rm.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    fp = sub.add_parser("paper-fastpush", help="paper trade fast-push level breaks on the 15-min markets (no orders are sent)")
    fp.add_argument("--out", default="runs/fastpush")
    fp.add_argument("--contracts", type=int, default=20, help="contracts per paper trade")
    fp.add_argument("--push-logs", action="store_true", help="git commit and push --out every hour")
    fp.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    fr = sub.add_parser("paper-fastpush-range", help="paper trade fast-push level breaks on the hourly range bands (no orders are sent)")
    fr.add_argument("--out", default="runs/fastpush-range")
    fr.add_argument("--contracts", type=int, default=20, help="contracts per paper trade")
    fr.add_argument("--push-logs", action="store_true", help="git commit and push --out every hour")
    fr.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    fv = sub.add_parser("paper-favorite", help="paper trade YES favorites (75-84c, last 3 min) on the hourly range bands (no orders are sent)")
    fv.add_argument("--out", default="runs/favorite")
    fv.add_argument("--contracts", type=int, default=20, help="contracts per paper trade")
    fv.add_argument("--push-logs", action="store_true", help="git commit and push --out every hour")
    fv.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    tp = sub.add_parser("paper-trend", help="paper trade trend-in YES favorites 4-20 min before the hourly close (no orders are sent)")
    tp.add_argument("--rule", choices=["broad", "tight"], default="tight",
                    help="broad = YES 61-80c within 80%% of the band centre or 81-90c within 40%%; tight = YES 70-84c within 40%%")
    tp.add_argument("--out", default="runs/trend")
    tp.add_argument("--contracts", type=int, default=20, help="contracts per paper trade")
    tp.add_argument("--push-logs", action="store_true", help="git commit and push --out every hour")
    tp.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    mp = sub.add_parser("paper-maker", help="paper trade the tight trend-in signal with a resting limit bid instead of taking the ask (no orders are sent)")
    mp.add_argument("--out", default="runs/maker")
    mp.add_argument("--contracts", type=int, default=20, help="contracts per paper bid")
    mp.add_argument("--maker-fee", type=float, default=0.0175, help="maker fee rate (fee = rate x C x P x (1-P)); 0 if the market has none")
    mp.add_argument("--push-logs", action="store_true", help="git commit and push --out every hour")
    mp.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    qp = sub.add_parser("paper-quiet", help="paper trade quiet YES favorites (61-95c) 1-18 min before the hourly close (no orders are sent)")
    qp.add_argument("--out", default="runs/quiet")
    qp.add_argument("--contracts", type=int, default=20, help="contracts per paper trade")
    qp.add_argument("--push-logs", action="store_true", help="git commit and push --out every hour")
    qp.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    sq = sub.add_parser("paper-squeeze", help="paper trade squeeze breaks on the 15-min markets, taking the ask (no orders are sent)")
    sq.add_argument("--out", default="runs/squeeze")
    sq.add_argument("--contracts", type=int, default=20, help="contracts per paper trade")
    sq.add_argument("--push-logs", action="store_true", help="git commit and push --out every hour")
    sq.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    sm = sub.add_parser("paper-squeeze-maker", help="paper trade squeeze breaks on the 15-min markets with a resting limit bid (no orders are sent)")
    sm.add_argument("--out", default="runs/squeeze-maker")
    sm.add_argument("--contracts", type=int, default=20, help="contracts per paper bid")
    sm.add_argument("--maker-fee", type=float, default=0.0175, help="maker fee rate (fee = rate x C x P x (1-P)); 0 if the market has none")
    sm.add_argument("--push-logs", action="store_true", help="git commit and push --out every hour")
    sm.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    mg = sub.add_parser("paper-model-gap", help="paper trade the Direct ML model-gap rule on the 15-min markets (NO side, no orders are sent)")
    mg.add_argument("--out", default="runs/model-gap")
    mg.add_argument("--contracts", type=int, default=20, help="contracts per paper trade")
    mg.add_argument("--push-logs", action="store_true", help="git commit and push --out every hour")
    mg.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    cg = sub.add_parser("paper-corr-gap", help="paper trade the Correction ML model-gap rule on the 15-min markets (NO side, no orders are sent)")
    cg.add_argument("--out", default="runs/corr-gap")
    cg.add_argument("--contracts", type=int, default=20, help="contracts per paper trade")
    cg.add_argument("--push-logs", action="store_true", help="git commit and push --out every hour")
    cg.add_argument("--minutes", type=float, help="stop after this many minutes (default: run forever)")

    d = sub.add_parser("dashboard", help="serve a live chart of the paper bots (read-only)")
    d.add_argument("--port", type=int, default=8080)
    d.add_argument("--token-file", default="/etc/kalshi-bot/dash_token", help="file holding the secret page path")
    d.add_argument("--breakdown", default="/opt/kalshi-logs/breakdown")
    d.add_argument("--range", default="/opt/kalshi-logs-range/range")
    d.add_argument("--fastpush", default="/opt/kalshi-logs-fastpush/fastpush")
    d.add_argument("--range5", default="/opt/kalshi-logs-range5/range5")
    d.add_argument("--fastpush-range", default="/opt/kalshi-logs-fastpush-range/fastpush-range")
    d.add_argument("--favorite", default="/opt/kalshi-logs-favorite/favorite")
    d.add_argument("--trend-broad", default="/opt/kalshi-logs-trend-broad/trend-broad")
    d.add_argument("--trend-tight", default="/opt/kalshi-logs-trend-tight/trend-tight")
    d.add_argument("--maker", default="/opt/kalshi-logs-maker/maker")
    d.add_argument("--quiet", default="/opt/kalshi-logs-quiet/quiet")
    d.add_argument("--squeeze", default="/opt/kalshi-logs-squeeze/squeeze")
    d.add_argument("--squeeze-maker", default="/opt/kalshi-logs-squeeze-maker/squeeze-maker")
    d.add_argument("--model-gap", default="/opt/kalshi-logs-model-gap/model-gap")
    d.add_argument("--corr-gap", default="/opt/kalshi-logs-corr-gap/corr-gap")

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
    elif a.cmd == "paper-range5":
        from .range_multi import from_env as r5_env
        r5_env(a.out, a.push_logs, a.scale).run(a.minutes * 60 if a.minutes else None)
    elif a.cmd == "paper-fastpush":
        from .fast_push import from_env as fp_env
        fp_env(a.out, a.push_logs, a.contracts).run(a.minutes * 60 if a.minutes else None)
    elif a.cmd == "paper-fastpush-range":
        from .fast_push_range import from_env as fpr_env
        fpr_env(a.out, a.push_logs, a.contracts).run(a.minutes * 60 if a.minutes else None)
    elif a.cmd == "paper-favorite":
        from .favorite_paper import from_env as fav_env
        fav_env(a.out, a.push_logs, a.contracts).run(a.minutes * 60 if a.minutes else None)
    elif a.cmd == "paper-trend":
        from .trend_paper import from_env as trend_env
        trend_env(a.out, a.push_logs, a.contracts, a.rule).run(a.minutes * 60 if a.minutes else None)
    elif a.cmd == "paper-maker":
        from .maker_paper import from_env as maker_env
        maker_env(a.out, a.push_logs, a.contracts, a.maker_fee).run(a.minutes * 60 if a.minutes else None)
    elif a.cmd == "paper-quiet":
        from .quiet_paper import from_env as quiet_env
        quiet_env(a.out, a.push_logs, a.contracts).run(a.minutes * 60 if a.minutes else None)
    elif a.cmd == "paper-squeeze":
        from .squeeze_paper import from_env as sq_env
        sq_env(a.out, a.push_logs, a.contracts).run(a.minutes * 60 if a.minutes else None)
    elif a.cmd == "paper-squeeze-maker":
        from .squeeze_paper import maker_from_env as sqm_env
        sqm_env(a.out, a.push_logs, a.contracts, a.maker_fee).run(a.minutes * 60 if a.minutes else None)
    elif a.cmd == "paper-model-gap":
        from .model_gap_paper import from_env as mg_env
        mg_env(a.out, a.push_logs, a.contracts).run(a.minutes * 60 if a.minutes else None)
    elif a.cmd == "paper-corr-gap":
        from .model_gap_paper import corr_from_env
        corr_from_env(a.out, a.push_logs, a.contracts).run(a.minutes * 60 if a.minutes else None)
    elif a.cmd == "dashboard":
        from pathlib import Path
        from .dashboard import serve
        serve(a.port, Path(a.token_file).read_text().strip(),
              {"breakdown": Path(a.breakdown), "range": Path(a.range), "fastpush": Path(a.fastpush),
               "range5": Path(a.range5), "fastpush_range": Path(a.fastpush_range), "favorite": Path(a.favorite),
               "trend_broad": Path(a.trend_broad), "trend_tight": Path(a.trend_tight), "maker": Path(a.maker), "quiet": Path(a.quiet),
               "squeeze": Path(a.squeeze), "squeeze_maker": Path(a.squeeze_maker),
               "model_gap": Path(a.model_gap), "corr_gap": Path(a.corr_gap)})
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
