"""Replay the shipped lock-in bot's own decision code (kalshi_btc_bot.lockin_paper) over the Kalshi daily trade files.

For every KXBTC15M contract it takes the bot's checks at 50/40/30/20/10 s before the close, with the real 1-second
BRTI (locked-in prints + trailing 15-minute volatility, through `inputs`) and, as the price, the last Kalshi trade
print ON THAT SIDE (a YES taker's price for YES, a NO taker's for NO) made within 10 s before the check second,
the backtest's definition; a live ask can be worse. `evaluate` and
`limit_price` decide exactly as the bot does; the first qualifying check is the contract's one trade, 20 contracts,
settled on markets.csv. The spread skip cannot be replayed from prints and is not applied.

    python replay_bot.py <kalshi-daily dir> <brti dir or zip> <out.csv> [YYYY-MM-DD from] [YYYY-MM-DD to]
"""

from __future__ import annotations

import csv
import gzip
import io
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kalshi_btc_bot.late_breakdown import kalshi_fee
from kalshi_btc_bot.lockin_paper import CHECK_SECS, MIN_EDGE, VOL_WINDOW, edge, evaluate, inputs, limit_price

CONTRACTS = 20
MAX_AGE = 10          # seconds: a print older than this is not a quote


def load_brti(src: Path, day_from: str, day_to: str) -> tuple[int, np.ndarray]:
    """1-second BRTI as an array indexed by (unix second - t0), NaN where missing, from the daily csv.gz files."""
    files = {}
    if src.is_dir():
        for p in src.rglob("BRTI_1s_*.csv.gz"):
            files[p.name[8:18]] = ("file", p)
    for z in ([src] if src.suffix == ".zip" else list(src.glob("*.zip"))):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if "BRTI_1s_" in n and n.endswith(".csv.gz"):
                    files.setdefault(Path(n).name[8:18], ("zip", z, n))
    days = sorted(d for d in files if day_from <= d <= day_to)
    t0 = int(datetime.strptime(days[0], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) - VOL_WINDOW - 60
    t1 = int(datetime.strptime(days[-1], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + 86400 + 60
    arr = np.full(t1 - t0, np.nan)
    prev = (datetime.strptime(days[0], "%Y-%m-%d").date() - __import__("datetime").timedelta(days=1)).isoformat()
    for d in [prev] + days:          # the day before, for the first checks' volatility window
        f = files.get(d)
        if not f:
            continue
        raw = f[1].read_bytes() if f[0] == "file" else zipfile.ZipFile(f[1]).read(f[2])
        for line in io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(raw))):
            if line.startswith("unix_s"):
                continue
            a = line.split(",")
            t = int(a[0]) - t0
            if 0 <= t < len(arr):
                arr[t] = float(a[2])
    return t0, arr


def ffill(a: np.ndarray) -> np.ndarray:
    idx = np.where(~np.isnan(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = a[idx]
    out[np.isnan(a) & (idx == 0)] = np.nan
    return out


def replay_day(z: Path, t0: int, brti: np.ndarray, out: list[dict]):
    with zipfile.ZipFile(z) as zf:
        mk = {}
        for r in csv.DictReader(io.TextIOWrapper(zf.open("markets.csv"))):
            close = int(datetime.fromisoformat(r["close_time"].replace("Z", "+00:00")).timestamp())
            if r.get("result", "").upper() not in ("YES", "NO") or not r.get("floor_strike"):
                continue
            mk[r["ticker"]] = {"close": close, "strike": float(r["floor_strike"]), "result": r["result"].lower(),
                               "open": int(datetime.fromisoformat(r["open_time"].replace("Z", "+00:00")).timestamp())}
        last: dict[str, list] = defaultdict(list)       # ticker -> prints in the last 70 s: (ts, taker side, price paid)
        name = [n for n in zf.namelist() if n.startswith("trades/")][0]
        for line in io.TextIOWrapper(zf.open(name), newline=""):
            if line.startswith("ticker"):
                continue
            a = line.rstrip("\n").split(",")
            m = mk.get(a[0])
            if not m:
                continue
            ts = float(a[3])
            if m["close"] - 70 <= ts <= m["close"]:
                side = a[7].strip().lower()
                if side in ("yes", "no"):
                    last[a[0]].append((ts, side, float(a[4]) if side == "yes" else float(a[5])))
    for tk, m in mk.items():
        prints = sorted(last.get(tk, []))
        for s in CHECK_SECS:
            tc = m["close"] - s
            e = tc - t0
            if e - VOL_WINDOW < 0 or e >= len(brti):
                continue
            sec = brti[e - VOL_WINDOW:e + 1]
            if np.isnan(sec).any():
                continue
            f = inputs(sec, tc, m["close"], m["strike"])
            if f["p_yes"] is None:
                continue
            sides, age = {}, {}
            for sd in ("yes", "no"):
                q = [p for p in prints if p[1] == sd and tc - MAX_AGE < p[0] <= tc]
                px = round(q[-1][2], 4) if q else None
                sides[sd] = {"asks": [[px, 1e9]] if px else [], "ask": px, "bid": None}
                age[sd] = round(tc - q[-1][0], 1) if q else None
            ev = evaluate(f, sides, MIN_EDGE)
            if ev["look"] is None:
                continue
            side, price, p = ev["look"], ev["ask"], ev["prob"]
            assert limit_price(p, sides[side]["asks"], MIN_EDGE) >= price
            won = int(m["result"] == side)
            fee = kalshi_fee(CONTRACTS, price)
            out.append({"time_utc": datetime.fromtimestamp(tc, timezone.utc).strftime("%Y-%m-%d %H:%M:%S"), "ticker": tk,
                        "secs_left": s, "side": side.upper(), "brti": round(f["brti"], 2), "strike": m["strike"],
                        "locked_avg": round(f["known_avg"], 2), "sigma_1s": round(f["sigma"], 8),
                        "model_prob": round(p, 4), "price": price, "edge_pts": round(100 * edge(p, price), 2),
                        "print_age_s": age[side], "won": won,
                        "pnl": round(CONTRACTS * (won - price) - fee, 2)})
            break


def summary(rows: list[dict], label: str) -> str:
    n = len(rows)
    if not n:
        return f"{label}: no trades"
    win = sum(r["won"] for r in rows) / n
    price = sum(r["price"] for r in rows) / n
    ev = sum(r["pnl"] for r in rows) / n / CONTRACTS
    evs = [((r["won"] - r["price"]) - 0.07 * r["price"] * (1 - r["price"])) for r in rows]
    sd = (sum((x - sum(evs) / n) ** 2 for x in evs) / max(n - 1, 1)) ** 0.5
    return (f"{label}: {n} trades, {100 * win:.1f}% won vs {100 * price:.1f}c paid, win-price {100 * (win - price):+.1f} pts, "
            f"EV {100 * ev:+.2f}c/contract (t {ev / (sd / n ** 0.5) if sd else float('nan'):.1f}), P/L ${sum(r['pnl'] for r in rows):+.2f}")


def main():
    daily, brti_src, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    day_from = sys.argv[4] if len(sys.argv) > 4 else "2026-08-01"
    day_to = sys.argv[5] if len(sys.argv) > 5 else "2026-10-25"
    t0, arr = load_brti(brti_src, day_from, day_to)
    arr = ffill(arr)
    print(f"BRTI loaded: {np.isfinite(arr).sum():,} seconds", flush=True)
    rows: list[dict] = []
    for z in sorted(daily.glob("Kalshi_BTC15M_*.zip")):
        d = z.stem[-10:]
        if not day_from <= d <= day_to:
            continue
        before = len(rows)
        replay_day(z, t0, arr, rows)
        print(f"{d}: {len(rows) - before} trades", flush=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print()
    print(summary(rows, "All"))
    for mo in sorted({r["time_utc"][:7] for r in rows}):
        print(summary([r for r in rows if r["time_utc"][:7] == mo], mo))
    for s in CHECK_SECS:
        print(summary([r for r in rows if r["secs_left"] == s], f"first fire at {s} s"))
    for sd in ("YES", "NO"):
        print(summary([r for r in rows if r["side"] == sd], sd))


if __name__ == "__main__":
    main()
