#!/bin/bash
# Binance 1-s klines, daily zips, Nov 30 2025 - Oct 25 2026
cd "$(dirname "$0")"
for s in ETHUSDT XRPUSDT; do mkdir -p $s; d=2025-11-30
  while [ "$d" != "2026-10-26" ]; do
    f=$s/${s}-1s-$d.zip
    if [ ! -s $f ]; then curl -sS -f -o $f "https://data.binance.vision/data/spot/daily/klines/$s/1s/${s}-1s-$d.zip" || echo "MISS $s $d"; fi
    d=$(date -u -d "$d + 1 day" +%F)
  done; echo "DONE $s"
done
