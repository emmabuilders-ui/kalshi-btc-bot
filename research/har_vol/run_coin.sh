#!/bin/bash
# usage: run_coin.sh ETHUSDT  -> stage 1 for that coin
cd "$(dirname "$0")"; sym=$1; c=${sym:0:3}
python3 bn/build.py $sym && python3 vol1.py work_$c && mkdir -p out_$c && python3 har1.py work_$c out_$c > out_$c/stage1.log 2>&1 && echo "STAGE1 $c OK" || echo "STAGE1 $c FAILED"
