"""Binance 1-s klines -> 1-s close array (brti.npy layout, T0 = 2025-11-30 00:00 UTC) in work_<COIN>/."""
import sys, os, glob, zipfile, io, numpy as np, pandas as pd
sym = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
T0 = 1764460800; end = int(pd.Timestamp('2026-10-26', tz='UTC').timestamp()); n = end - T0
arr = np.full(n, np.nan)
for f in sorted(glob.glob(f'{here}/{sym}/{sym}-1s-*.zip')):
    try:
        z = zipfile.ZipFile(f); df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None, usecols=[0, 4])
    except Exception as e:
        print('BAD', f, e); continue
    t = df[0].to_numpy()
    t = (t // 1_000_000 if t[0] > 1e14 else t // 1000).astype(np.int64) - T0
    ok = (t >= 0) & (t < n); arr[t[ok]] = df[4].to_numpy()[ok]
out = f'{here}/../work_{sym[:3]}'; os.makedirs(out, exist_ok=True); np.save(out + '/brti.npy', arr)
print(sym, 'missing seconds', int(np.isnan(arr).sum()), 'pct', round(np.isnan(arr).mean() * 100, 3))
