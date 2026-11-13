"""Calibration of Kalshi-implied 15-min bands (width fitted to the price at the strike) and of the model's 15-min bands,
on Kalshi's own settlement values. Price at minute m left = last trade price in the minute before (a proxy for the quote).
Usage: python calibrate15.py brti.npy out.csv zip1 zip2 ..."""
import sys, io, zipfile, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kalshi_btc_bot.range_paper import RangeModel
from kalshi_btc_bot import prob_bands as pb
T0 = 1764460800; brti = np.load(sys.argv[1]); M = RangeModel()
SEP = pd.Timestamp('2026-09-01', tz='UTC').timestamp(); MS = [1, 2, 3, 5, 7, 10, 14]
def px(t):
    i = int(t) - T0
    for k in range(30):
        if brti[i - k] == brti[i - k]: return brti[i - k]
    return np.nan
rows = []
for zf in sys.argv[3:]:
    z = zipfile.ZipFile(zf); mk = pd.read_csv(io.BytesIO(z.read('markets.csv')))
    mk['expiration_value'] = pd.to_numeric(mk.expiration_value, errors='coerce'); mk['floor_strike'] = pd.to_numeric(mk.floor_strike, errors='coerce')
    tr = pd.read_csv(io.BytesIO(z.read([n for n in z.namelist() if n.startswith('trades/')][0])), usecols=['ticker', 'ts_unix', 'yes_price']).dropna()
    tr = tr.sort_values('ts_unix'); g = {k: v for k, v in tr.groupby('ticker')}
    for _, r in mk.iterrows():
        if r.ticker not in g or r.result not in ('YES', 'NO') or not (r.expiration_value > 0) or not (r.floor_strike > 0): continue
        c = int(pd.Timestamp(r.close_time).timestamp()); K = float(r.floor_strike); S = float(r.expiration_value)
        d = g[r.ticker]; ts = d.ts_unix.to_numpy(); yp = d.yes_price.to_numpy()
        for m in MS:
            t = c - 60 * m; j = np.searchsorted(ts, t, side='right') - 1
            if j < 0 or t - ts[j] > 90: continue                     # no trade in the last 90 s
            p = px(t); cl = np.array([px(t - 1 - 60 * (100 - k)) for k in range(101)])
            if p != p or np.isnan(cl).any(): continue
            v = M.vol(cl)
            if v is None: continue
            f = pb.implied_width_mult(M, p, K, float(yp[j]), m, v)
            rec = dict(m=m, per='Dec-Aug' if t < SEP else 'Sep-Oct', f=f if f else np.nan)
            for tag, mult in (('model', 1.0), ('kalshi', f)):
                if mult is None: continue
                o = pb.model_offsets(M, m, v, mult=mult)
                for lv, a, b in pb.LEVELS: rec[f'{tag}_in{lv}'] = float(p * np.exp(o[a]) <= S <= p * np.exp(o[b]))
            rows.append(rec)
    print(zf, len(rows), flush=True)
df = pd.DataFrame(rows); df.to_csv(sys.argv[2], index=False)
print(df.groupby(['per', 'm']).agg(n=('f', 'size'), nk=('kalshi_in80', 'count'), f=('f', 'median'),
      **{f'{t}_{n}': (f'{t}_in{n}', 'mean') for t in ('model', 'kalshi') for n in (50, 80, 95)}).round(3).to_string())
