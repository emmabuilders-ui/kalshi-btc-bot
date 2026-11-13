"""Calibration of probability bands around BTC for the 15-min and 1-hour Kalshi windows.

Model bands : range model (s100 x sqrt(time left) -> empirical quantile table, frozen fit), quantiles of the 60-s-average settlement.
Kalshi bands: quantiles of the CDF implied by Kalshi's hourly band prices (see prob_bands.ladder_quantiles).
Check       : fraction of settlements that land inside the central 50 / 80 / 95 % band, by minutes left and period.
Usage: python calibrate.py brti.npy bands.parquet quotes.parquet out.csv
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kalshi_btc_bot.range_paper import RangeModel
from kalshi_btc_bot import prob_bands as pb

T0 = 1764460800
brti = np.load(sys.argv[1]); M = RangeModel()
LV = [(.25, .75, 50), (.10, .90, 80), (.025, .975, 95)]
SEP = pd.Timestamp('2026-09-01', tz='UTC').timestamp()
def px(t):
    i = int(t) - T0
    for k in range(30):
        v = brti[i - k]
        if v == v: return v
    return np.nan
def closes(t): return np.array([px(t - 1 - 60 * (100 - k)) for k in range(101)])
def settle(c):
    a = brti[c - 60 - T0:c - T0]
    return a[~np.isnan(a)].mean() if (~np.isnan(a)).sum() > 40 else np.nan

def model_q(price, left, v, qs):
    o = pb.model_offsets(M, left, v, qs); return [price * np.exp(o[q]) for q in qs]
QS = sorted({q for a, b, _ in LV for q in (a, b)})

rows = []
# ---- model bands, every 5 min for hourly (m in list) and 15-min windows
MH = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60]; M15 = [1, 2, 3, 5, 7, 10, 14]
d0 = int(pd.Timestamp('2025-12-02', tz='UTC').timestamp()); d1 = int(pd.Timestamp('2026-10-25', tz='UTC').timestamp())
for c in range(d0 // 900 * 900, d1, 900):
    hourly = c % 3600 == 0
    for kind, ms in (('15min', M15), ('1hour', MH if hourly else [])):
        for m in ms:
            t = c - 60 * m; p = px(t); cl = closes(t)
            if p != p or np.isnan(cl).any(): continue
            v = M.vol(cl); s = settle(c)
            if v is None or s != s: continue
            qv = dict(zip(QS, model_q(p, m, v, QS)))
            r = dict(src='model', kind=kind, m=m, per='Dec-Aug' if t < SEP else 'Sep-Oct', z=np.log(s / p) / (v[0] * np.sqrt(m)))
            for a, b, n in LV: r[f'in{n}'] = qv[a] <= s <= qv[b]; r[f'w{n}'] = (qv[b] - qv[a]) / p * 1e4
            rows.append(r)
# ---- Kalshi-implied hourly bands
b = pd.read_parquet(sys.argv[2]); b = b[b.strike_type == 'between']
q = pd.read_parquet(sys.argv[3]).merge(b[['ticker', 'close', 'floor_strike', 'cap_strike']], on='ticker')
q['m'] = ((q.close - q.t) // 60).astype(int); q = q[q.m.isin([1, 2, 3, 5, 10, 15, 20, 30])]
q = q[(q.yes_ask > 0) & (q.yes_ask <= 1)]
q = q.sort_values('floor_strike')
for (c, m), g in q.groupby(['close', 'm']):
    t = c - 60 * m
    if len(g) < 10 or (g.floor_strike.diff().dropna() > 150).any(): continue
    p0 = px(t); s = settle(int(c))
    if p0 != p0 or s != s: continue
    bl = [dict(lo=a, hi=bb, yes_bid=x, yes_ask=y) for a, bb, x, y in zip(g.floor_strike, g.cap_strike, g.yes_bid.fillna(0), g.yes_ask)]
    qv = pb.ladder_quantiles(bl, QS)
    if qv is None: continue
    r = dict(src='kalshi', kind='1hour', m=m, per='Dec-Aug' if t < SEP else 'Sep-Oct')
    for a, bb, n in LV:
        ok = qv[a] is not None and qv[bb] is not None
        r[f'in{n}'] = (qv[a] <= s <= qv[bb]) if ok else np.nan; r[f'w{n}'] = (qv[bb] - qv[a]) / p0 * 1e4 if ok else np.nan
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(sys.argv[4], index=False)
for n in (50, 80, 95): df[f'in{n}'] = df[f'in{n}'].astype(float)
out = df.groupby(['kind', 'src', 'per', 'm'])[['in50', 'in80', 'in95', 'w80']].agg(['mean']).round(3)
out['n'] = df.groupby(['kind', 'src', 'per', 'm']).size(); print(out.to_string())

cal = {}
for (kind, src, per, m), g in df.groupby(['kind', 'src', 'per', 'm']):
    cal.setdefault(f'{kind}|{src}', {}).setdefault(per, {})[int(m)] = [round(float(g[f'in{n}'].mean(skipna=True)) * 100, 1) for n in (50, 80, 95)] + [int(g['in80'].notna().sum())]
json.dump(cal, open(Path(sys.argv[4]).with_suffix('.json'), 'w'), indent=0)
