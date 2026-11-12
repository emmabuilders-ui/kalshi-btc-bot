"""Stage 2 (hourly bands): swap the range model's vol input; rebuild move tables on Dec-Aug; Brier vs Kalshi mid."""
import numpy as np, pandas as pd, json, sys, warnings
warnings.filterwarnings('ignore')
W, OUT = sys.argv[1], sys.argv[2]
HR = '/mnt/project-files/data-download/kalshi-hourly-range/'
T0 = 1764460800; B = pd.Series(np.load(W + '/brti.npy')).ffill().to_numpy()
V = pd.read_parquet(W + '/vol_minutes.parquet', columns=['t', 's100', 's10', 'ewma', 'atr14'] + [f'fwd{h}' for h in range(1, 31)])
HZ = pd.read_parquet(W + '/fc_hz_frozen.parquet'); HZX = pd.read_parquet(W + '/fc_hz_expanding.parquet')
ATRC = {int(k): v for k, v in json.load(open(W + '/atr_scale.json')).items()}
bands = pd.read_parquet(HR + 'bands_dec2025_oct2026.parquet')
Q = pd.read_parquet(HR + 'quotes_1min_last60_dec2025_oct2026.parquet')
Q = Q.merge(bands[['ticker', 'close', 'floor_strike', 'cap_strike', 'result', 'expiration_value']], on='ticker')
Q['L'] = ((Q.close - Q.t) // 60).astype(int)
Q = Q[(Q.L >= 1) & (Q.L <= 30) & Q.yes_ask.notna() & Q.yes_bid.notna()]
Q = Q[(Q.yes_ask - Q.yes_bid <= 0.0501) & (Q.yes_ask >= Q.yes_bid)].copy()
idx = (Q.t.to_numpy() - T0).astype(np.int64); ok = (idx >= 0) & (idx < len(B)); Q = Q[ok]; Q['price'] = B[idx[ok]]
Q = Q.merge(V, on='t', how='inner').merge(HZ, on='t', how='inner').merge(HZX, on='t', how='inner', suffixes=('', '_x'))
Q = Q.dropna(subset=['s100', 's10', 'ewma', 'atr14'])
Q['y'] = (Q.result.str.lower() == 'yes').astype(float)
Q['mid'] = (Q.yes_ask + Q.yes_bid) / 2
Q['month'] = pd.to_datetime(Q.t, unit='s').dt.strftime('%Y-%m')
Q['period'] = np.where(Q.month < '2026-09', 'Dec-Aug', 'Sep-Oct')
SHR = 40 / 60
Q['te'] = Q.L - SHR
Q['vb'] = np.digitize(np.log(Q.s10 / Q.s100), [-0.3, 0.3])
Q['tb'] = np.where(Q.L > 20, 30, np.where(Q.L > 8, 15, 5))
# sigma for the remaining time, per input
L = Q.L.to_numpy(); te = Q.te.to_numpy()
hz = np.stack([Q[f'h{h}'].to_numpy() for h in range(1, 31)], 1)[np.arange(len(Q)), L - 1]
hzx = np.stack([Q[f'h{h}_x'].to_numpy() for h in range(1, 31)], 1)[np.arange(len(Q)), L - 1]
atrc = np.array([ATRC[h] for h in L])
SIG = {
    's100 (live input)': Q.s100.to_numpy() * np.sqrt(te),
    'EWMA 0.97': np.sqrt(Q.ewma.to_numpy() * te),
    'ATR14 x scale': np.sqrt(atrc * Q.atr14.to_numpy() ** 2 * te / L),
    'HAR3 frozen': np.sqrt(hz * te / L),
    'HAR3 expanding': np.sqrt(hzx * te / L),
    'Oracle: actual future RV (not tradable)': np.sqrt(np.stack([Q[f'fwd{h}'].to_numpy() for h in range(1, 31)], 1)[np.arange(len(Q)), L - 1] * te / L),
}
# standardized settlement move per (hour, minute): one row per close x t
settle = pd.to_numeric(Q.expiration_value, errors='coerce').to_numpy(); price = Q.price.to_numpy()
zmove = np.log(settle / price)
first = ~Q.duplicated(['close', 't']).to_numpy()
fitrows = first & (Q.period.to_numpy() == 'Dec-Aug')
QS = np.linspace(0, 1, 2001)
def build_table(sig):
    z = zmove / sig; T = {}
    for tb in (30, 15, 5):
        for vb in (0, 1, 2):
            i = fitrows & (Q.tb.to_numpy() == tb) & (Q.vb.to_numpy() == vb) & np.isfinite(z)
            T[(tb, vb)] = np.quantile(z[i], QS)
    return T
lo = Q.floor_strike.to_numpy(); hi = Q.cap_strike.to_numpy() + 0.01
def p_inside(sig, T=None):
    xl = np.log(lo / price) / sig; xh = np.log(hi / price) / sig
    if T is None:
        from scipy.stats import norm
        return norm.cdf(xh) - norm.cdf(xl)
    p = np.full(len(Q), np.nan)
    for key, c in T.items():
        i = (Q.tb.to_numpy() == key[0]) & (Q.vb.to_numpy() == key[1])
        p[i] = np.interp(xh[i], c, QS, left=0, right=1) - np.interp(xl[i], c, QS, left=0, right=1)
    return p
P = {'Kalshi mid': Q.mid.to_numpy()}
for k, s in SIG.items():
    ok = np.isfinite(s) & (s > 0)
    T = build_table(np.where(ok, s, np.nan)); P[k] = p_inside(np.where(ok, s, 1.0), T); P[k][~ok] = np.nan
P['HAR3 frozen, normal CDF'] = p_inside(np.where(np.isfinite(SIG['HAR3 frozen']), SIG['HAR3 frozen'], 1.0))
y = Q.y.to_numpy()
rows = []
mon = Q.month.to_numpy(); per = Q.period.to_numpy()
common = np.all([np.isfinite(P[k]) for k in P if k not in ('HAR3 expanding', 'Oracle: actual future RV (not tradable)')], 0)
for k, p in P.items():
    for grp, i in [('Dec-Aug', per == 'Dec-Aug'), ('Sep-Oct', per == 'Sep-Oct')] + [(m, mon == m) for m in sorted(set(mon))]:
        j = i & common & np.isfinite(p)
        rows.append(dict(model=k, group=grp, n=int(j.sum()), brier=((p[j] - y[j]) ** 2).mean()))
    for g, lab in ((L <= 5, '1-5 min'), ((L > 5) & (L <= 15), '6-15 min'), (L > 15, '16-30 min')):
        for pr in ('Dec-Aug', 'Sep-Oct'):
            j = g & (per == pr) & common & np.isfinite(p)
            rows.append(dict(model=k, group=f'{lab} {pr}', n=int(j.sum()), brier=((p[j] - y[j]) ** 2).mean()))
R = pd.DataFrame(rows); R.to_parquet(OUT + '/stage2_hourly_brier.parquet')
pd.set_option('display.width', 250)
PV = R.pivot(index='model', columns='group', values='brier')
print('Brier (lower is better), hourly bands, 1-30 min left, spread<=5c, n Dec-Aug/Sep-Oct =', R[(R.model == 'Kalshi mid') & R.group.isin(['Dec-Aug', 'Sep-Oct'])].n.tolist())
print(PV[['Dec-Aug', 'Sep-Oct']].round(5).sort_values('Sep-Oct'))
print(PV[[c for c in PV.columns if 'min' in c]].round(5))
MP = PV[[c for c in PV.columns if c.startswith('20')]]
print('by month:'); print(MP.round(5).T)
print('months better than Kalshi mid:'); print((MP.lt(MP.loc['Kalshi mid'], axis=1)).sum(1))
print('months better than s100 table:'); print((MP.lt(MP.loc['s100 (live input)'], axis=1)).sum(1))
# ---- trades: YES at ask 75-99c, model >= ask + margin, first qualifying minute per band
ask = Q.yes_ask.to_numpy(); fee = lambda p: 0.07 * p * (1 - p)
tr = []
for k in [m for m in P if m != 'Kalshi mid']:
    for marg in (0.02, 0.05):
        for lo_c, hi_c in ((0.745, 0.895), (0.895, 0.995), (0.745, 0.995)):
            sel = (P[k] >= ask + marg) & (ask >= lo_c) & (ask < hi_c) & common & np.isfinite(P[k])
            d = Q[sel].sort_values('t').drop_duplicates('ticker')
            for pr in ('Dec-Aug', 'Sep-Oct'):
                e = d[d.period == pr]
                if len(e) == 0: tr.append(dict(model=k, margin=marg, price=f'{int(round(lo_c*100+0.5))}-{int(round(hi_c*100-0.5))}c', period=pr, n=0)); continue
                w = e.y.mean(); a = e.yes_ask.mean(); f = fee(e.yes_ask).mean()
                mths = e.groupby('month').apply(lambda g: (g.y - g.yes_ask - fee(g.yes_ask)).mean() > 0)
                tr.append(dict(model=k, margin=marg, price=f'{int(round(lo_c*100+0.5))}-{int(round(hi_c*100-0.5))}c', period=pr, n=len(e),
                               avg_price=round(a * 100, 1), win=round(w * 100, 1), edge=round((w - a - f) * 100, 2), per_day=round(len(e) / e.t.sub(T0).floordiv(86400).nunique(), 2),
                               months_pos=f'{int(mths.sum())}/{len(mths)}'))
TR = pd.DataFrame(tr); TR.to_parquet(OUT + '/stage2_hourly_trades.parquet')
print(TR[TR.margin == 0.02].to_string()); print(TR[TR.margin == 0.05].to_string())
Q[['ticker', 't', 'L', 'month', 'period', 'yes_ask', 'yes_bid', 'y']].assign(**{k: P[k] for k in P}).to_parquet(OUT + '/stage2_hourly_probs.parquet')
