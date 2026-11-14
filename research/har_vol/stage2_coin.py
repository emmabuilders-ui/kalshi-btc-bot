"""Stage 2 for ETH/XRP hourly bands: vol-input swap, tables rebuilt on Dec-Aug, Brier + AUC vs Kalshi mid.
Price proxy = Binance 1-s close + basis, basis = Kalshi expiration_value - Binance 60-s avg at the most recent PRIOR close."""
import numpy as np, pandas as pd, json, sys, glob, warnings
from scipy.stats import rankdata
warnings.filterwarnings('ignore')
coin, ser, W, KX, OUT = sys.argv[1:6]; TICK = float(sys.argv[6])
T0 = 1764460800; Braw = np.load(W + '/brti.npy'); B = pd.Series(Braw).ffill().to_numpy()
V = pd.read_parquet(W + '/vol_minutes.parquet', columns=['t', 's100', 's10', 'ewma', 'atr14'] + [f'fwd{h}' for h in range(1, 31)])
HZ = pd.read_parquet(W + '/fc_hz_frozen.parquet'); ATRC = {int(k): v for k, v in json.load(open(W + '/atr_scale.json')).items()}
M = pd.read_parquet(f'{KX}/{ser}_markets.parquet'); M = M[(M.strike_type == 'between') & M.expiration_value.notna()]
sel = pd.read_parquet(f'{KX}/{ser}_selected.parquet')
Q = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{KX}/cand/{ser}/*.parquet'))])
Q = Q[Q.t > 0].merge(M[['ticker', 'close', 'floor_strike', 'cap_strike', 'result', 'expiration_value']], on='ticker')
# basis per close (settlement value vs Binance 60-s average), then use the latest prior close's basis
cl = np.sort(M.close.unique()); idx = (cl - T0)[:, None] + np.arange(-60, 0)[None, :]
ok = (idx >= 0).all(1) & (idx < len(Braw)).all(1)
bavg = np.full(len(cl), np.nan); bavg[ok] = np.nanmean(Braw[idx[ok]], 1)
ev = M.drop_duplicates('close').set_index('close').expiration_value.reindex(cl).to_numpy()
basis = pd.Series(ev - bavg, index=cl).dropna()
print('basis $: median', round(basis.median(), 4), 'sd', round(basis.std(), 4), 'as % of price', round(100 * basis.median() / np.nanmedian(ev), 3))
Q['L'] = ((Q.close - Q.t) // 60).astype(int)
Q = Q[(Q.L >= 1) & (Q.L <= 30) & Q.yes_ask.notna() & Q.yes_bid.notna() & (Q.yes_ask - Q.yes_bid <= 0.0501) & (Q.yes_ask >= Q.yes_bid)].copy()
e = (Q.t.to_numpy() - T0).astype(np.int64); ok = (e >= 0) & (e < len(B)); Q = Q[ok]; e = e[ok]
bi = np.searchsorted(basis.index.to_numpy(), Q.t.to_numpy() - 3600, side='right') - 1  # latest close at or before t-60min... prior hour's close
bi = np.searchsorted(basis.index.to_numpy(), Q.close.to_numpy() - 3600, side='right') - 1
Q = Q[bi >= 0]; e = e[bi >= 0]; bi = bi[bi >= 0]
Q['price'] = B[e] + basis.to_numpy()[bi]
Q = Q.merge(V, on='t').merge(HZ, on='t').dropna(subset=['s100', 's10', 'ewma', 'atr14'])
Q['y'] = (Q.result.str.lower() == 'yes').astype(float); Q['mid'] = (Q.yes_ask + Q.yes_bid) / 2
Q['month'] = pd.to_datetime(Q.t, unit='s').dt.strftime('%Y-%m'); Q['period'] = np.where(Q.month < '2026-09', 'Dec-Aug', 'Sep-Oct')
L = Q.L.to_numpy(); te = L - 40 / 60
Q['vb'] = np.digitize(np.log(Q.s10 / Q.s100), [-0.3, 0.3]); Q['tb'] = np.where(L > 20, 30, np.where(L > 8, 15, 5))
hz = np.stack([Q[f'h{h}'].to_numpy() for h in range(1, 31)], 1)[np.arange(len(Q)), L - 1]
atrc = np.array([ATRC[h] for h in L])
SIG = {'s100 (live input)': Q.s100.to_numpy() * np.sqrt(te), 'EWMA 0.97': np.sqrt(Q.ewma.to_numpy() * te),
       'ATR14 x scale': np.sqrt(atrc * Q.atr14.to_numpy() ** 2 * te / L), 'HAR3 frozen': np.sqrt(hz * te / L),
       'Oracle: actual future RV (not tradable)': np.sqrt(np.stack([Q[f'fwd{h}'].to_numpy() for h in range(1, 31)], 1)[np.arange(len(Q)), L - 1] * te / L)}
settle = Q.expiration_value.to_numpy(); price = Q.price.to_numpy(); zmove = np.log(settle / price)
first = ~Q.duplicated(['close', 't']).to_numpy(); fitrows = first & (Q.period.to_numpy() == 'Dec-Aug'); QS = np.linspace(0, 1, 2001)
lo = Q.floor_strike.to_numpy(); hi = Q.cap_strike.to_numpy() + TICK
tb = Q.tb.to_numpy(); vb = Q.vb.to_numpy()
def p_inside(sig):
    z = zmove / sig; xl = np.log(lo / price) / sig; xh = np.log(hi / price) / sig; p = np.full(len(Q), np.nan)
    for t_ in (30, 15, 5):
        for v_ in (0, 1, 2):
            i = (tb == t_) & (vb == v_); f = i & fitrows & np.isfinite(z)
            if f.sum() < 50: continue
            c = np.quantile(z[f], QS); p[i] = np.interp(xh[i], c, QS, left=0, right=1) - np.interp(xl[i], c, QS, left=0, right=1)
    return p
P = {'Kalshi mid': Q.mid.to_numpy()}
for k, s in SIG.items():
    okk = np.isfinite(s) & (s > 0); p = p_inside(np.where(okk, s, 1.0)); p[~okk] = np.nan; P[k] = p
y = Q.y.to_numpy(); mon = Q.month.to_numpy(); per = Q.period.to_numpy()
common = np.all([np.isfinite(P[k]) for k in P if not k.startswith('Oracle')], 0)
def auc(yy, pp):
    r = rankdata(pp); n1 = yy.sum(); n0 = len(yy) - n1; return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
rows = []
for k, p in P.items():
    groups = [('Dec-Aug', per == 'Dec-Aug'), ('Sep-Oct', per == 'Sep-Oct')] + [(m, mon == m) for m in sorted(set(mon))] + \
             [(f'{lab} {pr}', g & (per == pr)) for g, lab in ((L <= 5, '1-5 min'), ((L > 5) & (L <= 15), '6-15 min'), (L > 15, '16-30 min')) for pr in ('Dec-Aug', 'Sep-Oct')]
    for grp, i in groups:
        j = i & common & np.isfinite(p)
        if j.sum() < 100: continue
        rows.append(dict(model=k, group=grp, n=int(j.sum()), brier=((p[j] - y[j]) ** 2).mean(), auc=auc(y[j], p[j])))
R = pd.DataFrame(rows); R.to_parquet(f'{OUT}/stage2_{coin}_hourly.parquet')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
PV = R.pivot(index='model', columns='group', values='brier'); AU = R.pivot(index='model', columns='group', values='auc'); N = R.pivot(index='model', columns='group', values='n')
print(coin, 'band-minutes Dec-Aug/Sep-Oct', N.loc['Kalshi mid'][['Dec-Aug', 'Sep-Oct']].tolist(), 'bands', Q.ticker.nunique())
print('BRIER'); print(PV[['Dec-Aug', 'Sep-Oct']].round(5).sort_values('Sep-Oct')); print('AUC'); print(AU[['Dec-Aug', 'Sep-Oct']].round(4))
print(PV[[c for c in PV.columns if 'min' in c]].round(5))
MP = PV[[c for c in PV.columns if c.startswith('20')]]; print(MP.round(5).T)
print('months better than Kalshi mid:'); print((MP.lt(MP.loc['Kalshi mid'], axis=1)).sum(1))
print('months better than s100:'); print((MP.lt(MP.loc['s100 (live input)'], axis=1)).sum(1))
# trades YES 75-99c, model >= ask + 2c / 5c, first minute per band
ask = Q.yes_ask.to_numpy(); fee = lambda p: 0.07 * p * (1 - p); tr = []
for k in [m for m in P if m != 'Kalshi mid']:
    for marg in (0.02, 0.05):
        for lo_c, hi_c in ((0.745, 0.895), (0.895, 0.995)):
            s_ = (P[k] >= ask + marg) & (ask >= lo_c) & (ask < hi_c) & common & np.isfinite(P[k])
            d = Q[s_].sort_values('t').drop_duplicates('ticker')
            for pr in ('Dec-Aug', 'Sep-Oct'):
                e_ = d[d.period == pr]
                if len(e_) == 0: continue
                mths = e_.groupby('month').apply(lambda g: (g.y - g.yes_ask - fee(g.yes_ask)).mean() > 0)
                tr.append(dict(model=k, margin=marg, price=f'{int(round(lo_c*100+0.5))}-{int(round(hi_c*100-0.5))}c', period=pr, n=len(e_), avg_price=round(e_.yes_ask.mean() * 100, 1),
                               win=round(e_.y.mean() * 100, 1), edge=round((e_.y - e_.yes_ask - fee(e_.yes_ask)).mean() * 100, 2), months_pos=f'{int(mths.sum())}/{len(mths)}'))
TR = pd.DataFrame(tr); TR.to_parquet(f'{OUT}/stage2_{coin}_trades.parquet'); print(TR.to_string())
