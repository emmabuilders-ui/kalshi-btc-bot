"""Stage 2 (15-min contracts): vol input swap, tables rebuilt on Dec-Aug per minute left x vol state, Brier vs Kalshi mid."""
import numpy as np, pandas as pd, json, sys, warnings
warnings.filterwarnings('ignore')
W, OUT = sys.argv[1], sys.argv[2]
T0 = 1764460800; Braw = np.load(W + '/brti.npy'); B = pd.Series(Braw).ffill().to_numpy()
M = pd.read_parquet(W + '/markets_audited.parquet')
# fill missing strikes with the BRTI 60-s average before the open (= previous settlement)
idx = (M.open_ts.to_numpy() - T0)[:, None] + np.arange(-60, 0)[None, :]
ok = (idx >= 0).all(1) & (idx < len(Braw)).all(1)
fill = np.full(len(M), np.nan); fill[ok] = np.nanmean(Braw[idx[ok]], 1)
M['strike'] = M.strike.where(M.strike > 1000, fill)
M = M[(M.n_prints == 60) & M.strike.notna() & (M.strike > 1000) & M.res.notna()]
S = pd.read_parquet(W + '/snaps_raw.parquet').merge(M[['ticker', 'open_ts', 'close_ts', 'strike', 'res', 'brti_avg', 'month']], on='ticker')
S = S[(S.ask_age < 20) & (S.bid_age < 20) & S.ask.notna() & S.bid.notna() & (S.ask >= S.bid) & (S.ask - S.bid <= 0.0501)].copy()
e = (S.snap_ts.to_numpy() - T0).astype(np.int64); S['price'] = B[e]
S['t'] = S.snap_ts  # snapshot at open + 60k = end of a BRTI minute
V = pd.read_parquet(W + '/vol_minutes.parquet', columns=['t', 's100', 's10', 'ewma', 'atr14'] + [f'fwd{h}' for h in range(1, 15)])
HZ = pd.read_parquet(W + '/fc_hz_frozen.parquet', columns=['t'] + [f'h{h}' for h in range(1, 15)])
HZX = pd.read_parquet(W + '/fc_hz_expanding.parquet', columns=['t'] + [f'h{h}' for h in range(1, 15)])
ATRC = {int(k): v for k, v in json.load(open(W + '/atr_scale.json')).items()}
S = S.merge(V, on='t').merge(HZ, on='t').merge(HZX, on='t', suffixes=('', '_x')).dropna(subset=['s100', 's10', 'ewma', 'atr14'])
S['L'] = 15 - S.k; S['mid'] = (S.ask + S.bid) / 2
S['period'] = np.where(S.month < '2026-09', 'Dec-Aug', 'Sep-Oct')
L = S.L.to_numpy(); te = L - 40 / 60
S['vb'] = np.digitize(np.log(S.s10 / S.s100), [-0.3, 0.3])
hz = np.stack([S[f'h{h}'].to_numpy() for h in range(1, 15)], 1)[np.arange(len(S)), L - 1]
hzx = np.stack([S[f'h{h}_x'].to_numpy() for h in range(1, 15)], 1)[np.arange(len(S)), L - 1]
atrc = np.array([ATRC[h] for h in L])
SIG = {'s100 (live input)': S.s100.to_numpy() * np.sqrt(te), 'EWMA 0.97': np.sqrt(S.ewma.to_numpy() * te),
       'ATR14 x scale': np.sqrt(atrc * S.atr14.to_numpy() ** 2 * te / L), 'HAR3 frozen': np.sqrt(hz * te / L), 'HAR3 expanding': np.sqrt(hzx * te / L),
       'Oracle: actual future RV (not tradable)': np.sqrt(np.stack([S[f'fwd{h}'].to_numpy() for h in range(1, 15)], 1)[np.arange(len(S)), L - 1] * te / L)}
price = S.price.to_numpy(); zmove = np.log(S.brti_avg.to_numpy() / price); xs = np.log(S.strike.to_numpy() / price)
fitrows = S.period.to_numpy() == 'Dec-Aug'; QS = np.linspace(0, 1, 2001)
y = S.res.to_numpy().astype(float); mon = S.month.to_numpy(); per = S.period.to_numpy()
def p_yes(sig, table=True):
    z = zmove / sig; x = xs / sig; p = np.full(len(S), np.nan)
    if not table:
        from scipy.stats import norm; return 1 - norm.cdf(x)
    for l in range(1, 15):
        for vb in (0, 1, 2):
            i = (L == l) & (S.vb.to_numpy() == vb); f = i & fitrows & np.isfinite(z)
            c = np.quantile(z[f], QS); p[i] = 1 - np.interp(x[i], c, QS, left=0, right=1)
    return p
P = {'Kalshi mid': S.mid.to_numpy()}
for k, s in SIG.items():
    ok = np.isfinite(s) & (s > 0); p = p_yes(np.where(ok, s, 1.0)); p[~ok] = np.nan; P[k] = p
P['HAR3 frozen, normal CDF'] = p_yes(np.where(np.isfinite(SIG['HAR3 frozen']), SIG['HAR3 frozen'], 1.0), table=False)
common = np.all([np.isfinite(P[k]) for k in P if k not in ('HAR3 expanding', 'Oracle: actual future RV (not tradable)')], 0)
rows = []
for k, p in P.items():
    for grp, i in [('Dec-Aug', per == 'Dec-Aug'), ('Sep-Oct', per == 'Sep-Oct')] + [(m, mon == m) for m in sorted(set(mon))] + \
                  [(f'{lab} {pr}', g & (per == pr)) for g, lab in ((L <= 3, '1-3 min'), ((L > 3) & (L <= 8), '4-8 min'), (L > 8, '9-14 min')) for pr in ('Dec-Aug', 'Sep-Oct')]:
        j = i & common & np.isfinite(p); rows.append(dict(model=k, group=grp, n=int(j.sum()), brier=((p[j] - y[j]) ** 2).mean()))
R = pd.DataFrame(rows); R.to_parquet(OUT + '/stage2_15m_brier.parquet')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
PV = R.pivot(index='model', columns='group', values='brier'); N = R.pivot(index='model', columns='group', values='n')
print('n:', N.loc['Kalshi mid'][['Dec-Aug', 'Sep-Oct']].tolist(), 'contracts', S.ticker.nunique())
print(PV[['Dec-Aug', 'Sep-Oct']].round(5).sort_values('Sep-Oct'))
print(PV[[c for c in PV.columns if 'min' in c]].round(5))
MP = PV[[c for c in PV.columns if c.startswith('20')]]; print(MP.round(5).T)
print('months better than Kalshi mid:'); print((MP.lt(MP.loc['Kalshi mid'], axis=1)).sum(1))
print('months better than s100 table:'); print((MP.lt(MP.loc['s100 (live input)'], axis=1)).sum(1))
# trades: YES at ask / NO at (1-bid), 75-99c, model >= price + margin, first qualifying snapshot per contract
ask = S.ask.to_numpy(); bid = S.bid.to_numpy(); fee = lambda p: 0.07 * p * (1 - p)
tr = []
for k in [m for m in P if m != 'Kalshi mid']:
    for side in ('YES', 'NO'):
        pr_ = ask if side == 'YES' else 1 - bid; pm = P[k] if side == 'YES' else 1 - P[k]; win = y if side == 'YES' else 1 - y
        for marg in (0.02, 0.05):
            for lo_c, hi_c in ((0.745, 0.895), (0.895, 0.995)):
                sel = (pm >= pr_ + marg) & (pr_ >= lo_c) & (pr_ < hi_c) & common & np.isfinite(pm)
                d = S[sel].assign(pr=pr_[sel], w=win[sel]).sort_values('t').drop_duplicates('ticker')
                for pr in ('Dec-Aug', 'Sep-Oct'):
                    e_ = d[d.period == pr]
                    if len(e_) == 0: tr.append(dict(model=k, side=side, margin=marg, price=f'{int(round(lo_c*100+0.5))}-{int(round(hi_c*100-0.5))}c', period=pr, n=0)); continue
                    mths = e_.groupby('month').apply(lambda g: (g.w - g.pr - fee(g.pr)).mean() > 0)
                    tr.append(dict(model=k, side=side, margin=marg, price=f'{int(round(lo_c*100+0.5))}-{int(round(hi_c*100-0.5))}c', period=pr, n=len(e_),
                                   avg_price=round(e_.pr.mean() * 100, 1), win=round(e_.w.mean() * 100, 1), edge=round((e_.w - e_.pr - fee(e_.pr)).mean() * 100, 2),
                                   per_day=round(len(e_) / e_.t.sub(T0).floordiv(86400).nunique(), 2), months_pos=f'{int(mths.sum())}/{len(mths)}'))
TR = pd.DataFrame(tr); TR.to_parquet(OUT + '/stage2_15m_trades.parquet'); print(TR.to_string())
S[['ticker', 't', 'L', 'month', 'period', 'ask', 'bid', 'res']].assign(**{k: P[k] for k in P}).to_parquet(OUT + '/stage2_15m_probs.parquet')
