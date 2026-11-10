# Technical add-ons (support/resistance, RSI, MACD, squeeze, OBV) on the pooled model-gap favourite rules.
# Indicators use only bars completed before the snapshot. BRTI 1-min bars from the 1-s index; OBV from Binance BTCUSDT 1-min volume.
import numpy as np, pandas as pd, zipfile, glob
T0 = 1764460800
b = np.load('work/brti.npy').astype(float)
b = pd.Series(b).ffill().to_numpy()
nmin = len(b) // 60
M = b[:nmin*60].reshape(nmin, 60)
C1 = pd.Series(M[:, -1]); H1 = pd.Series(M.max(1)); L1 = pd.Series(M.min(1))   # minute i covers [T0+60i, T0+60i+60)
def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn.replace(0, np.nan))
def macd_hist(c):
    m = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean(); return (m - m.ewm(span=9, adjust=False).mean()) / c
def squeeze(c, h, l, n=20):
    sd = c.rolling(n).std(); tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(1); atr = tr.rolling(n).mean()
    return (2*sd) / (1.5*atr)   # < 1 = Bollinger inside Keltner (squeeze on)
I1 = pd.DataFrame({'rsi1': rsi(C1), 'macd1': macd_hist(C1), 'sq1': squeeze(C1, H1, L1),
                   'hi60': H1.rolling(60).max(), 'lo60': L1.rolling(60).min(), 'hi240': H1.rolling(240).max(), 'lo240': L1.rolling(240).min()})
# 5-min bars
n5 = nmin // 5
C5 = pd.Series(C1[:n5*5].to_numpy().reshape(n5, 5)[:, -1]); H5 = pd.Series(H1[:n5*5].to_numpy().reshape(n5, 5).max(1)); L5 = pd.Series(L1[:n5*5].to_numpy().reshape(n5, 5).min(1))
I5 = pd.DataFrame({'rsi5': rsi(C5), 'macd5': macd_hist(C5), 'sq5': squeeze(C5, H5, L5)})
# Binance OBV (1-min)
K = []
for f in sorted(glob.glob('/tmp/claude-0/-home-user/964f9265-0f8f-5239-a8c8-6dac9f8eb350/scratchpad/binance/*.zip')):
    z = zipfile.ZipFile(f); K.append(pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 4, 5]))
K = pd.concat(K); K.columns = ['ot', 'close', 'vol']; K['ot'] = np.where(K.ot > 1e14, K.ot // 1_000_000, K.ot // 1000)
K = K.drop_duplicates('ot').set_index('ot').sort_index()
K['obv'] = (np.sign(K.close.diff()).fillna(0) * K.vol).cumsum(); vavg = K.vol.rolling(60).mean()
K['obv15'] = (K.obv - K.obv.shift(15)) / (vavg*15); K['obv60'] = (K.obv - K.obv.shift(60)) / (vavg*60)
K['px15'] = np.log(K.close / K.close.shift(15))
K['mi'] = (K.index - T0) // 60
K = K.set_index('mi')
def attach(T):
    last1 = ((T.snap_ts - T0) // 60 - 1).astype(int); last5 = ((T.snap_ts - T0) // 300 - 1).astype(int)
    T = T.copy()
    for c in I1: T[c] = I1[c].to_numpy()[last1]
    for c in I5: T[c] = I5[c].to_numpy()[last5]
    for c in ['obv15', 'obv60', 'px15']: T[c] = K[c].reindex(last1.to_numpy()).to_numpy()
    s = T.sgn
    T['RSI 1-min (own side, 50=neutral)'] = s*(T.rsi1 - 50); T['RSI 5-min (own side)'] = s*(T.rsi5 - 50)
    T['MACD hist 1-min (own side)'] = s*T.macd1; T['MACD hist 5-min (own side)'] = s*T.macd5
    T['Squeeze 1-min (BB/KC width)'] = T.sq1; T['Squeeze 5-min (BB/KC width)'] = T.sq5
    T['Squeeze on 1-min'] = np.where(T.sq1 < 1, 'squeeze on', 'off'); T['Squeeze on 5-min'] = np.where(T.sq5 < 1, 'squeeze on', 'off')
    # support/resistance: strike beyond the 60/240-min extreme on the side that must hold (own side positive = strike is past support/resistance)
    T['Strike past 60-min S/R (own side)'] = np.where(s > 0, np.log(T.lo60/T.strike), np.log(T.strike/T.hi60))
    T['Strike past 240-min S/R (own side)'] = np.where(s > 0, np.log(T.lo240/T.strike), np.log(T.strike/T.hi240))
    T['Price room to 60-min S/R (adverse)'] = np.where(s > 0, np.log(T.brti/T.lo60), np.log(T.hi60/T.brti))
    T['Strike outside 60-min range'] = np.where(T['Strike past 60-min S/R (own side)'] > 0, 'strike beyond S/R', 'strike inside range')
    T['OBV 15-min slope (own side)'] = s*T.obv15; T['OBV 60-min slope (own side)'] = s*T.obv60
    T['OBV vs price 15-min'] = np.where(np.sign(s*T.obv15) == np.sign(s*T.px15), 'OBV confirms', 'OBV diverges')
    return T
CONT = ['RSI 1-min (own side, 50=neutral)', 'RSI 5-min (own side)', 'MACD hist 1-min (own side)', 'MACD hist 5-min (own side)', 'Squeeze 1-min (BB/KC width)',
        'Squeeze 5-min (BB/KC width)', 'Strike past 60-min S/R (own side)', 'Strike past 240-min S/R (own side)', 'Price room to 60-min S/R (adverse)',
        'OBV 15-min slope (own side)', 'OBV 60-min slope (own side)']
CAT = ['Squeeze on 1-min', 'Squeeze on 5-min', 'Strike outside 60-min range', 'OBV vs price 15-min']
def s_(g):
    mo = g.groupby('month').pnl.mean()
    return dict(trades=len(g), win_pct=round(100*g.win.mean(), 1), avg_price=round(100*g.px.mean(), 1), breakeven=round(100*(g.px+g.fee).mean(), 1),
                after_fee_c=round(100*g.pnl.mean(), 2), t=round(g.pnl.mean()/(g.pnl.std()/np.sqrt(len(g))), 2) if len(g) > 2 else np.nan, months_pos=f'{(mo>0).sum()}/{len(mo)}')
rows = []
X = pd.read_parquet('work/models_scored.parquet', columns=['ticker', 'k', 'snap_ts', 'brti', 'strike']).drop_duplicates(['ticker', 'snap_ts'])
for m, mname in [('p_direct', 'Direct ML'), ('p_corr', 'Correction ML')]:
    T = pd.read_parquet(f'work/addf_{m}.parquet').merge(X[['ticker', 'snap_ts', 'brti', 'strike']], on=['ticker', 'snap_ts'], how='left')
    T = attach(T); T.to_parquet(f'work/ta_{m}.parquet')
    cz = T[T.per == 'Mar-Aug']['Cushion (z to strike, own side)'].quantile(2/3)
    for subset, S in [('base rule', T), ('base + cushion top third', T[T['Cushion (z to strike, own side)'] > cz])]:
        for per, g in S.groupby('per'): rows.append(dict(rule=mname, subset=subset, condition='(none)', level='all', per=per, **s_(g)))
        tr = S[S.per == 'Mar-Aug']
        for c in CONT:
            q = tr[c].quantile([1/3, 2/3]).values
            lab = pd.cut(S[c], [-np.inf, q[0], q[1], np.inf], labels=[f'low (< {q[0]:.3g})', 'mid', f'high (> {q[1]:.3g})']).astype(str)
            for (lv, per), g in S.assign(lv=lab).groupby(['lv', 'per']):
                if lv != 'nan': rows.append(dict(rule=mname, subset=subset, condition=c, level=lv, per=per, **s_(g)))
        for c in CAT:
            for (lv, per), g in S.groupby([c, 'per']): rows.append(dict(rule=mname, subset=subset, condition=c, level=lv, per=per, **s_(g)))
R = pd.DataFrame(rows)
W = R.pivot_table(index=['rule', 'subset', 'condition', 'level'], columns='per', values=['trades', 'win_pct', 'avg_price', 'breakeven', 'after_fee_c', 't', 'months_pos'], aggfunc='first')
W.columns = [f'{b} {a}' for a, b in W.columns]
W = W[[f'{p} {c}' for p in ['Mar-Aug', 'Sep-Oct'] for c in ['trades', 'win_pct', 'avg_price', 'breakeven', 'after_fee_c', 't', 'months_pos']]].reset_index()
base = W[W.condition == '(none)'].set_index(['rule', 'subset'])
W['beats base both (+0.5c)'] = W.apply(lambda r: (r['Mar-Aug after_fee_c'] > base.loc[(r.rule, r.subset), 'Mar-Aug after_fee_c'] + .5) and (r['Sep-Oct after_fee_c'] > base.loc[(r.rule, r.subset), 'Sep-Oct after_fee_c'] + .5), axis=1)
W.to_parquet('work/ta_filters.parquet'); W.to_excel('work/model_gap_rules_ta_filters.xlsx', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 400)
show = ['rule', 'subset', 'condition', 'level', 'Mar-Aug trades', 'Mar-Aug win_pct', 'Mar-Aug avg_price', 'Mar-Aug after_fee_c', 'Mar-Aug t', 'Mar-Aug months_pos', 'Sep-Oct trades', 'Sep-Oct win_pct', 'Sep-Oct avg_price', 'Sep-Oct after_fee_c', 'Sep-Oct t', 'Sep-Oct months_pos', 'beats base both (+0.5c)']
print(W[show].to_string(index=False))
print('flags', W['beats base both (+0.5c)'].sum(), 'of', (W.condition != '(none)').sum())
