# Support/resistance = high/low of the 60 (and 240) minutes BEFORE the contract opened. At trade time: is BTC above, below or between those levels?
import numpy as np, pandas as pd
T0 = 1764460800
b = pd.Series(np.load('work/brti.npy').astype(float)).ffill().to_numpy()
nmin = len(b)//60; M = b[:nmin*60].reshape(nmin, 60)
H1 = pd.Series(M.max(1)); L1 = pd.Series(M.min(1))
lev = {w: (H1.rolling(w).max().to_numpy(), L1.rolling(w).min().to_numpy()) for w in (60, 240)}
X = pd.read_parquet('work/models_scored.parquet', columns=['ticker', 'snap_ts', 'open_ts', 'brti', 'strike']).drop_duplicates(['ticker', 'snap_ts'])
def s_(g):
    mo = g.groupby('month').pnl.mean()
    return pd.Series(dict(trades=len(g), win=f'{100*g.win.mean():.1f} vs {100*g.px.mean():.1f}', breakeven=round(100*(g.px+g.fee).mean(), 1),
        after_fee_c=round(100*g.pnl.mean(), 2), t=round(g.pnl.mean()/(g.pnl.std()/np.sqrt(len(g))), 2) if len(g) > 2 else np.nan, months=f'{(mo>0).sum()}/{len(mo)}'))
out = []
for m, mname in [('p_direct', 'Direct ML'), ('p_corr', 'Correction ML')]:
    T = pd.read_parquet(f'work/addf_{m}.parquet').merge(X, on=['ticker', 'snap_ts'], how='left')
    cz = T[T.per == 'Mar-Aug']['Cushion (z to strike, own side)'].quantile(2/3)
    T['cushion'] = np.where(T['Cushion (z to strike, own side)'] > cz, 'cushion top third', 'other')
    idx = ((T.open_ts - T0)//60 - 1).astype(int)   # last full minute before the contract opened
    for w in (60, 240):
        hi, lo = lev[w][0][idx], lev[w][1][idx]
        pos = np.where(T.brti > hi, 'BTC above resistance', np.where(T.brti < lo, 'BTC below support', 'BTC between levels'))
        # same thing seen from the side bought: with you = broke out in the direction your side needs
        own = np.where(pos == 'BTC between levels', 'between', np.where((pos == 'BTC above resistance') == (T.side == 'YES'), 'broke out WITH my side', 'broke out AGAINST my side'))
        strike_pos = np.where(T.strike > hi, 'strike above resistance', np.where(T.strike < lo, 'strike below support', 'strike between'))
        for lab, col in [('BTC position', pos), ('BTC position vs side bought', own), ('Strike position', strike_pos)]:
            for (sub, side, lv, per), g in T.assign(lv=col).groupby(['cushion', 'side', 'lv', 'per']):
                out.append(dict(rule=mname, window=f'{w} min before open', view=lab, cushion=sub, side=side, level=lv, per=per, **s_(g)))
            for (lv, per), g in T.assign(lv=col).groupby(['lv', 'per']):
                out.append(dict(rule=mname, window=f'{w} min before open', view=lab, cushion='all', side='both', level=lv, per=per, **s_(g)))
R = pd.DataFrame(out)
W = R.pivot_table(index=['rule', 'window', 'view', 'cushion', 'side', 'level'], columns='per', values=['trades', 'win', 'breakeven', 'after_fee_c', 't', 'months'], aggfunc='first')
W.columns = [f'{b} {a}' for a, b in W.columns]
W = W[[f'{p} {c}' for p in ['Mar-Aug', 'Sep-Oct'] for c in ['trades', 'win', 'breakeven', 'after_fee_c', 't', 'months']]].reset_index()
W.to_excel('work/model_gap_rules_support_resistance.xlsx', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_rows', 300)
print(W[(W.cushion == 'all')].to_string(index=False))
print(W[(W.view == 'BTC position vs side bought') & (W.cushion != 'all')].to_string(index=False))
