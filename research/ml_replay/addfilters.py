# Add-on conditions for the pooled model-gap rules (Direct ML, Correction ML): gap>=3 pts over ask+fee, favourite 70-99c, 6-14 min left,
# first qualifying minute per contract. Conditions split into thirds with cutoffs fit on Mar-Aug; also ET-hour blocks and weekday.
import numpy as np, pandas as pd
cols = ['ticker','month','min_left','ask','bid','ask_age','bid_age','res','p_direct','p_corr','z','ret_60','ret_300','ret_900','sig15','sig60','vol_ratio','vol_ratio_short',
        'dist_hi','dist_lo','spread','log_cumvol','hour_utc','dow','snap_ts']
A = pd.read_parquet('work/models_scored.parquet', columns=cols); A = A[A.month >= '2026-03'].dropna(subset=['ask','bid'])
fee = lambda p: .07*p*(1-p)
def trades(m):
    B = A.dropna(subset=[m]); out = []
    for side, px, win, prob, sgn in [('YES', B.ask, B.res, B[m], 1), ('NO', 1-B.bid, 1-B.res, 1-B[m], -1)]:
        t = B.assign(side=side, px=px, win=win, prob=prob, sgn=sgn); out.append(t)
    T = pd.concat(out); T['fee'] = fee(T.px); T['gap'] = 100*(T.prob-T.px-T.fee); T['pnl'] = T.win-T.px-T.fee
    T = T[(T.px >= .7) & (T.px < 1) & (T.min_left >= 6) & (T.gap >= 3)]
    T = T.sort_values('min_left', ascending=False).drop_duplicates('ticker')
    T['per'] = np.where(T.month >= '2026-09', 'Sep-Oct', 'Mar-Aug')
    # conditions, signed toward the side bought where direction matters
    T['Cushion (z to strike, own side)'] = T.sgn*T.z
    T['Momentum 1 min (own side)'] = T.sgn*T.ret_60; T['Momentum 5 min (own side)'] = T.sgn*T.ret_300; T['Trend 15 min (own side)'] = T.sgn*T.ret_900
    T['Vol 15 min (level)'] = T.sig15; T['Vol 60 min (level)'] = T.sig60; T['Vol 15/60 ratio'] = T.vol_ratio; T['Vol 1/15 ratio (spike)'] = T.vol_ratio_short
    T['Room to recent extreme (own side)'] = np.where(T.sgn > 0, -T.dist_hi, T.dist_lo)
    T['Spread'] = T.spread; T['Volume so far (log)'] = T.log_cumvol; T['Model gap size'] = T.gap; T['Price paid'] = T.px; T['Minutes left'] = T.min_left
    et = (T.hour_utc - 4) % 24
    T['ET time block'] = pd.cut(et, [-1, 5, 11, 17, 23], labels=['00-06 ET', '06-12 ET', '12-18 ET', '18-24 ET']).astype(str)
    T['Weekday'] = np.where(T.dow >= 5, 'Weekend', 'Mon-Fri'); T['Side bought'] = T.side
    return T
CONT = ['Cushion (z to strike, own side)','Momentum 1 min (own side)','Momentum 5 min (own side)','Trend 15 min (own side)','Vol 15 min (level)','Vol 60 min (level)',
        'Vol 15/60 ratio','Vol 1/15 ratio (spike)','Room to recent extreme (own side)','Spread','Volume so far (log)','Model gap size','Price paid','Minutes left']
CAT = ['ET time block', 'Weekday', 'Side bought']
def s(g):
    mo = g.groupby('month').pnl.mean()
    return dict(trades=len(g), win_pct=round(100*g.win.mean(),1), avg_price=round(100*g.px.mean(),1), breakeven=round(100*(g.px+g.fee).mean(),1),
                after_fee_c=round(100*g.pnl.mean(),2), t=round(g.pnl.mean()/(g.pnl.std()/np.sqrt(len(g))),2) if len(g) > 2 else np.nan, months_pos=f'{(mo>0).sum()}/{len(mo)}')
rows = []
for m, mname in [('p_direct', 'Direct ML'), ('p_corr', 'Correction ML')]:
    T = trades(m)
    for per, g in T.groupby('per'): rows.append(dict(rule=mname, condition='(none - base rule)', level='all', per=per, **s(g)))
    tr = T[T.per == 'Mar-Aug']
    for c in CONT:
        q = tr[c].quantile([1/3, 2/3]).values
        if q[0] == q[1]: q = [q[0] - 1e-12, q[1]]
        lab = pd.cut(T[c], [-np.inf, q[0], q[1], np.inf], labels=[f'low (< {q[0]:.3g})', f'mid', f'high (> {q[1]:.3g})']).astype(str)
        for (lv, per), g in T.assign(lv=lab).groupby(['lv', 'per']): rows.append(dict(rule=mname, condition=c, level=lv, per=per, **s(g)))
    for c in CAT:
        for (lv, per), g in T.groupby([c, 'per']): rows.append(dict(rule=mname, condition=c, level=lv, per=per, **s(g)))
    T.to_parquet(f'work/addf_{m}.parquet')
R = pd.DataFrame(rows)
W = R.pivot_table(index=['rule','condition','level'], columns='per', values=['trades','win_pct','avg_price','breakeven','after_fee_c','t','months_pos'], aggfunc='first')
W.columns = [f'{b} {a}' for a, b in W.columns]
order = [f'{p} {c}' for p in ['Mar-Aug','Sep-Oct'] for c in ['trades','win_pct','avg_price','breakeven','after_fee_c','t','months_pos']]
W = W[order].reset_index()
base = W[W.condition == '(none - base rule)'].set_index('rule')
W['beats base both'] = W.apply(lambda r: (r['Mar-Aug after_fee_c'] > base.loc[r.rule, 'Mar-Aug after_fee_c'] + 0.5) and (r['Sep-Oct after_fee_c'] > base.loc[r.rule, 'Sep-Oct after_fee_c'] + 0.5), axis=1)
W.to_parquet('work/addfilters.parquet')
pd.set_option('display.width', 280); pd.set_option('display.max_rows', 300)
print(W[['rule','condition','level','Mar-Aug trades','Mar-Aug win_pct','Mar-Aug avg_price','Mar-Aug after_fee_c','Mar-Aug t','Mar-Aug months_pos','Sep-Oct trades','Sep-Oct win_pct','Sep-Oct avg_price','Sep-Oct after_fee_c','Sep-Oct t','Sep-Oct months_pos','beats base both']].to_string(index=False))
