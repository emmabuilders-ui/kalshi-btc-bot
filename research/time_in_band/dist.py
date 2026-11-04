import numpy as np, pandas as pd
c=pd.read_pickle('snaps_tib.pkl'); c=c[c.in_this_hour.notna()&c.lo.notna()&c.hi.notna()].copy()
fee=lambda p: np.ceil(np.round(0.07*20*p*(1-p)*100,6))/100/20
ctr=(c.lo+c.hi+0.01)/2; half=(c.hi+0.01-c.lo)/2
c['dist_pct']=(c.brti-ctr)/half*100; c['absd']=c.dist_pct.abs()
c['step']=(np.floor(c.absd/10)*10).clip(upper=300).astype(int)
out=[]
for side in ('yes','no'):
    px=c.ask if side=='yes' else 1-c.bid; won=(c.y==1) if side=='yes' else (c.y==0)
    d=c.assign(px=px,won=won,pnl=won-px-fee(px))
    d=d[d.px.between(0.02,0.98)]
    for filt,m in (('all',slice(None)),('prev hour 75%+ in band',d.in_prev_hour>=0.75),('last 15 min 75%+ in band',d.in_last15>=0.75)):
        g=d[m].groupby(['step','per']).agg(band_minutes=('won','size'),bands=('ticker','nunique'),win_pct=('won','mean'),avg_price=('px','mean'),pnl=('pnl','mean')).reset_index()
        g['win_pct']*=100; g['avg_price']*=100; g['win_minus_price']=g.win_pct-g.avg_price; g['pnl']*=100
        g.insert(0,'filter',filt); g.insert(0,'side',side.upper()); out.append(g)
o=pd.concat(out).round(2).rename(columns={'step':'dist_from_center_pct (from)','pnl':'pnl_per_contract_c'})
w=o.pivot_table(index=['side','filter','dist_from_center_pct (from)'],columns='per',values=['band_minutes','win_pct','avg_price','win_minus_price','pnl_per_contract_c']).round(2)
w.columns=[f'{a} {b}' for a,b in w.columns]; w=w.reset_index()
with pd.ExcelWriter('/mnt/project-files/strategy/range_results_by_distance_from_center.xlsx') as x:
    w.to_excel(x,sheet_name='wide',index=False); o.to_excel(x,sheet_name='long',index=False)
    pd.DataFrame({'note':['dist from center % = |BRTI - band center| / (half the band width) x 100, at each minute mark, 30 to 1 min left. 0% = center, 100% = band edge, 200% = one band width outside.',
      'Rows are 10% steps (e.g. 40 = 40-49.9%); 300 = 300% and beyond. Band-minutes: each band counts once per minute. Prices from 1-minute Kalshi quotes, 2-98c only, P/L after taker fee (20-contract order).']}).to_excel(x,sheet_name='read_me',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
cols=['dist_from_center_pct (from)','band_minutes Dec-Aug','band_minutes Sep-Oct','win_pct Dec-Aug','avg_price Dec-Aug','pnl_per_contract_c Dec-Aug','win_pct Sep-Oct','avg_price Sep-Oct','pnl_per_contract_c Sep-Oct']
for s in ('YES','NO'):
  for f in ('all','prev hour 75%+ in band'):
    print(s,f); print(w[(w.side==s)&(w['filter']==f)][cols].head(22).to_string(index=False))
