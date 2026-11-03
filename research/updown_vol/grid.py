import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
c=pd.read_pickle('updown_scored.pkl')
c['pb']=(np.floor(c.mid*100/5)*5).clip(0,95).astype(int)
c['dm']=(c.p_sym-c.y)**2-(c.mid-c.y)**2      # model minus Kalshi per snapshot (<0 = model better)
c['dc']=(c.model_now-c.y)**2-(c.mid-c.y)**2
g=c.groupby(['pb','left','per']).agg(n=('y','size'),bands=('ticker','nunique'),win=('y','mean'),kalshi=('mid','mean'),model=('p_sym','mean'),d=('dm','mean'),dsd=('dm','std'),dcur=('dc','mean')).reset_index()
g['se']=g.dsd/np.sqrt(g.bands.clip(lower=1))     # conservative: per band, not per minute
w=g.pivot_table(index=['pb','left'],columns='per',values=['n','bands','win','kalshi','model','d','se','dcur']); w.columns=[f'{a}_{b}' for a,b in w.columns]; w=w.reset_index()
both=w[(w['d_Dec-Aug']<0)&(w['d_Sep-Oct']<0)&(w['bands_Sep-Oct']>=30)&(w['bands_Dec-Aug']>=100)].copy()
both['strength']=-(both['d_Sep-Oct']/both['se_Sep-Oct'])
print(len(w),'cells;',len(both),'better in both periods')
cols=['pb','left','bands_Dec-Aug','bands_Sep-Oct','win_Dec-Aug','kalshi_Dec-Aug','model_Dec-Aug','win_Sep-Oct','kalshi_Sep-Oct','model_Sep-Oct','d_Dec-Aug','d_Sep-Oct','strength']
pd.set_option('display.width',250)
print(both.sort_values('strength',ascending=False)[cols].head(30).round(4).to_string(index=False))
# region summaries
both.to_pickle('both.pkl'); w.to_pickle('grid.pkl')
fig,axs=plt.subplots(1,2,figsize=(16,7),sharey=True)
for a,per in zip(axs,('Dec-Aug','Sep-Oct')):
    p=w.pivot(index='pb',columns='left',values=f'd_{per}')*1000
    m=w.pivot(index='pb',columns='left',values=f'bands_{per}')
    p=p.where(m>=(100 if per=='Dec-Aug' else 30))
    im=a.imshow(p.values,aspect='auto',origin='lower',cmap='RdYlGn_r',norm=TwoSlopeNorm(0,-3,3),extent=[0.5,30.5,-2.5,97.5])
    a.set_xlim(30.5,0.5); a.set_title(f'{per}: model minus Kalshi Brier (x1000). Green = model better',loc='left',fontsize=11)
    a.set_xlabel('Minutes left'); a.set_yticks(range(0,100,10))
    for (pb,l) in both[['pb','left']].values: a.add_patch(plt.Rectangle((l-0.5,pb-2.5),1,5,fill=False,ec='#000',lw=1.2))
axs[0].set_ylabel('Kalshi YES price (5¢ bins, start of bin)')
fig.colorbar(im,ax=axs,shrink=.8,label='Brier difference x1000 (negative = model better)')
fig.suptitle('Rebuilt range model vs Kalshi by price and minutes left (boxed = model better in BOTH periods)',x=0.01,ha='left',fontsize=13)
fig.savefig('/mnt/project-files/strategy/range_model_vs_kalshi_grid.png',dpi=100,bbox_inches='tight')
