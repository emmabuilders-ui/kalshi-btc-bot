import pandas as pd, numpy as np, math
d=pd.read_parquet('/mnt/project-files/strategy/inefficiency_map_rows.parquet')
d=d[d.lvl<=15].copy()
d['per']=np.where(d.month>='2026-09','SepOct','DecAug')
def fee(p): return math.ceil(7*p*(1-p)*100)/100  # per contract $, p in dollars... Kalshi fee = 0.07*C*P*(1-P) rounded up to cent
def summ(g):
    n=len(g); p=g.lvl.mean()/100; w=g.won.mean()
    f=0.07*p*(1-p)  # per contract in $, approx (no rounding)
    ev=(w-p-f)*100
    return pd.Series({'n':n,'price':round(p*100,2),'win%':round(w*100,2),'be%':round((p+f)*100,2),'win-price':round((w-p)*100,2),'ev_c':round(ev,2)})
pd.set_option('display.width',250)
# by market/side/period, all lvl<=15 pooled, and by 5c group
d['grp']=pd.cut(d.lvl,[0,5,10,15],labels=['1-5','6-10','11-15'])
print(d.groupby(['market','side','per','grp'],observed=True).apply(summ).to_string())
# months positive
m=d.groupby(['market','side','month']).apply(summ)
m['pos']=m.ev_c>0
print(m.groupby(level=[0,1]).pos.agg(['sum','count']))
# by minute left (hourly band 1-30; 15-min 1-14)
for mk in ['15-min','hourly band','hourly mark']:
    x=d[d.market==mk].groupby(['side','per','left']).apply(summ)
    print(mk); print(x.unstack('per')[['n','price','win%','ev_c']].to_string())
