import pandas as pd, numpy as np
from scipy.stats import norm
pd.set_option('display.width',250); pd.set_option('display.max_rows',600)
def summ(g):
    n=len(g); p=g.price.mean()*100; w=g.won.mean()*100; ev=g.pnl.mean()*100; se=g.pnl.std()*100/np.sqrt(max(n,1))
    return pd.Series({'trades':n,'avg_price_c':round(p,2),'win_pct':round(w,2),'breakeven_pct':round((g.price+g.fee).mean()*100,2),'win_minus_price':round(w-p,2),'ev_c':round(ev,2),'se_c':round(se,2),'t':round(ev/se if se>0 else np.nan,2)})
def fe(x): return x.sort_values('min_left',ascending=False).groupby(['ticker','side']).head(1)
R=pd.read_parquet('m15_long.parquet'); R['p_norm']=norm.cdf(-R.zgap)
zc=[-np.inf,0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.5,3.0,4.0,np.inf]
R['zb']=pd.cut(R.zgap,zc).astype(str)
R['mg']=pd.cut(R.min_left,[0,1,2,3,5,9,14],labels=['1','2','3','4-5','6-9','10-14']).astype(str)
R['mgc']=pd.cut(R.min_left,[0,3,14],labels=['1-3','4-14']).astype(str)   # coarse fallback
def table(fit):
    t=fit.groupby(['zb','mg']).won.agg(['mean','size']); tc=fit.groupby(['zb','mgc']).won.agg(['mean','size']); tz=fit.groupby('zb').won.agg(['mean','size'])
    return t,tc,tz
def apply(rows,tabs,minn=50):
    t,tc,tz=tabs
    a=rows.join(t.rename(columns={'mean':'p1','size':'n1'}),on=['zb','mg'])
    a=a.join(tc.rename(columns={'mean':'p2','size':'n2'}),on=['zb','mgc'])
    a=a.join(tz.rename(columns={'mean':'p3','size':'n3'}),on='zb')
    p=np.where(a.n1>=minn,a.p1,np.where(a.n2>=minn,a.p2,a.p3)); return p
# cross-fit Dec-Aug by month (table from other Dec-Aug months), frozen Dec-Aug table for Sep-Oct
DA=R[R.per=='DecAug']; SO=R[R.per=='SepOct'].copy()
parts=[]
for m in sorted(DA.month.unique()):
    te=DA[DA.month==m].copy(); fit=DA[DA.month!=m]
    te['p_emp']=apply(te,table(fit)); parts.append(te)
SO['p_emp']=apply(SO,table(DA))
A=pd.concat(parts+[SO]); A['edge_emp']=A.p_emp-A.price-A.fee; A['edge_norm']=A.p_norm-A.price-A.fee
# the Dec-Aug table itself (for the workbook)
t,tc,tz=table(DA); T=t.reset_index(); T.columns=['z_bin','minutes_left','win_rate','n']; T['win_rate']=(T.win_rate*100).round(2)
T['normal_curve_pct']=np.nan
print('Dec-Aug table (win % of <=15c rows by distance/SD bin x minutes left):'); print(T.pivot(index='z_bin',columns='minutes_left',values='win_rate').to_string()); print(T.pivot(index='z_bin',columns='minutes_left',values='n').to_string())
# calibration check of emp vs norm on Sep-Oct
for col in ['p_emp','p_norm']:
    SOx=A[A.per=='SepOct'].copy(); SOx['fb']=pd.cut(SOx[col],[0,0.03,0.06,0.1,0.15,0.2,0.3,1])
    print(col,'Sep-Oct calibration'); print(SOx.groupby('fb',observed=True).agg(n=('won','size'),pred=(col,'mean'),actual=('won','mean')).round(3).to_string())
rows=[]
for m in [0,0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05]:
    s=fe(A[A.edge_emp>=m]).groupby('per').apply(summ)
    for per in s.index: rows.append(dict(rule='empirical',margin_c=m*100,per=per,**s.loc[per].to_dict()))
for m in [0.05,0.08]:
    s=fe(A[A.edge_norm>=m]).groupby('per').apply(summ)
    for per in s.index: rows.append(dict(rule='normal',margin_c=m*100,per=per,**s.loc[per].to_dict()))
M=pd.DataFrame(rows); print(M.set_index(['rule','margin_c','per']).unstack('per').to_string())
M.to_csv('emp_margins.csv',index=False); T.to_csv('emp_table.csv',index=False); A.to_parquet('emp_rows.parquet')
