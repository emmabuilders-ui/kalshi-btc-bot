import numpy as np, pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score
pd.set_option('display.width',250); pd.set_option('display.max_rows',600)
S=pd.read_parquet('m15_rows.parquet')
Y=S[S.ask<=0.15].copy(); Y['side']='YES'; Y['price']=Y.ask; Y['won']=Y.res.astype(int)
N=S[(1-S.bid)<=0.15].copy(); N['side']='NO'; N['price']=1-N.bid; N['won']=1-N.res.astype(int)
R=pd.concat([Y,N]); R=R[R.price>0].copy()
R['fee']=0.07*R.price*(1-R.price); R['pnl']=R.won-R.price-R.fee
# own-side signed quantities: for YES longshot BTC below strike (dist<0): distance to strike = -dist; move toward = +ret
sgn=np.where(R.side=='YES',1.0,-1.0)
R['gap']=-sgn*R.dist            # log distance BTC must travel to win (positive = out of the money)
R['zgap']=-sgn*R.z; R['zgap60']=-sgn*R.z60
for w in (10,60,300,900): R[f'tow_{w}']=sgn*R[f'ret_{w}']
R['tow_open']=sgn*R.ret_open
R['pg']=pd.cut(R.price*100,[0,5,10,15],labels=['1-5c','6-10c','11-15c'])
R['lg']=pd.cut(R.min_left,[0,3,6,9,14],labels=['1-3m','4-6m','7-9m','10-14m'])
# extra: did BTC already touch the winning side this contract? (for YES: hi>=strike)
R['touched']=np.where(R.side=='YES',R.hi>=R.strike,R.lo<R.strike).astype(int)
R['beyond_prev']=np.where(R.side=='YES',R.brti>R.prev_hi,R.brti<R.prev_lo).astype(int)   # BTC broke prior-hour range toward win
R['strike_in_prev']=((R.strike<=R.prev_hi)&(R.strike>=R.prev_lo)).astype(int)
def summ(g):
    n=len(g); p=g.price.mean()*100; w=g.won.mean()*100; ev=g.pnl.mean()*100; se=g.pnl.std()*100/np.sqrt(max(n,1))
    return pd.Series({'n':n,'price':round(p,2),'win%':round(w,2),'be%':round((g.price+g.fee).mean()*100,2),'w-p':round(w-p,2),'ev':round(ev,2),'se':round(se,2)})
def first_entry(x): return x.sort_values('min_left',ascending=False).groupby(['ticker','side']).head(1)
print('rows',len(R),R.groupby(['side','per']).size().to_dict())
print(R.groupby(['side','per','pg','lg'],observed=True).apply(summ).to_string())
tr=R[R.per=='DecAug']
rows=[]
for side in ['YES','NO']:
    for pg in ['1-5c','6-10c','11-15c']:
        g=tr[(tr.side==side)&(tr.pg==pg)]
        for c in ['zgap','zgap60','gap','tow_10','tow_60','tow_300','tow_900','tow_open','vol_ratio','sig60','sig15','sig1','spread','min_left','hour_et','cumvol','ask_age','touched','beyond_prev','strike_in_prev']:
            x=g[c]; ok=x.notna()
            if ok.sum()>200 and g.won[ok].nunique()==2:
                rows.append(dict(side=side,pg=pg,cond=c,n=int(ok.sum()),auc=round(roc_auc_score(g.won[ok],x[ok]),3),tau=round(kendalltau(x[ok],g.won[ok])[0],3)))
A=pd.DataFrame(rows); print('\nAUC vs win Dec-Aug'); print(A.sort_values('auc').to_string()); A.to_csv('m15_auc.csv',index=False)
conds={}
def terc(col,k=3):
    cuts=tr[col].quantile(np.linspace(0,1,k+1)).to_numpy().copy(); cuts[0]=-np.inf; cuts[-1]=np.inf; cuts=np.unique(cuts)
    R[col+'_c']=pd.cut(R[col],cuts,labels=[f'{col}_T{i+1}' for i in range(len(cuts)-1)]); conds[col+'_c']=R[col+'_c']
for c in ['zgap','zgap60','tow_10','tow_60','tow_300','tow_900','tow_open','vol_ratio','sig60','sig1','cumvol']: terc(c)
R['et_c']=pd.cut(R.hour_et,[-1,3,8,12,16,20,23],labels=['et0-3','et4-8','et9-12','et13-16','et17-20','et21-23']); conds['et_c']=R.et_c
R['wkend']=np.where(R.dow>=5,'weekend','weekday'); conds['wkend']=R.wkend
R['touched_c']=np.where(R.touched==1,'touched','nottouched'); conds['touched_c']=R.touched_c
R['beyond_c']=np.where(R.beyond_prev==1,'beyondprev','inprev'); conds['beyond_c']=R.beyond_c
R['sip_c']=np.where(R.strike_in_prev==1,'strikeInPrevRange','strikeOutside'); conds['sip_c']=R.sip_c
R['tow60s']=np.where(R.tow_60>0,'toward1m','away1m'); conds['tow60s']=R.tow60s
R['tow300s']=np.where(R.tow_300>0,'toward5m','away5m'); conds['tow300s']=R.tow300s
R['spread_c']=pd.cut(R.spread*100+1e-9,[0,1.01,2.01,100],labels=['sp1','sp2','sp3+']); conds['spread_c']=R.spread_c
out=[]; tried=0
for cname,col in conds.items():
    for (side,pg,lg,lev),g in R.groupby(['side','pg','lg',col],observed=True):
        tried+=1
        a=g[g.per=='DecAug']; b=g[g.per=='SepOct']
        if len(a)<100: continue
        sa=summ(a); sb=summ(b) if len(b) else None; fa=summ(first_entry(a)); fb=summ(first_entry(b)) if len(b) else None
        ma=a.groupby('month').pnl.mean()
        out.append(dict(cond=cname,level=str(lev),side=side,pg=pg,lg=lg,n_DA=sa.n,price_DA=sa.price,win_DA=sa['win%'],ev_DA=sa.ev,se_DA=sa.se,mpos_DA=f"{(ma>0).sum()}/{len(ma)}",
            n_SO=(sb.n if sb is not None else 0),price_SO=(sb.price if sb is not None else np.nan),win_SO=(sb['win%'] if sb is not None else np.nan),ev_SO=(sb.ev if sb is not None else np.nan),se_SO=(sb.se if sb is not None else np.nan),
            fe_n_DA=fa.n,fe_ev_DA=fa.ev,fe_n_SO=(fb.n if fb is not None else 0),fe_ev_SO=(fb.ev if fb is not None else np.nan)))
O=pd.DataFrame(out); O.to_csv('m15_cells.csv',index=False)
print('\ncells tried',tried,'with n>=100',len(O))
sel=O[(O.ev_DA>0)&(O.ev_DA>1.5*O.se_DA)]
print('Dec-Aug positive >1.5SE:',len(sel),' of which Sep-Oct positive:',(sel.ev_SO>0).sum())
print(sel.sort_values('ev_DA',ascending=False).to_string())
print('\nall Dec-Aug positive cells:',(O.ev_DA>0).sum(),' both positive:',((O.ev_DA>0)&(O.ev_SO>0)).sum())
print(O[(O.ev_DA>0)].sort_values('ev_DA',ascending=False).head(40).to_string())
R[[c for c in R.columns if not str(R[c].dtype).startswith('category')]].to_parquet('m15_long.parquet')
