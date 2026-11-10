import numpy as np, pandas as pd, itertools, sys
from scipy.stats import kendalltau
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
d = pd.read_parquet('hb_rows.parquet')
d = d[d.spread<=0.10]
# longshot rows: YES side price = yes_ask; NO side price = 1-yes_bid
Y = d[d.yes_ask<=0.15].copy(); Y['side']='YES'; Y['price']=Y.yes_ask; Y['won']=Y.won_yes
N = d[(1-d.yes_bid)<=0.15].copy(); N['side']='NO'; N['price']=1-N.yes_bid; N['won']=1-N.won_yes
R = pd.concat([Y,N]); R = R[R.price>0]
R['fee'] = 0.07*R.price*(1-R.price)
R['pnl'] = R.won - R.price - R.fee          # $ per contract
R['pg'] = pd.cut(R.price*100,[0,5,10,15],labels=['1-5c','6-10c','11-15c'])
R['lg'] = pd.cut(R.left,[0,5,10,20,30],labels=['1-5m','6-10m','11-20m','21-30m'])
def summ(g):
    n=len(g); p=g.price.mean()*100; w=g.won.mean()*100; ev=g.pnl.mean()*100
    se = g.pnl.std()*100/np.sqrt(max(n,1))
    return pd.Series({'n':n,'price':round(p,2),'win%':round(w,2),'be%':round((g.price+g.fee).mean()*100,2),'w-p':round(w-p,2),'ev':round(ev,2),'se':round(se,2)})
def first_entry(x):  # one trade per ticker/side: first qualifying minute (earliest = largest left)
    return x.sort_values('left',ascending=False).groupby(['ticker','side']).head(1)
print('rows', len(R), R.groupby(['side','per']).size().to_dict())
print(R.groupby(['side','per','pg','lg'],observed=True).apply(summ).to_string())
# conditions
tr = R[R.per=='DecAug']
conds = {}
def terc(col, name=None, k=3):
    name = name or col+'_c'
    cuts = tr[col].quantile(np.linspace(0,1,k+1)).to_numpy().copy(); cuts[0]=-np.inf; cuts[-1]=np.inf
    R[name] = pd.cut(R[col], np.unique(cuts), labels=[f'{name}_T{i+1}' for i in range(len(np.unique(cuts))-1)])
    conds[name]=R[name]
for c in ['z_edge','z_edge60','tow_60','tow_300','tow_900','vol_ratio','sig60','sig15','ret_hour','dist_mid']: terc(c)
R['bands_away_c'] = pd.cut(R.bands_away,[-1,0,1,2,3,100],labels=['inside','1away','2away','3away','4+away']); conds['bands_away_c']=R['bands_away_c']
R['spread_c'] = pd.cut(R.spread*100+1e-9,[0,2.01,5.01,10.01],labels=['sp1-2','sp3-5','sp6-10']); conds['spread_c']=R['spread_c']
R['et_c'] = pd.cut(R.hour_et,[-1,3,8,12,16,20,23],labels=['et0-3','et4-8','et9-12','et13-16','et17-20','et21-23']); conds['et_c']=R['et_c']
R['wkend'] = np.where(R.dow>=5,'weekend','weekday'); conds['wkend']=R['wkend']
R['visited_c'] = np.where(R.visited==1,'visited','notvisited'); conds['visited_c']=R['visited_c']
R['tow60_sign'] = np.where(R.tow_60>0,'toward1m','away1m'); conds['tow60_sign']=R['tow60_sign']
R['tow300_sign'] = np.where(R.tow_300>0,'toward5m','away5m'); conds['tow300_sign']=R['tow300_sign']
R['z_cush_c'] = pd.cut(R.z_cush, [-1,0.5,1,2,100], labels=['cush<0.5','cush0.5-1','cush1-2','cush2+']); conds['z_cush_c']=R['z_cush_c']
# AUC / tau of continuous conditions vs won within side x price group (Dec-Aug)
from sklearn.metrics import roc_auc_score
rows=[]
for side in ['YES','NO']:
    for pg in ['1-5c','6-10c','11-15c']:
        g = tr[(tr.side==side)&(tr.pg==pg)]
        for c in ['z_edge','z_edge60','tow_60','tow_300','tow_900','vol_ratio','sig60','ret_hour','spread','left','bands_away','dist_mid','z_cush','hour_et']:
            x=g[c]; ok=x.notna()
            if ok.sum()>200 and g.won[ok].nunique()==2:
                rows.append(dict(side=side,pg=pg,cond=c,n=int(ok.sum()),auc=round(roc_auc_score(g.won[ok],x[ok]),3), tau=round(kendalltau(x[ok],g.won[ok])[0],3)))
A=pd.DataFrame(rows); print('\nAUC of condition vs win, Dec-Aug (0.5=coin flip)'); print(A.sort_values('auc').to_string())
# Full cell scan: side x pg x lg x each condition level; Dec-Aug choose (ev>0 and n>=100 and ev>1.5se), then Sep-Oct
out=[]; tried=0
for cname, col in conds.items():
    for (side,pg,lg,lev), g in R.groupby(['side','pg','lg',col],observed=True):
        tried+=1
        a=g[g.per=='DecAug']; b=g[g.per=='SepOct']
        if len(a)<100: continue
        sa=summ(a); sb=summ(b) if len(b)>=1 else None
        fa=summ(first_entry(a)); fb=summ(first_entry(b)) if len(b)>=1 else None
        ma = a.groupby('month').pnl.mean(); 
        out.append(dict(cond=cname,level=str(lev),side=side,pg=pg,lg=lg,n_DA=sa.n,price_DA=sa.price,win_DA=sa['win%'],ev_DA=sa.ev,se_DA=sa.se,mpos_DA=f"{(ma>0).sum()}/{len(ma)}",
                        n_SO=(sb.n if sb is not None else 0),price_SO=(sb.price if sb is not None else np.nan),win_SO=(sb['win%'] if sb is not None else np.nan),ev_SO=(sb.ev if sb is not None else np.nan),se_SO=(sb.se if sb is not None else np.nan),
                        fe_n_DA=fa.n,fe_ev_DA=fa.ev,fe_n_SO=(fb.n if fb is not None else 0),fe_ev_SO=(fb.ev if fb is not None else np.nan)))
O=pd.DataFrame(out); O.to_csv('hb_cells.csv',index=False)
print('\ncells tried',tried,'with n>=100 Dec-Aug',len(O))
sel = O[(O.ev_DA>0)&(O.ev_DA>1.5*O.se_DA)]
print('Dec-Aug positive & >1.5 SE:',len(sel)); print(sel.sort_values('ev_DA',ascending=False).to_string())
print('\nSep-Oct positive among them:',(sel.ev_SO>0).sum())
