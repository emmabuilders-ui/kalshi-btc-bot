import numpy as np, pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',600)
PB=[0,10,20,30,40,50,60,70,80,90,100]; PL=['01-10','11-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-99']
Tm=pd.read_parquet('work/sr_trades_all.parquet')
L=pd.read_parquet('work/lastmin.parquet',columns=['ticker','off','ask','ask_age','bid','bid_age','res','period'])
def mk(df,side):
    d=df.copy(); d['side']=side
    d['a']=d.ask if side=='YES' else 1-d.bid
    d['fresh']=(d.ask_age if side=='YES' else d.bid_age)<=10
    d['win']=((d.res==1) if side=='YES' else (d.res==0)).astype(float); return d
Tl=pd.concat([mk(L,'YES'),mk(L,'NO')]); Tl=Tl[Tl.fresh&(Tl.a>=.01)&(Tl.a<=.99)].copy()
Tl['period']=Tl.period.str.replace('(in-sample)','',regex=False); Tl['k']=Tl.off; Tl['mg']=Tl.off
Tl['fee']=.07*Tl.a*(1-Tl.a); Tl['pn']=100*(Tl.win-Tl.a-Tl.fee); Tl['pb']=pd.cut(Tl.a*100,PB,labels=PL)
c=['side','mg','pb','period']; Tl=Tl.join(Tl.groupby(c,observed=True).pn.mean().rename('base'),on=c); Tl['ex']=Tl.pn-Tl.base
def s(g):
    n=len(g); sd=g.ex.std(); return pd.Series(dict(n=n,ask=round(100*g.a.mean(),1),win=round(100*g.win.mean(),1),avg=round(g.pn.mean(),2),med=round(g.pn.median(),2),ctrl=round(g.base.mean(),2),exc=round(g.ex.mean(),2),t=round(g.ex.mean()/(sd/np.sqrt(n)),1) if n>30 else np.nan))
res={}
for tag,T in (('min',Tm),('last',Tl)):
    E=pd.read_parquet(f'work/sr2_events_{tag}.parquet'); X=E.merge(T,on=['ticker','k','side'])
    X['sdec']=X.groupby(['ev','dir']).strength.transform(lambda x:pd.qcut(x.rank(method='first'),10,labels=False)+1) if tag=='min' else X.groupby(['ev','dir']).strength.transform(lambda x:pd.qcut(x.rank(method='first'),5,labels=False)+1)
    D=X.sort_values('strength').drop_duplicates(['ticker','k','side','ev'],keep='last')
    res[tag]=(X,D)
    print('=====',tag,len(E),len(X))
    print(D.groupby(['ev','side','period']).apply(s,include_groups=False).to_string())
    if tag=='last':
        print(D.groupby(['ev','side','mg','period']).apply(s,include_groups=False).to_string())
        print(D.groupby(['ev','side','pb','period'],observed=True).apply(s,include_groups=False).to_string())
    print('--hard only'); print(X[X.level=='hard'].groupby(['ev','dir','period']).apply(s,include_groups=False).to_string())
    print('--strength'); print(D.assign(sd=X.loc[D.index,'sdec'] if False else 0).shape)
    # strength cut on deduped with dir
    D=D.copy(); D['sdec']=D.groupby(['ev','side']).strength.transform(lambda x:pd.qcut(x.rank(method='first'),10 if tag=='min' else 5,labels=False)+1)
    st=D.groupby(['ev','side','sdec','period']).apply(s,include_groups=False); print(st.to_string())
    with pd.ExcelWriter(f'work/sr2_results_{tag}.xlsx') as w:
        D.groupby(['ev','side','mg','pb','period'],observed=True).apply(s,include_groups=False).reset_index().to_excel(w,sheet_name='min_x_price',index=False)
        st.reset_index().to_excel(w,sheet_name='strength',index=False)
        X.groupby(['ev','dir','level','period']).apply(s,include_groups=False).reset_index().to_excel(w,sheet_name='by_level',index=False)
