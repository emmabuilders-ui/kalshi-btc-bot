import numpy as np, pandas as pd
T0=1764460800
b=pd.Series(np.load('work/brti.npy')).ffill().to_numpy().astype(float)
nmin=len(b)//60; M=b[:nmin*60].reshape(nmin,60); C1=pd.Series(M[:,-1])
n5=nmin//5; C5=pd.Series(C1[:n5*5].to_numpy().reshape(n5,5)[:,-1])
def rsi(c,n=14):
    d=c.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean(); dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return (100-100/(1+up/dn.replace(0,np.nan))).to_numpy()
R1=rsi(C1); R5=rsi(C5)
def prep(A,model):
    A=A.copy(); A['dev']=100*np.where(A.side=='YES',1,-1)*A.dist if 'dev' not in A else A['dev']
    A['d']=100*(A.p_s-A.ask_s)
    i1=((A.snap_ts-T0)//60-1).astype(int).to_numpy(); i5=((A.snap_ts-T0)//300-1).astype(int).to_numpy()
    r1=R1[i1]; r5=R5[i5]; sgn=np.where(A.side=='YES',1,-1)
    A['rsi1']=np.where(sgn>0,r1,100-r1); A['rsi5']=np.where(sgn>0,r5,100-r5)   # own side: high = BTC moving toward the side
    rb=[0,30,45,55,70,101]; rl=['<30','30-45','45-55','55-70','>70']
    A['r1b']=pd.cut(A.rsi1,rb,right=False,labels=rl); A['r5b']=pd.cut(A.rsi5,rb,right=False,labels=rl)
    A['db']=pd.cut(A.dev,[-100,-.1,0,.05,.1,.2,100],right=False,labels=['<-.1','-.1..0','0..+.05','.05..+.1','.1..+.2','>+.2'])
    A['gg']=pd.cut(A.d,[-100,0,3,5,100],right=False,labels=['<0','0-3','3-5','5+'])
    A['mg']=pd.cut(A.min_left,[1,4,8,12,15],right=False,labels=['1-3','4-7','8-11','12-14'])
    A['model']=model; return A
D=prep(pd.read_parquet('work/direct_all.parquet'),'Direct')
Cc=pd.read_parquet('work/unleashed_all.parquet'); Cc['dev']=100*np.where(Cc.side=='YES',1,-1)*Cc.dist
C=prep(Cc,'Correction')
for A in (D,C): A.drop(columns=[c for c in A.columns if c in ('day',)],inplace=True,errors='ignore')
ALL=pd.concat([D,C]); ALL.to_parquet('work/rsi_all.parquet'); print(len(ALL), ALL.rsi1.notna().mean(), ALL.rsi5.notna().mean())
