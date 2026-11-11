import sys, numpy as np, pandas as pd, time, pickle, os
from zoneinfo import ZoneInfo
import kget
from kget import get
T0=1764460800; B=np.load('work/brti.npy')
a,b=sys.argv[1],sys.argv[2]; tag=sys.argv[3]
kget_interval=0.13
ET=ZoneInfo('America/New_York')
live=pd.read_parquet('work/btcd_markets_all.parquet')
live['ct']=live.close
live_ev={e:g for e,g in live.groupby('event_ticker')}
def evt(ct):
    d=pd.Timestamp(ct,unit='s',tz='UTC').tz_convert(ET); return f"KXBTCD-{d.strftime('%y')}{d.strftime('%b').upper()}{d.strftime('%d')}{d.strftime('%H')}"
def tk(ev,K): return f'{ev}-T{K-0.01:.2f}'
def px(c,k):
    v=c.get(k)
    if v is None: return np.nan
    if isinstance(v,dict): v=v.get('close_dollars',v.get('close'))
    try: return float(v)
    except: return np.nan
def candles(t,ct,hist):
    path=f'/historical/markets/{t}/candlesticks' if hist else f'/series/KXBTCD/markets/{t}/candlesticks'
    try: j=get(path,min_interval=kget_interval,start_ts=ct-1860,end_ts=ct,period_interval=1)
    except Exception as e:
        if '404' in str(e): return None
        raise
    return [(c['end_period_ts'],px(c,'yes_ask'),px(c,'yes_bid')) for c in j.get('candlesticks',[])]
t0=int(pd.Timestamp(a,tz='UTC').timestamp()); t1=int(pd.Timestamp(b,tz='UTC').timestamp())
cut=int(pd.Timestamp('2026-09-04',tz='UTC').timestamp())
rows=[]; meta=[]; step_cache={}
for ct in range(t0,t1,3600):
    spot=B[ct-1800-T0]
    if np.isnan(spot): continue
    ev=evt(ct); hist=ct<cut
    if ev in live_ev:   # exact list (live era)
        g=live_ev[ev]; g=g.iloc[np.argsort(np.abs(g.floor_strike.to_numpy()-spot))[:7]]
        strikes=[(r.ticker,r.floor_strike,r.result) for r in g.itertuples()]
    else:
        day=ct//86400
        if day not in step_cache:
            hit=[]
            c19=day*86400+19*3600; base=int(round(B[c19-1800-T0]/50)*50)
            for off in range(-400,401,50):
                K=base+off; 
                if candles(tk(evt(c19),K),c19,True) is not None: hit.append(K)
                if len(hit)>=3: break
            hit=sorted(hit); step_cache[day]=(hit[1]-hit[0]) if len(hit)>=2 else None
        step=step_cache[day]
        if not step: continue
        base=int(round(spot/step)*step); Ks=[base+i*step for i in (-3,-2,-1,0,1,2,3)]
        Ks=sorted(Ks,key=lambda K:abs(K-spot))[:7]
        strikes=[(tk(ev,K),K-0.01,None) for K in Ks]
    for t,K,res in strikes:
        c=candles(t,ct,hist)
        if not c: continue
        avg=np.nanmean(B[ct-60-T0:ct-T0]); r=res if res else ('yes' if avg>=K else 'no')
        for tt,ya,yb in c: rows.append((t,ct,K,r,tt,ya,yb))
    if (ct//3600)%24==0:
        print(pd.Timestamp(ct,unit='s'),len(rows),flush=True)
        pd.DataFrame(rows,columns=['ticker','close','strike','result','t','yes_ask','yes_bid']).to_parquet(f'work/btcd_q_{tag}.parquet')
pd.DataFrame(rows,columns=['ticker','close','strike','result','t','yes_ask','yes_bid']).to_parquet(f'work/btcd_q_{tag}.parquet'); print('DONE',len(rows))
