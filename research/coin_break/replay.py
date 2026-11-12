import sys,json,gzip,math,datetime as dt,bisect
import numpy as np, pandas as pd
sys.path.insert(0,'/home/user/kalshi-btc-bot')
from kalshi_btc_bot.settlement import settle_prob
from pathlib import Path
coin=sys.argv[1]; limit_to_have=True
def ts(s): return int(dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp())
mk=json.load(gzip.open(f'mk_KX{coin}15M.json.gz','rt'))
mk=[m for m in mk if m['status'] in('finalized','settled') and m.get('floor_strike') is not None and m['result'] in('yes','no') and m['close_time']<'2026-11-12T00:00:00Z']
mk.sort(key=lambda m:m['close_time'])
cache={}
def hour(h):
    if h in cache: return cache[h]
    fn=Path(f'idx/{coin}/{h}.json.gz')
    if not fn.exists(): cache[h]=None; return None
    d={int(k):v for k,v in json.load(gzip.open(fn,'rt')).items()}
    cache[h]=d
    if len(cache)>6: cache.pop(next(iter(cache)))
    return d
def series(a,b):
    """array of prices for seconds a..b inclusive, ffill; None if hole >5 s at ends or missing hour"""
    out=np.full(b-a+1,np.nan)
    for h in range(a//3600*3600,b//3600*3600+1,3600):
        d=hour(h)
        if d is None: return None
        for t,v in d.items():
            if a<=t<=b: out[t-a]=v
    # forward fill
    m=np.isnan(out)
    if m.all(): return None
    idx=np.where(~m,np.arange(len(out)),0); np.maximum.accumulate(idx,out=idx); out=out[idx]
    if np.isnan(out[0]): return None
    return out
def fee(p): return 0.07*p*(1-p)
rows_li=[];rows_bk=[];rows_br=[]
miss=0
for m in mk:
    T=ts(m['close_time']); K=float(m['floor_strike']); yes=m['result']=='yes'
    s=series(T-60-700,T-1)
    if s is None: miss+=1; continue
    base=T-60-700
    at=lambda t:s[t-base]
    # validate settlement
    avg=float(np.mean([at(t) for t in range(T-60,T)]))
    if (avg>=K)!=yes: pass
    try: tr=json.load(gzip.open(f'tr/{coin}/{m["ticker"]}.json.gz','rt'))
    except Exception: tr=None
    if tr is None: miss+=1; continue
    tt=[(ts(c.split('.')[0]+'Z') if '.' in c else ts(c),float(p),side) for c,p,_,side in tr]
    tt.sort()
    times=[x[0] for x in tt]
    def fresh(t,side,maxage=10):
        i=bisect.bisect_right(times,t)-1
        while i>=0 and t-times[i]<=maxage:
            if tt[i][2]==side: return tt[i][1] if side=='yes' else 1-tt[i][1]
            i-=1
        return None
    def lastprice(t,maxage=10):
        i=bisect.bisect_right(times,t)-1
        if i>=0 and t-times[i]<=maxage: return tt[i][1]
        return None
    day=m['close_time'][:10]
    def sig(t):
        w=s[max(0,t-base-300):t-base+1]; lr=np.diff(np.log(w))
        return float(np.sqrt(max((lr*lr).mean()-lr.mean()**2,0)))
    # Brier snapshots
    for tau in (50,40,30,20,10):
        t=T-tau; sp=at(t); known=61-tau; ka=float(np.mean([at(x) for x in range(T-60,t+1)]))
        p=settle_prob(sp,K,tau,sig(t),ka,known); lp=lastprice(t)
        if lp is not None: rows_br.append(dict(coin=coin,day=day,tau=tau,p_sa=p,kal=lp,y=int(yes)))
    # lock-in
    done={5:False,2:False}
    for tau in range(55,4,-1):
        t=T-tau; sp=at(t); known=61-tau; ka=float(np.mean([at(x) for x in range(T-60,t+1)]))
        py=settle_prob(sp,K,tau,sig(t),ka,known)
        for side in ('yes','no'):
            pr=fresh(t,side)
            if pr is None or pr<=0.01 or pr>=0.99: continue
            p=py if side=='yes' else 1-py
            e=p-pr-fee(pr)
            for mg in (5,2):
                if not done[mg] and e>=mg/100:
                    done[mg]=True
                    win=(yes if side=='yes' else not yes)
                    rows_li.append(dict(coin=coin,day=day,margin=mg,tau=tau,side=side,price=pr,model=p,win=int(win),fee=fee(pr)))
    # break 60..150 s: spot crosses strike by >= thr (scaled $5 BTC at ~75k)
    thr=5.0*float(np.mean(s))/75000.0
    prevside=None; fired=False
    for tau in range(210,59,-1):
        t=T-tau; sp=at(t)
        side='yes' if sp>=K else 'no'
        if tau<=150 and not fired and prevside is not None and side!=prevside and abs(sp-K)>=thr:
            pass
        # event: first time spot is >= thr beyond strike on a side opposite to the side it was on before (cross)
        if tau<=150 and not fired and prevside is not None and ((side=='yes' and sp-K>=thr and prevside=='no') or (side=='no' and K-sp>=thr and prevside=='yes')):
            fired=True
            known=0; ka=None
            py=settle_prob(sp,K,tau,sig(t),None,0)
            pr=fresh(t,side,maxage=15)
            if pr is not None and 0.01<pr<0.99:
                p=py if side=='yes' else 1-py
                win=(yes if side=='yes' else not yes)
                rows_bk.append(dict(coin=coin,day=day,tau=tau,side=side,price=pr,model=p,win=int(win),fee=fee(pr),gate=int(p-pr-fee(pr)>=0)))
        # track last side beyond thr only (hysteresis): prevside = side once |sp-K|>=thr
        if abs(sp-K)>=thr or prevside is None: prevside=side if abs(sp-K)>=thr else prevside
pd.DataFrame(rows_li).to_csv(f'out_{coin}_lockin.csv',index=False)
pd.DataFrame(rows_bk).to_csv(f'out_{coin}_break.csv',index=False)
pd.DataFrame(rows_br).to_csv(f'out_{coin}_brier.csv',index=False)
print(coin,'contracts',len(mk),'missing',miss,'lockin',len(rows_li),'break',len(rows_bk),'brier',len(rows_br))
