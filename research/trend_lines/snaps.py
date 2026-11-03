"""Band-minute snapshots for the last 30 min of each hourly KXBTC event: Kalshi quotes, range-model chance, outcome."""
import sys, json, math, numpy as np, pandas as pd
sys.path.insert(0,'/home/user/kalshi-btc-bot')
from kalshi_btc_bot.range_paper import RangeModel
b=np.load('../tl/brti.npy'); T0=int(open('../tl/brti_t0.txt').read())
M=RangeModel()
bands=pd.read_csv('bands.csv').set_index('ticker')
rows=[]
for l in open('candles.jsonl'):
    j=json.loads(l)
    for tk,ts,ya,yb in j['rows']:
        rows.append((tk,int(ts),float(ya) if ya is not None else np.nan,float(yb) if yb is not None else np.nan))
for l in open('hcandles.jsonl'):
    j=json.loads(l)
    for tk,ts,ya,yb in j['rows']:
        rows.append((tk,int(ts),float(ya) if ya is not None else np.nan,float(yb) if yb is not None else np.nan))
c=pd.DataFrame(rows,columns=['ticker','t','ask','bid']).drop_duplicates(['ticker','t'])
c=c.join(bands[['event_ticker','close','floor_strike','cap_strike','strike_type','result']],on='ticker')
c['left']=(c.close-c.t)//60
c=c[(c.left>=1)&(c.left<=30)&c.ask.notna()&c.bid.notna()]
# BRTI, vol per (event, minute)
key=c[['t']].drop_duplicates()
def vol(t):
    i=t-T0; cl=b[[i-1-60*(100-k) for k in range(101)]]
    return M.vol(cl)
V={t:vol(t) for t in key.t}
P=[]
for r in c.itertuples():
    v=V[r.t]
    if v is None: P.append(np.nan); continue
    lo,hi=r.floor_strike,r.cap_strike
    if r.strike_type=='greater': hi=None
    elif r.strike_type=='less': lo=None
    lo=None if (lo is None or (isinstance(lo,float) and math.isnan(lo))) else lo
    hi=None if (hi is None or (isinstance(hi,float) and math.isnan(hi))) else hi
    P.append(M.p_inside(b[r.t-T0],lo,hi,r.left,v[0],v[1]))
c['model_yes']=P; c['brti']=b[c.t.values-T0]
c['s100']=[V[t][0] if V[t] else np.nan for t in c.t]; c['vb']=[V[t][1] if V[t] else np.nan for t in c.t]
c['y']=(c.result=='yes').astype(int)
c.to_pickle('snaps.pkl'); print(len(c), c.event_ticker.nunique())
