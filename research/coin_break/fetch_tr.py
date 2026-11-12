import sys,json,gzip,time,datetime as dt
sys.path.insert(0,'.')
import kget
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
def ts(s): return int(dt.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp())
jobs=[]
for c in ['ETH','SOL','XRP']:
    rows=json.load(gzip.open(f'mk_KX{c}15M.json.gz','rt'))
    for r in rows:
        if r['status'] in('finalized','settled') and r['close_time']<'2026-11-12T00:00:00Z': jobs.append((c,r))
jobs.sort(key=lambda x:x[1]['close_time'])
for c in 'ETH','SOL','XRP': Path('tr').joinpath(c).mkdir(exist_ok=True)
def run(j):
    c,r=j; fn=Path('tr')/c/(r['ticker']+'.json.gz')
    if fn.exists(): return
    cl=ts(r['close_time']); out=[]
    for path in ['/historical/trades','/markets/trades']:
        cur=None; got=[]
        while True:
            x=kget.get(path,min_interval=0.08,ticker=r['ticker'],min_ts=cl-200,max_ts=cl+5,limit=1000,cursor=cur)
            if not x: break
            got+=[(t['created_time'],t['yes_price_dollars'],t['count_fp'],t.get('taker_side')) for t in x.get('trades',[])]
            cur=x.get('cursor')
            if not cur or not x.get('trades'): break
        out+=got
        if out: break
    gzip.open(fn,'wt').write(json.dumps(out))
n=[0]
def w(j):
    run(j); n[0]+=1
    if n[0]%1000==0: print(n[0],len(jobs),j[1]['close_time'],flush=True)
with ThreadPoolExecutor(6) as ex: list(ex.map(w,jobs))
print('DONE')
