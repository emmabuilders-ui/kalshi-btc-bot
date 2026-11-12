import sys,json,gzip,time,datetime as dt,threading
sys.path.insert(0,'.')
import kget
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ranges={'ETH':('2025-12-10T21:00:00Z','ETHUSD_RTI'),'SOL':('2026-01-09T00:00:00Z','SOLUSD_RTI'),'XRP':('2026-02-11T05:00:00Z','XRPUSD_RTI')}
end=dt.datetime(2026,11,12,0,0,tzinfo=dt.timezone.utc)
jobs=[]
import itertools
lists={}
for c,(s,i) in ranges.items():
    t=dt.datetime.fromisoformat(s.replace('Z','+00:00')); l=[]
    while t<end: l.append((c,i,t)); t+=dt.timedelta(hours=1)
    lists[c]=l
# interleave coins so all progress together
for grp in itertools.zip_longest(*lists.values()):
    jobs+= [g for g in grp if g]
def run(j):
    c,i,h=j; fn=Path('idx')/c/f'{int(h.timestamp())}.json.gz'
    if fn.exists(): return
    r=kget.get('/cfbenchmarks/history/values',min_interval=0.3,tries=60,id=i,timespan='HOUR',timestamp=h.strftime('%Y-%m-%dT%H:%M:%SZ'))
    sec={}
    if r and r.get('data'):
        for p in r['data']['payload']:
            if p['time']%1000==0: sec[p['time']//1000]=float(p['value'])
    gzip.open(fn,'wt').write(json.dumps(sec))
n=[0]
def w(j):
    run(j); n[0]+=1
    if n[0]%200==0: print(n[0],len(jobs),j[0],j[2],flush=True)
with ThreadPoolExecutor(20) as ex: list(ex.map(w,jobs))
print('DONE')
