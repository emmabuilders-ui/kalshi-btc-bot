import json,gzip,sys,numpy as np,pandas as pd
from pathlib import Path
c=sys.argv[1]
ts=[];vs=[]
for f in sorted(Path('idx',c).glob('*.json.gz'),key=lambda p:int(p.name.split(".")[0])):
    d=json.load(gzip.open(f,'rt'))
    if d:
        k=np.array(sorted(map(int,d)),dtype=np.int64); ts.append(k); vs.append(np.array([d[str(x)] for x in k],dtype=np.float64))
t=np.concatenate(ts); v=np.concatenate(vs)
pd.DataFrame({'unix_s':t,'index':v}).to_csv(f'pkg/{c}_index_1s.csv.gz',index=False,compression='gzip')
rows=[];bad=0
for f in Path('tr',c).glob('*.json.gz'):
    try: L=json.load(gzip.open(f,'rt'))
    except Exception: bad+=1; continue
    for ct,p,n,s in L: rows.append((f.name[:-8],ct,float(p),float(n),s))
pd.DataFrame(rows,columns=['ticker','created_time','yes_price','count','taker_side']).to_csv(f'pkg/{c}_trades_last200s.csv.gz',index=False,compression='gzip')
print(c,len(t),len(rows),'bad trade files',bad)
