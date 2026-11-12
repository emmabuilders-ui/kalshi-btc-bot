import sys,json,gzip; sys.path.insert(0,'.')
import kget
ser=sys.argv[1]
out=[];cur=None;pg=0
# live endpoint (recent) then historical
for path in ['/markets','/historical/markets']:
    cur=None
    while True:
        j=kget.get(path,min_interval=0.3,series_ticker=ser,limit=1000,cursor=cur,**({'status':'settled'} if path=='/markets' else {}))
        if not j: break
        ms=j.get('markets',[]); out+=ms; pg+=1
        cur=j.get('cursor')
        print(ser,path,pg,len(out),ms[-1]['open_time'][:10] if ms else None,flush=True)
        if not cur or not ms: break
seen={};
for m in out: seen[m['ticker']]=m
keep=['ticker','event_ticker','open_time','close_time','floor_strike','result','expiration_value','volume_fp','status','settlement_ts','strike_type']
rows=[{k:m.get(k) for k in keep} for m in seen.values()]
json.dump(rows,gzip.open(f'mk_{ser}.json.gz','wt'))
print(ser,'DONE',len(rows),min(r['open_time'] for r in rows),max(r['open_time'] for r in rows))
