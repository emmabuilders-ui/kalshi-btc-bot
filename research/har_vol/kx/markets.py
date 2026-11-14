"""Market lists for an hourly-range series (KXETH / KXXRP): historical (cursor, newest first) + live settled."""
import sys, time, pandas as pd, numpy as np, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import kget
ser = sys.argv[1]; out = f"{os.path.dirname(os.path.abspath(__file__))}/{ser}_markets.parquet"
keep = ['ticker', 'event_ticker', 'close_time', 'floor_strike', 'cap_strike', 'strike_type', 'result', 'expiration_value']
rows = []; cur = None; n = 0
while True:
    j = kget.get('/historical/markets', series_ticker=ser, limit=1000, **({'cursor': cur} if cur else {}))
    ms = j.get('markets', []); rows += [{k: m.get(k) for k in keep} for m in ms]; n += 1
    cur = j.get('cursor')
    if n % 100 == 0: print('hist pages', n, len(rows), rows[-1]['close_time'], flush=True)
    if not cur or not ms: break
H = pd.DataFrame(rows); H['src'] = 'hist'
print('historical', len(H), H.close_time.min(), H.close_time.max(), '429s', kget.N429[0], flush=True)
# live (last ~2 months), settled
rows = []; cur = None
while True:
    j = kget.get('/markets', series_ticker=ser, status='settled', limit=1000, min_close_ts=int(pd.Timestamp('2026-08-15', tz='UTC').timestamp()),
                 max_close_ts=int(pd.Timestamp('2026-11-01', tz='UTC').timestamp()), **({'cursor': cur} if cur else {}))
    ms = j.get('markets', []); rows += [{k: m.get(k) for k in keep} for m in ms]; cur = j.get('cursor')
    if not cur or not ms: break
L = pd.DataFrame(rows); L['src'] = 'live'
print('live', len(L), L.close_time.min() if len(L) else None, L.close_time.max() if len(L) else None, flush=True)
M = pd.concat([L, H]).drop_duplicates('ticker')
M['close'] = ((pd.to_datetime(M.close_time, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()).astype('int64')
for c in ('floor_strike', 'cap_strike', 'expiration_value'): M[c] = pd.to_numeric(M[c], errors='coerce')
M.to_parquet(out); print('saved', len(M), M.strike_type.value_counts().to_dict(), M.groupby(M.close // 86400 * 86400).size().describe().to_dict(), flush=True)
