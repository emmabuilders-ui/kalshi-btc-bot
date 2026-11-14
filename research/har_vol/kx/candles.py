"""1-min candles (yes_ask/yes_bid close) for the nearest bands of every hourly event, last 31 min before the close. Per-day parquet, resumable."""
import sys, os, time, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + '/..')
import kget
ser, coin = sys.argv[1], sys.argv[2]          # KXETH ETH
NT = int(os.environ.get('NT', '3')); COVER = 0.025; NMIN, NMAX = 6, 14
T0 = 1764460800; B = pd.Series(np.load(f'{HERE}/../work_{coin}/brti.npy')).ffill().to_numpy()
M = pd.read_parquet(f'{HERE}/{ser}_markets.parquet'); M = M[M.strike_type == 'between'].copy()
M['mid'] = (M.floor_strike + M.cap_strike) / 2
cut = int(pd.Timestamp('2026-09-04', tz='UTC').timestamp())
def px(c, k):
    v = c.get(k)
    if isinstance(v, dict): v = v.get('close_dollars', v.get('close'))
    try: return float(v)
    except: return np.nan
def fetch(row):
    tk, ct = row
    j = None
    for path in (f'/historical/markets/{tk}/candlesticks', f'/series/{ser}/markets/{tk}/candlesticks'):
        try: j = kget.get(path, start_ts=ct - 1860, end_ts=ct, period_interval=1); break
        except Exception as e: continue
    if j is None: return [(tk, -1, np.nan, np.nan)]
    return [(tk, c['end_period_ts'], px(c, 'yes_ask'), px(c, 'yes_bid')) for c in j.get('candlesticks', [])]
out = f'{HERE}/cand/{ser}'; os.makedirs(out, exist_ok=True)
sel_all = []
for ev, g in M.groupby('event_ticker'):
    ct = int(g.close.iloc[0]); i = ct - 1800 - T0
    if i < 0 or i >= len(B) or np.isnan(B[i]): continue
    spot = B[i]; d = np.abs(g.mid.to_numpy() - spot)
    n = int(np.clip((d <= COVER * spot).sum(), NMIN, NMAX))
    sel_all.append(g.iloc[np.argsort(d)[:n]][['ticker', 'close']])
S = pd.concat(sel_all); S['day'] = pd.to_datetime(S.close, unit='s').dt.strftime('%Y-%m-%d')
S.to_parquet(f'{HERE}/{ser}_selected.parquet')
print('selected bands', len(S), 'events', S.close.nunique(), flush=True)
for day, g in S.groupby('day'):
    fn = f'{out}/{day}.parquet'
    if os.path.exists(fn): continue
    t0 = time.time(); rows = []
    with ThreadPoolExecutor(NT) as ex:
        for r in ex.map(fetch, list(zip(g.ticker, g.close))): rows += r
    pd.DataFrame(rows, columns=['ticker', 't', 'yes_ask', 'yes_bid']).to_parquet(fn)
    print(day, len(g), 'markets', len(rows), 'rows', f'{time.time()-t0:.0f}s', '429s', kget.N429[0], time.strftime('%H:%M:%S'), flush=True)
print('CANDLES_DONE', ser, flush=True)
