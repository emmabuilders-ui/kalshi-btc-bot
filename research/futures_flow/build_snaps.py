import glob, zipfile, io, re, numpy as np, pandas as pd, pyarrow.csv as pc
from multiprocessing import Pool
K = np.arange(1, 15)  # snapshot at open + 60*k  -> minutes left = 15-k
def ts(x): return (pd.to_datetime(x, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().astype('int64')
def from_zip(f):
    z = zipfile.ZipFile(f)
    mk = pd.read_csv(io.BytesIO(z.read('markets.csv')))
    if mk.empty: return None, None
    mk = mk[['ticker', 'open_time', 'close_time', 'strike', 'result', 'expiration_value']].copy()
    mk['open_ts'] = ts(mk.open_time); mk['close_ts'] = ts(mk.close_time)
    tn = [n for n in z.namelist() if n.startswith('trades/')][0]
    tb = pc.read_csv(io.BytesIO(z.read(tn)), convert_options=pc.ConvertOptions(include_columns=['ticker', 'ts_unix', 'yes_price', 'count', 'taker_side'])).to_pandas()
    p = tb.yes_price.to_numpy()
    sub = np.abs(np.round(p * 100) - p * 100) > 1e-6
    audit = dict(file=f.split('/')[-1], day=re.search(r'(\d{4}-\d{2}-\d{2})', f.split('/')[-1]).group(1), n_markets=len(mk), n_trades=len(tb), frac_subcent=float(sub.mean()) if len(p) else np.nan)
    rows = []
    for tk, g in tb.groupby('ticker', sort=False):
        m = mk[mk.ticker == tk]
        if m.empty: continue
        m = m.iloc[0]
        g = g.sort_values('ts_unix'); t = g.ts_unix.to_numpy(); pr = g.yes_price.to_numpy(); side = g.taker_side.to_numpy(); cnt = g['count'].to_numpy()
        cum = np.cumsum(cnt)
        isy = side == 'yes'; ty, py = t[isy], pr[isy]; tn_, pn = t[~isy], pr[~isy]
        for k in K:
            s = m.open_ts + 60 * k
            r = dict(ticker=tk, k=int(k), snap_ts=int(s))
            i = np.searchsorted(ty, s, side='right') - 1
            r['ask'] = py[i] if i >= 0 else np.nan; r['ask_age'] = s - ty[i] if i >= 0 else np.nan
            j = np.searchsorted(tn_, s, side='right') - 1
            r['bid'] = pn[j] if j >= 0 else np.nan; r['bid_age'] = s - tn_[j] if j >= 0 else np.nan
            a = np.searchsorted(t, s, side='right') - 1
            r['last'] = pr[a] if a >= 0 else np.nan; r['cumvol'] = cum[a] if a >= 0 else 0.0
            rows.append(r)
    S = pd.DataFrame(rows)
    return mk, S, audit
def work(f):
    try:
        out = from_zip(f)
        if out[0] is None: return None
        return out
    except Exception as e:
        print('ERR', f, e, flush=True); return None
if __name__ == '__main__':
    zips = sorted(glob.glob('/mnt/project-files/data-download/kalshi-daily/**/Kalshi_BTC15M_*.zip', recursive=True))
    M, S, A = [], [], []
    with Pool(4) as p:
        for i, o in enumerate(p.imap_unordered(work, zips)):
            if o: M.append(o[0]); S.append(o[1]); A.append(o[2])
            if i % 20 == 0: print(i, len(zips), flush=True)
    M = pd.concat(M); S = pd.concat(S); A = pd.DataFrame(A)
    M['src'] = 'trades'; S['src'] = 'trades'
    # May-Jul from 5-s quotes
    mj = pd.read_csv('/mnt/project-files/data-download/kalshi-daily/2026-05_to_07/Kalshi_BTC15M_markets_2026-05-01_to_07-31.csv')
    mj = mj[['ticker', 'open_time', 'close_time', 'strike', 'result', 'expiration_value']].copy()
    mj['open_ts'] = ts(mj.open_time); mj['close_ts'] = ts(mj.close_time); mj['src'] = 'q5s'
    Q = []
    for f in sorted(glob.glob('/mnt/project-files/data-download/kalshi-daily/2026-05_to_07/quotes_5s/*.parquet')):
        q = pd.read_parquet(f, columns=['ticker', 'open_unix', 't', 'last', 'cumvol', 'ask', 'ask_age', 'bid', 'bid_age'])
        q = q[q.t.isin(60 * K)].copy(); q['k'] = (q.t // 60).astype(int); q['snap_ts'] = q.open_unix + q.t
        Q.append(q[['ticker', 'k', 'snap_ts', 'ask', 'ask_age', 'bid', 'bid_age', 'last', 'cumvol']])
    Q = pd.concat(Q); Q['src'] = 'q5s'
    M = pd.concat([M, mj]).drop_duplicates('ticker'); S = pd.concat([S, Q]).drop_duplicates(['ticker', 'k'])
    for c in ['strike','expiration_value']: M[c]=pd.to_numeric(M[c],errors='coerce')
    M['result']=M['result'].astype(str)
    M.to_parquet('work/markets.parquet'); S.to_parquet('work/snaps_raw.parquet'); A.to_csv('work/zip_audit.csv', index=False)
    print('markets', len(M), 'snaps', len(S))
