"""15-min contract quotes at 60/45/30/15 s before close (settlement-window snapshots), May 14 2026 on.
Aug-Oct from trade ZIPs (ask = last yes-taker price, bid = last no-taker price); May 14-Jul from quotes_5s (t=840..885)."""
import glob, zipfile, io, numpy as np, pandas as pd, pyarrow.csv as pc
from multiprocessing import Pool
OFFS = [840, 855, 870, 885]  # seconds after open -> 60/45/30/15 s left
def ts(x): return (pd.to_datetime(x, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().astype('int64')
def from_zip(f):
    z = zipfile.ZipFile(f); mk = pd.read_csv(io.BytesIO(z.read('markets.csv')))
    if mk.empty: return None
    mk['open_ts'] = ts(mk.open_time)
    tn = [n for n in z.namelist() if n.startswith('trades/')][0]
    tb = pc.read_csv(io.BytesIO(z.read(tn)), convert_options=pc.ConvertOptions(include_columns=['ticker', 'ts_unix', 'yes_price', 'taker_side'])).to_pandas()
    rows = []
    for tk, g in tb.groupby('ticker', sort=False):
        m = mk[mk.ticker == tk]
        if m.empty: continue
        o = int(m.iloc[0].open_ts); g = g.sort_values('ts_unix'); t = g.ts_unix.to_numpy(); pr = g.yes_price.to_numpy(); isy = (g.taker_side == 'yes').to_numpy()
        ty, py, tn_, pn = t[isy], pr[isy], t[~isy], pr[~isy]
        for off in OFFS:
            s = o + off; i = np.searchsorted(ty, s, side='right') - 1; j = np.searchsorted(tn_, s, side='right') - 1
            rows.append(dict(ticker=tk, snap_ts=s, secs_left=900 - off, ask=py[i] if i >= 0 else np.nan, ask_age=s - ty[i] if i >= 0 else np.nan, bid=pn[j] if j >= 0 else np.nan, bid_age=s - tn_[j] if j >= 0 else np.nan))
    return pd.DataFrame(rows)
if __name__ == '__main__':
    zips = sorted(glob.glob('/mnt/project-files/data-download/kalshi-daily/Kalshi_BTC15M_2026-*.zip'))
    with Pool(4) as p: S = [x for x in p.imap_unordered(from_zip, zips) if x is not None]
    S = pd.concat(S)
    Q = []
    for f in sorted(glob.glob('/mnt/project-files/data-download/kalshi-daily/2026-05_to_07/quotes_5s/*.parquet')):
        q = pd.read_parquet(f, columns=['ticker', 'open_unix', 't', 'ask', 'ask_age', 'bid', 'bid_age']); q = q[q.t.isin(OFFS)].copy()
        q['snap_ts'] = q.open_unix + q.t; q['secs_left'] = 900 - q.t; Q.append(q[['ticker', 'snap_ts', 'secs_left', 'ask', 'ask_age', 'bid', 'bid_age']])
    S = pd.concat([S] + Q).drop_duplicates(['ticker', 'secs_left'])
    S.to_parquet('work/subminute_raw.parquet'); print(len(S), S.ticker.nunique())
