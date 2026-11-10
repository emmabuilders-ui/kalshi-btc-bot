import glob, zipfile, io, numpy as np, pandas as pd, pyarrow.csv as pc
from multiprocessing import Pool
K = np.arange(1, 15); WIN = (10, 30, 60, 180)
def ts(x): return (pd.to_datetime(x, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds().astype('int64')
def work(f):
    z = zipfile.ZipFile(f); mk = pd.read_csv(io.BytesIO(z.read('markets.csv')))
    if mk.empty: return None
    mk['open_ts'] = ts(mk.open_time); o = dict(zip(mk.ticker, mk.open_ts))
    tn = [n for n in z.namelist() if n.startswith('trades/')][0]
    tb = pc.read_csv(io.BytesIO(z.read(tn)), convert_options=pc.ConvertOptions(include_columns=['ticker', 'ts_unix', 'yes_price', 'count', 'taker_side'])).to_pandas()
    rows = []
    for tk, g in tb.groupby('ticker', sort=False):
        if tk not in o: continue
        g = g.sort_values('ts_unix'); t = g.ts_unix.to_numpy(); c = g['count'].to_numpy(); p = g.yes_price.to_numpy()
        sg = np.where(g.taker_side.to_numpy() == 'yes', 1.0, -1.0)
        big = c >= 100
        cv = np.concatenate([[0], np.cumsum(c)]); cs = np.concatenate([[0], np.cumsum(sg * c)]); cn = np.arange(len(t) + 1)
        cb = np.concatenate([[0], np.cumsum(sg * c * big)]); cpv = np.concatenate([[0], np.cumsum(p * c)])
        for k in K:
            s = o[tk] + 60 * k; r = dict(ticker=tk, k=int(k)); i1 = np.searchsorted(t, s, side='right')
            r['of_signed_open'] = cs[i1] / max(cv[i1], 1e-9)
            for w in WIN:
                i0 = np.searchsorted(t, s - w, side='right'); v = cv[i1] - cv[i0]
                r[f'of_imb_{w}'] = (cs[i1] - cs[i0]) / v if v > 0 else 0.0
                r[f'of_vol_{w}'] = np.log1p(v); r[f'of_n_{w}'] = cn[i1] - cn[i0]
                r[f'of_vwap_{w}'] = (cpv[i1] - cpv[i0]) / v if v > 0 else np.nan
            i0 = np.searchsorted(t, s - 180, side='right'); r['of_big_180'] = cb[i1] - cb[i0]
            j = i1 - 1; r['last_px'] = p[j] if j >= 0 else np.nan
            for w in (30, 60, 180):
                jw = np.searchsorted(t, s - w, side='right') - 1; r[f'of_dpx_{w}'] = (p[j] - p[jw]) if (j >= 0 and jw >= 0) else 0.0
            rows.append(r)
    return pd.DataFrame(rows)
if __name__ == '__main__':
    zips = sorted(glob.glob('/mnt/project-files/data-download/kalshi-daily/**/Kalshi_BTC15M_*.zip', recursive=True))
    out = []
    with Pool(4) as pl:
        for i, d in enumerate(pl.imap_unordered(work, zips)):
            if d is not None: out.append(d)
            if i % 40 == 0: print(i, len(zips), flush=True)
    F = pd.concat(out); F.to_parquet('work/flow.parquet'); print(F.shape)
