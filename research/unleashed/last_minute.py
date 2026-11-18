"""Inside the settlement minute: 45/30/15 s left. Extrapolation: the model was trained on 1..14 min left only. Info only."""
import glob, zipfile, io, json, sys, numpy as np, pandas as pd, pyarrow.csv as pc
from multiprocessing import Pool
sys.path.insert(0, '/home/user/kalshi-btc-bot')
from kalshi_btc_bot.model_gap_paper import DirectModel
T0 = 1764460800; B = np.load('work/brti.npy'); Bf = pd.Series(B).ffill().to_numpy()
OFF = (45, 30, 15)
def work(f):
    z = zipfile.ZipFile(f); mk = pd.read_csv(io.BytesIO(z.read('markets.csv')))
    if mk.empty: return None
    mk = mk[['ticker','open_time','close_time','strike','result']].copy()
    ts = lambda x: int(pd.Timestamp(x).timestamp())
    tn = [n for n in z.namelist() if n.startswith('trades/')][0]
    tb = pc.read_csv(io.BytesIO(z.read(tn)), convert_options=pc.ConvertOptions(include_columns=['ticker','ts_unix','yes_price','count','taker_side'])).to_pandas()
    rows = []
    for tk, g in tb.groupby('ticker', sort=False):
        m = mk[mk.ticker == tk]
        if m.empty: continue
        m = m.iloc[0]; o, c = ts(m.open_time), ts(m.close_time)
        g = g.sort_values('ts_unix'); t = g.ts_unix.to_numpy(); pr = g.yes_price.to_numpy(); sd = g.taker_side.to_numpy(); cnt = g['count'].to_numpy(); cum = np.cumsum(cnt)
        iy = sd == 'yes'; ty, py, tn_, pn = t[iy], pr[iy], t[~iy], pr[~iy]
        for off in OFF:
            s = c - off; r = dict(ticker=tk, off=off, snap_ts=s, open_ts=o, close_ts=c, strike=m.strike, res=1 if str(m.result).upper() in ('YES','1','1.0') else 0)
            i = np.searchsorted(ty, s, 'right') - 1; j = np.searchsorted(tn_, s, 'right') - 1; a = np.searchsorted(t, s, 'right') - 1
            r['ask'] = py[i] if i >= 0 else np.nan; r['ask_age'] = s - ty[i] if i >= 0 else np.nan
            r['bid'] = pn[j] if j >= 0 else np.nan; r['bid_age'] = s - tn_[j] if j >= 0 else np.nan
            r['cumvol'] = cum[a] if a >= 0 else 0.0; rows.append(r)
    return pd.DataFrame(rows)
if __name__ == '__main__':
    zips = sorted(glob.glob('/mnt/project-files/data-download/kalshi-daily/**/Kalshi_BTC15M_*.zip', recursive=True))
    with Pool(4) as p: S = pd.concat([x for x in p.imap_unordered(work, zips) if x is not None])
    S['strike'] = pd.to_numeric(S.strike, errors='coerce'); S = S[S.strike > 1000]
    S = S[(S.ask_age < 20) & (S.bid_age < 20) & S.ask.notna() & S.bid.notna() & (S.ask > S.bid - 1e-9)].copy()
    L = np.zeros(len(Bf)); L[1:] = np.diff(np.log(Bf))
    c1 = np.concatenate([[0], np.cumsum(L)]); c2 = np.concatenate([[0], np.cumsum(L ** 2)])
    sd_ = lambda end, w: np.sqrt(np.maximum((c2[end]-c2[end-w])/w - ((c1[end]-c1[end-w])/w)**2, 0))
    e = (S.snap_ts.to_numpy() - T0).astype(np.int64); S = S[e > 4000].copy(); e = (S.snap_ts.to_numpy() - T0).astype(np.int64)
    S['brti'] = Bf[e]; S['sig1'] = sd_(e+1, 60); S['sig5'] = sd_(e+1, 300); S['sig15'] = sd_(e+1, 900); S['sig60'] = sd_(e+1, 3600)
    S['secs_left'] = S.close_ts - S.snap_ts; S['min_left'] = S.secs_left / 60
    S['dist'] = np.log(S.brti / S.strike); S['mid'] = (S.ask + S.bid) / 2; S['spread'] = S.ask - S.bid
    Teff = (S.secs_left - 40).clip(lower=1)
    S['z'] = S.dist / (S.sig15.clip(lower=1e-7) * np.sqrt(Teff)); S['z60'] = S.dist / (S.sig60.clip(lower=1e-7) * np.sqrt(Teff))
    S['vol_ratio'] = S.sig15 / S.sig60.clip(lower=1e-9); S['vol_ratio_short'] = S.sig1 / S.sig15.clip(lower=1e-9)
    for w in (10, 30, 60, 300, 900): S[f'ret_{w}'] = np.log(B[e] / B[e - w]) if False else np.log(Bf[e] / Bf[e - w])
    op = (S.open_ts.to_numpy() - T0).astype(np.int64); hi = np.empty(len(S)); lo = np.empty(len(S))
    for i, (a, b) in enumerate(zip(op, e)): seg = Bf[a:b+1]; hi[i] = seg.max(); lo[i] = seg.min()
    S['dist_hi'] = np.log(S.brti / hi); S['dist_lo'] = np.log(S.brti / lo)
    dt = pd.to_datetime(S.snap_ts, unit='s'); S['hour_utc'] = dt.dt.hour; S['dow'] = dt.dt.dayofweek
    eps = 1e-4; m_ = S.mid.clip(eps, 1-eps); S['x'] = np.log(m_/(1-m_)); S['log_cumvol'] = np.log1p(S.cumvol.fillna(0))
    dm = DirectModel(json.load(open('/home/user/kalshi-btc-bot/kalshi_btc_bot/data/model_gap_correction.json')))
    S['p'] = [dm.predict(r) for r in S[dm.features].to_dict('records')]
    S['period'] = np.where(S.snap_ts >= pd.Timestamp('2026-09-01', tz='UTC').timestamp(), 'Sep-Oct', 'Dec-Aug(in-sample)')
    S['month'] = pd.to_datetime(S.snap_ts, unit='s').dt.strftime('%Y-%m')
    S.to_parquet('work/lastmin.parquet'); print(len(S), S.groupby(['period','off']).size())
