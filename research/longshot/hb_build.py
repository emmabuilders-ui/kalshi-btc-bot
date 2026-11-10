# Hourly band (KXBTC) longshot dataset: band-minutes with YES ask<=15c or NO ask<=15c, with conditions at or before the minute end.
import numpy as np, pandas as pd
T0 = 1764460800
B = np.load('/home/user/kalshi-btc-bot/research/ml_replay/work/brti.npy'); Bf = pd.Series(B).ffill().to_numpy()
L = np.zeros(len(Bf)); L[1:] = np.nan_to_num(np.diff(np.log(Bf)))
c1 = np.concatenate([[0], np.cumsum(L)]); c2 = np.concatenate([[0], np.cumsum(L**2)])
def sd(end, w): a = end - w; s1 = c1[end]-c1[a]; s2 = c2[end]-c2[a]; return np.sqrt(np.maximum(s2/w-(s1/w)**2, 0))
bands = pd.read_parquet('/mnt/project-files/data-download/kalshi-hourly-range/bands_dec2025_oct2026.parquet').drop_duplicates('ticker')
bands = bands[bands.strike_type=='between'].copy()
q = pd.read_parquet('/mnt/project-files/data-download/kalshi-hourly-range/quotes_1min_last60_dec2025_oct2026.parquet').drop_duplicates(['ticker','t'])
d = q.merge(bands[['ticker','floor_strike','cap_strike','result','close','event_ticker']], on='ticker')
d['left'] = ((d.close - d.t)//60).astype(int)   # minutes left at end of candle minute t
d = d[(d.left>=1)&(d.left<=30)]
d = d[d.yes_ask.notna() & d.yes_bid.notna() & (d.yes_ask>=d.yes_bid)]
d['spread'] = d.yes_ask - d.yes_bid
d['won_yes'] = (d.result=='yes').astype(int)
e = (d.t.to_numpy() - T0).astype(np.int64)
ok = (e>=3600)&(e<len(Bf)); d = d[ok].copy(); e = e[ok]
d['brti'] = Bf[e]
d['sig15'] = sd(e+1, 900); d['sig60'] = sd(e+1, 3600); d['sig5'] = sd(e+1,300)
for w in (60,300,900): d[f'ret_{w}'] = np.log(Bf[e]/Bf[e-w])
# hour open price (close-3600) and high/low since hour open
op = (d.close.to_numpy()-3600-T0).astype(np.int64)
d['ret_hour'] = np.log(Bf[e]/Bf[op])
hi = np.empty(len(d)); lo = np.empty(len(d))
# compute per unique (close, t) to save time
key = pd.DataFrame({'op':op,'e':e}).drop_duplicates()
hl = {}
for a,b in zip(key.op.to_numpy(), key.e.to_numpy()):
    seg = Bf[a:b+1]; hl[(a,b)] = (seg.max(), seg.min())
hi = np.array([hl[(a,b)][0] for a,b in zip(op,e)]); lo = np.array([hl[(a,b)][1] for a,b in zip(op,e)])
d['hour_hi']=hi; d['hour_lo']=lo
width = d.cap_strike - d.floor_strike
mid = (d.cap_strike + d.floor_strike)/2
d['width']=width
# signed distance to band: 0 if inside; positive $ distance to nearest edge if outside
below = d.brti < d.floor_strike; above = d.brti > d.cap_strike
d['inside'] = (~below & ~above).astype(int)
d['dist_edge'] = np.where(below, d.floor_strike - d.brti, np.where(above, d.brti - d.cap_strike, 0.0))
d['dist_mid'] = (d.brti - mid)  # signed
d['bands_away'] = np.where(below, np.ceil((d.floor_strike - d.brti)/width), np.where(above, np.ceil((d.brti-d.cap_strike)/width), 0))
Teff = (d.left*60 - 40).clip(lower=1)
d['z_edge'] = d.dist_edge/(d.brti*d.sig15.clip(lower=1e-7)*np.sqrt(Teff))   # vol units to nearest edge (0 inside)
d['z_edge60'] = d.dist_edge/(d.brti*d.sig60.clip(lower=1e-7)*np.sqrt(Teff))
# for inside: cushion to nearest edge in vol units
cush = np.minimum(d.brti-d.floor_strike, d.cap_strike-d.brti)
d['z_cush'] = np.where(d.inside==1, cush/(d.brti*d.sig15.clip(lower=1e-7)*np.sqrt(Teff)), np.nan)
# move toward band over 1/5 min: positive = toward band (for outside) ; for inside: toward nearest edge (positive = toward edge)
sgn_to_band = np.where(below, 1.0, np.where(above, -1.0, np.where(d.brti-d.floor_strike < d.cap_strike-d.brti, -1.0, 1.0)))
for w in (60,300,900): d[f'tow_{w}'] = sgn_to_band*d[f'ret_{w}']*d.brti  # $ moved toward band/edge
d['visited'] = ((d.hour_hi>=d.floor_strike)&(d.hour_lo<=d.cap_strike)).astype(int)  # BTC was in band this hour
d['vol_ratio'] = d.sig15/d.sig60.clip(lower=1e-9)
dt = pd.to_datetime(d.t, unit='s', utc=True).dt.tz_convert('US/Eastern')
d['hour_et'] = dt.dt.hour; d['dow'] = dt.dt.dayofweek
d['month'] = pd.to_datetime(d.close, unit='s').dt.strftime('%Y-%m')
d['per'] = np.where(d.month>='2026-09','SepOct','DecAug')
d.to_parquet('hb_rows.parquet')
print(len(d), d.ticker.nunique()); print(d.groupby('per').size())
print(d[['yes_ask','yes_bid','spread','dist_edge','z_edge','bands_away']].describe().to_string())
