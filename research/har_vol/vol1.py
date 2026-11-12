"""Stage 1: build 1-min BRTI bars + realized-variance series and forecaster inputs."""
import numpy as np, pandas as pd, sys
W = sys.argv[1]  # work dir holding brti.npy
T0 = 1764460800
B = np.load(W + '/brti.npy')
n = len(B) // 60 * 60
B = B[:n]
miss = np.isnan(B).reshape(-1, 60).sum(1)            # missing seconds per minute
Bf = pd.Series(B).ffill().to_numpy()
close = Bf[59::60]                                   # close of each minute (last second)
hi = np.nanmax(B.reshape(-1, 60), 1); lo = np.nanmin(B.reshape(-1, 60), 1)
m = len(close)
t = T0 + 60 * np.arange(m) + 60                      # end-of-minute unix time
r = np.full(m, np.nan); r[1:] = np.diff(np.log(close))
valid = (miss <= 30) & np.isfinite(close)
valid[1:] &= valid[:-1]                              # a return needs both closes
rv = np.where(valid, r ** 2, np.nan)
df = pd.DataFrame(dict(t=t, close=close, hi=hi, lo=lo, r=r, rv=rv, valid=valid, miss=miss))
# true range (1-min bars)
pc = np.roll(close, 1); pc[0] = np.nan
tr = np.maximum.reduce([hi - lo, np.abs(hi - pc), np.abs(lo - pc)]) / close
df['tr'] = np.where(valid, tr, np.nan)
def wsum(x, w):  # trailing sum over last w minutes INCLUDING current; nan if any nan inside
    c = np.concatenate([[0], np.nancumsum(x)]); bad = np.concatenate([[0], np.cumsum(np.isnan(x))])
    out = np.full(len(x), np.nan); out[w - 1:] = c[w:] - c[:-w]
    nb = np.full(len(x), np.nan); nb[w - 1:] = bad[w:] - bad[:-w]
    out[nb > 0] = np.nan
    return out
def fsum(x, h):  # forward sum over next h minutes (t+1..t+h)
    s = wsum(x, h); out = np.full(len(x), np.nan); out[:-h] = s[h:]; return out
for w in (5, 15, 60, 240, 1440):
    df[f'rv{w}'] = wsum(rv, w)
df['fwd15'] = fsum(rv, 15); df['fwd60'] = fsum(rv, 60)
for h in range(1, 31):
    if h not in (15,): df[f'fwd{h}'] = fsum(rv, h)
# s100: SD of last 100 1-min returns (ddof=1) as in the live model; s10 for the vol state
s1 = wsum(r, 100); s2 = wsum(r ** 2, 100)
df['s100'] = np.sqrt(np.maximum((s2 - s1 ** 2 / 100) / 99, 0))
s1 = wsum(r, 10); s2 = wsum(r ** 2, 10)
df['s10'] = np.sqrt(np.maximum((s2 - s1 ** 2 / 10) / 9, 0))
# EWMA(0.97) of squared returns, restart after invalid
ew = np.full(m, np.nan); v = np.nan
for i in range(m):
    if not valid[i]: v = np.nan; ew[i] = np.nan; continue
    v = rv[i] if np.isnan(v) else 0.97 * v + 0.03 * rv[i]
    ew[i] = v
df['ewma'] = ew
# ATR14 (Wilder)
atr = np.full(m, np.nan); a = np.nan; cnt = 0
trv = df.tr.to_numpy()
for i in range(m):
    if np.isnan(trv[i]): a = np.nan; cnt = 0; atr[i] = np.nan; continue
    if np.isnan(a): a = trv[i]; cnt = 1
    else: a = (a * 13 + trv[i]) / 14; cnt += 1
    atr[i] = a if cnt >= 14 else np.nan
df['atr14'] = atr
df['month'] = pd.to_datetime(df.t, unit='s').dt.strftime('%Y-%m')
df.to_parquet(W + '/vol_minutes.parquet')
print(len(df), 'minutes; valid', valid.mean().round(4)); print(df.month.value_counts().sort_index())
