"""Second-level combined flow (5 x @BTC + 0.1 x @MBT) from 1-sec bars, 2026-05-14 on. Bar stamped at close second t
covers (t-1, t]; stored at index t-1-S0 so a window [s-X, s) uses bars that closed at or before s."""
import numpy as np, pandas as pd
S0 = int(pd.Timestamp('2026-05-14', tz='UTC').timestamp()); NSEC = int(pd.Timestamp('2026-11-15', tz='UTC').timestamp()) - S0
PQ = '/mnt/project-files/tradestation/parquet/'
def build():
    up = np.zeros(NSEC, np.float32); dn = np.zeros(NSEC, np.float32); tr = np.zeros(NSEC, np.float32)
    for sym, mult in [('btc', 5.0), ('mbt', 0.1)]:
        d = pd.read_parquet(f'{PQ}{sym}_1sec.parquet', columns=['time', 'upvol', 'downvol', 'trades'])
        t = ((d['time'].dt.tz_convert('UTC') - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()).to_numpy().astype(np.int64)
        idx = t - 1 - S0; ok = (idx >= 0) & (idx < NSEC)
        np.add.at(up, idx[ok], mult * d.upvol.to_numpy()[ok]); np.add.at(dn, idx[ok], mult * d.downvol.to_numpy()[ok]); np.add.at(tr, idx[ok], d.trades.to_numpy()[ok])
    np.savez_compressed('work/flow_sec.npz', up=up, dn=dn, tr=tr)
    print('seconds', NSEC, 'with trades', (tr > 0).mean().round(4))
class FlowSec:
    def __init__(self):
        z = np.load('work/flow_sec.npz')
        self.cup = np.concatenate([[0.], np.cumsum(z['up'], dtype=np.float64)]); self.cdn = np.concatenate([[0.], np.cumsum(z['dn'], dtype=np.float64)]); self.ctr = np.concatenate([[0.], np.cumsum(z['tr'], dtype=np.float64)])
    def at(self, snap_ts, X):
        s = np.asarray(snap_ts, np.int64) - S0; a = s - X
        ok = (a >= 0) & (s <= NSEC); a_ = np.clip(a, 0, NSEC); s_ = np.clip(s, 0, NSEC)
        up = self.cup[s_] - self.cup[a_]; dn = self.cdn[s_] - self.cdn[a_]; tr = self.ctr[s_] - self.ctr[a_]
        with np.errstate(invalid='ignore', divide='ignore'):
            vi = np.where(up + dn > 0, (up - dn) / (up + dn), np.nan)
        vi[~ok] = np.nan
        return pd.DataFrame({f'svi_{X}': vi, f'snv_{X}': np.where(ok, up - dn, np.nan), f'str_{X}': np.where(ok, tr, np.nan)})
if __name__ == '__main__':
    build()
