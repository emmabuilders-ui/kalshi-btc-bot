"""CME BTC futures order flow (TradeStation @BTC + @MBT 1-min bars, tick-rule up/down volume).

Minute grid from T0 (2025-11-30 00:00 UTC). Bar stamped at its CLOSE time t covers [t-60, t), so it is stored at
grid index (t-60-T0)//60 and is known at any snapshot second s >= t.  A window of W minutes ending at snapshot s
(whole minute) uses grid indices (s-T0)//60-W .. (s-T0)//60-1, i.e. bars that closed at or before s.
Flow in BTC units: 5 x @BTC contracts + 0.1 x @MBT contracts.
"""
import numpy as np, pandas as pd
T0 = 1764460800
NMIN = (int(pd.Timestamp('2026-11-15', tz='UTC').timestamp()) - T0) // 60
WINDOWS = [1, 2, 3, 5, 10]
SYMS = {'btc': 5.0, 'mbt': 0.1}
PQ = '/mnt/project-files/tradestation/parquet/'

def load_grid():
    g = {}
    tr_all = np.zeros(NMIN)
    for sym in SYMS:
        d = pd.read_parquet(f'{PQ}{sym}_1min.parquet')
        t = ((d['time'].dt.tz_convert('UTC') - pd.Timestamp('1970-01-01', tz='UTC')).dt.total_seconds()).to_numpy().astype(np.int64)
        idx = (t - 60 - T0) // 60
        ok = (idx >= 0) & (idx < NMIN)
        for c in ['upvol', 'downvol', 'uptrades', 'downtrades', 'trades']:
            a = np.zeros(NMIN); np.add.at(a, idx[ok], d[c].to_numpy()[ok].astype(float)); g[f'{sym}_{c}'] = a
        # close price of each bar for reference
        px = np.full(NMIN, np.nan); px[idx[ok]] = d['close'].to_numpy()[ok]; g[f'{sym}_close'] = px
        tr_all += g[f'{sym}_trades']
    g['trades_all'] = tr_all
    return g

def closed_mask(g):
    """True where CME is closed: rule (Fri 17:00 ET - Sun 18:00 ET, daily 17:00-18:00 ET) or a data gap >= 60 min."""
    t = pd.to_datetime(T0 + 60 * np.arange(NMIN), unit='s', utc=True).tz_convert('America/New_York')
    h = t.hour.to_numpy(); mi = t.minute.to_numpy(); dow = t.dayofweek.to_numpy()  # Mon=0
    hm = h * 60 + mi
    rule = (hm >= 17 * 60) & (hm < 18 * 60)
    rule |= (dow == 4) & (hm >= 17 * 60)
    rule |= (dow == 5)
    rule |= (dow == 6) & (hm < 18 * 60)
    # data gaps: runs of >= 60 minutes with zero trades in both symbols
    z = g['trades_all'] == 0
    gap = np.zeros(NMIN, bool)
    i = 0
    edges = np.flatnonzero(np.diff(np.concatenate([[0], z.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= 60: gap[a:b] = True
    return rule | gap, rule, gap

def build(save='work/flow_grid.npz'):
    g = load_grid(); closed, rule, gap = closed_mask(g)
    g['closed'] = closed.astype(np.int8)
    np.savez_compressed(save, **g)
    print('grid minutes', NMIN, 'closed', closed.mean().round(3), 'rule', rule.mean().round(3), 'gap only', (gap & ~rule).mean().round(4))
    return g

class Flow:
    def __init__(self, path='work/flow_grid.npz'):
        z = np.load(path); self.g = {k: z[k] for k in z.files}
        g = self.g
        comb = {}
        for c in ['upvol', 'downvol']:
            comb[c] = 5.0 * g[f'btc_{c}'] + 0.1 * g[f'mbt_{c}']
        for c in ['uptrades', 'downtrades', 'trades']:
            comb[c] = g[f'btc_{c}'] + g[f'mbt_{c}']
        self.series = {'btc': {c: g[f'btc_{c}'] for c in ['upvol', 'downvol', 'uptrades', 'downtrades', 'trades']},
                       'mbt': {c: g[f'mbt_{c}'] for c in ['upvol', 'downvol', 'uptrades', 'downtrades', 'trades']},
                       'comb': comb}
        self.closed = g['closed'].astype(bool)
        self.cum = {s: {c: np.concatenate([[0.0], np.cumsum(v)]) for c, v in d.items()} for s, d in self.series.items()}
        self.cclosed = np.concatenate([[0], np.cumsum(self.closed)])
        # trailing-24h SD of net signed volume per window and symbol set (strictly prior minutes, active only)
        self.nv_sd = {}
        for s in self.series:
            up, dn = self.series[s]['upvol'], self.series[s]['downvol']
            for W in WINDOWS:
                nv = pd.Series(up - dn).rolling(W).sum().to_numpy().copy()
                # window valid if none of its W minutes is closed
                cl = pd.Series(self.closed.astype(float)).rolling(W).sum().to_numpy()
                nv[(cl > 0) | np.isnan(cl)] = np.nan
                sd = pd.Series(nv).rolling(1440, min_periods=480).std().shift(1).to_numpy()
                self.nv_sd[(s, W)] = sd

    def at(self, snap_ts):
        """Signals for an array of snapshot unix seconds. Returns DataFrame (same order)."""
        snap_ts = np.asarray(snap_ts, dtype=np.int64)
        i = (snap_ts - T0) // 60  # grid index of the minute STARTING at the snapshot; bars used are < i
        out = {}
        for W in WINDOWS:
            a = i - W
            okw = (a >= 0) & (i <= NMIN)
            a_ = np.clip(a, 0, NMIN); i_ = np.clip(i, 0, NMIN)
            ncl = self.cclosed[i_] - self.cclosed[a_]
            for s in self.series:
                cs = self.cum[s]
                up = cs['upvol'][i_] - cs['upvol'][a_]; dn = cs['downvol'][i_] - cs['downvol'][a_]
                ut = cs['uptrades'][i_] - cs['uptrades'][a_]; dt = cs['downtrades'][i_] - cs['downtrades'][a_]
                tr = cs['trades'][i_] - cs['trades'][a_]
                valid = okw & (ncl == 0) & (tr >= 10)
                with np.errstate(invalid='ignore', divide='ignore'):
                    vi = np.where(up + dn > 0, (up - dn) / (up + dn), np.nan)
                    ti = np.where(tr > 0, (ut - dt) / tr, np.nan)
                    sd = self.nv_sd[(s, W)][np.clip(i_ - 1, 0, NMIN - 1)]
                    nvz = (up - dn) / sd
                for nm, v in [('vi', vi), ('ti', ti), ('nvz', nvz)]:
                    v = v.astype(float); v[~valid] = np.nan; out[f'{nm}_{s}_{W}'] = v
                out[f'tr_{s}_{W}'] = tr; out[f'vol_{s}_{W}'] = up + dn
            out[f'valid_{W}'] = okw & (ncl == 0) & ((self.cum['comb']['trades'][i_] - self.cum['comb']['trades'][a_]) >= 10)
        return pd.DataFrame(out)

if __name__ == '__main__':
    build()
    F = Flow()
    ts = np.array([int(pd.Timestamp(x, tz='UTC').timestamp()) for x in ['2026-03-10 14:05', '2026-03-14 18:30', '2026-03-10 21:30', '2026-07-03 19:00', '2026-09-20 12:00']])
    r = F.at(ts); print(r[[c for c in r.columns if c.endswith('_comb_3') or c.startswith('valid')]].T)
