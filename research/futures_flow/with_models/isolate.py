"""Where does the edge sit? 90-99c favourites (the only cells with average and median both positive in both periods) bought at the ask,
all snapshots, by minute left. Before = all; after = exclude snapshots where BTC is within k sigma-sqrt(t) of the strike (the tail-loss zone),
and a flow condition. Reports avg, per-day median, 5%-day-trimmed mean, win% next to avg price, n, both periods."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
T0 = 1764460800; FEE = lambda p: np.ceil(7 * p * (1 - p))
prod = sys.argv[1]
b = np.load('work/brti.npy'); b = pd.Series(b).ffill().bfill().to_numpy(); lr = np.diff(np.log(b), prepend=0.0)
c1 = np.concatenate([[0], np.cumsum(lr)]); c2 = np.concatenate([[0], np.cumsum(lr ** 2)])
d = pd.read_parquet(f'work/merged_{prod}.parquet', columns=['snap_ts', 'ask', 'bid', 'res', 'min_left', 'secs_left', 'dist', 'period', 'month', 'nvz_comb_3'])
d = d[(d.ask > 0) & (d.ask < 1) & (d.bid > 0) & (d.bid < 1) & (d.snap_ts - T0 > 1000) & (d.snap_ts - T0 < len(b) - 2)]
a = (d.snap_ts.to_numpy() - T0)
sig = np.sqrt(np.maximum((c2[a] - c2[a - 900]) / 900 - ((c1[a] - c1[a - 900]) / 900) ** 2, 1e-14))
d['z'] = np.abs(d.dist.to_numpy()) / (sig * np.sqrt(np.maximum(d.secs_left.to_numpy() - 40, 1)))
d['day'] = pd.to_datetime(d.snap_ts, unit='s').dt.floor('D')
rows = []
for side in ('YES', 'NO'):
    x = d.copy(); x['side'] = side
    x['price'] = 100 * x.ask if side == 'YES' else 100 * (1 - x.bid); x['won'] = x.res if side == 'YES' else 1 - x.res
    x['fav_z'] = np.where((x.dist > 0) == (side == 'YES'), x.z, -x.z)  # signed: + = BTC on the side we bought
    x['flow'] = x.nvz_comb_3 * (1 if side == 'YES' else -1)
    rows.append(x[x.price.between(90, 99.99)])
X = pd.concat(rows); X['fee'] = FEE(X.price / 100); X['edge'] = 100 * X.won - X.price - X.fee
def st(g):
    if len(g) == 0: return pd.Series({'n': 0})
    bd = g.groupby('day').edge.agg(['mean', 'size']); lo, hi = bd['mean'].quantile([.05, .95]); k = bd[(bd['mean'] >= lo) & (bd['mean'] <= hi)]
    return pd.Series({'n': len(g), 'avg_price': g.price.mean(), 'win_pct': 100 * g.won.mean(), 'edge_avg': g.edge.mean(), 'edge_day_median': bd['mean'].median(),
                      'edge_trim5': np.average(k['mean'], weights=k['size']) if len(k) else np.nan, 'loss_rate_pct': 100 * (1 - g.won.mean()), 'avg_loss_c': g.edge[g.won == 0].mean()})
X['mgrp'] = pd.cut(X.min_left, [0, 1, 2, 3, 5, 8, 14, 30], labels=['1', '2', '3', '4-5', '6-8', '9-14', '15-30'])
X['zgrp'] = pd.cut(X.fav_z, [-99, 0, 1, 2, 3, 4, 99], labels=['<0 (wrong side)', '0-1', '1-2', '2-3', '3-4', '>4'])
out = {}
out['by_z'] = X.groupby(['period', 'mgrp', 'zgrp'], observed=True).apply(st, include_groups=False).reset_index()
for k in (1, 2, 3):
    Y = X[X.fav_z >= k]; out[f'after_z{k}'] = Y.groupby(['period', 'mgrp'], observed=True).apply(st, include_groups=False).reset_index()
out['before'] = X.groupby(['period', 'mgrp'], observed=True).apply(st, include_groups=False).reset_index()
# flow deciles inside the cleaned set (z>=2): does flow add anything once outliers are removed?
Y = X[X.fav_z >= 2].copy(); cuts = np.nanquantile(Y.loc[Y.period == 'Dec-Aug', 'flow'], np.linspace(0, 1, 6)); cuts[0], cuts[-1] = -np.inf, np.inf
Y['quint'] = pd.cut(Y.flow, cuts, labels=['Q1 flow against', 'Q2', 'Q3', 'Q4', 'Q5 flow with']).astype(str).replace('nan', 'no flow')
out['after_z2_flow_quint'] = Y.groupby(['period', 'quint']).apply(st, include_groups=False).reset_index()
for k, v in out.items(): v.to_parquet(f'work/iso_{k}_{prod}.parquet')
pd.set_option('display.width', 250)
print(prod, 'BEFORE'); print(out['before'].round(2).to_string()); print('AFTER z>=2'); print(out['after_z2'].round(2).to_string() if 'after_z2' in out else '')
print(out['after_z2_flow_quint'].round(2).to_string())
