"""Model-level view Michael asked for (2026-11-14 23:08Z): edge of buying YES / NO at the ask, by DECILE of the flow signal,
with average and per-day median, per price bucket x minute left x side x period, on every snapshot (not bot trades).
Deciles: Dec-Aug cut points of the signed 3-min net-volume z (D1 = heaviest selling ... D10 = heaviest buying), applied to Sep-Oct too;
'none' = CME closed / thin (no flow). Outlier analysis: mean vs per-day median vs 5% day-trimmed mean, share of the mean from the top 5% of days."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
FEE = lambda p: np.ceil(7 * p * (1 - p))
prod = sys.argv[1]; SIG = sys.argv[2] if len(sys.argv) > 2 else 'nvz_comb_3'
cols = ['ticker', 'snap_ts', 'ask', 'bid', 'res', 'min_left', 'period', 'month', SIG]
d = pd.read_parquet(f'work/merged_{prod}.parquet', columns=cols)
d = d[(d.ask > 0) & (d.ask < 1) & (d.bid > 0) & (d.bid < 1)]
d['day'] = pd.to_datetime(d.snap_ts, unit='s').dt.floor('D')
cuts = np.nanquantile(d.loc[d.period == 'Dec-Aug', SIG], np.linspace(0, 1, 11)); cuts[0], cuts[-1] = -np.inf, np.inf
d['decile'] = pd.cut(d[SIG], cuts, labels=[f'D{i}' for i in range(1, 11)]).astype(str).replace('nan', 'none')
rows = []
for side in ('YES', 'NO'):
    x = d.copy(); x['side'] = side
    if side == 'YES': x['price'] = 100 * x.ask; x['won'] = x.res
    else: x['price'] = 100 * (1 - x.bid); x['won'] = 1 - x.res
    x['fee'] = FEE(x.price / 100); x['edge'] = 100 * x.won - x.price - x.fee; x['band10'] = (np.clip(x.price, 0, 99) // 10 * 10).astype(int)
    rows.append(x[['period', 'month', 'day', 'side', 'band10', 'min_left', 'decile', 'price', 'won', 'edge', SIG]])
X = pd.concat(rows, ignore_index=True)
X.to_parquet(f'work/dec_rows_{prod}.parquet')

def stats(g):
    byday = g.groupby('day').edge.agg(['mean', 'size']); bym = g.groupby('month').edge.mean()
    s = {'n': len(g), 'avg_price': g.price.mean(), 'win_pct': 100 * g.won.mean(), 'win_minus_price': 100 * g.won.mean() - g.price.mean(),
         'edge_avg': g.edge.mean(), 'edge_day_median': byday['mean'].median(), 'days': len(byday), 'months_pos': f'{int((bym > 0).sum())}/{len(bym)}', 'edge_trim5_days': np.nan, 'top5pct_days_share_of_sum': np.nan}
    if len(byday) >= 20:
        lo, hi = byday['mean'].quantile([0.05, 0.95]); keep = byday[(byday['mean'] >= lo) & (byday['mean'] <= hi)]
        s['edge_trim5_days'] = np.average(keep['mean'], weights=keep['size'])
        top = byday.sort_values('mean', ascending=False).head(max(1, len(byday) // 20))
        s['top5pct_days_share_of_sum'] = (top['mean'] * top['size']).sum() / max(abs((byday['mean'] * byday['size']).sum()), 1e-9)
    return pd.Series(s)
DEC = [f'D{i}' for i in range(1, 11)] + ['none']
out = {}
out['pooled'] = X.groupby(['period', 'side', 'decile']).apply(stats, include_groups=False).reset_index()
out['by_band'] = X.groupby(['period', 'side', 'band10', 'decile']).apply(stats, include_groups=False).reset_index()
out['by_minute'] = X.groupby(['period', 'side', 'min_left', 'decile']).apply(stats, include_groups=False).reset_index()
out['full'] = X.groupby(['period', 'side', 'band10', 'min_left', 'decile']).apply(stats, include_groups=False).reset_index()
for k, v in out.items():
    v['decile'] = pd.Categorical(v.decile, DEC, ordered=True); v.sort_values([c for c in v.columns if c in ('period', 'side', 'band10', 'min_left', 'decile')], inplace=True)
    v.to_parquet(f'work/dec_{k}_{prod}.parquet')
pd.Series(cuts).to_csv(f'work/dec_cuts_{prod}.csv')
print('cuts', np.round(cuts, 2))
print(out['pooled'].round(2).to_string())
