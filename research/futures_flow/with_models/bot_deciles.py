"""Bot-level view: each bot's backtest trades by flow decile (same Dec-Aug cut points as the snapshot view, signed toward the bought side
for directional contracts; |z| deciles for band bots), avg + per-day median edge, both periods."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
T = pd.read_parquet('work/wm_trades.parquet'); SIG = 'nvz_comb_3'
cuts = pd.read_csv('work/dec_cuts_15m.csv', index_col=0).iloc[:, 0].to_numpy().copy(); cuts[0], cuts[-1] = -np.inf, np.inf
T['z_signed'] = np.where(T.kind == 'directional', T[SIG] * T.dir, T[SIG])
T['decile'] = pd.cut(T.z_signed, cuts, labels=[f'D{i}' for i in range(1, 11)]).astype(str).replace('nan', 'none')
def stats(g):
    byday = g.groupby('day').edge.mean(); bym = g.groupby('month').edge.mean()
    return pd.Series({'n': len(g), 'avg_price': g.price.mean(), 'win_pct': 100 * g.won.mean(), 'edge_avg': g.edge.mean(), 'edge_day_median': byday.median(), 'days': len(byday), 'months_pos': f'{int((bym > 0).sum())}/{len(bym)}'})
R = T.groupby(['bot', 'period', 'decile']).apply(stats, include_groups=False).reset_index()
R2 = T.groupby(['bot', 'period', 'side', 'band10', 'min_left', 'decile']).apply(stats, include_groups=False).reset_index()
DEC = [f'D{i}' for i in range(1, 11)] + ['none']
for r in (R, R2): r['decile'] = pd.Categorical(r.decile, DEC, ordered=True)
R.sort_values(['bot', 'period', 'decile']).to_parquet('work/botdec_pooled.parquet'); R2.sort_values(['bot', 'period', 'side', 'band10', 'min_left', 'decile']).to_parquet('work/botdec_full.parquet')
print(R.sort_values(['bot', 'period', 'decile']).round(1).to_string())
