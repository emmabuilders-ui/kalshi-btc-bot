import numpy as np, pandas as pd
D = pd.read_parquet('work/autofeat.parquet'); ts = lambda s: pd.Timestamp(s, tz='UTC').timestamp()
D['per'] = np.where(D.snap_ts >= ts('2026-09-01'), 'Sep-Oct', np.where(D.snap_ts >= ts('2026-07-01'), 'Jul-Aug', 'Dec-Jun'))
THREE = [('vol_300', 12, '5-min volatility'), ('vol_600', 12, '10-min volatility'), ('vol_900', 9, '15-min volatility')]
out = []
for f, m, lab in THREE:
    d = D[D.min_left == m].copy()
    cuts = d[d.per == 'Dec-Jun'][f].quantile(np.linspace(0, 1, 11)).to_numpy().copy(); cuts[0], cuts[-1] = -np.inf, np.inf  # deciles set on Dec-Jun only
    d['decile'] = np.digitize(d[f], cuts[1:-1]) + 1
    for scope, s in (('all prices', d), ('10-39c', d[(d.mid >= 0.10) & (d.mid < 0.40)])):
        for (dec, per), h in s.groupby(['decile', 'per']):
            out.append(dict(feature=lab, minutes_left=m, prices=scope, decile=dec, period=per, vol_bps_per_sec=round(1e4 * h[f].mean(), 3),
                            snapshots=len(h), win_pct=round(100 * h.res.mean(), 1), avg_price_c=round(100 * h.mid.mean(), 1),
                            win_minus_price_pts=round(100 * (h.res.mean() - h.mid.mean()), 2), se_pts=round(100 * (h.res - h.mid).std() / np.sqrt(len(h)), 2)))
R = pd.DataFrame(out)
W = R.pivot_table(index=['feature', 'prices', 'decile'], columns='period', values=['snapshots', 'win_minus_price_pts'], aggfunc='first')
W.columns = [f'{a} {b}' for a, b in W.columns]; W = W.reset_index()
pd.set_option('display.width', 250); print(W.to_string())
with pd.ExcelWriter('work/three_vol_deciles.xlsx') as w:
    W.to_excel(w, sheet_name='Deciles_wide', index=False); R.to_excel(w, sheet_name='Deciles_long', index=False)
