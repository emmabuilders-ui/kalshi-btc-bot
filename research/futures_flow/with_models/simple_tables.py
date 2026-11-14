"""Michael 2026-11-14 23:25Z: pricing buckets x minutes to settle x YES/NO x (all / against / agree): n, avg price, win rate. Spreadsheet.
Universe: every Kalshi snapshot, buy at the ask. Agree = 3-min flow z > +0.5 for YES / < -0.5 for NO; Against = opposite; All = everything."""
import numpy as np, pandas as pd
W = '/mnt/project-files/strategy/futures_flow/futures_flow_price_minute_side_all_against_agree.xlsx'
def agg(g):
    return pd.Series({'n': len(g), 'avg_price': round(g.price.mean(), 1) if len(g) else np.nan, 'win_pct': round(100 * g.won.mean(), 1) if len(g) else np.nan})
readme = pd.DataFrame({'note': [
 'Every Kalshi snapshot (15-min: minutes 1-14 left; hourly: 1-30 left), bought at the ask. n = snapshots (one per contract per minute). Avg price in cents. Win % next to it.',
 'Price bucket = price paid (the ask of the side bought), 10c bands.',
 'Agree = CME futures flow over the last 3 minutes points the way of the side bought (z > +0.5 for YES, z < -0.5 for NO). Against = opposite. All = every snapshot incl. weak flow and CME closed/thin.',
 'Periods: Dec-Aug (Dec 10 2025 - Aug 31 2026) and Sep-Oct (Sep 1 - Oct 26 2026). One sheet per product and price band, plus a summary by band.']})
with pd.ExcelWriter(W) as xw:
    readme.to_excel(xw, sheet_name='README', index=False)
    for prod in ['15m', 'hourly']:
        X = pd.read_parquet(f'work/dec_rows_{prod}.parquet', columns=['period', 'side', 'band10', 'min_left', 'price', 'won', 'nvz_comb_3'])
        z = X.nvz_comb_3.to_numpy(); sgn = np.where(X.side == 'YES', 1, -1)
        X['grp'] = np.where(np.isnan(z), 'none', np.where(np.abs(z) <= 0.5, 'weak', np.where(z * sgn > 0, 'Agree', 'Against')))
        summ = []
        for band in sorted(X.band10.unique()):
            B = X[X.band10 == band]; rows = []
            for (side, ml), g in B.groupby(['side', 'min_left']):
                r = {'min_left': ml, 'side': side}
                for per in ['Dec-Aug', 'Sep-Oct']:
                    gp = g[g.period == per]
                    for nm, sub in [('All', gp), ('Against', gp[gp.grp == 'Against']), ('Agree', gp[gp.grp == 'Agree'])]:
                        a = agg(sub); r[f'{per} {nm} n'] = int(a.n); r[f'{per} {nm} avg price'] = a.avg_price; r[f'{per} {nm} win %'] = a.win_pct
                rows.append(r)
            D = pd.DataFrame(rows).sort_values(['side', 'min_left'], ascending=[False, False])
            D.to_excel(xw, sheet_name=f'{prod}_{band}-{band+9}c', index=False)
        for (side, band, per), g in X.groupby(['side', 'band10', 'period']):
            r = {'side': side, 'price band': f'{band}-{band+9}c', 'period': per}
            for nm, sub in [('All', g), ('Against', g[g.grp == 'Against']), ('Agree', g[g.grp == 'Agree'])]:
                a = agg(sub); r[f'{nm} n'] = int(a.n); r[f'{nm} avg price'] = a.avg_price; r[f'{nm} win %'] = a.win_pct
            summ.append(r)
        pd.DataFrame(summ).sort_values(['side', 'period', 'price band'], ascending=[False, True, True]).to_excel(xw, sheet_name=f'{prod}_summary', index=False)
print('ok')
