"""Per minute-to-settlement breakdown of the primary signal (nvz_comb_3): AUC, residual coef, and with-flow trades at |z|>1.5."""
import numpy as np, pandas as pd, statsmodels.api as sm
from analyze import auc, fee, logit
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
out = []
for prod, maxm in [('15m', 14), ('hourly', 30)]:
    S = pd.read_parquet(f'work/merged_{prod}.parquet', columns=['ticker', 'secs_left', 'ask', 'bid', 'mid', 'res', 'move', 'up', 'period', 'month', 'nvz_comb_3'])
    S = S[S.nvz_comb_3.notna()].copy(); S['ml'] = (S.secs_left // 60).astype(int); S['pk'] = S.mid.clip(0.01, 0.99)
    for ml in range(1, maxm + 1):
        for per in ['Dec-Aug', 'Sep-Oct']:
            d = S[(S.ml == ml) & (S.period == per)]
            r = dict(product=prod, min_left=ml, period=per, n=len(d), auc=auc(d.nvz_comb_3.to_numpy(), d.up.to_numpy()))
            try:
                f = sm.Logit(d.res.astype(float).to_numpy(), sm.add_constant(np.column_stack([logit(d.pk.to_numpy()), np.clip(d.nvz_comb_3.to_numpy(), -6, 6)]))).fit(disp=0)
                r['coef'] = f.params[2]; r['p'] = f.pvalues[2]
            except Exception: r['coef'] = r['p'] = np.nan
            h = d[d.nvz_comb_3.abs() > 1.5]; yes = h.nvz_comb_3 > 0
            price = np.where(yes, h.ask, 1 - h.bid); win = np.where(yes, h.res == 1, h.res == 0).astype(float); pnl = win - price - fee(price)
            r.update(trades=len(h), avg_price=100 * price.mean(), win_pct=100 * win.mean(), win_minus_price=100 * (win.mean() - price.mean()), edge_after_fee_c=100 * pnl.mean(), t=pnl.mean() / (pnl.std() / np.sqrt(len(h))) if len(h) > 2 else np.nan)
            m = pd.Series(pnl).groupby(h.month.to_numpy()).mean(); r['months_pos'] = f'{(m > 0).sum()}/{len(m)}'
            out.append(r)
R = pd.DataFrame(out); R.to_parquet('work/per_minute.parquet'); print(R.round(3).to_string())
# add sheet to workbook
from openpyxl import load_workbook
p = '/mnt/project-files/strategy/futures_flow/futures_flow_test.xlsx'
with pd.ExcelWriter(p, engine='openpyxl', mode='a', if_sheet_exists='replace') as xw: R.round(3).to_excel(xw, sheet_name='per_minute', index=False)
