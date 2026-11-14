import pandas as pd, numpy as np
OUT = '/mnt/project-files/strategy/futures_flow/futures_flow_test.xlsx'
def rd(n): return pd.read_parquet(f'work/{n}.parquet')
sc = pd.concat([rd('screen_15m'), rd('screen_hourly')]); br = pd.concat([rd('brier_15m'), rd('brier_hourly')])
dec = pd.concat([rd('deciles_15m'), rd('deciles_hourly')]); tr = pd.concat([rd('trades_15m'), rd('trades_hourly')]); cov = pd.concat([rd('coverage_15m'), rd('coverage_hourly')])
ab = pd.concat([rd('absorbed_15m'), rd('absorbed_hourly')]); ss = rd('seconds_screen'); st = rd('seconds_trades')
# pass-bar table
rows = []
for prod in ['15m', 'hourly']:
    s = sc[sc['product'] == prod]; b = br[br['product'] == prod]; t = tr[tr['product'] == prod].copy(); t['months_n'] = t.months_pos.str.split('/').str[0].astype(int)
    idx = ['signal', 'rule', 'k', 'side', 'band10', 'tgroup']
    j = t[t.period == 'Dec-Aug'].set_index(idx).join(t[t.period == 'Sep-Oct'].set_index(idx), lsuffix='_A', rsuffix='_B', how='inner')
    cells_pass = int(((j.edge_after_fee_c_A >= 1.5) & (j.months_n_A >= 7) & (j.t_A >= 2) & (j.edge_after_fee_c_B >= 1.0) & (j.trades_B >= 300)).sum())
    p = s[s.signal == 'nvz_comb_3'].iloc[0]; pb = b[b.signal == 'nvz_comb_3'].iloc[0]
    rows += [dict(product=prod, test='Direction: AUC >= 0.53 both periods (primary)', result=f"{p['auc_Dec-Aug']:.3f} / {p['auc_Sep-Oct']:.3f}", variants_passing=int(((s['auc_Dec-Aug'] >= 0.53) & (s['auc_Sep-Oct'] >= 0.53)).sum()), of=len(s), passed='NO'),
             dict(product=prod, test='Residual: flow coef same sign, p<0.01 both periods', result=f"coef {p['coef_Dec-Aug']:+.3f} (p {p['p_Dec-Aug']:.2f}) / {p['coef_Sep-Oct']:+.3f} (p {p['p_Sep-Oct']:.4f})",
                  variants_passing=int(((np.sign(s['coef_Dec-Aug']) == np.sign(s['coef_Sep-Oct'])) & (s['p_Dec-Aug'] < 0.01) & (s['p_Sep-Oct'] < 0.01)).sum()), of=len(s), passed='NO (primary); hourly 6 secondary variants pass this line only'),
             dict(product=prod, test='Brier: Kalshi+flow beats Kalshi by >= 0.0005, CI below 0 (Sep-Oct)', result=f"{pb.diff_vs_kalshi:+.5f} [{pb.ci_lo:+.5f}, {pb.ci_hi:+.5f}]; vs recalibrated Kalshi {pb.diff_vs_kalshi_recal:+.5f}",
                  variants_passing=int(((b.diff_vs_kalshi <= -0.0005) & (b.ci_hi < 0)).sum()), of=len(b), passed='NO'),
             dict(product=prod, test='Trading: cell >= +1.5c Dec-Aug (7/9 mo, t>=2) and >= +1.0c Sep-Oct (n>=300)', result=f'{cells_pass} cells pass', variants_passing=cells_pass, of=len(j), passed='NO')]
pb = pd.DataFrame(rows)
readme = pd.DataFrame({'line': [
 'BTC futures order flow (CME @BTC + @MBT, TradeStation tick-rule up/down volume) as a predictor of Kalshi KXBTC15M and KXBTCD outcomes. Run 2026-11-14.',
 'Setup: strategy/futures_flow/futures-flow-test-setup.md (written and approved before the run). Scripts: repo research/futures_flow/.',
 'RESULT: FAIL on all four pre-set pass-bar lines, both products. Kalshi already prices the futures flow: Kalshi mid moves with the flow (deciles), the subsequent BRTI move does not (AUC 0.49-0.50).',
 'Signals: nvz = net signed volume (BTC units, 5x@BTC+0.1x@MBT) / trailing 24h SD; vi = (upvol-downvol)/(upvol+downvol); ti = same on trade counts. Suffix = symbol set _ window minutes. Primary = nvz_comb_3.',
 'Snapshots: 15m = whole minutes 1-14 left, fresh quotes (<20 s); hourly = 1-min candle ask/bid, 1-30 min left. Skipped when CME closed or <10 futures trades in the window.',
 'Trades: one per contract per rule, first qualifying minute; with flow = buy YES at ask if flow > +k SD, NO at 1-bid if < -k; fade = opposite. Fee ceil(7*p*(1-p)) c. Edge = win% - price - fee.',
 'Periods: Dec 2025-Aug 2026 in-sample, Sep 1-Oct 25 2026 out-of-sample. Seconds test May 14-Aug / Sep-Oct.',
 'Sheets: pass_bar, coverage, screen (tau/AUC/residual coef), brier, deciles (primary), trades_summary, trades_cells, absorbed_flow, seconds_screen, seconds_trades.']})
with pd.ExcelWriter(OUT, engine='openpyxl') as xw:
    readme.to_excel(xw, sheet_name='README', index=False); pb.to_excel(xw, sheet_name='pass_bar', index=False); cov.to_excel(xw, sheet_name='coverage', index=False)
    sc.round(5).to_excel(xw, sheet_name='screen', index=False); br.round(6).to_excel(xw, sheet_name='brier', index=False); dec.round(2).to_excel(xw, sheet_name='deciles', index=False)
    tr[(tr.band10 == 'all')].round(2).to_excel(xw, sheet_name='trades_summary', index=False); tr[(tr.band10 != 'all')].round(2).to_excel(xw, sheet_name='trades_cells', index=False)
    ab.round(2).to_excel(xw, sheet_name='absorbed_flow', index=False); ss.round(4).to_excel(xw, sheet_name='seconds_screen', index=False); st.round(2).to_excel(xw, sheet_name='seconds_trades', index=False)
    for ws in xw.book.worksheets:
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(60, max(10, max(len(str(c.value)) if c.value is not None else 0 for c in col[:200]) + 2))
print(pb.to_string()); print('wrote', OUT)
