import pandas as pd, numpy as np
W = '/mnt/project-files/strategy/futures_flow/futures_flow_with_models.xlsx'
sp = pd.read_parquet('work/wm_t1_splits.parquet'); gr = pd.read_parquet('work/wm_t1_grids.parquet'); se = pd.read_parquet('work/wm_t1_settle.parquet')
pb = pd.read_parquet('work/wm_t1_passbar.parquet'); sh = pd.read_parquet('work/wm_t1_shuffle.parquet')
rules = pd.read_parquet('work/wm_t2_rules.parquet'); br = pd.read_parquet('work/wm_t2_brier.parquet'); co = pd.read_parquet('work/wm_t2_coefs.parquet')
lk = pd.read_parquet('work/wm_t2_lockin.parquet'); lg = pd.read_parquet('work/wm_t2_logit_bots.parquet'); T = pd.read_parquet('work/wm_trades.parquet')
readme = pd.DataFrame({'note': [
 'Futures flow combined with our models (setup: futures-flow-with-models-setup.md). Built 2026-11-14.',
 'Flow = CME @BTC/@MBT tick-rule order flow (5x@BTC + 0.1x@MBT). nvz_comb_W = net signed volume over W minutes / trailing 24-h SD; vi = volume imbalance; ti = trade-count imbalance; svi_X = seconds imbalance over X s.',
 'agrees = flow points to the side bought (|z|>0.5; vi/ti: |value| above the Dec-Aug median); against = points the other way; neutral = weak flow, CME closed, or <10 futures trades.',
 'Band bots (hourly, YES = BTC stays inside band): direction = toward the band centre. quiet_* variants: quiet flow = agrees, strong flow either way = against.',
 'Edge = win% - price - Kalshi taker fee, cents per contract. edge_day_median = median over days of the day mean edge (per-trade median is 0/1).',
 'Dec-Aug for the Direct/Correction gap rules is IN-SAMPLE for those models (fit Dec-Aug); use the agrees-vs-against comparison, not the level.',
 'Sheets: t1_* = Test 1 (filter); grid_<bot> = minute to settlement x 10c price band x YES/NO x period for agrees and against (primary nvz_comb_3); settle_* = seconds-left cuts for Break and Lock-in; t2_* = Test 2 (model input).',
 'Trades after 2026-10-26 (end of the BRTI file) dropped: 228. Lock-in Dec-Aug = Aug only.']})
with pd.ExcelWriter(W) as xw:
    readme.to_excel(xw, sheet_name='README', index=False)
    cols = ['bot', 'variant', 'period', 'split', 'n', 'avg_price', 'win_pct', 'win_minus_price', 'edge_avg', 'edge_day_median', 'days', 'months_pos', 't_stat']
    sp[sp.variant == 'nvz_comb_3'][cols].round(2).to_excel(xw, sheet_name='t1_splits_primary', index=False)
    sp[cols].round(2).to_excel(xw, sheet_name='t1_splits_all_variants', index=False)
    pb.round(3).to_excel(xw, sheet_name='t1_passbar', index=False)
    sh.round(4).to_excel(xw, sheet_name='t1_shuffle', index=False)
    gcols = ['bot', 'split', 'period', 'side', 'band10', 'min_left', 'n', 'avg_price', 'win_pct', 'win_minus_price', 'edge_avg', 'edge_day_median', 'days', 'months_pos']
    g3 = gr[gr.variant == 'nvz_comb_3']
    for bot, g in g3.groupby('bot'):
        g.sort_values(['split', 'period', 'side', 'band10', 'min_left'])[gcols].round(2).to_excel(xw, sheet_name=f'grid_{bot[:24]}', index=False)
    gr[gcols + ['variant']].round(2).to_excel(xw, sheet_name='grids_all_variants', index=False)
    scols = ['bot', 'variant', 'split', 'period', 'side', 'band10', 'secs_bucket', 'n', 'avg_price', 'win_pct', 'win_minus_price', 'edge_avg', 'edge_day_median', 'months_pos']
    se[scols].round(2).to_excel(xw, sheet_name='settle_minute_cuts', index=False)
    rules.round(3).to_excel(xw, sheet_name='t2_gap_rules_rerun', index=False)
    br.round(6).to_excel(xw, sheet_name='t2_brier', index=False)
    co.round(4).to_excel(xw, sheet_name='t2_fit_coefs', index=False)
    lk.round(3).to_excel(xw, sheet_name='t2_lockin', index=False)
    lg.round(4).to_excel(xw, sheet_name='t2_logit_all_bots', index=False)
    T.drop(columns=[c for c in T.columns if c.startswith(('vi_comb', 'ti_comb', 'tr_comb', 'snv', 'str'))]).to_excel(xw, sheet_name='trades_tagged', index=False)
print('wrote', W)
