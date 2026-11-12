"""Stage 1: HAR-RV vs s100 / EWMA / ATR14 / persistence. QLIKE + RMSE, frozen and expanding fits."""
import numpy as np, pandas as pd, sys, warnings
from scipy.stats import kendalltau
warnings.filterwarnings('ignore')
W = sys.argv[1]; OUT = sys.argv[2]
D = pd.read_parquet(W + '/vol_minutes.parquet')
D = D[D.month >= '2025-12'].copy()
MONTHS = sorted(D.month.unique()); SO = ['2026-09', '2026-10']; DA = [m for m in MONTHS if m < '2026-09']
EPS = 1e-14
def L(x): return np.log(np.maximum(x, EPS))
def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X]); b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ b; return b, res.var()
def pred(X, b): return np.column_stack([np.ones(len(X)), X]) @ b
SPECS = {  # name -> list of log-RV regressors
    'HAR3 (15m,1h,24h)': ['rv15', 'rv60', 'rv1440'],
    'HAR4 (+4h)': ['rv15', 'rv60', 'rv240', 'rv1440'],
    'HAR2 (15m,1h)': ['rv15', 'rv60'],
}
def forecasts(fit, app, h):
    """fit/app: DataFrames. returns dict name -> variance forecast on app rows, for horizon h minutes."""
    y = app[f'fwd{h}'].to_numpy(); F = {}
    F['s100 (live input)'] = h * app.s100.to_numpy() ** 2
    F['EWMA 0.97'] = h * app.ewma.to_numpy()
    yf = fit[f'fwd{h}'].to_numpy()
    c = np.nanmean(yf / fit.atr14.to_numpy() ** 2); F['ATR14 x scale'] = c * app.atr14.to_numpy() ** 2
    base = app[f'rv{h}'].to_numpy() if h in (15, 60) else app.rv15.to_numpy() * h / 15
    F['last-window RV'] = base
    c = np.nanmean(yf / (h * fit.s100.to_numpy() ** 2)); F['s100 x scale'] = c * h * app.s100.to_numpy() ** 2
    for name, cols in SPECS.items():
        Xf = np.column_stack([L(fit[c].to_numpy()) for c in cols]); b, s2 = ols(Xf, L(yf))
        Xa = np.column_stack([L(app[c].to_numpy()) for c in cols])
        F[name] = np.exp(pred(Xa, b) + s2 / 2)
    # linear HAR on levels
    cols = SPECS['HAR3 (15m,1h,24h)']
    b, _ = ols(np.column_stack([fit[c].to_numpy() for c in cols]), yf)
    F['HAR3 linear'] = np.maximum(pred(np.column_stack([app[c].to_numpy() for c in cols]), b), EPS)
    return F
def qlike(y, f): return y / f - np.log(y / f) - 1
rows = []
for h in (15, 60):
    need = ['rv15', 'rv60', 'rv240', 'rv1440', 's100', 'ewma', 'atr14', f'fwd{h}']
    X = D.dropna(subset=need); X = X[X[f'fwd{h}'] > 0]
    fit = X[X.month.isin(DA)]
    # frozen: fit Dec-Aug, apply everywhere (Dec-Aug in-sample, Sep-Oct OOS)
    Ffro = forecasts(fit, X, h)
    # expanding: for each month Jan.., fit on earlier months
    Fexp = {k: np.full(len(X), np.nan) for k in Ffro}
    mon = X.month.to_numpy()
    for M in MONTHS[1:]:
        f = X[X.month < M]; a = X[X.month == M]
        if len(a) == 0: continue
        Fm = forecasts(f, a, h)
        for k in Fm: Fexp[k][mon == M] = Fm[k]
    y = X[f'fwd{h}'].to_numpy()
    step = h
    for scheme, FF in (('frozen', Ffro), ('expanding', Fexp)):
        for k, f in FF.items():
            ok = np.isfinite(f) & (f > 0)
            for M in MONTHS:
                i = ok & (mon == M)
                if i.sum() == 0: continue
                rows.append(dict(h=h, scheme=scheme, model=k, month=M, n=int(i.sum()), qlike=qlike(y[i], f[i]).mean(),
                                 rmse=np.sqrt(((np.sqrt(y[i]) - np.sqrt(f[i])) ** 2).mean()) * 1e4,
                                 bias=np.mean(np.sqrt(f[i])) / np.mean(np.sqrt(y[i]))))
            for per, ms in (('Dec-Aug', DA), ('Sep-Oct', SO), ('Jan-Oct OOS', MONTHS[1:])):
                i = ok & np.isin(mon, ms)
                if scheme == 'expanding' and per == 'Dec-Aug': i = ok & np.isin(mon, DA[1:])
                sub = np.where(i)[0][::step]
                tau = kendalltau(f[sub], y[sub]).statistic
                rows.append(dict(h=h, scheme=scheme, model=k, month=per, n=int(i.sum()), qlike=qlike(y[i], f[i]).mean(),
                                 rmse=np.sqrt(((np.sqrt(y[i]) - np.sqrt(f[i])) ** 2).mean()) * 1e4,
                                 bias=np.mean(np.sqrt(f[i])) / np.mean(np.sqrt(y[i])), tau=tau))
    # save the forecasts for stage 2 (frozen)
    pd.DataFrame({'t': X.t.to_numpy(), **{k: v for k, v in Ffro.items()}}).to_parquet(f'{W}/fc_h{h}_frozen.parquet')
    pd.DataFrame({'t': X.t.to_numpy(), **{k: v for k, v in Fexp.items()}}).to_parquet(f'{W}/fc_h{h}_expanding.parquet')
    # coefficients (frozen)
    cols = SPECS['HAR3 (15m,1h,24h)']
    b, s2 = ols(np.column_stack([L(fit[c].to_numpy()) for c in cols]), L(fit[f'fwd{h}'].to_numpy()))
    print(f'h={h} frozen HAR3 coefs (const, 15m, 1h, 24h):', np.round(b, 3), 'resid var', round(s2, 3))
R = pd.DataFrame(rows)
R.to_parquet(OUT + '/stage1_scores.parquet')
pd.set_option('display.width', 200)
for h in (15, 60):
    for scheme in ('frozen', 'expanding'):
        S = R[(R.h == h) & (R.scheme == scheme) & R.month.isin(['Dec-Aug', 'Sep-Oct', 'Jan-Oct OOS'])]
        print(f'\n=== next {h} min, {scheme} fit ===')
        print(S.pivot(index='model', columns='month', values='qlike').round(4).join(
              S.pivot(index='model', columns='month', values='rmse').round(2), rsuffix=' rmse').join(
              S.pivot(index='model', columns='month', values='tau').round(3), rsuffix=' tau').sort_values('Sep-Oct'))
        M = R[(R.h == h) & (R.scheme == scheme) & R.month.str.startswith('20')]
        P = M.pivot(index='month', columns='model', values='qlike')
        win = (P.lt(P['s100 (live input)'], axis=0)).sum()
        print('months beating s100 on QLIKE (of', len(P), '):'); print(win.sort_values(ascending=False).to_string())
        print('month-by-month QLIKE:'); print(P[['s100 (live input)', 'EWMA 0.97', 'ATR14 x scale', 'last-window RV', 'HAR3 (15m,1h,24h)', 'HAR4 (+4h)']].round(4).to_string())
