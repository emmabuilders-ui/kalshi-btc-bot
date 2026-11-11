import numpy as np, pandas as pd
T0=1764460800
B=pd.Series(np.load('work/brti.npy')).ffill().bfill().to_numpy(); lB=np.log(B)
L=np.zeros(len(B)); L[1:]=np.diff(lB)
c1=np.concatenate([[0],np.cumsum(L)]); c2=np.concatenate([[0],np.cumsum(L**2)])
rmax={w:pd.Series(lB).rolling(w,min_periods=1).max().to_numpy() for w in (900,3600,86400)}
rmin={w:pd.Series(lB).rolling(w,min_periods=1).min().to_numpy() for w in (900,3600,86400)}
SHR=40
def sdv(E,w):
    a=E-w; s1=c1[E]-c1[a]; s2=c2[E]-c2[a]; return np.sqrt(np.maximum(s2/w-(s1/w)**2,1e-14))
def ncdf(x):
    from scipy.special import ndtr; return ndtr(x)
def features(snap_ts, close_ts, lo, hi):
    """snap_ts,close_ts unix sec arrays; lo/hi strike arrays (hi NaN for above/below). Uses BRTI strictly up to second before snap."""
    e=(snap_ts-T0).astype(np.int64)-1  # last whole second known
    E=e+1; spot=B[e]; secs=(close_ts-snap_ts).astype(float)
    F={'secs_left':secs,'spot':spot}
    v={w:sdv(E,w) for w in (60,300,900,3600)}
    for w in v: F[f'vol_{w}']=v[w]
    Teff=np.clip(secs-SHR,1,None)
    sd=spot*v[900]*np.sqrt(Teff)
    F['dist_lo']=np.log(spot/lo)
    F['z_lo']=(spot-lo)/np.maximum(sd,1e-9)
    p_lo=ncdf(F['z_lo'])
    has_hi=~np.isnan(hi)
    hi2=np.where(has_hi,hi+0.01,1.0)
    F['dist_hi']=np.where(has_hi,np.log(spot/hi2),0.0)
    F['z_hi']=np.where(has_hi,(spot-hi2)/np.maximum(sd,1e-9),np.nan)
    p_hi=np.where(has_hi,ncdf(np.nan_to_num(F['z_hi'])),0.0)
    F['p_sa']=np.clip(p_lo-p_hi,0,1)
    F['inside']=np.where(has_hi,((spot>=lo)&(spot<=hi2)).astype(float),(spot>=lo).astype(float))
    F['abs_dist_edge']=np.where(has_hi,np.minimum(np.abs(F['dist_lo']),np.abs(F['dist_hi'])),np.abs(F['dist_lo']))
    F['z_edge']=F['abs_dist_edge']/np.maximum(v[900]*np.sqrt(Teff),1e-9)
    for l in (10,30,60,300,900,1800,3600): F[f'ret_{l}']=lB[e]-lB[e-l]
    for w in (900,3600,86400):
        F[f'from_hi_{w}']=lB[e]-rmax[w][e]; F[f'from_lo_{w}']=lB[e]-rmin[w][e]
    F['volratio_60_900']=v[60]/v[900]; F['volratio_300_3600']=v[300]/v[3600]
    F['prior_hr_vol_pct']=v[3600]*np.sqrt(3600)*100
    t=pd.to_datetime(snap_ts,unit='s'); F['hour']=t.hour.to_numpy(); F['dow']=t.dayofweek.to_numpy()
    return pd.DataFrame(F).replace([np.inf,-np.inf],np.nan)
