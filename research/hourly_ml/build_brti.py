import glob, zipfile, io, numpy as np, pandas as pd
T0 = 1764460800  # 2025-11-30 00:00 UTC
end = int(pd.Timestamp('2026-11-01', tz='UTC').timestamp()); n = end - T0
arr = np.full(n, np.nan)
def put(df):
    s = df['unix_s'].to_numpy().astype(np.int64) - T0; ok = (s >= 0) & (s < n); arr[s[ok]] = df['brti'].to_numpy()[ok]
for f in sorted(glob.glob('/mnt/project-files/data-download/brti/1sec_2025-11-30_to_2026-07-31/*.csv.gz')): put(pd.read_csv(f))
z = zipfile.ZipFile('/mnt/project-files/data-download/brti/BRTI_1sec_2026-08-01_to_2026-10-25.zip')
for nm in sorted(z.namelist()):
    if nm.endswith('.gz'): put(pd.read_csv(io.BytesIO(z.read(nm)), compression='gzip'))
put(pd.read_csv('/mnt/project-files/data-download/kalshi-hourly-range/brti_1s_2026-10-26_to_10-31.csv.gz'))
np.save('work/brti.npy', arr); print('n', n, 'missing', int(np.isnan(arr).sum()), round(np.isnan(arr).mean()*100, 3), '%')
