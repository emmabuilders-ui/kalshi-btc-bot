# BRTI history (Aug 1 - Oct 25, 2026)

No free public BRTI dataset exists (CF Benchmarks licenses it). These values are BRTI
numbers Kalshi published as KXBTC15M settlement values, taken from the markets.csv in
each kalshi-daily ZIP.

## BRTI_15min_settlements_2026-08-01_to_2026-10-25.csv
- time_utc: a 15-minute mark
- brti_60s_avg: simple average of the 60 BRTI prints in the minute before that mark
  (exactly what Kalshi settles on: a contract's expiration_value, which also equals the
  next contract's strike; 8,141 overlaps match to the cent)
- binance_btcusdt_60s_avg: mean of Binance BTCUSDT 1s closes over the same 60 seconds
- brti_minus_binance: basis. Mean -$42.35, median -$34.12, SD $35.23
- 8,170 marks. Missing marks are Kalshi's Thursday 06:45-09:00 UTC maintenance gaps.

Binance gives the same up/down result as BRTI in 97.8% of consecutive windows
(7,973 of 8,156), so about 1 in 45 contracts would be mis-settled by a Binance proxy.

## BRTI_daily_2026-08-01_to_2026-10-25.csv
Daily open (00:00 UTC mark), high/low over the 15-minute marks, last mark of the day.
This is not an intraday high/low of the true index.

## For finer granularity
Kalshi's authenticated passthrough GET /trade-api/v2/cfbenchmarks/values?id=BRTI serves
historical BRTI (intra-second). It returns 401 without a signed request and needs an
account entitlement.

## BRTI_1sec_2026-08-01_to_2026-10-25.zip (added 2026-10-29)
Real 1-second BRTI from Kalshi's authenticated CF Benchmarks passthrough
(GET /trade-api/v2/cfbenchmarks/history/values?id=BRTI&timespan=HOUR&timestamp=<ISO hour start>,
signed with Michael's read-only key). The endpoint returns 5 ticks per second; these files keep
the tick on each whole second.
- One gzip CSV per UTC day: unix_s,time_utc,brti. 7,429,304 rows.
- 1,096 seconds (about 0.15%) have no print on the whole second, spread over 35 days.
- Check: the mean of the 60 prints from T-60s to T-1s equals Kalshi's settlement value at all
  8,164 fully covered 15-minute marks (6 marks fall on a missing second and are not compared).
- Kalshi notes that values pulled after expiry can reflect later revisions; none showed up here.
