# Correction Unleashed ML: test scripts (2026-11-17)

Run from `research/ml_replay/` after the replay pipeline: `build_brti.py`, `build_snaps.py`, `audit.py`, `replay.py`,
`baselines.py`, `features.py` (outputs in `work/`, not committed).

1. `run_unleashed.py`: Correction ML (BRTI features + Kalshi mid, spread, volume; no quote ages), monthly out-of-fold for Mar-Aug,
   one frozen Dec-Aug fit for Sep-Oct. `shipped.py` replaces the Sep-Oct predictions with the shipped
   `model_gap_correction.json` (mean absolute difference from the retrained fit about 1 point).
2. `analyze.py`: both sides at ask + fee, predicted return (p - cost) / cost, first qualifying look per contract.
3. `tables.py`, `export.py`: model test (Brier by price, minute, side), deciles of predicted return, bot results by
   minute x 10c bucket x side x period, isolation cuts, workbook.
4. `last_minute.py`, `lm_tables.py`: 45/30/15 s left. Extrapolation, the model was trained on 14 to 1 minutes left only.
Results: /mnt/project-files/strategy/correction_unleashed/.
