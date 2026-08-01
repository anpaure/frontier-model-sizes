# ECI functional-form tournament audit

Generated 2026-07-31. Status: **RETAIN LIVE FORM**.

## What changed

Fifteen hash-pinned Epoch vintages create 27 first-observed parameter checkpoints. Twenty-five backfills form the selection set; Kimi K2.5 and Kimi K2.7 Code, each released between adjacent archive captures, are reserved as the closest available prospective check.

The inverse-ECI-CI tournament selects `ridge_flexible` on the 25-row selection set. Median error moves from 2.70x to 1.36x and RMSE from 0.495 to 0.364 log10.

On the two interval-prospective checkpoints, mean absolute error moves from 0.166 to 0.163 log10. This tiny validation set is decisive only as a veto, not as proof of a winner.

## Why the central forecast is not changed

Fable and Sol are 5.1 and 6.1 ECI points beyond the strongest open-weight calibrator. Flexible score curves therefore extrapolate rather than interpolate.

- Claude Fable 5: the selected challenger spans 5.49–7.34T across weighting and same-base-collapse sensitivities (1.34x).
- GPT-5.6 Sol: the selected challenger spans 6.43–8.88T across weighting and same-base-collapse sensitivities (1.38x).

The live 60/40 linear blend is retained because the selected flexible challenger fails at least one prospective/stability gate. Movement from a nonlinear curve without this veto would be model-selection and extrapolation noise, not stronger evidence.

## Data integrity

- Frozen archive: `sources/epoch_eci_historical_snapshots_2026-07-18.tar.gz` (0ce76d6662dd34efe9847e418cb0ade2654b320b8df49c998e2f7df989ba2150); terminal capture 2026-07-16 (`20260716153134`).
- Archive collection metadata: `sources/epoch_eci_historical_collection_metadata_2026-07-18.json` (8123f9fbaa0d33ad92d56d3c7ca6608c74a5ecc1fa014fab461d9accfbd0b178)
- Historical score ledger: `sources/epoch_eci_historical_model_scores_2026-07-18.csv` (98b18176aeb56d9b5015c80ab50ed7d27c3bac28e7d699c11968b45dc052c3fa)
- Fixed-code fit metadata: `sources/epoch_eci_historical_fit_metadata_2026-07-18.json` (3547e7e9477dde263c69a83be2325782daec49050f0a8211af0e6e9e5b5e81ce)
- Prediction ledger: `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_fit_tournament_predictions_2026-07-18.csv` (a60f852b1d4304d5d0334dd7335f86501e997f5f2a60632e66754fb1b42623fe)
- Frontier sensitivity ledger: `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_fit_tournament_frontier_sensitivity_2026-07-18.csv` (862f06d3d6d73bde8d726b5efe7ca2accf654ff1b60519c19aaab69dbe7e7437)

The July 31 canonical ECI panel is a separately hashed live-successor reference. It is used by the current frontier model, but it is not an equality target or an input to the frozen historical refit.
