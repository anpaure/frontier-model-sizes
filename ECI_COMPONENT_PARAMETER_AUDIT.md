# ECI component and parameter-signal audit

Snapshot: 2026-07-18

## Decision

No individual ECI benchmark receives a separate forecast weight. The aggregate ECI branch and current frontier centers remain unchanged.

The audit found a real-looking unadjusted signal for OTIS Mock AIME, but it does not survive correction for testing 13 component benchmarks. Its active-parameter median error falls from 2.07× to 1.77× and its unadjusted family-cluster interval is favorable, while the global familywise one-sided p-value is 0.27. Treating that as an independent signal would be benchmark selection after seeing the outcomes.

## Expanded aggregate-ECI panel

The original workbook calibration contains 58 disclosed open checkpoints. The canonical megafile supplies 25 more checkpoints that meet every admission rule:

- a unique exact alphanumeric ECI-to-Epoch checkpoint match;
- high match confidence and checkpoint-level identity;
- exact release-date agreement;
- open weights;
- identical parameter values in the ECI row and Epoch source.

This produces an 83-checkpoint total-parameter panel across 36 families. The original workbook weights are exactly `15.3664 / (ECI confidence-interval width)^2`; the same law is used for all 25 additions.

On all 64 chronologically eligible tests, the expanded panel improves median error from 2.63× to 2.40× and mean absolute log10 error from 0.445 to 0.398. The equal-family bootstrap 90% interval for the improvement is fully favorable. On the narrower 46-test original panel, median error improves from 1.93× to 1.85× but the paired interval crosses zero.

The full-panel frontier estimates are stable: Fable moves from 2.99T to 2.97T and Sol from 3.40T to 3.39T. Every audited frontier ECI center moves by less than 2.5%, so this extension is retained as a robustness check rather than used to create a cosmetic forecast change.

## Active-parameter component audit

The second audit uses the 88-model ECI parameter map, covering 38 developer families, 707 model-by-benchmark measurements, and 48 benchmarks. Thirteen benchmarks have enough data for a held-out comparison.

For each test checkpoint:

1. training rows must have strictly earlier release dates;
2. the test checkpoint's entire family is removed;
3. the baseline predicts log10 active or total parameters from aggregate ECI, exact date, reasoning status, and MoE status;
4. the candidate adds `component-implied ECI − aggregate ECI`;
5. training gives equal total weight to each family and halves the weight of broad-ECI-interval rows;
6. inference uses a global family-level max-T sign-flip correction across all 13 eligible benchmarks with 100,000 permutations.

No benchmark survives the familywise correction. Fable and Sol have OTIS, WeirdML, and/or GPQA component coverage, but none is admitted to the live parameter ensemble.

## Limitations

- Epoch's component table is a current snapshot, not a sequence of historical benchmark vintages.
- Benchmark availability is non-random and strongly correlated with release date.
- Some active-parameter values in the 88-model map are inferred rather than directly disclosed.
- New disclosed checkpoints or a later independently frozen benchmark snapshot can prospectively retest the apparent OTIS and knowledge-benchmark signals.

## Reproducible artifacts

- `run_eci_component_extended_audit.py`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_component_extended_audit_2026-07-18.json`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_component_expanded_parameter_panel_2026-07-18.csv`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_component_expanded_aggregate_predictions_2026-07-18.csv`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_component_active_incremental_comparison_2026-07-18.csv`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_component_active_incremental_predictions_2026-07-18.csv`
- `tests/test_eci_component_extended_audit.py`
