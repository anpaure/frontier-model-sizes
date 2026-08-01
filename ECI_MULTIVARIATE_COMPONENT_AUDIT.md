# Multivariate ECI component audit

Date: 2026-07-18
Live incremental weight: **0%**

## Outcome

A regularized combination of individual ECI benchmark residuals is directionally promising, but it does not meet the evidentiary threshold for a live parameter-count factor.

| Chronological developer-held-out test | Aggregate ECI baseline | Multivariate components | Equal-family 90% interval |
|---|---:|---:|---:|
| Primary panel, 72 tests / 32 families | 2.28× median error | **1.99×** | −0.052 to +0.002 log10 MAE |
| Narrow-ECI-CI outcomes within primary run, 42 / 20 | 2.28× | **2.21×** | −0.057 to +0.001 |
| Narrow-ECI-CI-only training and tests, 29 / 10 | **2.00×** | 2.04× | −0.111 to +0.047 |
| Frontier-like primary folds, 9 / 5 | 2.71× | **2.43×** | −0.095 to −0.017 |

The primary point estimate improves, but the narrow-ECI-CI-only replication is slightly worse. Both main intervals cross zero, and the frontier result comes from only five families. These results justify collecting more component data, not changing the live forecast.

## Leakage controls

Every outer prediction:

1. uses only checkpoints with a strictly earlier exact release date;
2. removes the test checkpoint's entire developer family;
3. recomputes benchmark eligibility and robust median/MAD scaling from training data only;
4. selects the feature class and ridge penalty through leave-one-family-out validation inside the outer training set;
5. drops folds with no honestly tunable policy rather than substituting a default.

The three fixed feature classes are all eligible benchmarks, a pretraining-like subset, and a narrower knowledge subset. Aggregate ECI, exact date, reasoning status, and MoE status remain unpenalized baseline covariates; ridge shrinkage applies only to component residuals. Families receive equal total weight. Checkpoints whose aggregate ECI confidence interval exceeds 10 points receive half weight in the primary run and are excluded entirely in the narrow-ECI-CI-only replication. This is score-uncertainty metadata, not parameter-disclosure metadata; the underlying panel does not classify parameter disclosure.

## Target-applicability failure

Target fitting now follows the same chronology and family-holdout rule as the backtest. This corrects an earlier exploratory implementation that used the full panel for target sensitivity.

| Target | Full-training adjustment | Narrow-ECI-CI-only adjustment | Observed selected components | Applicability finding |
|---|---:|---:|---:|---|
| Claude Fable 5 | 1.05× | 0.80× | 0 full / 4 narrow-CI | Directions disagree; the full policy has no observed selected benchmark |
| GPT-5.6 Sol | 0.52× | 0.29× | 2 / 2 | Direction agrees, but the magnitude is large and training-scope sensitive |

Fable therefore fails both the minimum target-component coverage gate and the full-versus-narrow-CI direction gate. Sol does not rescue the branch: its implied ECI adjustment is too dependent on which ECI-uncertainty rows are admitted. Both sensitivities remain visible with 0% weight and do not alter the 4.5T Fable or 3.1T Sol headline forecasts.

## Data audit

- 88 unique ECI parameter-map checkpoints across 38 families;
- 42 narrow-ECI-CI and 46 broad-ECI-CI checkpoints;
- zero parameter-disclosure classifications inferred from this flag;
- 707 unique checkpoint/benchmark measurements;
- 48 component benchmarks after exact identity intersection;
- no duplicate checkpoint/benchmark keys;
- every admitted checkpoint retains at least one component measurement;
- source hashes are recorded and tested.

The component measurements are from the current Epoch snapshot, not release-vintage historical snapshots. Benchmark availability is non-random and the residuals are correlated with aggregate ECI, so even a future supported branch should receive only a small incremental weight.

## Reproducible artifacts

- `analyze_eci_multivariate_components.py`
- `tests/test_eci_multivariate_components.py`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_multivariate_component_audit_2026-07-18.json`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_multivariate_component_predictions_2026-07-18.csv`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_multivariate_component_narrow_eci_ci_predictions_2026-07-18.csv`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_multivariate_component_targets_2026-07-18.csv`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_multivariate_component_coverage_2026-07-18.csv`
