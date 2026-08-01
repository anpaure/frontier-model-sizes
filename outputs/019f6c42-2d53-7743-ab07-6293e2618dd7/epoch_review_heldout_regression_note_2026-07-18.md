# Held-out parameter-count regression and ECI component audit

Date: 2026-07-18  
Target: `log10(disclosed total parameters in billions)`  
Primary question: do individual ECI benchmarks predict total parameter count better than aggregate ECI after controlling for release date and model family?

## Bottom line

The reviewer's proposed validation standard is correct, and the present pipeline now implements it. Every outer-fold prediction is trained only on strictly earlier releases, with the test model's entire family removed. Component benchmarks and aggregate ECI are evaluated on identical train/test rows.

No individual benchmark or predeclared composite robustly beats aggregate ECI. The live 58-model ECI calibration does contain out-of-sample signal—its strict chronological family-holdout median error is 1.93×—but its 80th-percentile error is 4.21×. This supports using ECI as one probabilistic branch, not treating it as a precise parameter counter.

The broader current-like evidence ensemble previously achieved a 1.98× median held-out error on the 37 folds with available components, compared with 2.47× for ECI alone in that broader panel. Because the live crowd cannot yet be scored against undisclosed ground truth, it is not included in that backtest.

## Data inventory

- Official Epoch component snapshot: 2,010 unique model/benchmark rows, 211 models, 52 benchmarks.
- Exact live open-weight calibration: 58 models with disclosed total parameters.
- Matched component calibration panel: 502 measurements, all 58 models, 47 benchmarks.
- Unified megafile: 7,817 observations and 19,439 long-form measurements.
- Structural audit: zero duplicate observation IDs, duplicate measurement IDs, orphan measurements, source-record JSON errors, or raw-record mismatches—including all 2,010 component rows.

## Validation design

For each held-out model:

1. Keep only models with an earlier release date.
2. Remove every model in the held-out model's family.
3. Require at least 12 training models and five training families.
4. Convert component performance to the ECI latent scale using Epoch's published benchmark difficulty and slope: `difficulty + logit(performance) / slope`.
5. Fit parameter count in log space and pool 60% score-only with 40% score-plus-date.
6. Compare the component and aggregate ECI on exactly the same fold.
7. Bootstrap paired absolute-error differences by model family; only a predeclared or independently replicated panel with a 90% family-cluster interval below zero is eligible for promotion.

This is pseudo-chronological because the current benchmark snapshot is used for all folds. It validates cross-model mapping and temporal extrapolation, not a fully vintage real-time forecast.

## Component results

Median multiplicative error; lower is better.

| Panel | Held-out n | Component | Aggregate ECI | Result |
|---|---:|---:|---:|---|
| MMLU | 19 | 2.01× | 1.65× | Component does not improve |
| Pretraining-like composite | 13 | 2.44× | 2.31× | Statistically tied; no promotion |
| Knowledge composite | 22 | 2.41× | 2.17× | Statistically tied; no promotion |
| WeirdML | 9 | 2.55× | 2.40× | Component significantly worse |
| GPQA Diamond | 24 | 3.42× | 2.20× | Component significantly worse |
| MATH Level 5 | 12 | 3.74× | 3.19× | Component significantly worse |
| OTIS Mock AIME | 16 | 3.85× | 2.64× | Component lost all 16 paired folds |

ARC AI2 had a superficially better median (2.03× versus 2.41×) but worse RMSE, only nine held-out models, and a family-bootstrap interval crossing zero. It is not robust evidence of improvement.

The component-derived Fable/Sol estimates are deliberately marked exploratory and excluded from the forecast mixture. The qualifying target benchmarks are precisely those that backtest worse than ECI, so promoting their low implied counts would be selection on noise.

## RL/post-training compute

Epoch coverage is currently insufficient for the requested direct RL-to-pretraining-compute ratio regression:

| Coverage among matched open models | Models |
|---|---:|
| Training compute | 45 |
| Finetune compute | 6 |
| Post-training compute | 1 |
| Both training and finetune compute | 4 |
| Both training and post-training compute | 1 |

A same-family/same-size sensitivity diagnostic is therefore used instead. Median reasoning-model uplift on the latent ECI scale is approximately 0.4 points for WeirdML, 3.9 for GPQA, 8.6 for OTIS AIME, and 11.6 for MATH Level 5, albeit from only one or two mixed reasoning groups per benchmark. This is consistent with math/reasoning scores being much more post-training and inference-time-compute sensitive than pretraining scale.

## Forecast update after the final anonymous respondent batch

Respondent R21's Fable 5–10T and Sol 3–8T intervals are recorded with geometric interval centers of 7.1T and 4.9T. They increase the crowd pools to 20 Fable forecasts and 19 Sol forecasts. The repository retains no name-to-ID mapping.

| Model | Evidence model | Crowd center | 50/50 log pool |
|---|---:|---:|---:|
| Claude Fable 5 | 4.7T | 4.4T | **4.5T** |
| GPT-5.6 Sol | 3.1T | 3.2T | **3.2T** |

## Decision

- Retain aggregate ECI rather than substituting a cherry-picked component benchmark.
- Do not fit a direct RL/pretraining-compute coefficient until materially more than one complete observation exists.
- Keep the component analysis in the automatic pipeline and rerun it whenever the Epoch snapshot changes.
- Treat approximately 2× median held-out error as the current empirical resolution of the parameter model; narrower uncertainty would not be evidence-backed.

## Direct factor-weight optimization

The available AA, ECI, no-CoT, and compute weights were also optimized directly in log-parameter space. This cannot include API price or the human crowd: comparable provider-price observations are concentrated in closed models without disclosed parameter counts, while Fable and Sol remain undisclosed.

Coverage is sparse and unbalanced. Among 37 matched held-out checkpoints, AA appears in 7, ECI in 32, no-CoT in 4, and compute in 33; 26 rows contain only ECI and compute.

An in-sample squared-error optimizer chooses 26% AA, 8% ECI, 50% no-CoT, and 16% compute. This reduces RMSE from 0.439 to 0.413 log10 points but worsens median error from 1.98× to 2.53×. Changing the objective produces radically different weights: optimizing median error chooses 24% AA, 66% ECI, 6% no-CoT, and 4% compute.

The decisive test is nested. For each outer checkpoint, weights are learned only from earlier already-held-out predictions, with the outer checkpoint's family removed. On the 25 eligible outer folds:

| Weight rule | Median error | RMSE log10 | P80 error | Within 2× |
|---|---:|---:|---:|---:|
| Current judgmental weights | **1.98×** | **0.420** | 4.10× | **52%** |
| Equal available weights | 2.05× | 0.419 | **3.69×** | 48% |
| Nested MSE-optimized | 2.60× | 0.496 | 4.66× | 40% |
| Nested MAE-optimized | 2.53× | 0.502 | 4.15× | 44% |

The nested MSE optimizer increases mean absolute log error by 0.075 versus the current weights; a family-cluster bootstrap gives a 90% interval of +0.018 to +0.137 and only a 1.7% probability that optimization is better. Family-bootstrap intervals for individual globally optimized weights span most of the simplex—for example AA 0–82%, ECI 0–78%, and no-CoT 2–94%—confirming weak identification.

Therefore the live weights were not replaced. Direct optimization currently overfits the small, missing-not-at-random overlap panel.

## Reproducible outputs

- `eci_component_chronological_backtest_2026-07-18.json`: full method, inventory, metrics, hashes, and promotion result.
- `eci_component_benchmark_comparison_2026-07-18.csv`: one row per eligible component/composite comparison.
- `eci_component_backtest_predictions_2026-07-18.csv`: all outer-fold predictions and training cutoffs.
- `eci_component_same_base_sensitivity_2026-07-18.csv`: same-family/same-size post-training diagnostic.
- `unified_model_observations_compute_enriched_2026-07-17.csv` and `unified_model_measurements_long_compute_enriched_2026-07-17.csv`: lossless megafiles.
- `unified_model_data_test_results_compute_enriched_2026-07-17.csv`: 97/97 structural tests passed.
- `factor_weight_optimization_2026-07-18.json`: global, objective-sensitivity, family-bootstrap, and nested weight tests.
- `factor_weight_optimization_predictions_2026-07-18.csv`: all 25 meta-level held-out predictions and learned weights.
