# Epoch employee calibration feedback audit

Snapshot date: 2026-07-21

## Bottom line

The feedback identifies a genuine weakness in the aggregate AA/ECI parameter mapping. The supplied eight-row sheet is reproduced exactly from pinned sources, and its residuals are architecture-structured. However, neither tested architecture correction is robust enough to replace the live branch: both family-clustered confidence intervals cross zero, both MoE-only intervals cross zero, and both worsen recovery of the disclosed 1.5T Grok 4.5 frontier anchor. The live forecasts and weights therefore remain unchanged.

## What was supplied

The public sheet compares eight known-size open-weight checkpoints against:

```text
P_sheet = sqrt(P_AA * P_ECI)
```

It contains Kimi K2.6, Kimi K2.7 Code, Kimi K2 Thinking, DeepSeek-V3.1, Llama 4 Maverick, Llama 3.1-405B, Mistral Small 3.1, and Phi-4. The employee suggested that MoE structure may explain the misses, that price is noisy, and that a smaller benchmark set may generalize better than a large aggregate.

The normalized source record is [`sources/epoch_employee_calibration_feedback_2026-07-21.csv`](sources/epoch_employee_calibration_feedback_2026-07-21.csv). It preserves the sheet values, Epoch confidence labels, architecture labels, base-cluster identities, source URL, and audit notes.

## Exact source reconciliation

Every row is manually mapped to:

- the exact Artificial Analysis configuration in the pinned detailed panel;
- the exact ECI checkpoint in `regression_results.json`;
- one exact Epoch model row and its total parameter label; and
- the live formula coefficients and exact release date.

All eight AA scores, ECI scores, parameter labels, AA-implied predictions, ECI-implied predictions, geometric model estimates, and displayed ratios reproduce within their displayed rounding. The three Kimi rows share one lineage and are collapsed to one base cluster for uncertainty, leaving six independent clusters.

This is not treated as a clean statistical holdout. The rows were hand-selected after the live formulas existed, and three of the eight observations share a base lineage. It is a valuable structural challenge set.

## What the critique shows

| Scope | Rows | Median error | Geometric-mean error | Signed prediction / actual |
|---|---:|---:|---:|---:|
| All | 8 | 3.23× | 2.69× | 0.82× |
| MoE | 5 | 1.97× | 1.99× | 0.50× |
| Dense | 3 | 4.10× | 4.46× | 1.88× |

All five MoE checkpoints are underpredicted on average. That supports adding architecture to the candidate set. But binary MoE status is not sufficient: the dense subset contains Llama 3.1-405B, which is strongly underpredicted, and Mistral Small 3.1/Phi-4, which are strongly overpredicted. Active parameters, routing sparsity, model efficiency, and lineage matter in addition to the dense/MoE label.

## Candidate models

The audit refits candidates on all 88 ECI checkpoints with total and active parameter labels across 38 developer families. Every test prediction:

1. uses only checkpoints released strictly before the test checkpoint;
2. excludes the entire test family;
3. gives each training family equal aggregate weight; and
4. is compared on identical held-out rows when calculating paired effects.

The two substantive candidates are:

- `score + exact date + MoE`: predicts total parameters directly;
- `active then dated sparsity`: predicts active parameters from score/date, then predicts the total/active ratio among MoE training checkpoints.

A score-only model and a score/date/MoE/reasoning sensitivity are also retained, but reasoning is not used as a promoted correction.

## Held-out results

| Specification | Predictions | Median error | Within 2× | Equal-family Δ absolute log10 error vs live | 90% interval |
|---|---:|---:|---:|---:|---:|
| Live ECI 60/40 | 68 | 2.47× | 42.6% | — | — |
| Score + date + MoE | 68 | 2.27× | 44.1% | −0.017 | [−0.082, +0.050] |
| Active then dated sparsity | 62 | 1.82× | 53.2% | +0.010 | [−0.081, +0.105] |

The active/sparsity candidate has the best unweighted headline median, but it is not better after families receive equal weight. The uncertainty intervals for both candidates cross zero. Their MoE-only family intervals also cross zero.

## Frontier-anchor veto

Grok 4.5 is a disclosed 1.5T frontier anchor and is excluded from its candidate training folds by family. The live ECI estimate is 1.24T, a 1.21× error. The architecture candidate estimates 0.57T, a 2.61× error; the active/sparsity candidate estimates 0.83T, a 1.80× error. Both fail the rule that a new model may be at most 10% worse than the current branch on this disclosed anchor.

This anchor check matters because the objective is current frontier-model extrapolation, not only average recovery over a heterogeneous historical panel.

## Less-is-more and price checks

The existing nested ECI component audit already allows the data to choose among all eligible, pretraining-like, and knowledge-only benchmark sets. It selects `knowledge_only` for Fable and Sol and improves the all-panel median from 2.28× to 1.99× and the frontier-like median from 2.72× to 2.43×. It remains unpromoted because its primary family interval barely crosses zero, its narrow-ECI-confidence replication is not favorable, and target coverage is incomplete.

Removing the weak price signal changes the final estimate by only +1.38% for Fable and +0.30% for Sol. Price is noisy and deliberately downweighted, but it is not responsible for the sheet mismatch.

## Promotion gates

An architecture candidate receives nonzero live weight only if all of these hold:

- at least 60 paired predictions across at least 25 families;
- an all-family 90% bootstrap interval wholly below zero;
- at least 20 MoE paired predictions across at least eight families;
- a MoE-family 90% bootstrap interval wholly below zero; and
- disclosed Grok anchor error no more than 1.10× the current ECI anchor error.

Both candidates pass coverage and fail the other gates. Their incremental live weight is therefore 0%.

## Automatic artifacts

`analyze_epoch_feedback_signal.py` regenerates:

- [`epoch_feedback_lean_architecture_audit_2026-07-21.json`](outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/epoch_feedback_lean_architecture_audit_2026-07-21.json)
- [`epoch_feedback_critique_panel_2026-07-21.csv`](outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/epoch_feedback_critique_panel_2026-07-21.csv)
- [`lean_architecture_predictions_2026-07-21.csv`](outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/lean_architecture_predictions_2026-07-21.csv)
- [`lean_architecture_target_sensitivity_2026-07-21.csv`](outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/lean_architecture_target_sensitivity_2026-07-21.csv)

`tests/test_epoch_feedback_signal.py` verifies identities, source hashes, exact formula reproduction, prediction chronology, family exclusion, inventory counts, bootstrap decisions, anchor vetoes, and the zero-weight integration decision. Both are mandatory steps in `run_forecast_pipeline.py`.
