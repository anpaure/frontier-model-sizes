# ECI architecture-blend challenger

Generated 2026-07-31. Status: **ZERO LIVE WEIGHT**.

The challenger geometrically combines 60% aggregate ECI score-only with 40% aggregate ECI score/date/MoE/reasoning. The comparator is the current direct 60% score-only + 40% score/date mapping.

## Held-out results

| Evaluation | Rows / developers | Baseline median | Challenger median | Baseline MAE | Challenger MAE | 90% developer CI on MAE delta |
|---|---:|---:|---:|---:|---:|---:|
| Fixed, all | 69 / 13 | 2.536× | 2.356× | 0.465 | 0.430 | [-0.045, -0.005] |
| Fixed, frontier-like | 46 / 11 | 2.547× | 2.501× | 0.477 | 0.442 | [-0.049, -0.000] |
| Nested, all | 57 / 12 | 2.612× | 2.589× | 0.488 | 0.448 | [-0.053, -0.007] |
| Nested, frontier-like | 42 / 10 | 2.638× | 2.596× | 0.495 | 0.459 | [-0.050, 0.003] |

The nested chooser uses only earlier outer-fold errors and excludes the target developer again. It selects the architecture challenger 52 times and the baseline 5 times.

## Anchor checks

Kimi K3: baseline 6.275× error; challenger 6.681×. Grok 4.5: baseline 2.853×; the explicitly non-observed MoE+reasoning working scenario gives 3.063×. The full four-scenario Grok sensitivity is retained in the JSON.

## Decision

Do not change the live ECI factor or any headline parameter forecast. The frontier confidence intervals are not wholly favorable, within-2× accuracy deteriorates, live-target architecture is unobserved, and there are no frozen prospective disclosures.

Failed gates:

- `fixed_all_metrics_non_worse`
- `fixed_frontier_metrics_non_worse`
- `live_target_architectures_observed`
- `nested_all_metrics_non_worse`
- `nested_frontier_ci_wholly_favorable`
- `nested_frontier_metrics_non_worse`
- `prospective_disclosures_at_least_3`
- `prospective_frontier_disclosures_at_least_2`

The machine-readable audit and fold ledger are `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_architecture_blend_challenger_2026-07-31.json` and `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_architecture_blend_challenger_predictions_2026-07-31.csv`.
