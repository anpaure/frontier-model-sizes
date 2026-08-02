# Predictive uncertainty audit

Generated: 2026-08-01

Central forecasts changed by this uncertainty step: **No**. The upstream Fable center uses the baseline evidence mix because IKP failed its current chronological-coverage promotion gate and receives 0% live weight.

## Result

The published 4.5T Fable, 3.1T Sol, and 3.0T Opus 5 values are centers, not precise parameter censuses. Empirical error bands are generated automatically from realized errors in both the strictly chronological model-lineage-held-out and whole-developer-held-out ensemble backtests. Each developer contributes the largest absolute residual among checkpoints on its latest eligible release date, and each displayed factor is the conservative envelope across the two holdout specifications.

| Target | Evidence center | Final displayed center | Empirical 50% interval | Empirical 80% interval |
|---|---:|---:|---:|---:|
| Claude Fable 5 | 4.7T | 4.5T | 2.0–11.3T | 0.9–25.0T |
| GPT-5.6 Sol | 3.1T | 3.1T | 1.0–9.9T | 0.6–16.3T |
| Claude Opus 5 | 3.0T | 3.0T | 1.0–9.1T | 0.6–15.8T |

These intervals are intentionally much wider than coefficient-only regression intervals. They include realized model-form, architecture, benchmark, date-law, and proxy-transport error. The crowd shifts the displayed center slightly but does not narrow the interval because the crowd has no disclosed Fable/Sol outcomes on which to calibrate coverage.

## Calibration policy

- Source: two matched 44-checkpoint ensembles. One excludes the test model-series lineage; the stricter refit excludes the test developer from every component fit and equalizes training weight by developer.
- Primary cohort: 27 frontier-like predictions across 16 lineages and 11 canonical developers.
- Developer balance: each developer contributes one score. If multiple checkpoints share the latest date, the largest absolute residual is retained. The old lexical tie-break silently selected Llama 4 Scout over the much larger-error Maverick; model names no longer affect the band.
- Band: symmetric multiplicative empirical prequential order statistic `ceil((developers + 1) × requested coverage)`. This is a transparent error summary, **not** a formal split-conformal guarantee: the fitted model changes over time, most benchmark measurements are current-snapshot rather than release-vintage, and developer clusters are not exchangeable with a new closed frontier lab.
- Unsupported ranks: if the requested order-statistic rank exceeds the number of developer scores, the factor is `null`/unsupported. It is never clipped to the observed maximum and presented as a finite band.
- Tail policy: the descriptive 50% factor remains target-chronological within each holdout specification. It is 2.38× for Fable, 3.22× for Sol, and 3.02× for Opus 5. The 80% and 90% factors first take the maximum of target-chronological and currently observed values; every level then takes the conservative envelope across holdout specifications. The current frontier-cohort factors are 3.02×, 5.28×, and 6.19×, and the target 80% factor is exactly 5.276400× for all three models.
- Transparency: the per-specification bands, model-lineage grouping comparison, target-date factors, and selected envelope source all remain in the JSON.

Kimi K3 required a retrospective join correction. K3 is deliberately withheld from the AA parameter-target table, but its exact score is observed. The prior available-component row silently treated this as missing AA evidence and combined only ECI (449.8B) with speculative compute (211.1B), yielding 384.5B / 7.23×. The corrected row adds a leakage-safe AA prediction trained on 46 strictly earlier checkpoints across 21 families with every Kimi lineage excluded; AA alone predicts 2.243T / 1.24×. The unchanged weighted available-component result is 838.7B / 3.31×. This correction lowers the diagnostic tails but does not alter any central forecast or rewrite the immutable prospective freeze.

## K3 efficiency upper-tail stress test

The separate user-supplied statement that the targets are at least as parameter-efficient as disclosed 2.780T Kimi K3 is now implemented as a center-preserving, winsorized upper-tail structural sensitivity. It is not literal conditioning, an extra AA/ECI likelihood, or a point-estimate factor. Its point-center weight is 0%, and the exact crowd contribution remains 50% for Fable and Sol.

The equal-efficiency transport holds target and K3 dates equal and uses only retained laws that are linear in `log10(parameters)`. Those laws already encode diminishing returns in raw parameters. K3's parameter label is excluded from the slope fit and used only to anchor the level. The flexible quadratic/ridge challenger remains rejected at 0% weight; the earlier 8.5–12.2T claim used an out-of-support convex derivative and transported K3's full residual.

| Target | AA equivalent | Canonical ECI equivalent | Pooled reference median | Reference 10–90% |
|---|---:|---:|---:|---:|
| Claude Fable 5 | 3.7T | 5.1T | 4.1T | 3.6–4.9T |
| GPT-5.6 Sol | 3.3T | 5.7T | 3.9T | 3.4–4.6T |
| Claude Opus 5 | 4.0T | 4.3T | 4.0T | 3.5–4.6T |

Five retained ECI slope variants and the published target/K3 ECI intervals generate 20,000 deterministic reference draws. AA and ECI receive equal log weight but are explicitly labeled correlated alternative mappings. Their geometric pool is not a strict ceiling under either mapping, and the uniform slope-specification weights are not learned posterior probabilities. The user-supplied `Sol < Fable` judgment projects only Sol's reference draws, not actual parameter draws. The maximum non-K3 calibration ECI is 151.37, so K3 and all three targets are out of support; the result is therefore an extrapolative sensitivity, not a newly validated scaling law.

At the default 80% projection strength, the layer winsorizes only upper-half uncertainty draws:

| Target | Raw 80% band | Projected 80% stress test | Lower endpoint changed | Center changed |
|---|---:|---:|---:|---:|
| Claude Fable 5 | 0.9–25.0T | 0.9–5.5T | No | No |
| GPT-5.6 Sol | 0.6–16.3T | 0.6–4.7T | No | No |
| Claude Opus 5 | 0.6–15.8T | 0.6–4.8T | No | No |

Fable's pooled equal-efficiency reference falls below its 4.7T evidence center in about 84% of draws. The center overrides the reference in exactly those draws, so even 100% projection strength is not a literal hard constraint. The pipeline exposes that override rate rather than silently recentering Fable or falsely claiming conditioning. The projected values are not empirical coverage intervals, conformal intervals, or formal Bayesian credible intervals. The raw bands remain authoritative and unchanged; projection strength is exposed from 0% to 100% in the site.

Sequential coverage is no longer summarized as though all repeated checkpoints were independent:

| Holdout fit | Requested level | Raw checkpoints | Developer-balanced | Latest developer | Supported tests |
|---|---:|---:|---:|---:|---:|
| Model lineage | 50% | 45.0% | 42.8% | 33.3% | 20 |
| Model lineage | 80% | 100.0% | 100.0% | 100.0% | 20 |
| Model lineage | 90% | 100.0% | 100.0% | 100.0% | 7 |
| Whole developer | 50% | 45.0% | 44.3% | 33.3% | 20 |
| Whole developer | 80% | 90.0% | 95.6% | 88.9% | 20 |
| Whole developer | 90% | 100.0% | 100.0% | 100.0% | 7 |

At requested 90% coverage, 13 of the 20 sequential checkpoints have too few earlier developers for the rank to exist. The previous finite maximum for those cases was mathematically unsupported. With only 11 current calibration developers and only seven supported sequential 90% tests, none of these percentages identifies tail coverage precisely.

## Architecture-matched diagnostic

Restricting the panel to recent, frontier-like, ≥100B-predicted, MoE reasoning checkpoints leaves 11 tests across five developers. Its 80% envelope is 5.17×, but a 90% rank is unsupported (`ceil(6 × 0.9) = 6 > 5`). The diagnostic is retained but is too small to replace the 11-developer primary cohort.

The separate developer/vintage audit preserves the paired row-level comparison and the ECI-only historical sensitivity. Whole-developer holdout improves the paired current-score frontier median from 2.02× to 1.89×, but its latest-date developer factors remain wide and are reported from the actual whole-developer predictions rather than reusing lineage predictions. The 22-row first-observed vintage ECI frontier remains worse (3.08× lineage holdout; 3.13× developer holdout). Only two archive targets are genuinely interval-prospective, and both are Moonshot Kimi checkpoints, so the vintage result cannot calibrate the full ensemble.

## Crowd and frozen forecast status

The crowd changes the displayed center but never narrows these bands. This is essential because the forecasters share public evidence and are not independent calibration outcomes. Bands remain centered on the evidence-model estimate; the separate final center is shown explicitly.

This correction was performed after the immutable `2026-07-31-frontier-parameters-v1` prospective forecast freeze. It does not rewrite that freeze, its hash, its recorded crowd pools, or its central forecasts. It is a later diagnostic correction to uncertainty construction and labeling.

## Reproducible artifacts

- `analyze_parameter_predictive_uncertainty.py`
- `tests/test_parameter_predictive_uncertainty.py`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/frontier_parameter_predictive_uncertainty_2026-07-18.json`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/frontier_parameter_predictive_uncertainty_calibration_2026-07-18.csv`
- `site/public/data/predictive-uncertainty.json`
- `analyze_k3_efficiency_prior.py`
- `K3_EFFICIENCY_PRIOR_AUDIT.md`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/k3_efficiency_prior_2026-08-01.json`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/k3_efficiency_prior_cap_draws_2026-08-01.csv`
- `tests/test_k3_efficiency_prior.py`
- `analyze_parameter_vintage_sensitivity.py`
- `tests/test_parameter_vintage_sensitivity.py`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/parameter_developer_vintage_sensitivity_2026-07-31.json`

The main chart remains free of error bars. The selected-model inspector displays the raw empirical band, the K3-projected stress test, its center-override rate, and its independent projection-strength control separately, so changing factor weights does not falsely rescale or reinterpret a fixed historical error statement.
