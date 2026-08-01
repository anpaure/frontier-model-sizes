# Artificial Analysis operational-signal audit

Source snapshot date: 2026-07-31

Decision: validate the direction of the existing small price branch; add no new operational weight.

## Question

Do Artificial Analysis's standardized API price, output-speed, latency, cost-per-task, or time-per-task measurements improve parameter-count prediction beyond AA Intelligence Index and exact release date?

This is a distinct source-level validation of the OpenRouter branch. Artificial Analysis reports the first-party API when one exists and otherwise the median across providers. Output speed is measured after the first token and normalized to OpenAI-token units. The values are live/current serving measurements, not historical vintages.

Methodology references:

- https://artificialanalysis.ai/methodology
- https://artificialanalysis.ai/methodology/performance-benchmarking
- https://artificialanalysis.ai/models/

## Data audit

The frozen 587-configuration AA payload is joined to the already-audited 275-checkpoint calibration panel after two fail-closed Motif openness corrections. The existing rule is retained exactly: configurations sharing a weight checkpoint, parameter count, and date collapse to the highest Intelligence Index score. No raw configuration is deleted from the frozen source ledger. Motif-2-12.7B-Reasoning has no populated operational field, so it changes the complete-panel inventory but not any price/speed/latency comparison row.

Coverage after checkpoint deduplication:

| Signal | Checkpoints | Developers |
|---|---:|---:|
| 7:2:1 blended price | 168 | 34 |
| Output speed | 125 | 26 |
| Time to first chunk | 125 | 26 |
| Intelligence Index cost per task | 61 | 21 |
| Intelligence Index time per task | 59 | 21 |

The operational panel contains 81 first-party configurations and 194 provider-median configurations. Price is available for all 81 first-party configurations and 87 provider-median configurations.

## Held-out design

Each primary prediction uses:

- target: log10 disclosed total parameters in billions;
- baseline: AA Intelligence Index plus exact release date;
- candidate: the identical baseline plus the stated operational field;
- outer split: only rows whose `max(checkpoint release date, parameter-label availability date)` is strictly earlier;
- family protection: the entire test developer is removed from training;
- training weights: equal total weight per developer, with estimated AA scores receiving half weight;
- main minimum: 30 prior checkpoints and 8 prior developers;
- frontier-like scope: test score at or above the 90th percentile of its training set.

Cost-per-task and time-per-task use a predeclared lower-coverage exploratory minimum of 20 checkpoints and 6 developers. They are not eligible to change the live model.

## Results

### Price

| Serving regime / scope | N | Baseline median error | + price median error | Baseline MAE log10 | + price MAE log10 | Developer-bootstrap 90% CI, candidate − baseline |
|---|---:|---:|---:|---:|---:|---:|
| All prices / all tests | 138 | 2.76× | 2.75× | 0.502 | 0.497 | −0.139 to −0.024 |
| All prices / frontier-like | 53 | 2.13× | 2.52× | 0.447 | 0.496 | −0.144 to +0.045 |
| Provider median / all tests | 57 | 3.07× | 2.79× | 0.513 | 0.476 | −0.084 to +0.001 |
| Provider median / frontier-like | 22 | 2.57× | 2.41× | 0.463 | 0.421 | −0.160 to −0.012 |
| First party / all tests | 39 | 2.47× | 2.89× | 0.549 | 0.552 | −0.123 to +0.057 |
| First party / frontier-like | 22 | 2.25× | 2.62× | 0.427 | 0.538 | −0.053 to +0.166 |

Provider-median price is predictive. First-party price is not, and its frontier-like point estimates worsen substantially. Claude Fable 5 and GPT-5.6 Sol are first-party API targets, so the provider-median result is not transferred to them as a new weight.

### Speed and latency

| Signal / scope | N | Baseline median error | Candidate median error | Baseline MAE log10 | Candidate MAE log10 | Developer-bootstrap 90% CI |
|---|---:|---:|---:|---:|---:|---:|
| Output speed / all | 89 | 2.58× | 2.68× | 0.489 | 0.491 | −0.026 to +0.032 |
| Output speed / frontier-like | 34 | 1.87× | 1.91× | 0.372 | 0.383 | −0.024 to +0.061 |
| TTFC / all | 89 | 2.58× | 2.72× | 0.489 | 0.498 | −0.003 to +0.020 |
| TTFC / frontier-like | 34 | 1.87× | 1.79× | 0.372 | 0.377 | −0.005 to +0.010 |
| Speed + TTFC / frontier-like | 34 | 1.87× | 1.89× | 0.372 | 0.394 | −0.013 to +0.085 |

Speed is neutral. Latency makes mean held-out error worse. Cost per task and time per task are also unsupported in their smaller exploratory panels.

## Exact AA/OpenRouter cross-source check

The AA operational panel and OpenRouter calibration share 28 exact Epoch checkpoint IDs. On positive-value pairs:

| Comparison | N | log10 Pearson | Spearman |
|---|---:|---:|---:|
| AA price vs OpenRouter price | 26 | 0.823 | 0.830 |
| AA speed vs OpenRouter raw tok/s | 24 | 0.501 | 0.418 |
| AA speed vs provider-normalized OpenRouter tok/s | 24 | 0.497 | 0.560 |

The independent price implementations agree strongly in rank and scale direction. Speed agreement is weak, consistent with hardware, routing, quantization, prompt-length, and provider effects dominating any parameter-count signal.

## Bayesian interpretation

This audit strengthens confidence in the sign of the existing small API-price branch, but does not create a new independent likelihood:

1. AA and OpenRouter prices are strongly correlated measurements of the same commercial serving decision.
2. The predictive result is concentrated in provider-median open-weight serving, while Fable and Sol are first-party targets.
3. First-party frontier-like held-out recovery worsens when price is added.
4. Current prices and performance are measured after release, so a historical-vintage causal interpretation is unavailable.

Accordingly:

- existing final price weight for Fable/Sol: unchanged at 3.375%;
- incremental AA operational price weight: 0%;
- tok/s weight: 0%;
- latency weight: 0%.

## Reproduction

```bash
python3 analyze_aa_operational_signal.py
python3 -m unittest -v tests.test_aa_operational_signal
```

Primary artifacts:

- `sources/aa_detailed_snapshot_2026-07-31.html.gz`
- `sources/aa_detailed_model_signals_2026-07-31.csv`
- `sources/aa_detailed_snapshot_manifest_2026-07-31.json`

Generated operational-audit outputs currently retain a legacy `2026-07-18` filename suffix for compatibility. Their embedded `metadata.generated_on` date is `2026-07-31`; current tests bind the 275-row panel, current counts, and source hashes to that date.
