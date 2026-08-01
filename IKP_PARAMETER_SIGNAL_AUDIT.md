# IKP direct-capacity parameter signal audit

Date: 2026-07-18

Live decision: **0% inside Fable's evidence model and 0% of its final crowd-plus-evidence forecast.**

Sol decision: **Unobserved; no transport and no numerical change.**

## Result

The current Incompressible Knowledge Probes (IKP) benchmark provides a genuinely different, direct factual-capacity sensitivity. It is not treated as a parameter disclosure or a live likelihood. The source's current Fable estimate is 3.5T. Re-estimating from only open-weight models released before Fable, while excluding Anthropic entirely and collapsing explicit thinking/non-thinking duplicates, gives 3.6T. The point diagnostics are favorable, but the current identity-resolved chronological subset contains only seven models against the predeclared minimum of eight, so the promotion gate fails.

| Quantity | Result |
|---|---:|
| Source calibration configurations | 93 |
| Distinct weight bases after serving collapse | 87 |
| Strict chronological/vendor-held-out predictions | 960 |
| Exact overlap with existing ensemble | 13 models / 11 families |
| Existing overlap median error | 2.16× |
| IKP overlap median error | 1.68× |
| Fixed 10% diagnostic blend median error | 1.80× |
| Families improved by fixed blend | 10 / 11 |
| Existing-vs-IKP signed-error correlation | 0.007 |
| Later chronological subset | 7 models / 7 families |
| Required later-subset coverage | 8 models / 7 families |
| Strict Fable point | 3.6T |
| Strict model-form range | 2.3–3.9T |
| Source 90% model prediction interval | 1.1–11.2T |

The fixed 10% diagnostic blend's family-balanced mean absolute log-error improvement has a 90% bootstrap interval of **−0.054 to −0.022** on the full exact overlap. On the later chronological subset, after at least five earlier overlap rows from five other families exist, the interval is **−0.054 to −0.012**. Both are wholly favorable. A leave-one-family-out meta-layer also excludes every scored test family when selecting the blend weight; it is retained as a diagnostic rather than used to set a live weight. Promotion nevertheless requires every encoded gate, and the chronological-model coverage gate is `7 ≥ 8`, which is false.

## Conditional benchmark audit

The pinned upstream repository also contains 81 vendor-reported MMLU, MMLU-Pro, GPQA Diamond, SimpleQA, and HLE rows with raw primary-source citation tables. All 81 model rows and all 173 populated score cells reconcile exactly to the raw markdown tables. The generated 93-row joined panel, nine simple-regression rows, and nine time-coefficient rows are reproduced exactly after the upstream rounding policy.

The conditional test asks a narrower question: does IKP improve a parameter regression that already knows the standard benchmark score and the exact day-level release date? Every test fold uses only strictly earlier releases and excludes the test vendor. Explicit thinking/non-thinking variants are again collapsed to 87 weight bases. MoE architecture and equal-total-weight-per-vendor fitting are crossed as two independent sensitivities.

| Benchmark | Panel bases | Strict test bases / vendors | Baseline median | +IKP median | Passing sensitivities |
|---|---:|---:|---:|---:|---:|
| MMLU | 30 | 16 / 11 | 2.66× | 1.70× | 3 / 4 |
| MMLU-Pro | 25 | 15 / 7 | 2.68× | 2.25× | 0 / 4 |
| GPQA Diamond | 30 | 18 / 9 | 4.60× | 2.19× | 4 / 4 |
| SimpleQA | 10 | 0 / 0 | — | — | 0 / 4 |

GPQA is robust across row-equal/vendor-equal and with/without-MoE specifications. Its primary equal-vendor paired interval is **−0.774 to −0.154** log10 error, with 99.86% bootstrap probability of improvement. MMLU is supportive: its primary interval is **−0.384 to −0.139**, but the vendor-balanced plus architecture specification crosses zero. MMLU-Pro has only seven held-out vendors and no specification with a wholly favorable interval; SimpleQA cannot form a qualifying strict fold. These failures are retained rather than pooled away.

This corroborates that IKP is not merely re-expressing a standard knowledge benchmark, but it does **not** override the failed primary coverage gate or increase the live weight. The conditional panel and Fable target still share one IKP source, Fable remains out of the open-weight calibration range, and Sol has no IKP observation.

### Upstream narrative discrepancy

The pinned generated files report 93 post-exclusion configurations, full-set R² **0.9105**, and an IKP time coefficient of **+0.0864 pp/month**. The accompanying `SUMMARY.md` still says 89, 0.917, and a negative coefficient; it also refers to 96 raw models where the pinned panel has 100. Six such stale claims are parsed and tested explicitly. Generated CSV outputs are authoritative; the narrative is never silently substituted.

## Leakage and duplicate controls

- Every scored calibration prediction uses only models with a strictly earlier release date.
- The test model's vendor is excluded from its training fold.
- Six explicit `-think` serving duplicates collapse to one base observation. Mean, non-thinking, and maximum-accuracy collapse policies are all retained as sensitivity checks.
- Fable's live 3.6T point uses 86 earlier open-weight bases across 31 families and 19 non-Anthropic vendors.
- GPT-5.6 Sol is absent from the pinned evaluation. It remains null rather than borrowing an OpenAI-family value.
- Exact checkpoint mappings are enumerated manually. Fuzzy name matching is prohibited.

## Why the live weight is zero

IKP measures effective factual recall, not literal architecture. Its result can move with pretraining data coverage, refusals, distillation, and serving configuration. Fable's observed refusal rate is 12.1%, placing it in the source's caution tier. Same-base controls also demonstrate material serving sensitivity: the IKP-implied sizes for asserted GPT-5/5.5 and Opus 4.7/4.8 same-base pairs differ substantially.

The independent replication audited an earlier benchmark version and found probe-ambiguity and calibration sensitivity. The current upstream v2 addresses several issues by using a cleaned set, λ=0, and refusal-aware reporting, but it does not eliminate all data-diversity and serving-policy confounding. More decisively, the live policy requires at least 12 overlap models, 10 overlap families, eight later chronological models, seven later chronological families, and favorable full and chronological bootstrap gates. The current audit passes every condition except the eight-model chronological minimum: it has seven. The direct 3.6T Fable estimate therefore remains available as a sensitivity at 0% weight.

## Forecast effect

| Target | IKP-free baseline | Live after IKP decision | Display |
|---|---:|---:|---:|
| Fable evidence center | 4.7319T | 4.7319T | 4.7T |
| Fable final center | 4.5483T | 4.5483T | 4.5T |
| Sol evidence center | 3.1T | 3.1T | 3.1T |
| Sol final center | 3.1457T | 3.1457T | 3.1T |

The failed promotion gate means IKP makes no numerical change. The exact live centers are 4.7319T evidence / 4.5483T final for Fable and 3.0810T evidence / 3.1457T final for Sol. The current empirical 50% Fable interval is 2.4–9.5T. One-decimal display is 4.5T for Fable and 3.1T for Sol.

## Pinned sources and reproducibility

- Upstream repository: `19PINE-AI/ikp`, commit `e5c4231985048bb2db5dc2611b6eb659b891791d`
- Independent replication: `BenSturgeon/ikp-replication`, commit `c44e4dc82132e268dc9a2c86350863f59282fddb`
- `collect_ikp_source.py` verifies immutable commit URLs and exact SHA-256 hashes.
- `analyze_ikp_parameter_signal.py` reproduces the source fit, generates all strict predictions, runs incremental validation, and writes the site contract.
- `analyze_ikp_conditional_benchmark_signal.py` verifies all 21 pinned IKP/replication files, reconciles raw benchmark citations, reproduces the upstream comparison outputs, detects stale narrative claims, and generates 196 conditional prediction rows.
- `tests/test_ikp_parameter_signal.py` checks hashes, chronology, vendor exclusion, duplicate collapse, target construction, promotion gates, and site-output identity.
- `tests/test_ikp_conditional_benchmark_signal.py` checks exact upstream reproduction, all 196 fold identities, conditional gates, non-promotion of sparse benchmarks, and site-output identity.
- The independent re-audit refits all 49 primary conditional predictions from the pinned source rows and recomputes every sensitivity bootstrap.
- The generated workbook contains an `IKP Signal Audit` sheet with the full 13-model overlap ledger, conditional benchmark table, failed-gate explanation, and six-row stale-narrative reconciliation.
