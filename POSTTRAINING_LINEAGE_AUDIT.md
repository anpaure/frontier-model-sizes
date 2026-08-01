# Post-training lineage and shared-base audit

Snapshot: 2026-07-18

## Decision

Post-training is a measured source of parameter-inference error, but no new correction receives live weight. Headline forecasts remain unchanged.

The exact Epoch lineage graph contains 685 non-empty `Base model` links. Requiring a unique normalized parent match, total parameters within 1%, and open weights leaves 239 links. Restricting to language models leaves 211 candidates; only eight edges across six distinct bases have a matched AA, ECI, No-CoT, METR, or ECI-component measurement for both parent and child.

## What is directly supported

- Fifty-three open-weight checkpoints have both reasoning and non-reasoning AA configurations on the same weights. Reasoning raises AA by a creator-balanced median of 5.51 points, with a 90% creator bootstrap interval of 3.78–7.48 points. This measures inference-compute configuration, not RL training compute.
- Seven unchanged-size lineage edges have ECI predictions. Their median later-child/earlier-parent date-adjusted parameter implication is 1.36× despite identical total parameters.
- Collapsing parent and child ECI scores improves equal-base mean absolute log error by only 0.004, with a 90% interval of -0.074 to +0.065. The result is not decisive.
- Three unchanged-size lineage edges have detailed AA predictions. Parent/child collapse improves median error from 1.37× to 1.31× and the three-base bootstrap is favorable, but three bases are far below the predeclared six-base signal gate.
- There is one No-CoT lineage edge and no METR lineage edge. Neither horizon dataset can presently identify a portable post-training correction.

## Benchmark specificity

Across 38 overlapping ECI-component measurements, knowledge benchmarks have a median within-lineage uplift of 1.47 ECI-equivalent points, versus 5.33 points for other benchmarks. This direction supports the hypothesis that knowledge-heavy benchmarks are less post-training-sensitive. It is still exploratory: knowledge coverage is five bases, other-benchmark coverage is four, and the broader pretraining-like non-knowledge category is dominated by one Llama lineage.

## Proprietary same-base claims

The workbook continues to collapse Opus 4.5–4.8 and GPT-5 through GPT-5.5 as user-supplied modeling assumptions. They are not labeled public disclosures:

- Anthropic publishes separate system cards describing pretraining followed by substantial post-training for Opus releases, but no primary-source same-underlying-model statement was found.
- OpenAI describes GPT-5.5 as a new model and does not disclose a shared pretrained base with GPT-5.
- OpenAI does explicitly state that GPT-5.5 Pro is the same underlying model as GPT-5.5 with parallel test-time compute. These are one base.
- OpenAI says GPT-5.6 Sol and Terra improve substantially over GPT-5.5 on small-scale pretraining optimization. The GPT-5-through-5.5 assumption is therefore not extended to GPT-5.6.

This is distinct from Claude Fable 5 and Claude Mythos 5: Anthropic's first-party system card explicitly says those two deployments use the same underlying weights. The canonical registry now applies that identity. Anthropic's documented Opus 4.8 client fallback is serving behavior, not evidence that Opus shares the Fable/Mythos base.

Naive date-adjusted AA regressions imply extreme scale growth along the asserted proprietary chains. Those ratios are deliberately shown only as counterfactual diagnostics, with calibration-range extrapolation flags. If the same-base assertions are true, the extreme ratios demonstrate why successive capability scores cannot be treated as independent parameter-count observations.

## Promotion gates

A live post-training correction requires all of the following:

1. at least eight verified measured open-weight bases;
2. at least six bases for each promoted signal;
3. a wholly favorable equal-base bootstrap interval;
4. public verification or direct open-weight evidence for any proprietary lineage identity used in the correction.

The current audit fails coverage, ECI interval, METR/No-CoT coverage, and proprietary-identity gates. Incremental weight is 0%.

## Reproducible artifacts

The lineage audit reads the base unified observations file. The later operational extension consumes the lineage outputs, avoiding a circular dependency.

- `analyze_posttraining_lineage_signal.py`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/posttraining_lineage_audit_2026-07-18.json`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/posttraining_lineage_edges_2026-07-18.csv`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/posttraining_lineage_measurements_2026-07-18.csv`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/posttraining_lineage_predictions_2026-07-18.csv`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/frontier_shared_base_sensitivity_2026-07-18.csv`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/frontier_lineage_evidence_2026-07-18.csv`
- `tests/test_posttraining_lineage_signal.py`
