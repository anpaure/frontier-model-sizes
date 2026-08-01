# First-party frontier evidence audit

Snapshot: 2026-07-18

## Decision

The official evidence improves model identity and audit coverage, but it does not change the headline forecasts. Claude Fable 5 remains 4.5T and GPT-5.6 Sol remains 3.1T after one-decimal display rounding.

Anthropic's system card directly establishes that Claude Fable 5 and Claude Mythos 5 use the same underlying weights. The canonical registry therefore assigns both checkpoints to `base:anthropic-claude-fable-mythos-5`. Anthropic's separate statement that some client requests can fall back to Claude Opus 4.8 describes serving behavior; it is not treated as evidence that Fable/Mythos and Opus share a base.

OpenAI's GPT-5.6 system card reports a 3.6-minute UK AISI no-CoT math horizon for Sol and a 2.3-minute comparator for GPT-5.5. These are direct capability measurements, not parameter disclosures. Both are preserved in the unified megafile, workbook, site, and source manifests.

## Why the Sol point has zero incremental weight

The predeclared model-level candidate is:

`ln(total parameters) ~ exact release date + MoE + reasoning + ln(no-CoT horizon)`

Its baseline omits the horizon term. Every test checkpoint is predicted only from strictly earlier rows, and the entire test developer is excluded from training. Only folds with a full-rank design are admitted. This yields 16 chronological predictions across five held-out developers.

The horizon candidate has a mildly better equal-developer point estimate: mean absolute log error falls from 1.125 to 1.057. The 20,000-draw developer-balanced bootstrap assigns 61.6% probability to an improvement, but its 90% interval for candidate-minus-baseline error is -0.479 to +0.421. It therefore does not establish a portable improvement.

More importantly, defensible mappings of the same official 3.6-minute point disagree sharply:

| Mapping | Sol total-parameter sensitivity |
|---|---:|
| Direct model-level regression | 1.7T |
| Pooled paper elasticity anchored to Kimi K3 | 6.5T |
| MoE paper elasticity anchored to Kimi K3 | 9.5T |
| GPT-5.5-suite-rebased pooled elasticity | 11.2T |

The non-baseline maximum/minimum ratio is 6.6×. The official measurement is precise as a reported point, but the capability-to-parameters transport is not. Promotion requires at least 30 chronological predictions, eight held-out developers, a wholly favorable bootstrap interval, mapping dispersion below 2×, and a same-suite uncertainty interval. None of those statistical gates currently passes.

## Same-size controls

The audit retains rather than suppresses counterexamples:

- Kimi K2 and DeepSeek V3 family sequences keep the same reported total parameter count while their no-CoT horizons move materially.
- The exact open-weight Kimi K2.5 → Kimi K2.6 Epoch lineage keeps the same parameter count while no-CoT horizon falls from 1.102 to 0.508 minutes.
- GPT-5 and Opus same-base sequences remain user-asserted sensitivities. They are not promoted to public parameter identities.

These controls are why a frontier capability increase is not automatically converted into a larger pretrain.

## Source integrity

Ordinary builds are offline and verify frozen first-party artifacts. A live refresh is explicit:

```bash
./run_forecast_pipeline.py --refresh-frontier-primary
```

The OpenAI HTML capture is stored deterministically. The 27 MB Anthropic PDF is not committed; its URL, byte count, SHA-256, page count, per-page text hashes, and exact claim-page locations are committed and verified. The workbook manifest includes the raw OpenAI capture, the Anthropic verified-claim ledger, the normalized evidence ledger, collection metadata, statistical audit, and control ledger.

## Reproducible artifacts

- `collect_frontier_primary_evidence.py`
- `analyze_frontier_primary_evidence.py`
- `sources/openai_gpt_5_6_system_card_2026-07-18.html.gz`
- `sources/anthropic_fable_mythos_primary_claims_2026-07-18.json`
- `sources/frontier_primary_evidence_2026-07-18.csv`
- `sources/frontier_primary_evidence_collection_metadata_2026-07-18.json`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/frontier_primary_evidence_audit_2026-07-18.json`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/frontier_primary_evidence_controls_2026-07-18.csv`
- `tests/test_frontier_primary_evidence.py`
- `tests/test_frontier_primary_evidence_signal.py`
