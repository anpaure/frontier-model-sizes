# Artificial Analysis inference-budget audit

Source snapshot: 2026-07-31

## Outcome

Artificial Analysis's full machine-readable model payload materially expands the auditable open-weight parameter panel and directly measures reasoning-mode uplift. It does **not yet justify changing the live frontier forecast**.

The frozen source contains 587 model configurations. Of these, 333 raw-snapshot open-weight configurations have a parameter count, Intelligence Index score, and exact release date; collapsing only identical checkpoints produces 273 raw-snapshot groups across 47 creators. Two exact, hash-pinned official-repository overlays correct stale AA openness metadata for Motif 3 Beta and Motif-2-12.7B-Reasoning. The resulting calibration view contains 335 eligible configurations and 275 checkpoints across 48 creators; the raw AA ledger is never rewritten.

| Held-out comparison | Baseline median error | Candidate median error | Cluster-bootstrap conclusion |
|---|---:|---:|---|
| Current 50-model AA panel vs detailed 275-panel, all tests | 2.18× | 2.03× | Mean improves, but 90% interval crosses zero |
| Current 50-model AA panel vs detailed 275-panel, frontier-like | 1.90× | 1.83× | Inconclusive; 90% interval crosses zero |
| Score/date vs score/date + measured token budget, all tests | 2.69× | 2.42× | Unweighted metrics improve, but equal-developer interval is inconclusive |
| Score/date vs token-budget adjustment, frontier-like | 2.48× | 2.34× | Unweighted metrics improve, but the interval crosses zero |
| Raw score vs portable reasoning-standardized score, frontier-like | 2.49× | 2.22× | Interval still crosses zero |

Accordingly, the detailed panel, measured token-budget feature, and reasoning-standardization branch each receive 0% incremental live weight. They remain prospective diagnostics for the next disclosed frontier-scale checkpoint.

## Direct measurement of reasoning uplift

The snapshot contains 100 same-checkpoint reasoning/non-reasoning pairs across 17 creators:

- 61 open-weight pairs and 39 proprietary pairs;
- 53 pairs satisfy the stricter exact open-weight parameter/weight-group rule;
- all-pair checkpoint median uplift: 5.87 AA points;
- equal-creator median uplift: 6.43 points;
- equal-creator bootstrap 90% interval: 4.59–7.41 points.

This independently validates the existing six-point global correction. It also shows why a single lab-specific number is risky: creator medians range from 0.22 to 18.57 points. Anthropic's median is 5.92 points, OpenAI's is 18.57, xAI's is 13.74, and open-weight Kimi's is 7.81. These are configuration-level benchmark lifts, not pure RL-compute estimates.

The chronological standardization backtest uses a direct non-reasoning score when the identical checkpoint supplies one. Otherwise it uses only strictly earlier reasoning pairs. The portable version excludes the test creator; the creator-aware version requires at least two earlier same-creator pairs and falls back to the portable estimate. Both remove the entire test developer from parameter-model fitting.

## Exact Epoch crosscheck

All 62 exact Epoch checkpoints in the AA expansion panel are reconciled to the detailed payload. The median AA/Epoch parameter ratio is exactly 1.00 and the 5th–95th percentile range is 0.989–1.072. Three identity matches expose source-metadata conflicts and are retained with explicit flags:

- **Phi-4 Mini:** exact identity and parameter agreement, but AA reports a 2024 date inconsistent with the 2025 public release.
- **Granite 4.0 H 1B / Granite-4.0-H-Tiny:** exact active-parameter alias, but AA reports 1.5B while Epoch documents 7B total and 1B active.
- **Ring-flash-2.0 / Ring-flash-linear-2.0:** exact architecture alias and parameter agreement, but the creator label differs (InclusionAI versus Ant).

The crosscheck never overwrites either source's value. Epoch remains authoritative for regression targets on matched checkpoints.

## Inference-token data

AA exposes Intelligence Index output-token measurements for 79 selected open-weight checkpoint groups. A strictly chronological developer-held-out regression compares identical test rows with complete output/answer/reasoning token fields:

- baseline: AA score + exact date;
- candidate: baseline + `log10(1 + answer tokens/task)` + `log10(1 + reasoning tokens/task)`.

The token branch reduces the unweighted median and several mean/tail errors, including in the frontier-like subset, but its equal-developer cluster intervals still cross zero. Output tokens are also not inference FLOPs: architecture, tool calls, speculative decoding, and cascades remain only partially observed.

## Reproducibility

Use the frozen snapshot:

```bash
python3 run_forecast_pipeline.py
```

Explicitly refresh Artificial Analysis first:

```bash
python3 run_forecast_pipeline.py --refresh-aa
```

Important artifacts:

- `sources/aa_detailed_snapshot_2026-07-31.html.gz`: complete public React-Flight payload.
- `sources/aa_detailed_model_signals_2026-07-31.csv`: 587 flattened configurations with the complete raw record JSON preserved per row.
- `sources/aa_detailed_snapshot_manifest_2026-07-31.json`: source hashes, row counts, and refresh contract.
- `sources/aa_calibration_primary_overrides_2026-07-31.json`: fail-closed overlay ledger, exact checkpoint identities, chronology, lineage, and hashes.
- `sources/aa_calibration_evidence/`: vendored commit-pinned model cards, model API responses, Motif-2 config, and the weight-upload commit history used for offline validation.
- `sources/aa_parameter_label_availability_2026-07-31.json`: six-record release/label/weight timing ledger, backed by 29 vendored primary-source artifacts. See `AA_PARAMETER_LABEL_TIMING_AUDIT.md`.

Generated inference-audit outputs currently retain a legacy `2026-07-18` filename suffix for compatibility. Their embedded `generated_on` date is `2026-07-31`; current tests require 275 selected checkpoint groups, 100 reasoning/non-reasoning pairs, 62 exact Epoch reconciliations, two primary-source overrides, and the current hashes. The embedded date and hashes, not the compatibility suffix, determine the audit vintage.

Automated tests reparse the compressed payload, compare every flattened record to its preserved raw JSON, verify hashes and counts, enforce unique checkpoint and pair identities, prove strict chronology and developer exclusion for every prediction, and require the three source disagreements to remain visible.
