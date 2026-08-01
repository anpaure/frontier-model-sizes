# Epoch ECI reproduction and date audit

Snapshot: 2026-07-31

## Outcome

The aggregate Epoch Capabilities Index is reproduced from the current 2,059-row canonical benchmark input with Epoch's official `eci-public` implementation pinned to commit `542567e72a415b72624e5bbd12603cfd3f485179`.

- 213 aggregate model scores are reproduced from 54 benchmark names.
- The official anchors remain exactly Claude 3.5 Sonnet = 130 and GPT-5 = 150.
- All 196 nonblank scores in Epoch's published capabilities archive agree with the reproduced scores within the archive's two-decimal display rounding; 17 published rows are blank.
- Bootstrap intervals are retained at full precision for all 211 non-anchor models.
- Source input, output, metadata, release-date crosscheck, and audit hashes are verified on every pipeline build.

## Release dates are a separate field

The canonical `eci_benchmarks.csv` date is necessary to reproduce Epoch's exported ECI metadata, but it is not always the release date of the exact `model_version`. The published capabilities archive disagrees with that input date for 43 of the 211 reproduced checkpoints that have a published release date. Examples include:

- Kimi K2 (Jul 2025): canonical input date 2025-07-11; published checkpoint release 2025-07-12.
- Mistral Large 2 (Nov 2024): canonical input date 2024-07-24; published checkpoint release 2024-11-18.
- DeepSeek Coder 1.3B: canonical input date 2024-01-25; published checkpoint release 2023-11-02.

The pipeline therefore preserves both dates and applies this deterministic policy:

1. Reproduce scores from the untouched canonical ECI input.
2. Use the published checkpoint-level `Release date` for chronological regression.
3. Fall back to the canonical input date only when the published date is blank. The two fallbacks are Claude Instant and Gemma 2 9B.

This prevents report or family-aggregation dates from leaking into the algorithmic-improvement coefficient while preserving exact score reproducibility.

## Generated artifacts

- `sources/epoch_snapshot_manifest_2026-07-31.json`: atomic snapshot inventory, source hashes, and canonical within-snapshot identity policy.
- `sources/epoch_eci_reproduced_scores_2026-07-31.csv`: full-precision official reproduction output.
- `sources/epoch_eci_reproduction_metadata_2026-07-31.json`: upstream commit, command, runtime, hashes, bootstrap seed, and upstream-test context.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/epoch_eci_reproduction_crosscheck_2026-07-31.csv`: one row per model with both date fields, date policy, reproduced score, published score, and score difference.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/epoch_eci_reproduction_audit_2026-07-31.json`: machine-readable coverage and integrity contract.
- `tests/test_epoch_eci_reproduction.py`: frozen inventory, score, date-policy, example, uniqueness, and hash checks.
