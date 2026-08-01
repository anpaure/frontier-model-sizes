# METR primary-source audit

The METR branch is sourced from METR's official
[`benchmark_results_1_1.yaml`](https://metr.org/assets/benchmark_results_1_1.yaml)
asset. The build no longer writes or depends on a hardcoded transcription of
the 26 model rows.

`collect_metr_primary_source.py` has two modes:

- the default offline mode reads the committed raw YAML and deterministically
  regenerates the normalized signals, metadata, and reconciliation audit;
- `--refresh` fetches the first-party asset and then performs the same checks.

The parser is deliberately fail-closed against the published document schema.
It requires 26 unique result IDs; the exact benchmark and source-version
identifiers; all average, p50, p80, and confidence-interval fields; valid
release dates and SOTA flags; and the published trend-law constants. An
upstream structural or value change fails the build rather than silently
changing the regression.

## Current reconciliation

- Official result rows: **26 unique models**
- Full official scaffold entries preserved: **114**
- Official trend: **128.744 days** from 2023, 95% interval
  **104.428–158.012 days**
- Earlier user-copy rows matching across all 14 common fields: **26/26**
- Field mismatches: **0**
- Official raw SHA-256:
  `aae31902b0519a4da73e16643915e5e8aca13cd3315c3aac893ce3d6dfe92ad9`

The historical user-supplied CSV remains committed only as an independent
crosscheck. It is not authoritative and never overwrites the official source.
The official normalized ledger adds `scaffolds_json`, so the prior collapsed
`scaffold_family` field is preserved while the complete source arrays are no
longer lost.

## Artifacts

- `sources/metr_benchmark_results_1_1_2026-07-18.yaml` — verbatim primary asset
- `sources/metr_horizon_official_signals_2026-07-18.csv` — normalized 26-model ledger
- `sources/metr_horizon_official_metadata_2026-07-18.json` — provenance, trend, inventory, and hashes
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/metr_primary_source_audit_2026-07-18.json` — exact crosscheck and losslessness audit
- `tests/test_metr_primary_source.py` — pinned source, schema, count, scaffold, and hash tests

This provenance correction does not change the parameter forecast by itself:
the official numeric values exactly match the prior copy. It strengthens the
claim that the METR evidence entering the model is lossless, reproducible, and
first-party.
