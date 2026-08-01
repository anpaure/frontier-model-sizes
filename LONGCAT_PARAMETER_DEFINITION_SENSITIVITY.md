# LongCat 2.0 parameter-definition sensitivity

## Verdict

Keep **1.6T total / about 48B average active** as the canonical regression target. It is the publisher's explicit model-level definition. Treat **1.775560491136T** as an exact serialized-tensor inventory sensitivity, not as a silent correction to the disclosed model total. The sensitivity receives **zero live weight**.

## Why the two numbers differ

| Quantity | Parameters | Interpretation |
|---|---:|---|
| Publisher model total | 1600.000B | First-party rounded semantic total |
| HF safetensors elements | 1775.560491B | Exact stored tensor elements |
| `model.mtp.*` elements | 136.943684B | Auxiliary multi-token-prediction tensors in the checkpoint |
| Serialized elements excluding MTP | 1638.616807B | Rounds to 1.6T at the publisher's displayed precision |

MTP tensors explain 78.0% of the nominal 175.6B gap. The remaining 38.6B is inside the rounding interval for a one-decimal 1.6T label. The config explicitly declares three replicated MTP layers. This makes the publisher/HF difference consistent with a definition boundary, but it does not prove every detail of Meituan's counting convention; serialized inventories can also differ on tied, replicated, auxiliary, or inference-only tensors.

## Diagnostic regression effect

| Target | Publisher 1.6T | HF serialized 1.7756T | Change |
|---|---:|---:|---:|
| Claude Fable 5 | 3.4234T | 3.4517T | +0.83% |
| GPT-5.6 Sol | 2.5147T | 2.5329T | +0.72% |
| Claude Opus 5 | 2.1741T | 2.1914T | +0.79% |

The strict AA median held-out error moves from 2.7008× to 2.6885×, while AA p80 moves from 11.6091× to 12.1028×. The available-component ensemble is unchanged at 2.1364× because LongCat has no independently matched second component in that ensemble, and its convention changes only the LongCat target plus later AA fits.

## Live forecast effect

| Target | Current evidence | HF counterfactual | Change |
|---|---:|---:|---:|
| Claude Fable 5 | 4.7319T | 4.7319T | +0.00% |
| GPT-5.6 Sol | 3.0810T | 3.0810T | +0.00% |
| Claude Opus 5 | 3.0038T | 3.0038T | +0.00% |

The exact live effect is zero: the K3-calibrated live branch consumes frontier AA scores, ECI scores, and dates, not the LongCat calibration target or the legacy `moe_total_b` field. This is a dependency-graph result, not an assertion that the diagnostic regression itself is invariant.

## Provenance and policy

- Publisher source: https://www.meituan.com/news/NN260630164005904
- HF repository API: https://huggingface.co/api/models/meituan-longcat/LongCat-2.0
- Pinned tensor index: https://huggingface.co/meituan-longcat/LongCat-2.0/resolve/834bf5ffe3047aa9f6cc7a64a9bc068b146b8274/model.safetensors.index.json
- Full machine-readable audit: `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/longcat_parameter_definition_sensitivity_2026-07-31.json`
- Target sensitivity CSV: `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/longcat_parameter_definition_target_sensitivity_2026-07-31.csv`
- Backtest sensitivity CSV: `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/longcat_parameter_definition_backtest_sensitivity_2026-07-31.csv`
- Raw source snapshots, canonical parameter truth, live weights, and forecast centers are not modified by this audit.
