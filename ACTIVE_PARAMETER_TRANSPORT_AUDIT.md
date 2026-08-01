# Active-parameter transport and Kimi K3 architecture audit

Current evidence date: 2026-07-31

Forecast target: total pretrained parameters

Intermediate quantity: parameters activated per token

## Bottom line

Moonshot's official Kimi K3 technical report now supplies both quantities needed for a direct architecture constraint: **2.780T total parameters and 104.2B activated parameters per token**. K3 selects 16 of 896 routed experts, but `16/896 = 1.7857%` is not the activated-parameter fraction. The disclosed activated fraction is `104.2/2780 = 3.7482%`, because attention, shared experts, embeddings, routers, and other non-routed components also execute. The older `2780B × 16/896 = 49.64B` shortcut is superseded and receives no evidential weight.

The refreshed audit predicts active parameters from AA score and exact date on 127 sparse checkpoints across 33 developers, yielding 96 strict chronological developer-held-out predictions, including 40 frontier-like folds. Active parameters are modestly easier to recover than total parameters, including at the capability frontier, and the refreshed developer-cluster interval is narrowly favorable. On 47 high-sparsity conversion folds, converting the active prediction back to total parameters lowers mean and tail error with a favorable clustered-mean interval, but slightly worsens the primary median. Fable and Sol sparsity remain unobserved, so the transport still fails the live-use policy.

Accordingly, the active-parameter route receives **0% incremental live weight**. It is retained as a structural sensitivity:

| Model | Raw predicted active | K3-calibrated active | K3-ratio total | Existing final |
|---|---:|---:|---:|---:|
| Claude Fable 5 | 85.6B | 139.5B | **3.7T** | **4.5T** |
| GPT-5.6 Sol | 73.1B | 119.1B | **3.2T** | **3.1T** |
| Kimi K3 | 64.0B | 104.2B disclosed | 2.8T disclosed | 2.8T |

The headline forecast remains unchanged. The current result says the Sol center is architecture-consistent under a K3-like sparsity assumption and gives Fable a lower structural sensitivity near 3.7T. Neither target's actual sparsity is disclosed.

## Official architecture facts

Moonshot's [Kimi K3 technical report](https://arxiv.org/abs/2607.24653), official [model card](https://github.com/MoonshotAI/Kimi-K3), and [launch page](https://www.kimi.com/blog/kimi-k3) jointly establish:

- exactly 2.780T total parameters and 104.2B activated parameters;
- 16 of 896 routed experts selected per token;
- 93 layers: one dense layer, 69 Kimi Delta Attention layers, and 24 gated-MLA layers;
- two shared experts, 896 routed experts, and 16 selected routed experts per token;
- approximately 2.5× scaling-efficiency improvement over Kimi K2;
- a 401M-parameter, 27-layer MoonViT-V2 vision encoder and a 1M-token maximum context; and
- no exact pretraining tokens, pretraining FLOPs, post-training FLOPs, or GPU-hours disclosure.

The same K3 report's K2 comparator gives exactly 1.040T total and 32.6B activated parameters, with 8 of 384 routed experts, one shared expert, and a 131,072-token training context.

Those K2 values are applied to every named K2-family checkpoint through the canonical parameter-truth overlay while preserving the original 1.0T/32B and 1.04T/32.6B labels in their raw source views. The same overlay uses the commit-pinned Hugging Face tensor inventory—exactly 228.703644928B total and 10B active—for each of MiniMax M2.5 and M2.7, while preserving rounded 229B/230B source labels. Distinct checkpoints remain distinct observations; equal parameter truth is not a deduplication rule. No live weight was chosen after observing these corrected outcomes.

These facts imply:

| Quantity | Value | Interpretation |
|---|---:|---|
| K2 disclosed total/activated ratio | 31.9018× | `1040/32.6` |
| K3 selected-routed-expert fraction | 1.7857% | `16/896`; not the activated-parameter fraction |
| K3 disclosed activated fraction | 3.7482% | `104.2/2780` |
| K3 disclosed total/activated ratio | 26.6795× | `2780/104.2` |
| Activated fraction / selected-expert fraction | 2.0990× | Quantifies why routed-expert fraction cannot stand in for active parameters |

## Held-out validation

Every fold:

1. uses only strictly earlier checkpoints;
2. removes the entire test developer;
3. requires at least 30 active-parameter training rows and eight developers;
4. weights each developer equally, with estimated AA scores receiving half weight;
5. predicts in log parameter space from AA score and exact release date.

### Active versus total parameter predictability

The identical 127-checkpoint sparse panel yields 96 eligible chronological developer-held-out tests, of which 40 are frontier-like.

| Target | Median error | Mean absolute log10 error | P80 error |
|---|---:|---:|---:|
| Active parameters | **1.94×** | **0.332** | **3.49×** |
| Total parameters, same panel | 2.09× | 0.374 | 4.13× |
| Total parameters, full 275-checkpoint training panel | 2.02× | 0.378 | 4.23× |

The active-minus-total equal-developer error difference is −0.043 log10 and its refreshed 90% cluster interval is **[−0.093, −0.0001]**. On 40 frontier-like folds, active parameters have 1.64× median error versus 1.72× for total parameters. The active-size signal is narrowly favorable, but it still cannot identify a hidden model's total size without a target-specific sparsity ratio.

The added Motif-2-12.7B-Reasoning checkpoint is verified dense and therefore enters only the full total-parameter panel. It does not change the 127-checkpoint sparse active subset, the 96 active held-out folds, or the Fable/Sol K3-ratio sensitivities.

### Converting active back to total

For the preregistered-like ≥15× sparsity scope, the conversion uses the equal-developer geometric mean ratio among only earlier high-sparsity training checkpoints.

| Method | Tests | Median error | Mean absolute log10 error | P80 error |
|---|---:|---:|---:|---:|
| Active prediction × high-sparsity ratio | 47 | 2.10× | **0.349** | **3.20×** |
| Direct total-parameter baseline | 47 | **1.99×** | 0.403 | 4.60× |

The candidate improves mean and tail error but slightly worsens the median. Its equal-developer paired mean-error interval is **[−0.184, −0.015]**, which is favorable. It is still not eligible for promotion because the primary median does not improve and applying the conversion to Fable or Sol would require an undisclosed K3-like sparsity assumption.

## K3 external structural check

After removing every Kimi checkpoint, 117 earlier active-parameter rows across 31 developers predict K3 at 64.0B activated parameters. The disclosed 104.2B count is 1.629× larger. The audit therefore calibrates the active prediction by that observed residual before applying K3's exact `2780/104.2 = 26.6795×` total-to-activated ratio. This produces the zero-weight 3.721T Fable and 3.178T Sol sensitivities above. No expert-fraction midpoint enters the calculation.

## Compute-branch independence correction

One of the nine live target rows—Kimi K3—has an Epoch training-compute estimate, but it is explicitly classified as speculative rather than a primary-source disclosure. No live target has disclosed training compute, and all hidden targets still use compute predicted from AA score and date. The workbook then maps that predicted compute and date to parameters. Algebraically:

`log C = a + b·AA + c·date`

`log P = d + e·log C + f·date`

therefore:

`log P = (d + ea) + eb·AA + (ec + f)·date`.

So the live compute branch is not independent target evidence. It is a compute-structured AA/date regularizer learned from a different calibration panel. Its numerical weight remains unchanged because there is no target-faithful paired backtest justifying removal, but the pipeline and site now describe the dependency explicitly.

## Reproducible artifacts

- `sources/kimi_k3_release_evidence_2026-07-31.json`
- `sources/kimi_k3_technical_report_2026-07-31.pdf`
- `analyze_active_parameter_transport.py`
- `tests/test_active_parameter_transport.py`

The generated transport artifacts currently retain a legacy `2026-07-18` filename suffix for compatibility, but their embedded `metadata.generated_on` date is `2026-07-31` and their source hash points to the current K3 evidence bundle. The date inside the artifact, not the compatibility suffix, governs the audit vintage.
