# No-CoT exact-date audit

## Result

All 49 checkpoints in the no-CoT paper panel now have day-level release dates. The prior unified file already resolved 45 through the audited ECI/Epoch registry; this increment resolves the remaining four without inferring any new parameter identity.

| Paper checkpoint | Exact date | Date source | Parameter-join policy |
|---|---:|---|---|
| GPT-2 | 2019-02-14 | Epoch exact checkpoint row, backed by OpenAI | Date only; no Epoch parameter join |
| GPT-3 | 2020-05-28 | Epoch exact checkpoint row, backed by the GPT-3 paper | Date only; no Epoch parameter join |
| GPT-3.5 | 2022-03-15 | Epoch exact checkpoint row for `davinci-002` | Date only; no Epoch parameter join |
| Qwen 3 30B-A3B (2507) | 2025-07-28 | First commit in the first-party Hugging Face repository | Date only; no Epoch parameter join |

The Qwen raw commit history is frozen in `sources/qwen3_30b_a3b_instruct_2507_hf_commits_2026-07-18.json.gz`. Refreshing it is explicit through `--refresh-no-cot-dates`.

## Scaling-law correction

The paper publishes month-level dates and bootstrap scaling-law estimates, but not its bootstrap samples. The audit therefore does not pretend to recreate the unavailable bootstrap. It performs a method-matched deterministic sensitivity:

1. Recreate the paper's log-linear Pareto-frontier OLS approximation using month-start dates.
2. Rerun the same calculation with exact dates, retaining the paper's GPT-2/GPT-3 trend exclusions.
3. Multiply the published point and confidence interval by the exact-date/month-date slope ratio.

This changes the time-horizon law from 373.0 to 365.9 days (ratio 0.9809) and the token-horizon law from 437.0 to 438.4 days (ratio 1.0031). Opus 4.7 enters the exact-date time frontier because its 2026-04-16 release precedes GPT-5.5 on 2026-04-23; month-only data treated both as April 2026 and suppressed Opus 4.7 through same-date dominance.

The no-CoT evidence weight remains 50%. This is a source-fidelity correction, not a new likelihood or discretionary reweighting.

## Forecast impact

The correction is small but nonzero:

These values isolate the exact-date correction at that pipeline stage; they are not the current all-evidence headline. After the later source and weight refreshes, the current Sol final is 3.1457T, displayed as 3.1T.

| Forecast | Before | After | Relative change | Displayed headline |
|---|---:|---:|---:|---:|
| Claude Fable 5 | 4.5133T | 4.5325T | +0.42% | 4.5T |
| GPT-5.6 Sol | 3.1549T | 3.1611T | +0.20% | 3.2T |

## Enforced invariants

- 49 no-CoT model rows and 49 exact dates.
- Zero month-only rows.
- Exactly four explicit date-only overrides.
- Zero parameter identities added by those overrides.
- Every exact-date row links back to its pre-existing no-CoT checkpoint identity.
- The 49 model audit rows, one law audit row, and 256 numeric audit measurements are included in the operational megafiles.
- Frozen source hashes, workbook formulas, rendered audit sheet, site contract, and pipeline tests all cover this increment.

Primary references: [no-CoT paper](https://arxiv.org/pdf/2606.07157), [Qwen first-party model repository](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507), [OpenAI GPT-2 release](https://openai.com/index/better-language-models/), and [GPT-3 paper](https://arxiv.org/abs/2005.14165).
