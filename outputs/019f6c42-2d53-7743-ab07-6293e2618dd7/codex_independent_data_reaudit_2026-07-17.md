# Independent data and inference re-audit

Date: 2026-07-17  
Auditor: Codex, using a fresh Python reconstruction rather than the workbook's existing test suite  
Scope: source preservation, record identity, dates, AA duplicate policy, Epoch archive reconciliation, compute regressions, crowd pool, final formulas, workbook hashes, and forecast sensitivity

## Bottom line

The source-data layer passes. I independently re-parsed and compared all 5,807 observations to their upstream source records: 3,555 Epoch all-model rows, 1,690 Epoch archive-view rows, 211 ECI rows, 274 Artificial Analysis rows, 50 No-CoT paper records, and 27 METR records/law rows. There were zero raw-record mismatches, duplicate observation IDs, duplicate measurement IDs, orphan measurements, or JSON parse errors.

The inference layer passes arithmetically but not as a calibrated Bayesian posterior. The workbook exactly implements the stated judgmental log-pooling policy. Its 4.3T Fable and 2.9T Sol values reproduce to floating-point precision. However, the result is sensitive to modeling choices—especially the pooled No-CoT parameter elasticity for Fable—and several evidence branches are correlated. I endorse the displayed numbers as policy-weighted central estimates, not as precise or statistically calibrated parameter estimates.

## Severity summary

| Severity | Finding | Effect on displayed forecast |
|---|---|---|
| PASS | All 5,807 upstream/derived raw records were independently reconstructed with zero mismatches. | None |
| PASS | Epoch archive reconciliation independently reproduces 137/137 frontier, 1,035/1,035 notable, and 515/518 large-scale exact matches with zero shared-field disagreement. | None |
| PASS | The three large-scale rows absent from `all_ai_models.csv`—MAI-Thinking-1, MiMo-V2.5, and Hunyuan T1—are explicitly unmatched and preserved rather than forced. | None |
| PASS | AA's “highest score per identical display name” policy is implemented correctly for all 241 display names across 274 rows. | None |
| PASS | Crowd centers and the final 50/50 geometric ensemble reproduce exactly: Fable 4.269178T and Sol 2.870251T. | None |
| PASS | All listed workbook source-manifest hashes match; no spreadsheet formula-error token is present. | None |
| MAJOR | “Bayesian posterior” overstates the method. This is a judgmental logarithmic opinion pool with fixed weights, correlated evidence, and no fully specified likelihoods or posterior uncertainty. | Interpretation/confidence, not central arithmetic |
| MAJOR | The No-CoT branch uses the pooled 4.2× total-parameter factor per TH doubling. The paper itself reports 2.2× for dense models and 8.1× for MoEs, while Fable's architecture is unknown. | Fable moves from 3.9T to 4.7T under dense-versus-MoE sensitivity; Sol stays about 2.9T |
| MODERATE | The compute branch uses sequential residual regressions, which are order-dependent, rather than a simultaneous regression. A simultaneous refit changes raw compute priors by 2.00× for Fable and 1.47× for Sol. | Because compute is only 2.5% of the final pool, the final becomes 4.34T/2.90T—still 4.3T/2.9T at one decimal |
| MAJOR (contained) | Before its 6.51× two-anchor scale calibration, the compute chain predicts only 329B for K3 and 301B for Grok 4.5. After calibration it still deliberately splits the anchor error rather than hitting either anchor exactly. | Confirms that compute should remain a weak check; its 2.5% final weight limits the effect to a few percent |
| MODERATE | The 50% crowd allocation treats contributors equally, collapses ranges to geometric centers, and does not model correlation or forecaster calibration. | Central choice is explicit but not data-derived |
| MODERATE | Fable's AA=60 observation is explicitly a “with fallback” system configuration linked to the regular checkpoint. It is not a clean base-model measurement. | Confounds AA/post-training/system performance with pretraining scale; partly mitigated by branch diversification |
| MODERATE | Same-base claims for GPT-5 through 5.5 and Opus 4.5 onward are model assumptions supplied in the task, not established by the audited source files. | Makes the RL/pretraining decomposition conditional on those assumptions |
| MINOR | The final workbook's direct source-manifest sheet omits the AA attachment and METR snapshot, although their exact records and provenance are transitively bound by the hashed unified-observations file and fully listed in the unified source manifest. | Audit presentation only |
| MINOR | No-CoT floor values such as GPT-2's `<0.01` are represented as numeric 0.01 in typed columns, with censoring retained in raw text. | No effect; GPT-2 is excluded from the model |

## Source-data verification

### Exact preservation

| Source layer | Rows checked | Raw mismatches |
|---|---:|---:|
| Epoch `all_ai_models.csv` | 3,555 | 0 |
| Epoch frontier/notable/large-scale views | 1,690 | 0 |
| ECI Graph Data plus matching Regression Data records | 211 | 0 |
| Artificial Analysis leaderboard | 274 | 0 |
| No-CoT LaTeX tables and scaling-law macros | 50 | 0 |
| METR model snapshot plus trend-law record | 27 | 0 |
| **Total** | **5,807** | **0** |

The long measurement table contains 17,429 unique measurements. Every measurement references an existing observation; no correlated Epoch view row is marked for independent model-level inclusion.

### Epoch archive views

- `all_ai_models.csv` inside `ai_models.zip` is byte-identical to the primary Epoch snapshot.
- Frontier view: 137/137 exact-key matches; all shared fields agree.
- Notable view: 1,035/1,035 matches; all shared fields agree. One key has multiple master candidates and is resolved by zero shared-field disagreement.
- Large-scale view: 515/518 matches; all matched rows have zero shared-field disagreement. The three unmatched rows are preserved with explicit nonmatch status.

### Artificial Analysis identity layer

- 274 raw rows reduce to 241 unique display names only for model-level selection; no raw row is deleted.
- The selected row is the maximum AA score for every duplicated display name.
- Link statuses: 147 checkpoint rows, one checkpoint/system-configuration row, nine base-only rows, two family rows, six ambiguous records covering five ambiguous display names, and 109 records absent from Epoch after review.
- The five explicitly ambiguous display names are Gemini 2.5 Pro, Devstral 2, NVIDIA Nemotron 3 Nano, MiMo-V2-Flash (Feb 2026), and Olmo 3 7B. None is forced to an Epoch checkpoint.
- The sole low-confidence link is Command A+ → Cohere Command A, and it is correctly classified as base-only rather than a checkpoint match.

This validates structural correctness and explicit adjudication. It cannot prove that every manual semantic alias is factually true; that remains human judgment.

## Date audit

- All 211 ECI rows retain day-level source dates.
- AA supplies no release dates. Of its 274 rows, 170 acquire a canonical date through an exact ECI/Epoch/manual bridge and 104 remain undated rather than receiving an invented date.
- The target rows use day-level dates: Fable 2026-06-09, Sol/Terra/Luna 2026-07-09, Opus 4.8 2026-05-28, GPT-5.5 2026-04-23, Sonnet 5 2026-06-30, and Grok 4.5 2026-07-08 from ECI; Kimi K3 uses the user-supplied 2026-07-16 date.
- The No-CoT raw table correctly preserves the paper's month-only dates. The target projections use exact ECI endpoint dates, but the 373-day time-horizon law is inherited from the paper rather than independently refit with exact-day dates.
- All source-versus-canonical conflicts over 31 days are flagged and both values are preserved. The relevant No-CoT conflicts are DeepSeek V3.2, Kimi K2-0905, and Kimi K2.5; none is a Fable/Sol target row.

## Regression recomputation

### Epoch compute branch

The workbook's exact sequential procedure was independently rebuilt.

| Stage | n | In-sample R² | In-sample multiplicative RMSE | Simultaneous-OLS LOOCV multiplicative RMSE |
|---|---:|---:|---:|---:|
| AA/date → Epoch training compute | 31 | 0.467 | 4.72× | 4.68× |
| Compute/date → total parameters | 19 | 0.608 | 2.15× | 2.31× |

The compute prior has 0.888 log correlation with the existing benchmark branch. Its low 5% pre-crowd weight, or 2.5% final weight, is therefore justified. The regression nevertheless should be rewritten as a simultaneous model in a future methodological revision.

The uncalibrated chain is also badly mis-scaled: it predicts 328.8B for K3 and 301.2B for Grok 4.5. The 6.5126× calibration factor moves those to 2.14T and 1.96T, splitting the error around the disclosed 2.8T/1.5T anchors. This makes the branch a relative-ordering regularizer, not an independent absolute parameter estimator.

### No-CoT architecture sensitivity

Keeping every other weight unchanged and substituting the paper's architecture-specific total-parameter scaling factors gives:

| No-CoT elasticity choice | Fable horizon prior | Final Fable | Final Sol |
|---|---:|---:|---:|
| Dense: 2.2× params per TH doubling | 4.46T | 3.88T | 2.85T |
| Pooled: 4.2× | 6.53T | 4.27T | 2.87T |
| MoE: 8.1× | 9.63T | 4.70T | 2.89T |

This is not a confidence interval. It is a concrete model-specification sensitivity range. It shows that 4.3T is a reasonable midpoint for Fable under architecture uncertainty, while Sol is robust near 2.9T.

The pooled result is not the aggressive edge of this sensitivity. The paper's pooled 4.2× factor is almost exactly the geometric mean of its dense and MoE factors: `sqrt(2.2 × 8.1) = 4.22`. Likewise, the geometric midpoint of the dense and MoE final Fable forecasts is 4.27T. Therefore, under an equal log-prior over the two architecture-specific transport models, 4.3T is the model-averaged central value. If Fable is more likely MoE—as a multi-trillion-parameter frontier system plausibly is—the architecture prior would move upward, not downward. The larger unresolved question is whether either open-weight slope transports to Fable at all.

Fable's horizon input also extrapolates beyond the measured frontier range: K3's 2.4-minute value is an imputed pretrain-equivalent, and Fable's 3.61-minute value exceeds the paper's largest measured frontier point. The Ryan Greenblatt gap contributes half of Fable's horizon prior in log space—25% of the pre-crowd evidence model and 12.5% of the final crowd-and-model ensemble.

## Crowd and final-ensemble recomputation

- Registry: 14 named contributors; 13 usable Fable estimates and 13 usable Sol estimates.
- Exact geometric crowd centers: Fable 3.911661T; Sol 2.658296T.
- Evidence-model centers before crowd: Fable 4.659373T; Sol 3.099106T.
- Final 50/50 log pool: Fable 4.269178T; Sol 2.870251T.
- Workbook differences from the independent recomputation: Fable `8.9e-16` T; Sol exactly zero.

Effective final weights for Fable/Sol are 50% crowd, 25% No-CoT, 9.5625% AA, 9.5625% ECI, 3.375% API price, 2.5% Epoch compute, and 0% direct METR. METR is used only for the RL-versus-pretraining decomposition.

## Final judgment

Keep the displayed central estimates at **Fable 4.3T** and **Sol 2.9T** if the intended policy remains a 50% crowd / 50% model ensemble. The data pipeline is clean enough for that use. Do not describe these as precise parameter counts or a calibrated Bayesian posterior. “Crowd-and-evidence ensemble central estimate” is accurate.

The strongest unresolved uncertainty is not a missing or duplicated row. It is whether the pooled open-weight parameter/TH relationship is transportable to an undisclosed frontier architecture and how much the observed scores reflect post-training, scaffolding, and system configuration rather than pretraining scale.

## Reconciliation with the external Fable audit

The separately launched Claude Fable 5 audit agrees on all core data and arithmetic findings: strong structural-data pass, exact crowd and workbook calculations, weak/correlated compute evidence, uncalibrated weights, and a robust Sol estimate near 2.9T. It independently reproduced the same dense/pooled/MoE horizon sensitivity.

Fable prefers an approximately 3.8T central estimate for Fable because it discounts the entire horizon branch more heavily. I do not adopt that adjustment as the primary displayed result: the discount is judgmental and was not converted into an alternative explicit weight model, while its description of the pooled 4.2× factor as the aggressive choice is inconsistent with the paper's larger 8.1× MoE factor. A 3.8–4.0T conservative center is defensible if one assigns little transport weight to the No-CoT relation; 4.3T is the correct center under the workbook's declared weights and is almost exactly the dense/MoE log-model average. The disagreement is therefore about prior/model weights, not data correctness or arithmetic.

## Reproducibility artifacts

- Independent audit program: `run_codex_independent_reaudit.py`
- Machine-readable metrics: `codex_independent_reaudit_metrics_2026-07-17.json`
- Unified observations: `unified_model_observations_compute_enriched_2026-07-17.csv`
- Long measurements: `unified_model_measurements_long_compute_enriched_2026-07-17.csv`
- AA match audit: `aa_epoch_match_audit_compute_enriched_2026-07-17.csv`
- Epoch archive-view audit: `epoch_archive_view_match_audit_2026-07-17.csv`
- Final workbook: `frontier_parameter_model_crowd_50pct_2026-07-17.xlsx`
