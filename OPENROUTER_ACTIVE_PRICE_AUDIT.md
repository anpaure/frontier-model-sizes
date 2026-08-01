# OpenRouter active-parameter price audit

Snapshot status: frozen 2026-07-31 audit. Generated filenames retain their original `2026-07-18` suffix for compatibility; the embedded date and source hashes govern the actual vintage.

Current Kimi K3 primary evidence supersedes the snapshot's architecture uncertainty: Moonshot now discloses exactly 2.780T total and 104.2B activated parameters. The frozen active-price fit excluded Kimi from training and used only its total count as an external lock, so the historical numerical result remains reproducible; K3's activated count is the disclosed 104.2B, never the routed-fraction product `2780 × 16/896`.

## Decision

Current API price is plausibly more closely related to parameters activated per token than to a sparse model's full parameter inventory. The exact-identity audit supports that mechanism, but the evidence is not strong or prospective enough to change the live forecast. The branch remains a **0% incremental-weight diagnostic**.

The audited panel contains 63 active-parameter labels from 15 developers. It is constructed from 93 existing OpenRouter↔Epoch calibration checkpoints using only exact Hugging Face repository IDs and the independently audited AA↔Epoch checkpoint crosswalk. Forty-five labels are disclosed active counts from Artificial Analysis. Eighteen additional controls are models whose successfully parsed primary Hugging Face config is dense, so active parameters equal the Epoch total. The config collector covers 87 exact repositories: 73 configs parse successfully and 14 gated repositories remain explicitly unavailable. No HTTP failure is treated as dense. The two raw repository ambiguities are retained in the ledger; neither leaves an unresolved active-parameter match. There are no identity conflicts or duplicate checkpoints.

## Held-out evidence

The release-ordered developer-family holdout produces 45 predictions from 11 developers. Active parameters are easier to predict than total parameters across the tested specifications. With AA score, exact release date, and current blended price, median error is 1.77× for active parameters versus 2.42× for totals; the equal-developer difference is −0.118 log10 error with a 90% interval of −0.222 to −0.021. However, adding price to score and date reduces equal-developer active-parameter absolute log error by only 0.026, with a 90% interval of −0.127 to +0.088. Price's incremental contribution is therefore not separately established.

For the 16 held-out models whose disclosed total-to-active ratio is at least 15×, active prediction followed by a training-only sparse-MoE ratio gives 1.61× median error, versus 2.16× for directly predicting totals. The paired equal-developer improvement is −0.117 log10 error, but its 90% interval is −0.275 to +0.042. This misses both predeclared promotion requirements: the interval must be wholly favorable, and the comparison must contain at least 20 tests from eight developers. It has 16 tests from seven.

## Frontier sensitivity

A common 55-checkpoint, 13-developer training fit excludes Anthropic, OpenAI, Kimi/Moonshot, and all target-dated observations. Relative to K3's exact disclosed 2.780T total anchor (displayed as 2.8T), the score/date/price active-capacity sensitivity implies 5.6T for Claude Fable 5 and 3.9T for GPT-5.6 Sol.

These are not live estimates. Fable's current price is 6.5× the maximum training price and Sol's is 3.7×; even K3 is 1.9×. The sensitivity is strongly extrapolative, overlaps the existing AA/date and API-price branches, and receives 0% weight.

## Historical-price follow-up

The original audit's main limitation has now been directly tested. A separate hash-pinned ledger reconstructs all 2,566 price changes from 1,062 official OpenRouter model-catalog snapshots. Across predeclared 1–90 day launch windows, launch-vintage price robustly beats date alone for total parameters. On the exact active-parameter/AA common panel, however, the developer-bootstrap interval for price added to score and date crosses zero in every window. The prospective result therefore validates price as a correlated mechanism but does not promote the active-price branch or add another live weight. See `OPENROUTER_HISTORICAL_PRICE_AUDIT.md`.

## Reproducible artifacts

All dated artifacts in this section intentionally retain their `2026-07-18` compatibility names; their embedded observation date is 2026-07-31.

- `analyze_openrouter_active_price_signal.py`: exact joins, backtest, target sensitivity, promotion gates, hashes.
- `openrouter_active_parameter_match_audit_2026-07-18.csv`: all 93 calibration rows and every identity decision.
- `huggingface_architecture_config_snapshot_2026-07-18.json.gz`: verbatim primary configs and explicit HTTP failures for all 87 repositories.
- `huggingface_architecture_config_signals_2026-07-18.csv`: nested expert fields and conservative dense/MoE/unavailable classifications.
- `huggingface_architecture_config_collection_audit_2026-07-18.json`: inventory, HTTP statuses, classifications, and source hashes.
- `openrouter_active_price_predictions_2026-07-18.csv`: all 45 release-ordered held-out predictions.
- `openrouter_active_price_targets_2026-07-18.csv`: K3-anchored frontier sensitivities and extrapolation ratios.
- `openrouter_active_price_audit_2026-07-18.json`: machine-readable results, caveats, and decision.
- `tests/test_openrouter_active_price_signal.py`: identity, chronology, arithmetic, hash, gate, and target tests.
