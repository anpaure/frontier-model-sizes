# OpenRouter historical launch-price audit

Snapshot status: frozen through 2026-07-31. Generated filenames retain their original `2026-07-18` suffix as a compatibility contract; the metadata and source hashes govern the actual vintage. Current Kimi K3 primary evidence gives exactly 2.780T total and 104.2B activated parameters; this historical price audit uses K3 only as a total-parameter anchor.

## Decision

Historical launch-vintage price contains real information about total parameter scale beyond release date alone. That result is robust across all seven predeclared price windows. On the stricter active-parameter panel, however, launch price does not add robust information beyond Artificial Analysis score and exact release date. The audit therefore validates the existing API-price mechanism but contributes **0% additional live weight** and leaves the headline forecasts unchanged.

## Source integrity

The source is pinned to `jvrck/openrouterlist` commit `1cd0e2ec5fccb271df9d1140abc91aaf20b3e878`. That repository archives the official public `https://openrouter.ai/api/v1/models` response approximately twice daily. The compact ledger contains 914 OpenRouter model IDs and 2,566 ordered price-change points from 2024-09-21 through 2026-07-31, including 226 free or partly-free states, five unavailable states, and 80 additional same-day changes.

A clean rebuild from all 1,062 upstream snapshot commits produced the same raw ledger SHA-256, `d572cdb4b64b0b4ed878e570490b937666a09b49f5cef1199dc8869825473acc`. The frozen pipeline is offline and verifies this hash. Every model, change index, first/last-seen date, prompt price, completion price, free state, unavailable state, and same-day ordering is retained.

## Identity and timing rules

All 93 regression checkpoints and their OpenRouter aliases are audited explicitly. Every alias exists in the historical ledger; there are no duplicate checkpoint IDs and no fuzzy joins. A checkpoint is eligible only when its release is on or after the 2024-09-21 history floor and its absolute OpenRouter onboarding lag is at most 30 days. Thirteen earlier or late-onboarded checkpoints remain in the match ledger but are explicitly excluded.

The predeclared windows are 1, 3, 7, 14, 30, 60, and 90 days. A model's window price is the equal-day median of positive prompt/output geometric means over the complete window. The feature becomes available only at the window end. Within every held-out fold:

- the test developer is excluded from training;
- every training price-availability date strictly precedes the test availability date;
- free and unavailable states are preserved but never log-modeled;
- the 2026-07-31 current price appears only as an explicitly nonprospective comparator.

The 1–30 day panels contain 73 total-parameter checkpoints and 55 active-label checkpoints. The 60-day panel contains 70 and 52; the 90-day panel contains 68 and 51. Chronology and developer exclusion produce 3,098 recorded held-out predictions across all panels and specifications.

## Held-out results

For total parameters, date plus historical price beats date alone in every window. The equal-developer paired absolute-log10-error deltas range from −0.221 to −0.329. Every 90% developer-bootstrap interval is wholly below zero; the least favorable upper bound is −0.064. The result is therefore not an artifact of using today's repriced catalog.

On the exact active-parameter/AA-score common panel, score + date + historical price never has a 90% developer-bootstrap interval wholly below zero relative to score + date. The point delta ranges from a small improvement to a small worsening depending on the window, and every interval crosses zero. Price is informative, but it overlaps the capability/date signal and does not earn a second independent mixture weight.

## Frontier sensitivity

Using only the first complete launch day and excluding Anthropic, OpenAI, and Kimi from the common training fit, a K3-anchored score/date/price sensitivity gives:

- Claude Fable 5: $10/M input, $50/M output, 5.4T sensitivity;
- GPT-5.6 Sol: $5/M input, $30/M output, 3.8T sensitivity;
- Kimi K3: $3/M input, $15/M output, fixed at its exact disclosed 2.780T total anchor (displayed as 2.8T).

These are zero-weight sensitivities, not headline estimates. They are recorded because they are useful directional checks; the combined final forecast remains 4.5T for Fable and 3.1T for Sol.

## Reproducible artifacts

- `collect_openrouter_historical_prices.py`: hash-pinned refresh, frozen offline rebuild, validation, deterministic gzip, and lossless change-point normalization.
- `sources/openrouter_historical_price_ledger_2026-07-18.json.gz`: pinned compact raw ledger.
- `sources/openrouter_historical_price_change_points_2026-07-18.csv`: all 2,566 ordered price states.
- `sources/openrouter_historical_price_collection_metadata_2026-07-18.json`: commit, blob, hashes, full-history rebuild verification, inventory, and no-loss policy.
- `analyze_openrouter_historical_price_signal.py`: exact joins, complete-window feature construction, prospective holdouts, developer bootstraps, and frontier sensitivities.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/openrouter_historical_price_match_audit_2026-07-18.csv`: all 93 checkpoint identities and every window eligibility decision.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/openrouter_historical_price_backtest_predictions_2026-07-18.csv`: all 3,098 held-out predictions.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/openrouter_historical_price_frontier_targets_2026-07-18.csv`: exact first-day frontier prices and sensitivities.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/openrouter_historical_price_audit_2026-07-18.json`: machine-readable metrics, bootstraps, decision, and hashes.
- `tests/test_openrouter_historical_price_signal.py`: source inventory, identity, chronology, held-out, decision, and target tests.
