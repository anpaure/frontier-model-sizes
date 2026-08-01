# Crowd robustness audit

The 50% crowd layer for Fable and Sol is a judgmental ensemble policy, not a statistically independent likelihood. `analyze_crowd_robustness.py` rebuilds its active pools directly from the append-only anonymous forecast ledger, verifies the supersession graph and one-active-forecast-per-respondent rule, and reconciles the exact centers to the generated site contract.

The centers are not controlled by one forecast. Removing any one respondent moves Fable's final from 4.55T into a 4.45–4.73T range and Sol's from 3.15T into a 3.08–3.27T range. Median and 10%-per-tail log-trimmed centers are retained as estimator sensitivities. This is reassuring about outlier robustness even though individual points span more than 9× for each target.

Dependence remains substantial. Eighteen anonymous respondents forecast both targets, and their log-point correlation is about 0.77. Most entries were relayed into the project rather than collected under a blinded, randomized elicitation protocol. The crowd can therefore shift the displayed center under the user's explicit 50/50 policy, but it cannot narrow the empirical parameter intervals or validate the regression by agreement alone. Stable IDs preserve paired analysis; no name-to-ID mapping is stored.

Generated artifact: `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/crowd_robustness_audit_2026-07-31.json`.
