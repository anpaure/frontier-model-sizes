# Artificial Analysis parameter-panel expansion audit

Source snapshot: 2026-07-31

## Decision

Retain the current Kimi-K3-anchored Artificial Analysis branch. The expanded panel is a valuable robustness dataset, but it receives 0% incremental live weight until a prospective high-score disclosure distinguishes its steeper score and date slopes.

## Data reconciliation

The live AA calibration contains 50 manually curated recent open-weight checkpoints. The canonical AA↔Epoch megafile independently yields 63 exact open-weight checkpoints after applying all of these rules:

- model-level AA row selected under the highest-score-per-display-name rule;
- unique, high-confidence checkpoint-level Epoch link;
- unique Epoch model label;
- exact Epoch release-date and parameter agreement;
- open model weights;
- highest AA score retained when several AA configurations map to one checkpoint.

Nine lower-scoring duplicate configurations are discarded. Nineteen exact checkpoints overlap the live panel; normalized exact model labels plus the manually verified `Nemotron 3 Ultra 550B A55B` / `Nemotron 3 Ultra` alias resolve them. The exact Epoch-backed row supersedes each duplicate, producing 94 unique models across 27 developers. Kimi K3 is the one parameter override: the official report's exact 2.780T replaces Epoch's rounded 2.800T value.

## Held-out test

The predeclared specification is the same direct orientation used by the main leakage audit:

`log10(total parameters) ~ AA score + exact release date`

For every test checkpoint, training rows must be strictly earlier and the test developer is entirely removed. Training gives equal total weight to every developer; AA scores marked with an asterisk receive half weight. The current 50-row panel and expanded 94-row panel predict the identical 44 eligible test checkpoints.

Across all 44 tests, median error improves from 2.11× to 1.85×, mean absolute log10 error from 0.529 to 0.411, and p80 error from 7.60× to 5.25×. The equal-developer bootstrap 90% interval is fully favorable. On the 33 checkpoints represented in the original panel, mean and tail errors improve, although median error is essentially unchanged at 2.11× versus 2.16×.

The relevant frontier-like subset is much weaker evidence: 16 tests across only seven developers. Median error improves from 1.84× to 1.79×, but mean error is slightly worse, p80 error rises from 2.17× to 3.13×, and the equal-developer interval crosses zero. The extension mainly fixes broad small/mid-scale extrapolation failures; it has not demonstrated a better frontier mapping.

## Coefficient and anchor checks

The expanded full fit is materially steeper:

- score slope: 0.0582 versus the current live 0.0452 log10 parameters per AA point;
- date slope: −0.7206 versus the current live −0.3509 log10 parameters per year.

The developer bootstrap for the expanded date slope remains broad (5th–95th percentile approximately −1.24 to −0.21). Before K3 intercept calibration, the expanded fit predicts K3 at 4.24T, or 3.99T with Moonshot held out, versus the exact disclosed 2.780T. The current panel predicts 2.67T and 2.36T respectively.

After forcing both variants through K3's exact 2.780T anchor, the expanded coefficients would move the AA-only Fable estimate from 4.08T to 4.84T and Sol from 3.44T to 3.70T. It improves the disclosed Grok 4.5 check modestly (2.05T to 1.90T against 1.5T), but one closed-model check does not overcome the neutral frontier-like backtest and worsened K3 pre-calibration check.

## Limitations

- AA scores are from the current leaderboard snapshot, not launch-date benchmark vintages.
- AA does not provide a machine-readable reasoning-budget field for every row, so the main comparison avoids a subjective reasoning label.
- Developer holdout cannot eliminate cross-developer distillation or algorithm diffusion.
- Only 16 frontier-like predictions across seven developers are eligible.

## Reproducible artifacts

- `run_aa_expanded_parameter_audit.py`
- `sources/aa_detailed_snapshot_manifest_2026-07-31.json`
- `sources/kimi_k3_release_evidence_2026-07-31.json`
- `tests/test_aa_expanded_parameter_audit.py`

Generated expansion-audit outputs currently retain a legacy `2026-07-18` filename suffix for compatibility. Their embedded `generated_on` date is `2026-07-31`; current tests bind the 94-row expanded panel, exact K3 override, and source hashes to that date.
