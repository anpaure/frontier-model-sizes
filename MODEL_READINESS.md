# Current frontier-parameter model readiness

Generated automatically from the audited pipeline on 2026-07-31.

## Current forecasts

| Model | Evidence | Crowd | Final | Empirical 50% band | Empirical 80% band |
|---|---:|---:|---:|---:|---:|
| Claude Fable 5 | 4.7T | 4.4T | **4.5T** | 2.0T–11.3T | 0.9T–25.0T |
| GPT-5.6 Sol | 3.1T | 3.2T | **3.1T** | 1.0T–9.9T | 0.6T–16.3T |
| Claude Opus 5 | 3.0T | — | **3.0T** | 1.0T–9.1T | 0.6T–15.8T |

The intervals are calibrated around the evidence centers. Crowd forecasts shift Fable/Sol's displayed centers but do not narrow coverage.

## Precision

- Frontier lineage holdout: 27 predictions, median error 2.02×, 48% within 2× and 74% within 3×.
- Whole-developer frontier holdout: 27 predictions, median error 1.89×.
- Latest residual per developer under whole-developer refits: 11 developers; median factor 3.02×.
- Conservative envelope across lineage- and whole-developer-holdout specifications: 3.02× at 50%, 5.28× at 80%, and 6.19× at 90%.
- Kimi K3 audit correction: the exact-score AA fit with all Kimi lineages held out predicts 2.24T (1.24×). The prior 0.385T / 7.23× row used only ECI and speculative compute because K3 was external to the AA target table; after incorporating the leakage-safe AA component, the available-component result is 0.839T / 3.31×.
- Practical reading: row-level point errors are roughly factor-of-two, but a new developer's empirical central band is closer to factor-of-three and the tails are much wider. These are prequential error bands, not formal conformal coverage guarantees.

## External and vintage validation

- Frozen ECI form on four retrospective interval targets: median error 1.46×; 100% within 2×.
- Kimi K3 later score-vintage check: 2.06× error; not project-prospective because K3's size was already known.
- Four-model common GPQA/MATH/AIME panel: median error 3.08×, 25% within 2×; live weight 0% after fixed gates fail.
- Vintage knowledge challenger: 18 rows/5 developers, 3.65× → 3.49×; live weight 0% because coverage gates fail.
- High-sparsity shrinkage challenger: 47 rows/15 developers, 1.99× → 1.77×; live weight 0% because target sparsity and preregistration gates fail.

## Prospective commitment

- Freeze `frontier-parameters-2026-07-31-v1` is `LOCKED_PRE_DISCLOSURE` for Claude Fable 5, GPT-5.6 Sol, Claude Opus 5.
- Artifact SHA-256: `e859c32aa80a0b6bb0f0afec6654135c5a75f21163d3ea75032f8754a537912d`; post-outcome refitting is **FORBIDDEN**.
- The frozen point centers exactly equal the current centers. The table's uncertainty bands are later, corrected diagnostics; prospective interval scoring must use the immutable bands inside the freeze artifact, which was not rewritten.
- Poll identities are privacy-redacted to stable anonymous respondent IDs in the public freeze. The redacted artifact preserves the prior digest and all numerical fields, and the project retains no name-to-ID mapping.

## Retained decisions

- Active-parameter recovery has a 90% developer interval [-0.093, -0.000] (wholly favorable), while total transport worsens median error (1.99× → 2.10×) and target sparsity is unobserved; live weight remains 0%.
- Direct factor-weight optimization worsens median error on 32 outer tests (2.14× → 2.32×); weights remain unchanged.
- ECI architecture-blend challenger: fixed whole-developer median improves on 69 rows from 2.54× to 2.36×, but within-2× accuracy worsens, both frontier developer intervals cross zero, K3 error worsens, and target architecture is unobserved; live weight remains 0%.
- LongCat parameter-definition audit: retain the publisher's 1.6T semantic model total; the exact 1.776T serialized inventory falls to 1.639T after excluding MTP tensors. The alternative moves legacy target fits by at most 0.83%, leaves the matched ensemble invariant, and receives 0% live weight.
- AA calibration view: 587 raw configurations, 335 eligible configurations, 275 checkpoints, and 48 creators after 2 explicit primary-source overrides.
- AA parameter-label timing: 6 pinned records, of which 5 become eligible after nominal model release; chronological folds use the later date while the current fit is unchanged.
- AA score-publication timing: 28 of 50 live AA checkpoints have verified non-null changelog dates and 16 were published after nominal release. Correcting information dates changes the all-ensemble audit from 43 rows/2.15× to 44 rows/2.14×, while the frontier median changes from 2.09× to 2.02×; centers and live weights remain unchanged.
- Parameter-truth reconciliation: 3 narrow primary-source overlays canonicalize Moonshot K2 and MiniMax M2.5/M2.7 coarse labels while preserving raw values and all distinct checkpoints; no global match tolerance is widened.
- Crowd center robustness: leave-one-contributor-out final ranges are 4.5T–4.7T for Fable and 3.1T–3.3T for Sol. The 18 paired contributors have log-point correlation 0.77; crowd agreement remains correlated and does not narrow intervals.

## Bottom line

Ready for comparative forecasting and sensitivity analysis; not precise enough to claim literal hidden counts to a decimal place. The strongest remaining validation is a future unadjusted comparison against a parameter disclosure for Fable, Sol, or Opus 5.
