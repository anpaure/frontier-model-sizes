# Active-parameter shrinkage challenger

Status: deterministic retrospective audit; **0% live weight**.

The current AA architecture audit produces two strictly chronological,
whole-developer-held-out estimates for a sparse checkpoint's total parameter
count:

- a direct total-parameter regression; and
- a predicted-active-parameter estimate transported through a
  training-fold-only high-sparsity ratio.

This challenger takes the geometric 50/50 mean of those estimates. It is
evaluated only on disclosed checkpoints with total/active ratio at least 15x.
The direct and transported estimates share AA score/date inputs, so this is a
shrinkage rule, not independent evidence.

Scoring uses the canonical parameter-truth overlay: 1.040T total / 32.6B
active for the publisher-reported Kimi K2 family, and exactly
228.703644928B total / 10B active for each MiniMax M2.5 and M2.7 official
Hugging Face tensor inventory. Rounded source labels remain preserved, and
distinct checkpoints are not deduplicated merely because they share a truth.
The overlay corrects outcomes consistently; it was not used to choose a live
weight after seeing the revised errors.

## Results

On 47 held-out high-sparsity checkpoints from 15 developers, the fixed blend
changes median error from **1.99x to 1.77x**, mean absolute log10 error from
0.403 to 0.329, 80th-percentile error from 4.60x to 3.37x, and within-2x
accuracy from 51.1% to 59.6%. The equal-developer paired change in mean
absolute log10 error is -0.082, with a 90% developer-cluster bootstrap interval
of **[-0.128, -0.037]**.

On the 28 frontier-like rows from 10 developers, median error changes from
**1.89x to 1.73x**, within-2x accuracy from 60.7% to 67.9%, and the paired
equal-developer change is -0.063, with 90% interval **[-0.090, -0.037]**.

A second-level nested chooser uses only earlier eligible held-out predictions,
removes the target developer again, and chooses among active weights
0/25/50/75/100%. It covers 31 rows from 14 developers (21 frontier rows from
9 developers). Its equal-developer change is -0.062 overall, 90% interval
**[-0.110, -0.016]**, and -0.058 on the frontier subset, interval
**[-0.105, -0.011]**.

Kimi K3 is an external scale check for the component fits: Kimi is excluded
from training, and only data available before its release enter the fold. The
direct branch predicts 5.73T and the active-transport branch 1.42T. Their fixed
geometric mean is 2.848T, a 1.025x error against the exact primary-source 2.780T
total. This is still retrospective because the shrinkage rule was examined
after K3's disclosure. The audit transparently replaces AA's rounded
2.8T/104B metadata with the pinned exact 2.78T/104.2B truth when scoring K3.

## Why it is not live

The empirical gates pass, but the applicability gates do not. Membership in
the >=15x cohort is known here only because both total and active counts were
eventually disclosed. Fable, Sol, and Opus 5 have no independent pre-outcome
active fraction or equivalent architecture-to-sparsity observation in the
project data. Applying the favorable conditional error to them would therefore
convert an architecture belief into seemingly measured evidence.

The challenger can receive nonzero weight only after a pre-outcome source,
independent of the target total-parameter label, establishes qualifying
high-sparsity architecture, and after the frozen 50/50 rule is scored without
refitting on disclosures from at least three new developers.

Reproduce with:

```bash
python3 analyze_active_parameter_shrinkage_challenger.py
python3 -m unittest -v tests.test_active_parameter_shrinkage_challenger
```

Machine-readable outputs:

- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/active_parameter_shrinkage_challenger_2026-07-31.json`
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/active_parameter_shrinkage_challenger_predictions_2026-07-31.csv`
