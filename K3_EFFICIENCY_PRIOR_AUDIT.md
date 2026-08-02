# K3 relative-efficiency reference audit

Generated 2026-08-01. The center-preserving projection is **live for upper-tail sensitivity at 80% strength** and has **0% point-center weight**.

## Contract

The assumption ‘at least as parameter-efficient as Kimi K3’ is one-sided. For a fixed capability mapping it implies `target parameters <= K3-relative parameter equivalent`. It does not imply that a target is at least 2.78T; that is a separate size-floor assumption.

The retained models are linear in log10(parameters), so they already encode diminishing returns in raw parameters. No quadratic, ridge-flexible, spline, or score-asymptote curve enters this reference.

| Target | AA equivalent | Canonical ECI equivalent | Pooled reference median | Reference 10–90% |
|---|---:|---:|---:|---:|
| Claude Fable 5 | 3.70T | 5.07T | 4.11T | 3.55–4.94T |
| GPT-5.6 Sol | 3.35T | 5.69T | 3.92T | 3.42–4.55T |
| Claude Opus 5 | 4.04T | 4.33T | 4.00T | 3.49–4.65T |

The pooled reference distribution propagates current ECI confidence intervals and five retained linear slope sensitivities. AA and ECI receive equal log weight; they are explicitly correlated alternative mappings, not independent likelihoods. Their geometric mean is not a strict ceiling under either mapping.

## Support audit

K3 is excluded from the ECI slope fit. The largest remaining calibration ECI is 151.37; K3 is 4.23 points beyond it. Fable, Sol, and Opus 5 are respectively 9.36, 10.36, and 8.01 points beyond it. The linear transport is therefore an explicit extrapolative sensitivity, not a newly validated frontier law.

## Integration decision

- Preserve every evidence center and the exact 50% crowd blend.
- Winsorize only draws above the evidence center; lower tails are unchanged.
- When the pooled reference is below the center, preserve the center and record that override explicitly.
- Publish both raw empirical intervals and center-preserving K3-efficiency projection intervals.
- The projected intervals are not conditioning, conformal intervals, or Bayesian credible intervals.
- The user-supplied Sol < Fable ordering is applied to the Sol reference draws only; it does not alter centers or enforce actual-size ordering.

## Why the rejected nonlinear result is absent

The ECI functional-form tournament did not promote its flexible ridge challenger. The earlier 8.5–12.2T K3-rebased result transported an out-of-support quadratic derivative and K3's entire residual. It remains a rejected stress test and contributes exactly zero weight here.
