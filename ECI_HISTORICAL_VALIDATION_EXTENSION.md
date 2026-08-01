# ECI historical validation extension

This is a validation-only extension. It cannot tune candidates, select weights, or change the live forecast.

## Four historical interval targets

The expanded retrospective interval set spans Moonshot, xAI, and Z.ai. GLM-5.2 is admitted by exact timestamps: its public Hugging Face repository appeared at 07:20:33Z, after the prior Epoch capture at 02:44:13Z on the same day.

For the frozen live 60/40 form under inverse-ECI-CI weighting, median multiplicative error is 1.46x and 100% fall within 2x.

- Kimi K2.5: 0.630T predicted vs 1.040T disclosed (1.65x error).
- Kimi K2.7 Code: 0.799T predicted vs 1.040T disclosed (1.30x error).
- Grok 4.5: 1.362T predicted vs 1.500T disclosed (1.10x error).
- GLM-5.2: 1.227T predicted vs 0.744T disclosed (1.65x error).

## Kimi K3

K3 is a score-vintage holdout only: 1.353T predicted vs 2.780T disclosed. It is not project-prospective because the project already used its disclosed size before incorporating the July 22 score vintage.

## Decision

No live-weight or functional-form change is permitted from this small retrospective sample. The four-model GPQA/MATH/AIME check is implemented as a separate zero-weight audit and is never mixed into this aggregate result.
