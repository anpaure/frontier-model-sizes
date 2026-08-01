# ECI historical common-component audit

This rolling-origin panel uses exactly GPQA diamond, MATH level 5, and OTIS Mock AIME 2024-2025. The fold ledger fixes all targets, versions, snapshots, developer exclusions, family exclusions, and benchmark selection before fitting.

Each target is scored in its first later pinned snapshot. Training parameters and scores come only from the prior snapshot, training releases are strictly earlier, and any row matching the target developer or base-family token is removed. Canonical model names are then collapsed to the latest eligible scored version.

The primary equal-developer 60/40 component/date blend has 3.08x median error and 25% within 2x.

- Gemma 3 27B: 89.0B predicted vs 27.0B snapshot truth (3.30x error).
- Mistral Small 3.1: 28.4B predicted vs 24.0B snapshot truth (1.19x error).
- Llama 4 Scout: 37.8B predicted vs 109.0B snapshot truth (2.88x error).
- Llama 4 Maverick: 113.7B predicted vs 400.0B snapshot truth (3.52x error).

## Promotion decision

Live weight remains 0%. Failed fixed gates: target_count, target_developer_count, within_2x, median_error, project_preregistration.

These four retrospective observations are a useful failure-mode check, not independent evidence precise enough to tighten frontier parameter forecasts.
