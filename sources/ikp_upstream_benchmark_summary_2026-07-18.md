# Benchmark comparison results

## Setup
- 81 vendor-published benchmark scores collected from primary sources (model cards, system cards, technical reports, vendor blog posts) — see `raw_*.md` files for citations.
- Joined against `data/densing_analysis_data.csv` (89 calibration models after applying the same exclusion list as `loo_cv_analysis.py`).
- For each benchmark, fit OLS `score ~ log10(params)` and `score ~ log10(params) + months`. The IKP fit is run on the same model subset for an apples-to-apples comparison.

## Headline numbers (with paper's standard calibration excludes)

| Metric | n | R² vs log₁₀(params) | IKP R² (same n) | Time slope (pp/month) |
|---|---:|---:|---:|---:|
| **IKP (full set)** | **89** | **0.917** | — | **−0.06** |
| SimpleQA | 10 | 0.904 | 0.991 | +0.03 |
| MMLU | 30 | 0.705 | 0.886 | +0.58 |
| MMLU-Pro | 25 | 0.689 | 0.900 | +0.82 |
| GPQA Diamond | 30 | 0.520 | 0.903 | +1.99 |

## Interpretation

1. **R² gap is real.** On every matched subset, IKP explains substantially more variance in log-parameters than the standard benchmark does. GPQA Diamond is the worst proxy at R² = 0.52 (vs. IKP 0.90 on the same 30 models).
2. **Time drift confirms compressibility.** Standard benchmarks gain 0.6–2.0 percentage points per month at fixed params, consistent with the Densing-Law thesis. IKP time slopes hover near zero (−0.02 on the full set; small/noisy on subsets). Reasoning-heavy benchmarks (GPQA at +1.99 pp/mo) drift fastest — a 33B model improves ~24 points across one year of releases.
3. **SimpleQA is the closest analog to IKP** — it's the only standard benchmark targeting pure factual recall, and its R² (0.90) and time slope (~0) most closely resemble IKP. The residual gap is exactly the long-tail vs. head-of-distribution distinction IKP was designed to isolate. (Sample only n=10, so treat as suggestive.)
4. **Sample-selection asymmetry.** Vendors only report MMLU-Pro / GPQA / SimpleQA on roughly the upper half of the parameter range. IKP has identical methodology across 96 models from 135M to 1.6T, including the small/older models that vendors quietly drop from benchmark tables. This is itself a methodological argument for IKP.

## Files

- `benchmark_scores.csv` — collected vendor scores keyed by `model`
- `regression_summary.csv` — R²/slope per benchmark
- `time_coefficients.csv` — joint log-params + time fits
- `joined_per_model.csv` — full per-model table for the appendix
- `raw_anthropic.md`, `raw_openai.md`, `raw_google_meta.md`, `raw_deepseek_qwen_kimi_glm.md`, `raw_others.md` — source tables with primary URLs
- `paper/figures/benchmark_comparison.{pdf,png}` — 2×3 panel scatter
- `scripts/16_benchmark_comparison.py` — analysis script
