#!/usr/bin/env python3
"""Leakage-controlled tests of ECI component benchmarks for parameter inference.

The aggregate ECI and each component benchmark are compared on exactly the same
outer folds.  Each outer prediction uses only strictly earlier releases and
removes the test model's entire family from training.  Component scores are
mapped to the official ECI scale with Epoch's published IRT difficulty/slope
parameters before the parameter-count regression is fit.

This is still pseudo-chronological: it uses the current benchmark snapshot, not
historical benchmark vintages.  Per-benchmark rankings are exploratory and are
not used to change the live forecast weights automatically.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from artifact_paths import portable_path


WORK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORK_DIR / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
REGRESSION_PATH = WORK_DIR / "regression_results.json"
COMPONENT_PATH = WORK_DIR / "sources" / "epoch_eci_benchmarks_2026-07-31.csv"
BENCHMARK_ZIP_PATH = WORK_DIR / "sources" / "epoch_benchmark_data_2026-07-31.zip"
EPOCH_MODELS_PATH = WORK_DIR / "sources" / "epoch_all_ai_models_2026-07-31.csv"
RESULT_PATH = OUTPUT_DIR / "eci_component_chronological_backtest_2026-07-18.json"
COMPARISON_PATH = OUTPUT_DIR / "eci_component_benchmark_comparison_2026-07-18.csv"
PREDICTION_PATH = OUTPUT_DIR / "eci_component_backtest_predictions_2026-07-18.csv"
SENSITIVITY_PATH = OUTPUT_DIR / "eci_component_same_base_sensitivity_2026-07-18.csv"
FRONTIER_PATH = OUTPUT_DIR / "eci_component_frontier_estimates_2026-07-18.csv"

DATE_ORIGIN = date(2023, 1, 1)
MIN_TRAIN = 12
MIN_FAMILIES = 5
MIN_COMPONENT_ROWS = 16
MIN_ELIGIBLE_PREDICTIONS = 8

# Predeclared before looking at the component backtest results.
KNOWLEDGE_BENCHMARKS = {
    "MMLU",
    "GPQA diamond",
    "SimpleQA Verified",
    "TriviaQA",
    "OpenBookQA",
    "ScienceQA",
}
PRETRAINING_LIKE_BENCHMARKS = KNOWLEDGE_BENCHMARKS | {
    "ARC AI2",
    "BBH",
    "GSM8K",
    "HellaSwag",
    "LAMBADA",
    "PIQA",
    "Winogrande",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _years(value: str) -> float:
    return (date.fromisoformat(value) - DATE_ORIGIN).days / 365.25


def _logit(value: float, epsilon: float = 1e-3) -> float:
    clipped = min(1.0 - epsilon, max(epsilon, float(value)))
    return math.log(clipped / (1.0 - clipped))


def _family_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = Counter(row["family"] for row in rows)
    weights = np.asarray([1.0 / counts[row["family"]] for row in rows], dtype=float)
    return weights / weights.mean()


def _design(rows: list[dict[str, Any]], signal: str, include_date: bool) -> np.ndarray:
    matrix = []
    for row in rows:
        values = [1.0, float(row[signal])]
        if include_date:
            values.append(_years(row["release_date"]))
        matrix.append(values)
    return np.asarray(matrix, dtype=float)


def _fit_predict(
    train: list[dict[str, Any]],
    test: dict[str, Any],
    signal: str,
    include_date: bool,
) -> float:
    x = _design(train, signal, include_date)
    y = np.log10(np.asarray([row["total_b"] for row in train], dtype=float))
    weights = _family_weights(train)
    root = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)
    return float((_design([test], signal, include_date) @ beta).item())


def _fit_predict_live_wls(
    train: list[dict[str, Any]],
    test: dict[str, Any],
    include_date: bool,
) -> float:
    """Reproduce the live ECI branch's source-row weights inside each outer fold."""
    x = _design(train, "eci", include_date)
    y = np.log10(np.asarray([row["total_b"] for row in train], dtype=float))
    weights = np.asarray([row["workbook_wls_weight"] for row in train], dtype=float)
    root = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)
    return float((_design([test], "eci", include_date) @ beta).item())


def _metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    if not rows:
        return {f"{prefix}_n": 0}
    errors = np.asarray([row[f"{prefix}_log10_error"] for row in rows], dtype=float)
    absolute = np.abs(errors)
    return {
        f"{prefix}_n": int(len(rows)),
        f"{prefix}_families": int(len({row["family"] for row in rows})),
        f"{prefix}_median_factor": float(10 ** np.median(absolute)),
        f"{prefix}_geomean_factor": float(10 ** np.mean(absolute)),
        f"{prefix}_rmse_log10": float(np.sqrt(np.mean(errors**2))),
        f"{prefix}_p80_factor": float(10 ** np.quantile(absolute, 0.80)),
        f"{prefix}_bias_factor": float(10 ** np.mean(errors)),
        f"{prefix}_within_2x": float(np.mean(absolute <= math.log10(2.0))),
    }


def _cluster_bootstrap_delta(
    rows: list[dict[str, Any]], samples: int = 4000, seed: int = 20260718
) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    families = sorted(by_family)
    if len(families) < 2:
        return {
            "paired_mean_abs_log10_delta_component_minus_eci": None,
            "paired_delta_ci_low": None,
            "paired_delta_ci_high": None,
            "bootstrap_probability_component_better": None,
        }

    def delta(sampled: list[str]) -> float:
        selected = [row for family in sampled for row in by_family[family]]
        return float(
            np.mean([abs(row["component_log10_error"]) for row in selected])
            - np.mean([abs(row["eci_log10_error"]) for row in selected])
        )

    observed = delta(families)
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [delta(list(rng.choice(families, size=len(families), replace=True))) for _ in range(samples)]
    )
    return {
        "paired_mean_abs_log10_delta_component_minus_eci": observed,
        "paired_delta_ci_low": float(np.quantile(draws, 0.05)),
        "paired_delta_ci_high": float(np.quantile(draws, 0.95)),
        "bootstrap_probability_component_better": float(np.mean(draws < 0)),
    }


def _load_difficulties() -> dict[str, tuple[float, float]]:
    with zipfile.ZipFile(BENCHMARK_ZIP_PATH) as archive:
        with archive.open("additional_eci_data/eci_benchmark_difficulties_and_slopes.csv") as handle:
            frame = pd.read_csv(handle)
    return {
        str(row.benchmark_name): (float(row.edi), float(row.estimated_slope_scaled))
        for row in frame.itertuples()
    }


def _load_open_models() -> list[dict[str, Any]]:
    regression = json.loads(REGRESSION_PATH.read_text())
    rows = []
    for attribute in regression["eci"]["open_models"]:
        ci_width = float(attribute["ci_width"])
        if ci_width <= 0:
            raise ValueError(f"Non-positive ECI interval width for {attribute['model']}")
        rows.append(
            {
                "model": str(attribute["model"]),
                "release_date": str(attribute["release_date"]),
                "total_b": float(attribute["total_b"]),
                "family": attribute["family"],
                "eci": float(attribute["score"]),
                "reasoning": int(attribute.get("reasoning", 0) or 0),
                "moe": int(attribute.get("moe", 0) or 0),
                "workbook_wls_weight": 15.3664 / ci_width**2,
            }
        )
    if len(rows) != 89 or len({row["model"] for row in rows}) != 89:
        raise ValueError(f"Expected 89 unique current ECI calibration rows; found {len(rows)}")
    return rows


def _load_component_rows() -> tuple[list[dict[str, Any]], dict[str, tuple[float, float]]]:
    open_models = {row["model"]: row for row in _load_open_models()}
    difficulties = _load_difficulties()
    frame = pd.read_csv(COMPONENT_PATH)
    rows = []
    for source in frame.itertuples(index=False):
        model = str(source.model)
        benchmark = str(source.benchmark)
        if model not in open_models or benchmark not in difficulties:
            continue
        difficulty, slope = difficulties[benchmark]
        base = dict(open_models[model])
        performance = float(source.performance)
        base.update(
            {
                "benchmark": benchmark,
                "performance": performance,
                "optimized": bool(source.optimized),
                "component_eci": difficulty + _logit(performance) / slope,
            }
        )
        rows.append(base)
    return rows, difficulties


def _paired_backtest(
    label: str,
    rows: list[dict[str, Any]],
    component_signal: str = "component_eci",
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (row["release_date"], row["model"]))
    output = []
    for test in ordered:
        train = [
            row
            for row in ordered
            if row["release_date"] < test["release_date"] and row["family"] != test["family"]
        ]
        if len(train) < MIN_TRAIN or len({row["family"] for row in train}) < MIN_FAMILIES:
            continue
        component_only = _fit_predict(train, test, component_signal, False)
        component_date = _fit_predict(train, test, component_signal, True)
        eci_only = _fit_predict(train, test, "eci", False)
        eci_date = _fit_predict(train, test, "eci", True)
        component_prediction = 0.60 * component_only + 0.40 * component_date
        eci_prediction = 0.60 * eci_only + 0.40 * eci_date
        actual = math.log10(float(test["total_b"]))
        output.append(
            {
                "panel": label,
                "release_date": test["release_date"],
                "model": test["model"],
                "family": test["family"],
                "actual_b": test["total_b"],
                "performance": test.get("performance"),
                "component_signal": test[component_signal],
                "eci": test["eci"],
                "component_predicted_b": float(10**component_prediction),
                "eci_predicted_b": float(10**eci_prediction),
                "component_log10_error": float(component_prediction - actual),
                "eci_log10_error": float(eci_prediction - actual),
                "component_wins": int(abs(component_prediction - actual) < abs(eci_prediction - actual)),
                "train_n": len(train),
                "train_family_n": len({row["family"] for row in train}),
                "train_max_date": max(row["release_date"] for row in train),
            }
        )
    return output


def _aggregate_eci_live_backtest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strict chronological family holdout of the current 89-row calibration."""
    ordered = sorted(rows, key=lambda row: (row["release_date"], row["model"]))
    output = []
    for test in ordered:
        train = [
            row
            for row in ordered
            if row["release_date"] < test["release_date"] and row["family"] != test["family"]
        ]
        if len(train) < MIN_TRAIN or len({row["family"] for row in train}) < MIN_FAMILIES:
            continue
        score_only = _fit_predict_live_wls(train, test, False)
        score_date = _fit_predict_live_wls(train, test, True)
        prediction = 0.60 * score_only + 0.40 * score_date
        actual = math.log10(float(test["total_b"]))
        output.append(
            {
                "panel": "aggregate_eci_exact_live_calibration",
                "release_date": test["release_date"],
                "model": test["model"],
                "family": test["family"],
                "actual_b": test["total_b"],
                "performance": None,
                "component_signal": test["eci"],
                "eci": test["eci"],
                "component_predicted_b": float(10**prediction),
                "eci_predicted_b": float(10**prediction),
                "component_log10_error": float(prediction - actual),
                "eci_log10_error": float(prediction - actual),
                "component_wins": 0,
                "train_n": len(train),
                "train_family_n": len({row["family"] for row in train}),
                "train_max_date": max(row["release_date"] for row in train),
            }
        )
    return output


def _summary(label: str, rows: list[dict[str, Any]], coverage: int) -> dict[str, Any]:
    component = _metrics(rows, "component")
    eci = _metrics(rows, "eci")
    return {
        "panel": label,
        "coverage_models": int(coverage),
        "eligible": len(rows) >= MIN_ELIGIBLE_PREDICTIONS,
        **component,
        **eci,
        "component_win_rate": float(np.mean([row["component_wins"] for row in rows])) if rows else None,
        **_cluster_bootstrap_delta(rows),
    }


def _composite_rows(
    component_rows: list[dict[str, Any]],
    benchmarks: set[str],
    label: str,
    minimum_components: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        if row["benchmark"] in benchmarks:
            grouped[row["model"]].append(row)
    output = []
    for model, rows in grouped.items():
        if len(rows) < minimum_components:
            continue
        base = dict(rows[0])
        base["component_eci"] = float(np.median([row["component_eci"] for row in rows]))
        base["performance"] = None
        base["benchmark"] = label
        base["component_count"] = len(rows)
        output.append(base)
    return output


def _same_base_sensitivity(component_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        by_benchmark[row["benchmark"]].append(row)
    output = []
    for benchmark, rows in sorted(by_benchmark.items()):
        groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[(row["family"], row["total_b"])].append(row)
        repeated = [group for group in groups.values() if len(group) >= 2]
        mixed_reasoning = [
            group for group in repeated if {row["reasoning"] for row in group} == {0, 1}
        ]
        within_ranges = [
            max(row["component_eci"] for row in group) - min(row["component_eci"] for row in group)
            for group in repeated
        ]
        reasoning_uplifts = []
        for group in mixed_reasoning:
            reasoning = np.mean([row["component_eci"] for row in group if row["reasoning"] == 1])
            nonreasoning = np.mean([row["component_eci"] for row in group if row["reasoning"] == 0])
            reasoning_uplifts.append(float(reasoning - nonreasoning))
        output.append(
            {
                "benchmark": benchmark,
                "coverage_models": len(rows),
                "same_size_family_groups": len(repeated),
                "same_size_family_models": sum(len(group) for group in repeated),
                "median_within_base_eci_range": float(np.median(within_ranges)) if within_ranges else None,
                "p80_within_base_eci_range": float(np.quantile(within_ranges, 0.80)) if within_ranges else None,
                "mixed_reasoning_groups": len(mixed_reasoning),
                "median_reasoning_uplift_eci": float(np.median(reasoning_uplifts)) if reasoning_uplifts else None,
            }
        )
    return output


def _compute_coverage(open_names: set[str]) -> dict[str, Any]:
    frame = pd.read_csv(EPOCH_MODELS_PATH, low_memory=False)
    frame = frame[frame["Model"].isin(open_names)].copy()
    training = frame["Training compute (FLOP)"].notna()
    finetune = frame["Finetune compute (FLOP)"].notna()
    post = frame["Post-training compute (FLOP)"].notna()
    return {
        "matched_open_models": int(frame["Model"].nunique()),
        "training_compute_models": int(frame.loc[training, "Model"].nunique()),
        "finetune_compute_models": int(frame.loc[finetune, "Model"].nunique()),
        "post_training_compute_models": int(frame.loc[post, "Model"].nunique()),
        "training_and_finetune_models": int(frame.loc[training & finetune, "Model"].nunique()),
        "training_and_post_training_models": int(frame.loc[training & post, "Model"].nunique()),
        "ratio_conclusion": "Insufficient coverage for a direct RL/pretraining compute-ratio regression.",
    }


def _frontier_estimates(
    component_rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    difficulties: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    frame = pd.read_csv(COMPONENT_PATH)
    open_by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        open_by_benchmark[row["benchmark"]].append(row)
    eligible = {
        row["panel"]
        for row in summaries
        if row["eligible"] and row["panel"] not in {"knowledge_composite", "pretraining_like_composite"}
    }
    targets = frame[frame["model"].isin(["Claude Fable 5", "GPT-5.6 Sol"])]
    output = []
    for source in targets.itertuples(index=False):
        benchmark = str(source.benchmark)
        if benchmark not in eligible or benchmark not in difficulties:
            continue
        train = open_by_benchmark[benchmark]
        if len(train) < MIN_TRAIN or len({row["family"] for row in train}) < MIN_FAMILIES:
            continue
        difficulty, slope = difficulties[benchmark]
        target = {
            "model": str(source.model),
            "release_date": "2026-06-09" if str(source.model) == "Claude Fable 5" else "2026-07-09",
            "total_b": 1.0,
            "family": "anthropic" if str(source.model) == "Claude Fable 5" else "openai",
            "component_eci": difficulty + _logit(float(source.performance)) / slope,
        }
        component_only = _fit_predict(train, target, "component_eci", False)
        component_date = _fit_predict(train, target, "component_eci", True)
        estimate = 10 ** (0.60 * component_only + 0.40 * component_date) / 1000.0
        output.append(
            {
                "model": target["model"],
                "benchmark": benchmark,
                "performance": float(source.performance),
                "component_implied_eci": target["component_eci"],
                "estimate_t": float(estimate),
                "training_models": len(train),
                "training_families": len({row["family"] for row in train}),
                "exploratory_only": True,
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    component_rows, difficulties = _load_component_rows()
    open_models = _load_open_models()
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        by_benchmark[row["benchmark"]].append(row)

    summaries = []
    predictions = []
    for benchmark, rows in sorted(by_benchmark.items()):
        if len(rows) < MIN_COMPONENT_ROWS:
            continue
        fold_rows = _paired_backtest(benchmark, rows)
        predictions.extend(fold_rows)
        summaries.append(_summary(benchmark, fold_rows, len(rows)))

    composites = [
        (
            "knowledge_composite",
            _composite_rows(component_rows, KNOWLEDGE_BENCHMARKS, "knowledge_composite", 2),
        ),
        (
            "pretraining_like_composite",
            _composite_rows(
                component_rows,
                PRETRAINING_LIKE_BENCHMARKS,
                "pretraining_like_composite",
                3,
            ),
        ),
    ]
    for label, rows in composites:
        fold_rows = _paired_backtest(label, rows)
        predictions.extend(fold_rows)
        summaries.append(_summary(label, fold_rows, len(rows)))

    summaries.sort(
        key=lambda row: (
            not row["eligible"],
            row.get("component_rmse_log10", float("inf")),
            row["panel"],
        )
    )
    sensitivity = _same_base_sensitivity(component_rows)
    frontier = _frontier_estimates(component_rows, summaries, difficulties)
    compute_coverage = _compute_coverage({row["model"] for row in open_models})
    aggregate_eci_predictions = _aggregate_eci_live_backtest(open_models)
    predictions.extend(aggregate_eci_predictions)
    aggregate_eci_summary = {
        "split": "strictly-earlier chronological family holdout",
        "training_weights": "ECI-interval precision weights retained inside each fold",
        **_metrics(aggregate_eci_predictions, "eci"),
    }

    eligible_summaries = [row for row in summaries if row["eligible"]]
    materially_better = [
        row
        for row in eligible_summaries
        if row["paired_delta_ci_high"] is not None and row["paired_delta_ci_high"] < 0
    ]
    result = {
        "metadata": {
            "generated_on": "2026-07-31",
            "target": "log10 disclosed total parameters in billions",
            "outer_split": "strictly-earlier chronological family holdout",
            "paired_comparison": "Each component and aggregate ECI use the identical train/test rows in every fold.",
            "component_transform": "Official ECI IRT inverse: benchmark difficulty + logit(performance)/slope; performance clipped to [0.001, 0.999].",
            "prediction_spec": "60% score-only plus 40% score+date in log10 parameter space. Component comparisons use family-balanced training weights; the aggregate audit separately uses ECI-interval precision weights.",
            "minimums": {
                "component_coverage": MIN_COMPONENT_ROWS,
                "outer_train_rows": MIN_TRAIN,
                "outer_train_families": MIN_FAMILIES,
                "eligible_predictions": MIN_ELIGIBLE_PREDICTIONS,
            },
            "benchmark_vintage_caveat": "Current benchmark snapshot, not historical vintages.",
            "selection_caveat": "Per-benchmark rankings are exploratory multiple comparisons; only the two composites were predeclared.",
        },
        "inventory": {
            "component_rows_all": int(pd.read_csv(COMPONENT_PATH).shape[0]),
            "component_models_all": int(pd.read_csv(COMPONENT_PATH)["model"].nunique()),
            "component_benchmarks_all": int(pd.read_csv(COMPONENT_PATH)["benchmark"].nunique()),
            "open_ground_truth_models": len(open_models),
            "matched_component_rows": len(component_rows),
            "matched_component_models": len({row["model"] for row in component_rows}),
            "matched_component_benchmarks": len(by_benchmark),
        },
        "compute_coverage": compute_coverage,
        "aggregate_eci_exact_live_backtest": aggregate_eci_summary,
        "benchmark_comparisons": summaries,
        "materially_better_than_eci_after_family_cluster_bootstrap": materially_better,
        "frontier_component_estimates": frontier,
        "interpretation": {
            "promotion_rule": "Do not add a component branch unless a predeclared composite or independently replicated benchmark beats aggregate ECI on paired chronological family-held-out folds.",
            "rl_compute": "Use same-size/same-family score variation as a post-training sensitivity diagnostic; do not fit an explicit ratio with one complete data point.",
        },
        "source_files": {
            portable_path(COMPONENT_PATH): _sha256(COMPONENT_PATH),
            portable_path(BENCHMARK_ZIP_PATH): _sha256(BENCHMARK_ZIP_PATH),
            portable_path(REGRESSION_PATH): _sha256(REGRESSION_PATH),
            portable_path(EPOCH_MODELS_PATH): _sha256(EPOCH_MODELS_PATH),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    _write_csv(COMPARISON_PATH, summaries)
    _write_csv(PREDICTION_PATH, predictions)
    _write_csv(SENSITIVITY_PATH, sensitivity)
    _write_csv(FRONTIER_PATH, frontier)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
