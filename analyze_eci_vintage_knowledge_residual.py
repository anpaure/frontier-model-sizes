#!/usr/bin/env python3
"""Archive-vintage ECI knowledge-residual parameter audit.

This audit is a quarantined challenger to the live ECI parameter mapping.  It
uses only the Epoch component and aggregate scores that existed in each frozen
archive capture.  Every outer prediction:

* is made at the checkpoint's first archived ECI observation;
* trains only on checkpoints released strictly earlier than the target;
* removes the target developer, not merely the target model lineage;
* selects one global ridge penalty inside the outer training set using
  chronological, whole-developer-held-out validation; and
* applies a predeclared coverage attenuation and coverage-weighted global ridge
  to six benchmark-specific knowledge residuals.

The challenger remains diagnostic.  It cannot receive live weight unless all
predeclared archive-vintage coverage, clustered-error, source-format,
frontier, prospective, and current-target-coverage gates pass.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from analyze_parameter_vintage_sensitivity import (
    canonical_developer,
    developer_lookup,
    fallback_developer,
)
from run_parameter_backtest import _normal_model_name


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
REGRESSION = ROOT / "regression_results.json"
UNIFIED = OUT / "unified_model_observations_compute_enriched_2026-07-17.csv"
ARCHIVE = ROOT / "sources/epoch_eci_historical_snapshots_2026-07-18.tar.gz"
COLLECTION_METADATA = (
    ROOT / "sources/epoch_eci_historical_collection_metadata_2026-07-18.json"
)
HISTORICAL_COMPONENTS = (
    ROOT / "sources/epoch_eci_historical_reconstructed_inputs_2026-07-18.csv.gz"
)
HISTORICAL_SCORES = (
    ROOT / "sources/epoch_eci_historical_model_scores_2026-07-18.csv"
)
HISTORICAL_BENCHMARK_PARAMETERS = (
    ROOT / "sources/epoch_eci_historical_benchmark_parameters_2026-07-18.csv"
)
CURRENT_COMPONENTS = ROOT / "sources/epoch_eci_benchmarks_2026-07-31.csv"

RESULT = OUT / "eci_vintage_knowledge_residual_audit_2026-07-31.json"
PREDICTIONS = OUT / "eci_vintage_knowledge_residual_predictions_2026-07-31.csv"

GENERATED_ON = "2026-07-31"
DATE_ORIGIN = date(2023, 1, 1)
KNOWLEDGE_BENCHMARKS = (
    "GPQA diamond",
    "MMLU",
    "OpenBookQA",
    "ScienceQA",
    "SimpleQA Verified",
    "TriviaQA",
)
RIDGE_ALPHAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
ROBUST_CLIP = 4.0
COVERAGE_PRIOR_BENCHMARKS = 2.0
TUNING_NEAR_TIE_LOG10_MAE = 0.002
MIN_OUTER_TRAIN_ROWS = 20
MIN_OUTER_TRAIN_DEVELOPERS = 6
MIN_INNER_TRAIN_ROWS = 16
MIN_INNER_TRAIN_DEVELOPERS = 5
MIN_INNER_PREDICTIONS = 12
MIN_INNER_VALIDATION_DEVELOPERS = 5
MIN_BENCHMARK_ROWS = 10
MIN_BENCHMARK_DEVELOPERS = 4
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_731

# These are deliberately harder than the currently available prospective and
# target-coverage inventories.  Failing them is an honest zero-weight result,
# not a reason to relax the gate after looking at outcomes.
PROMOTION_MIN_OUTER_ROWS = 20
PROMOTION_MIN_OUTER_DEVELOPERS = 6
PROMOTION_MIN_FRONTIER_ROWS = 8
PROMOTION_MIN_FRONTIER_DEVELOPERS = 4
PROMOTION_MIN_FORMAT_ROWS = 6
PROMOTION_MIN_FORMAT_DEVELOPERS = 3
PROMOTION_MIN_PROSPECTIVE_ROWS = 5
PROMOTION_MIN_PROSPECTIVE_DEVELOPERS = 3
PROMOTION_MIN_TARGET_KNOWLEDGE_BENCHMARKS = 3

TARGET_MODELS = ("Claude Fable 5", "GPT-5.6 Sol")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def years(value: str) -> float:
    return (date.fromisoformat(value) - DATE_ORIGIN).days / 365.25


def timestamp_date(timestamp: str) -> date:
    return datetime.strptime(timestamp[:8], "%Y%m%d").date()


def logit(value: float, epsilon: float = 1e-3) -> float:
    clipped = min(1.0 - epsilon, max(epsilon, float(value)))
    return math.log(clipped / (1.0 - clipped))


def parameter_panel() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    regression = json.loads(REGRESSION.read_text(encoding="utf-8"))
    source = regression["eci"]["open_models"]
    if len(source) != 89:
        raise ValueError(f"Expected 89 ECI parameter checkpoints; found {len(source)}")
    lookup = developer_lookup()
    panel: dict[str, dict[str, Any]] = {}
    developers: dict[str, str] = {}
    for row in source:
        model = row["model"]
        developer = lookup.get(_normal_model_name(model)) or fallback_developer(
            model, row["family"]
        )
        developer = canonical_developer(developer)
        developers[model] = developer
        panel[model] = {
            "model": model,
            "release_date": row["release_date"],
            "lineage_family": row["family"],
            "developer": developer,
            "total_b": float(row["total_b"]),
            "active_b": float(row["active_b"]),
            "moe": int(row.get("moe", 0) or 0),
            "reasoning": int(row.get("reasoning", 0) or 0),
        }
    return panel, developers


def load_vintage_sources() -> dict[str, Any]:
    score_rows = read_csv(HISTORICAL_SCORES)
    component_rows = read_gzip_csv(HISTORICAL_COMPONENTS)
    benchmark_rows = read_csv(HISTORICAL_BENCHMARK_PARAMETERS)
    snapshots = sorted({row["snapshot_timestamp"] for row in score_rows})
    if len(snapshots) != 15:
        raise ValueError(f"Expected 15 archived ECI snapshots; found {len(snapshots)}")
    if len(score_rows) != 2308:
        raise ValueError(f"Expected 2,308 archived ECI scores; found {len(score_rows)}")
    if len(component_rows) != 20_350:
        raise ValueError(
            f"Expected 20,350 archived component rows; found {len(component_rows)}"
        )

    scores: dict[str, dict[str, float]] = defaultdict(dict)
    snapshot_kinds: dict[str, str] = {}
    for row in score_rows:
        timestamp = row["snapshot_timestamp"]
        model = row["Model"]
        if model in scores[timestamp]:
            raise ValueError(f"Duplicate archived ECI score: {(timestamp, model)}")
        scores[timestamp][model] = float(row["eci"])
        snapshot_kinds[timestamp] = row["snapshot_kind"]

    parameters: dict[tuple[str, str], tuple[float, float]] = {}
    for row in benchmark_rows:
        key = (row["snapshot_timestamp"], row["benchmark"])
        if key in parameters:
            raise ValueError(f"Duplicate archived benchmark parameter: {key}")
        parameters[key] = (float(row["edi"]), float(row["discriminability_scaled"]))

    raw_components: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    knowledge_rows = 0
    seen: set[tuple[str, str, str]] = set()
    for row in component_rows:
        benchmark = row["benchmark"]
        if benchmark not in KNOWLEDGE_BENCHMARKS:
            continue
        timestamp = row["snapshot_timestamp"]
        model = row["Model"]
        key = (timestamp, model, benchmark)
        if key in seen:
            raise ValueError(f"Duplicate archived component measurement: {key}")
        seen.add(key)
        if (timestamp, benchmark) not in parameters:
            raise ValueError(f"Missing vintage benchmark parameters for {key}")
        raw_components[timestamp][model][benchmark] = float(row["performance"])
        knowledge_rows += 1
    if knowledge_rows != 4066:
        raise ValueError(
            f"Expected 4,066 archived knowledge measurements; found {knowledge_rows}"
        )

    residuals: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for timestamp, models in raw_components.items():
        for model, components in models.items():
            aggregate = scores[timestamp].get(model)
            if aggregate is None:
                continue
            for benchmark, performance in components.items():
                edi, slope = parameters[(timestamp, benchmark)]
                residuals[timestamp][model][benchmark] = (
                    edi + logit(performance) / slope - aggregate
                )

    return {
        "snapshots": snapshots,
        "snapshot_kinds": snapshot_kinds,
        "scores": scores,
        "raw_components": raw_components,
        "residuals": residuals,
        "parameters": parameters,
        "score_rows": score_rows,
        "component_rows": component_rows,
        "knowledge_rows": knowledge_rows,
    }


def snapshot_panel(
    timestamp: str,
    source: dict[str, Any],
    parameters: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for model, score in source["scores"][timestamp].items():
        if model not in parameters:
            continue
        output.append(
            {
                **parameters[model],
                "score": float(score),
                "components": dict(source["residuals"][timestamp].get(model, {})),
            }
        )
    return sorted(output, key=lambda row: (row["release_date"], row["model"]))


def developer_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = Counter(row["developer"] for row in rows)
    weights = np.asarray([1.0 / counts[row["developer"]] for row in rows], dtype=float)
    return weights / weights.mean()


def fit_linear(
    train: list[dict[str, Any]], features: tuple[str, ...]
) -> np.ndarray:
    matrix = np.asarray(
        [
            [
                1.0,
                *[
                    float(row[feature]) if feature != "date" else years(row["release_date"])
                    for feature in features
                ],
            ]
            for row in train
        ],
        dtype=float,
    )
    values = np.log10(np.asarray([row["total_b"] for row in train], dtype=float))
    root = np.sqrt(developer_weights(train))
    beta, *_ = np.linalg.lstsq(matrix * root[:, None], values * root, rcond=None)
    return beta


def linear_predict(
    row: dict[str, Any], beta: np.ndarray, features: tuple[str, ...]
) -> float:
    values = [
        1.0,
        *[
            float(row[feature]) if feature != "date" else years(row["release_date"])
            for feature in features
        ],
    ]
    return float(np.asarray(values, dtype=float) @ beta)


def baseline_fit(train: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    return fit_linear(train, ("score",)), fit_linear(train, ("score", "date"))


def baseline_predict(row: dict[str, Any], fit: tuple[np.ndarray, np.ndarray]) -> float:
    score_beta, dated_beta = fit
    return 0.60 * linear_predict(row, score_beta, ("score",)) + 0.40 * linear_predict(
        row, dated_beta, ("score", "date")
    )


def eligible_benchmarks(train: list[dict[str, Any]]) -> list[str]:
    output = []
    for benchmark in KNOWLEDGE_BENCHMARKS:
        observed = [row for row in train if benchmark in row["components"]]
        if len(observed) < MIN_BENCHMARK_ROWS:
            continue
        if len({row["developer"] for row in observed}) < MIN_BENCHMARK_DEVELOPERS:
            continue
        output.append(benchmark)
    return output


def component_statistics(
    train: list[dict[str, Any]], benchmarks: list[str]
) -> dict[str, tuple[float, float, int]]:
    output = {}
    for benchmark in benchmarks:
        values = np.asarray(
            [row["components"][benchmark] for row in train if benchmark in row["components"]],
            dtype=float,
        )
        center = float(np.median(values))
        scale = float(np.median(np.abs(values - center)) * 1.4826)
        if scale < 1e-6:
            scale = float(np.std(values))
        if scale < 1e-6:
            scale = 1.0
        developers = len(
            {row["developer"] for row in train if benchmark in row["components"]}
        )
        output[benchmark] = (center, scale, developers)
    return output


def component_design(
    rows: list[dict[str, Any]],
    benchmarks: list[str],
    statistics: dict[str, tuple[float, float, int]],
) -> tuple[np.ndarray, list[int], list[float]]:
    matrix = []
    observed_counts = []
    shrinkages = []
    for row in rows:
        observed_count = sum(benchmark in row["components"] for benchmark in benchmarks)
        shrinkage = observed_count / (observed_count + COVERAGE_PRIOR_BENCHMARKS)
        values = []
        for benchmark in benchmarks:
            if benchmark not in row["components"]:
                values.append(0.0)
                continue
            center, scale, _ = statistics[benchmark]
            standardized = (row["components"][benchmark] - center) / scale
            values.append(float(np.clip(standardized, -ROBUST_CLIP, ROBUST_CLIP)) * shrinkage)
        matrix.append(values)
        observed_counts.append(observed_count)
        shrinkages.append(float(shrinkage))
    return np.asarray(matrix, dtype=float), observed_counts, shrinkages


def fit_predict_pair(
    train: list[dict[str, Any]], test: dict[str, Any], alpha: float
) -> dict[str, Any]:
    base_fit = baseline_fit(train)
    base_train = np.asarray([baseline_predict(row, base_fit) for row in train])
    baseline = baseline_predict(test, base_fit)
    benchmarks = eligible_benchmarks(train)
    if not benchmarks:
        return {
            "baseline_log10": baseline,
            "candidate_log10": baseline,
            "benchmarks": [],
            "observed_count": 0,
            "coverage_shrinkage": 0.0,
            "coefficients": [],
        }
    statistics = component_statistics(train, benchmarks)
    matrix, _, _ = component_design(train, benchmarks, statistics)
    residual = np.log10(np.asarray([row["total_b"] for row in train])) - base_train
    root = np.sqrt(developer_weights(train))
    weighted = matrix * root[:, None]
    max_developers = max(statistics[name][2] for name in benchmarks)
    penalty = np.asarray(
        [alpha * max_developers / statistics[name][2] for name in benchmarks],
        dtype=float,
    )
    gamma = np.linalg.pinv(
        weighted.T @ weighted + np.diag(penalty), rcond=1e-10
    ) @ (weighted.T @ (residual * root))
    test_matrix, observed, shrinkage = component_design([test], benchmarks, statistics)
    candidate = baseline + float(test_matrix[0] @ gamma)
    return {
        "baseline_log10": float(baseline),
        "candidate_log10": float(candidate),
        "benchmarks": benchmarks,
        "observed_count": int(observed[0]),
        "coverage_shrinkage": float(shrinkage[0]),
        "coefficients": [float(value) for value in gamma],
    }


def equal_developer_mae(rows: Iterable[tuple[str, float]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for developer, error in rows:
        grouped[developer].append(abs(float(error)))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def choose_alpha(outer_train: list[dict[str, Any]]) -> dict[str, Any] | None:
    scores: dict[float, float] = {}
    counts: dict[float, tuple[int, int]] = {}
    chronology_flags: dict[float, bool] = {}
    ordered = sorted(outer_train, key=lambda row: (row["release_date"], row["model"]))
    for alpha in RIDGE_ALPHAS:
        predictions: list[tuple[str, float]] = []
        strict = True
        for validation in ordered:
            train = [
                row
                for row in ordered
                if row["release_date"] < validation["release_date"]
                and row["developer"] != validation["developer"]
            ]
            if (
                len(train) < MIN_INNER_TRAIN_ROWS
                or len({row["developer"] for row in train})
                < MIN_INNER_TRAIN_DEVELOPERS
                or not eligible_benchmarks(train)
            ):
                continue
            fit = fit_predict_pair(train, validation, alpha)
            actual = math.log10(float(validation["total_b"]))
            predictions.append(
                (validation["developer"], fit["candidate_log10"] - actual)
            )
            strict = strict and max(row["release_date"] for row in train) < validation[
                "release_date"
            ] and all(
                row["developer"] != validation["developer"] for row in train
            )
        developers = len({developer for developer, _ in predictions})
        if (
            len(predictions) < MIN_INNER_PREDICTIONS
            or developers < MIN_INNER_VALIDATION_DEVELOPERS
        ):
            continue
        scores[alpha] = equal_developer_mae(predictions)
        counts[alpha] = (len(predictions), developers)
        chronology_flags[alpha] = strict
    if not scores:
        return None
    best = min(scores.values())
    eligible = [
        alpha
        for alpha, score in scores.items()
        if score <= best + TUNING_NEAR_TIE_LOG10_MAE
    ]
    selected = max(eligible)
    return {
        "alpha": float(selected),
        "inner_equal_developer_mae": float(scores[selected]),
        "inner_predictions": int(counts[selected][0]),
        "inner_developers": int(counts[selected][1]),
        "inner_strict_chronology": bool(chronology_flags[selected]),
        "candidate_alphas_evaluated": len(scores),
    }


def first_observed_predictions(
    source: dict[str, Any], parameters: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    snapshots: list[str] = source["snapshots"]
    first_snapshot: dict[str, str] = {}
    for timestamp in snapshots:
        for model in source["scores"][timestamp]:
            first_snapshot.setdefault(model, timestamp)
    initial = snapshots[0]
    output = []
    for model, timestamp in sorted(first_snapshot.items(), key=lambda item: (item[1], item[0])):
        if timestamp == initial or model not in parameters:
            continue
        panel = snapshot_panel(timestamp, source, parameters)
        target = next((row for row in panel if row["model"] == model), None)
        if target is None or not target["components"]:
            continue
        train = [
            row
            for row in panel
            if row["release_date"] < target["release_date"]
            and row["developer"] != target["developer"]
        ]
        if (
            len(train) < MIN_OUTER_TRAIN_ROWS
            or len({row["developer"] for row in train})
            < MIN_OUTER_TRAIN_DEVELOPERS
            or not eligible_benchmarks(train)
        ):
            continue
        policy = choose_alpha(train)
        if policy is None:
            continue
        fit = fit_predict_pair(train, target, policy["alpha"])
        actual = math.log10(float(target["total_b"]))
        previous = snapshots[snapshots.index(timestamp) - 1]
        observed = timestamp_date(timestamp)
        released = date.fromisoformat(target["release_date"])
        prior_scores = [row["score"] for row in train]
        rank = sum(value <= target["score"] for value in prior_scores) / len(prior_scores)
        target_components = {
            name: float(target["components"][name])
            for name in fit["benchmarks"]
            if name in target["components"]
        }
        output.append(
            {
                "snapshot_timestamp": timestamp,
                "snapshot_date": observed.isoformat(),
                "snapshot_kind": source["snapshot_kinds"][timestamp],
                "previous_snapshot_date": timestamp_date(previous).isoformat(),
                "model": model,
                "release_date": target["release_date"],
                "developer": target["developer"],
                "lineage_family": target["lineage_family"],
                "availability_lag_days": (observed - released).days,
                "interval_prospective": timestamp_date(previous) < released <= observed,
                "target_eci": float(target["score"]),
                "frontier_signal_rank": float(rank),
                "actual_b": float(target["total_b"]),
                "train_n": len(train),
                "train_developers": len({row["developer"] for row in train}),
                "train_max_date": max(row["release_date"] for row in train),
                "test_developer_excluded": all(
                    row["developer"] != target["developer"] for row in train
                ),
                "selected_alpha": policy["alpha"],
                "inner_equal_developer_mae": policy["inner_equal_developer_mae"],
                "inner_predictions": policy["inner_predictions"],
                "inner_developers": policy["inner_developers"],
                "inner_strict_chronology": policy["inner_strict_chronology"],
                "candidate_alphas_evaluated": policy["candidate_alphas_evaluated"],
                "eligible_benchmarks": "|".join(fit["benchmarks"]),
                "eligible_benchmark_count": len(fit["benchmarks"]),
                "observed_benchmarks": "|".join(target_components),
                "observed_benchmark_count": fit["observed_count"],
                "coverage_shrinkage": fit["coverage_shrinkage"],
                "target_component_residuals_json": json.dumps(
                    target_components, sort_keys=True, separators=(",", ":")
                ),
                "component_coefficients_json": json.dumps(
                    fit["coefficients"], separators=(",", ":")
                ),
                "baseline_predicted_b": float(10 ** fit["baseline_log10"]),
                "candidate_predicted_b": float(10 ** fit["candidate_log10"]),
                "baseline_log10_error": float(fit["baseline_log10"] - actual),
                "candidate_log10_error": float(fit["candidate_log10"] - actual),
                "baseline_multiplicative_error": float(
                    10 ** abs(fit["baseline_log10"] - actual)
                ),
                "candidate_multiplicative_error": float(
                    10 ** abs(fit["candidate_log10"] - actual)
                ),
                "candidate_over_baseline_prediction": float(
                    10 ** (fit["candidate_log10"] - fit["baseline_log10"])
                ),
            }
        )
    return output


def metric_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    errors = np.asarray([float(row[f"{prefix}_log10_error"]) for row in rows])
    factors = 10 ** np.abs(errors)
    return {
        "n": len(rows),
        "median_multiplicative_error": float(np.median(factors)),
        "geomean_multiplicative_error": float(10 ** np.mean(np.abs(errors))),
        "mean_absolute_log10_error": float(np.mean(np.abs(errors))),
        "rmse_log10": float(np.sqrt(np.mean(errors**2))),
        "p80_multiplicative_error": float(np.quantile(factors, 0.8)),
        "within_2x": float(np.mean(factors <= 2.0)),
        "signed_bias_factor": float(10 ** np.mean(errors)),
    }


def paired_developer_bootstrap(
    rows: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["developer"]].append(
            abs(float(row["candidate_log10_error"]))
            - abs(float(row["baseline_log10_error"]))
        )
    effects = np.asarray(
        [float(np.mean(grouped[name])) for name in sorted(grouped)], dtype=float
    )
    if not len(effects):
        return {"developers": 0, "paired_rows": 0}
    rng = np.random.default_rng(seed)
    draws = effects[
        rng.integers(0, len(effects), size=(BOOTSTRAP_SAMPLES, len(effects)))
    ].mean(axis=1)
    return {
        "metric": "equal-developer mean absolute log10 error; candidate minus baseline",
        "paired_rows": len(rows),
        "developers": len(effects),
        "observed_delta": float(np.mean(effects)),
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "probability_candidate_better": float(np.mean(draws < 0)),
        "samples": BOOTSTRAP_SAMPLES,
        "random_seed": seed,
    }


def comparison(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "developers": len({row["developer"] for row in rows}),
        "models": sorted(row["model"] for row in rows),
        "baseline": metric_summary(rows, "baseline"),
        "candidate": metric_summary(rows, "candidate"),
        "paired_developer_bootstrap": paired_developer_bootstrap(rows, seed),
    }


def target_component_coverage() -> list[dict[str, Any]]:
    counts: dict[str, set[str]] = {model: set() for model in TARGET_MODELS}
    for row in read_csv(CURRENT_COMPONENTS):
        if row["model"] in counts and row["benchmark"] in KNOWLEDGE_BENCHMARKS:
            counts[row["model"]].add(row["benchmark"])
    return [
        {
            "model": model,
            "observed_knowledge_benchmarks": sorted(counts[model]),
            "observed_count": len(counts[model]),
            "coverage_shrinkage": len(counts[model])
            / (len(counts[model]) + COVERAGE_PRIOR_BENCHMARKS),
            "passes_minimum": len(counts[model])
            >= PROMOTION_MIN_TARGET_KNOWLEDGE_BENCHMARKS,
        }
        for model in TARGET_MODELS
    ]


def main() -> None:
    parameters, developers = parameter_panel()
    source = load_vintage_sources()
    predictions = first_observed_predictions(source, parameters)
    if not predictions:
        raise ValueError("No archive-vintage knowledge-residual predictions")

    cohorts = {
        "all_first_observed": predictions,
        "frontier_like": [
            row for row in predictions if row["frontier_signal_rank"] >= 0.90
        ],
        "available_within_90_days": [
            row for row in predictions if row["availability_lag_days"] <= 90
        ],
        "interval_prospective": [
            row for row in predictions if row["interval_prospective"]
        ],
        "benchmark_zip": [
            row
            for row in predictions
            if row["snapshot_kind"] == "benchmark_zip_fixed_code_reconstruction"
        ],
        "canonical_csv": [
            row
            for row in predictions
            if row["snapshot_kind"] == "canonical_eci_csv"
        ],
    }
    summaries = {
        name: comparison(rows, BOOTSTRAP_SEED + index)
        for index, (name, rows) in enumerate(cohorts.items())
    }
    target_coverage = target_component_coverage()

    all_summary = summaries["all_first_observed"]
    frontier = summaries["frontier_like"]
    prospective = summaries["interval_prospective"]
    format_summaries = [summaries["benchmark_zip"], summaries["canonical_csv"]]
    gates = {
        "outer_coverage": all_summary["rows"] >= PROMOTION_MIN_OUTER_ROWS
        and all_summary["developers"] >= PROMOTION_MIN_OUTER_DEVELOPERS,
        "all_clustered_interval_wholly_favorable": all_summary[
            "paired_developer_bootstrap"
        ].get("ci_90", [math.inf, math.inf])[1]
        < 0,
        "all_median_improves": all_summary["candidate"].get(
            "median_multiplicative_error", math.inf
        )
        < all_summary["baseline"].get("median_multiplicative_error", -math.inf),
        "frontier_coverage": frontier["rows"] >= PROMOTION_MIN_FRONTIER_ROWS
        and frontier["developers"] >= PROMOTION_MIN_FRONTIER_DEVELOPERS,
        "frontier_clustered_interval_wholly_favorable": frontier[
            "paired_developer_bootstrap"
        ].get("ci_90", [math.inf, math.inf])[1]
        < 0,
        "source_format_stability": all(
            block["rows"] >= PROMOTION_MIN_FORMAT_ROWS
            and block["developers"] >= PROMOTION_MIN_FORMAT_DEVELOPERS
            and block["paired_developer_bootstrap"].get("observed_delta", math.inf) < 0
            for block in format_summaries
        ),
        "prospective_coverage": prospective["rows"]
        >= PROMOTION_MIN_PROSPECTIVE_ROWS
        and prospective["developers"] >= PROMOTION_MIN_PROSPECTIVE_DEVELOPERS,
        "prospective_median_non_degradation": prospective["rows"] > 0
        and prospective["candidate"].get("median_multiplicative_error", math.inf)
        <= prospective["baseline"].get("median_multiplicative_error", -math.inf),
        "current_target_component_coverage": all(
            row["passes_minimum"] for row in target_coverage
        ),
    }
    promote = all(gates.values())

    collection = json.loads(COLLECTION_METADATA.read_text(encoding="utf-8"))
    result = {
        "generated_on": GENERATED_ON,
        "question": (
            "Does a six-benchmark, coverage-shrunk knowledge-residual ridge improve "
            "total-parameter recovery in archive-vintage nested chronological "
            "whole-developer-held-out tests?"
        ),
        "predeclared_model": {
            "baseline": "60% score-only + 40% score-and-exact-date ECI log-parameter regression",
            "candidate": (
                "baseline offset plus six separate snapshot-vintage component-ECI "
                "residual coefficients"
            ),
            "knowledge_benchmarks": list(KNOWLEDGE_BENCHMARKS),
            "global_ridge_alphas": list(RIDGE_ALPHAS),
            "coefficient_penalty": (
                "one inner-selected global alpha, multiplied by maximum developer "
                "coverage / benchmark developer coverage"
            ),
            "sparse_row_shrinkage": "observed/(observed+2) multiplies every component residual",
            "coverage_prior_benchmarks": COVERAGE_PRIOR_BENCHMARKS,
            "outer_split": (
                "first archived observation; same vintage; strictly earlier releases; "
                "entire target developer excluded"
            ),
            "inner_split": (
                "within outer training only; each validation row uses strictly earlier "
                "releases and excludes the entire validation developer"
            ),
            "inner_selection_metric": "equal-developer mean absolute log10 error",
            "near_tie_rule": (
                f"within {TUNING_NEAR_TIE_LOG10_MAE} log10 MAE, choose stronger shrinkage"
            ),
        },
        "inventory": {
            "historical_snapshots": len(source["snapshots"]),
            "historical_score_rows": len(source["score_rows"]),
            "historical_component_rows": len(source["component_rows"]),
            "historical_knowledge_rows": source["knowledge_rows"],
            "parameter_checkpoints": len(parameters),
            "parameter_developers": len(set(developers.values())),
            "archive_canonical_csv_captures": sum(
                capture["kind"] == "canonical_eci_csv"
                for capture in collection["captures"]
            ),
            "archive_benchmark_zip_eci_captures": sum(
                capture["kind"] == "benchmark_zip"
                and capture["inventory"].get("has_eci_table", False)
                for capture in collection["captures"]
            ),
            "outer_predictions": len(predictions),
            "outer_developers": len({row["developer"] for row in predictions}),
        },
        "cohorts": summaries,
        "selected_alpha_counts": dict(
            sorted(Counter(str(row["selected_alpha"]) for row in predictions).items())
        ),
        "current_target_coverage": target_coverage,
        "promotion_gates": {
            "thresholds": {
                "minimum_outer_rows": PROMOTION_MIN_OUTER_ROWS,
                "minimum_outer_developers": PROMOTION_MIN_OUTER_DEVELOPERS,
                "minimum_frontier_rows": PROMOTION_MIN_FRONTIER_ROWS,
                "minimum_frontier_developers": PROMOTION_MIN_FRONTIER_DEVELOPERS,
                "minimum_rows_per_source_format": PROMOTION_MIN_FORMAT_ROWS,
                "minimum_developers_per_source_format": PROMOTION_MIN_FORMAT_DEVELOPERS,
                "minimum_prospective_rows": PROMOTION_MIN_PROSPECTIVE_ROWS,
                "minimum_prospective_developers": PROMOTION_MIN_PROSPECTIVE_DEVELOPERS,
                "minimum_current_target_knowledge_benchmarks": PROMOTION_MIN_TARGET_KNOWLEDGE_BENCHMARKS,
                "maximum_clustered_ci90_upper": 0.0,
            },
            "results": gates,
            "all_pass": promote,
        },
        "decision": {
            "promote_archive_vintage_knowledge_branch": promote,
            "incremental_live_weight": 0.10 if promote else 0.0,
            "change_live_forecasts": bool(promote),
            "reason": (
                "All predeclared archive-vintage gates passed."
                if promote
                else "At least one predeclared archive-vintage coverage, stability, or target-coverage gate failed; retain zero live weight."
            ),
        },
        "limitations": [
            "Most first-observed archive targets are historical backfills, not models released between adjacent captures.",
            "The truly interval-prospective cohort remains small and developer-concentrated.",
            "Only six knowledge benchmarks are predeclared; benchmark availability is non-random across model and date.",
            "Fable and Sol expose fewer knowledge components than the promotion minimum, so sparse-coverage shrinkage is material.",
            "The audit tests an incremental ECI component branch only and makes no full-ensemble vintage claim.",
        ],
        "source_files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                ARCHIVE,
                COLLECTION_METADATA,
                HISTORICAL_COMPONENTS,
                HISTORICAL_SCORES,
                HISTORICAL_BENCHMARK_PARAMETERS,
                CURRENT_COMPONENTS,
                REGRESSION,
                UNIFIED,
            )
        },
        "outputs": {
            "predictions": str(PREDICTIONS.relative_to(ROOT)),
        },
    }

    write_csv(PREDICTIONS, predictions)
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {RESULT}")
    print(f"Wrote {PREDICTIONS}")


if __name__ == "__main__":
    main()
