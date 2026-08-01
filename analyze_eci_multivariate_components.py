#!/usr/bin/env python3
"""Nested audit of multivariate ECI benchmark structure for parameter inference.

The individual-component audit tests one benchmark at a time.  This script asks
the harder and more useful question: can a regularized combination of benchmark
residuals improve on aggregate ECI without selecting components on the outer
test outcomes?

For every outer prediction:

* only strictly earlier checkpoints are eligible for training;
* the test checkpoint's entire developer family is removed;
* the component subset and ridge penalty are selected inside the outer training
  data by leave-one-family-out validation;
* component eligibility, robust centering/scaling, and missing-value imputation
  are recomputed from the training data only.

The source is a current benchmark snapshot rather than a historical vintage.
The input's legacy ``estimated`` flag is *not* parameter-disclosure metadata:
it is exactly an indicator that the aggregate ECI score interval is wider than
10 points.  The replication therefore uses only narrow-ECI-CI rows.  It must
not be described as a disclosed-parameter subset.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
REGRESSION = ROOT / "regression_results.json"
COMPONENTS = ROOT / "sources/epoch_eci_benchmarks_2026-07-31.csv"
BENCHMARK_ARCHIVE = ROOT / "sources/epoch_benchmark_data_2026-07-31.zip"
UNIFIED = OUT / "unified_model_observations_compute_enriched_2026-07-17.csv"
EXTENDED_AUDIT = OUT / "eci_component_extended_audit_2026-07-18.json"

RESULT = OUT / "eci_multivariate_component_audit_2026-07-18.json"
PREDICTIONS = OUT / "eci_multivariate_component_predictions_2026-07-18.csv"
NARROW_CI_PREDICTIONS = (
    OUT / "eci_multivariate_component_narrow_eci_ci_predictions_2026-07-18.csv"
)
TARGETS = OUT / "eci_multivariate_component_targets_2026-07-18.csv"
COVERAGE = OUT / "eci_multivariate_component_coverage_2026-07-18.csv"

DATE_ORIGIN = date(2023, 1, 1)
MIN_TRAIN_ROWS = 12
MIN_TRAIN_FAMILIES = 5
MIN_INNER_TRAIN_ROWS = 10
MIN_INNER_TRAIN_FAMILIES = 4
MIN_COMPONENT_ROWS = 10
MIN_COMPONENT_FAMILIES = 4
ROBUST_CLIP = 4.0
ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
TUNING_TOLERANCE = 0.002
BOOTSTRAP_SAMPLES = 20_000
SEED = 20_260_718

KNOWLEDGE_BENCHMARKS = frozenset(
    {
        "MMLU",
        "GPQA diamond",
        "SimpleQA Verified",
        "TriviaQA",
        "OpenBookQA",
        "ScienceQA",
    }
)
PRETRAINING_LIKE_BENCHMARKS = KNOWLEDGE_BENCHMARKS | frozenset(
    {
        "ARC AI2",
        "BBH",
        "GSM8K",
        "HellaSwag",
        "LAMBADA",
        "PIQA",
        "Winogrande",
    }
)
FEATURE_SETS: dict[str, frozenset[str] | None] = {
    "all_eligible": None,
    "pretraining_like": PRETRAINING_LIKE_BENCHMARKS,
    "knowledge_only": KNOWLEDGE_BENCHMARKS,
}
TARGET_MODELS = ("Claude Fable 5", "GPT-5.6 Sol")

PROMOTION_MIN_ALL_TESTS = 60
PROMOTION_MIN_ALL_FAMILIES = 25
PROMOTION_MIN_NARROW_CI_TESTS = 25
PROMOTION_MIN_NARROW_CI_FAMILIES = 15
PROMOTION_MIN_FRONTIER_TESTS = 6
PROMOTION_MIN_NARROW_CI_TRAINING_TESTS = 20
PROMOTION_MIN_NARROW_CI_TRAINING_FAMILIES = 10
PROMOTION_MIN_TARGET_OBSERVED_COMPONENTS = 2
MAX_TARGET_ADJUSTMENT_INSTABILITY = 1.5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def years(value: str) -> float:
    return (date.fromisoformat(value) - DATE_ORIGIN).days / 365.25


def logit(value: float, epsilon: float = 1e-3) -> float:
    clipped = min(1.0 - epsilon, max(epsilon, float(value)))
    return math.log(clipped / (1.0 - clipped))


def load_difficulties() -> dict[str, tuple[float, float]]:
    with zipfile.ZipFile(BENCHMARK_ARCHIVE) as archive:
        raw = archive.read(
            "additional_eci_data/eci_benchmark_difficulties_and_slopes.csv"
        ).decode("utf-8-sig")
    return {
        row["benchmark_name"]: (float(row["edi"]), float(row["estimated_slope_scaled"]))
        for row in csv.DictReader(io.StringIO(raw))
    }


def load_panel() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    regression = json.loads(REGRESSION.read_text(encoding="utf-8"))
    raw_attributes = regression["eci"]["open_models"]
    if len(raw_attributes) != 89:
        raise ValueError(f"Expected 89 ECI parameter-map checkpoints; found {len(raw_attributes)}")
    attributes = {row["model"]: row for row in raw_attributes}
    if len(attributes) != 89:
        raise ValueError("Duplicate model name in the ECI parameter map")
    for name, row in attributes.items():
        expected = int(float(row["ci_width"]) > 10.0)
        if int(row["estimated"]) != expected:
            raise ValueError(
                f"Legacy ECI uncertainty flag does not equal ci_width > 10 for {name}"
            )
        if int(row["broad_eci_ci"]) != expected:
            raise ValueError(f"ECI uncertainty flag does not equal ci_width > 10 for {name}")
        if row["parameter_disclosure_status"] != "not classified":
            raise ValueError(f"Unexpected parameter-disclosure classification for {name}")
    difficulties = load_difficulties()
    panel = {
        name: {
            "model": name,
            "release_date": row["release_date"],
            "family": row["family"],
            "active_b": float(row["active_b"]),
            "total_b": float(row["total_b"]),
            "eci": float(row["score"]),
            "reasoning": int(row["reasoning"]),
            "moe": int(row["moe"]),
            "broad_eci_ci": int(row["broad_eci_ci"]),
            "components": {},
        }
        for name, row in attributes.items()
    }
    seen: set[tuple[str, str]] = set()
    coverage: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"models": set(), "families": set(), "optimized": Counter()}
    )
    admitted = 0
    for source in read_csv(COMPONENTS):
        model = source["model"]
        benchmark = source["benchmark"]
        if model not in panel or benchmark not in difficulties:
            continue
        key = (model, benchmark)
        if key in seen:
            raise ValueError(f"Duplicate ECI model/benchmark measurement: {key}")
        seen.add(key)
        difficulty, slope = difficulties[benchmark]
        component_eci = difficulty + logit(float(source["performance"])) / slope
        panel[model]["components"][benchmark] = component_eci - panel[model]["eci"]
        coverage[benchmark]["models"].add(model)
        coverage[benchmark]["families"].add(panel[model]["family"])
        coverage[benchmark]["optimized"][source["optimized"]] += 1
        admitted += 1
    if admitted != 723 or len(seen) != 723:
        raise ValueError(f"Expected 723 unique component measurements; found {admitted}")
    if any(not row["components"] for row in panel.values()):
        raise ValueError("Every ECI parameter-map checkpoint must retain a component measurement")
    rows = sorted(panel.values(), key=lambda row: (row["release_date"], row["model"]))
    coverage_rows = []
    for benchmark, values in sorted(
        coverage.items(), key=lambda item: (-len(item[1]["models"]), item[0])
    ):
        coverage_rows.append(
            {
                "benchmark": benchmark,
                "models": len(values["models"]),
                "families": len(values["families"]),
                "knowledge_predeclared": benchmark in KNOWLEDGE_BENCHMARKS,
                "pretraining_like_predeclared": benchmark in PRETRAINING_LIKE_BENCHMARKS,
                "eligible_on_full_panel": len(values["models"]) >= MIN_COMPONENT_ROWS
                and len(values["families"]) >= MIN_COMPONENT_FAMILIES,
                "optimized_true_rows": values["optimized"]["True"],
                "optimized_false_rows": values["optimized"]["False"],
            }
        )
    return rows, coverage_rows


def family_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = Counter(row["family"] for row in rows)
    values = np.asarray(
        [
            (0.5 if row["broad_eci_ci"] else 1.0) / counts[row["family"]]
            for row in rows
        ],
        dtype=float,
    )
    return values / values.mean()


def eligible_benchmarks(
    rows: list[dict[str, Any]], subset: frozenset[str] | None
) -> list[str]:
    counts: Counter[str] = Counter()
    families: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for benchmark in row["components"]:
            if subset is not None and benchmark not in subset:
                continue
            counts[benchmark] += 1
            families[benchmark].add(row["family"])
    return sorted(
        benchmark
        for benchmark in counts
        if counts[benchmark] >= MIN_COMPONENT_ROWS
        and len(families[benchmark]) >= MIN_COMPONENT_FAMILIES
    )


def robust_stats(
    rows: list[dict[str, Any]], benchmarks: list[str]
) -> dict[str, tuple[float, float]]:
    output = {}
    for benchmark in benchmarks:
        values = np.asarray(
            [row["components"][benchmark] for row in rows if benchmark in row["components"]],
            dtype=float,
        )
        center = float(np.median(values))
        scale = float(np.median(np.abs(values - center)) * 1.4826)
        if scale < 1e-6:
            scale = float(np.std(values))
        output[benchmark] = (center, scale if scale >= 1e-6 else 1.0)
    return output


def design(
    rows: list[dict[str, Any]],
    benchmarks: list[str],
    statistics: dict[str, tuple[float, float]],
) -> np.ndarray:
    matrix = []
    for row in rows:
        values = [
            1.0,
            float(row["eci"]),
            years(row["release_date"]),
            float(row["reasoning"]),
            float(row["moe"]),
        ]
        for benchmark in benchmarks:
            if benchmark not in row["components"]:
                values.append(0.0)
                continue
            center, scale = statistics[benchmark]
            standardized = (row["components"][benchmark] - center) / scale
            values.append(float(np.clip(standardized, -ROBUST_CLIP, ROBUST_CLIP)))
        matrix.append(values)
    return np.asarray(matrix, dtype=float)


def fit_predict(
    train: list[dict[str, Any]],
    test: dict[str, Any],
    target: str,
    alpha: float,
    subset: frozenset[str] | None,
) -> tuple[float, list[str], list[float]]:
    benchmarks = eligible_benchmarks(train, subset)
    statistics = robust_stats(train, benchmarks)
    matrix = design(train, benchmarks, statistics)
    values = np.log10(np.asarray([row[target] for row in train], dtype=float))
    root_weight = np.sqrt(family_weights(train))
    weighted = matrix * root_weight[:, None]
    penalty = np.zeros(matrix.shape[1], dtype=float)
    penalty[5:] = alpha
    beta = np.linalg.pinv(weighted.T @ weighted + np.diag(penalty), rcond=1e-10) @ (
        weighted.T @ (values * root_weight)
    )
    prediction = float(design([test], benchmarks, statistics)[0] @ beta)
    return prediction, benchmarks, [float(value) for value in beta]


def equal_family_mae(rows: Iterable[tuple[str, float]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for family, error in rows:
        grouped[family].append(abs(float(error)))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def choose_policy(
    outer_train: list[dict[str, Any]], target: str
) -> dict[str, Any] | None:
    families = sorted({row["family"] for row in outer_train})
    scores: dict[tuple[str, float], float] = {}
    counts: dict[tuple[str, float], tuple[int, int]] = {}
    for feature_name, subset in FEATURE_SETS.items():
        for alpha in ALPHAS:
            predictions = []
            for validation_family in families:
                train = [
                    row for row in outer_train if row["family"] != validation_family
                ]
                validation = [
                    row for row in outer_train if row["family"] == validation_family
                ]
                if (
                    len(train) < MIN_INNER_TRAIN_ROWS
                    or len({row["family"] for row in train})
                    < MIN_INNER_TRAIN_FAMILIES
                    or not eligible_benchmarks(train, subset)
                ):
                    continue
                for test in validation:
                    predicted, _, _ = fit_predict(train, test, target, alpha, subset)
                    predictions.append(
                        (test["family"], predicted - math.log10(float(test[target])))
                    )
            prediction_families = len({family for family, _ in predictions})
            if (
                len(predictions) < MIN_INNER_TRAIN_ROWS
                or prediction_families < MIN_INNER_TRAIN_FAMILIES
            ):
                continue
            key = (feature_name, alpha)
            scores[key] = equal_family_mae(predictions)
            counts[key] = (len(predictions), prediction_families)
    if not scores:
        return None
    best = min(scores.values())
    eligible = [key for key, value in scores.items() if value <= best + TUNING_TOLERANCE]
    # Prefer stronger shrinkage inside the near-tie band, then the narrower
    # predeclared feature set.  This rule is fixed and does not inspect outer outcomes.
    feature_tie_order = {"knowledge_only": 2, "pretraining_like": 1, "all_eligible": 0}
    selected = max(eligible, key=lambda key: (key[1], feature_tie_order[key[0]]))
    return {
        "feature_set": selected[0],
        "alpha": selected[1],
        "inner_equal_family_mae": scores[selected],
        "inner_predictions": counts[selected][0],
        "inner_families": counts[selected][1],
        "candidate_policies_evaluated": len(scores),
    }


def chronological_backtest(
    panel: list[dict[str, Any]], training_scope: str
) -> list[dict[str, Any]]:
    if training_scope == "all_half_weight":
        eligible_panel = panel
        test_panel = panel
        minimum_predictions = PROMOTION_MIN_ALL_TESTS
        minimum_families = PROMOTION_MIN_ALL_FAMILIES
    elif training_scope == "narrow_eci_ci_only":
        eligible_panel = [row for row in panel if not row["broad_eci_ci"]]
        test_panel = eligible_panel
        minimum_predictions = PROMOTION_MIN_NARROW_CI_TRAINING_TESTS
        minimum_families = PROMOTION_MIN_NARROW_CI_TRAINING_FAMILIES
    else:
        raise ValueError(f"Unknown training scope: {training_scope}")
    ranks = {
        row["model"]: float(
            np.mean([other["eci"] <= row["eci"] for other in panel])
        )
        for row in panel
    }
    output = []
    for test in test_panel:
        train = [
            row
            for row in eligible_panel
            if row["release_date"] < test["release_date"]
            and row["family"] != test["family"]
        ]
        if (
            len(train) < MIN_TRAIN_ROWS
            or len({row["family"] for row in train}) < MIN_TRAIN_FAMILIES
        ):
            continue
        policies = {
            target: choose_policy(train, target)
            for target in ("active_b", "total_b")
        }
        if any(policy is None for policy in policies.values()):
            continue
        record: dict[str, Any] = {
            "training_scope": training_scope,
            "release_date": test["release_date"],
            "model": test["model"],
            "family": test["family"],
            "broad_eci_ci": test["broad_eci_ci"],
            "eci": test["eci"],
            "eci_rank": ranks[test["model"]],
            "reasoning": test["reasoning"],
            "moe": test["moe"],
            "actual_active_b": test["active_b"],
            "actual_total_b": test["total_b"],
            "available_components": len(test["components"]),
            "train_n": len(train),
            "train_families": len({row["family"] for row in train}),
            "train_max_date": max(row["release_date"] for row in train),
            "test_family_excluded": not any(
                row["family"] == test["family"] for row in train
            ),
        }
        for target, label in (("active_b", "active"), ("total_b", "total")):
            policy = policies[target]
            assert policy is not None
            subset = FEATURE_SETS[policy["feature_set"]]
            candidate, benchmarks, _ = fit_predict(
                train, test, target, float(policy["alpha"]), subset
            )
            baseline, _, _ = fit_predict(train, test, target, 0.0, frozenset())
            actual = math.log10(float(test[target]))
            observed = [name for name in benchmarks if name in test["components"]]
            record.update(
                {
                    f"{label}_feature_set": policy["feature_set"],
                    f"{label}_alpha": policy["alpha"],
                    f"{label}_inner_equal_family_mae": policy[
                        "inner_equal_family_mae"
                    ],
                    f"{label}_inner_predictions": policy["inner_predictions"],
                    f"{label}_inner_families": policy["inner_families"],
                    f"{label}_selected_benchmarks": "|".join(benchmarks),
                    f"{label}_selected_benchmark_count": len(benchmarks),
                    f"{label}_observed_selected_benchmarks": "|".join(observed),
                    f"{label}_observed_selected_count": len(observed),
                    f"baseline_{label}_predicted_b": 10**baseline,
                    f"candidate_{label}_predicted_b": 10**candidate,
                    f"baseline_{label}_log10_error": baseline - actual,
                    f"candidate_{label}_log10_error": candidate - actual,
                }
            )
        output.append(record)
    if (
        len(output) < minimum_predictions
        or len({row["family"] for row in output}) < minimum_families
    ):
        raise ValueError(
            f"Nested component audit has insufficient outer coverage for {training_scope}: "
            f"{len(output)} predictions / {len({row['family'] for row in output})} families"
        )
    return output


def metric_summary(errors: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(errors), dtype=float)
    if not len(values):
        return {"n": 0}
    absolute = np.abs(values)
    return {
        "n": int(len(values)),
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "mean_absolute_log10_error": float(np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(values**2))),
        "p80_multiplicative_error": float(10 ** np.quantile(absolute, 0.80)),
        "within_2x": float(np.mean(absolute <= math.log10(2.0))),
        "signed_bias_factor": float(10 ** np.mean(values)),
    }


def paired_bootstrap(
    rows: list[dict[str, Any]], baseline_key: str, candidate_key: str, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(
            abs(float(row[candidate_key])) - abs(float(row[baseline_key]))
        )
    effects = np.asarray(
        [float(np.mean(grouped[family])) for family in sorted(grouped)], dtype=float
    )
    if not len(effects):
        return {"family_clusters": 0}
    rng = np.random.default_rng(seed)
    draws = np.mean(
        effects[
            rng.integers(0, len(effects), size=(BOOTSTRAP_SAMPLES, len(effects)))
        ],
        axis=1,
    )
    return {
        "metric": "equal-family mean absolute log10 error; candidate minus baseline",
        "observed_delta": float(np.mean(effects)),
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "probability_candidate_better": float(np.mean(draws < 0)),
        "family_clusters": int(len(effects)),
        "samples": BOOTSTRAP_SAMPLES,
    }


def comparison(
    predictions: list[dict[str, Any]], scope: str, target: str, seed: int
) -> dict[str, Any]:
    if scope == "all":
        selected = predictions
    elif scope == "narrow_eci_ci":
        selected = [row for row in predictions if not row["broad_eci_ci"]]
    elif scope == "broad_eci_ci":
        selected = [row for row in predictions if row["broad_eci_ci"]]
    elif scope == "frontier_like":
        selected = [row for row in predictions if row["eci_rank"] >= 0.90]
    elif scope == "recent_narrow_eci_ci":
        selected = [
            row
            for row in predictions
            if not row["broad_eci_ci"] and row["release_date"] >= "2024-01-01"
        ]
    else:
        raise ValueError(scope)
    baseline_key = f"baseline_{target}_log10_error"
    candidate_key = f"candidate_{target}_log10_error"
    return {
        "scope": scope,
        "n": len(selected),
        "families": len({row["family"] for row in selected}),
        "baseline": metric_summary(float(row[baseline_key]) for row in selected),
        "candidate": metric_summary(float(row[candidate_key]) for row in selected),
        "paired_family_bootstrap": paired_bootstrap(
            selected, baseline_key, candidate_key, seed
        ),
    }


def load_targets(
    panel: list[dict[str, Any]], coverage_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    difficulties = load_difficulties()
    unified = {
        row["source_model_name"]: row
        for row in read_csv(UNIFIED)
        if row["source"] == "ECI"
    }
    raw_components: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(COMPONENTS):
        if row["model"] in TARGET_MODELS and row["benchmark"] in difficulties:
            raw_components[row["model"]].append(row)
    current_anchors = {
        row["model"]: float(row["legacy_57_estimate_t"])
        for row in json.loads(EXTENDED_AUDIT.read_text(encoding="utf-8"))[
            "expanded_total_parameter_panel"
        ]["frontier_estimate_stability"]
    }
    output = []
    fit_records = []
    for model in TARGET_MODELS:
        source = unified[model]
        target = {
            "model": model,
            "release_date": source["canonical_release_date"],
            "family": "anthropic" if model.startswith("Claude") else "openai",
            "active_b": 1.0,
            "total_b": 1.0,
            "eci": float(source["eci_score"]),
            "reasoning": 1,
            "moe": 1,
            "broad_eci_ci": 0,
            "components": {},
        }
        for row in raw_components[model]:
            difficulty, slope = difficulties[row["benchmark"]]
            target["components"][row["benchmark"]] = (
                difficulty + logit(float(row["performance"])) / slope - target["eci"]
            )

        full_train = [
            row
            for row in panel
            if row["release_date"] < target["release_date"]
            and row["family"] != target["family"]
        ]
        narrow_ci_train = [row for row in full_train if not row["broad_eci_ci"]]
        full_policy = choose_policy(full_train, "total_b")
        narrow_ci_policy = choose_policy(narrow_ci_train, "total_b")
        if full_policy is None or narrow_ci_policy is None:
            raise ValueError(f"No honest nested target policy for {model}")

        def fit_branch(
            train: list[dict[str, Any]], policy: dict[str, Any]
        ) -> dict[str, Any]:
            subset = FEATURE_SETS[policy["feature_set"]]
            candidate, benchmarks, coefficients = fit_predict(
                train, target, "total_b", float(policy["alpha"]), subset
            )
            baseline, _, _ = fit_predict(
                train, target, "total_b", 0.0, frozenset()
            )
            observed = [name for name in benchmarks if name in target["components"]]
            adjustment = float(10 ** (candidate - baseline))
            return {
                "policy": policy,
                "benchmarks": benchmarks,
                "coefficients": coefficients,
                "observed": observed,
                "baseline": baseline,
                "candidate": candidate,
                "adjustment": adjustment,
            }

        full_fit = fit_branch(full_train, full_policy)
        narrow_ci_fit = fit_branch(narrow_ci_train, narrow_ci_policy)
        adjustment_ratio = max(
            full_fit["adjustment"], narrow_ci_fit["adjustment"]
        ) / min(full_fit["adjustment"], narrow_ci_fit["adjustment"])
        direction_agrees = (
            (full_fit["adjustment"] >= 1 and narrow_ci_fit["adjustment"] >= 1)
            or (full_fit["adjustment"] <= 1 and narrow_ci_fit["adjustment"] <= 1)
        )
        output.append(
            {
                "model": model,
                "release_date": target["release_date"],
                "eci_score": target["eci"],
                "available_component_measurements": len(target["components"]),
                "full_training_n": len(full_train),
                "full_training_families": len(
                    {row["family"] for row in full_train}
                ),
                "full_training_max_date": max(
                    row["release_date"] for row in full_train
                ),
                "target_family_excluded": not any(
                    row["family"] == target["family"] for row in full_train
                ),
                "selected_feature_set": full_policy["feature_set"],
                "selected_alpha": full_policy["alpha"],
                "eligible_training_benchmarks": "|".join(full_fit["benchmarks"]),
                "observed_selected_benchmarks": "|".join(full_fit["observed"]),
                "observed_selected_count": len(full_fit["observed"]),
                "raw_multivariate_baseline_t": 10 ** full_fit["baseline"] / 1000,
                "raw_multivariate_candidate_t": 10 ** full_fit["candidate"] / 1000,
                "component_adjustment_factor": full_fit["adjustment"],
                "narrow_ci_training_n": len(narrow_ci_train),
                "narrow_ci_training_families": len(
                    {row["family"] for row in narrow_ci_train}
                ),
                "narrow_ci_training_feature_set": narrow_ci_policy["feature_set"],
                "narrow_ci_training_alpha": narrow_ci_policy["alpha"],
                "narrow_ci_training_eligible_benchmarks": "|".join(
                    narrow_ci_fit["benchmarks"]
                ),
                "narrow_ci_training_observed_benchmarks": "|".join(
                    narrow_ci_fit["observed"]
                ),
                "narrow_ci_training_observed_count": len(narrow_ci_fit["observed"]),
                "narrow_ci_training_raw_baseline_t": 10
                ** narrow_ci_fit["baseline"]
                / 1000,
                "narrow_ci_training_raw_candidate_t": 10
                ** narrow_ci_fit["candidate"]
                / 1000,
                "narrow_ci_training_adjustment_factor": narrow_ci_fit["adjustment"],
                "full_vs_narrow_ci_adjustment_ratio": adjustment_ratio,
                "full_vs_narrow_ci_direction_agrees": direction_agrees,
                "legacy_57_eci_t": current_anchors[model],
                "component_adjusted_eci_t": current_anchors[model]
                * full_fit["adjustment"],
                "narrow_ci_training_adjusted_eci_t": current_anchors[model]
                * narrow_ci_fit["adjustment"],
                "incremental_live_weight": 0.0,
                "status": "diagnostic pending promotion gates",
            }
        )
        fit_records.append(
            {
                "model": model,
                "full_panel_policy": full_policy,
                "full_eligible_benchmarks": full_fit["benchmarks"],
                "full_coefficients": full_fit["coefficients"],
                "narrow_ci_panel_policy": narrow_ci_policy,
                "narrow_ci_eligible_benchmarks": narrow_ci_fit["benchmarks"],
                "narrow_ci_coefficients": narrow_ci_fit["coefficients"],
            }
        )
    return output, {
        "target_models": fit_records,
        "coverage_rows": len(coverage_rows),
    }


def main() -> None:
    panel, coverage_rows = load_panel()
    predictions = chronological_backtest(panel, "all_half_weight")
    narrow_ci_predictions = chronological_backtest(panel, "narrow_eci_ci_only")
    scopes: dict[str, dict[str, Any]] = {}
    seed = SEED
    for target in ("active", "total"):
        scopes[target] = {}
        for scope in (
            "all",
            "narrow_eci_ci",
            "broad_eci_ci",
            "frontier_like",
            "recent_narrow_eci_ci",
        ):
            scopes[target][scope] = comparison(predictions, scope, target, seed)
            seed += 1
    narrow_ci_training_backtest = {}
    for target in ("active", "total"):
        narrow_ci_training_backtest[target] = comparison(
            narrow_ci_predictions, "all", target, seed
        )
        seed += 1

    total_all = scopes["total"]["all"]
    total_narrow_ci = scopes["total"]["narrow_eci_ci"]
    total_frontier = scopes["total"]["frontier_like"]
    all_point_gate = all(
        total_all["candidate"][metric] < total_all["baseline"][metric]
        for metric in (
            "median_multiplicative_error",
            "mean_absolute_log10_error",
            "p80_multiplicative_error",
        )
    )
    narrow_ci_point_gate = all(
        total_narrow_ci["candidate"][metric]
        < total_narrow_ci["baseline"][metric]
        for metric in ("median_multiplicative_error", "mean_absolute_log10_error")
    )
    all_interval_gate = total_all["paired_family_bootstrap"]["ci_90"][1] < 0
    narrow_ci_interval_gate = (
        total_narrow_ci["paired_family_bootstrap"]["ci_90"][1] < 0
    )
    coverage_gate = (
        total_all["n"] >= PROMOTION_MIN_ALL_TESTS
        and total_all["families"] >= PROMOTION_MIN_ALL_FAMILIES
        and total_narrow_ci["n"] >= PROMOTION_MIN_NARROW_CI_TESTS
        and total_narrow_ci["families"] >= PROMOTION_MIN_NARROW_CI_FAMILIES
        and total_frontier["n"] >= PROMOTION_MIN_FRONTIER_TESTS
    )
    frontier_non_degradation_gate = (
        total_frontier["candidate"]["median_multiplicative_error"]
        <= total_frontier["baseline"]["median_multiplicative_error"]
    )
    narrow_ci_training_total = narrow_ci_training_backtest["total"]
    narrow_ci_training_point_gate = all(
        narrow_ci_training_total["candidate"][metric]
        < narrow_ci_training_total["baseline"][metric]
        for metric in ("median_multiplicative_error", "mean_absolute_log10_error")
    )
    narrow_ci_training_interval_gate = (
        narrow_ci_training_total["paired_family_bootstrap"]["ci_90"][1] < 0
    )
    narrow_ci_training_coverage_gate = (
        narrow_ci_training_total["n"] >= PROMOTION_MIN_NARROW_CI_TRAINING_TESTS
        and narrow_ci_training_total["families"]
        >= PROMOTION_MIN_NARROW_CI_TRAINING_FAMILIES
    )

    targets, target_fit = load_targets(panel, coverage_rows)
    target_adjustment_stability_gate = all(
        row["full_vs_narrow_ci_adjustment_ratio"]
        <= MAX_TARGET_ADJUSTMENT_INSTABILITY
        and row["full_vs_narrow_ci_direction_agrees"]
        for row in targets
    )
    target_component_coverage_gate = all(
        row["observed_selected_count"] >= PROMOTION_MIN_TARGET_OBSERVED_COMPONENTS
        and row["narrow_ci_training_observed_count"]
        >= PROMOTION_MIN_TARGET_OBSERVED_COMPONENTS
        for row in targets
    )
    target_chronology_gate = all(
        row["full_training_max_date"] < row["release_date"]
        and row["target_family_excluded"]
        for row in targets
    )
    promote = all(
        (
            all_point_gate,
            narrow_ci_point_gate,
            all_interval_gate,
            narrow_ci_interval_gate,
            coverage_gate,
            frontier_non_degradation_gate,
            narrow_ci_training_point_gate,
            narrow_ci_training_interval_gate,
            narrow_ci_training_coverage_gate,
            target_adjustment_stability_gate,
            target_component_coverage_gate,
            target_chronology_gate,
        )
    )

    for row in targets:
        row["incremental_live_weight"] = 0.05 if promote else 0.0
        row["status"] = (
            "eligible 5% correlated ECI sub-branch"
            if promote
            else "0%-weight diagnostic; promotion gates failed"
        )

    result = {
        "metadata": {
            "generated_on": "2026-07-31",
            "question": "Does a nested regularized combination of ECI benchmark residuals improve total-parameter inference beyond aggregate ECI?",
            "outer_split": "strictly earlier release date; entire test family removed",
            "inner_selection": "leave-one-family-out within outer training data; selects feature class and ridge alpha",
            "baseline": "log10 parameters ~ aggregate ECI + exact date + reasoning + MoE",
            "candidate": "baseline plus training-standardized benchmark-implied-ECI residuals with ridge shrinkage",
            "missing_component_policy": "training-median residual after robust centering, represented as zero",
            "training_weights": "primary run gives equal total weight per family and half weight to aggregate ECI scores whose CI width exceeds 10 points; the replication uses only narrow-ECI-CI rows",
            "legacy_flag_definition": "regression_results.eci.open_models[].estimated is exactly int(eci_ci_width > 10); it contains no parameter-disclosure information",
            "target_fit": "strictly earlier checkpoints only; target developer removed; full and narrow-ECI-CI policies tuned separately",
            "current_snapshot_caveat": "Benchmark measurements are from the current Epoch snapshot rather than historical vintages.",
        },
        "inventory": {
            "parameter_map_checkpoints": len(panel),
            "parameter_map_families": len({row["family"] for row in panel}),
            "narrow_eci_ci_checkpoints": sum(not row["broad_eci_ci"] for row in panel),
            "broad_eci_ci_checkpoints": sum(bool(row["broad_eci_ci"]) for row in panel),
            "unique_component_measurements": sum(len(row["components"]) for row in panel),
            "component_benchmarks": len(coverage_rows),
            "outer_predictions": len(predictions),
            "outer_prediction_families": len({row["family"] for row in predictions}),
            "narrow_eci_ci_only_outer_predictions": len(narrow_ci_predictions),
            "narrow_eci_ci_only_outer_prediction_families": len(
                {row["family"] for row in narrow_ci_predictions}
            ),
        },
        "predeclared_model": {
            "feature_sets": {
                "all_eligible": "all training-eligible benchmarks",
                "pretraining_like": sorted(PRETRAINING_LIKE_BENCHMARKS),
                "knowledge_only": sorted(KNOWLEDGE_BENCHMARKS),
            },
            "minimum_component_rows": MIN_COMPONENT_ROWS,
            "minimum_component_families": MIN_COMPONENT_FAMILIES,
            "minimum_inner_training_rows": MIN_INNER_TRAIN_ROWS,
            "minimum_inner_training_families": MIN_INNER_TRAIN_FAMILIES,
            "ridge_alphas": ALPHAS,
            "robust_clip_standard_deviations": ROBUST_CLIP,
            "tuning_near_tie_tolerance_log10_mae": TUNING_TOLERANCE,
        },
        "backtest": scopes,
        "narrow_eci_ci_only_training_backtest": narrow_ci_training_backtest,
        "target_sensitivity": targets,
        "target_fit": target_fit,
        "promotion_gates": {
            "all_point_metrics_improve": all_point_gate,
            "narrow_eci_ci_point_metrics_improve": narrow_ci_point_gate,
            "all_equal_family_ci_wholly_favorable": all_interval_gate,
            "narrow_eci_ci_equal_family_ci_wholly_favorable": narrow_ci_interval_gate,
            "coverage_gate": coverage_gate,
            "frontier_median_non_degradation": frontier_non_degradation_gate,
            "narrow_eci_ci_only_training_point_metrics_improve": narrow_ci_training_point_gate,
            "narrow_eci_ci_only_training_ci_wholly_favorable": narrow_ci_training_interval_gate,
            "narrow_eci_ci_only_training_coverage_gate": narrow_ci_training_coverage_gate,
            "target_full_vs_narrow_eci_ci_adjustment_stable": target_adjustment_stability_gate,
            "target_component_coverage_gate": target_component_coverage_gate,
            "target_chronology_and_family_holdout_gate": target_chronology_gate,
            "minimum_all_tests": PROMOTION_MIN_ALL_TESTS,
            "minimum_all_families": PROMOTION_MIN_ALL_FAMILIES,
            "minimum_narrow_eci_ci_tests": PROMOTION_MIN_NARROW_CI_TESTS,
            "minimum_narrow_eci_ci_families": PROMOTION_MIN_NARROW_CI_FAMILIES,
            "minimum_frontier_tests": PROMOTION_MIN_FRONTIER_TESTS,
            "minimum_narrow_eci_ci_only_training_tests": PROMOTION_MIN_NARROW_CI_TRAINING_TESTS,
            "minimum_narrow_eci_ci_only_training_families": PROMOTION_MIN_NARROW_CI_TRAINING_FAMILIES,
            "minimum_target_observed_components_per_branch": PROMOTION_MIN_TARGET_OBSERVED_COMPONENTS,
            "maximum_full_vs_narrow_eci_ci_target_adjustment_ratio": MAX_TARGET_ADJUSTMENT_INSTABILITY,
        },
        "decision": {
            "promote_multivariate_component_branch": promote,
            "incremental_live_weight": 0.05 if promote else 0.0,
            "change_headline_forecasts": False,
            "reason": (
                "All predeclared outer, narrow-ECI-CI, coverage, and frontier gates pass; admit only a small correlated ECI sub-branch."
                if promote
                else "The nested component model is retained as a zero-weight diagnostic because at least one predeclared narrow-ECI-CI, interval, coverage, or frontier gate fails."
            ),
        },
        "limitations": [
            "The component snapshot is current, not release-vintage historical data.",
            "Benchmark availability is highly non-random across release date and developer.",
            "Forty-eight of 89 aggregate ECI scores have confidence intervals wider than 10 points; this is score uncertainty, not parameter-disclosure status.",
            "The narrow-ECI-CI-only replication has substantially less chronological training coverage than the primary half-weight run.",
            "No parameter-disclosure classification is available in this panel, so this audit makes no parameter-disclosure claim.",
            "The benchmark residuals are correlated with aggregate ECI and cannot be counted as an independent likelihood.",
            "Target architecture flags assume Fable and Sol are reasoning MoEs, consistent with the working model but not public disclosures.",
        ],
        "source_files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                REGRESSION,
                COMPONENTS,
                BENCHMARK_ARCHIVE,
                UNIFIED,
                EXTENDED_AUDIT,
            )
        },
        "outputs": {
            "predictions": str(PREDICTIONS.relative_to(ROOT)),
            "narrow_eci_ci_only_predictions": str(
                NARROW_CI_PREDICTIONS.relative_to(ROOT)
            ),
            "targets": str(TARGETS.relative_to(ROOT)),
            "coverage": str(COVERAGE.relative_to(ROOT)),
        },
    }
    write_csv(PREDICTIONS, predictions)
    write_csv(NARROW_CI_PREDICTIONS, narrow_ci_predictions)
    write_csv(TARGETS, targets)
    write_csv(COVERAGE, coverage_rows)
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "inventory": result["inventory"],
                "total_backtest": scopes["total"],
                "narrow_eci_ci_only_training_total_backtest": narrow_ci_training_backtest[
                    "total"
                ],
                "targets": targets,
                "promotion_gates": result["promotion_gates"],
                "decision": result["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
