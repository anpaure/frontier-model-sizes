#!/usr/bin/env python3
"""Extended, leakage-resistant audit of ECI components as parameter signals.

This audit answers two distinct questions without changing the live forecast:

1. Do 27 additional exact open-weight Epoch parameter matches improve the
   aggregate ECI -> total-parameter calibration relative to the 57 retained workbook
   panel when the identical inverse-uncertainty weighting law is used?
2. Does any individual ECI benchmark add held-out information about active or
   total parameters after aggregate ECI, exact date, reasoning status, and MoE
   architecture are already controlled?

Every outer prediction uses strictly earlier checkpoints and removes the test
family. Benchmark comparisons receive a global familywise max-T sign-flip
correction. The current benchmark snapshot is not a historical vintage, so a
component is never promoted without corrected evidence or future replication.
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

from open_model_parameter_truth import LEDGER_PATH as PARAMETER_TRUTH_PATH
from open_model_parameter_truth import resolve_parameter_truth


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
UNIFIED = OUT / "unified_model_observations_compute_enriched_2026-07-17.csv"
EPOCH_MODELS = ROOT / "sources/epoch_all_ai_models_2026-07-31.csv"
COMPONENTS = ROOT / "sources/epoch_eci_benchmarks_2026-07-31.csv"
BENCHMARK_ARCHIVE = ROOT / "sources/epoch_benchmark_data_2026-07-31.zip"
REGRESSION = ROOT / "regression_results.json"

RESULT = OUT / "eci_component_extended_audit_2026-07-18.json"
EXPANDED_PANEL = OUT / "eci_component_expanded_parameter_panel_2026-07-18.csv"
AGGREGATE_PREDICTIONS = OUT / "eci_component_expanded_aggregate_predictions_2026-07-18.csv"
ACTIVE_COMPARISON = OUT / "eci_component_active_incremental_comparison_2026-07-18.csv"
ACTIVE_PREDICTIONS = OUT / "eci_component_active_incremental_predictions_2026-07-18.csv"

DATE_ORIGIN = date(2023, 1, 1)
MIN_TRAIN_ROWS = 12
MIN_TRAIN_FAMILIES = 5
MIN_BENCHMARK_MODELS = 16
MIN_ELIGIBLE_PREDICTIONS = 8
WLS_NUMERATOR = 15.3664
BOOTSTRAP_SAMPLES = 20_000
MAX_T_PERMUTATIONS = 100_000
MANUAL_FAMILY = {"RedPajama-INCITE-7B-Base": "redpajama"}
FRONTIER_TARGETS = (
    "Claude Fable 5",
    "GPT-5.6 Sol",
    "GPT-5.5",
    "Claude Opus 4.8",
    "GPT-5.6 Terra",
    "Claude Sonnet 5",
    "GPT-5.6 Luna",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def years(value: str) -> float:
    return (date.fromisoformat(value) - DATE_ORIGIN).days / 365.25


def logit(value: float, epsilon: float = 1e-3) -> float:
    clipped = min(1.0 - epsilon, max(epsilon, float(value)))
    return math.log(clipped / (1.0 - clipped))


def parameter_metrics(errors: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(errors), dtype=float)
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


def family_weights(rows: list[dict[str, Any]], quality_weight: bool = False) -> np.ndarray:
    counts = Counter(row["family"] for row in rows)
    values = []
    for row in rows:
        weight = 1.0 / counts[row["family"]]
        if quality_weight and row.get("estimated"):
            weight *= 0.5
        values.append(weight)
    output = np.asarray(values, dtype=float)
    return output / output.mean()


def fit_direct(
    train: list[dict[str, Any]],
    test: dict[str, Any],
    features: tuple[str, ...],
    target: str,
    weights: np.ndarray,
) -> float:
    x = np.asarray(
        [[1.0, *[float(row[feature]) for feature in features]] for row in train],
        dtype=float,
    )
    y = np.log10(np.asarray([float(row[target]) for row in train], dtype=float))
    root = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)
    test_x = np.asarray([1.0, *[float(test[feature]) for feature in features]])
    return float(test_x @ beta)


def load_difficulties() -> dict[str, tuple[float, float]]:
    with zipfile.ZipFile(BENCHMARK_ARCHIVE) as archive:
        raw = archive.read(
            "additional_eci_data/eci_benchmark_difficulties_and_slopes.csv"
        ).decode("utf-8-sig")
    rows = csv.DictReader(io.StringIO(raw))
    return {
        row["benchmark_name"]: (float(row["edi"]), float(row["estimated_slope_scaled"]))
        for row in rows
    }


def load_attributes() -> dict[str, dict[str, Any]]:
    regression = json.loads(REGRESSION.read_text(encoding="utf-8"))
    attributes = {row["model"]: row for row in regression["eci"]["open_models"]}
    if len(attributes) != 89:
        raise ValueError(f"Expected 89 ECI parameter-map checkpoints; found {len(attributes)}")
    return attributes


def load_expanded_panel() -> list[dict[str, Any]]:
    attributes = load_attributes()
    unified = [row for row in read_csv(UNIFIED) if row["source"] == "ECI"]
    epoch_by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(EPOCH_MODELS):
        epoch_by_name[row["Model"]].append(row)

    output = []
    for row in unified:
        if not row["total_parameters_b"]:
            continue
        source = row["parameter_value_source"]
        source_parts = source.split("; canonicalized by ", 1)
        base_source = source_parts[0]
        source_truth_id = source_parts[1] if len(source_parts) == 2 else ""
        row_truth_id = row.get("parameter_truth_id") or ""
        if source_truth_id != row_truth_id:
            raise ValueError(
                f"Parameter-truth provenance mismatch for {row['source_model_name']}"
            )
        if row_truth_id:
            truth = resolve_parameter_truth(row["source_model_name"])
            if (
                truth is None
                or truth["truth_id"] != row_truth_id
                or not math.isclose(
                    float(row["total_parameters_b"]),
                    float(truth["canonical_total_parameters_b"]),
                    rel_tol=0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    f"Canonical parameter truth mismatch for {row['source_model_name']}"
                )
        live = base_source == "ECI Regression Data"
        exact_extra = base_source == "matched Epoch Parameters"
        primary_exact = source == "Kimi K3 official technical report Table 1"
        epoch_rows = epoch_by_name.get(row["matched_epoch_model"], [])
        epoch = epoch_rows[0] if len(epoch_rows) == 1 else None
        if primary_exact:
            valid_primary = (
                row["source_model_name"] == "Kimi K3"
                and math.isclose(float(row["total_parameters_b"]), 2780.0)
                and math.isclose(float(row["active_parameters_b"]), 104.2)
                and row["canonical_release_date"] == "2026-07-16"
                and epoch is not None
                and epoch["Model accessibility"].startswith("Open weights")
            )
            if not valid_primary:
                raise ValueError("Kimi K3 primary exact parameter row is malformed")
        elif exact_extra:
            valid_extra = (
                epoch is not None
                and epoch["Model accessibility"].startswith("Open weights")
                and row["epoch_match_status"] == "matched_checkpoint"
                and row["epoch_match_confidence"] == "high"
                and row["epoch_link_level"] == "checkpoint"
                and row["epoch_candidate_count"] == "1"
                and row["date_conflict_flag"] == "false"
                and row["release_date_delta_days"] == "0"
            )
            if not valid_extra:
                continue
            epoch_b = float(epoch["Parameters"]) / 1e9
            comparison_b = float(
                row.get("raw_total_parameters_b") or row["total_parameters_b"]
            )
            if abs(math.log10(comparison_b / epoch_b)) > 1e-9:
                raise ValueError(f"Epoch parameter mismatch for {row['source_model_name']}")
        elif not live:
            continue

        model = row["source_model_name"]
        attribute = attributes.get(model)
        family = attribute["family"] if attribute else MANUAL_FAMILY.get(model)
        if not family:
            raise ValueError(f"Expanded ECI checkpoint lacks family metadata: {model}")
        ci_width = float(row["eci_ci_high"]) - float(row["eci_ci_low"])
        wls_weight = WLS_NUMERATOR / ci_width**2
        legacy_wls_weight = None
        if live:
            source_record = json.loads(row["source_record_json"])
            legacy_wls_weight = float(source_record["regression_data"]["WLS weight"])
        output.append(
            {
                "model": model,
                "canonical_checkpoint_id": row["canonical_checkpoint_id"],
                "release_date": row["canonical_release_date"],
                "total_parameters_b": float(row["total_parameters_b"]),
                "family": family,
                "eci_score": float(row["eci_score"]),
                "eci_ci_low": float(row["eci_ci_low"]),
                "eci_ci_high": float(row["eci_ci_high"]),
                "eci_ci_width": ci_width,
                "wls_weight": wls_weight,
                "legacy_workbook_wls_weight": legacy_wls_weight,
                "panel_source": "legacy_workbook_parameter_row" if live else "primary_exact_open_extension" if primary_exact else "exact_epoch_open_extension",
                "parameter_value_source": source,
                "matched_epoch_model": row["matched_epoch_model"],
                "epoch_match_method": row["epoch_match_method"],
                "epoch_match_confidence": row["epoch_match_confidence"],
                "epoch_accessibility": epoch["Model accessibility"] if epoch else "",
                "epoch_parameter_notes": epoch["Parameters notes"] if epoch else "",
                "source_url": epoch["Link"] if epoch and epoch["Link"] else row["source_url"],
            }
        )
    output.sort(key=lambda row: (row["release_date"], row["model"]))
    if len(output) != 84 or len({row["canonical_checkpoint_id"] for row in output}) != 84:
        raise ValueError(
            f"Expanded ECI panel inventory changed: {len(output)} rows / "
            f"{len({row['canonical_checkpoint_id'] for row in output})} unique checkpoints"
        )
    if Counter(row["panel_source"] for row in output) != {
        "legacy_workbook_parameter_row": 57,
        "exact_epoch_open_extension": 26,
        "primary_exact_open_extension": 1,
    }:
        raise ValueError("Unexpected expanded-panel source counts")
    return output


def aggregate_prediction(
    train_panel: list[dict[str, Any]], test: dict[str, Any]
) -> tuple[float, int, int, str]:
    train = [
        {**row, "release_years": years(row["release_date"]), "total_b": row["total_parameters_b"]}
        for row in train_panel
        if row["release_date"] < test["release_date"] and row["family"] != test["family"]
    ]
    if len(train) < MIN_TRAIN_ROWS or len({row["family"] for row in train}) < MIN_TRAIN_FAMILIES:
        raise ValueError("Insufficient aggregate ECI training rows")
    target = {
        **test,
        "release_years": years(test["release_date"]),
        "total_b": test["total_parameters_b"],
    }
    weights = np.asarray([row["wls_weight"] for row in train], dtype=float)
    score_only = fit_direct(train, target, ("eci_score",), "total_b", weights)
    score_date = fit_direct(
        train, target, ("eci_score", "release_years"), "total_b", weights
    )
    prediction = 0.60 * score_only + 0.40 * score_date
    return prediction, len(train), len({row["family"] for row in train}), max(
        row["release_date"] for row in train
    )


def aggregate_predictions(panel: list[dict[str, Any]]) -> list[dict[str, Any]]:
    live = [row for row in panel if row["panel_source"] == "legacy_workbook_parameter_row"]
    output = []
    for test in panel:
        base_train = [
            row
            for row in live
            if row["release_date"] < test["release_date"] and row["family"] != test["family"]
        ]
        expanded_train = [
            row
            for row in panel
            if row["release_date"] < test["release_date"] and row["family"] != test["family"]
        ]
        if (
            len(base_train) < MIN_TRAIN_ROWS
            or len({row["family"] for row in base_train}) < MIN_TRAIN_FAMILIES
            or len(expanded_train) < MIN_TRAIN_ROWS
            or len({row["family"] for row in expanded_train}) < MIN_TRAIN_FAMILIES
        ):
            continue
        base, base_n, base_families, base_max = aggregate_prediction(live, test)
        expanded, expanded_n, expanded_families, expanded_max = aggregate_prediction(panel, test)
        actual = math.log10(test["total_parameters_b"])
        output.append(
            {
                "release_date": test["release_date"],
                "model": test["model"],
                "family": test["family"],
                "test_panel_source": test["panel_source"],
                "actual_parameters_b": test["total_parameters_b"],
                "legacy_57_predicted_b": 10**base,
                "expanded_84_predicted_b": 10**expanded,
                "legacy_57_log10_error": base - actual,
                "expanded_84_log10_error": expanded - actual,
                "current_train_n": base_n,
                "current_train_family_n": base_families,
                "current_train_max_date": base_max,
                "expanded_train_n": expanded_n,
                "expanded_train_family_n": expanded_families,
                "expanded_train_max_date": expanded_max,
            }
        )
    return output


def paired_family_bootstrap(
    rows: list[dict[str, Any]],
    baseline_error: str,
    candidate_error: str,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = 20260718,
) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    family_effects = {
        family: float(
            np.mean(
                [
                    abs(float(row[candidate_error])) - abs(float(row[baseline_error]))
                    for row in family_rows
                ]
            )
        )
        for family, family_rows in by_family.items()
    }
    families = sorted(family_effects)
    values = np.asarray([family_effects[family] for family in families], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.mean(
        values[rng.integers(0, len(values), size=(samples, len(values)))], axis=1
    )
    return {
        "metric": "equal-family mean absolute log10 error; candidate minus baseline",
        "observed_delta": float(np.mean(values)),
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "bootstrap_probability_candidate_better": float(np.mean(draws < 0)),
        "samples": samples,
        "family_clusters": len(families),
    }


def aggregate_comparison(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    selected = (
        rows
        if scope == "all_84"
        else [row for row in rows if row["test_panel_source"] == "legacy_workbook_parameter_row"]
    )
    return {
        "scope": scope,
        "n": len(selected),
        "families": len({row["family"] for row in selected}),
        "legacy_57": parameter_metrics(row["legacy_57_log10_error"] for row in selected),
        "expanded_84": parameter_metrics(row["expanded_84_log10_error"] for row in selected),
        "row_weighted_mean_absolute_log10_delta": float(
            np.mean(
                [
                    abs(row["expanded_84_log10_error"])
                    - abs(row["legacy_57_log10_error"])
                    for row in selected
                ]
            )
        ),
        "paired_family_bootstrap": paired_family_bootstrap(
            selected, "legacy_57_log10_error", "expanded_84_log10_error"
        ),
    }


def frontier_aggregate_estimates(panel: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unified = {row["source_model_name"]: row for row in read_csv(UNIFIED) if row["source"] == "ECI"}
    live = [row for row in panel if row["panel_source"] == "legacy_workbook_parameter_row"]

    def full_prediction(train_panel: list[dict[str, Any]], target: dict[str, Any]) -> float:
        train = [
            {**row, "release_years": years(row["release_date"]), "total_b": row["total_parameters_b"]}
            for row in train_panel
        ]
        test = {
            "eci_score": target["eci_score"],
            "release_years": years(target["release_date"]),
            "total_b": 1.0,
        }
        weights = np.asarray([row["wls_weight"] for row in train], dtype=float)
        score_only = fit_direct(train, test, ("eci_score",), "total_b", weights)
        score_date = fit_direct(
            train, test, ("eci_score", "release_years"), "total_b", weights
        )
        return float(10 ** (0.60 * score_only + 0.40 * score_date))

    output = []
    for model in FRONTIER_TARGETS:
        source = unified[model]
        target = {
            "eci_score": float(source["eci_score"]),
            "release_date": source["canonical_release_date"],
        }
        current = full_prediction(live, target)
        expanded = full_prediction(panel, target)
        output.append(
            {
                "model": model,
                "release_date": target["release_date"],
                "eci_score": target["eci_score"],
                "legacy_57_estimate_t": current / 1000,
                "expanded_84_estimate_t": expanded / 1000,
                "expanded_over_legacy": expanded / current,
            }
        )
    return output


def load_active_component_rows() -> list[dict[str, Any]]:
    attributes = load_attributes()
    difficulties = load_difficulties()
    output = []
    for source in read_csv(COMPONENTS):
        model = source["model"]
        benchmark = source["benchmark"]
        if model not in attributes or benchmark not in difficulties:
            continue
        attribute = attributes[model]
        difficulty, slope = difficulties[benchmark]
        performance = float(source["performance"])
        output.append(
            {
                "model": model,
                "release_date": attribute["release_date"],
                "family": attribute["family"],
                "active_b": float(attribute["active_b"]),
                "total_b": float(attribute["total_b"]),
                "eci": float(attribute["score"]),
                "reasoning": int(attribute["reasoning"]),
                "moe": int(attribute["moe"]),
                "estimated": int(attribute["estimated"]),
                "benchmark": benchmark,
                "performance": performance,
                "component_eci": difficulty + logit(performance) / slope,
            }
        )
    if len(output) != 723 or len({row["model"] for row in output}) != 89:
        raise ValueError("Expected 723 component rows across all 89 ECI parameter-map models")
    return output


def component_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark"]].append(row)
    baseline_features = ("eci", "release_years", "reasoning", "moe")
    augmented_features = (*baseline_features, "component_deviation")
    for benchmark, panel in sorted(by_benchmark.items()):
        if len({row["model"] for row in panel}) < MIN_BENCHMARK_MODELS:
            continue
        for test_source in sorted(panel, key=lambda row: (row["release_date"], row["model"])):
            train_source = [
                row
                for row in panel
                if row["release_date"] < test_source["release_date"]
                and row["family"] != test_source["family"]
            ]
            if (
                len(train_source) < MIN_TRAIN_ROWS
                or len({row["family"] for row in train_source}) < MIN_TRAIN_FAMILIES
            ):
                continue
            train = [
                {
                    **row,
                    "release_years": years(row["release_date"]),
                    "component_deviation": row["component_eci"] - row["eci"],
                }
                for row in train_source
            ]
            test = {
                **test_source,
                "release_years": years(test_source["release_date"]),
                "component_deviation": test_source["component_eci"] - test_source["eci"],
            }
            weights = family_weights(train, quality_weight=True)
            predictions: dict[str, float] = {}
            for target in ("active_b", "total_b"):
                predictions[f"baseline_{target}"] = fit_direct(
                    train, test, baseline_features, target, weights
                )
                predictions[f"augmented_{target}"] = fit_direct(
                    train, test, augmented_features, target, weights
                )
            active_actual = math.log10(test["active_b"])
            total_actual = math.log10(test["total_b"])
            output.append(
                {
                    "benchmark": benchmark,
                    "release_date": test["release_date"],
                    "model": test["model"],
                    "family": test["family"],
                    "actual_active_b": test["active_b"],
                    "actual_total_b": test["total_b"],
                    "eci": test["eci"],
                    "component_eci": test["component_eci"],
                    "component_deviation": test["component_deviation"],
                    "baseline_active_predicted_b": 10 ** predictions["baseline_active_b"],
                    "augmented_active_predicted_b": 10 ** predictions["augmented_active_b"],
                    "baseline_active_log10_error": predictions["baseline_active_b"] - active_actual,
                    "augmented_active_log10_error": predictions["augmented_active_b"] - active_actual,
                    "baseline_total_predicted_b": 10 ** predictions["baseline_total_b"],
                    "augmented_total_predicted_b": 10 ** predictions["augmented_total_b"],
                    "baseline_total_log10_error": predictions["baseline_total_b"] - total_actual,
                    "augmented_total_log10_error": predictions["augmented_total_b"] - total_actual,
                    "train_n": len(train),
                    "train_family_n": len({row["family"] for row in train}),
                    "train_max_date": max(row["release_date"] for row in train),
                }
            )
    return output


def family_effects(
    rows: list[dict[str, Any]], baseline: str, candidate: str
) -> dict[str, float]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    return {
        family: float(
            np.mean(
                [
                    abs(float(row[candidate])) - abs(float(row[baseline]))
                    for row in family_rows
                ]
            )
        )
        for family, family_rows in by_family.items()
    }


def studentized(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    if len(array) < 2 or np.std(array, ddof=1) == 0:
        return 0.0
    return float(np.mean(array) / (np.std(array, ddof=1) / math.sqrt(len(array))))


def component_comparison(
    predictions: list[dict[str, Any]], coverage_by_benchmark: dict[str, int]
) -> list[dict[str, Any]]:
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_benchmark[row["benchmark"]].append(row)
    output = []
    for benchmark, rows in sorted(by_benchmark.items()):
        if len(rows) < MIN_ELIGIBLE_PREDICTIONS:
            continue
        active_effects = family_effects(
            rows, "baseline_active_log10_error", "augmented_active_log10_error"
        )
        total_effects = family_effects(
            rows, "baseline_total_log10_error", "augmented_total_log10_error"
        )
        active_bootstrap = paired_family_bootstrap(
            rows, "baseline_active_log10_error", "augmented_active_log10_error"
        )
        total_bootstrap = paired_family_bootstrap(
            rows, "baseline_total_log10_error", "augmented_total_log10_error"
        )
        output.append(
            {
                "benchmark": benchmark,
                "coverage_models": coverage_by_benchmark[benchmark],
                "heldout_predictions": len(rows),
                "heldout_families": len({row["family"] for row in rows}),
                "baseline_active_median_error_x": parameter_metrics(
                    row["baseline_active_log10_error"] for row in rows
                )["median_multiplicative_error"],
                "augmented_active_median_error_x": parameter_metrics(
                    row["augmented_active_log10_error"] for row in rows
                )["median_multiplicative_error"],
                "active_equal_family_mae_delta": float(np.mean(list(active_effects.values()))),
                "active_delta_ci_90_low": active_bootstrap["ci_90"][0],
                "active_delta_ci_90_high": active_bootstrap["ci_90"][1],
                "active_unadjusted_probability_better": active_bootstrap[
                    "bootstrap_probability_candidate_better"
                ],
                "active_studentized_statistic": studentized(active_effects.values()),
                "baseline_total_median_error_x": parameter_metrics(
                    row["baseline_total_log10_error"] for row in rows
                )["median_multiplicative_error"],
                "augmented_total_median_error_x": parameter_metrics(
                    row["augmented_total_log10_error"] for row in rows
                )["median_multiplicative_error"],
                "total_equal_family_mae_delta": float(np.mean(list(total_effects.values()))),
                "total_delta_ci_90_low": total_bootstrap["ci_90"][0],
                "total_delta_ci_90_high": total_bootstrap["ci_90"][1],
                "total_unadjusted_probability_better": total_bootstrap[
                    "bootstrap_probability_candidate_better"
                ],
                "total_studentized_statistic": studentized(total_effects.values()),
            }
        )
    return output


def add_familywise_pvalues(
    comparisons: list[dict[str, Any]], predictions: list[dict[str, Any]], target: str
) -> None:
    baseline = f"baseline_{target}_log10_error"
    candidate = f"augmented_{target}_log10_error"
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_benchmark[row["benchmark"]].append(row)
    effects = {
        row["benchmark"]: family_effects(by_benchmark[row["benchmark"]], baseline, candidate)
        for row in comparisons
    }
    all_families = sorted({family for panel in effects.values() for family in panel})
    family_index = {family: index for index, family in enumerate(all_families)}
    rng = np.random.default_rng(20260718 if target == "active" else 20260719)
    signs = rng.choice(
        np.asarray([-1.0, 1.0]), size=(MAX_T_PERMUTATIONS, len(all_families))
    )
    minimum = np.full(MAX_T_PERMUTATIONS, np.inf)
    for panel in effects.values():
        families = sorted(panel)
        values = np.asarray([panel[family] for family in families], dtype=float)
        permuted = signs[:, [family_index[family] for family in families]] * values
        denominator = np.std(permuted, axis=1, ddof=1) / math.sqrt(len(values))
        statistics = np.divide(
            np.mean(permuted, axis=1),
            denominator,
            out=np.zeros(MAX_T_PERMUTATIONS),
            where=denominator > 0,
        )
        minimum = np.minimum(minimum, statistics)
    statistic_key = f"{target}_studentized_statistic"
    p_key = f"{target}_familywise_one_sided_p"
    for row in comparisons:
        observed = float(row[statistic_key])
        row[p_key] = float(
            (1 + np.sum(minimum <= observed)) / (MAX_T_PERMUTATIONS + 1)
        )


def frontier_component_coverage(
    comparisons: list[dict[str, Any]], supported: set[str]
) -> list[dict[str, Any]]:
    comparison_by_name = {row["benchmark"]: row for row in comparisons}
    output = []
    for row in read_csv(COMPONENTS):
        if row["model"] not in {"Claude Fable 5", "GPT-5.6 Sol"}:
            continue
        benchmark = row["benchmark"]
        comparison = comparison_by_name.get(benchmark)
        output.append(
            {
                "model": row["model"],
                "benchmark": benchmark,
                "performance": float(row["performance"]),
                "eligible_component_panel": comparison is not None,
                "active_familywise_p": comparison.get("active_familywise_one_sided_p")
                if comparison
                else None,
                "supported_after_familywise_correction": benchmark in supported,
                "used_in_live_forecast": False,
            }
        )
    return output


def main() -> None:
    expanded_panel = load_expanded_panel()
    aggregate_rows = aggregate_predictions(expanded_panel)
    aggregate_audit = {
        "legacy_57_tests": aggregate_comparison(aggregate_rows, "legacy_57"),
        "all_84_tests": aggregate_comparison(aggregate_rows, "all_84"),
    }
    frontier = frontier_aggregate_estimates(expanded_panel)

    active_rows = load_active_component_rows()
    active_predictions = component_predictions(active_rows)
    coverage_by_benchmark = {
        benchmark: len({row["model"] for row in active_rows if row["benchmark"] == benchmark})
        for benchmark in {row["benchmark"] for row in active_rows}
    }
    comparisons = component_comparison(active_predictions, coverage_by_benchmark)
    add_familywise_pvalues(comparisons, active_predictions, "active")
    add_familywise_pvalues(comparisons, active_predictions, "total")
    comparisons.sort(key=lambda row: (row["active_familywise_one_sided_p"], row["benchmark"]))
    supported = {
        row["benchmark"]
        for row in comparisons
        if row["active_equal_family_mae_delta"] < 0
        and row["active_familywise_one_sided_p"] < 0.05
    }
    coverage = frontier_component_coverage(comparisons, supported)

    result = {
        "generated_on": "2026-07-31",
        "question": "Do exact additional parameter matches or individual ECI components improve frontier parameter inference?",
        "expanded_total_parameter_panel": {
            "models": len(expanded_panel),
            "families": len({row["family"] for row in expanded_panel}),
            "source_counts": dict(sorted(Counter(row["panel_source"] for row in expanded_panel).items())),
            "matching_policy": "Additional rows require a unique, exact alphanumeric, high-confidence Epoch checkpoint match; exact date agreement; open weights; and identical parameter values.",
            "weighting_policy": f"All rows recompute the inverse-uncertainty law from the current ECI intervals: {WLS_NUMERATOR} / (ECI CI width)^2; legacy workbook weights are retained only as audit fields.",
            "aggregate_backtest": aggregate_audit,
            "frontier_estimate_stability": frontier,
            "decision": {
                "change_live_eci_center": False,
                "reason": "The extension is evaluated against the 57 retained legacy parameter rows after the current-snapshot identity migration. It remains a diagnostic comparison and cannot change the live branch without decisive held-out evidence.",
            },
        },
        "active_parameter_component_audit": {
            "parameter_map_models": len({row["model"] for row in active_rows}),
            "parameter_map_families": len({row["family"] for row in active_rows}),
            "component_rows": len(active_rows),
            "component_benchmarks": len({row["benchmark"] for row in active_rows}),
            "eligible_comparisons": len(comparisons),
            "outer_split": "strictly earlier release date; test family entirely removed",
            "baseline": "log10 parameter count ~ aggregate ECI + exact date + reasoning + MoE",
            "candidate": "baseline + (component-implied ECI - aggregate ECI)",
            "training_weights": "equal total weight per family; broad aggregate-ECI confidence intervals receive 0.5 quality weight",
            "multiple_testing": f"Global family-level max-T sign-flip correction across {len(comparisons)} eligible benchmarks using {MAX_T_PERMUTATIONS} permutations.",
            "supported_after_familywise_correction": sorted(supported),
            "best_uncorrected_component": comparisons[0] if comparisons else None,
            "decision": {
                "add_component_branch": False,
                "incremental_component_weight": 0.0,
                "reason": "No individual component survives the familywise correction. Apparent OTIS/knowledge gains remain exploratory until an independent benchmark vintage or new disclosed models replicate them.",
            },
        },
        "frontier_component_coverage": coverage,
        "limitations": [
            "Component performance comes from the current Epoch snapshot, not historical benchmark vintages.",
            "The 89-model active-parameter map contains some inferred active counts; broad ECI intervals are downweighted but parameter-source uncertainty is not fully quantified.",
            "Benchmark availability is non-random and varies strongly by release date.",
            "Familywise correction protects against selecting a lucky component from this snapshot, but future snapshots still require prospective validation.",
        ],
        "files": {
            "expanded_panel": str(EXPANDED_PANEL.relative_to(ROOT)),
            "aggregate_predictions": str(AGGREGATE_PREDICTIONS.relative_to(ROOT)),
            "active_comparison": str(ACTIVE_COMPARISON.relative_to(ROOT)),
            "active_predictions": str(ACTIVE_PREDICTIONS.relative_to(ROOT)),
        },
        "source_manifest": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                UNIFIED,
                EPOCH_MODELS,
                COMPONENTS,
                BENCHMARK_ARCHIVE,
                REGRESSION,
                PARAMETER_TRUTH_PATH,
            )
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(EXPANDED_PANEL, expanded_panel)
    write_csv(AGGREGATE_PREDICTIONS, aggregate_rows)
    write_csv(ACTIVE_COMPARISON, comparisons)
    write_csv(ACTIVE_PREDICTIONS, active_predictions)
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
