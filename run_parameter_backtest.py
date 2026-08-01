#!/usr/bin/env python3
"""Leakage-controlled backtests for the frontier parameter forecasting pipeline.

The target is disclosed total parameter count in billions.  Every outer-fold
prediction is trained only on models with a strictly earlier release date.
The preferred split also removes the held-out checkpoint's entire model/lab
family from the training fold.

This is a pseudo-chronological backtest: benchmark scores are taken from the
current audited snapshots, not from historical benchmark vintages.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from aa_calibration_overrides import parameter_training_eligibility_date
from aa_parameter_label_availability import (
    LEDGER_PATH as AA_PARAMETER_LABEL_AVAILABILITY_PATH,
    load_parameter_label_availability,
)
from aa_score_availability import (
    LEDGER_PATH as AA_SCORE_AVAILABILITY_PATH,
    RAW_PATH as AA_CHANGELOG_PATH,
    aa_prediction_information_date,
    load_aa_score_availability,
)
from k3_primary_evidence import K3_EVIDENCE_PATH, K3_PARAMETER_SOURCE, K3_TOTAL_B
from open_model_parameter_truth import (
    LEDGER_PATH as OPEN_MODEL_PARAMETER_TRUTH_PATH,
    apply_parameter_truth,
)
from artifact_paths import portable_path
from frontier_target_signals import AA_DETAILED_PATH, AA_TARGET_SIGNALS


WORK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = WORK_DIR / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
REGRESSION_PATH = WORK_DIR / "regression_results.json"
UNIFIED_PATH = OUTPUT_DIR / "unified_model_observations_compute_enriched_2026-07-17.csv"
RESULT_PATH = OUTPUT_DIR / "frontier_parameter_chronological_backtest_2026-07-17.json"
PREDICTION_CSV_PATH = OUTPUT_DIR / "frontier_parameter_backtest_predictions_2026-07-17.csv"
COMPARISON_CSV_PATH = OUTPUT_DIR / "frontier_parameter_backtest_model_comparison_2026-07-17.csv"

DATE_ORIGIN = date(2023, 1, 1)
CURRENT_WEIGHTS = {"AA": 19.125, "ECI": 19.125, "No-CoT": 50.0, "Compute": 5.0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _years(iso: str) -> float:
    return (date.fromisoformat(iso) - DATE_ORIGIN).days / 365.25


def _feature_value(row: dict[str, Any], feature: str) -> float:
    if feature == "date":
        return _years(row["release_date"])
    return float(row.get(feature, 0.0) or 0.0)


def _optional_binary(row: dict[str, Any], field: str) -> int | None:
    value = row.get(field)
    if value in (None, ""):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "reasoning", "hybrid-reasoning"}:
            return 1
        if normalized in {"false", "no", "non-reasoning"}:
            return 0
    return int(float(value))


def _architecture_moe(value: Any) -> int | None:
    normalized = str(value or "").strip().lower()
    if not normalized or normalized in {"unknown", "not specified"}:
        return None
    if "moe" in normalized or "mixture of experts" in normalized:
        return 1
    if "dense" in normalized:
        return 0
    return None


def _design(rows: list[dict[str, Any]], features: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[1.0, *[_feature_value(row, feature) for feature in features]] for row in rows],
        dtype=float,
    )


def _family_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = Counter(row["family"] for row in rows)
    weights = np.asarray(
        [
            (0.5 if int(row.get("estimated", 0) or 0) else 1.0) / counts[row["family"]]
            for row in rows
        ],
        dtype=float,
    )
    return weights / weights.mean()


def _fit_predict(
    train: list[dict[str, Any]], test: dict[str, Any], features: tuple[str, ...]
) -> tuple[float, list[float]]:
    x_train = _design(train, features)
    y_train = np.log10(np.asarray([row["total_b"] for row in train], dtype=float))
    weights = _family_weights(train)
    sqrt_w = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(x_train * sqrt_w[:, None], y_train * sqrt_w, rcond=None)
    prediction = float((_design([test], features) @ beta).item())
    return prediction, [float(value) for value in beta]


def _metric_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not predictions:
        return {"n": 0}
    errors = np.asarray([row["log10_error"] for row in predictions], dtype=float)
    absolute = np.abs(errors)
    return {
        "n": int(len(predictions)),
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "geomean_multiplicative_error": float(10 ** np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(errors**2))),
        "signed_bias_factor": float(10 ** np.mean(errors)),
        "within_1_5x": float(np.mean(absolute <= math.log10(1.5))),
        "within_2x": float(np.mean(absolute <= math.log10(2.0))),
        "within_3x": float(np.mean(absolute <= math.log10(3.0))),
        "p80_multiplicative_error": float(10 ** np.quantile(absolute, 0.80)),
        "p90_multiplicative_error": float(10 ** np.quantile(absolute, 0.90)),
    }


def _backtest(
    panel: str,
    rows: list[dict[str, Any]],
    spec: str,
    features: tuple[str, ...],
    min_train: int,
    min_families: int,
    exclude_family: bool,
    primary_signal: str,
) -> list[dict[str, Any]]:
    def information_date(row: dict[str, Any]) -> str:
        return (
            aa_prediction_information_date(row)
            if panel == "AA"
            else str(row["release_date"])
        )

    ordered = sorted(
        rows,
        key=lambda row: (information_date(row), row["release_date"], row["model"]),
    )
    output: list[dict[str, Any]] = []
    for test in ordered:
        test_information_date = information_date(test)
        prior = [
            row
            for row in ordered
            if parameter_training_eligibility_date(row) < test_information_date
        ]
        if not prior:
            continue
        rank = sum(_feature_value(row, primary_signal) <= _feature_value(test, primary_signal) for row in prior) / len(prior)
        train = [row for row in prior if not exclude_family or row["family"] != test["family"]]
        if len(train) < min_train or len({row["family"] for row in train}) < min_families:
            continue
        prediction_log10, coefficients = _fit_predict(train, test, features)
        actual_log10 = math.log10(float(test["total_b"]))
        error = prediction_log10 - actual_log10
        output.append(
            {
                "panel": panel,
                "split": "chronological_family_holdout" if exclude_family else "chronological",
                "spec": spec,
                "release_date": test["release_date"],
                "prediction_information_date": test_information_date,
                "model": test["model"],
                "family": test["family"],
                "moe": _optional_binary(test, "moe"),
                "reasoning": _optional_binary(test, "reasoning"),
                "actual_b": float(test["total_b"]),
                "predicted_b": float(10**prediction_log10),
                "log10_error": float(error),
                "multiplicative_error": float(10 ** abs(error)),
                "signed_prediction_ratio": float(10**error),
                "train_n": len(train),
                "train_family_n": len({row["family"] for row in train}),
                "train_max_date": max(
                    parameter_training_eligibility_date(row) for row in train
                ),
                "test_family_excluded": not any(row["family"] == test["family"] for row in train),
                "frontier_signal_rank": float(rank),
                "coefficients": coefficients,
            }
        )
    return output


def _blend_predictions(
    panel: str,
    spec: str,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    left_weight: float,
) -> list[dict[str, Any]]:
    left_map = {(row["release_date"], row["model"]): row for row in left}
    right_map = {(row["release_date"], row["model"]): row for row in right}
    output = []
    for key in sorted(left_map.keys() & right_map.keys()):
        a, b = left_map[key], right_map[key]
        prediction_log10 = left_weight * math.log10(a["predicted_b"]) + (1 - left_weight) * math.log10(b["predicted_b"])
        actual_log10 = math.log10(a["actual_b"])
        error = prediction_log10 - actual_log10
        row = dict(a)
        row.update(
            {
                "panel": panel,
                "spec": spec,
                "predicted_b": float(10**prediction_log10),
                "log10_error": float(error),
                "multiplicative_error": float(10 ** abs(error)),
                "signed_prediction_ratio": float(10**error),
                "train_n": min(a["train_n"], b["train_n"]),
                "train_family_n": min(a["train_family_n"], b["train_family_n"]),
                "train_max_date": max(a["train_max_date"], b["train_max_date"]),
                "coefficients": [],
            }
        )
        output.append(row)
    return output


def _horizon_family(model: str) -> str:
    lower = model.lower()
    if "ministral" in lower:
        return "mistral"
    for family in ("llama", "mistral", "qwen", "gemma", "deepseek", "kimi"):
        if family in lower:
            return family
    raise ValueError(f"Unmapped No-CoT family: {model}")


def _normal_model_name(model: str) -> str:
    text = model.lower().replace("non-reasoning", "").replace("reasoning", "").replace("thinking", "")
    text = re.sub(r"\b(instruct|chat|high|medium|low|preview|latest|max|hosted)\b", "", text)
    text = re.sub(r"\((jul|mar|may|sep|nov)\s+\d{4}\)", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


PARAMETER_TRUTH_SOURCE_PRIORITY = {
    "Primary evidence": -100,
    "Epoch": 10,
    "ECI": 20,
    "AA": 30,
    "No-CoT": 40,
    "METR": 50,
    "Regression parameter registry": 60,
}
PARAMETER_TRUTH_MATCH_RATIO = 1.02
PARAMETER_TRUTH_DATE_TOLERANCE_DAYS = 62


def _parameter_truth_priority(row: dict[str, str]) -> int:
    provenance = str(row.get("parameter_value_source", "")).lower()
    if "official technical report" in provenance or "primary evidence" in provenance:
        return 0
    return PARAMETER_TRUTH_SOURCE_PRIORITY.get(row.get("source", ""), 100)


def _load_parameter_truth_registry() -> dict[str, list[dict[str, Any]]]:
    """Load one provenance-bearing target registry from the canonical megafile.

    Component panels sometimes retain rounded copies of the same disclosed
    parameter count.  Those copies are useful reconciliation evidence, but
    they are not independent target labels and must never be averaged.  The
    registry keeps every eligible source row and resolves one source-priority
    truth only after the ensemble identity/date checks have passed.
    """

    registry: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with UNIFIED_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("model_level_include") != "true"
                or not row.get("total_parameters_b")
                or not row.get("canonical_release_date")
            ):
                continue
            key = _normal_model_name(row["canonical_display_name"])
            canonical = apply_parameter_truth(
                {
                    "model": row["canonical_display_name"],
                    "total_b": float(row["total_parameters_b"]),
                }
            )
            registry[key].append(
                {
                    "actual_b": float(canonical["total_b"]),
                    "canonical_checkpoint_id": row["canonical_checkpoint_id"],
                    "canonical_display_name": row["canonical_display_name"],
                    "release_date": row["canonical_release_date"],
                    "source": row["source"],
                    "parameter_value_source": row["parameter_value_source"],
                    "source_url": row["source_url"],
                    "source_locator": row["source_locator"],
                    "priority": _parameter_truth_priority(row),
                }
            )

    regression = json.loads(REGRESSION_PATH.read_text(encoding="utf-8"))
    for panel_name, rows in (
        ("AA", regression["open_models"]),
        ("ECI", regression["eci"]["open_models"]),
    ):
        for row in rows:
            key = _normal_model_name(row["model"])
            registry[key].append(
                {
                    "actual_b": float(row["total_b"]),
                    "canonical_checkpoint_id": f"regression:{key}",
                    "canonical_display_name": row["model"],
                    "release_date": row["release_date"],
                    "source": "Regression parameter registry",
                    "parameter_value_source": row.get("parameter_source")
                    or f"{panel_name} audited parameter map",
                    "source_url": row.get("parameter_source") or row.get("source", ""),
                    "source_locator": portable_path(REGRESSION_PATH),
                    "priority": PARAMETER_TRUTH_SOURCE_PRIORITY[
                        "Regression parameter registry"
                    ],
                }
            )

    k3_evidence = json.loads(K3_EVIDENCE_PATH.read_text(encoding="utf-8"))
    k3_url = k3_evidence["source_files"]["official_technical_report"]["url"]
    registry[_normal_model_name("Kimi K3")].append(
        {
            "actual_b": K3_TOTAL_B,
            "canonical_checkpoint_id": "checkpoint:moonshot:kimi-k3",
            "canonical_display_name": "Kimi K3",
            "release_date": "2026-07-16",
            "source": "Primary evidence",
            "parameter_value_source": K3_PARAMETER_SOURCE,
            "source_url": k3_url,
            "source_locator": portable_path(K3_EVIDENCE_PATH),
            "priority": PARAMETER_TRUTH_SOURCE_PRIORITY["Primary evidence"],
        }
    )
    return dict(registry)


def _resolve_parameter_truth(
    key: str,
    components: list[tuple[str, dict[str, Any]]],
    registry: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    component_actuals = [float(row["actual_b"]) for _, row in components]
    component_dates = [date.fromisoformat(row["release_date"]) for _, row in components]
    eligible: list[tuple[int, int, str, str, dict[str, Any]]] = []
    for candidate in registry.get(key, []):
        value = float(candidate["actual_b"])
        if any(max(value, actual) / min(value, actual) >= PARAMETER_TRUTH_MATCH_RATIO for actual in component_actuals):
            continue
        candidate_date = date.fromisoformat(candidate["release_date"])
        date_distance = max(abs((candidate_date - observed).days) for observed in component_dates)
        if date_distance > PARAMETER_TRUTH_DATE_TOLERANCE_DAYS:
            continue
        eligible.append(
            (
                int(candidate["priority"]),
                date_distance,
                str(candidate["canonical_checkpoint_id"]),
                str(candidate["source_locator"]),
                candidate,
            )
        )
    if not eligible:
        return None
    return dict(min(eligible, key=lambda item: item[:-1])[-1])


def _load_panels() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    regression = json.loads(REGRESSION_PATH.read_text())
    with UNIFIED_PATH.open(newline="", encoding="utf-8") as handle:
        unified = list(csv.DictReader(handle))

    aa = [dict(row) for row in regression["open_models"]]
    eci = [dict(row) for row in regression["eci"]["open_models"]]

    horizon: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in unified:
        checkpoint = row["canonical_checkpoint_id"]
        if (
            not row["nocot_time_horizon_minutes"]
            or not row["total_parameters_b"]
            or checkpoint in seen
            or row["canonical_display_name"].startswith("GPT-4")
        ):
            continue
        seen.add(checkpoint)
        horizon.append(
            apply_parameter_truth({
                "release_date": row["canonical_release_date"],
                "model": row["canonical_display_name"],
                "total_b": float(row["total_parameters_b"]),
                "family": _horizon_family(row["canonical_display_name"]),
                "log_horizon": math.log10(float(row["nocot_time_horizon_minutes"])),
                "moe": _architecture_moe(row["architecture"]),
                "reasoning": _optional_binary(row, "reasoning"),
                "estimated": 0,
            })
        )

    compute: list[dict[str, Any]] = []
    seen.clear()
    for row in unified:
        checkpoint = row["canonical_checkpoint_id"]
        if (
            row["source"] != "Epoch"
            or checkpoint in seen
            or row["epoch_open_model_weights"] != "Yes"
            or not row["epoch_training_compute_flop"]
            or not row["total_parameters_b"]
            or row["canonical_release_date"] < "2022-01-01"
            or row["epoch_base_model"]
        ):
            continue
        try:
            source_record = json.loads(row["source_record_json"])
        except json.JSONDecodeError:
            continue
        if "Language" not in str(source_record.get("Domain", "")):
            continue
        seen.add(checkpoint)
        organization = row["source_organization"] or row["source_provider"] or "unknown"
        compute.append(
            apply_parameter_truth({
                "release_date": row["canonical_release_date"],
                "model": row["canonical_display_name"],
                "total_b": float(row["total_parameters_b"]),
                "family": re.sub(r"[^a-z0-9]+", "_", organization.lower()).strip("_"),
                "log_compute": math.log10(float(row["epoch_training_compute_flop"])),
                "moe": _architecture_moe(row["architecture"]),
                "reasoning": _optional_binary(row, "reasoning"),
                "estimated": 0,
            })
        )

    panels = {"AA": aa, "ECI": eci, "No-CoT": horizon, "Compute": compute}
    inventory = {
        panel: {
            "rows": len(rows),
            "families": len({row["family"] for row in rows}),
            "min_date": min(row["release_date"] for row in rows),
            "max_date": max(row["release_date"] for row in rows),
        }
        for panel, rows in panels.items()
    }
    return panels, inventory


def _k3_aa_external_holdout_prediction(
    aa_rows: list[dict[str, Any]],
    *,
    target_family: str,
    exclude_all_kimi_lineages: bool,
) -> dict[str, Any]:
    """Build K3's leakage-controlled AA prediction outside the AA target panel.

    K3 is intentionally absent from ``regression_results.json/open_models`` so
    that it remains an external calibration check.  The ordinary ensemble used
    to mistake that absence for a missing AA measurement and evaluated K3 from
    only ECI plus speculative compute.  This helper makes the independently
    held-out AA prediction explicit without adding K3's disclosed parameter
    count to the fit.

    ``target_family`` is a lineage label in the core backtest and a canonical
    developer label in the developer-holdout sensitivity.  The latter has
    already replaced every row's family with its developer.
    """

    signal = AA_TARGET_SIGNALS["Kimi K3"]
    target = {
        "release_date": signal["release_date"],
        "aa_slug": signal["slug"],
        "model": "Kimi K3",
        "total_b": K3_TOTAL_B,
        "score": signal["score"],
        "family": target_family,
        "moe": 1,
        "reasoning": 1,
        "estimated": 0,
    }
    information_date = aa_prediction_information_date(target)
    prior = [
        row
        for row in aa_rows
        if parameter_training_eligibility_date(row) < information_date
    ]
    if exclude_all_kimi_lineages:
        train = [
            row for row in prior if "kimi" not in str(row["family"]).lower()
        ]
    else:
        train = [row for row in prior if row["family"] != target_family]
    if len(train) < 16 or len({row["family"] for row in train}) < 6:
        raise ValueError("Insufficient leakage-controlled AA training data for K3")

    prediction_log10, coefficients = _fit_predict(train, target, ("score", "date"))
    actual_log10 = math.log10(K3_TOTAL_B)
    error = prediction_log10 - actual_log10
    rank = sum(float(row["score"]) <= float(target["score"]) for row in prior) / len(prior)
    return {
        "panel": "AA",
        "split": "chronological_family_holdout",
        "spec": "score_date_external_k3_holdout",
        "release_date": target["release_date"],
        "prediction_information_date": information_date,
        "model": target["model"],
        "family": target_family,
        "moe": 1,
        "reasoning": 1,
        "actual_b": K3_TOTAL_B,
        "predicted_b": float(10**prediction_log10),
        "log10_error": float(error),
        "multiplicative_error": float(10 ** abs(error)),
        "signed_prediction_ratio": float(10**error),
        "train_n": len(train),
        "train_family_n": len({row["family"] for row in train}),
        "train_max_date": max(
            parameter_training_eligibility_date(row) for row in train
        ),
        "test_family_excluded": not any(
            row["family"] == target_family for row in train
        ),
        "frontier_signal_rank": float(rank),
        "coefficients": coefficients,
        "external_target_not_in_aa_parameter_panel": True,
        "exclusion_policy": (
            "all Kimi lineages held out"
            if exclude_all_kimi_lineages
            else f"entire developer held out: {target_family}"
        ),
    }


def _current_ensemble(
    selected: dict[str, list[dict[str, Any]]],
    equal_weight: bool,
    parameter_registry: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    by_panel: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for panel, predictions in selected.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            grouped[_normal_model_name(row["model"])].append(row)
        by_panel[panel] = grouped

    keys = set().union(*(set(grouped) for grouped in by_panel.values()))
    output = []
    for key in sorted(keys):
        components = []
        for panel, grouped in by_panel.items():
            candidates = grouped.get(key, [])
            if len(candidates) == 1:
                components.append((panel, candidates[0]))
        if len(components) < 2:
            continue
        actuals = [row["actual_b"] for _, row in components]
        dates = [date.fromisoformat(row["release_date"]) for _, row in components]
        if (
            max(actuals) / min(actuals) >= PARAMETER_TRUTH_MATCH_RATIO
            or (max(dates) - min(dates)).days
            > PARAMETER_TRUTH_DATE_TOLERANCE_DAYS
        ):
            continue
        parameter_truth = _resolve_parameter_truth(key, components, parameter_registry)
        if parameter_truth is None:
            continue
        weighted_logs = []
        weights = []
        for panel, row in components:
            weight = 1.0 if equal_weight else CURRENT_WEIGHTS[panel]
            weighted_logs.append(weight * math.log10(row["predicted_b"]))
            weights.append(weight)
        prediction_log10 = sum(weighted_logs) / sum(weights)
        actual_b = float(parameter_truth["actual_b"])
        error = prediction_log10 - math.log10(actual_b)
        moe_flags = {
            row["moe"] for _, row in components if row.get("moe") is not None
        }
        reasoning_flags = {
            row["reasoning"]
            for _, row in components
            if row.get("reasoning") is not None
        }
        moe = next(iter(moe_flags)) if len(moe_flags) == 1 else None
        reasoning = next(iter(reasoning_flags)) if len(reasoning_flags) == 1 else None
        output.append(
            {
                "panel": "Available-components ensemble",
                "split": "chronological_family_holdout",
                "spec": "equal_available_weights" if equal_weight else "current_available_weights",
                "release_date": max(row["release_date"] for _, row in components),
                "prediction_information_date": max(
                    row.get("prediction_information_date") or row["release_date"]
                    for _, row in components
                ),
                "model": components[0][1]["model"],
                "normalized_model": key,
                "family": components[0][1]["family"],
                "moe": moe,
                "reasoning": reasoning,
                "moe_flag_conflict": len(moe_flags) > 1,
                "reasoning_flag_conflict": len(reasoning_flags) > 1,
                "actual_b": actual_b,
                "actual_canonical_checkpoint_id": parameter_truth[
                    "canonical_checkpoint_id"
                ],
                "actual_parameter_source_family": parameter_truth["source"],
                "actual_parameter_source": parameter_truth["parameter_value_source"],
                "actual_parameter_source_url": parameter_truth["source_url"],
                "component_actuals_b": {
                    panel: float(row["actual_b"]) for panel, row in components
                },
                "component_actual_max_over_min": float(max(actuals) / min(actuals)),
                "predicted_b": float(10**prediction_log10),
                "log10_error": float(error),
                "multiplicative_error": float(10 ** abs(error)),
                "signed_prediction_ratio": float(10**error),
                "component_count": len(components),
                "components": [
                    {
                        "panel": panel,
                        "actual_b": float(row["actual_b"]),
                        "predicted_b": row["predicted_b"],
                        "prediction_information_date": row.get(
                            "prediction_information_date"
                        )
                        or row["release_date"],
                        "weight": 1.0 if equal_weight else CURRENT_WEIGHTS[panel],
                    }
                    for panel, row in components
                ],
                "train_n": min(row["train_n"] for _, row in components),
                "train_family_n": min(row["train_family_n"] for _, row in components),
                "train_max_date": max(row["train_max_date"] for _, row in components),
                "test_family_excluded": all(row["test_family_excluded"] for _, row in components),
                "frontier_signal_rank": float(sum(row["frontier_signal_rank"] for _, row in components) / len(components)),
                "coefficients": [],
            }
        )
    return output


def _external_checks(panels: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    k3 = {
        "release_date": "2026-07-16",
        "model": "Kimi K3",
        "total_b": K3_TOTAL_B,
        "score": 57.1123394372091,
        "family": "kimi_k3",
        "estimated": 0,
    }
    checks = []
    for label, remove_kimi in (("AA expanding fit", False), ("AA expanding fit; all Kimi held out", True)):
        if remove_kimi:
            heldout = _k3_aa_external_holdout_prediction(
                panels["AA"],
                target_family="kimi_k2",
                exclude_all_kimi_lineages=True,
            )
            train_n = heldout["train_n"]
            coefficients = heldout["coefficients"]
            prediction_b = heldout["predicted_b"]
        else:
            train = [
                row
                for row in panels["AA"]
                if parameter_training_eligibility_date(row) < k3["release_date"]
            ]
            prediction_log10, coefficients = _fit_predict(
                train, k3, ("score", "date")
            )
            train_n = len(train)
            prediction_b = 10**prediction_log10
        checks.append(
            {
                "anchor": "Kimi K3",
                "actual_b": K3_TOTAL_B,
                "method": label,
                "predicted_b": prediction_b,
                "multiplicative_error": max(
                    prediction_b / K3_TOTAL_B, K3_TOTAL_B / prediction_b
                ),
                "train_n": train_n,
                "coefficients": coefficients,
                "note": "Pre-calibration external check; current AA snapshot, exact release date.",
                "actual_parameter_source": K3_PARAMETER_SOURCE,
            }
        )

    regression = json.loads(REGRESSION_PATH.read_text(encoding="utf-8"))
    grok_input = next(
        row for row in regression["frontier_predictions"] if row["model"] == "Grok 4.5"
    )
    grok_target = {
        "release_date": grok_input["release_date"],
        "model": "Grok 4.5",
        "family": "grok45",
        "total_b": 1500.0,
        "estimated": 0,
    }
    aa_target = {**grok_target, "score": float(grok_input["aa_score"])}
    aa_train = [
        row
        for row in panels["AA"]
        if parameter_training_eligibility_date(row) < aa_target["release_date"]
        and row["family"] != aa_target["family"]
    ]
    aa_log10, aa_coefficients = _fit_predict(
        aa_train, aa_target, ("score", "date")
    )
    aa_prediction_b = float(10**aa_log10)

    eci_target = {**grok_target, "score": float(grok_input["eci"])}
    eci_train = [
        row
        for row in panels["ECI"]
        if parameter_training_eligibility_date(row) < eci_target["release_date"]
        and row["family"] != eci_target["family"]
    ]
    eci_score_log10, eci_score_coefficients = _fit_predict(
        eci_train, eci_target, ("score",)
    )
    eci_date_log10, eci_date_coefficients = _fit_predict(
        eci_train, eci_target, ("score", "date")
    )
    eci_prediction_log10 = 0.60 * eci_score_log10 + 0.40 * eci_date_log10
    eci_prediction_b = float(10**eci_prediction_log10)

    direct_grok_checks = (
        (
            "AA direct chronological panel fit",
            aa_prediction_b,
            len(aa_train),
            aa_coefficients,
            float(grok_input["aa_score"]),
        ),
        (
            "ECI direct chronological 60/40 panel fit",
            eci_prediction_b,
            len(eci_train),
            [*eci_score_coefficients, *eci_date_coefficients],
            float(grok_input["eci"]),
        ),
    )
    for method, prediction_b, train_n, coefficients, input_score in direct_grok_checks:
        checks.append(
            {
                "anchor": "Grok 4.5",
                "actual_b": 1500.0,
                "method": method,
                "predicted_b": prediction_b,
                "multiplicative_error": max(
                    prediction_b / 1500.0, 1500.0 / prediction_b
                ),
                "train_n": train_n,
                "train_max_date": max(
                    row["release_date"]
                    for row in (aa_train if method.startswith("AA") else eci_train)
                ),
                "input_score": input_score,
                "coefficients": coefficients,
                "note": (
                    "Directly refit from the pinned regression panel using only "
                    "strictly earlier releases. The disclosed 1.5T target is hidden "
                    "from fitting; weak branch performance is retained as a stress check."
                ),
                "actual_parameter_source": grok_input["classification"],
            }
        )

    aa_eci_b = math.sqrt(aa_prediction_b * eci_prediction_b)
    checks.append(
        {
            "anchor": "Grok 4.5",
            "actual_b": 1500.0,
            "method": "AA/ECI direct geometric ensemble before anchor lock",
            "predicted_b": aa_eci_b,
            "multiplicative_error": max(aa_eci_b / 1500.0, 1500.0 / aa_eci_b),
            "train_n": min(len(aa_train), len(eci_train)),
            "train_max_date": max(
                max(row["release_date"] for row in aa_train),
                max(row["release_date"] for row in eci_train),
            ),
            "coefficients": [],
            "note": (
                "Geometric pool of the two independently refit direct panel checks. "
                "No site, compute, price, horizon, or locked-anchor value enters the prediction."
            ),
            "actual_parameter_source": grok_input["classification"],
        }
    )
    return checks


def main() -> None:
    label_timing_ledger = load_parameter_label_availability()
    score_timing_ledger = load_aa_score_availability()
    label_timing_evidence_paths = tuple(
        WORK_DIR / evidence["path"]
        for record in label_timing_ledger["records"]
        for evidence in record["local_evidence"]
    )
    panels, inventory = _load_panels()
    parameter_registry = _load_parameter_truth_registry()
    panel_specs = {
        "AA": {
            "rows": panels["AA"],
            "min_train": 16,
            "min_families": 6,
            "signal": "score",
            "specs": {
                "geometric_prior": (),
                "score_only": ("score",),
                "score_date": ("score", "date"),
                "score_date_moe": ("score", "date", "moe"),
                "score_date_moe_reasoning": ("score", "date", "moe", "reasoning"),
            },
            "current": "score_date",
        },
        "ECI": {
            "rows": panels["ECI"],
            "min_train": 20,
            "min_families": 6,
            "signal": "score",
            "specs": {
                "geometric_prior": (),
                "score_only": ("score",),
                "score_date": ("score", "date"),
                "score_date_moe": ("score", "date", "moe"),
                "score_date_moe_reasoning": ("score", "date", "moe", "reasoning"),
            },
            "current": "blend_60_score_40_score_date",
        },
        "No-CoT": {
            "rows": panels["No-CoT"],
            "min_train": 12,
            "min_families": 3,
            "signal": "log_horizon",
            "specs": {
                "geometric_prior": (),
                "log_horizon_only": ("log_horizon",),
                "log_horizon_date": ("log_horizon", "date"),
                "log_horizon_date_moe": ("log_horizon", "date", "moe"),
            },
            "current": "log_horizon_date_moe",
        },
        "Compute": {
            "rows": panels["Compute"],
            "min_train": 100,
            "min_families": 20,
            "signal": "log_compute",
            "specs": {
                "geometric_prior": (),
                "log_compute_only": ("log_compute",),
                "log_compute_date": ("log_compute", "date"),
            },
            "current": "log_compute_date",
        },
    }

    all_predictions: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    comparisons: list[dict[str, Any]] = []
    selected_strict: dict[str, list[dict[str, Any]]] = {}

    for panel, definition in panel_specs.items():
        for exclude_family in (False, True):
            split = "chronological_family_holdout" if exclude_family else "chronological"
            for spec, features in definition["specs"].items():
                predictions = _backtest(
                    panel,
                    definition["rows"],
                    spec,
                    features,
                    definition["min_train"],
                    definition["min_families"],
                    exclude_family,
                    definition["signal"],
                )
                all_predictions[(panel, split, spec)] = predictions
                comparisons.append(
                    {
                        "panel": panel,
                        "split": split,
                        "spec": spec,
                        "is_current_like": spec == definition["current"],
                        **_metric_summary(predictions),
                    }
                )

            if panel == "ECI":
                blend = _blend_predictions(
                    panel,
                    "blend_60_score_40_score_date",
                    all_predictions[(panel, split, "score_only")],
                    all_predictions[(panel, split, "score_date")],
                    0.60,
                )
                all_predictions[(panel, split, "blend_60_score_40_score_date")] = blend
                comparisons.append(
                    {
                        "panel": panel,
                        "split": split,
                        "spec": "blend_60_score_40_score_date",
                        "is_current_like": True,
                        **_metric_summary(blend),
                    }
                )

        selected_strict[panel] = all_predictions[
            (panel, "chronological_family_holdout", definition["current"])
        ]

    k3_external_aa = _k3_aa_external_holdout_prediction(
        panels["AA"],
        target_family="kimi_k2",
        exclude_all_kimi_lineages=True,
    )
    ensemble_selected = {
        **selected_strict,
        "AA": [*selected_strict["AA"], k3_external_aa],
    }
    current_ensemble = _current_ensemble(
        ensemble_selected, equal_weight=False, parameter_registry=parameter_registry
    )
    equal_ensemble = _current_ensemble(
        ensemble_selected, equal_weight=True, parameter_registry=parameter_registry
    )
    for predictions in (current_ensemble, equal_ensemble):
        comparisons.append(
            {
                "panel": "Available-components ensemble",
                "split": "chronological_family_holdout",
                "spec": predictions[0]["spec"] if predictions else "unavailable",
                "is_current_like": predictions is current_ensemble,
                **_metric_summary(predictions),
            }
        )

    current_metrics = {
        panel: _metric_summary(predictions) for panel, predictions in selected_strict.items()
    }
    current_metrics["Available-components ensemble"] = _metric_summary(current_ensemble)
    frontier_like_metrics = {
        panel: _metric_summary(
            [row for row in predictions if row["frontier_signal_rank"] >= 0.90]
        )
        for panel, predictions in selected_strict.items()
    }
    frontier_like_metrics["Available-components ensemble"] = _metric_summary(
        [row for row in current_ensemble if row["frontier_signal_rank"] >= 0.90]
    )

    detailed_predictions = [
        row for panel in ("AA", "ECI", "No-CoT", "Compute") for row in selected_strict[panel]
    ]
    detailed_predictions.append(k3_external_aa)
    detailed_predictions.extend(current_ensemble)
    external_checks = _external_checks(panels)

    result = {
        "metadata": {
            "generated_on": "2026-07-31",
            "target": "log10 disclosed total parameters in billions",
            "date_rule": "For AA, max(training release date, public parameter-label date, verified score-publication date) must be strictly earlier than the test score-publication date; release date remains the date feature. Other panels retain strict release-date ordering.",
            "preferred_split": "chronological_family_holdout",
            "family_weighting": "inverse checkpoint count per training family; AA estimated scores and broad ECI CIs receive 0.5 weight",
            "benchmark_vintage_caveat": "Pseudo-chronological: current benchmark snapshots are used, so the exercise validates cross-model mapping and temporal extrapolation, not a fully vintage real-time forecast.",
            "selection_caveat": "Specifications are compared transparently. Current-like specifications were designated from the live pipeline before reading this backtest.",
            "crowd_caveat": "The human crowd cannot yet be backtested because Fable and Sol remain undisclosed and the forecasts were collected after benchmark outcomes were visible.",
            "parameter_truth_policy": (
                "Ensemble targets resolve once against the canonical megafile with "
                "source-priority provenance and primary-source overrides; correlated "
                "component copies are reconciled but never averaged."
            ),
            "parameter_label_timing_records": len(label_timing_ledger["records"]),
            "parameter_label_timing_policy": "Checkpoint release, public parameter-label availability, score publication, and weight availability are stored separately; release, label, and verified score dates determine AA training eligibility, while weight availability is only provenance.",
            "aa_score_timing_records": len(score_timing_ledger["records"]),
            "aa_score_timing_policy": "Exact AA slug events with a non-null Intelligence Index use their first changelog publication date. Unmatched checkpoints retain nominal release ordering and are explicitly not claimed as historically reconstructed scores.",
            "k3_external_aa_ensemble_policy": (
                "K3 remains excluded from the AA parameter target panel. Its exact-score, "
                "all-Kimi-held-out AA prediction is nevertheless included as an ensemble "
                "component so the external target is not evaluated from ECI and speculative "
                "compute alone. The disclosed 2.78T count never enters the AA fit."
            ),
        },
        "inventory": inventory,
        "current_like_metrics": current_metrics,
        "frontier_like_metrics": frontier_like_metrics,
        "model_comparisons": comparisons,
        "predictions": detailed_predictions,
        "ensemble_predictions": current_ensemble,
        "equal_weight_ensemble_predictions": equal_ensemble,
        "external_checks": external_checks,
        "interpretation": {
            "headline": "The regressions carry real signal versus an unconditional prior, but literal parameter recovery remains only factor-of-roughly-two-to-three accurate in strict family-held-out chronological tests.",
            "date_finding": "Adding release date does not consistently improve AA or ECI out-of-sample accuracy; the date law should remain model-averaged or shrunk, not treated as known.",
            "horizon_finding": "No-CoT horizon improves strongly over the unconditional prior, but its strict median error is still about 2.5x and architecture matters; 50% weight is too strong if interpreted as an independent precise measurement.",
            "ensemble_finding": (
                "The available-component ensemble improves median error to about "
                f"{frontier_like_metrics['Available-components ensemble']['median_multiplicative_error']:.1f}x "
                f"on {frontier_like_metrics['Available-components ensemble']['n']} frontier-like, conservatively matched checkpoints, "
                f"but its 80th-percentile error is {frontier_like_metrics['Available-components ensemble']['p80_multiplicative_error']:.1f}x."
            ),
            "external_finding": "K3 and Grok are encouraging external scale checks, but two anchors are not enough to establish calibration of the full frontier posterior.",
        },
        "source_files": {
            portable_path(REGRESSION_PATH): _sha256(REGRESSION_PATH),
            portable_path(UNIFIED_PATH): _sha256(UNIFIED_PATH),
            portable_path(K3_EVIDENCE_PATH): _sha256(K3_EVIDENCE_PATH),
            portable_path(AA_PARAMETER_LABEL_AVAILABILITY_PATH): _sha256(
                AA_PARAMETER_LABEL_AVAILABILITY_PATH
            ),
            portable_path(AA_SCORE_AVAILABILITY_PATH): _sha256(
                AA_SCORE_AVAILABILITY_PATH
            ),
            portable_path(AA_CHANGELOG_PATH): _sha256(AA_CHANGELOG_PATH),
            portable_path(OPEN_MODEL_PARAMETER_TRUTH_PATH): _sha256(
                OPEN_MODEL_PARAMETER_TRUTH_PATH
            ),
            portable_path(AA_DETAILED_PATH): _sha256(AA_DETAILED_PATH),
            **{
                portable_path(path): _sha256(path)
                for path in label_timing_evidence_paths
            },
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    prediction_fields = [
        "panel", "split", "spec", "release_date", "prediction_information_date", "model", "family", "actual_b", "predicted_b",
        "log10_error", "multiplicative_error", "signed_prediction_ratio", "train_n", "train_family_n",
        "train_max_date", "test_family_excluded", "frontier_signal_rank", "component_count",
    ]
    with PREDICTION_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(detailed_predictions)

    comparison_fields = [
        "panel", "split", "spec", "is_current_like", "n", "median_multiplicative_error",
        "geomean_multiplicative_error", "rmse_log10", "signed_bias_factor", "within_1_5x",
        "within_2x", "within_3x", "p80_multiplicative_error", "p90_multiplicative_error",
    ]
    with COMPARISON_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(comparisons)

    print(
        json.dumps(
            {
                "result": portable_path(RESULT_PATH),
                "prediction_csv": portable_path(PREDICTION_CSV_PATH),
                "comparison_csv": portable_path(COMPARISON_CSV_PATH),
                "inventory": inventory,
                "current_like_metrics": current_metrics,
                "frontier_like_metrics": frontier_like_metrics,
                "external_checks": external_checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
