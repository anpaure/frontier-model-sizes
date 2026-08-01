#!/usr/bin/env python3
"""Leakage-controlled audit of an architecture-aware aggregate-ECI blend.

This file is deliberately isolated from the live forecast pipeline.  It compares
the current direct aggregate-ECI mapping

    60% score-only + 40% score/date

with a fixed challenger

    60% score-only + 40% score/date/MoE/reasoning.

Both mixtures are geometric (linear in log10 total parameters).  Every source
prediction uses only parameter labels whose public-eligibility date is strictly
earlier than the test checkpoint's information date, and removes the whole
target developer.  A genuinely nested evaluation then decides between the two
fixed specifications using only earlier outer-fold residuals, again excluding
the target developer.

The audit never writes regression_results.json or any live-pipeline artifact.
Even a favorable retrospective result remains a zero-live-weight challenger
until all predeclared promotion gates, including target-architecture and
prospective-validation gates, pass.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from aa_calibration_overrides import parameter_training_eligibility_date
from aa_parameter_label_availability import LEDGER_PATH as PARAMETER_LABEL_LEDGER
from analyze_parameter_vintage_sensitivity import developer_lookup, with_developers
from k3_primary_evidence import (
    K3_ACTIVE_B,
    K3_EVIDENCE_PATH,
    K3_PARAMETER_SOURCE,
    K3_TOTAL_B,
)
from run_parameter_backtest import (
    OUTPUT_DIR,
    REGRESSION_PATH,
    UNIFIED_PATH,
    _backtest,
    _blend_predictions,
    _fit_predict,
    _load_panels,
)


ROOT = Path(__file__).resolve().parent
RESULT = OUTPUT_DIR / "eci_architecture_blend_challenger_2026-07-31.json"
PREDICTIONS = (
    OUTPUT_DIR / "eci_architecture_blend_challenger_predictions_2026-07-31.csv"
)
REPORT = ROOT / "ECI_ARCHITECTURE_BLEND_CHALLENGER.md"

GENERATED_ON = "2026-07-31"
TARGET = "log10 disclosed total parameters in billions"

SCORE_ONLY_WEIGHT = 0.60
ARCHITECTURE_COMPONENT_WEIGHT = 0.40
MIN_OUTER_TRAIN_ROWS = 20
MIN_OUTER_TRAIN_DEVELOPERS = 6
MIN_META_TRAIN_ROWS = 12
MIN_META_TRAIN_DEVELOPERS = 6
FRONTIER_RANK_THRESHOLD = 0.90
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED_FIXED_ALL = 20_260_731
BOOTSTRAP_SEED_FIXED_FRONTIER = 20_260_732
BOOTSTRAP_SEED_NESTED_ALL = 20_260_733
BOOTSTRAP_SEED_NESTED_FRONTIER = 20_260_734
MAX_ANCHOR_ERROR_RELATIVE_TO_BASELINE = 1.10

MIN_FIXED_ALL_ROWS = 60
MIN_FIXED_ALL_DEVELOPERS = 12
MIN_FIXED_FRONTIER_ROWS = 30
MIN_FIXED_FRONTIER_DEVELOPERS = 8
MIN_NESTED_ALL_ROWS = 50
MIN_NESTED_ALL_DEVELOPERS = 10
MIN_NESTED_FRONTIER_ROWS = 30
MIN_NESTED_FRONTIER_DEVELOPERS = 8
MIN_PROSPECTIVE_DISCLOSURES = 3
MIN_PROSPECTIVE_FRONTIER_DISCLOSURES = 2

LIVE_TARGET_MODELS = ("Claude Opus 5", "Claude Fable 5", "GPT-5.6 Sol")
GROK_MODEL = "Grok 4.5"
GROK_TOTAL_B = 1500.0
GROK_WORKING_ASSUMPTION = (1, 1)

SOURCE_PATHS = (
    REGRESSION_PATH,
    UNIFIED_PATH,
    PARAMETER_LABEL_LEDGER,
    K3_EVIDENCE_PATH,
    ROOT / "run_parameter_backtest.py",
    ROOT / "analyze_parameter_vintage_sensitivity.py",
    ROOT / "aa_calibration_overrides.py",
    ROOT / "aa_parameter_label_availability.py",
    ROOT / "k3_primary_evidence.py",
)

PREDICTION_FIELDS = (
    "release_date",
    "prediction_information_date",
    "parameter_training_eligibility_date",
    "model",
    "lineage_family",
    "developer",
    "eci_score",
    "moe",
    "reasoning",
    "actual_b",
    "frontier_signal_rank",
    "source_train_n",
    "source_train_developers",
    "source_train_max_eligibility_date",
    "source_test_developer_excluded",
    "score_only_predicted_b",
    "score_date_predicted_b",
    "architecture_component_predicted_b",
    "baseline_predicted_b",
    "baseline_log10_error",
    "baseline_multiplicative_error",
    "challenger_predicted_b",
    "challenger_log10_error",
    "challenger_multiplicative_error",
    "fixed_absolute_log10_delta",
    "nested_eligible",
    "nested_meta_train_n",
    "nested_meta_train_developers",
    "nested_meta_train_max_eligibility_date",
    "nested_baseline_equal_developer_mae",
    "nested_challenger_equal_developer_mae",
    "nested_selected_specification",
    "nested_selected_predicted_b",
    "nested_selected_log10_error",
    "nested_selected_multiplicative_error",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _source_hashes() -> dict[str, str]:
    missing = [path for path in SOURCE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing audit input(s): {missing}")
    return {_relative(path): sha256(path) for path in SOURCE_PATHS}


def _prediction_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["release_date"]), str(row["model"])


def _unique_map(
    rows: Iterable[dict[str, Any]], label: str
) -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _prediction_key(row)
        if key in output:
            raise ValueError(f"Duplicate {label} prediction key: {key}")
        output[key] = row
    return output


def metric_summary(
    rows: list[dict[str, Any]], error_field: str
) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "developers": 0}
    errors = np.asarray([float(row[error_field]) for row in rows], dtype=float)
    if not np.isfinite(errors).all():
        raise ValueError(f"Non-finite values in {error_field}")
    absolute = np.abs(errors)
    return {
        "n": len(rows),
        "developers": len({str(row["developer"]) for row in rows}),
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "geomean_multiplicative_error": float(10 ** np.mean(absolute)),
        "mean_absolute_log10_error": float(np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(errors**2))),
        "p80_multiplicative_error": float(10 ** np.quantile(absolute, 0.80)),
        "p90_multiplicative_error": float(10 ** np.quantile(absolute, 0.90)),
        "within_2x": float(np.mean(absolute <= math.log10(2.0))),
        "signed_bias_factor": float(10 ** np.mean(errors)),
    }


def equal_developer_mae(
    rows: list[dict[str, Any]], error_field: str
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["developer"])].append(abs(float(row[error_field])))
    if not grouped:
        raise ValueError("Cannot compute equal-developer MAE on an empty cohort")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def paired_developer_bootstrap(
    rows: list[dict[str, Any]],
    candidate_error_field: str,
    *,
    seed: int,
) -> dict[str, Any]:
    """Resample whole developers and compare per-developer absolute error."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["developer"])].append(
            abs(float(row[candidate_error_field]))
            - abs(float(row["baseline_log10_error"]))
        )
    developers = sorted(grouped)
    if not developers:
        raise ValueError("Cannot bootstrap an empty cohort")
    effects = np.asarray(
        [np.mean(grouped[developer]) for developer in developers], dtype=float
    )
    rng = np.random.default_rng(seed)
    draws = effects[
        rng.integers(0, len(effects), size=(BOOTSTRAP_SAMPLES, len(effects)))
    ].mean(axis=1)
    return {
        "metric": (
            "equal-developer mean absolute log10 error; challenger minus "
            "current direct aggregate-ECI baseline"
        ),
        "observed_delta": float(np.mean(effects)),
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "probability_challenger_better": float(np.mean(draws < 0)),
        "developers": len(developers),
        "samples": BOOTSTRAP_SAMPLES,
        "random_seed": seed,
    }


def _build_outer_predictions() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    panels, inventory = _load_panels()
    converted = with_developers(
        {"ECI": panels["ECI"]}, developer_lookup()
    )["ECI"]
    if len(converted) != len(panels["ECI"]):
        raise ValueError("Developer mapping changed ECI row count")
    if any(row.get("moe") is None or row.get("reasoning") is None for row in converted):
        raise ValueError("Open-model ECI panel has an unclassified architecture flag")

    score_only = _backtest(
        "ECI",
        converted,
        "score_only",
        ("score",),
        MIN_OUTER_TRAIN_ROWS,
        MIN_OUTER_TRAIN_DEVELOPERS,
        True,
        "score",
    )
    score_date = _backtest(
        "ECI",
        converted,
        "score_date",
        ("score", "date"),
        MIN_OUTER_TRAIN_ROWS,
        MIN_OUTER_TRAIN_DEVELOPERS,
        True,
        "score",
    )
    architecture = _backtest(
        "ECI",
        converted,
        "score_date_moe_reasoning",
        ("score", "date", "moe", "reasoning"),
        MIN_OUTER_TRAIN_ROWS,
        MIN_OUTER_TRAIN_DEVELOPERS,
        True,
        "score",
    )
    baseline = _blend_predictions(
        "ECI",
        "blend_60_score_40_score_date",
        score_only,
        score_date,
        SCORE_ONLY_WEIGHT,
    )
    challenger = _blend_predictions(
        "ECI",
        "blend_60_score_40_score_date_moe_reasoning",
        score_only,
        architecture,
        SCORE_ONLY_WEIGHT,
    )

    maps = {
        "source": _unique_map(converted, "source"),
        "score_only": _unique_map(score_only, "score-only"),
        "score_date": _unique_map(score_date, "score/date"),
        "architecture": _unique_map(architecture, "architecture"),
        "baseline": _unique_map(baseline, "baseline"),
        "challenger": _unique_map(challenger, "challenger"),
    }
    paired_keys = sorted(maps["baseline"].keys() & maps["challenger"].keys())
    if not paired_keys:
        raise ValueError("No paired ECI predictions")
    if any(set(mapping) != set(maps["baseline"]) for mapping in maps.values() if mapping is not maps["source"]):
        raise ValueError("Component ECI folds do not have identical prediction coverage")

    output: list[dict[str, Any]] = []
    for key in paired_keys:
        source = maps["source"][key]
        score = maps["score_only"][key]
        dated = maps["score_date"][key]
        arch = maps["architecture"][key]
        base = maps["baseline"][key]
        challenge = maps["challenger"][key]
        information_date = str(base["prediction_information_date"])
        eligibility_date = parameter_training_eligibility_date(source)
        component_rows = (score, dated, arch, base, challenge)
        if any(not row["test_family_excluded"] for row in component_rows):
            raise ValueError(f"Target developer leaked into source fold for {key}")
        if any(str(row["train_max_date"]) >= information_date for row in component_rows):
            raise ValueError(f"Non-chronological source fold for {key}")
        if any(not math.isclose(float(row["actual_b"]), float(source["total_b"]), abs_tol=1e-12) for row in component_rows):
            raise ValueError(f"Target parameter mismatch for {key}")
        expected_baseline = (
            SCORE_ONLY_WEIGHT * math.log10(float(score["predicted_b"]))
            + ARCHITECTURE_COMPONENT_WEIGHT
            * math.log10(float(dated["predicted_b"]))
        )
        expected_challenger = (
            SCORE_ONLY_WEIGHT * math.log10(float(score["predicted_b"]))
            + ARCHITECTURE_COMPONENT_WEIGHT
            * math.log10(float(arch["predicted_b"]))
        )
        if not math.isclose(
            math.log10(float(base["predicted_b"])), expected_baseline, abs_tol=1e-12
        ):
            raise ValueError(f"Baseline mixture arithmetic mismatch for {key}")
        if not math.isclose(
            math.log10(float(challenge["predicted_b"])),
            expected_challenger,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Challenger mixture arithmetic mismatch for {key}")

        baseline_error = float(base["log10_error"])
        challenger_error = float(challenge["log10_error"])
        output.append(
            {
                "release_date": str(source["release_date"]),
                # Epoch does not publish a historical aggregate-score availability
                # ledger.  The release day is therefore the explicitly declared
                # information-date proxy, and the audit remains pseudo-chronological.
                "prediction_information_date": information_date,
                "parameter_training_eligibility_date": eligibility_date,
                "model": str(source["model"]),
                "lineage_family": str(source["lineage_family"]),
                "developer": str(source["developer"]),
                "eci_score": float(source["score"]),
                "moe": int(source["moe"]),
                "reasoning": int(source["reasoning"]),
                "actual_b": float(source["total_b"]),
                "frontier_signal_rank": float(base["frontier_signal_rank"]),
                "source_train_n": min(int(row["train_n"]) for row in component_rows),
                "source_train_developers": min(
                    int(row["train_family_n"]) for row in component_rows
                ),
                "source_train_max_eligibility_date": max(
                    str(row["train_max_date"]) for row in component_rows
                ),
                "source_test_developer_excluded": True,
                "score_only_predicted_b": float(score["predicted_b"]),
                "score_date_predicted_b": float(dated["predicted_b"]),
                "architecture_component_predicted_b": float(arch["predicted_b"]),
                "baseline_predicted_b": float(base["predicted_b"]),
                "baseline_log10_error": baseline_error,
                "baseline_multiplicative_error": float(10 ** abs(baseline_error)),
                "challenger_predicted_b": float(challenge["predicted_b"]),
                "challenger_log10_error": challenger_error,
                "challenger_multiplicative_error": float(
                    10 ** abs(challenger_error)
                ),
                "fixed_absolute_log10_delta": float(
                    abs(challenger_error) - abs(baseline_error)
                ),
                "nested_eligible": False,
                "nested_meta_train_n": "",
                "nested_meta_train_developers": "",
                "nested_meta_train_max_eligibility_date": "",
                "nested_baseline_equal_developer_mae": "",
                "nested_challenger_equal_developer_mae": "",
                "nested_selected_specification": "",
                "nested_selected_predicted_b": "",
                "nested_selected_log10_error": "",
                "nested_selected_multiplicative_error": "",
            }
        )

    developer_inventory = {
        "source_eci_rows": len(converted),
        "source_lineage_families": len(
            {str(row["lineage_family"]) for row in converted}
        ),
        "source_developers": len({str(row["developer"]) for row in converted}),
        "outer_paired_rows": len(output),
        "outer_paired_developers": len({row["developer"] for row in output}),
        "source_inventory": inventory["ECI"],
    }
    return output, developer_inventory


def _add_nested_selection(rows: list[dict[str, Any]]) -> None:
    """Choose baseline/challenger using earlier outer errors only."""

    for test in rows:
        prior = [
            row
            for row in rows
            if row["parameter_training_eligibility_date"]
            < test["prediction_information_date"]
            and row["developer"] != test["developer"]
        ]
        developers = {str(row["developer"]) for row in prior}
        if (
            len(prior) < MIN_META_TRAIN_ROWS
            or len(developers) < MIN_META_TRAIN_DEVELOPERS
        ):
            continue
        baseline_mae = equal_developer_mae(prior, "baseline_log10_error")
        challenger_mae = equal_developer_mae(prior, "challenger_log10_error")
        use_challenger = challenger_mae < baseline_mae
        selected_prefix = "challenger" if use_challenger else "baseline"
        selected_error = float(test[f"{selected_prefix}_log10_error"])
        test.update(
            {
                "nested_eligible": True,
                "nested_meta_train_n": len(prior),
                "nested_meta_train_developers": len(developers),
                "nested_meta_train_max_eligibility_date": max(
                    str(row["parameter_training_eligibility_date"])
                    for row in prior
                ),
                "nested_baseline_equal_developer_mae": baseline_mae,
                "nested_challenger_equal_developer_mae": challenger_mae,
                "nested_selected_specification": selected_prefix,
                "nested_selected_predicted_b": float(
                    test[f"{selected_prefix}_predicted_b"]
                ),
                "nested_selected_log10_error": selected_error,
                "nested_selected_multiplicative_error": float(
                    10 ** abs(selected_error)
                ),
            }
        )


def _cohort_evaluation(
    rows: list[dict[str, Any]],
    candidate_error_field: str,
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        "baseline": metric_summary(rows, "baseline_log10_error"),
        "challenger": metric_summary(rows, candidate_error_field),
        "baseline_equal_developer_mae": equal_developer_mae(
            rows, "baseline_log10_error"
        ),
        "challenger_equal_developer_mae": equal_developer_mae(
            rows, candidate_error_field
        ),
        "paired_developer_bootstrap": paired_developer_bootstrap(
            rows, candidate_error_field, seed=seed
        ),
    }


def _fixed_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frontier = [
        row
        for row in rows
        if float(row["frontier_signal_rank"]) >= FRONTIER_RANK_THRESHOLD
    ]
    return {
        "all": _cohort_evaluation(
            rows,
            "challenger_log10_error",
            seed=BOOTSTRAP_SEED_FIXED_ALL,
        ),
        "frontier_like": _cohort_evaluation(
            frontier,
            "challenger_log10_error",
            seed=BOOTSTRAP_SEED_FIXED_FRONTIER,
        ),
    }


def _nested_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["nested_eligible"]]
    frontier = [
        row
        for row in eligible
        if float(row["frontier_signal_rank"]) >= FRONTIER_RANK_THRESHOLD
    ]
    selections: dict[str, int] = defaultdict(int)
    for row in eligible:
        selections[str(row["nested_selected_specification"])] += 1
    return {
        "selection_counts": dict(sorted(selections.items())),
        "selection_rule": (
            "At each outer target, choose the fixed specification with lower "
            "equal-developer MAE among eligible earlier outer-fold residuals; "
            "exclude the target developer again and break ties toward baseline."
        ),
        "all": _cohort_evaluation(
            eligible,
            "nested_selected_log10_error",
            seed=BOOTSTRAP_SEED_NESTED_ALL,
        ),
        "frontier_like": _cohort_evaluation(
            frontier,
            "nested_selected_log10_error",
            seed=BOOTSTRAP_SEED_NESTED_FRONTIER,
        ),
    }


def _fit_grok_anchor(eci_rows: list[dict[str, Any]]) -> dict[str, Any]:
    regression = json.loads(REGRESSION_PATH.read_text(encoding="utf-8"))
    raw = next(
        row for row in regression["frontier_predictions"] if row["model"] == GROK_MODEL
    )
    if "disclosed 1.5T total" not in str(raw.get("classification", "")):
        raise ValueError("Grok 4.5 disclosed-total provenance is no longer pinned")
    target = {
        "release_date": str(raw["release_date"]),
        "model": GROK_MODEL,
        "family": "xAI",
        "score": float(raw["eci"]),
        "total_b": GROK_TOTAL_B,
        "estimated": 0,
    }
    train = [
        row
        for row in eci_rows
        if parameter_training_eligibility_date(row) < target["release_date"]
        and row["developer"] != target["family"]
    ]
    if (
        len(train) < MIN_OUTER_TRAIN_ROWS
        or len({row["developer"] for row in train}) < MIN_OUTER_TRAIN_DEVELOPERS
    ):
        raise ValueError("Insufficient Grok anchor training support")
    score_log10, score_beta = _fit_predict(train, target, ("score",))
    dated_log10, dated_beta = _fit_predict(train, target, ("score", "date"))
    baseline_log10 = (
        SCORE_ONLY_WEIGHT * score_log10
        + ARCHITECTURE_COMPONENT_WEIGHT * dated_log10
    )
    baseline_b = float(10**baseline_log10)
    baseline_error = max(baseline_b / GROK_TOTAL_B, GROK_TOTAL_B / baseline_b)

    scenarios = []
    for moe in (0, 1):
        for reasoning in (0, 1):
            scenario_target = {**target, "moe": moe, "reasoning": reasoning}
            architecture_log10, coefficients = _fit_predict(
                train,
                scenario_target,
                ("score", "date", "moe", "reasoning"),
            )
            prediction_log10 = (
                SCORE_ONLY_WEIGHT * score_log10
                + ARCHITECTURE_COMPONENT_WEIGHT * architecture_log10
            )
            predicted_b = float(10**prediction_log10)
            error = max(
                predicted_b / GROK_TOTAL_B, GROK_TOTAL_B / predicted_b
            )
            scenarios.append(
                {
                    "moe": moe,
                    "reasoning": reasoning,
                    "architecture_observed": False,
                    "predicted_b": predicted_b,
                    "multiplicative_error": error,
                    "error_relative_to_baseline": float(error / baseline_error),
                    "architecture_coefficients": coefficients,
                }
            )
    working = next(
        row
        for row in scenarios
        if (row["moe"], row["reasoning"]) == GROK_WORKING_ASSUMPTION
    )
    return {
        "model": GROK_MODEL,
        "release_date": target["release_date"],
        "eci_score": target["score"],
        "disclosed_total_b": GROK_TOTAL_B,
        "actual_parameter_source": str(raw["classification"]),
        "target_developer": target["family"],
        "target_developer_excluded": not any(
            row["developer"] == target["family"] for row in train
        ),
        "train_n": len(train),
        "train_developers": len({row["developer"] for row in train}),
        "train_max_eligibility_date": max(
            parameter_training_eligibility_date(row) for row in train
        ),
        "score_only_coefficients": score_beta,
        "score_date_coefficients": dated_beta,
        "baseline_predicted_b": baseline_b,
        "baseline_multiplicative_error": baseline_error,
        "target_architecture_observed": False,
        "target_architecture_note": (
            "No independently observed MoE/reasoning classification is available; "
            "the challenger is reported as a four-scenario sensitivity."
        ),
        "scenario_predictions": scenarios,
        "working_assumption": {
            "moe": GROK_WORKING_ASSUMPTION[0],
            "reasoning": GROK_WORKING_ASSUMPTION[1],
            "not_an_observation": True,
            "predicted_b": working["predicted_b"],
            "multiplicative_error": working["multiplicative_error"],
            "error_relative_to_baseline": working["error_relative_to_baseline"],
        },
    }


def _anchor_checks(
    rows: list[dict[str, Any]], eci_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    k3 = next(row for row in rows if row["model"] == "Kimi K3")
    if not math.isclose(float(k3["actual_b"]), K3_TOTAL_B, abs_tol=1e-12):
        raise ValueError("K3 outer target is not using exact primary-source truth")
    k3_baseline_error = float(k3["baseline_multiplicative_error"])
    k3_challenger_error = float(k3["challenger_multiplicative_error"])
    return {
        "kimi_k3": {
            "model": "Kimi K3",
            "release_date": k3["release_date"],
            "disclosed_total_b": K3_TOTAL_B,
            "disclosed_active_b": K3_ACTIVE_B,
            "actual_parameter_source": K3_PARAMETER_SOURCE,
            "moe_observed": True,
            "reasoning_flag_used": int(k3["reasoning"]),
            "target_developer_excluded": k3["source_test_developer_excluded"],
            "train_max_eligibility_date": k3[
                "source_train_max_eligibility_date"
            ],
            "baseline_predicted_b": k3["baseline_predicted_b"],
            "baseline_multiplicative_error": k3_baseline_error,
            "challenger_predicted_b": k3["challenger_predicted_b"],
            "challenger_multiplicative_error": k3_challenger_error,
            "error_relative_to_baseline": float(
                k3_challenger_error / k3_baseline_error
            ),
            "status": (
                "retrospective current-score anchor; parameter truth is hidden "
                "from fitting, but this is not a frozen prospective forecast"
            ),
        },
        "grok_4_5": _fit_grok_anchor(eci_rows),
    }


def _live_target_applicability() -> list[dict[str, Any]]:
    regression = json.loads(REGRESSION_PATH.read_text(encoding="utf-8"))
    by_name = {row["model"]: row for row in regression["frontier_predictions"]}
    missing = [model for model in LIVE_TARGET_MODELS if model not in by_name]
    if missing:
        raise ValueError(f"Live target(s) absent from regression registry: {missing}")
    output = []
    for model in LIVE_TARGET_MODELS:
        row = by_name[model]
        moe = row.get("moe")
        reasoning = row.get("reasoning")
        output.append(
            {
                "model": model,
                "release_date": row["release_date"],
                "moe": moe,
                "reasoning": reasoning,
                "architecture_observed": moe is not None and reasoning is not None,
                "status": "target architecture unobserved",
            }
        )
    if any(row["architecture_observed"] for row in output):
        raise ValueError(
            "A live target architecture is now observed; review the frozen gate manually"
        )
    return output


def _metrics_non_worse(block: dict[str, Any]) -> bool:
    baseline = block["baseline"]
    challenger = block["challenger"]
    return bool(
        challenger["median_multiplicative_error"]
        <= baseline["median_multiplicative_error"]
        and challenger["mean_absolute_log10_error"]
        <= baseline["mean_absolute_log10_error"]
        and challenger["rmse_log10"] <= baseline["rmse_log10"]
        and challenger["p80_multiplicative_error"]
        <= baseline["p80_multiplicative_error"]
        and challenger["within_2x"] >= baseline["within_2x"]
    )


def _promotion_gates(
    fixed: dict[str, Any],
    nested: dict[str, Any],
    anchors: dict[str, Any],
    target_applicability: list[dict[str, Any]],
) -> dict[str, Any]:
    fixed_all = fixed["all"]
    fixed_frontier = fixed["frontier_like"]
    nested_all = nested["all"]
    nested_frontier = nested["frontier_like"]
    k3 = anchors["kimi_k3"]
    grok = anchors["grok_4_5"]
    observed = {
        "fixed_all_coverage": fixed_all["baseline"]["n"] >= MIN_FIXED_ALL_ROWS
        and fixed_all["baseline"]["developers"] >= MIN_FIXED_ALL_DEVELOPERS,
        "fixed_frontier_coverage": fixed_frontier["baseline"]["n"]
        >= MIN_FIXED_FRONTIER_ROWS
        and fixed_frontier["baseline"]["developers"]
        >= MIN_FIXED_FRONTIER_DEVELOPERS,
        "nested_all_coverage": nested_all["baseline"]["n"] >= MIN_NESTED_ALL_ROWS
        and nested_all["baseline"]["developers"] >= MIN_NESTED_ALL_DEVELOPERS,
        "nested_frontier_coverage": nested_frontier["baseline"]["n"]
        >= MIN_NESTED_FRONTIER_ROWS
        and nested_frontier["baseline"]["developers"]
        >= MIN_NESTED_FRONTIER_DEVELOPERS,
        "fixed_all_ci_wholly_favorable": fixed_all[
            "paired_developer_bootstrap"
        ]["ci_90"][1]
        < 0,
        "fixed_frontier_ci_wholly_favorable": fixed_frontier[
            "paired_developer_bootstrap"
        ]["ci_90"][1]
        < 0,
        "nested_all_ci_wholly_favorable": nested_all[
            "paired_developer_bootstrap"
        ]["ci_90"][1]
        < 0,
        "nested_frontier_ci_wholly_favorable": nested_frontier[
            "paired_developer_bootstrap"
        ]["ci_90"][1]
        < 0,
        "fixed_all_metrics_non_worse": _metrics_non_worse(fixed_all),
        "fixed_frontier_metrics_non_worse": _metrics_non_worse(fixed_frontier),
        "nested_all_metrics_non_worse": _metrics_non_worse(nested_all),
        "nested_frontier_metrics_non_worse": _metrics_non_worse(nested_frontier),
        "k3_anchor_within_10pct_of_baseline_error": k3[
            "error_relative_to_baseline"
        ]
        <= MAX_ANCHOR_ERROR_RELATIVE_TO_BASELINE,
        "grok_working_assumption_within_10pct_of_baseline_error": grok[
            "working_assumption"
        ]["error_relative_to_baseline"]
        <= MAX_ANCHOR_ERROR_RELATIVE_TO_BASELINE,
        "live_target_architectures_observed": all(
            row["architecture_observed"] for row in target_applicability
        ),
        # No post-freeze disclosures have yet been registered by this isolated
        # challenger.  These gates must be changed only by a future frozen audit.
        "prospective_disclosures_at_least_3": False,
        "prospective_frontier_disclosures_at_least_2": False,
    }
    return {
        "policy": {
            "fixed_all_minimum": {
                "rows": MIN_FIXED_ALL_ROWS,
                "developers": MIN_FIXED_ALL_DEVELOPERS,
            },
            "fixed_frontier_minimum": {
                "rows": MIN_FIXED_FRONTIER_ROWS,
                "developers": MIN_FIXED_FRONTIER_DEVELOPERS,
            },
            "nested_all_minimum": {
                "rows": MIN_NESTED_ALL_ROWS,
                "developers": MIN_NESTED_ALL_DEVELOPERS,
            },
            "nested_frontier_minimum": {
                "rows": MIN_NESTED_FRONTIER_ROWS,
                "developers": MIN_NESTED_FRONTIER_DEVELOPERS,
            },
            "statistical": (
                "Developer-cluster 90% CI upper bound must be below zero for "
                "fixed and nested evaluations, both all-model and frontier-like."
            ),
            "metric_bundle": (
                "Median multiplicative error, MAE, RMSE, and p80 must be no "
                "higher, and within-2x accuracy no lower, in every cohort."
            ),
            "anchor_tolerance": MAX_ANCHOR_ERROR_RELATIVE_TO_BASELINE,
            "applicability": (
                "MoE and reasoning status must be independently observed for "
                "every live target before architecture coefficients may be used."
            ),
            "prospective_minimum": {
                "new_developer_disclosures": MIN_PROSPECTIVE_DISCLOSURES,
                "frontier_like_disclosures": MIN_PROSPECTIVE_FRONTIER_DISCLOSURES,
            },
        },
        "results": observed,
        "failed_gates": sorted(name for name, passed in observed.items() if not passed),
        "all_pass": all(observed.values()),
    }


def build_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hashes_before = _source_hashes()
    rows, inventory = _build_outer_predictions()
    _add_nested_selection(rows)
    fixed = _fixed_evaluation(rows)
    nested = _nested_evaluation(rows)

    panels, _ = _load_panels()
    developer_eci = with_developers(
        {"ECI": panels["ECI"]}, developer_lookup()
    )["ECI"]
    anchors = _anchor_checks(rows, developer_eci)
    targets = _live_target_applicability()
    gates = _promotion_gates(fixed, nested, anchors, targets)
    hashes_after = _source_hashes()
    if hashes_before != hashes_after:
        raise RuntimeError("Audit inputs changed while the challenger was being built")

    result = {
        "metadata": {
            "generated_on": GENERATED_ON,
            "status": "zero-live-weight retrospective challenger",
            "target": TARGET,
            "baseline": (
                "60% aggregate ECI score-only + 40% aggregate ECI score/date, "
                "mixed in log10 parameter space"
            ),
            "challenger": (
                "60% aggregate ECI score-only + 40% aggregate ECI "
                "score/date/MoE/reasoning, mixed in log10 parameter space"
            ),
            "outer_split": (
                "strict earlier parameter-label eligibility; whole target "
                "developer excluded; equal aggregate weight per training developer"
            ),
            "information_date_policy": (
                "ECI target release date is the information-date proxy because "
                "there is no historical aggregate-score publication ledger."
            ),
            "benchmark_vintage_caveat": (
                "Pseudo-chronological: the current pinned aggregate ECI snapshot "
                "is used, not benchmark scores frozen at each historical date."
            ),
            "selection_caveat": (
                "The fixed 60/40 architecture specification was motivated after "
                "examining existing retrospective results; nested selection "
                "reduces but does not erase research-selection bias."
            ),
        },
        "inventory": inventory,
        "fixed_evaluation": fixed,
        "nested_evaluation": nested,
        "anchor_checks": anchors,
        "live_target_applicability": targets,
        "promotion_gates": gates,
        "decision": {
            "promote_to_live_eci": False,
            "incremental_live_weight": 0.0,
            "change_live_weights": False,
            "change_central_forecasts": False,
            "preserve_as_challenger": True,
            "reason": (
                "The overall retrospective error reduction is real enough to "
                "preserve, but frontier uncertainty crosses zero, within-2x "
                "accuracy deteriorates, target architecture is unobserved, and "
                "no prospective disclosures exist."
            ),
        },
        "source_files": hashes_after,
    }
    return result, rows


def render_csv(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=PREDICTION_FIELDS,
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(
        {field: row.get(field, "") for field in PREDICTION_FIELDS} for row in rows
    )
    return buffer.getvalue()


def render_report(result: dict[str, Any]) -> str:
    fixed_all = result["fixed_evaluation"]["all"]
    fixed_frontier = result["fixed_evaluation"]["frontier_like"]
    nested_all = result["nested_evaluation"]["all"]
    nested_frontier = result["nested_evaluation"]["frontier_like"]
    k3 = result["anchor_checks"]["kimi_k3"]
    grok = result["anchor_checks"]["grok_4_5"]
    failed = result["promotion_gates"]["failed_gates"]

    def line(label: str, block: dict[str, Any]) -> str:
        baseline = block["baseline"]
        challenger = block["challenger"]
        ci = block["paired_developer_bootstrap"]["ci_90"]
        return (
            f"| {label} | {baseline['n']} / {baseline['developers']} | "
            f"{baseline['median_multiplicative_error']:.3f}× | "
            f"{challenger['median_multiplicative_error']:.3f}× | "
            f"{baseline['mean_absolute_log10_error']:.3f} | "
            f"{challenger['mean_absolute_log10_error']:.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] |"
        )

    report = [
        "# ECI architecture-blend challenger",
        "",
        f"Generated {GENERATED_ON}. Status: **ZERO LIVE WEIGHT**.",
        "",
        "The challenger geometrically combines 60% aggregate ECI score-only "
        "with 40% aggregate ECI score/date/MoE/reasoning. The comparator is the "
        "current direct 60% score-only + 40% score/date mapping.",
        "",
        "## Held-out results",
        "",
        "| Evaluation | Rows / developers | Baseline median | Challenger median | Baseline MAE | Challenger MAE | 90% developer CI on MAE delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
        line("Fixed, all", fixed_all),
        line("Fixed, frontier-like", fixed_frontier),
        line("Nested, all", nested_all),
        line("Nested, frontier-like", nested_frontier),
        "",
        "The nested chooser uses only earlier outer-fold errors and excludes the "
        "target developer again. It selects the architecture challenger "
        f"{result['nested_evaluation']['selection_counts'].get('challenger', 0)} times "
        "and the baseline "
        f"{result['nested_evaluation']['selection_counts'].get('baseline', 0)} times.",
        "",
        "## Anchor checks",
        "",
        f"Kimi K3: baseline {k3['baseline_multiplicative_error']:.3f}× error; "
        f"challenger {k3['challenger_multiplicative_error']:.3f}×. Grok 4.5: "
        f"baseline {grok['baseline_multiplicative_error']:.3f}×; the explicitly "
        "non-observed MoE+reasoning working scenario gives "
        f"{grok['working_assumption']['multiplicative_error']:.3f}×. The full "
        "four-scenario Grok sensitivity is retained in the JSON.",
        "",
        "## Decision",
        "",
        "Do not change the live ECI factor or any headline parameter forecast. "
        "The frontier confidence intervals are not wholly favorable, within-2× "
        "accuracy deteriorates, live-target architecture is unobserved, and there "
        "are no frozen prospective disclosures.",
        "",
        "Failed gates:",
        "",
        *[f"- `{gate}`" for gate in failed],
        "",
        "The machine-readable audit and fold ledger are "
        f"`{_relative(RESULT)}` and `{_relative(PREDICTIONS)}`.",
        "",
    ]
    return "\n".join(report)


def write_outputs(
    result: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    PREDICTIONS.write_text(render_csv(rows), encoding="utf-8")
    REPORT.write_text(render_report(result), encoding="utf-8")


def main() -> None:
    result, rows = build_audit()
    write_outputs(result, rows)
    fixed = result["fixed_evaluation"]["all"]
    print(
        "ECI architecture challenger: "
        f"median {fixed['baseline']['median_multiplicative_error']:.3f}x -> "
        f"{fixed['challenger']['median_multiplicative_error']:.3f}x; "
        "live weight remains 0."
    )


if __name__ == "__main__":
    main()
