#!/usr/bin/env python3
"""Audit a conservative direct-total / active-transport shrinkage challenger.

The existing active-parameter audit predicts total parameters in two correlated
ways from the Artificial Analysis panel:

1. directly from score and release date; and
2. through predicted active parameters and a training-fold-only high-sparsity
   ratio.

This audit asks whether a geometric blend of those two held-out predictions is
more accurate on checkpoints that are independently known, after disclosure,
to have total/active ratios of at least 15.  Every component prediction was
already generated using strictly earlier parameter-label availability and with
the whole test developer removed.  The nested weight chooser below adds a
second chronology layer: it may use only earlier eligible prediction rows and
again removes the test developer.

The result is a zero-weight challenger.  Selecting the evaluation cohort uses
the disclosed target total/active ratio, while Fable/Sol/Opus sparsity is not
independently observed.  Therefore favorable retrospective accuracy cannot be
transported into the live forecast until the applicability gate is satisfied.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
PREDICTION_INPUT = OUT / "active_parameter_transport_predictions_2026-07-18.csv"
ACTIVE_AUDIT_INPUT = OUT / "active_parameter_transport_audit_2026-07-18.json"
AA_PANEL_INPUT = OUT / "aa_detailed_parameter_panel_2026-07-18.csv"
K3_FACTS_INPUT = ROOT / "sources/kimi_k3_release_evidence_2026-07-31.json"

RESULT = OUT / "active_parameter_shrinkage_challenger_2026-07-31.json"
PREDICTIONS = OUT / "active_parameter_shrinkage_challenger_predictions_2026-07-31.csv"

GENERATED_ON = "2026-07-31"
HIGH_SPARSITY_THRESHOLD = 15.0
FIXED_ACTIVE_WEIGHT = 0.50
NESTED_WEIGHT_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
MIN_META_TRAIN_ROWS = 15
MIN_META_TRAIN_DEVELOPERS = 6
FRONTIER_RANK_THRESHOLD = 0.90
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED_FIXED = 20_260_731
BOOTSTRAP_SEED_NESTED = 20_260_732


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _float(value: str, field: str) -> float:
    if value in (None, ""):
        raise ValueError(f"Missing required numeric field {field!r}")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"Non-finite numeric field {field!r}: {value!r}")
    return output


def _metric_summary(rows: list[dict[str, Any]], error_field: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "developers": 0}
    errors = np.asarray([float(row[error_field]) for row in rows], dtype=float)
    absolute = np.abs(errors)
    return {
        "n": len(rows),
        "developers": len({row["developer"] for row in rows}),
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "mean_absolute_log10_error": float(np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(errors**2))),
        "p80_multiplicative_error": float(10 ** np.quantile(absolute, 0.80)),
        "within_2x": float(np.mean(absolute <= math.log10(2.0))),
        "signed_bias_factor": float(10 ** np.mean(errors)),
    }


def _equal_developer_mae(rows: list[dict[str, Any]], error_field: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["developer"]].append(abs(float(row[error_field])))
    if not grouped:
        raise ValueError("Cannot compute equal-developer MAE on an empty cohort")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _paired_developer_bootstrap(
    rows: list[dict[str, Any]],
    candidate_error: str,
    *,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["developer"]].append(
            abs(float(row[candidate_error]))
            - abs(float(row["baseline_log10_error"]))
        )
    developers = sorted(grouped)
    if not developers:
        raise ValueError("Cannot bootstrap an empty cohort")
    deltas = np.asarray(
        [np.mean(grouped[developer]) for developer in developers], dtype=float
    )
    rng = np.random.default_rng(seed)
    draws = deltas[
        rng.integers(0, len(deltas), size=(BOOTSTRAP_SAMPLES, len(deltas)))
    ].mean(axis=1)
    return {
        "metric": (
            "equal-developer mean absolute log10 error; candidate minus "
            "direct-total baseline"
        ),
        "observed_delta": float(np.mean(deltas)),
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "probability_candidate_better": float(np.mean(draws < 0)),
        "developers": len(developers),
        "samples": BOOTSTRAP_SAMPLES,
        "random_seed": seed,
    }


def _blend_error(row: dict[str, Any], active_weight: float) -> float:
    return float(
        (1.0 - active_weight) * float(row["baseline_log10_error"])
        + active_weight * float(row["active_transport_log10_error"])
    )


def _load_rows() -> list[dict[str, Any]]:
    raw_predictions = read_csv(PREDICTION_INPUT)
    raw_panel = read_csv(AA_PANEL_INPUT)
    k3_facts = json.loads(K3_FACTS_INPUT.read_text(encoding="utf-8"))["kimi_k3"]
    if len({row["checkpoint_group_id"] for row in raw_predictions}) != len(
        raw_predictions
    ):
        raise ValueError("Active-transport predictions contain duplicate checkpoints")
    panel_by_checkpoint = {
        row["checkpoint_group_id"]: row for row in raw_panel
    }
    if len(panel_by_checkpoint) != len(raw_panel):
        raise ValueError("Detailed AA panel contains duplicate checkpoint groups")

    output: list[dict[str, Any]] = []
    for raw in raw_predictions:
        checkpoint = raw["checkpoint_group_id"]
        if checkpoint not in panel_by_checkpoint:
            raise ValueError(f"Prediction checkpoint is absent from AA panel: {checkpoint}")
        panel = panel_by_checkpoint[checkpoint]
        if raw["test_developer_excluded"].strip().lower() != "true":
            raise ValueError(f"Developer was not excluded for {raw['model']}")
        if raw["active_converted_total_log10_error"] in (None, ""):
            continue
        release_date = raw["release_date"]
        prediction_information_date = (
            raw.get("prediction_information_date") or release_date
        )
        eligibility_date = (
            panel.get("parameter_training_eligibility_date") or release_date
        )
        source_train_max_date = raw["train_max_date"]
        if source_train_max_date >= prediction_information_date:
            raise ValueError(
                f"Non-chronological source fold for {raw['model']}: "
                f"{source_train_max_date} >= {prediction_information_date}"
            )

        source_actual_total_b = _float(raw["actual_total_b"], "actual_total_b")
        source_actual_active_b = _float(raw["actual_active_b"], "actual_active_b")
        source_ratio = _float(
            raw["actual_total_to_active_ratio"], "actual_total_to_active_ratio"
        )
        actual_total_b = source_actual_total_b
        actual_active_b = source_actual_active_b
        actual_value_override = ""
        if raw["model"] == "Kimi K3":
            # Artificial Analysis exposes rounded 2.8T/104B metadata.  The
            # primary technical report discloses exact 2.78T/104.2B values.
            # Preserve the raw inputs below, but score the external check and
            # aggregate metrics against the higher-priority exact truth.
            actual_total_b = float(k3_facts["total_parameters_b_exact"])
            actual_active_b = float(k3_facts["activated_parameters_b_exact"])
            actual_value_override = "K3 exact primary evidence: 2780B total / 104.2B active"
        ratio = actual_total_b / actual_active_b
        if ratio < HIGH_SPARSITY_THRESHOLD:
            continue
        baseline_predicted_b = _float(
            raw["predicted_full_panel_total_b"], "predicted_full_panel_total_b"
        )
        active_predicted_b = _float(
            raw["active_converted_total_b"], "active_converted_total_b"
        )
        source_baseline_error = _float(
            raw["full_panel_total_log10_error"], "full_panel_total_log10_error"
        )
        source_active_error = _float(
            raw["active_converted_total_log10_error"],
            "active_converted_total_log10_error",
        )
        expected_source_baseline_error = math.log10(
            baseline_predicted_b / source_actual_total_b
        )
        expected_source_active_error = math.log10(
            active_predicted_b / source_actual_total_b
        )
        if not math.isclose(
            source_baseline_error, expected_source_baseline_error, abs_tol=1e-12
        ):
            raise ValueError(f"Baseline error mismatch for {raw['model']}")
        if not math.isclose(
            source_active_error, expected_source_active_error, abs_tol=1e-12
        ):
            raise ValueError(f"Active-transport error mismatch for {raw['model']}")

        baseline_error = math.log10(baseline_predicted_b / actual_total_b)
        active_error = math.log10(active_predicted_b / actual_total_b)

        fixed_error = (1.0 - FIXED_ACTIVE_WEIGHT) * baseline_error + FIXED_ACTIVE_WEIGHT * active_error
        fixed_prediction = actual_total_b * 10**fixed_error
        output.append(
            {
                "checkpoint_group_id": checkpoint,
                "release_date": release_date,
                "prediction_information_date": prediction_information_date,
                "parameter_training_eligibility_date": eligibility_date,
                "model": raw["model"],
                "developer": raw["developer"],
                "frontier_score_rank": _float(
                    raw["frontier_score_rank"], "frontier_score_rank"
                ),
                "actual_total_b": actual_total_b,
                "actual_active_b": actual_active_b,
                "actual_total_to_active_ratio": ratio,
                "source_input_actual_total_b": source_actual_total_b,
                "source_input_actual_active_b": source_actual_active_b,
                "source_input_total_to_active_ratio": source_ratio,
                "actual_value_override": actual_value_override,
                "baseline_predicted_total_b": baseline_predicted_b,
                "active_transport_predicted_total_b": active_predicted_b,
                "fixed_active_transport_weight": FIXED_ACTIVE_WEIGHT,
                "fixed_blend_predicted_total_b": fixed_prediction,
                "baseline_log10_error": baseline_error,
                "active_transport_log10_error": active_error,
                "fixed_blend_log10_error": fixed_error,
                "source_train_max_date": source_train_max_date,
                "source_test_developer_excluded": True,
            }
        )

    output.sort(
        key=lambda row: (
            row["prediction_information_date"],
            row["release_date"],
            row["model"],
        )
    )
    if not output:
        raise ValueError("No high-sparsity active-transport rows are eligible")
    return output


def _add_nested_predictions(rows: list[dict[str, Any]]) -> None:
    for test in rows:
        prior = [
            row
            for row in rows
            if row["parameter_training_eligibility_date"]
            < test["prediction_information_date"]
            and row["developer"] != test["developer"]
        ]
        developers = {row["developer"] for row in prior}
        if len(prior) < MIN_META_TRAIN_ROWS or len(developers) < MIN_META_TRAIN_DEVELOPERS:
            test.update(
                {
                    "nested_eligible": False,
                    "nested_meta_train_n": None,
                    "nested_meta_train_developers": None,
                    "nested_meta_train_max_eligibility_date": None,
                    "nested_selected_active_weight": None,
                    "nested_predicted_total_b": None,
                    "nested_log10_error": None,
                }
            )
            continue

        scores: list[tuple[float, float]] = []
        for weight in NESTED_WEIGHT_GRID:
            candidate_rows = [
                {**row, "candidate_error": _blend_error(row, weight)}
                for row in prior
            ]
            scores.append(
                (_equal_developer_mae(candidate_rows, "candidate_error"), weight)
            )
        _, selected_weight = min(scores)
        nested_error = _blend_error(test, selected_weight)
        nested_prediction = float(test["actual_total_b"] * 10**nested_error)
        max_eligibility = max(
            row["parameter_training_eligibility_date"] for row in prior
        )
        if max_eligibility >= test["prediction_information_date"]:
            raise ValueError(f"Nested chronology failed for {test['model']}")
        if any(row["developer"] == test["developer"] for row in prior):
            raise ValueError(f"Nested developer exclusion failed for {test['model']}")
        test.update(
            {
                "nested_eligible": True,
                "nested_meta_train_n": len(prior),
                "nested_meta_train_developers": len(developers),
                "nested_meta_train_max_eligibility_date": max_eligibility,
                "nested_selected_active_weight": selected_weight,
                "nested_predicted_total_b": nested_prediction,
                "nested_log10_error": nested_error,
            }
        )


def _cohort_result(
    rows: list[dict[str, Any]],
    candidate_error: str,
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        "baseline": _metric_summary(rows, "baseline_log10_error"),
        "candidate": _metric_summary(rows, candidate_error),
        "paired_developer_bootstrap": _paired_developer_bootstrap(
            rows, candidate_error, seed=seed
        ),
    }


def _k3_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [row for row in rows if row["model"] == "Kimi K3"]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Kimi K3 row; found {len(matches)}")
    row = matches[0]
    facts = json.loads(K3_FACTS_INPUT.read_text(encoding="utf-8"))
    disclosed_total = float(facts["kimi_k3"]["total_parameters_b_exact"])
    disclosed_active = float(facts["kimi_k3"]["activated_parameters_b_exact"])
    if not math.isclose(row["actual_total_b"], disclosed_total, abs_tol=1e-12):
        raise ValueError("K3 total does not match pinned primary evidence")
    if not math.isclose(row["actual_active_b"], disclosed_active, abs_tol=1e-12):
        raise ValueError("K3 active count does not match pinned primary evidence")
    return {
        "model": row["model"],
        "release_date": row["release_date"],
        "developer": row["developer"],
        "source_component_developer_excluded": row[
            "source_test_developer_excluded"
        ],
        "source_train_max_date": row["source_train_max_date"],
        "disclosed_total_b": disclosed_total,
        "disclosed_active_b": disclosed_active,
        "disclosed_total_to_active_ratio": row["actual_total_to_active_ratio"],
        "direct_total_predicted_b": row["baseline_predicted_total_b"],
        "active_transport_predicted_b": row[
            "active_transport_predicted_total_b"
        ],
        "fixed_50_50_predicted_b": row["fixed_blend_predicted_total_b"],
        "direct_total_error_factor": float(
            10 ** abs(row["baseline_log10_error"])
        ),
        "active_transport_error_factor": float(
            10 ** abs(row["active_transport_log10_error"])
        ),
        "fixed_50_50_error_factor": float(
            10 ** abs(row["fixed_blend_log10_error"])
        ),
        "fixed_50_50_signed_log10_error": row["fixed_blend_log10_error"],
        "nested_selected_active_weight": row["nested_selected_active_weight"],
        "nested_error_factor": (
            float(10 ** abs(row["nested_log10_error"]))
            if row["nested_log10_error"] is not None
            else None
        ),
        "status": (
            "retrospective external scale check: K3 was excluded from every "
            "component training fold, but this challenger was examined after "
            "K3's parameter disclosure"
        ),
    }


def build_audit() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _load_rows()
    _add_nested_predictions(rows)
    frontier = [
        row
        for row in rows
        if row["frontier_score_rank"] >= FRONTIER_RANK_THRESHOLD
    ]
    nested = [row for row in rows if row["nested_eligible"]]
    nested_frontier = [
        row
        for row in nested
        if row["frontier_score_rank"] >= FRONTIER_RANK_THRESHOLD
    ]

    fixed = {
        "all_high_sparsity": _cohort_result(
            rows,
            "fixed_blend_log10_error",
            seed=BOOTSTRAP_SEED_FIXED,
        ),
        "frontier_like_high_sparsity": _cohort_result(
            frontier,
            "fixed_blend_log10_error",
            seed=BOOTSTRAP_SEED_FIXED + 100,
        ),
    }
    nested_result = {
        "weight_grid": list(NESTED_WEIGHT_GRID),
        "minimum_meta_training_rows": MIN_META_TRAIN_ROWS,
        "minimum_meta_training_developers": MIN_META_TRAIN_DEVELOPERS,
        "selection_loss": "equal-developer mean absolute log10 error",
        "tie_break": "lower active-transport weight",
        "selected_weight_counts": {
            str(weight): sum(
                row["nested_selected_active_weight"] == weight for row in nested
            )
            for weight in NESTED_WEIGHT_GRID
        },
        "all_high_sparsity": _cohort_result(
            nested,
            "nested_log10_error",
            seed=BOOTSTRAP_SEED_NESTED,
        ),
        "frontier_like_high_sparsity": _cohort_result(
            nested_frontier,
            "nested_log10_error",
            seed=BOOTSTRAP_SEED_NESTED + 100,
        ),
    }

    fixed_all = fixed["all_high_sparsity"]
    fixed_frontier = fixed["frontier_like_high_sparsity"]
    nested_all = nested_result["all_high_sparsity"]
    nested_frontier_result = nested_result["frontier_like_high_sparsity"]
    empirical_gates = {
        "at_least_40_high_sparsity_rows": len(rows) >= 40,
        "at_least_12_high_sparsity_developers": (
            len({row["developer"] for row in rows}) >= 12
        ),
        "at_least_20_frontier_rows": len(frontier) >= 20,
        "at_least_8_frontier_developers": (
            len({row["developer"] for row in frontier}) >= 8
        ),
        "fixed_all_ci90_wholly_favorable": (
            fixed_all["paired_developer_bootstrap"]["ci_90"][1] < 0
        ),
        "fixed_frontier_ci90_wholly_favorable": (
            fixed_frontier["paired_developer_bootstrap"]["ci_90"][1] < 0
        ),
        "fixed_all_median_improves": (
            fixed_all["candidate"]["median_multiplicative_error"]
            < fixed_all["baseline"]["median_multiplicative_error"]
        ),
        "fixed_frontier_median_improves": (
            fixed_frontier["candidate"]["median_multiplicative_error"]
            < fixed_frontier["baseline"]["median_multiplicative_error"]
        ),
        "nested_all_ci90_wholly_favorable": (
            nested_all["paired_developer_bootstrap"]["ci_90"][1] < 0
        ),
        "nested_frontier_ci90_wholly_favorable": (
            nested_frontier_result["paired_developer_bootstrap"]["ci_90"][1]
            < 0
        ),
    }
    applicability_gates = {
        "target_high_sparsity_status_independently_observed_pre_outcome": False,
        "target_active_fraction_or_equivalent_architecture_observed_pre_outcome": False,
        "challenger_preregistered_before_evaluated_outcomes": False,
    }
    all_gates = all(empirical_gates.values()) and all(applicability_gates.values())

    result = {
        "metadata": {
            "generated_on": GENERATED_ON,
            "status": "zero-weight retrospective challenger",
            "target": "total pretrained parameters in billions",
            "fixed_rule": (
                "geometric 50/50 blend of the direct-total prediction and the "
                "predicted-active-to-high-sparsity transport"
            ),
            "high_sparsity_threshold_total_to_active": HIGH_SPARSITY_THRESHOLD,
            "component_outer_split": (
                "strictly earlier parameter-training eligibility date; entire "
                "test developer removed"
            ),
            "nested_meta_split": (
                "only earlier high-sparsity held-out prediction rows whose "
                "parameter labels were eligible before the test release; "
                "entire test developer removed again"
            ),
            "important_selection_caveat": (
                "Evaluation membership uses the disclosed total/active ratio. "
                "That is valid for retrospective conditional accuracy but is "
                "not an available feature for the undisclosed frontier targets."
            ),
        },
        "inventory": {
            "high_sparsity_rows": len(rows),
            "high_sparsity_developers": len({row["developer"] for row in rows}),
            "frontier_like_rows": len(frontier),
            "frontier_like_developers": len(
                {row["developer"] for row in frontier}
            ),
            "nested_eligible_rows": len(nested),
            "nested_eligible_developers": len(
                {row["developer"] for row in nested}
            ),
            "nested_frontier_rows": len(nested_frontier),
            "nested_frontier_developers": len(
                {row["developer"] for row in nested_frontier}
            ),
        },
        "fixed_50_50_evaluation": fixed,
        "nested_weight_evaluation": nested_result,
        "kimi_k3_external_check": _k3_check(rows),
        "promotion_gates": {
            "empirical": empirical_gates,
            "applicability_and_prospective": applicability_gates,
            "all_gates_pass": all_gates,
        },
        "decision": {
            "promote_to_live_factor": False,
            "incremental_live_weight": 0.0,
            "change_headline_forecasts": False,
            "preserve_as_challenger": True,
            "reason": (
                "The fixed and nested shrinkage rules improve developer-clustered "
                "held-out accuracy, including the frontier subset, but cohort "
                "membership depends on target sparsity that is not independently "
                "observed for Fable, Sol, or Opus 5. The rule was also examined "
                "retrospectively rather than frozen before these outcomes."
            ),
            "applicability_requirement": (
                "Before this branch can receive nonzero weight for a target, a "
                "pre-outcome source independent of the target's total-parameter "
                "label must establish a qualifying high-sparsity active fraction "
                "or an architecture-to-sparsity mapping validated prospectively."
            ),
            "next_validation": (
                "Freeze the 50/50 rule now and score it without refitting on the "
                "next independently identified high-sparsity disclosures from at "
                "least three new developers."
            ),
        },
        "limitations": [
            "Artificial Analysis measurements are current-snapshot rather than release-vintage benchmark observations.",
            "The direct-total and active-transport predictions share AA score/date inputs and are correlated; the blend is shrinkage, not independent evidence.",
            "The >=15x evaluation cohort is defined using disclosed total and active counts, so target applicability must be established independently before prediction.",
            "K3 is held out from component training but the 50/50 challenger was examined after K3's disclosure; it is not a prospective validation.",
            "The nested selector validates historical weight transport but does not remove the target-classification problem.",
        ],
        "source_files": {
            str(PREDICTION_INPUT.relative_to(ROOT)): sha256(PREDICTION_INPUT),
            str(ACTIVE_AUDIT_INPUT.relative_to(ROOT)): sha256(ACTIVE_AUDIT_INPUT),
            str(AA_PANEL_INPUT.relative_to(ROOT)): sha256(AA_PANEL_INPUT),
            str(K3_FACTS_INPUT.relative_to(ROOT)): sha256(K3_FACTS_INPUT),
        },
        "outputs": {
            "predictions": str(PREDICTIONS.relative_to(ROOT)),
            "result": str(RESULT.relative_to(ROOT)),
        },
    }
    return result, rows


PREDICTION_FIELDS = (
    "checkpoint_group_id",
    "release_date",
    "prediction_information_date",
    "parameter_training_eligibility_date",
    "model",
    "developer",
    "frontier_score_rank",
    "actual_total_b",
    "actual_active_b",
    "actual_total_to_active_ratio",
    "source_input_actual_total_b",
    "source_input_actual_active_b",
    "source_input_total_to_active_ratio",
    "actual_value_override",
    "baseline_predicted_total_b",
    "active_transport_predicted_total_b",
    "fixed_active_transport_weight",
    "fixed_blend_predicted_total_b",
    "baseline_log10_error",
    "active_transport_log10_error",
    "fixed_blend_log10_error",
    "nested_eligible",
    "nested_meta_train_n",
    "nested_meta_train_developers",
    "nested_meta_train_max_eligibility_date",
    "nested_selected_active_weight",
    "nested_predicted_total_b",
    "nested_log10_error",
    "source_train_max_date",
    "source_test_developer_excluded",
)


def write_outputs(result: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=PREDICTION_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in PREDICTION_FIELDS}
            for row in rows
        )


def main() -> None:
    result, rows = build_audit()
    write_outputs(result, rows)
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "predictions": str(PREDICTIONS),
                "inventory": result["inventory"],
                "fixed_50_50_evaluation": result["fixed_50_50_evaluation"],
                "nested_weight_evaluation": result["nested_weight_evaluation"],
                "kimi_k3_external_check": result["kimi_k3_external_check"],
                "decision": result["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
