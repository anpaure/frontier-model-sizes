#!/usr/bin/env python3
"""Audit active-parameter transport and the independence of the compute branch.

The live forecast targets *total* pretrained parameters, while capability and
serving cost are more directly related to parameters activated per token.  This
script uses the complete deduplicated Artificial Analysis panel and its
checkpoints with both total and active parameter metadata to test whether an
active-parameter decomposition improves genuinely held-out inference.

Every chronological fold removes the entire test developer.  The architecture
transport is deliberately kept out of the live mixture unless its paired,
developer-cluster bootstrap interval is narrowly wholly favorable.  Kimi K3 is an
external disclosed total-and-activated-parameter check: no Kimi checkpoint is
used to fit its active-parameter prediction.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from aa_calibration_overrides import (
    OVERRIDES_PATH as AA_CALIBRATION_OVERRIDES_PATH,
    parameter_label_available_before,
    parameter_training_eligibility_date,
)
from aa_score_availability import aa_prediction_information_date
from frontier_target_signals import AA_TARGET_SIGNALS


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
PANEL_INPUT = OUT / "aa_detailed_parameter_panel_2026-07-18.csv"
FACTS_INPUT = ROOT / "sources/kimi_k3_release_evidence_2026-07-31.json"
UNIFIED_INPUT = OUT / "unified_model_observations_compute_enriched_2026-07-17.csv"

RESULT = OUT / "active_parameter_transport_audit_2026-07-18.json"
PREDICTIONS = OUT / "active_parameter_transport_predictions_2026-07-18.csv"
TARGETS = OUT / "active_parameter_transport_targets_2026-07-18.csv"

DATE_ORIGIN = date(2023, 1, 1)
MIN_TRAIN_ROWS = 30
MIN_TRAIN_DEVELOPERS = 8
HIGH_SPARSITY_THRESHOLD = 15.0
BOOTSTRAP_SAMPLES = 20_000
SEED = 20_260_718

TARGET_SPECS = (
    ("Claude Fable 5", AA_TARGET_SIGNALS["Claude Fable 5"]["score"], "2026-06-09"),
    ("GPT-5.6 Sol", AA_TARGET_SIGNALS["GPT-5.6 Sol"]["score"], "2026-07-09"),
    ("Kimi K3", AA_TARGET_SIGNALS["Kimi K3"]["score"], "2026-07-16"),
)

COMPUTE_TARGETS = (
    "Claude Fable 5",
    "GPT-5.6 Sol",
    "Kimi K3",
    "Claude Opus 4.8",
    "GPT-5.5",
    "GPT-5.6 Terra",
    "Claude Sonnet 5",
    "GPT-5.6 Luna",
    "Grok 4.5",
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
    if not rows:
        raise ValueError(f"Refusing to write empty audit table: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def years(value: str) -> float:
    return (date.fromisoformat(value) - DATE_ORIGIN).days / 365.25


def metric_summary(errors: list[float]) -> dict[str, Any]:
    values = np.asarray(errors, dtype=float)
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


def family_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = Counter(row["developer"] for row in rows)
    weights = np.asarray(
        [
            (0.5 if row["estimated_score"] else 1.0) / counts[row["developer"]]
            for row in rows
        ],
        dtype=float,
    )
    return weights / weights.mean()


def design(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[1.0, row["score"], years(row["release_date"])] for row in rows],
        dtype=float,
    )


def fit(rows: list[dict[str, Any]], target: str) -> np.ndarray:
    matrix = design(rows)
    values = np.log10(np.asarray([row[target] for row in rows], dtype=float))
    sqrt_weight = np.sqrt(family_weights(rows))
    beta, *_ = np.linalg.lstsq(
        matrix * sqrt_weight[:, None], values * sqrt_weight, rcond=None
    )
    return beta


def predict(beta: np.ndarray, row: dict[str, Any]) -> float:
    return float(10 ** (design([row]) @ beta).item())


def load_panel() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = read_csv(PANEL_INPUT)
    if not raw:
        raise ValueError("Detailed AA panel is empty")
    if len({row["checkpoint_group_id"] for row in raw}) != len(raw):
        raise ValueError("Detailed AA panel contains duplicate checkpoint groups")

    rows: list[dict[str, Any]] = []
    for row in raw:
        rows.append(
            {
                "checkpoint_group_id": row["checkpoint_group_id"],
                "model": row["selected_name"],
                "developer": row["creator_slug"],
                "release_date": row["release_date"],
                "score": float(row["intelligence_index"]),
                "estimated_score": row["intelligence_index_estimated"].lower() == "true",
                "total_b": float(row["parameters_b"]),
                "raw_parameter_total_b": row.get("raw_parameter_total_b", ""),
                "active_b": (
                    float(row["active_parameters_b"])
                    if row["active_parameters_b"]
                    else None
                ),
                "source_page_url": row["source_page_url"],
                "parameter_source": row["parameter_source"],
                "parameter_label_available_date": row.get(
                    "parameter_label_available_date"
                )
                or row["release_date"],
                "selected_slug": row.get("selected_slug", ""),
                "aa_score_available_date": row.get(
                    "aa_score_available_date", row["release_date"]
                ),
                "aa_score_availability_verified": row.get(
                    "aa_score_availability_verified", "false"
                ),
            }
        )
    active = [row for row in rows if row["active_b"] is not None]
    if (
        len(active) < MIN_TRAIN_ROWS
        or len({row["developer"] for row in active}) < MIN_TRAIN_DEVELOPERS
    ):
        raise ValueError(
            "Active-parameter panel is too small for the declared backtest: "
            f"{len(active)} rows across {len({row['developer'] for row in active})} developers"
        )
    if any(row["active_b"] >= row["total_b"] for row in active):
        raise ValueError("Active-parameter subset unexpectedly contains a dense/equal row")
    return rows, active


def paired_bootstrap(
    rows: list[dict[str, Any]], left_error: str, right_error: str
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["developer"]].append(
            abs(float(row[left_error])) - abs(float(row[right_error]))
        )
    developer_deltas = {
        developer: float(np.mean(values)) for developer, values in grouped.items()
    }
    developers = sorted(developer_deltas)
    observed = float(np.mean(list(developer_deltas.values())))
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for index in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(developers, size=len(developers), replace=True)
        draws[index] = np.mean([developer_deltas[name] for name in sampled])
    return {
        "metric": f"equal-developer absolute log10 error; {left_error} minus {right_error}",
        "observed_delta": observed,
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "probability_left_better": float(np.mean(draws < 0)),
        "samples": BOOTSTRAP_SAMPLES,
        "developers": len(developers),
    }


def chronological_backtest(
    all_rows: list[dict[str, Any]], active_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for test in sorted(
        active_rows,
        key=lambda row: (
            aa_prediction_information_date(row),
            row["release_date"],
            row["model"],
        ),
    ):
        prediction_date = aa_prediction_information_date(test)
        prior_active = [
            row
            for row in active_rows
            if parameter_label_available_before(row, prediction_date)
            and row["developer"] != test["developer"]
        ]
        prior_total = [
            row
            for row in all_rows
            if parameter_label_available_before(row, prediction_date)
            and row["developer"] != test["developer"]
        ]
        if (
            len(prior_active) < MIN_TRAIN_ROWS
            or len({row["developer"] for row in prior_active}) < MIN_TRAIN_DEVELOPERS
        ):
            continue

        active_beta = fit(prior_active, "active_b")
        same_panel_total_beta = fit(prior_active, "total_b")
        full_panel_total_beta = fit(prior_total, "total_b")
        predicted_active = predict(active_beta, test)
        predicted_same_panel_total = predict(same_panel_total_beta, test)
        predicted_full_panel_total = predict(full_panel_total_beta, test)
        all_prior_scores = [
            row["score"]
            for row in active_rows
            if parameter_label_available_before(row, prediction_date)
        ]
        frontier_rank = sum(score <= test["score"] for score in all_prior_scores) / len(
            all_prior_scores
        )

        high_sparsity_train = [
            row
            for row in prior_active
            if row["total_b"] / row["active_b"] >= HIGH_SPARSITY_THRESHOLD
        ]
        high_sparsity_developers = sorted(
            {row["developer"] for row in high_sparsity_train}
        )
        reference_ratio = None
        converted_total = None
        converted_error = None
        if len(high_sparsity_train) >= 8 and len(high_sparsity_developers) >= 4:
            ratios_by_developer: dict[str, list[float]] = defaultdict(list)
            for row in high_sparsity_train:
                ratios_by_developer[row["developer"]].append(
                    math.log10(row["total_b"] / row["active_b"])
                )
            reference_ratio = float(
                10
                ** np.mean(
                    [np.mean(values) for values in ratios_by_developer.values()]
                )
            )
            converted_total = predicted_active * reference_ratio
            converted_error = math.log10(converted_total / test["total_b"])

        output.append(
            {
                "checkpoint_group_id": test["checkpoint_group_id"],
                "release_date": test["release_date"],
                "prediction_information_date": prediction_date,
                "model": test["model"],
                "developer": test["developer"],
                "aa_score": test["score"],
                "frontier_score_rank": frontier_rank,
                "actual_active_b": test["active_b"],
                "predicted_active_b": predicted_active,
                "active_log10_error": math.log10(predicted_active / test["active_b"]),
                "actual_total_b": test["total_b"],
                "actual_total_to_active_ratio": test["total_b"] / test["active_b"],
                "predicted_same_panel_total_b": predicted_same_panel_total,
                "same_panel_total_log10_error": math.log10(
                    predicted_same_panel_total / test["total_b"]
                ),
                "predicted_full_panel_total_b": predicted_full_panel_total,
                "full_panel_total_log10_error": math.log10(
                    predicted_full_panel_total / test["total_b"]
                ),
                "high_sparsity_reference_ratio": reference_ratio,
                "active_converted_total_b": converted_total,
                "active_converted_total_log10_error": converted_error,
                "active_train_n": len(prior_active),
                "active_train_developers": len(
                    {row["developer"] for row in prior_active}
                ),
                "total_train_n": len(prior_total),
                "train_max_date": max(
                    parameter_training_eligibility_date(row)
                    for row in prior_active
                ),
                "test_developer_excluded": not any(
                    row["developer"] == test["developer"] for row in prior_active
                ),
            }
        )
    if not output:
        raise ValueError("No eligible chronological active-parameter folds")
    if len({row["checkpoint_group_id"] for row in output}) != len(output):
        raise ValueError("Chronological active-parameter folds contain duplicate checkpoints")
    return output


def target_transport(
    active_rows: list[dict[str, Any]], facts: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    k3_date = next(
        release_date
        for model, _, release_date in TARGET_SPECS
        if model == "Kimi K3"
    )
    training = [
        row
        for row in active_rows
        if parameter_label_available_before(row, k3_date)
        and row["developer"] not in {"kimi", "moonshot"}
    ]
    beta = fit(training, "active_b")
    k3_total_b = float(facts["kimi_k3"]["total_parameters_b_exact"])
    k3_active_b = float(facts["kimi_k3"]["activated_parameters_b_exact"])
    k3_disclosed_ratio = float(
        facts["derived_quantities"]["total_to_activated_parameter_ratio"]
    )
    k2_disclosed_ratio = float(
        facts["derived_quantities"]["k2_total_to_activated_parameter_ratio"]
    )
    target_rows = [
        {
            "model": model,
            "score": score,
            "release_date": release_date,
            "developer": "target",
            "estimated_score": False,
        }
        for model, score, release_date in TARGET_SPECS
    ]
    predictions = {row["model"]: predict(beta, row) for row in target_rows}
    predicted_k3_active = predictions["Kimi K3"]
    active_calibration_factor = k3_active_b / predicted_k3_active

    output: list[dict[str, Any]] = []
    for row in target_rows:
        predicted_active_b = predictions[row["model"]]
        calibrated_active_b = predicted_active_b * active_calibration_factor
        k3_ratio_total_b = calibrated_active_b * k3_disclosed_ratio
        k2_ratio_total_b = calibrated_active_b * k2_disclosed_ratio
        output.append(
            {
                "model": row["model"],
                "release_date": row["release_date"],
                "aa_score": row["score"],
                "predicted_active_b": predicted_active_b,
                "k3_calibrated_active_b": calibrated_active_b,
                "k3_anchored_total_t": k3_ratio_total_b / 1000,
                "k3_sparsity_total_t": k3_ratio_total_b / 1000,
                "k2_sparsity_total_t": k2_ratio_total_b / 1000,
                "status": "disclosed external check" if row["model"] == "Kimi K3" else "diagnostic only",
            }
        )
    return output, {
        "training_rows": len(training),
        "training_developers": len({row["developer"] for row in training}),
        "kimi_developer_removed": True,
        "coefficients_intercept_score_date": [float(value) for value in beta],
        "predicted_k3_active_b": predicted_k3_active,
        "k3_disclosed_active_b": k3_active_b,
        "k3_active_prediction_multiplicative_error": max(
            predicted_k3_active / k3_active_b,
            k3_active_b / predicted_k3_active,
        ),
        "k3_active_prediction_calibration_factor": active_calibration_factor,
        "k3_disclosed_total_b": k3_total_b,
        "k3_disclosed_total_to_active_ratio": k3_disclosed_ratio,
        "k3_disclosed_active_fraction": facts["derived_quantities"][
            "activated_parameter_fraction"
        ],
        "k2_disclosed_total_to_active_ratio": k2_disclosed_ratio,
        "selected_routed_expert_fraction": facts["derived_quantities"][
            "selected_routed_expert_fraction"
        ],
        "active_fraction_over_selected_expert_fraction": (
            facts["derived_quantities"]["activated_parameter_fraction"]
            / facts["derived_quantities"]["selected_routed_expert_fraction"]
        ),
    }


def compute_independence_audit(unified: list[dict[str, str]]) -> dict[str, Any]:
    normalized_targets = {
        "Claude Opus 4.7 / 4.8 shared base": "Claude Opus 4.8",
    }
    estimated: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in unified:
        display = normalized_targets.get(row["canonical_display_name"], row["canonical_display_name"])
        if (
            display in COMPUTE_TARGETS
            and row["source"] == "Epoch"
            and row["model_level_include"] == "true"
            and row["epoch_training_compute_flop"]
        ):
            estimated[display].append(row)
    estimated_names = sorted(estimated)
    estimate_records = []
    for model in estimated_names:
        for row in estimated[model]:
            raw = json.loads(row["source_record_json"])
            estimate_records.append(
                {
                    "model": model,
                    "training_compute_flop": float(row["epoch_training_compute_flop"]),
                    "epoch_confidence": raw.get("Confidence", ""),
                    "epoch_notes": raw.get("Training compute notes", ""),
                    "classification": "Epoch estimate; not a primary-source disclosure",
                }
            )
    return {
        "target_models": len(COMPUTE_TARGETS),
        "target_models_with_epoch_training_compute_estimate": len(estimated_names),
        "epoch_compute_estimate_target_names": estimated_names,
        "target_models_with_disclosed_training_compute": 0,
        "disclosed_training_compute_target_names": [],
        "epoch_compute_estimate_records": estimate_records,
        "live_target_compute_input": "The live branch predicts compute from AA score and exact release date. K3 now has a speculative Epoch compute estimate, but no target has a primary-source training-compute disclosure.",
        "algebra": "If logC = a + b*AA + c*date and logP = d + e*logC + f*date, then logP = (d+e*a) + (e*b)*AA + (e*c+f)*date. The target prediction remains in the same intercept/AA/date feature span.",
        "independent_target_evidence": False,
        "recommended_classification": "compute-structured AA/date regularizer, not an independent target likelihood",
        "change_numeric_weight": False,
        "reason": "The current small weight is correlated regularization. K3's Epoch value is explicitly speculative and cannot be treated as an independent target disclosure; the other hidden targets still use score/date-predicted compute.",
    }


def main() -> None:
    all_rows, active_rows = load_panel()
    facts = json.loads(FACTS_INPUT.read_text(encoding="utf-8"))
    unified = read_csv(UNIFIED_INPUT)
    predictions = chronological_backtest(all_rows, active_rows)
    write_csv(PREDICTIONS, predictions)

    active_metrics = metric_summary(
        [float(row["active_log10_error"]) for row in predictions]
    )
    same_total_metrics = metric_summary(
        [float(row["same_panel_total_log10_error"]) for row in predictions]
    )
    full_total_metrics = metric_summary(
        [float(row["full_panel_total_log10_error"]) for row in predictions]
    )
    frontier = [row for row in predictions if row["frontier_score_rank"] >= 0.90]

    high_sparsity = [
        row
        for row in predictions
        if row["actual_total_to_active_ratio"] >= HIGH_SPARSITY_THRESHOLD
        and row["active_converted_total_log10_error"] is not None
    ]
    converted_metrics = metric_summary(
        [float(row["active_converted_total_log10_error"]) for row in high_sparsity]
    )
    converted_baseline_metrics = metric_summary(
        [float(row["full_panel_total_log10_error"]) for row in high_sparsity]
    )

    targets, target_diagnostic = target_transport(active_rows, facts)
    write_csv(TARGETS, targets)
    active_bootstrap = paired_bootstrap(
        predictions, "active_log10_error", "same_panel_total_log10_error"
    )
    conversion_bootstrap = paired_bootstrap(
        high_sparsity,
        "active_converted_total_log10_error",
        "full_panel_total_log10_error",
    )

    # A favorable clustered mean alone is not enough to make this a live
    # likelihood.  The refreshed panel still has a slightly worse primary
    # median, and no hidden target discloses the sparsity ratio needed for the
    # active→total conversion.
    has_independent_target_architecture = False
    promote = (
        conversion_bootstrap["ci_90"][1] < 0
        and converted_metrics["median_multiplicative_error"]
        < converted_baseline_metrics["median_multiplicative_error"]
        and has_independent_target_architecture
    )
    result = {
        "metadata": {
            "generated_on": "2026-07-31",
            "target": "total pretrained parameters, with active parameters as an architecture-aware intermediate",
            "outer_split": "strictly earlier parameter-training eligibility date; entire test developer removed",
            "training_weights": "equal total weight per developer; estimated AA scores receive half weight",
            "minimum_training_rows": MIN_TRAIN_ROWS,
            "minimum_training_developers": MIN_TRAIN_DEVELOPERS,
            "high_sparsity_threshold_total_to_active": HIGH_SPARSITY_THRESHOLD,
        },
        "inventory": {
            "detailed_total_parameter_checkpoints": len(all_rows),
            "active_parameter_checkpoints": len(active_rows),
            "active_parameter_developers": len({row["developer"] for row in active_rows}),
            "chronological_predictions": len(predictions),
            "frontier_like_predictions": len(frontier),
            "high_sparsity_conversion_predictions": len(high_sparsity),
        },
        "active_parameter_predictability": {
            "active_score_date": active_metrics,
            "total_score_date_same_active_checkpoint_panel": same_total_metrics,
            "total_score_date_full_checkpoint_panel": full_total_metrics,
            "paired_active_vs_same_panel_total": active_bootstrap,
            "frontier_like": {
                "active_score_date": metric_summary(
                    [float(row["active_log10_error"]) for row in frontier]
                ),
                "total_score_date_same_panel": metric_summary(
                    [float(row["same_panel_total_log10_error"]) for row in frontier]
                ),
            },
            "interpretation": "Active parameters are modestly easier to recover from AA score/date and the refreshed developer-cluster 90% interval is narrowly wholly favorable. This does not identify total parameters without an observed target sparsity ratio; the high-sparsity transport still slightly worsens the primary median.",
        },
        "high_sparsity_total_transport": {
            "candidate": converted_metrics,
            "direct_total_baseline": converted_baseline_metrics,
            "paired_cluster_bootstrap": conversion_bootstrap,
            "interpretation": "The active-parameter conversion lowers mean and tail error for the >=15x sparsity scope and its developer-clustered mean interval is favorable, but the primary median is slightly worse and hidden-target sparsity is unobserved.",
        },
        "kimi_k3_external_architecture_check": target_diagnostic,
        "target_sensitivity": targets,
        "compute_branch_independence": compute_independence_audit(unified),
        "decision": {
            "promote_active_transport_to_live_factor": promote,
            "independent_target_architecture_observed": has_independent_target_architecture,
            "incremental_live_weight": 0.0,
            "change_headline_forecasts": False,
            "classify_compute_as_independent_evidence": False,
            "update_compute_description_to_show_dependency": True,
            "reason": "K3 provides exact 2.78T total and 104.2B activated counts. The refreshed high-sparsity comparison improves clustered mean and tail error, but slightly worsens the primary median and Fable/Sol sparsity is undisclosed. Preserve live centers and expose the K3-sparsity sensitivity at zero weight.",
        },
        "limitations": [
            "AA scores are a current snapshot rather than historical benchmark vintages.",
            f"AA active-parameter metadata exists for {len(active_rows)} sparse checkpoints and omits dense rows, so availability is non-random.",
            "K3's exact 104.2B activated count is disclosed and is 2.10x the naive 2.78T*(16/896) calculation; routed-expert selection fraction must not be used as active-parameter fraction.",
            "K3's 2.78T total and 104.2B activated counts are paired architecture facts, not independent likelihood terms.",
            "Assuming Fable and Sol have K3-like sparsity is a user-supplied architecture prior, not a public disclosure from Anthropic or OpenAI.",
        ],
        "source_files": {
            str(PANEL_INPUT.relative_to(ROOT)): sha256(PANEL_INPUT),
            str(AA_CALIBRATION_OVERRIDES_PATH.relative_to(ROOT)): sha256(
                AA_CALIBRATION_OVERRIDES_PATH
            ),
            str(FACTS_INPUT.relative_to(ROOT)): sha256(FACTS_INPUT),
            str(UNIFIED_INPUT.relative_to(ROOT)): sha256(UNIFIED_INPUT),
        },
        "outputs": {
            "predictions": str(PREDICTIONS.relative_to(ROOT)),
            "targets": str(TARGETS.relative_to(ROOT)),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "inventory": result["inventory"],
                "active": active_metrics,
                "high_sparsity_candidate": converted_metrics,
                "high_sparsity_baseline": converted_baseline_metrics,
                "target_sensitivity": targets,
                "decision": result["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
