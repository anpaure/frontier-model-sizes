#!/usr/bin/env python3
"""Audit robustness and dependence of the judgmental crowd layer.

The crowd weight is a user-selected ensemble policy, not an empirically
calibrated likelihood.  This audit checks whether its center is driven by one
contributor and measures cross-target dependence without changing weights or
narrowing predictive intervals.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
LEDGER = ROOT / "sources" / "human_parameter_forecasts_2026-07-17.csv"
FORECAST = ROOT / "site" / "public" / "data" / "forecast-model.json"
RESULT = OUT / "crowd_robustness_audit_2026-07-31.json"
TARGETS = ("Claude Fable 5", "GPT-5.6 Sol")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("Geometric means require nonempty positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        raise ValueError("Correlation requires at least three paired observations")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    covariance = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    if left_ss <= 0 or right_ss <= 0:
        raise ValueError("Correlation is undefined for a constant forecast pool")
    return covariance / math.sqrt(left_ss * right_ss)


def crowd_point(row: dict[str, str]) -> float:
    low = float(row["low_t"])
    high = float(row["high_t"])
    if low <= 0 or high < low:
        raise ValueError(f"Invalid range in {row['forecast_id']}")
    central_text = row.get("central_t", "").strip()
    if central_text:
        central = float(central_text)
        if not low <= central <= high:
            raise ValueError(f"Central estimate outside range in {row['forecast_id']}")
        return central
    return math.sqrt(low * high)


def load_active_rows() -> list[dict[str, Any]]:
    with LEDGER.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "forecast_id",
        "contributor",
        "date",
        "model",
        "low_t",
        "high_t",
        "central_t",
        "provenance",
        "supersedes",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Human forecast ledger schema mismatch")
    ids = [row["forecast_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate forecast_id")
    superseded = {row["supersedes"] for row in rows if row["supersedes"]}
    if not superseded.issubset(ids):
        raise ValueError("Unknown superseded forecast_id")
    active = []
    for row in rows:
        if row["forecast_id"] in superseded:
            continue
        copied: dict[str, Any] = dict(row)
        copied["point_t"] = crowd_point(row)
        active.append(copied)
    keys = [(row["contributor"], row["model"]) for row in active]
    if len(keys) != len(set(keys)):
        raise ValueError("Multiple active forecasts for a contributor/model")
    return active


def target_summary(
    model: str, rows: list[dict[str, Any]], evidence_center: float
) -> dict[str, Any]:
    selected = [row for row in rows if row["model"] == model]
    if len(selected) < 10:
        raise ValueError(f"Crowd pool is too small for {model}")
    values = [float(row["point_t"]) for row in selected]
    center = geometric_mean(values)
    final = math.sqrt(center * evidence_center)
    jackknife_crowd = [
        geometric_mean(values[:index] + values[index + 1 :])
        for index in range(len(values))
    ]
    jackknife_final = [math.sqrt(value * evidence_center) for value in jackknife_crowd]
    trim_each_tail = max(1, math.floor(0.1 * len(values)))
    sorted_logs = sorted(math.log(value) for value in values)
    trimmed_center = math.exp(
        statistics.fmean(sorted_logs[trim_each_tail:-trim_each_tail])
    )
    median_center = statistics.median(values)
    robust_finals = {
        "geometric_mean": final,
        "median": math.sqrt(median_center * evidence_center),
        "ten_percent_each_tail_log_trimmed_mean": math.sqrt(
            trimmed_center * evidence_center
        ),
    }
    maximum_jackknife_shift = max(abs(value / final - 1) for value in jackknife_final)
    maximum_robust_center_shift = max(
        abs(value / final - 1) for value in robust_finals.values()
    )
    return {
        "model": model,
        "n": len(values),
        "evidence_center_t": evidence_center,
        "crowd_geometric_center_t": center,
        "displayed_final_t": final,
        "individual_point_min_t": min(values),
        "individual_point_max_t": max(values),
        "individual_point_span_x": max(values) / min(values),
        "crowd_median_t": median_center,
        "crowd_ten_percent_each_tail_log_trimmed_mean_t": trimmed_center,
        "leave_one_contributor_out": {
            "crowd_center_min_t": min(jackknife_crowd),
            "crowd_center_max_t": max(jackknife_crowd),
            "final_min_t": min(jackknife_final),
            "final_max_t": max(jackknife_final),
            "maximum_absolute_final_shift_fraction": maximum_jackknife_shift,
        },
        "robust_final_sensitivity_t": robust_finals,
        "maximum_robust_center_final_shift_fraction": maximum_robust_center_shift,
        "provenance_counts": dict(sorted(Counter(row["provenance"] for row in selected).items())),
    }


def main() -> None:
    rows = load_active_rows()
    forecast = json.loads(FORECAST.read_text(encoding="utf-8"))
    model_rows = {row["name"]: row for row in forecast["models"]}
    if not set(TARGETS).issubset(model_rows):
        raise ValueError("Forecast contract lacks a required crowd target")
    summaries = [
        target_summary(model, rows, float(model_rows[model]["currentEvidenceT"]))
        for model in TARGETS
    ]
    for summary in summaries:
        observed = float(model_rows[summary["model"]]["currentFinalT"])
        if not math.isclose(
            observed, summary["displayed_final_t"], rel_tol=0, abs_tol=1e-10
        ):
            raise ValueError(f"Crowd arithmetic mismatch for {summary['model']}")

    paired: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["model"] in TARGETS:
            paired[row["contributor"]][row["model"]] = float(row["point_t"])
    complete = [values for values in paired.values() if set(values) == set(TARGETS)]
    log_correlation = correlation(
        [math.log(values[TARGETS[0]]) for values in complete],
        [math.log(values[TARGETS[1]]) for values in complete],
    )

    result = {
        "generated_on": "2026-07-31",
        "role": "robustness/dependence audit; never an outcome-calibrated likelihood",
        "aggregation_policy": {
            "active_revision_rule": "exclude every forecast_id named by a later supersedes field",
            "range_point_rule": "use separately stated central_t; otherwise geometric midpoint sqrt(low*high)",
            "contributor_weighting": "equal weight per active contributor in log space",
            "final_rule": "50% evidence and 50% crowd in log space for Fable/Sol",
        },
        "targets": summaries,
        "cross_target_dependence": {
            "paired_contributors": len(complete),
            "pearson_correlation_of_log_points": log_correlation,
            "interpretation": "The same forecasters tend to move both targets together, so the two crowd pools are not independent replications.",
        },
        "decision": {
            "center_is_single_forecaster_robust": all(
                row["leave_one_contributor_out"][
                    "maximum_absolute_final_shift_fraction"
                ]
                < 0.10
                for row in summaries
            ),
            "crowd_is_independent_calibration_evidence": False,
            "narrow_predictive_intervals": False,
            "change_user_selected_crowd_weight": False,
            "reason": "Jackknife and robust centers show that no single contributor controls the final center. The forecasts remain relayed, unscored, mutually correlated judgments and therefore cannot supply a calibrated likelihood or narrower interval.",
        },
        "limitations": [
            "The ledger records statements relayed in this project, not a randomized or blinded elicitation protocol.",
            "Forecasters may share public information or may have seen related parameter estimates.",
            "No hidden target has disclosed its parameter count, so crowd accuracy is not yet outcome-calibrated.",
        ],
        "source_files": {
            str(LEDGER.relative_to(ROOT)): sha256(LEDGER),
            str(FORECAST.relative_to(ROOT)): sha256(FORECAST),
        },
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "result": str(RESULT.relative_to(ROOT)),
                "paired_contributors": len(complete),
                "log_point_correlation": log_correlation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
