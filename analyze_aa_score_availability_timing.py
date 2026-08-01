#!/usr/bin/env python3
"""Measure the effect of exact AA score-publication dates on validation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

import run_parameter_backtest as backtest
from aa_score_availability import (
    LEDGER_PATH,
    RAW_PATH,
    aa_prediction_information_date,
    aa_score_availability_verified,
    load_aa_score_availability,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "aa_score_availability_timing_audit_2026-07-31.json"
PREDICTIONS = OUT / "aa_score_availability_timing_changes_2026-07-31.csv"
REPORT = ROOT / "AA_SCORE_AVAILABILITY_TIMING_AUDIT.md"
DETAILED_PANEL = OUT / "aa_detailed_parameter_panel_2026-07-18.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_predictions(
    panels: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    selected["AA"] = backtest._backtest(
        "AA",
        panels["AA"],
        "score_date",
        ("score", "date"),
        16,
        6,
        True,
        "score",
    )
    eci_score = backtest._backtest(
        "ECI", panels["ECI"], "score_only", ("score",), 20, 6, True, "score"
    )
    eci_date = backtest._backtest(
        "ECI",
        panels["ECI"],
        "score_date",
        ("score", "date"),
        20,
        6,
        True,
        "score",
    )
    selected["ECI"] = backtest._blend_predictions(
        "ECI", "blend_60_score_40_score_date", eci_score, eci_date, 0.60
    )
    selected["No-CoT"] = backtest._backtest(
        "No-CoT",
        panels["No-CoT"],
        "log_horizon_date_moe",
        ("log_horizon", "date", "moe"),
        12,
        3,
        True,
        "log_horizon",
    )
    selected["Compute"] = backtest._backtest(
        "Compute",
        panels["Compute"],
        "log_compute_date",
        ("log_compute", "date"),
        100,
        20,
        True,
        "log_compute",
    )
    ensemble = backtest._current_ensemble(
        selected,
        equal_weight=False,
        parameter_registry=backtest._load_parameter_truth_registry(),
    )
    return selected, ensemble


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return backtest._metric_summary(rows)


def scope(rows: list[dict[str, Any]], frontier: bool) -> list[dict[str, Any]]:
    return (
        [row for row in rows if float(row["frontier_signal_rank"]) >= 0.90]
        if frontier
        else rows
    )


def compare_rows(
    baseline: list[dict[str, Any]], corrected: list[dict[str, Any]], label: str
) -> list[dict[str, Any]]:
    left = {(row["release_date"], row["model"]): row for row in baseline}
    right = {(row["release_date"], row["model"]): row for row in corrected}
    output = []
    for key in sorted(left.keys() | right.keys()):
        before = left.get(key)
        after = right.get(key)
        output.append(
            {
                "comparison": label,
                "release_date": key[0],
                "model": key[1],
                "prediction_information_date": (
                    after or before
                ).get("prediction_information_date", key[0]),
                "baseline_eligible": before is not None,
                "corrected_eligible": after is not None,
                "baseline_predicted_b": before["predicted_b"] if before else "",
                "corrected_predicted_b": after["predicted_b"] if after else "",
                "actual_b": (after or before)["actual_b"],
                "baseline_log10_error": before["log10_error"] if before else "",
                "corrected_log10_error": after["log10_error"] if after else "",
                "absolute_log10_error_change": (
                    abs(float(after["log10_error"]))
                    - abs(float(before["log10_error"]))
                    if before and after
                    else ""
                ),
            }
        )
    return output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ledger = load_aa_score_availability()
    panels, _ = backtest._load_panels()
    corrected_panels = deepcopy(panels)
    baseline_panels = deepcopy(panels)
    for row in baseline_panels["AA"]:
        row["aa_score_available_date"] = row["release_date"]
        row["aa_score_availability_verified"] = False

    baseline_selected, baseline_ensemble = selected_predictions(baseline_panels)
    corrected_selected, corrected_ensemble = selected_predictions(corrected_panels)

    live_aa = corrected_panels["AA"]
    verified = [row for row in live_aa if aa_score_availability_verified(row)]
    delayed = [
        row
        for row in verified
        if aa_prediction_information_date(row) > row["release_date"]
    ]
    lags = [
        (
            date.fromisoformat(aa_prediction_information_date(row))
            - date.fromisoformat(row["release_date"])
        ).days
        for row in delayed
    ]

    detailed = read_csv(DETAILED_PANEL)
    detailed_verified = [
        row
        for row in detailed
        if str(row.get("aa_score_availability_verified", "")).lower() == "true"
    ]
    detailed_delayed = [
        row
        for row in detailed_verified
        if row["aa_score_available_date"] > row["release_date"]
    ]

    results = {}
    for name, baseline_rows, corrected_rows in (
        ("aa", baseline_selected["AA"], corrected_selected["AA"]),
        ("available_component_ensemble", baseline_ensemble, corrected_ensemble),
    ):
        results[name] = {
            "all": {
                "release_order_baseline": metrics(scope(baseline_rows, False)),
                "score_timing_corrected": metrics(scope(corrected_rows, False)),
            },
            "frontier_like": {
                "release_order_baseline": metrics(scope(baseline_rows, True)),
                "score_timing_corrected": metrics(scope(corrected_rows, True)),
            },
        }

    changes = compare_rows(
        baseline_selected["AA"], corrected_selected["AA"], "AA"
    ) + compare_rows(
        baseline_ensemble, corrected_ensemble, "Available-components ensemble"
    )
    write_csv(PREDICTIONS, changes)

    payload = {
        "generated_on": "2026-07-31",
        "question": "How much does exact AA score-publication timing change chronological parameter validation?",
        "method": {
            "baseline": "nominal release ordering plus the existing parameter-label timing correction",
            "corrected": "first exact AA changelog modelAdded date with non-null Intelligence Index for an exact slug; unmatched rows retain release ordering",
            "date_feature": "nominal model release date in both variants",
            "test_cutoff": "AA score-publication date when verified, otherwise release date",
            "training_cutoff": "max(release, parameter-label date, verified AA score date)",
            "vintage_limit": "The score value itself remains the pinned current AA snapshot; this corrects availability ordering, not historical index-version revisions.",
        },
        "coverage": {
            "raw_changelog_events_returned": ledger["summary"][
                "total_changelog_events"
            ],
            "raw_changelog_events_api_reported": ledger["summary"][
                "api_reported_total_events"
            ],
            "api_total_reconciles": ledger["summary"]["api_total_reconciles"],
            "verified_score_slugs": len(ledger["records"]),
            "live_aa_rows": len(live_aa),
            "live_aa_verified": len(verified),
            "live_aa_fallback": len(live_aa) - len(verified),
            "live_aa_verified_after_release": len(delayed),
            "live_aa_delay_days_median": float(np.median(lags)) if lags else None,
            "live_aa_delay_days_max": max(lags) if lags else None,
            "detailed_aa_rows": len(detailed),
            "detailed_aa_verified": len(detailed_verified),
            "detailed_aa_fallback": len(detailed) - len(detailed_verified),
            "detailed_aa_verified_after_release": len(detailed_delayed),
        },
        "validation_impact": results,
        "changed_rows": {
            "aa_common_predictions_changed": sum(
                row["baseline_eligible"]
                and row["corrected_eligible"]
                and not math.isclose(
                    float(row["baseline_predicted_b"]),
                    float(row["corrected_predicted_b"]),
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                for row in changes
                if row["comparison"] == "AA"
            ),
            "aa_newly_eligible_predictions": sum(
                not row["baseline_eligible"] and row["corrected_eligible"]
                for row in changes
                if row["comparison"] == "AA"
            ),
            "ensemble_common_predictions_changed": sum(
                row["baseline_eligible"]
                and row["corrected_eligible"]
                and not math.isclose(
                    float(row["baseline_predicted_b"]),
                    float(row["corrected_predicted_b"]),
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                for row in changes
                if row["comparison"] == "Available-components ensemble"
            ),
            "ensemble_newly_eligible_predictions": sum(
                not row["baseline_eligible"] and row["corrected_eligible"]
                for row in changes
                if row["comparison"] == "Available-components ensemble"
            ),
        },
        "decision": {
            "change_current_fit": False,
            "change_live_weights": False,
            "change_headline_centers": False,
            "change_validation_and_uncertainty": True,
            "reason": "Publication timing changes only what was available in historical folds. Every score and current-fit row is unchanged, but the corrected held-out residual ledger must feed precision calibration.",
        },
        "source_files": {
            str(LEDGER_PATH.relative_to(ROOT)): sha256(LEDGER_PATH),
            str(RAW_PATH.relative_to(ROOT)): sha256(RAW_PATH),
            str(backtest.REGRESSION_PATH.relative_to(ROOT)): sha256(
                backtest.REGRESSION_PATH
            ),
            str(DETAILED_PANEL.relative_to(ROOT)): sha256(DETAILED_PANEL),
        },
        "outputs": {
            "prediction_changes": str(PREDICTIONS.relative_to(ROOT)),
            "result": str(RESULT.relative_to(ROOT)),
        },
    }
    RESULT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    ensemble = results["available_component_ensemble"]
    text = f"""# Artificial Analysis score-publication timing audit

Status: exact changelog events applied to chronological validation; current fits and centers unchanged.

Artificial Analysis release dates are retained as the algorithmic-progress feature. For exact slugs with a dated non-null Intelligence Index event, chronological tests now use the score-publication date as the prediction cutoff and `max(release, parameter-label date, score date)` as training eligibility.

## Coverage

- Official API returned {payload['coverage']['raw_changelog_events_returned']:,} events while advertising {payload['coverage']['raw_changelog_events_api_reported']:,}; this unresolved source discrepancy is preserved explicitly.
- {len(ledger['records'])} unique score-publication slugs are hash-pinned.
- Live AA panel: {len(verified)}/{len(live_aa)} verified, {len(delayed)} later than nominal release; {len(live_aa) - len(verified)} retain explicit release-date fallback.
- Detailed AA panel: {len(detailed_verified)}/{len(detailed)} verified, {len(detailed_delayed)} later than nominal release.

## Validation effect

| Scope | Release-order baseline | Score-timing corrected |
|---|---:|---:|
| AA median error | {results['aa']['all']['release_order_baseline']['median_multiplicative_error']:.2f}x | {results['aa']['all']['score_timing_corrected']['median_multiplicative_error']:.2f}x |
| Frontier AA median error | {results['aa']['frontier_like']['release_order_baseline']['median_multiplicative_error']:.2f}x | {results['aa']['frontier_like']['score_timing_corrected']['median_multiplicative_error']:.2f}x |
| Ensemble median error | {ensemble['all']['release_order_baseline']['median_multiplicative_error']:.2f}x | {ensemble['all']['score_timing_corrected']['median_multiplicative_error']:.2f}x |
| Frontier ensemble median error | {ensemble['frontier_like']['release_order_baseline']['median_multiplicative_error']:.2f}x | {ensemble['frontier_like']['score_timing_corrected']['median_multiplicative_error']:.2f}x |

This is a validation correction, not new capability evidence. The published current AA score values still come from the July 31 snapshot, so the exercise remains pseudo-chronological with respect to index-version changes and score revisions.
"""
    REPORT.write_text(text, encoding="utf-8")
    print(json.dumps({"coverage": payload["coverage"], "validation_impact": results}, indent=2))


if __name__ == "__main__":
    main()
