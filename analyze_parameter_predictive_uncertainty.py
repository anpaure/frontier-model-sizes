#!/usr/bin/env python3
"""Build empirical frontier parameter-count error bands from historical errors.

The central forecasts combine several correlated capability proxies.  This audit
does not pretend their coefficient/bootstrap uncertainty is the full predictive
uncertainty.  Instead it uses the realized errors from the strictly earlier,
model-lineage-held-out and whole-developer-held-out ensemble backtests.

For a new developer, each canonical developer contributes exactly one score:
its latest-date eligible absolute log10 error, with the worst residual retained
when a developer released multiple checkpoints on that date.  Each published
factor is the conservative envelope across the two holdout specifications.

The familiar ceil((D + 1) * coverage) order statistic is retained as a clear,
finite-sample summary.  These are empirical prequential error bands, not formal
split-conformal coverage guarantees: the fits expand over time, the benchmark
snapshots are mostly current rather than release-vintage, and the developer
clusters are not exchangeable with a new closed frontier lab.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from analyze_parameter_vintage_sensitivity import (
    developer_lookup,
    fallback_developer,
    selected_developer_predictions,
    with_developers,
)
from k3_primary_evidence import K3_EVIDENCE_PATH
from run_parameter_backtest import (
    REGRESSION_PATH,
    UNIFIED_PATH,
    _current_ensemble,
    _load_panels,
    _load_parameter_truth_registry,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
BACKTEST = OUT / "frontier_parameter_chronological_backtest_2026-07-17.json"
SITE_DATA = ROOT / "site/public/data/forecast-model.json"
OPUS_5_EVIDENCE = ROOT / "sources/claude_opus_5_evidence_2026-07-31.json"
RESULT = OUT / "frontier_parameter_predictive_uncertainty_2026-07-18.json"
LEDGER = OUT / "frontier_parameter_predictive_uncertainty_calibration_2026-07-18.csv"
SITE_OUTPUT = ROOT / "site/public/data/predictive-uncertainty.json"
K3_EFFICIENCY_PRIOR = OUT / "k3_efficiency_prior_2026-08-01.json"
K3_EFFICIENCY_DRAWS = OUT / "k3_efficiency_prior_cap_draws_2026-08-01.csv"

LEVELS = (0.50, 0.80, 0.90)
MIN_SEQUENTIAL_CALIBRATION_FAMILIES = 5
PRIMARY_COHORT = "frontier_like"
EXPECTED_ENSEMBLE_CHECKPOINTS = 44
TARGETS = {
    "claude-fable-5": {"developer": "Anthropic", "release_date": "2026-06-09"},
    "gpt-56-sol": {"developer": "OpenAI", "release_date": "2026-07-09"},
}

LINEAGE_HOLDOUT = "model_lineage_holdout"
DEVELOPER_HOLDOUT = "whole_developer_holdout"

K3_EFFICIENCY_DRAW_FIELDS = {
    "claude-fable-5": "fable_ordered_reference_t",
    "gpt-56-sol": "sol_ordered_reference_t",
    "claude-opus-5": "opus5_ordered_reference_t",
}
EFFICIENCY_STRENGTH_GRID = tuple(range(0, 101, 5))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_k3_efficiency_reference() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reference = json.loads(K3_EFFICIENCY_PRIOR.read_text(encoding="utf-8"))
    decision = reference["decision"]
    if (
        not decision["apply_center_preserving_upper_tail_projection"]
        or decision["change_point_centers"]
        or decision["incremental_point_center_weight"] != 0
        or decision["change_crowd_weight"]
        or decision["crowd_weight_for_fable_and_sol"] != 0.5
        or decision["rejected_nonlinear_eci_weight"] != 0
        or decision["literal_constraint_enforced_when_reference_below_center"]
    ):
        raise ValueError("K3 efficiency reference violates the center-preserving policy")
    with K3_EFFICIENCY_DRAWS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(reference["method"]["draws"]):
        raise ValueError("K3 efficiency draw ledger does not match its audit")
    if sha256(K3_EFFICIENCY_DRAWS) != reference["outputs"]["draw_ledger_sha256"]:
        raise ValueError("K3 efficiency draw ledger hash does not match its audit")
    draws = {
        model_id: np.asarray([float(row[field]) for row in rows], dtype=float)
        for model_id, field in K3_EFFICIENCY_DRAW_FIELDS.items()
    }
    if any(np.any(values <= 0) for values in draws.values()):
        raise ValueError("K3 efficiency reference draws must be positive")
    return reference, draws


def upper_quantile_draws(
    center: float,
    intervals: dict[str, dict[str, Any]],
    size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a deterministic upper-half quantile sample.

    The interpolation passes through the three published upper endpoints
    exactly in the continuum: q75/q90/q95 correspond to the 50/80/90 central
    bands. This lets the center-preserving structural projection alter only the
    upper half while the empirical intervals remain authoritative and unchanged.
    """

    probabilities = np.asarray([0.50, 0.75, 0.90, 0.95, 0.9995])
    factors = np.asarray(
        [
            1.0,
            float(intervals["50"]["multiplicative_factor"]),
            float(intervals["80"]["multiplicative_factor"]),
            float(intervals["90"]["multiplicative_factor"]),
            0.0,
        ]
    )
    tail_log_slope = (
        math.log(factors[3]) - math.log(factors[2])
    ) / (probabilities[3] - probabilities[2])
    factors[4] = math.exp(
        math.log(factors[3])
        + tail_log_slope * (probabilities[4] - probabilities[3])
    )
    upper_probability = 0.50 + 0.4995 * (
        np.arange(size, dtype=float) + 0.5
    ) / size
    rng = np.random.default_rng(seed)
    rng.shuffle(upper_probability)
    log_factor = np.interp(upper_probability, probabilities, np.log(factors))
    raw_upper = center * np.exp(log_factor)
    application_uniform = rng.random(size)
    return raw_upper, application_uniform


def k3_efficiency_projection(
    model_id: str,
    center: float,
    intervals: dict[str, dict[str, Any]],
    reference_draws: np.ndarray,
    reference_metadata: dict[str, Any],
    default_strength: float,
) -> dict[str, Any]:
    raw_upper, application_uniform = upper_quantile_draws(
        center, intervals, len(reference_draws), 20260810 + sum(map(ord, model_id))
    )
    # This is intentionally a center-preserving winsorization, not literal
    # conditioning. If a reference draw lies below the published evidence
    # center, the center overrides it and that conflict is reported below.
    projected_upper = np.minimum(raw_upper, np.maximum(center, reference_draws))
    quantile_by_level = {"50": 0.50, "80": 0.80, "90": 0.90}
    strength_grid: dict[str, Any] = {}
    for percent in EFFICIENCY_STRENGTH_GRID:
        strength = percent / 100
        projected = np.where(
            application_uniform < strength, projected_upper, raw_upper
        )
        level_rows: dict[str, Any] = {}
        for level, upper_half_quantile in quantile_by_level.items():
            raw = intervals[level]
            high = (
                float(raw["high_t"])
                if percent == 0
                else float(np.quantile(projected, upper_half_quantile))
            )
            high = max(center, min(float(raw["high_t"]), high))
            level_rows[level] = {
                "low_t": float(raw["low_t"]),
                "high_t": high,
                "raw_low_t": float(raw["low_t"]),
                "raw_high_t": float(raw["high_t"]),
                "upper_reduction_t": float(raw["high_t"]) - high,
                "upper_reduction_fraction": 1 - high / float(raw["high_t"]),
            }
        strength_grid[str(percent)] = level_rows

    default_percent = int(round(100 * default_strength))
    if str(default_percent) not in strength_grid:
        raise ValueError("Default K3 efficiency strength is absent from the grid")
    return {
        "status": "live center-preserving upper-tail projection; point centers unchanged",
        "default_projection_strength": default_strength,
        "default_projection_percent": default_percent,
        "pooled_reference_quantiles_t": reference_metadata[
            "pooled_parameter_equivalent_reference_t"
        ],
        "raw_unordered_pooled_reference_quantiles_t": reference_metadata[
            "raw_unordered_pooled_reference_t"
        ],
        "order_projection_applied": reference_metadata["order_projection_applied"],
        "binding_probability_against_raw_upper_draws": float(
            np.mean(reference_draws < raw_upper)
        ),
        "center_override_probability": float(np.mean(reference_draws < center)),
        "literal_reference_violation_probability_at_100pct": float(
            np.mean(reference_draws < center)
        ),
        "projected_intervals": strength_grid[str(default_percent)],
        "strength_grid": strength_grid,
        "transformation": "min(raw upper draw, max(evidence center, pooled K3-relative reference draw))",
        "literal_conditioning": False,
        "literal_ceiling_enforced_when_reference_below_center": False,
        "lower_tail_changed": False,
        "point_center_changed": False,
        "formal_coverage_guarantee": False,
    }


def latest_per_group(
    rows: list[dict[str, Any]], group_field: str
) -> list[dict[str, Any]]:
    """Return one conservative latest-date score per group.

    A developer can release multiple checkpoints on the same day.  Selecting
    the lexicographically last model made the result depend on its name (Meta's
    Llama 4 Scout happened to replace Maverick).  At the latest observed date,
    retain the checkpoint with the largest absolute error and use model name
    only as a deterministic final tie-breaker.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_field])].append(row)
    latest: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        latest_date = max(str(row["release_date"]) for row in group_rows)
        same_day = [
            row for row in group_rows if str(row["release_date"]) == latest_date
        ]
        latest.append(
            max(
                same_day,
                key=lambda row: (
                    abs(float(row["log10_error"])),
                    str(row["model"]),
                ),
            )
        )
    return sorted(latest, key=lambda item: (item["release_date"], item["model"]))


def order_statistic_factor(
    rows: list[dict[str, Any]], coverage: float, group_field: str = "developer"
) -> dict[str, Any]:
    calibration = latest_per_group(rows, group_field)
    if not calibration:
        raise ValueError("No calibration groups")
    scores = sorted(abs(float(row["log10_error"])) for row in calibration)
    raw_rank = math.ceil((len(scores) + 1) * coverage)
    supported = raw_rank <= len(scores)
    rank = raw_rank if supported else None
    factor = 10 ** scores[rank - 1] if rank is not None else None
    return {
        "coverage": coverage,
        "group_field": group_field,
        "calibration_groups": len(scores),
        # Compatibility field consumed by the site. It now counts developers
        # for the primary intervals rather than model-series lineages.
        "calibration_families": len(scores),
        "calibration_developers": len(scores) if group_field == "developer" else None,
        "order_statistic_rank": rank,
        "raw_order_statistic_rank": raw_rank,
        "supported": supported,
        "unsupported_reason": (
            None
            if supported
            else "requested order-statistic rank exceeds calibration groups"
        ),
        # Retained as a compatibility/audit field. A clipped finite factor is
        # deliberately no longer emitted.
        "upper_rank_clipped": not supported,
        "multiplicative_factor": factor,
    }


def conservative_factor_envelope(
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    supported = {
        name: row
        for name, row in candidates.items()
        if row.get("supported") and row.get("multiplicative_factor") is not None
    }
    candidate_factors = {
        name: row.get("multiplicative_factor") for name, row in candidates.items()
    }
    if not supported:
        first = next(iter(candidates.values()))
        return {
            **first,
            "supported": False,
            "order_statistic_rank": None,
            "multiplicative_factor": None,
            "selected_holdout_spec": None,
            "candidate_multiplicative_factors": candidate_factors,
            "unsupported_reason": "no holdout specification supports the requested rank",
            "conservative_envelope_across_holdout_specs": True,
        }
    selected_name, selected = max(
        supported.items(), key=lambda item: float(item[1]["multiplicative_factor"])
    )
    return {
        **selected,
        "selected_holdout_spec": selected_name,
        "candidate_multiplicative_factors": candidate_factors,
        "conservative_envelope_across_holdout_specs": True,
    }


def cohort_filters() -> dict[str, Callable[[dict[str, Any]], bool]]:
    return {
        "all_ensemble": lambda row: True,
        "frontier_like": lambda row: float(row["frontier_signal_rank"]) >= 0.90,
        "frontier_moe_reasoning": lambda row: (
            float(row["frontier_signal_rank"]) >= 0.90
            and row["release_date"] >= "2025-01-01"
            and float(row["predicted_b"]) >= 100.0
            and row.get("moe") == 1
            and row.get("reasoning") == 1
            and not row.get("moe_flag_conflict")
            and not row.get("reasoning_flag_conflict")
        ),
    }


def with_canonical_developers(
    rows: list[dict[str, Any]], lookup: dict[str, str]
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        developer = lookup.get(row["normalized_model"]) or fallback_developer(
            row["model"], row["family"]
        )
        output.append({**row, "developer": developer})
    return output


def build_whole_developer_holdout_ensemble(
    lineage_ensemble: list[dict[str, Any]], lookup: dict[str, str]
) -> list[dict[str, Any]]:
    panels, _ = _load_panels()
    developer_panels = with_developers(panels, lookup)
    selected = selected_developer_predictions(developer_panels)
    raw = _current_ensemble(
        selected,
        equal_weight=False,
        parameter_registry=_load_parameter_truth_registry(),
    )
    lineage_by_key = {row["normalized_model"]: row for row in lineage_ensemble}
    if len(raw) != EXPECTED_ENSEMBLE_CHECKPOINTS or set(lineage_by_key) != {
        row["normalized_model"] for row in raw
    }:
        raise ValueError("Whole-developer and model-lineage ensembles do not reconcile")
    output = []
    for row in raw:
        matched = lineage_by_key[row["normalized_model"]]
        developer = str(row["family"])
        if (
            row["release_date"] != matched["release_date"]
            or not math.isclose(float(row["actual_b"]), float(matched["actual_b"]))
            or not row["test_family_excluded"]
            or row["train_max_date"]
            >= (row.get("prediction_information_date") or row["release_date"])
        ):
            raise ValueError(
                f"Whole-developer prediction failed reconciliation: {row['model']}"
            )
        output.append(
            {
                **row,
                # The refit temporarily uses developer as the fitting family.
                # Restore the model-series lineage for reporting and retain the
                # held-out fitting group explicitly.
                "family": matched["family"],
                "developer": developer,
            }
        )
    return output


def chronological_coverage(
    cohort: list[dict[str, Any]], coverage: float, group_field: str = "developer"
) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    for test in sorted(cohort, key=lambda row: (row["release_date"], row["model"])):
        calibration = [
            row
            for row in cohort
            if row["release_date"] < test["release_date"]
            and row[group_field] != test[group_field]
        ]
        groups = len({row[group_field] for row in calibration})
        if groups < MIN_SEQUENTIAL_CALIBRATION_FAMILIES:
            continue
        interval = order_statistic_factor(calibration, coverage, group_field)
        factor = interval["multiplicative_factor"]
        covered = (
            float(test["multiplicative_error"]) <= float(factor)
            if factor is not None
            else None
        )
        tests.append(
            {
                "model": test["model"],
                "release_date": test["release_date"],
                "family": test["family"],
                "developer": test["developer"],
                "calibration_groups": groups,
                "calibration_families": groups,
                "calibration_developers": groups,
                "factor": factor,
                "supported": interval["supported"],
                "actual_error_factor": float(test["multiplicative_error"]),
                "covered": covered,
            }
        )
    supported_tests = [row for row in tests if row["supported"]]
    by_developer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in supported_tests:
        by_developer[row["developer"]].append(row)
    developer_rates = [
        sum(bool(row["covered"]) for row in rows) / len(rows)
        for rows in by_developer.values()
    ]
    latest_developer_tests: list[dict[str, Any]] = []
    for rows in by_developer.values():
        latest_date = max(row["release_date"] for row in rows)
        same_day = [row for row in rows if row["release_date"] == latest_date]
        # A same-day developer cluster counts as covered only when every
        # checkpoint released on that date is covered.
        representative = max(
            same_day,
            key=lambda row: (row["actual_error_factor"] / row["factor"], row["model"]),
        )
        latest_developer_tests.append(representative)

    raw_covered = sum(bool(row["covered"]) for row in supported_tests)
    latest_covered = sum(bool(row["covered"]) for row in latest_developer_tests)
    raw_rate = raw_covered / len(supported_tests) if supported_tests else None
    developer_balanced_rate = (
        sum(developer_rates) / len(developer_rates) if developer_rates else None
    )
    latest_rate = (
        latest_covered / len(latest_developer_tests)
        if latest_developer_tests
        else None
    )
    return {
        "nominal_coverage": coverage,
        "group_field": group_field,
        "candidate_tests": len(tests),
        "eligible_tests": len(supported_tests),
        "unsupported_tests": len(tests) - len(supported_tests),
        "test_families": len({row["family"] for row in supported_tests}),
        "test_developers": len({row["developer"] for row in supported_tests}),
        # Compatibility alias: this is explicitly the raw checkpoint rate.
        "observed_coverage": raw_rate,
        "raw_checkpoint_coverage": {
            "covered": raw_covered,
            "tests": len(supported_tests),
            "rate": raw_rate,
        },
        "developer_balanced_coverage": {
            "developers": len(developer_rates),
            "rate": developer_balanced_rate,
        },
        "latest_developer_coverage": {
            "covered": latest_covered,
            "developers": len(latest_developer_tests),
            "rate": latest_rate,
            "same_day_policy": "least-covered checkpoint at the latest supported date",
        },
        "geometric_mean_interval_factor": (
            math.exp(
                sum(math.log(float(row["factor"])) for row in supported_tests)
                / len(supported_tests)
            )
            if supported_tests
            else None
        ),
        "tests": tests,
    }


def main() -> None:
    backtest = json.loads(BACKTEST.read_text(encoding="utf-8"))
    site = json.loads(SITE_DATA.read_text(encoding="utf-8"))
    efficiency_reference, efficiency_draws = load_k3_efficiency_reference()
    efficiency_targets = {
        row["model_id"]: row for row in efficiency_reference["targets"]
    }
    if set(TARGETS) - efficiency_targets.keys():
        raise ValueError("K3 efficiency reference is missing a required uncertainty target")
    default_efficiency_strength = float(
        efficiency_reference["decision"]["default_projection_strength"]
    )
    opus_5_evidence = json.loads(OPUS_5_EVIDENCE.read_text(encoding="utf-8"))
    opus_5_identity = opus_5_evidence["identity"]
    if (
        opus_5_identity["canonical_name"] != "Claude Opus 5"
        or opus_5_identity["parameter_disclosed"]
        or opus_5_identity["base_identity_policy"] != "unique_base"
    ):
        raise ValueError("Opus 5 uncertainty target must be a distinct undisclosed base")
    targets = {
        **TARGETS,
        "claude-opus-5": {
            "developer": "Anthropic",
            "release_date": opus_5_identity["release_date"],
        },
    }
    raw_ensemble = backtest["ensemble_predictions"]
    if (
        len(raw_ensemble) != EXPECTED_ENSEMBLE_CHECKPOINTS
        or len({row["normalized_model"] for row in raw_ensemble})
        != EXPECTED_ENSEMBLE_CHECKPOINTS
    ):
        raise ValueError(
            f"Expected {EXPECTED_ENSEMBLE_CHECKPOINTS} unique held-out ensemble checkpoints"
        )
    if not all(
        row["train_max_date"]
        < (row.get("prediction_information_date") or row["release_date"])
        for row in raw_ensemble
    ):
        raise ValueError("Chronology violation in uncertainty source")
    if not all(row["test_family_excluded"] for row in raw_ensemble):
        raise ValueError("Family-holdout violation in uncertainty source")

    lookup = developer_lookup()
    lineage_ensemble = with_canonical_developers(raw_ensemble, lookup)
    developer_ensemble = build_whole_developer_holdout_ensemble(
        lineage_ensemble, lookup
    )
    ensembles = {
        LINEAGE_HOLDOUT: lineage_ensemble,
        DEVELOPER_HOLDOUT: developer_ensemble,
    }

    filters = cohort_filters()
    cohorts: dict[str, Any] = {}
    ledger_rows: list[dict[str, Any]] = []
    for name, predicate in filters.items():
        rows_by_spec = {
            spec: [row for row in rows if predicate(row)]
            for spec, rows in ensembles.items()
        }
        lineage_rows = rows_by_spec[LINEAGE_HOLDOUT]
        families = len({row["family"] for row in lineage_rows})
        developers = len({row["developer"] for row in lineage_rows})
        holdout_specs: dict[str, Any] = {}
        for spec, rows in rows_by_spec.items():
            holdout_specs[spec] = {
                "rows": len(rows),
                "families": len({row["family"] for row in rows}),
                "developers": len({row["developer"] for row in rows}),
                "intervals": {
                    str(int(level * 100)): order_statistic_factor(
                        rows, level, "developer"
                    )
                    for level in LEVELS
                },
                "chronological_coverage": {
                    str(int(level * 100)): chronological_coverage(
                        rows, level, "developer"
                    )
                    for level in LEVELS
                },
            }
        intervals = {
            str(int(level * 100)): conservative_factor_envelope(
                {
                    spec: block["intervals"][str(int(level * 100))]
                    for spec, block in holdout_specs.items()
                }
            )
            for level in LEVELS
        }
        lineage_intervals = {
            str(int(level * 100)): order_statistic_factor(
                lineage_rows, level, "family"
            )
            for level in LEVELS
        }
        cohorts[name] = {
            "rows": len(lineage_rows),
            "families": families,
            "developers": developers,
            "latest_per_family_rows": len(
                latest_per_group(lineage_rows, "family")
            ),
            "latest_per_developer_rows": len(
                latest_per_group(lineage_rows, "developer")
            ),
            "intervals": intervals,
            "lineage_intervals": lineage_intervals,
            # Compatibility path: this remains the model-lineage-held-out
            # sequence, while both specifications are now published below.
            "chronological_coverage": holdout_specs[LINEAGE_HOLDOUT][
                "chronological_coverage"
            ],
            "holdout_specs": holdout_specs,
            "eligible_for_primary": name == PRIMARY_COHORT,
            "target_specificity_status": (
                "insufficient: fewer than 8 independent developers"
                if name == "frontier_moe_reasoning" and developers < 8
                else "usable"
            ),
        }
        for spec, rows in rows_by_spec.items():
            latest_family = latest_per_group(rows, "family")
            latest_developer = latest_per_group(rows, "developer")
            for row in rows:
                ledger_rows.append(
                    {
                        "cohort": name,
                        "holdout_spec": spec,
                        "release_date": row["release_date"],
                        "model": row["model"],
                        "family": row["family"],
                        "developer": row["developer"],
                        "moe": row.get("moe"),
                        "reasoning": row.get("reasoning"),
                        "frontier_signal_rank": row["frontier_signal_rank"],
                        "actual_b": row["actual_b"],
                        "predicted_b": row["predicted_b"],
                        "absolute_error_factor": row["multiplicative_error"],
                        "latest_in_family_for_cohort": row in latest_family,
                        "latest_in_developer_for_cohort": row in latest_developer,
                    }
                )

    primary_by_spec = {
        spec: [row for row in rows if filters[PRIMARY_COHORT](row)]
        for spec, rows in ensembles.items()
    }
    models_by_id = {row["id"]: row for row in site["models"]}
    target_intervals: list[dict[str, Any]] = []
    for model_id, target in targets.items():
        model = models_by_id[model_id]
        center = float(model["currentEvidenceT"])
        final_center = float(model["currentFinalT"])
        calibrations_by_spec: dict[str, dict[str, list[dict[str, Any]]]] = {}
        target_holdout_specs: dict[str, Any] = {}
        for spec, primary in primary_by_spec.items():
            chronological_calibration = [
                row
                for row in primary
                if row["release_date"] < target["release_date"]
                and row["developer"] != target["developer"]
            ]
            current_calibration = [
                row for row in primary if row["developer"] != target["developer"]
            ]
            calibrations_by_spec[spec] = {
                "chronological": chronological_calibration,
                "current": current_calibration,
            }
            spec_intervals: dict[str, Any] = {}
            chronological_intervals: dict[str, Any] = {}
            current_intervals: dict[str, Any] = {}
            for level in LEVELS:
                key = str(int(level * 100))
                chronological = order_statistic_factor(
                    chronological_calibration, level, "developer"
                )
                current = order_statistic_factor(
                    current_calibration, level, "developer"
                )
                chronological_intervals[key] = chronological
                current_intervals[key] = current
                if level == 0.50:
                    calibrated = {
                        **chronological,
                        "calibration_scope": "target_pre_release_developers",
                        "conservative_max_of_current_and_target_chronological": False,
                    }
                else:
                    date_envelope = conservative_factor_envelope(
                        {
                            "target_pre_release_developers": chronological,
                            "current_frontier_developers": current,
                        }
                    )
                    calibrated = {
                        **date_envelope,
                        "calibration_scope": date_envelope[
                            "selected_holdout_spec"
                        ],
                        "conservative_max_of_current_and_target_chronological": True,
                    }
                spec_intervals[key] = calibrated
            target_holdout_specs[spec] = {
                "calibration_rows": len(current_calibration),
                "calibration_developers": len(
                    {row["developer"] for row in current_calibration}
                ),
                "target_chronological_calibration_rows": len(
                    chronological_calibration
                ),
                "target_chronological_calibration_developers": len(
                    {row["developer"] for row in chronological_calibration}
                ),
                "intervals": spec_intervals,
                "target_chronological_intervals": chronological_intervals,
                "current_intervals": current_intervals,
            }

        intervals: dict[str, Any] = {}
        chronological_intervals: dict[str, Any] = {}
        for level in LEVELS:
            key = str(int(level * 100))
            envelope = conservative_factor_envelope(
                {
                    spec: block["intervals"][key]
                    for spec, block in target_holdout_specs.items()
                }
            )
            factor = envelope["multiplicative_factor"]
            intervals[key] = {
                **envelope,
                "low_t": center / float(factor) if factor is not None else None,
                "high_t": center * float(factor) if factor is not None else None,
            }
            chronological_intervals[key] = conservative_factor_envelope(
                {
                    spec: block["target_chronological_intervals"][key]
                    for spec, block in target_holdout_specs.items()
                }
            )

        lineage_current = calibrations_by_spec[LINEAGE_HOLDOUT]["current"]
        lineage_intervals = {
            str(int(level * 100)): order_statistic_factor(
                lineage_current, level, "family"
            )
            for level in LEVELS
        }
        compatibility_current = calibrations_by_spec[LINEAGE_HOLDOUT]["current"]
        compatibility_chronological = calibrations_by_spec[LINEAGE_HOLDOUT][
            "chronological"
        ]
        target_intervals.append(
            {
                "model_id": model_id,
                "model": model["name"],
                "release_date": target["release_date"],
                "center_t": center,
                "center_kind": "evidence-model center used by the historical calibration",
                "displayed_final_center_t": final_center,
                "final_over_calibrated_center": final_center / center,
                "calibration_cohort": PRIMARY_COHORT,
                "calibration_rows": len(compatibility_current),
                "calibration_families": len(
                    {row["developer"] for row in compatibility_current}
                ),
                "calibration_developers": len(
                    {row["developer"] for row in compatibility_current}
                ),
                "lineage_calibration_families": len(
                    {row["family"] for row in compatibility_current}
                ),
                "target_chronological_calibration_rows": len(
                    compatibility_chronological
                ),
                "target_chronological_calibration_developers": len(
                    {row["developer"] for row in compatibility_chronological}
                ),
                "intervals": intervals,
                "target_chronological_intervals": chronological_intervals,
                "lineage_intervals": lineage_intervals,
                "holdout_specs": target_holdout_specs,
                "k3_efficiency_projection": k3_efficiency_projection(
                    model_id,
                    center,
                    intervals,
                    efficiency_draws[model_id],
                    efficiency_targets[model_id],
                    default_efficiency_strength,
                ),
            }
        )

    primary_family_count = cohorts[PRIMARY_COHORT]["families"]
    primary_developer_count = cohorts[PRIMARY_COHORT]["developers"]
    diagnostic_family_count = cohorts["frontier_moe_reasoning"]["families"]
    diagnostic_developer_count = cohorts["frontier_moe_reasoning"]["developers"]
    primary_coverage_by_spec = {
        spec: block["chronological_coverage"]
        for spec, block in cohorts[PRIMARY_COHORT]["holdout_specs"].items()
    }
    developer_coverage = primary_coverage_by_spec[DEVELOPER_HOLDOUT]
    result = {
        "generated_on": "2026-08-01",
        "method": {
            "primary_cohort": PRIMARY_COHORT,
            "error_source": (
                "strictly-earlier chronological model-lineage-held-out and "
                "whole-developer-held-out ensemble predictions"
            ),
            "developer_cluster_policy": (
                "one residual per canonical developer: largest absolute error among "
                "checkpoints on that developer's latest eligible release date"
            ),
            "interval_policy": (
                "symmetric multiplicative empirical prequential order-statistic band; "
                "conservative per-level envelope across both holdout specifications"
            ),
            "formal_conformal_coverage_claim": False,
            "unsupported_rank_policy": (
                "return null when ceil((developers + 1) * requested coverage) "
                "exceeds the number of calibration developers; never clip to a finite maximum"
            ),
            "tail_policy": (
                "50% remains target-chronological and descriptive within each holdout "
                "specification; 80%/90% take the maximum of target-chronological and "
                "currently observed factors before the cross-specification envelope"
            ),
            "levels": list(LEVELS),
            "crowd_policy": "crowd forecasts affect the displayed center but are not treated as independent calibration data and do not narrow intervals",
            "k3_efficiency_policy": (
                "retain the empirical intervals unchanged; separately transform only "
                "upper-half draws with a center-preserving winsorization against the pooled "
                f"K3-relative reference at an explicit {100 * default_efficiency_strength:.0f}% projection strength; "
                "this is not literal conditioning"
            ),
        },
        "cohorts": cohorts,
        "targets": target_intervals,
        "decision": {
            "publish_empirical_intervals": True,
            "use_frontier_moe_reasoning_cohort": False,
            "reason": (
                f"The architecture-matched cohort has {diagnostic_family_count} model-series lineages but only "
                f"{diagnostic_developer_count} canonical developers; retain it as a diagnostic. The published empirical "
                f"bands use the conservative envelope of lineage- and whole-developer-held-out residuals on the "
                f"{primary_developer_count}-developer frontier-like cohort ({primary_family_count} model-series lineages)."
            ),
            "change_central_forecasts": False,
            "k3_efficiency_projection_live_for_upper_tail": True,
            "k3_efficiency_projection_changes_centers": False,
            "k3_efficiency_default_projection_strength": default_efficiency_strength,
            "formal_coverage_guarantee": False,
            "sequential_coverage_warning": (
                "Sequential checks are reported three ways rather than treating repeated "
                "checkpoints from one developer as independent. For the whole-developer "
                f"refit, raw/developer-balanced/latest-developer coverage is "
                f"{100 * developer_coverage['50']['raw_checkpoint_coverage']['rate']:.1f}%/"
                f"{100 * developer_coverage['50']['developer_balanced_coverage']['rate']:.1f}%/"
                f"{100 * developer_coverage['50']['latest_developer_coverage']['rate']:.1f}% at 50%, "
                f"{100 * developer_coverage['80']['raw_checkpoint_coverage']['rate']:.1f}%/"
                f"{100 * developer_coverage['80']['developer_balanced_coverage']['rate']:.1f}%/"
                f"{100 * developer_coverage['80']['latest_developer_coverage']['rate']:.1f}% at 80%, and "
                f"{100 * developer_coverage['90']['raw_checkpoint_coverage']['rate']:.1f}%/"
                f"{100 * developer_coverage['90']['developer_balanced_coverage']['rate']:.1f}%/"
                f"{100 * developer_coverage['90']['latest_developer_coverage']['rate']:.1f}% at 90%."
            ),
        },
        "post_freeze_diagnostic_correction": {
            "applied_after_forecast_freeze": True,
            "immutable_v1_freeze": "forecast_freezes/2026-07-31-frontier-parameters-v1/forecast_freeze.json",
            "freeze_rewritten": False,
            "central_forecasts_changed": False,
            "note": (
                "This later diagnostic correction recomputes whole-developer holdouts, "
                "removes same-day lexical tie selection, and relabels the bands; it does "
                "not rewrite the immutable v1 forecast or its recorded centers."
            ),
        },
        "limitations": [
            "Benchmark snapshots are current rather than release-vintage historical snapshots.",
            f"Only {primary_developer_count} frontier-like calibration developers make order statistics coarse.",
            "The targets lie beyond the observed open-weight capability frontier.",
            f"MoE/reasoning matching leaves only {diagnostic_developer_count} independent developers and cannot justify narrower intervals.",
            "The expanding fits, current benchmark snapshots, clustered residuals, and out-of-support targets violate the exchangeability conditions required for a formal split-conformal guarantee.",
            "The requested 50% sequential band undercovers in the available historical sample and should not be read as a validated 50% probability statement.",
            "The intervals calibrate total-parameter prediction error, not uncertainty about active parameters or system-level inference compute.",
            "The K3-efficiency-projected bands are center-preserving winsorized stress tests, not literal conditioning, empirical coverage intervals, or formal Bayesian credible intervals.",
        ],
        "source_files": {
            str(BACKTEST.relative_to(ROOT)): sha256(BACKTEST),
            str(SITE_DATA.relative_to(ROOT)): sha256(SITE_DATA),
            str(OPUS_5_EVIDENCE.relative_to(ROOT)): sha256(OPUS_5_EVIDENCE),
            str(REGRESSION_PATH.relative_to(ROOT)): sha256(REGRESSION_PATH),
            str(UNIFIED_PATH.relative_to(ROOT)): sha256(UNIFIED_PATH),
            str(K3_EVIDENCE_PATH.relative_to(ROOT)): sha256(K3_EVIDENCE_PATH),
            str(K3_EFFICIENCY_PRIOR.relative_to(ROOT)): sha256(K3_EFFICIENCY_PRIOR),
            str(K3_EFFICIENCY_DRAWS.relative_to(ROOT)): sha256(K3_EFFICIENCY_DRAWS),
        },
        "outputs": {
            "calibration_ledger": str(LEDGER.relative_to(ROOT)),
            "site_data": str(SITE_OUTPUT.relative_to(ROOT)),
        },
    }

    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    SITE_OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with LEDGER.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger_rows[0]))
        writer.writeheader()
        writer.writerows(ledger_rows)
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "frontier_rows": cohorts[PRIMARY_COHORT]["rows"],
                "frontier_families": cohorts[PRIMARY_COHORT]["families"],
                "frontier_developers": cohorts[PRIMARY_COHORT]["developers"],
                "architecture_matched_families": cohorts["frontier_moe_reasoning"]["families"],
                "architecture_matched_developers": cohorts["frontier_moe_reasoning"]["developers"],
                "targets": target_intervals,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
