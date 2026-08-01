#!/usr/bin/env python3
"""Developer-held-out and Epoch-vintage sensitivity for parameter recovery.

This audit is deliberately separate from the live forecast.  It asks two
questions that the pseudo-chronological core backtest cannot answer directly:

1. What changes when the whole developer, rather than a model-series lineage,
   is removed from every training fold?
2. What changes when ECI scores come from the first archived Epoch snapshot in
   which a checkpoint appears, rather than the current ECI snapshot?

Only ECI has enough historical benchmark vintages for the second question.
AA, No-CoT, and Epoch compute remain current-snapshot evidence, so this script
does not label any full-ensemble result as a vintage backtest and never changes
forecast weights.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from run_parameter_backtest import (
    CURRENT_WEIGHTS,
    OUTPUT_DIR,
    REGRESSION_PATH,
    RESULT_PATH,
    UNIFIED_PATH,
    _backtest,
    _blend_predictions,
    _current_ensemble,
    _fit_predict,
    _k3_aa_external_holdout_prediction,
    _load_panels,
    _load_parameter_truth_registry,
    _metric_summary,
    _normal_model_name,
)


ROOT = Path(__file__).resolve().parent
HISTORICAL_ECI = ROOT / "sources/epoch_eci_historical_model_scores_2026-07-18.csv"
RESULT = OUTPUT_DIR / "parameter_developer_vintage_sensitivity_2026-07-31.json"
GENERATED_ON = "2026-07-31"
MIN_ECI_TRAIN_ROWS = 20
MIN_ECI_TRAIN_GROUPS = 6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_developer(value: str) -> str:
    text = value.strip()
    first = text.split(",", 1)[0].strip()
    aliases = {
        "Microsoft Research": "Microsoft",
        "Meta": "Meta AI",
        "Zhipu AI": "Z.ai (Zhipu AI)",
    }
    return aliases.get(first, first)


def fallback_developer(model: str, family: str) -> str:
    text = f"{model} {family}".lower()
    rules = (
        (("qwen",), "Alibaba"),
        (("seed",), "ByteDance"),
        (("olmo",), "AI2"),
        (("mistral", "mixtral", "ministral"), "Mistral AI"),
        (("nemotron",), "NVIDIA"),
        (("inkling",), "Thinking Machines Lab"),
        (("smollm",), "Hugging Face"),
        (("llama",), "Meta AI"),
        (("gemma",), "Google DeepMind"),
        (("deepseek",), "DeepSeek"),
        (("kimi",), "Moonshot"),
        (("phi",), "Microsoft"),
    )
    for tokens, developer in rules:
        if any(token in text for token in tokens):
            return developer
    raise ValueError(f"No canonical developer for {model!r} / {family!r}")


def developer_lookup() -> dict[str, str]:
    candidates: dict[str, Counter[str]] = defaultdict(Counter)
    with UNIFIED_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            organization = row.get("source_organization") or row.get("source_provider")
            if not organization:
                continue
            key = _normal_model_name(row["canonical_display_name"])
            source_weight = 3 if row["source"] in {"Epoch", "ECI"} else 1
            candidates[key][canonical_developer(organization)] += source_weight
    return {
        key: counts.most_common(1)[0][0]
        for key, counts in candidates.items()
    }


def with_developers(
    panels: dict[str, list[dict[str, Any]]], lookup: dict[str, str]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for panel, rows in panels.items():
        converted = []
        for row in rows:
            key = _normal_model_name(row["model"])
            developer = lookup.get(key) or fallback_developer(
                row["model"], row["family"]
            )
            converted.append(
                {
                    **row,
                    "lineage_family": row["family"],
                    "developer": developer,
                    "family": developer,
                }
            )
        output[panel] = converted
    return output


def selected_developer_predictions(
    panels: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    definitions = {
        "AA": (16, 6, "score", ("score", "date")),
        "No-CoT": (12, 3, "log_horizon", ("log_horizon", "date", "moe")),
        "Compute": (100, 20, "log_compute", ("log_compute", "date")),
    }
    for panel, (min_rows, min_groups, signal, features) in definitions.items():
        selected[panel] = _backtest(
            panel,
            panels[panel],
            "developer_holdout",
            features,
            min_rows,
            min_groups,
            True,
            signal,
        )
    score = _backtest(
        "ECI", panels["ECI"], "score_only", ("score",), 20, 6, True, "score"
    )
    dated = _backtest(
        "ECI",
        panels["ECI"],
        "score_date",
        ("score", "date"),
        20,
        6,
        True,
        "score",
    )
    selected["ECI"] = _blend_predictions(
        "ECI", "developer_holdout_60_40", score, dated, 0.60
    )
    selected["AA"].append(
        _k3_aa_external_holdout_prediction(
            panels["AA"],
            target_family="Moonshot",
            exclude_all_kimi_lineages=False,
        )
    )
    return selected


def paired_metrics(
    baseline: list[dict[str, Any]], challenger: list[dict[str, Any]]
) -> dict[str, Any]:
    left = {_normal_model_name(row["model"]): row for row in baseline}
    right = {_normal_model_name(row["model"]): row for row in challenger}
    keys = sorted(left.keys() & right.keys())
    return {
        "paired_models": len(keys),
        "baseline": _metric_summary([left[key] for key in keys]),
        "challenger": _metric_summary([right[key] for key in keys]),
        "models": [left[key]["model"] for key in keys],
    }


def latest_per_developer_metrics(
    rows: list[dict[str, Any]], lookup: dict[str, str]
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = _normal_model_name(row["model"])
        developer = lookup.get(key) or fallback_developer(row["model"], row["family"])
        grouped[developer].append(row)
    selected = []
    for developer_rows in grouped.values():
        latest_date = max(row["release_date"] for row in developer_rows)
        same_day = [
            row for row in developer_rows if row["release_date"] == latest_date
        ]
        selected.append(
            max(
                same_day,
                key=lambda row: (abs(float(row["log10_error"])), row["model"]),
            )
        )
    errors = sorted(abs(float(row["log10_error"])) for row in selected)
    intervals = {}
    for coverage in (0.50, 0.80, 0.90):
        raw_rank = math.ceil((len(errors) + 1) * coverage)
        supported = raw_rank <= len(errors)
        rank = raw_rank if supported else None
        intervals[str(int(coverage * 100))] = {
            "coverage": coverage,
            "rank": rank,
            "raw_rank": raw_rank,
            "supported": supported,
            "multiplicative_factor": (
                float(10 ** errors[rank - 1]) if rank is not None else None
            ),
        }
    return {
        "developers": len(selected),
        "metrics": _metric_summary(selected),
        "cluster_policy": (
            "largest absolute residual among checkpoints on each developer's "
            "latest eligible release date"
        ),
        "order_statistic_factors": intervals,
        # Compatibility alias for downstream readers; these are descriptive
        # order statistics, not a formal conformal guarantee.
        "conformal_factors": intervals,
        "latest_models": sorted(row["model"] for row in selected),
    }


def timestamp_date(timestamp: str) -> date:
    return datetime.strptime(timestamp[:8], "%Y%m%d").date()


def vintage_eci_predictions(
    eci_rows: list[dict[str, Any]], lookup: dict[str, str]
) -> list[dict[str, Any]]:
    with HISTORICAL_ECI.open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    snapshots = sorted({row["snapshot_timestamp"] for row in history})
    if len(snapshots) != 15:
        raise ValueError(f"Expected 15 Epoch ECI snapshots; found {len(snapshots)}")
    by_snapshot: dict[str, dict[str, float]] = {
        snapshot: {
            row["Model"]: float(row["eci"])
            for row in history
            if row["snapshot_timestamp"] == snapshot
        }
        for snapshot in snapshots
    }
    first_snapshot = {
        model: min(
            row["snapshot_timestamp"] for row in history if row["Model"] == model
        )
        for model in {row["Model"] for row in history}
    }
    parameter_rows = {row["model"]: dict(row) for row in eci_rows}
    initial = snapshots[0]
    output: list[dict[str, Any]] = []

    for model, snapshot in sorted(
        first_snapshot.items(), key=lambda item: (item[1], item[0])
    ):
        if snapshot == initial or model not in parameter_rows:
            continue
        target = {
            **parameter_rows[model],
            "score": by_snapshot[snapshot][model],
        }
        target_developer = lookup.get(_normal_model_name(model)) or fallback_developer(
            model, target["family"]
        )
        available = []
        for training_model, score in by_snapshot[snapshot].items():
            if training_model not in parameter_rows:
                continue
            row = {
                **parameter_rows[training_model],
                "score": score,
            }
            row["developer"] = lookup.get(_normal_model_name(training_model)) or fallback_developer(
                training_model, row["family"]
            )
            available.append(row)
        prior = [
            row for row in available if row["release_date"] < target["release_date"]
        ]
        lineage_train = [
            row for row in prior if row["family"] != target["family"]
        ]
        developer_train = [
            {**row, "family": row["developer"]}
            for row in prior
            if row["developer"] != target_developer
        ]
        if (
            len(lineage_train) < MIN_ECI_TRAIN_ROWS
            or len({row["family"] for row in lineage_train}) < MIN_ECI_TRAIN_GROUPS
            or len(developer_train) < MIN_ECI_TRAIN_ROWS
            or len({row["family"] for row in developer_train}) < MIN_ECI_TRAIN_GROUPS
        ):
            continue

        lineage_score, _ = _fit_predict(lineage_train, target, ("score",))
        lineage_dated, _ = _fit_predict(lineage_train, target, ("score", "date"))
        lineage_log10 = 0.60 * lineage_score + 0.40 * lineage_dated

        developer_target = {**target, "family": target_developer}
        developer_score, _ = _fit_predict(
            developer_train, developer_target, ("score",)
        )
        developer_dated, _ = _fit_predict(
            developer_train, developer_target, ("score", "date")
        )
        developer_log10 = 0.60 * developer_score + 0.40 * developer_dated

        actual_log10 = math.log10(float(target["total_b"]))
        previous = snapshots[snapshots.index(snapshot) - 1]
        observed_date = timestamp_date(snapshot)
        release_date = date.fromisoformat(target["release_date"])
        rank = sum(row["score"] <= target["score"] for row in prior) / len(prior)
        output.append(
            {
                "model": model,
                "release_date": target["release_date"],
                "family": target["family"],
                "developer": target_developer,
                "first_snapshot_timestamp": snapshot,
                "first_snapshot_date": observed_date.isoformat(),
                "previous_snapshot_date": timestamp_date(previous).isoformat(),
                "availability_lag_days": (observed_date - release_date).days,
                "interval_prospective": (
                    timestamp_date(previous) < release_date <= observed_date
                ),
                "target_eci": float(target["score"]),
                "frontier_signal_rank": float(rank),
                "actual_b": float(target["total_b"]),
                "lineage_train_n": len(lineage_train),
                "lineage_train_groups": len(
                    {row["family"] for row in lineage_train}
                ),
                "lineage_predicted_b": float(10**lineage_log10),
                "lineage_log10_error": float(lineage_log10 - actual_log10),
                "developer_train_n": len(developer_train),
                "developer_train_groups": len(
                    {row["family"] for row in developer_train}
                ),
                "developer_predicted_b": float(10**developer_log10),
                "developer_log10_error": float(developer_log10 - actual_log10),
            }
        )
    return output


def remap_error(
    rows: list[dict[str, Any]], error_field: str
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "log10_error": float(row[error_field]),
            "multiplicative_error": float(10 ** abs(float(row[error_field]))),
        }
        for row in rows
    ]


def vintage_cohorts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    filters: dict[str, Callable[[dict[str, Any]], bool]] = {
        "all_first_observed": lambda row: True,
        "frontier_like": lambda row: row["frontier_signal_rank"] >= 0.90,
        "available_within_90_days": lambda row: row["availability_lag_days"] <= 90,
        "interval_prospective": lambda row: bool(row["interval_prospective"]),
    }
    output = {}
    for name, predicate in filters.items():
        selected = [row for row in rows if predicate(row)]
        output[name] = {
            "rows": len(selected),
            "developers": len({row["developer"] for row in selected}),
            "models": sorted(row["model"] for row in selected),
            "lineage_holdout": _metric_summary(
                remap_error(selected, "lineage_log10_error")
            ),
            "developer_holdout": _metric_summary(
                remap_error(selected, "developer_log10_error")
            ),
        }
    return output


def current_vs_vintage_eci(
    vintage: list[dict[str, Any]], current_backtest: dict[str, Any]
) -> dict[str, Any]:
    current = {
        row["model"]: row
        for row in current_backtest["predictions"]
        if row["panel"] == "ECI"
    }
    paired = [row for row in vintage if row["model"] in current]
    output = {}
    filters = {
        "all_first_observed": lambda row: True,
        "frontier_like": lambda row: row["frontier_signal_rank"] >= 0.90,
        "available_within_90_days": lambda row: row["availability_lag_days"] <= 90,
        "interval_prospective": lambda row: bool(row["interval_prospective"]),
    }
    for name, predicate in filters.items():
        selected = [row for row in paired if predicate(row)]
        current_rows = [current[row["model"]] for row in selected]
        output[name] = {
            "n": len(selected),
            "current_snapshot": _metric_summary(current_rows),
            "first_observed_vintage": _metric_summary(
                remap_error(selected, "lineage_log10_error")
            ),
        }
    return output


def vintage_eci_swap(
    vintage: list[dict[str, Any]], current_backtest: dict[str, Any]
) -> dict[str, Any]:
    vintage_by_key = {_normal_model_name(row["model"]): row for row in vintage}
    original = []
    swapped = []
    for row in current_backtest["ensemble_predictions"]:
        if row["frontier_signal_rank"] < 0.90:
            continue
        key = row["normalized_model"]
        if key not in vintage_by_key or not any(
            component["panel"] == "ECI" for component in row["components"]
        ):
            continue
        weighted_logs = []
        weights = []
        for component in row["components"]:
            predicted_b = (
                vintage_by_key[key]["lineage_predicted_b"]
                if component["panel"] == "ECI"
                else float(component["predicted_b"])
            )
            weighted_logs.append(float(component["weight"]) * math.log10(predicted_b))
            weights.append(float(component["weight"]))
        prediction_log10 = sum(weighted_logs) / sum(weights)
        error = prediction_log10 - math.log10(float(row["actual_b"]))
        original.append(row)
        swapped.append(
            {
                **row,
                "predicted_b": float(10**prediction_log10),
                "log10_error": float(error),
                "multiplicative_error": float(10 ** abs(error)),
            }
        )
    return {
        "n": len(original),
        "models": [row["model"] for row in original],
        "current_eci_component": _metric_summary(original),
        "first_observed_vintage_eci_component": _metric_summary(swapped),
    }


def main() -> None:
    panels, _ = _load_panels()
    lookup = developer_lookup()
    developer_panels = with_developers(panels, lookup)
    selected = selected_developer_predictions(developer_panels)
    developer_ensemble = _current_ensemble(
        selected,
        equal_weight=False,
        parameter_registry=_load_parameter_truth_registry(),
    )
    current_backtest = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    baseline_ensemble = current_backtest["ensemble_predictions"]
    baseline_frontier = [
        row for row in baseline_ensemble if row["frontier_signal_rank"] >= 0.90
    ]
    developer_frontier = [
        row for row in developer_ensemble if row["frontier_signal_rank"] >= 0.90
    ]

    vintage = vintage_eci_predictions(panels["ECI"], lookup)
    if len(vintage) != 25:
        raise ValueError(f"Expected 25 current-canonical first-observed ECI targets; found {len(vintage)}")
    prospective = {row["model"] for row in vintage if row["interval_prospective"]}
    if prospective != {"Kimi K2.5", "Kimi K2.7 Code"}:
        raise ValueError(f"Unexpected prospective ECI targets: {sorted(prospective)}")

    result = {
        "generated_on": GENERATED_ON,
        "question": (
            "How sensitive is parameter recovery to developer-level holdout and "
            "first-observed Epoch ECI vintages?"
        ),
        "decision": {
            "change_forecast_weights": False,
            "change_central_forecasts": False,
            "reason": (
                "The developer sensitivity is diagnostic and only ECI has historical "
                "measurement vintages. The two genuinely interval-prospective ECI "
                "targets share one developer, so they cannot identify full-system accuracy."
            ),
        },
        "method": {
            "current_weights": CURRENT_WEIGHTS,
            "developer_holdout": (
                "Replace model-series family keys with canonical source organizations, "
                "exclude the entire target developer, and equalize aggregate training "
                "weight by developer through the unchanged core weighting function."
            ),
            "latest_developer_cluster": (
                "Use the largest absolute residual among checkpoints on each developer's "
                "latest eligible release date; never select same-day checkpoints by name."
            ),
            "vintage_target": (
                "Use each checkpoint's first archived Epoch ECI score after the initial "
                "2025-11-13 left-censored snapshot. Fit only same-vintage models with "
                "strictly earlier release dates."
            ),
            "vintage_scope": "ECI only; no full-ensemble vintage claim",
            "k3_external_aa_policy": (
                "K3 is absent from the AA parameter target panel by design. The exact-score "
                "AA prediction excludes the whole Moonshot developer and is included in the "
                "developer-held-out ensemble; K3's disclosed total never enters the fit."
            ),
        },
        "developer_holdout_current_snapshot": {
            "lineage_frontier": _metric_summary(baseline_frontier),
            "developer_frontier": _metric_summary(developer_frontier),
            "paired_frontier": paired_metrics(
                baseline_frontier, developer_frontier
            ),
            "lineage_latest_per_developer": latest_per_developer_metrics(
                baseline_frontier, lookup
            ),
            "latest_per_developer": latest_per_developer_metrics(
                developer_frontier, lookup
            ),
            "developer_ensemble_rows": len(developer_ensemble),
            "developer_frontier_rows": len(developer_frontier),
        },
        "eci_vintage": {
            "inventory": {
                "historical_snapshots": 15,
                "historical_score_rows": sum(
                    1 for _ in csv.DictReader(HISTORICAL_ECI.open(newline="", encoding="utf-8"))
                ),
                "current_parameter_checkpoints": len(panels["ECI"]),
                "first_observed_targets": len(vintage),
                "interval_prospective_targets": len(prospective),
            },
            "cohorts": vintage_cohorts(vintage),
            "paired_current_vs_vintage": current_vs_vintage_eci(
                vintage, current_backtest
            ),
            "frontier_ensemble_eci_swap": vintage_eci_swap(
                vintage, current_backtest
            ),
            "predictions": vintage,
        },
        "interpretation": {
            "headline": (
                "The paired vintage penalty is modest on available ECI rows, but the "
                "fully prospective sample is too small to validate the 1.9x full-ensemble median."
            ),
            "independence": (
                "Developer grouping reduces the effective independent calibration count; "
                "lineage labels must not be described as independent developers."
            ),
            "tail": (
                "Developer-balanced calibration changes tail factors more materially than "
                "the paired row median, so uncertainty claims should emphasize tails."
            ),
        },
        "limitations": [
            "AA, No-CoT, and compute measurements are current-snapshot rather than release-vintage.",
            "Twenty-three of 25 first-observed ECI targets are archive backfills, not interval-prospective releases.",
            "The two interval-prospective targets are both Moonshot Kimi checkpoints.",
            "The frozen ECI archive ends on 2026-07-16, before the July 31 ECI refresh.",
            "Historical ECI central scores do not include vintage bootstrap confidence intervals.",
            "Retired historical-only parameter rows remain covered by the separate ECI fit tournament; this sensitivity uses the current 89-checkpoint canonical map.",
        ],
        "sources": {
            str(REGRESSION_PATH.relative_to(ROOT)): sha256(REGRESSION_PATH),
            str(UNIFIED_PATH.relative_to(ROOT)): sha256(UNIFIED_PATH),
            str(HISTORICAL_ECI.relative_to(ROOT)): sha256(HISTORICAL_ECI),
            str(RESULT_PATH.relative_to(ROOT)): sha256(RESULT_PATH),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "developer_frontier": result["developer_holdout_current_snapshot"][
                    "developer_frontier"
                ],
                "vintage_frontier": result["eci_vintage"]["cohorts"][
                    "frontier_like"
                ],
                "prospective": result["eci_vintage"]["cohorts"][
                    "interval_prospective"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
