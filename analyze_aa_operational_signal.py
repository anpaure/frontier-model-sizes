#!/usr/bin/env python3
"""Audit standardized Artificial Analysis price, speed, and latency signals.

The frozen AA model payload reports current per-token prices and live API
performance.  AA represents first-party APIs directly and otherwise uses the
median across providers; speed is normalized to OpenAI-token units.  This
script keeps those two serving regimes explicit, deduplicates only by the
already-audited highest-score checkpoint policy, and asks whether any
operational field improves strictly chronological developer-held-out recovery
of disclosed total parameter counts.

These current serving measurements are correlated with the OpenRouter branch
and are not historical vintages.  They therefore enter as a validation audit,
not as an automatic new likelihood term.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from aa_calibration_overrides import (
    OVERRIDES_PATH as AA_CALIBRATION_OVERRIDES_PATH,
    parameter_label_available_before,
    parameter_training_eligibility_date,
)
from aa_score_availability import aa_prediction_information_date
from analyze_aa_inference_budget_signal import (
    paired_cluster_bootstrap,
    parameter_metrics,
    predict,
)
from open_model_parameter_truth import LEDGER_PATH as PARAMETER_TRUTH_PATH
from open_model_parameter_truth import resolve_parameter_truth


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
DETAIL = ROOT / "sources/aa_detailed_model_signals_2026-07-31.csv"
RAW = ROOT / "sources/aa_detailed_snapshot_2026-07-31.html.gz"
PANEL_INPUT = OUT / "aa_detailed_parameter_panel_2026-07-18.csv"
EPOCH_CROSSCHECK_INPUT = OUT / "aa_detailed_epoch_crosscheck_2026-07-18.csv"
OPENROUTER_INPUT = OUT / "openrouter_parameter_calibration_2026-07-18.csv"

RESULT = OUT / "aa_operational_signal_audit_2026-07-18.json"
PANEL = OUT / "aa_operational_parameter_panel_2026-07-18.csv"
PREDICTIONS = OUT / "aa_operational_backtest_predictions_2026-07-18.csv"
CROSSCHECK = OUT / "aa_openrouter_operational_crosscheck_2026-07-18.csv"

MAIN_MIN_ROWS = 30
MAIN_MIN_DEVELOPERS = 8
EXPLORATORY_MIN_ROWS = 20
EXPLORATORY_MIN_DEVELOPERS = 6
FRONTIER_RANK = 0.90


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
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


def number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def boolean(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"Invalid serialized boolean {value!r}")
    return value == "True"


def enrich_panel() -> list[dict[str, Any]]:
    detailed = {row["slug"]: row for row in read_csv(DETAIL)}
    panel = read_csv(PANEL_INPUT)
    if len(detailed) != 587 or not panel:
        raise ValueError(f"Unexpected AA inputs: {len(detailed)} raw / {len(panel)} panel")

    output: list[dict[str, Any]] = []
    for row in panel:
        source = detailed[row["selected_slug"]]
        if source["release_date"] != row["release_date"]:
            raise ValueError(f"Release mismatch for {row['selected_slug']}")
        source_total = float(source["parameters_b"])
        canonical_total = float(row["parameters_b"])
        raw_total = float(row.get("raw_parameter_total_b") or canonical_total)
        truth = resolve_parameter_truth(row["selected_name"])
        if not row.get("calibration_override_id"):
            if not math.isclose(source_total, raw_total, rel_tol=0, abs_tol=1e-9):
                raise ValueError(
                    f"Raw parameter mismatch for {row['selected_slug']}: "
                    f"{source_total} vs {raw_total}"
                )
            if row.get("parameter_truth_id"):
                if truth is None or row["parameter_truth_id"] != truth["truth_id"]:
                    raise ValueError(
                        f"Parameter-truth identity mismatch for {row['selected_slug']}"
                    )
                if not math.isclose(
                    canonical_total,
                    float(truth["canonical_total_parameters_b"]),
                    rel_tol=0,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        f"Canonical parameter mismatch for {row['selected_slug']}"
                    )
            elif truth is not None or not math.isclose(
                source_total, canonical_total, rel_tol=0, abs_tol=1e-9
            ):
                raise ValueError(
                    f"Unproven parameter canonicalization for {row['selected_slug']}"
                )
        if not math.isclose(
            float(source["intelligence_index"]), float(row["intelligence_index"])
        ):
            raise ValueError(f"AA score mismatch for {row['selected_slug']}")

        price = number(source["price_blended_7_2_1_usd_per_mtoken"])
        speed = number(source["median_output_speed_tps"])
        ttfc = number(source["median_time_to_first_chunk_seconds"])
        cost = number(source["intelligence_cost_per_task_usd"])
        task_time = number(source["intelligence_time_per_task_seconds"])
        source_type = source["performance_data_source_type"]
        if source_type not in {"", "firstParty", "median"}:
            raise ValueError(f"Unknown AA performance source type {source_type!r}")
        output.append(
            {
                **row,
                "parameters_b": float(row["parameters_b"]),
                "active_parameters_b": number(row["active_parameters_b"]),
                "intelligence_index": float(row["intelligence_index"]),
                "intelligence_index_estimated": boolean(
                    row["intelligence_index_estimated"]
                ),
                "price_input_usd_per_mtoken": number(
                    source["price_input_usd_per_mtoken"]
                ),
                "price_output_usd_per_mtoken": number(
                    source["price_output_usd_per_mtoken"]
                ),
                "price_cache_hit_usd_per_mtoken": number(
                    source["price_cache_hit_usd_per_mtoken"]
                ),
                "price_blended_7_2_1_usd_per_mtoken": price,
                "median_output_speed_tps": speed,
                "median_time_to_first_chunk_seconds": ttfc,
                "intelligence_cost_per_task_usd": cost,
                "intelligence_time_per_task_seconds": task_time,
                "performance_data_source_type": source_type,
                "performance_provider_name": source[
                    "performance_provider_name"
                ],
                "log1p_blended_price": None
                if price is None
                else math.log10(1.0 + price),
                "log_output_speed": None
                if speed is None
                else math.log10(speed),
                "log1p_ttfc": None
                if ttfc is None
                else math.log10(1.0 + ttfc),
                "log1p_cost_per_task": None
                if cost is None
                else math.log10(1.0 + cost),
                "log_task_time": None
                if task_time is None
                else math.log10(task_time),
                "first_party_indicator": 1.0 if source_type == "firstParty" else 0.0,
                "operational_source_page_url": source["source_page_url"],
            }
        )
    if len({row["checkpoint_group_id"] for row in output}) != len(output):
        raise ValueError("Operational panel checkpoint IDs are not unique")
    return output


SPECIFICATIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "price_global",
        "features": ("log1p_blended_price",),
        "required": ("price_blended_7_2_1_usd_per_mtoken",),
        "filter": lambda row: True,
        "minimums": (MAIN_MIN_ROWS, MAIN_MIN_DEVELOPERS),
        "status": "primary validation",
    },
    {
        "id": "price_source_adjusted",
        "features": ("log1p_blended_price", "first_party_indicator"),
        "required": ("price_blended_7_2_1_usd_per_mtoken",),
        "filter": lambda row: True,
        "minimums": (MAIN_MIN_ROWS, MAIN_MIN_DEVELOPERS),
        "status": "primary validation",
    },
    {
        "id": "price_provider_median",
        "features": ("log1p_blended_price",),
        "required": ("price_blended_7_2_1_usd_per_mtoken",),
        "filter": lambda row: row["performance_data_source_type"] == "median",
        "minimums": (MAIN_MIN_ROWS, MAIN_MIN_DEVELOPERS),
        "status": "primary serving-regime check",
    },
    {
        "id": "price_first_party",
        "features": ("log1p_blended_price",),
        "required": ("price_blended_7_2_1_usd_per_mtoken",),
        "filter": lambda row: row["performance_data_source_type"] == "firstParty",
        "minimums": (MAIN_MIN_ROWS, MAIN_MIN_DEVELOPERS),
        "status": "primary serving-regime check",
    },
    {
        "id": "output_speed",
        "features": ("log_output_speed",),
        "required": ("median_output_speed_tps",),
        "filter": lambda row: True,
        "minimums": (MAIN_MIN_ROWS, MAIN_MIN_DEVELOPERS),
        "status": "primary validation",
    },
    {
        "id": "latency_ttfc",
        "features": ("log1p_ttfc",),
        "required": ("median_time_to_first_chunk_seconds",),
        "filter": lambda row: True,
        "minimums": (MAIN_MIN_ROWS, MAIN_MIN_DEVELOPERS),
        "status": "primary validation",
    },
    {
        "id": "speed_and_latency",
        "features": ("log_output_speed", "log1p_ttfc"),
        "required": (
            "median_output_speed_tps",
            "median_time_to_first_chunk_seconds",
        ),
        "filter": lambda row: True,
        "minimums": (MAIN_MIN_ROWS, MAIN_MIN_DEVELOPERS),
        "status": "primary validation",
    },
    {
        "id": "cost_per_task",
        "features": ("log1p_cost_per_task",),
        "required": ("intelligence_cost_per_task_usd",),
        "filter": lambda row: True,
        "minimums": (EXPLORATORY_MIN_ROWS, EXPLORATORY_MIN_DEVELOPERS),
        "status": "exploratory low coverage",
    },
    {
        "id": "time_per_task",
        "features": ("log_task_time",),
        "required": ("intelligence_time_per_task_seconds",),
        "filter": lambda row: True,
        "minimums": (EXPLORATORY_MIN_ROWS, EXPLORATORY_MIN_DEVELOPERS),
        "status": "exploratory low coverage",
    },
)


def run_specification(
    panel: list[dict[str, Any]], specification: dict[str, Any]
) -> list[dict[str, Any]]:
    required = specification["required"]
    eligible = [
        row
        for row in panel
        if specification["filter"](row)
        and all(row[field] is not None for field in required)
    ]
    minimum_rows, minimum_developers = specification["minimums"]
    baseline_features = ("score", "date")
    candidate_features = (*baseline_features, *specification["features"])
    output: list[dict[str, Any]] = []
    for test in sorted(
        eligible,
        key=lambda row: (
            aa_prediction_information_date(row),
            row["release_date"],
            row["selected_name"],
        ),
    ):
        prediction_date = aa_prediction_information_date(test)
        train = [
            row
            for row in eligible
            if parameter_label_available_before(row, prediction_date)
            and row["creator_slug"] != test["creator_slug"]
        ]
        if (
            len(train) < minimum_rows
            or len({row["creator_slug"] for row in train}) < minimum_developers
        ):
            continue
        train_rows = [
            {
                **row,
                "developer": row["creator_slug"],
                "score": row["intelligence_index"],
                "estimated": row["intelligence_index_estimated"],
            }
            for row in train
        ]
        test_row = {
            **test,
            "developer": test["creator_slug"],
            "score": test["intelligence_index"],
            "estimated": test["intelligence_index_estimated"],
        }
        baseline, baseline_beta = predict(
            train_rows, test_row, baseline_features, "developer"
        )
        candidate, candidate_beta = predict(
            train_rows, test_row, candidate_features, "developer"
        )
        actual = math.log10(test["parameters_b"])
        output.append(
            {
                "specification": specification["id"],
                "specification_status": specification["status"],
                "checkpoint_group_id": test["checkpoint_group_id"],
                "selected_slug": test["selected_slug"],
                "model": test["selected_name"],
                "developer": test["creator_slug"],
                "release_date": test["release_date"],
                "prediction_information_date": prediction_date,
                "actual_parameters_b": test["parameters_b"],
                "aa_score": test["intelligence_index"],
                "performance_data_source_type": test[
                    "performance_data_source_type"
                ],
                "price_blended_7_2_1_usd_per_mtoken": test[
                    "price_blended_7_2_1_usd_per_mtoken"
                ],
                "median_output_speed_tps": test["median_output_speed_tps"],
                "median_time_to_first_chunk_seconds": test[
                    "median_time_to_first_chunk_seconds"
                ],
                "intelligence_cost_per_task_usd": test[
                    "intelligence_cost_per_task_usd"
                ],
                "intelligence_time_per_task_seconds": test[
                    "intelligence_time_per_task_seconds"
                ],
                "frontier_score_rank": sum(
                    row["intelligence_index"] <= test["intelligence_index"]
                    for row in train
                )
                / len(train),
                "baseline_log10_error": baseline - actual,
                "candidate_log10_error": candidate - actual,
                "baseline_predicted_b": 10**baseline,
                "candidate_predicted_b": 10**candidate,
                "train_n": len(train),
                "train_developers": len({row["creator_slug"] for row in train}),
                "train_max_date": max(
                    parameter_training_eligibility_date(row) for row in train
                ),
                "test_developer_excluded": True,
                "minimum_train_rows": minimum_rows,
                "minimum_train_developers": minimum_developers,
                "baseline_coefficients": json.dumps(
                    baseline_beta, separators=(",", ":")
                ),
                "candidate_coefficients": json.dumps(
                    candidate_beta, separators=(",", ":")
                ),
            }
        )
    return output


def summarize_predictions(
    predictions: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for index, (scope, selected) in enumerate(
        (
            ("all", predictions),
            (
                "frontier_like",
                [
                    row
                    for row in predictions
                    if row["frontier_score_rank"] >= FRONTIER_RANK
                ],
            ),
        )
    ):
        output[scope] = {
            "n": len(selected),
            "developers": len({row["developer"] for row in selected}),
            "baseline": parameter_metrics(
                row["baseline_log10_error"] for row in selected
            ),
            "candidate": parameter_metrics(
                row["candidate_log10_error"] for row in selected
            ),
            "paired_developer_bootstrap": paired_cluster_bootstrap(
                selected,
                "developer",
                "candidate_log10_error",
                "baseline_log10_error",
                seed + index,
            ),
        }
    return output


def rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def correlation(
    rows: list[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row[left] not in (None, "")
        and row[right] not in (None, "")
        and float(row[left]) > 0
        and float(row[right]) > 0
    ]
    x = [math.log10(float(row[left])) for row in selected]
    y = [math.log10(float(row[right])) for row in selected]
    if len(selected) < 3:
        raise ValueError(f"Too little overlap for {left} vs {right}")
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rank_values(x), rank_values(y))[0, 1])
    return {
        "n_positive_pairs": len(selected),
        "log10_pearson": pearson,
        "spearman": spearman,
        "median_left_over_right": float(
            np.median(
                [float(row[left]) / float(row[right]) for row in selected]
            )
        ),
    }


def build_openrouter_crosscheck(
    panel: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_group = {row["checkpoint_group_id"]: row for row in panel}
    openrouter = {
        row["canonical_checkpoint_id"]: row for row in read_csv(OPENROUTER_INPUT)
    }
    output: list[dict[str, Any]] = []
    for match in read_csv(EPOCH_CROSSCHECK_INPUT):
        checkpoint_id = match["epoch_checkpoint_id"]
        if checkpoint_id not in openrouter:
            continue
        aa = by_group[match["aa_detailed_group_id"]]
        router = openrouter[checkpoint_id]
        output.append(
            {
                "canonical_checkpoint_id": checkpoint_id,
                "epoch_model": match["epoch_model"],
                "aa_model": aa["selected_name"],
                "aa_slug": aa["selected_slug"],
                "release_date": aa["release_date"],
                "parameters_b": aa["parameters_b"],
                "aa_performance_source_type": aa[
                    "performance_data_source_type"
                ],
                "aa_blended_price_usd_per_mtoken": aa[
                    "price_blended_7_2_1_usd_per_mtoken"
                ],
                "openrouter_blended_price_usd_per_mtoken": number(
                    router["blended_price_usd_per_mtoken"]
                ),
                "aa_output_speed_tps": aa["median_output_speed_tps"],
                "openrouter_raw_throughput_tps": number(
                    router["raw_throughput_tps_1w"]
                ),
                "openrouter_provider_normalized_throughput_ratio": number(
                    router["provider_normalized_throughput_ratio"]
                ),
                "aa_source_url": aa["operational_source_page_url"],
                "openrouter_model_ids": router["openrouter_model_ids"],
            }
        )
    output.sort(key=lambda row: (row["release_date"], row["canonical_checkpoint_id"]))
    if len(output) != 28 or len({row["canonical_checkpoint_id"] for row in output}) != 28:
        raise ValueError(f"Unexpected AA/OpenRouter exact overlap: {len(output)}")
    summary = {
        "exact_epoch_checkpoint_intersection": len(output),
        "price": correlation(
            output,
            "aa_blended_price_usd_per_mtoken",
            "openrouter_blended_price_usd_per_mtoken",
        ),
        "raw_speed": correlation(
            output,
            "aa_output_speed_tps",
            "openrouter_raw_throughput_tps",
        ),
        "provider_normalized_speed": correlation(
            output,
            "aa_output_speed_tps",
            "openrouter_provider_normalized_throughput_ratio",
        ),
    }
    return output, summary


def coverage(panel: list[dict[str, Any]], field: str) -> dict[str, int]:
    rows = [row for row in panel if row[field] is not None]
    return {
        "checkpoints": len(rows),
        "developers": len({row["creator_slug"] for row in rows}),
    }


def main() -> None:
    panel = enrich_panel()
    all_predictions: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for index, specification in enumerate(SPECIFICATIONS):
        predictions = run_specification(panel, specification)
        all_predictions.extend(predictions)
        summaries[specification["id"]] = {
            "status": specification["status"],
            "eligible_checkpoints": len(
                [
                    row
                    for row in panel
                    if specification["filter"](row)
                    and all(row[field] is not None for field in specification["required"])
                ]
            ),
            "minimum_train_rows": specification["minimums"][0],
            "minimum_train_developers": specification["minimums"][1],
            "scopes": summarize_predictions(predictions, 20260730 + index * 10),
        }

    crosscheck, crosscheck_summary = build_openrouter_crosscheck(panel)
    result = {
        "metadata": {
            "generated_on": "2026-07-31",
            "target": "log10 disclosed total parameters in billions",
            "outer_split": "strictly earlier parameter-training eligibility date; entire test developer removed",
            "training_weights": "equal total weight per developer; estimated AA scores receive 0.5 weight",
            "duplicate_policy": "highest Intelligence Index configuration per exact parameter/date/weight checkpoint",
            "price_definition": "AA 7:2:1 cache-hit/input/output blended USD per million native tokens",
            "speed_definition": "AA median output speed after first token, standardized to OpenAI-token units",
            "performance_representation": "first-party API when available; otherwise median across providers",
            "methodology_urls": [
                "https://artificialanalysis.ai/methodology",
                "https://artificialanalysis.ai/methodology/performance-benchmarking",
                "https://artificialanalysis.ai/models/",
            ],
            "benchmark_vintage_caveat": "Current serving measurements and current AA benchmark snapshot, not historical vintages.",
        },
        "data_audit": {
            "raw_model_configurations": 587,
            "deduplicated_open_parameter_checkpoints": len(panel),
            "coverage": {
                "blended_price": coverage(
                    panel, "price_blended_7_2_1_usd_per_mtoken"
                ),
                "output_speed": coverage(panel, "median_output_speed_tps"),
                "latency_ttfc": coverage(
                    panel, "median_time_to_first_chunk_seconds"
                ),
                "cost_per_task": coverage(
                    panel, "intelligence_cost_per_task_usd"
                ),
                "time_per_task": coverage(
                    panel, "intelligence_time_per_task_seconds"
                ),
            },
            "performance_source_type_counts": {
                source_type: sum(
                    row["performance_data_source_type"] == source_type
                    for row in panel
                )
                for source_type in ("firstParty", "median", "")
            },
        },
        "backtests": summaries,
        "aa_openrouter_exact_crosscheck": crosscheck_summary,
        "decision": {
            "validates_existing_price_direction": True,
            "change_live_price_weight": False,
            "incremental_aa_operational_price_weight": 0.0,
            "incremental_aa_speed_weight": 0.0,
            "incremental_aa_latency_weight": 0.0,
            "reason": "AA provider-median price improves broad and frontier-like held-out recovery and agrees strongly with OpenRouter prices, validating the existing small price branch. But first-party price worsens frontier-like recovery, Fable/Sol are first-party targets, and AA/OpenRouter price is highly correlated rather than independent. Speed and latency do not improve held-out mean error. Keep the current conservative price weight and add no operational weight.",
        },
        "source_manifest": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                DETAIL,
                RAW,
                AA_CALIBRATION_OVERRIDES_PATH,
                PANEL_INPUT,
                EPOCH_CROSSCHECK_INPUT,
                OPENROUTER_INPUT,
                PARAMETER_TRUTH_PATH,
            )
        },
        "outputs": {
            "operational_panel": str(PANEL.relative_to(ROOT)),
            "backtest_predictions": str(PREDICTIONS.relative_to(ROOT)),
            "openrouter_crosscheck": str(CROSSCHECK.relative_to(ROOT)),
        },
    }
    write_csv(PANEL, panel)
    write_csv(PREDICTIONS, all_predictions)
    write_csv(CROSSCHECK, crosscheck)
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(RESULT),
                "panel": len(panel),
                "predictions": len(all_predictions),
                "openrouter_exact_overlap": len(crosscheck),
                "price_checkpoints": result["data_audit"]["coverage"][
                    "blended_price"
                ]["checkpoints"],
                "speed_checkpoints": result["data_audit"]["coverage"][
                    "output_speed"
                ]["checkpoints"],
                "change_live_price_weight": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
