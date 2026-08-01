#!/usr/bin/env python3
"""Quantify the effect of replacing no-CoT paper months with audited exact dates.

The paper reports bootstrap-median frontier doubling laws but only month-level
release dates.  Its bootstrap samples are not included in the LaTeX archive.
We therefore isolate the date effect with the transparent deterministic
approximation described in the paper: OLS on log2 model horizons along the
release-date Pareto frontier.  The exact-date/month-date slope ratio is applied
to the paper's reported median and interval; all horizon measurements remain
the paper's values.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATE = "2026-07-18"
THREAD = "019f6c42-2d53-7743-ab07-6293e2618dd7"
OUTPUT = ROOT / "outputs" / THREAD
OBSERVATIONS = OUTPUT / "unified_model_observations_compute_enriched_2026-07-17.csv"
OVERRIDES = ROOT / f"sources/no_cot_exact_date_overrides_{DATE}.csv"
METADATA = ROOT / f"sources/no_cot_exact_date_collection_metadata_{DATE}.json"
RESULT = OUTPUT / f"no_cot_exact_date_audit_{DATE}.json"
MODEL_AUDIT = OUTPUT / f"no_cot_exact_date_model_audit_{DATE}.csv"

PAPER_TIME_DAYS = 373.0
PAPER_TIME_CI = (167.0, 691.0)
PAPER_TOKEN_DAYS = 437.0
PAPER_TOKEN_CI = (341.0, 571.0)
EXCLUDED_FROM_TREND = {"GPT-2", "GPT-3"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pareto_ols(rows: list[dict[str, Any]], metric: str, date_field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["model"] in EXCLUDED_FROM_TREND:
            continue
        grouped[row[date_field]].append(row)

    # A lower-horizon model on the same release date is strictly dominated.
    date_winners = [
        max(group, key=lambda row: row[metric])
        for _, group in sorted(grouped.items())
    ]
    frontier: list[dict[str, Any]] = []
    best = -math.inf
    for row in date_winners:
        if row[metric] > best:
            frontier.append(row)
            best = row[metric]

    origin = date(2020, 1, 1)
    x = [(date.fromisoformat(row[date_field]) - origin).days for row in frontier]
    y = [math.log2(row[metric]) for row in frontier]
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    denominator = sum((value - x_mean) ** 2 for value in x)
    slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denominator
    intercept = y_mean - slope * x_mean
    residual = sum((b - (intercept + slope * a)) ** 2 for a, b in zip(x, y))
    total = sum((b - y_mean) ** 2 for b in y)
    return {
        "metric": metric,
        "date_field": date_field,
        "frontier_models": [row["model"] for row in frontier],
        "frontier_rows": [
            {"model": row["model"], "date": row[date_field], "value": row[metric]}
            for row in frontier
        ],
        "n": len(frontier),
        "slope_log2_per_day": slope,
        "doubling_time_days": 1.0 / slope,
        "r_squared": 1.0 - residual / total,
    }


def adjusted_law(paper_days: float, paper_ci: tuple[float, float], ratio: float) -> dict[str, Any]:
    return {
        "paper_reported_point_days": paper_days,
        "paper_reported_ci_95_days": list(paper_ci),
        "exact_date_adjustment_ratio": ratio,
        "adjusted_point_days": paper_days * ratio,
        "adjusted_ci_95_days": [paper_ci[0] * ratio, paper_ci[1] * ratio],
    }


def main() -> None:
    all_rows = read_csv(OBSERVATIONS)
    no_cot = [
        row
        for row in all_rows
        if row["source"] == "No-CoT" and row["record_type"] == "model"
    ]
    if len(no_cot) != 49:
        raise ValueError(f"Expected 49 no-CoT model rows, found {len(no_cot)}")
    if len({row["source_model_name"] for row in no_cot}) != 49:
        raise ValueError("Duplicate no-CoT model names")
    missing_exact = [row["source_model_name"] for row in no_cot if not row["canonical_release_date"]]
    if missing_exact:
        raise ValueError(f"Missing exact no-CoT dates: {missing_exact}")

    frontier_source = [row for row in no_cot if row["source_locator"] == "tab:horizons-per-model"]
    if len(frontier_source) != 14:
        raise ValueError(f"Expected 14 frontier rows, found {len(frontier_source)}")
    frontier: list[dict[str, Any]] = []
    for row in frontier_source:
        frontier.append(
            {
                "model": row["source_model_name"],
                "paper_month": row["source_release_date"],
                "exact_date": row["canonical_release_date"],
                "time_horizon_minutes": float(row["nocot_time_horizon_minutes"]),
                "token_horizon_tokens": float(row["nocot_token_horizon_tokens"]),
            }
        )

    time_month = pareto_ols(frontier, "time_horizon_minutes", "paper_month")
    time_exact = pareto_ols(frontier, "time_horizon_minutes", "exact_date")
    token_month = pareto_ols(frontier, "token_horizon_tokens", "paper_month")
    token_exact = pareto_ols(frontier, "token_horizon_tokens", "exact_date")
    time_ratio = time_exact["doubling_time_days"] / time_month["doubling_time_days"]
    token_ratio = token_exact["doubling_time_days"] / token_month["doubling_time_days"]
    time_law = adjusted_law(PAPER_TIME_DAYS, PAPER_TIME_CI, time_ratio)
    token_law = adjusted_law(PAPER_TOKEN_DAYS, PAPER_TOKEN_CI, token_ratio)

    override_rows = read_csv(OVERRIDES)
    override_by_model = {row["paper_model"]: row for row in override_rows}
    month_only_after = [
        row["source_model_name"]
        for row in no_cot
        if row["canonical_release_date"] == row["source_release_date"]
        and row["canonical_release_date_source"] == "source release date"
    ]
    audit_rows: list[dict[str, Any]] = []
    time_month_frontier = set(time_month["frontier_models"])
    time_exact_frontier = set(time_exact["frontier_models"])
    token_month_frontier = set(token_month["frontier_models"])
    token_exact_frontier = set(token_exact["frontier_models"])
    for row in no_cot:
        model = row["source_model_name"]
        override = override_by_model.get(model)
        audit_rows.append(
            {
                "model": model,
                "source_locator": row["source_locator"],
                "paper_month_date": row["source_release_date"],
                "exact_release_date": row["canonical_release_date"],
                "day_offset_from_month_start": (
                    date.fromisoformat(row["canonical_release_date"])
                    - date.fromisoformat(row["source_release_date"])
                ).days,
                "exact_date_source": row["canonical_release_date_source"],
                "explicit_override": str(override is not None),
                "parameter_join_policy": override["parameter_join_policy"] if override else "existing exact checkpoint crosswalk",
                "time_pareto_with_month_dates": str(model in time_month_frontier),
                "time_pareto_with_exact_dates": str(model in time_exact_frontier),
                "token_pareto_with_month_dates": str(model in token_month_frontier),
                "token_pareto_with_exact_dates": str(model in token_exact_frontier),
            }
        )

    MODEL_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_AUDIT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit_rows)

    result = {
        "metadata": {
            "generated_on": DATE,
            "question": "Does replacing paper month-first dates with audited day-level dates materially change the no-CoT frontier laws?",
            "method": "Deterministic OLS on log2 horizon along the release-date Pareto frontier, matching the paper's documented approximation. Apply the exact/month slope ratio to the paper's bootstrap-median point and interval because bootstrap samples are not in the LaTeX archive.",
            "trend_exclusions": sorted(EXCLUDED_FROM_TREND),
        },
        "inventory": {
            "no_cot_models": len(no_cot),
            "frontier_models": len(frontier),
            "models_with_day_level_dates": len(no_cot),
            "models_remaining_month_only": len(month_only_after),
            "explicit_date_only_overrides": len(override_rows),
            "parameter_identities_added_by_overrides": 0,
        },
        "time_horizon": {
            "month_date_approximation": time_month,
            "exact_date_approximation": time_exact,
            "adjusted_reported_law": time_law,
        },
        "token_horizon": {
            "month_date_approximation": token_month,
            "exact_date_approximation": token_exact,
            "adjusted_reported_law": token_law,
        },
        "decision": {
            "use_exact_dates_in_no_cot_evidence": True,
            "use_exact_date_adjusted_time_law": True,
            "use_exact_date_adjusted_token_law": True,
            "change_live_no_cot_weight": False,
            "reason": "Exact dates change the time-law point by about two percent and add Opus 4.7 to the exact-date Pareto frontier. This is a source-fidelity correction, not new independent evidence, so the no-CoT weight stays fixed.",
        },
        "limitations": [
            "The paper's per-bootstrap model horizons and iteration-level trend fits are not present in the submitted LaTeX archive.",
            "The exact-date adjustment transfers a deterministic slope ratio onto the published bootstrap median and interval; it does not recreate the original bootstrap.",
            "Release date is a public-availability proxy, not a training-completion date.",
        ],
        "outputs": {"model_audit": str(MODEL_AUDIT.relative_to(ROOT))},
        "source_hashes": {
            str(OBSERVATIONS.relative_to(ROOT)): sha256(OBSERVATIONS),
            str(OVERRIDES.relative_to(ROOT)): sha256(OVERRIDES),
            str(METADATA.relative_to(ROOT)): sha256(METADATA),
            str(MODEL_AUDIT.relative_to(ROOT)): sha256(MODEL_AUDIT),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(RESULT),
                "exact_dates": len(no_cot),
                "remaining_month_only": len(month_only_after),
                "time_days_paper": PAPER_TIME_DAYS,
                "time_days_adjusted": time_law["adjusted_point_days"],
                "token_days_paper": PAPER_TOKEN_DAYS,
                "token_days_adjusted": token_law["adjusted_point_days"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
