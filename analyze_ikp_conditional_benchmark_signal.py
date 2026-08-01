#!/usr/bin/env python3
"""Test whether IKP adds size information beyond standard benchmarks and date.

The upstream IKP repository contains a useful but previously unused comparison
panel of vendor-reported MMLU, MMLU-Pro, GPQA Diamond, and SimpleQA scores.  This
script first reproduces every generated upstream table from the pinned inputs,
audits the raw markdown citation tables, and records discrepancies in the
upstream narrative summary.  It then collapses explicit thinking/non-thinking
serving variants and runs strictly earlier, vendor-held-out parameter
predictions.

The held-out comparison is deliberately conditional: benchmark score plus an
exact day-level release date is the baseline; IKP is the candidate addition.
MoE architecture and equal-vendor training weights are separate sensitivity
specifications.  No target checkpoint can appear in its own training set, and
no checkpoint from the target vendor can appear in that fold.
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


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"

DENSING = ROOT / "sources/ikp_upstream_densing_analysis_2026-07-18.csv"
BENCHMARK_SCORES = ROOT / "sources/ikp_upstream_benchmark_scores_2026-07-18.csv"
PINNED_JOINED = ROOT / "sources/ikp_upstream_benchmark_joined_2026-07-18.csv"
PINNED_REGRESSION = ROOT / "sources/ikp_upstream_benchmark_regression_summary_2026-07-18.csv"
PINNED_TIME = ROOT / "sources/ikp_upstream_benchmark_time_coefficients_2026-07-18.csv"
NARRATIVE = ROOT / "sources/ikp_upstream_benchmark_summary_2026-07-18.md"
SOURCE_METADATA = ROOT / "sources/ikp_source_metadata_2026-07-18.json"
RAW_MARKDOWN = (
    ROOT / "sources/ikp_upstream_benchmark_raw_anthropic_2026-07-18.md",
    ROOT / "sources/ikp_upstream_benchmark_raw_deepseek_qwen_kimi_glm_2026-07-18.md",
    ROOT / "sources/ikp_upstream_benchmark_raw_google_meta_2026-07-18.md",
    ROOT / "sources/ikp_upstream_benchmark_raw_openai_2026-07-18.md",
    ROOT / "sources/ikp_upstream_benchmark_raw_others_2026-07-18.md",
)

RESULT = OUT / "ikp_conditional_benchmark_signal_audit_2026-07-18.json"
PREDICTIONS = OUT / "ikp_conditional_benchmark_predictions_2026-07-18.csv"
SITE_OUTPUT = ROOT / "site/public/data/ikp-conditional-benchmark-signal.json"
PARAMETER_SIGNAL_AUDIT = OUT / "ikp_parameter_signal_audit_2026-07-18.json"

BENCHMARKS = ("mmlu", "mmlu_pro", "gpqa_diamond", "simpleqa")
BENCHMARK_LABELS = {
    "mmlu": "MMLU",
    "mmlu_pro": "MMLU-Pro",
    "gpqa_diamond": "GPQA Diamond",
    "simpleqa": "SimpleQA",
}
CALIBRATION_EXCLUDE = {
    "minimax-m1-think",
    "hunyuan-a13b",
    "hunyuan-a13b-think",
    "hermes-3-405b",
    "ling-2.6-flash",
    "deepseek-v3.1-nex-n1",
    "intellect-3-think",
}

MIN_TRAIN_ROWS = 10
MIN_TRAIN_VENDORS = 5
BOOTSTRAP_SAMPLES = 20_000
RANDOM_SEED = 20260718
DATE_ORIGIN = date(2024, 1, 1)

# Raw source tables use more descriptive labels for these benchmark rows.
RAW_MODEL_ALIASES = {
    "deepseek-v3": "DeepSeek-V3 (671B-A37B)",
    "deepseek-r1-think": "DeepSeek-R1",
    "deepseek-r1-distill-llama-70b-think": "R1-Distill-Llama-70B",
    "deepseek-r1-distill-qwen-32b-think": "R1-Distill-Qwen-32B",
    "qwen-2.5-72b": "Qwen2.5-72B-Instruct",
    "qwen-2.5-7b": "Qwen2.5-7B-Instruct",
    "qwen3-32b-think": "Qwen3-32B (base)",
    "qwen3-30b-a3b-think": "Qwen3-30B-A3B (base)",
    "qwen3-235b-a22b-think": "Qwen3-235B-A22B-Thinking-2507",
    "qwen3-next-80b-a3b": "Qwen3-Next-80B-A3B-Instruct",
    "kimi-k2": "Kimi-K2-Instruct (1T-A32B)",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def markdown_number(value: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else None


def audit_raw_benchmark_tables(
    benchmark_rows: list[dict[str, str]],
) -> dict[str, Any]:
    raw_rows: dict[str, dict[str, Any]] = {}
    raw_row_count = 0
    for path in RAW_MARKDOWN:
        for line in path.read_text(encoding="utf-8").splitlines():
            if (
                not line.startswith("|")
                or line.startswith("|---")
                or "| Model |" in line
            ):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 7:
                continue
            key = normalized_name(cells[0])
            if key in raw_rows:
                raise ValueError(f"Duplicate raw benchmark model label: {cells[0]}")
            raw_rows[key] = {"cells": cells, "path": path}
            raw_row_count += 1

    metric_columns = {
        "mmlu": 1,
        "mmlu_pro": 2,
        "gpqa_diamond": 3,
        "simpleqa": 4,
        "hle": 5,
    }
    mismatches = []
    matched_score_cells = 0
    cited_urls = set()
    for benchmark in benchmark_rows:
        raw_label = RAW_MODEL_ALIASES.get(benchmark["model"], benchmark["model"])
        raw = raw_rows.get(normalized_name(raw_label))
        if raw is None:
            mismatches.append(
                {"model": benchmark["model"], "field": "model", "issue": "missing raw row"}
            )
            continue
        cells = raw["cells"]
        source = cells[6]
        if not source.startswith("http"):
            mismatches.append(
                {"model": benchmark["model"], "field": "source", "issue": source}
            )
        else:
            cited_urls.add(source)
        for metric, index in metric_columns.items():
            expected = optional_float(benchmark[metric])
            observed = markdown_number(cells[index])
            if expected is None and observed is None:
                continue
            if expected is None or observed is None or not math.isclose(
                expected, observed, rel_tol=0, abs_tol=1e-12
            ):
                mismatches.append(
                    {
                        "model": benchmark["model"],
                        "field": metric,
                        "benchmark_csv": expected,
                        "raw_markdown": observed,
                        "raw_cell": cells[index],
                    }
                )
            else:
                matched_score_cells += 1
    if mismatches:
        raise ValueError(f"Raw benchmark provenance mismatch: {mismatches[:3]}")
    return {
        "benchmark_rows": len(benchmark_rows),
        "raw_markdown_rows": raw_row_count,
        "benchmark_rows_matched": len(benchmark_rows),
        "populated_score_cells_matched": matched_score_cells,
        "distinct_primary_urls": len(cited_urls),
        "mismatches": mismatches,
    }


def joined_panel(
    densing_rows: list[dict[str, str]], benchmark_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    benchmark_by_model = {row["model"]: row for row in benchmark_rows}
    if len(benchmark_by_model) != len(benchmark_rows):
        raise ValueError("Duplicate model ID in upstream benchmark score table")
    output = []
    for source in densing_rows:
        if source["model"] in CALIBRATION_EXCLUDE:
            continue
        benchmark = benchmark_by_model.get(source["model"], {})
        row: dict[str, Any] = dict(source)
        for metric in (*BENCHMARKS, "hle"):
            row[metric] = optional_float(benchmark.get(metric))
        output.append(row)
    return output


def compare_joined_output(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pinned = read_csv(PINNED_JOINED)
    fields = (
        "model",
        "params_B",
        "log10_params",
        "release_date",
        "pen_acc",
        *BENCHMARKS,
    )
    mismatches = []
    if len(rows) != len(pinned):
        mismatches.append({"field": "row_count", "reproduced": len(rows), "pinned": len(pinned)})
    for index, (left, right) in enumerate(zip(rows, pinned)):
        for field in fields:
            if field in {"model", "release_date"}:
                if str(left[field]) != right[field]:
                    mismatches.append(
                        {"row": index, "field": field, "reproduced": left[field], "pinned": right[field]}
                    )
            else:
                left_value = optional_float(str(left[field])) if left[field] is not None else None
                right_value = optional_float(right[field])
                if left_value is None and right_value is None:
                    continue
                if left_value is None or right_value is None or not math.isclose(
                    left_value, right_value, rel_tol=0, abs_tol=1e-12
                ):
                    mismatches.append(
                        {"row": index, "field": field, "reproduced": left_value, "pinned": right_value}
                    )
    if mismatches:
        raise ValueError(f"Failed to reproduce upstream joined panel: {mismatches[:3]}")
    return {
        "rows": len(rows),
        "order_identical": True,
        "fields_compared": list(fields),
        "mismatches": mismatches,
    }


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Numerically stable continued fraction used by incomplete beta."""

    maximum_iterations = 400
    epsilon = 3e-14
    floor = 1e-300
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    h = d
    for iteration in range(1, maximum_iterations + 1):
        twice = 2 * iteration
        term = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
        d = 1.0 + term * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + term / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        h *= d * c
        term = -(a + iteration) * (qab + iteration) * x / (
            (a + twice) * (qap + twice)
        )
        d = 1.0 + term * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + term / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            return h
    raise ArithmeticError("Incomplete beta continued fraction did not converge")


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return front * beta_continued_fraction(a, b, x) / a
    return 1 - front * beta_continued_fraction(b, a, 1 - x) / b


def student_t_two_sided_p(t_statistic: float, degrees_of_freedom: int) -> float:
    x = degrees_of_freedom / (degrees_of_freedom + t_statistic**2)
    return regularized_incomplete_beta(x, degrees_of_freedom / 2, 0.5)


def simple_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    y_centered = y - y_mean
    ss_x = float(x_centered @ x_centered)
    ss_y = float(y_centered @ y_centered)
    ss_xy = float(x_centered @ y_centered)
    slope = ss_xy / ss_x
    intercept = y_mean - slope * x_mean
    r_value = ss_xy / math.sqrt(ss_x * ss_y)
    degrees_of_freedom = len(x) - 2
    stderr = math.sqrt((1 - r_value**2) * ss_y / ss_x / degrees_of_freedom)
    t_statistic = abs(r_value) * math.sqrt(
        degrees_of_freedom / ((1 - r_value) * (1 + r_value))
    )
    p_value = student_t_two_sided_p(t_statistic, degrees_of_freedom)
    return {
        "n": len(x),
        "r2": round(float(r_value**2), 4),
        "slope": round(float(slope), 4),
        "intercept": float(intercept),
        "p": f"{p_value:.2e}",
        "stderr": float(stderr),
    }


def time_fit(x_params: np.ndarray, x_months: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    design = np.column_stack([np.ones_like(x_params), x_params, x_months])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = float(np.sum((y - fitted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": len(y),
        "intercept": round(float(coefficients[0]), 4),
        "slope_params": round(float(coefficients[1]), 4),
        "slope_months": round(float(coefficients[2]), 4),
        "r2": round(float(1 - residual / total), 4),
    }


def compare_generated_rows(
    reproduced: list[dict[str, Any]], pinned_path: Path
) -> dict[str, Any]:
    pinned = read_csv(pinned_path)
    mismatches = []
    if len(reproduced) != len(pinned):
        mismatches.append(
            {"field": "row_count", "reproduced": len(reproduced), "pinned": len(pinned)}
        )
    for index, (left, right) in enumerate(zip(reproduced, pinned)):
        if left["metric"] != right["metric"]:
            mismatches.append(
                {"row": index, "field": "metric", "reproduced": left["metric"], "pinned": right["metric"]}
            )
            continue
        for field, value in left.items():
            if field == "metric":
                continue
            if field == "p":
                if value != right[field]:
                    mismatches.append(
                        {"row": index, "field": field, "reproduced": value, "pinned": right[field]}
                    )
                continue
            pinned_value = float(right[field])
            tolerance = 0 if field in {"n", "r2", "slope", "slope_params", "slope_months"} else 1e-12
            if not math.isclose(float(value), pinned_value, rel_tol=0, abs_tol=tolerance):
                mismatches.append(
                    {"row": index, "field": field, "reproduced": value, "pinned": pinned_value}
                )
    if mismatches:
        raise ValueError(f"Failed to reproduce {pinned_path.name}: {mismatches[:3]}")
    return {"rows": len(reproduced), "mismatches": mismatches, "exact_after_upstream_rounding": True}


def reproduce_upstream_fits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x_full = np.asarray([float(row["log10_params"]) for row in rows])
    ikp_full = np.asarray([float(row["pen_acc"]) * 100 for row in rows])
    regression_rows = [{"metric": "IKP (full set)", **simple_fit(x_full, ikp_full)}]
    for benchmark in BENCHMARKS:
        subset = [row for row in rows if row[benchmark] is not None]
        x = np.asarray([float(row["log10_params"]) for row in subset])
        benchmark_y = np.asarray([float(row[benchmark]) for row in subset])
        ikp_y = np.asarray([float(row["pen_acc"]) * 100 for row in subset])
        regression_rows.append({"metric": benchmark, **simple_fit(x, benchmark_y)})
        regression_rows.append(
            {"metric": f"IKP (subset matching {benchmark})", **simple_fit(x, ikp_y)}
        )

    time_rows = []
    for benchmark in BENCHMARKS:
        subset = [row for row in rows if row[benchmark] is not None]
        x = np.asarray([float(row["log10_params"]) for row in subset])
        months = np.asarray([float(row["months"]) for row in subset])
        benchmark_y = np.asarray([float(row[benchmark]) for row in subset])
        ikp_y = np.asarray([float(row["pen_acc"]) * 100 for row in subset])
        time_rows.append({"metric": benchmark, **time_fit(x, months, benchmark_y)})
        time_rows.append({"metric": f"IKP (subset {benchmark})", **time_fit(x, months, ikp_y)})
    time_rows.append(
        {
            "metric": "IKP (full set)",
            **time_fit(
                x_full,
                np.asarray([float(row["months"]) for row in rows]),
                ikp_full,
            ),
        }
    )
    return {
        "regression_summary": {
            "rows": regression_rows,
            "reproduction": compare_generated_rows(regression_rows, PINNED_REGRESSION),
        },
        "time_coefficients": {
            "rows": time_rows,
            "reproduction": compare_generated_rows(time_rows, PINNED_TIME),
        },
    }


def parse_signed_number(value: str) -> float:
    return float(value.replace("−", "-"))


def audit_narrative_summary(
    panel_rows: int, full_r2: float, full_time_slope: float, raw_rows: int
) -> dict[str, Any]:
    text = NARRATIVE.read_text(encoding="utf-8")
    setup = re.search(r"\((\d+) calibration models after applying", text)
    table = re.search(
        r"\*\*IKP \(full set\)\*\* \| \*\*(\d+)\*\* \| \*\*([0-9.]+)\*\* \| — \| \*\*([+−\-0-9.]+)\*\*",
        text,
    )
    interpretation = re.search(r"\(([-+−0-9.]+) on the full set", text)
    methodology = re.search(r"identical methodology across (\d+) models", text)
    if not all((setup, table, interpretation, methodology)):
        raise ValueError("Could not parse pinned upstream benchmark narrative")

    claims = [
        {
            "claim": "setup_post_exclusion_configurations",
            "narrative": int(setup.group(1)),
            "generated_output": panel_rows,
        },
        {
            "claim": "headline_full_set_configurations",
            "narrative": int(table.group(1)),
            "generated_output": panel_rows,
        },
        {
            "claim": "headline_full_set_r_squared",
            "narrative": float(table.group(2)),
            "generated_output": full_r2,
        },
        {
            "claim": "headline_full_set_time_slope_pp_per_month",
            "narrative": parse_signed_number(table.group(3)),
            "generated_output": full_time_slope,
        },
        {
            "claim": "interpretation_full_set_time_slope_pp_per_month",
            "narrative": parse_signed_number(interpretation.group(1)),
            "generated_output": full_time_slope,
        },
        {
            "claim": "methodology_raw_configurations",
            "narrative": int(methodology.group(1)),
            "generated_output": raw_rows,
        },
    ]
    for claim in claims:
        claim["matches_generated_output"] = math.isclose(
            float(claim["narrative"]),
            float(claim["generated_output"]),
            rel_tol=0,
            abs_tol=1e-12,
        )
    stale = [claim for claim in claims if not claim["matches_generated_output"]]
    return {
        "authority_policy": "Pinned generated CSV outputs override the stale narrative summary.",
        "claims": claims,
        "stale_claims": stale,
        "stale_claim_count": len(stale),
        "all_claims_match_generated_outputs": not stale,
    }


def base_key(model: str) -> str:
    return model.removesuffix("-think")


def collapse_weight_bases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[base_key(row["model"])].append(row)
    output = []
    identity_fields = ("vendor", "family", "arch", "params_B", "active_B", "release_date")
    for key, variants in sorted(groups.items()):
        for field in identity_fields:
            if len({row[field] for row in variants}) != 1:
                raise ValueError(f"Serving-variant identity mismatch for {key}: {field}")
        benchmark_values: dict[str, float | None] = {}
        for benchmark in BENCHMARKS:
            values = [float(row[benchmark]) for row in variants if row[benchmark] is not None]
            if values and max(values) - min(values) > 1e-12:
                raise ValueError(f"Conflicting benchmark values within {key}: {benchmark}")
            benchmark_values[benchmark] = values[0] if values else None
        release_date = variants[0]["release_date"]
        output.append(
            {
                "base_key": key,
                "variants": sorted(row["model"] for row in variants),
                "variant_count": len(variants),
                "vendor": variants[0]["vendor"],
                "family": variants[0]["family"],
                "arch": variants[0]["arch"],
                "moe": int(variants[0]["arch"] == "moe"),
                "params_b": float(variants[0]["params_B"]),
                "active_b": float(variants[0]["active_B"]),
                "release_date": release_date,
                "release_day_years": (date.fromisoformat(release_date) - DATE_ORIGIN).days / 365.25,
                "ikp_score": float(np.mean([float(row["pen_acc"]) * 100 for row in variants])),
                **benchmark_values,
            }
        )
    return output


def fit_prediction(
    train: list[dict[str, Any]],
    test: dict[str, Any],
    benchmark: str,
    include_architecture: bool,
    include_ikp: bool,
    vendor_balanced: bool,
) -> tuple[float, list[float]]:
    features = [benchmark, "release_day_years"]
    if include_architecture:
        features.append("moe")
    if include_ikp:
        features.append("ikp_score")
    design = np.asarray(
        [[1.0, *[float(row[field]) for field in features]] for row in train],
        dtype=float,
    )
    target = np.log10([float(row["params_b"]) for row in train])
    if vendor_balanced:
        counts = Counter(row["vendor"] for row in train)
        weights = np.asarray([1 / counts[row["vendor"]] for row in train])
        root_weights = np.sqrt(weights)
        design = design * root_weights[:, None]
        target = target * root_weights
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    test_design = np.asarray(
        [1.0, *[float(test[field]) for field in features]], dtype=float
    )
    return float(test_design @ coefficients), [float(value) for value in coefficients]


def strict_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for benchmark in BENCHMARKS:
        panel = [row for row in rows if row[benchmark] is not None]
        for test in sorted(panel, key=lambda row: (row["release_date"], row["base_key"])):
            train = [
                row
                for row in panel
                if row["release_date"] < test["release_date"]
                and row["vendor"] != test["vendor"]
            ]
            if (
                len(train) < MIN_TRAIN_ROWS
                or len({row["vendor"] for row in train}) < MIN_TRAIN_VENDORS
            ):
                continue
            for include_architecture in (False, True):
                specification = "score_date_arch" if include_architecture else "score_date"
                for vendor_balanced in (False, True):
                    training_weighting = "vendor_equal" if vendor_balanced else "row_equal"
                    baseline_log10, baseline_coefficients = fit_prediction(
                        train,
                        test,
                        benchmark,
                        include_architecture,
                        False,
                        vendor_balanced,
                    )
                    candidate_log10, candidate_coefficients = fit_prediction(
                        train,
                        test,
                        benchmark,
                        include_architecture,
                        True,
                        vendor_balanced,
                    )
                    actual_log10 = math.log10(test["params_b"])
                    baseline_error = baseline_log10 - actual_log10
                    candidate_error = candidate_log10 - actual_log10
                    output.append(
                        {
                            "benchmark": benchmark,
                            "benchmark_label": BENCHMARK_LABELS[benchmark],
                            "specification": specification,
                            "training_weighting": training_weighting,
                            "base_key": test["base_key"],
                            "variants": "|".join(test["variants"]),
                            "release_date": test["release_date"],
                            "vendor": test["vendor"],
                            "family": test["family"],
                            "arch": test["arch"],
                            "moe": test["moe"],
                            "benchmark_score": test[benchmark],
                            "ikp_score": test["ikp_score"],
                            "actual_b": test["params_b"],
                            "baseline_predicted_b": float(10**baseline_log10),
                            "candidate_predicted_b": float(10**candidate_log10),
                            "baseline_log10_error": float(baseline_error),
                            "candidate_log10_error": float(candidate_error),
                            "baseline_abs_log10_error": float(abs(baseline_error)),
                            "candidate_abs_log10_error": float(abs(candidate_error)),
                            "candidate_minus_baseline_abs_log10_error": float(
                                abs(candidate_error) - abs(baseline_error)
                            ),
                            "train_rows": len(train),
                            "train_vendors": len({row["vendor"] for row in train}),
                            "train_families": len({row["family"] for row in train}),
                            "train_max_date": max(row["release_date"] for row in train),
                            "test_vendor_excluded": all(
                                row["vendor"] != test["vendor"] for row in train
                            ),
                            "baseline_coefficients_json": json.dumps(baseline_coefficients),
                            "candidate_coefficients_json": json.dumps(candidate_coefficients),
                        }
                    )
    return output


def prediction_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "vendors": 0}
    errors = np.asarray(
        [math.log10(float(row[field]) / float(row["actual_b"])) for row in rows]
    )
    absolute = np.abs(errors)
    vendor_absolute: dict[str, list[float]] = defaultdict(list)
    for row, error in zip(rows, absolute):
        vendor_absolute[row["vendor"]].append(float(error))
    return {
        "n": len(rows),
        "vendors": len(vendor_absolute),
        "families": len({row["family"] for row in rows}),
        "mean_absolute_log10_error": float(np.mean(absolute)),
        "equal_vendor_mean_absolute_log10_error": float(
            np.mean([np.mean(values) for values in vendor_absolute.values()])
        ),
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "p80_multiplicative_error": float(10 ** np.quantile(absolute, 0.8)),
        "rmse_log10": float(np.sqrt(np.mean(errors**2))),
        "within_2x": float(np.mean(absolute <= math.log10(2))),
        "signed_bias_factor": float(10 ** np.mean(errors)),
    }


def vendor_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_vendor: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_vendor[row["vendor"]].append(
            float(row["candidate_minus_baseline_abs_log10_error"])
        )
    if not by_vendor:
        return {"vendors": 0, "samples": 0}
    vendor_means = np.asarray([np.mean(values) for values in by_vendor.values()])
    rng = np.random.default_rng(RANDOM_SEED)
    draws = rng.choice(
        vendor_means,
        size=(BOOTSTRAP_SAMPLES, len(vendor_means)),
        replace=True,
    ).mean(axis=1)
    return {
        "metric": "equal-vendor mean absolute log10 error; IKP candidate minus benchmark/date baseline",
        "observed_delta": float(vendor_means.mean()),
        "ci_90": [float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))],
        "probability_candidate_better": float(np.mean(draws < 0)),
        "vendors": len(vendor_means),
        "samples": BOOTSTRAP_SAMPLES,
        "random_seed": RANDOM_SEED,
    }


def summarize_holdouts(
    weight_bases: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        panel = [row for row in weight_bases if row[benchmark] is not None]
        benchmark_predictions = [row for row in predictions if row["benchmark"] == benchmark]
        specifications: dict[str, Any] = {}
        for training_weighting in ("row_equal", "vendor_equal"):
            for specification in ("score_date", "score_date_arch"):
                selected = [
                    row
                    for row in benchmark_predictions
                    if row["training_weighting"] == training_weighting
                    and row["specification"] == specification
                ]
                bootstrap = vendor_bootstrap(selected)
                pass_gate = bool(
                    len(selected) >= 12
                    and len({row["vendor"] for row in selected}) >= 8
                    and bootstrap.get("ci_90", [1, 1])[1] < 0
                    and bootstrap.get("probability_candidate_better", 0) >= 0.90
                )
                specifications[f"{training_weighting}__{specification}"] = {
                    "baseline": prediction_metrics(selected, "baseline_predicted_b"),
                    "candidate_with_ikp": prediction_metrics(
                        selected, "candidate_predicted_b"
                    ),
                    "paired_vendor_bootstrap": bootstrap,
                    "passes_predeclared_gate": pass_gate,
                }
        output[benchmark] = {
            "label": BENCHMARK_LABELS[benchmark],
            "weight_base_panel_rows": len(panel),
            "panel_vendors": len({row["vendor"] for row in panel}),
            "strict_prediction_models": len(
                {
                    row["base_key"]
                    for row in benchmark_predictions
                    if row["training_weighting"] == "row_equal"
                    and row["specification"] == "score_date"
                }
            ),
            "strict_prediction_vendors": len(
                {
                    row["vendor"]
                    for row in benchmark_predictions
                    if row["training_weighting"] == "row_equal"
                    and row["specification"] == "score_date"
                }
            ),
            "specifications": specifications,
            "passing_specifications": sum(
                row["passes_predeclared_gate"] for row in specifications.values()
            ),
        }
    return output


def write_prediction_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "benchmark",
        "benchmark_label",
        "specification",
        "training_weighting",
        "base_key",
        "variants",
        "release_date",
        "vendor",
        "family",
        "arch",
        "moe",
        "benchmark_score",
        "ikp_score",
        "actual_b",
        "baseline_predicted_b",
        "candidate_predicted_b",
        "baseline_log10_error",
        "candidate_log10_error",
        "baseline_abs_log10_error",
        "candidate_abs_log10_error",
        "candidate_minus_baseline_abs_log10_error",
        "train_rows",
        "train_vendors",
        "train_families",
        "train_max_date",
        "test_vendor_excluded",
        "baseline_coefficients_json",
        "candidate_coefficients_json",
    ]
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    densing_rows = read_csv(DENSING)
    benchmark_rows = read_csv(BENCHMARK_SCORES)
    if len(densing_rows) != 100 or len({row["model"] for row in densing_rows}) != 100:
        raise ValueError("Expected 100 unique upstream IKP configurations")
    if len(benchmark_rows) != 81 or len({row["model"] for row in benchmark_rows}) != 81:
        raise ValueError("Expected 81 unique upstream benchmark rows")

    raw_provenance = audit_raw_benchmark_tables(benchmark_rows)
    joined = joined_panel(densing_rows, benchmark_rows)
    joined_reproduction = compare_joined_output(joined)
    fit_reproduction = reproduce_upstream_fits(joined)
    full_regression = fit_reproduction["regression_summary"]["rows"][0]
    full_time = fit_reproduction["time_coefficients"]["rows"][-1]
    narrative_audit = audit_narrative_summary(
        len(joined),
        full_regression["r2"],
        full_time["slope_months"],
        len(densing_rows),
    )

    weight_bases = collapse_weight_bases(joined)
    if len(weight_bases) != 87:
        raise ValueError(f"Expected 87 distinct IKP weight bases, found {len(weight_bases)}")
    predictions = strict_predictions(weight_bases)
    if not all(row["train_max_date"] < row["release_date"] for row in predictions):
        raise ValueError("Conditional IKP chronology violation")
    if not all(row["test_vendor_excluded"] for row in predictions):
        raise ValueError("Conditional IKP vendor-holdout violation")
    holdout = summarize_holdouts(weight_bases, predictions)

    gpqa_all = holdout["gpqa_diamond"]["passing_specifications"] == 4
    mmlu_supportive = holdout["mmlu"]["passing_specifications"] >= 3
    primary_benchmarks_pass = all(
        holdout[benchmark]["specifications"]["row_equal__score_date"][
            "passes_predeclared_gate"
        ]
        for benchmark in ("mmlu", "gpqa_diamond")
    )
    conditional_corroboration = bool(gpqa_all and mmlu_supportive and primary_benchmarks_pass)
    parameter_signal = json.loads(
        PARAMETER_SIGNAL_AUDIT.read_text(encoding="utf-8")
    )
    parameter_decision = parameter_signal["decision"]
    retained_final_weight = float(
        parameter_decision["incremental_final_weight_when_crowd_is_50pct"]
    )
    if not math.isclose(
        retained_final_weight,
        float(parameter_decision["incremental_evidence_weight"]) * 0.5,
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError("Primary IKP audit has inconsistent evidence/final weights")

    source_paths = (
        DENSING,
        BENCHMARK_SCORES,
        PINNED_JOINED,
        PINNED_REGRESSION,
        PINNED_TIME,
        NARRATIVE,
        SOURCE_METADATA,
        PARAMETER_SIGNAL_AUDIT,
        *RAW_MARKDOWN,
    )
    result = {
        "generated_on": "2026-07-18",
        "question": (
            "Does IKP add held-out parameter-count information beyond a standard "
            "knowledge benchmark, exact release date, and architecture sensitivity?"
        ),
        "protocol": {
            "target": "log10 total parameters in billions",
            "baseline": "benchmark score plus exact day-level release date",
            "candidate": "baseline plus collapsed mean IKP score",
            "architecture_sensitivity": "MoE binary included in both baseline and candidate",
            "training_weight_sensitivity": "row-equal OLS and equal-total-weight-per-vendor WLS",
            "test_fold": "release dates strictly earlier than test and test vendor fully excluded",
            "minimum_train_rows": MIN_TRAIN_ROWS,
            "minimum_train_vendors": MIN_TRAIN_VENDORS,
            "bootstrap": "equal-vendor cluster bootstrap of paired absolute log10 error",
            "promotion_gate": (
                "at least 12 test bases from 8 vendors, candidate CI90 upper bound below zero, "
                "and >=90% bootstrap probability of improvement"
            ),
        },
        "source_inventory": {
            "raw_ikp_configurations": len(densing_rows),
            "benchmark_rows": len(benchmark_rows),
            "post_exclusion_configurations": len(joined),
            "post_collapse_weight_bases": len(weight_bases),
            "serving_variants_collapsed": len(joined) - len(weight_bases),
            "excluded_configurations": sorted(CALIBRATION_EXCLUDE),
            "raw_benchmark_provenance": raw_provenance,
        },
        "upstream_reproduction": {
            "joined_panel": joined_reproduction,
            **fit_reproduction,
            "narrative_summary_audit": narrative_audit,
        },
        "heldout_results": holdout,
        "decision": {
            "conditional_incremental_signal_corroborated": conditional_corroboration,
            "gpqa_passes_all_four_sensitivity_specifications": gpqa_all,
            "mmlu_passing_specifications_out_of_four": holdout["mmlu"][
                "passing_specifications"
            ],
            "mmlu_pro_passing_specifications_out_of_four": holdout["mmlu_pro"][
                "passing_specifications"
            ],
            "simpleqa_has_sufficient_strict_predictions": holdout["simpleqa"][
                "strict_prediction_models"
            ]
            >= 12,
            "change_live_ikp_weight": False,
            "retain_current_final_fable_ikp_weight": retained_final_weight,
            "primary_parameter_signal_promoted": parameter_decision[
                "promote_incremental_ikp_weight"
            ],
            "primary_parameter_signal_reason": parameter_decision["reason"],
            "reason": (
                "GPQA shows robust incremental IKP information in all four date/architecture/"
                "training-weight sensitivities, and MMLU passes three of four. This strengthens "
                "the case that IKP is not merely a standard-benchmark proxy. The conditional "
                "audit does not independently set or override the live weight: it retains the "
                "primary IKP audit's current decision because both analyses use the same IKP "
                "source, Fable remains an extrapolation, and no Sol IKP result exists."
            ),
        },
        "limitations": [
            "Vendor-reported benchmark scores are not a randomized or uniformly scaffolded panel.",
            "MMLU-Pro does not pass the cluster-interval gate, and SimpleQA is too sparse for strict folds.",
            "The MoE control is binary and cannot identify undisclosed active fractions.",
            "This audit validates conditional signal on open-weight bases; it is not an independent Fable measurement.",
            "Upstream generated CSVs are internally reproducible, but the pinned narrative summary is stale and is not treated as evidence.",
        ],
        "outputs": {"predictions": str(PREDICTIONS.relative_to(ROOT))},
        "source_files": {
            str(path.relative_to(ROOT)): sha256(path) for path in source_paths
        },
    }

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    SITE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_prediction_csv(predictions)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    RESULT.write_text(encoded, encoding="utf-8")
    SITE_OUTPUT.write_text(encoded, encoding="utf-8")
    print(f"Wrote {RESULT}")
    print(f"Wrote {PREDICTIONS}")
    print(f"Wrote {SITE_OUTPUT}")
    print(
        json.dumps(
            {
                "conditional_incremental_signal_corroborated": conditional_corroboration,
                "gpqa_passing_specs": holdout["gpqa_diamond"]["passing_specifications"],
                "mmlu_passing_specs": holdout["mmlu"]["passing_specifications"],
                "mmlu_pro_passing_specs": holdout["mmlu_pro"]["passing_specifications"],
                "stale_upstream_narrative_claims": narrative_audit["stale_claim_count"],
                "change_live_ikp_weight": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
