#!/usr/bin/env python3
"""Audit post-training and asserted shared-base distortions in scale inference.

The audit separates three evidence grades that must not be conflated:

1. Epoch-structured, open-weight base -> descendant links with unchanged total
   parameter counts.  These are the only model-lineage observations admitted to
   the backtest.
2. Same-checkpoint reasoning/non-reasoning configurations.  These measure
   inference-budget uplift, not training-stage uplift.
3. User-supplied shared-base claims for proprietary GPT-5 and Opus lineages.
   These are reported as sensitivities unless a primary source explicitly says
   that the underlying pretrained model is the same.

Every fitted prediction uses only calibration checkpoints released strictly
before the parent model and excludes the endpoint's developer/family.  The
candidate replaces a descendant score with the arithmetic parent/child score
mean while retaining the descendant date.  This mirrors a conservative lineage
collapse without assuming that the parent is a pure pretrained checkpoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
EPOCH = ROOT / "sources/epoch_all_ai_models_2026-07-31.csv"
ECI_COMPONENTS = ROOT / "sources/epoch_eci_benchmarks_2026-07-31.csv"
ECI_BENCHMARK_ZIP = ROOT / "sources/epoch_benchmark_data_2026-07-31.zip"
AA_DETAIL = ROOT / "sources/aa_detailed_model_signals_2026-07-31.csv"
UNIFIED = OUT / "unified_model_observations_compute_enriched_2026-07-17.csv"
ECI_PANEL = OUT / "eci_component_expanded_parameter_panel_2026-07-18.csv"
AA_PANEL = OUT / "aa_detailed_parameter_panel_2026-07-18.csv"
AA_AUDIT = OUT / "aa_inference_budget_audit_2026-07-18.json"

RESULT = OUT / "posttraining_lineage_audit_2026-07-18.json"
EDGES = OUT / "posttraining_lineage_edges_2026-07-18.csv"
MEASUREMENTS = OUT / "posttraining_lineage_measurements_2026-07-18.csv"
PREDICTIONS = OUT / "posttraining_lineage_predictions_2026-07-18.csv"
FRONTIER_SENSITIVITY = OUT / "frontier_shared_base_sensitivity_2026-07-18.csv"
EVIDENCE = OUT / "frontier_lineage_evidence_2026-07-18.csv"

DATE_ORIGIN = date(2023, 1, 1)
BOOTSTRAPS = 20_000
MIN_ECI_TRAIN = 12
MIN_ECI_FAMILIES = 5
MIN_AA_TRAIN = 30
MIN_AA_CREATORS = 8
MIN_VERIFIED_BASES_FOR_PROMOTION = 8
MIN_SIGNAL_BASES_FOR_PROMOTION = 6

KNOWLEDGE_BENCHMARKS = {
    "MMLU",
    "GPQA diamond",
    "SimpleQA Verified",
    "TriviaQA",
    "OpenBookQA",
    "ScienceQA",
}
PRETRAINING_LIKE_BENCHMARKS = KNOWLEDGE_BENCHMARKS | {
    "ARC AI2",
    "BBH",
    "GSM8K",
    "HellaSwag",
    "LAMBADA",
    "PIQA",
    "Winogrande",
}

FRONTIER_CHAINS = {
    "Claude Opus 4.5-4.8": {
        "creator": "anthropic",
        "evidence_grade": "user_asserted_not_publicly_disclosed",
        "max_reasoning_slugs": [
            "claude-opus-4-5-thinking",
            "claude-opus-4-6-adaptive",
            "claude-opus-4-7",
            "claude-opus-4-8",
        ],
        "nonreasoning_slugs": [
            "claude-opus-4-5",
            "claude-opus-4-6",
            "claude-opus-4-7-non-reasoning",
        ],
    },
    "GPT-5 through GPT-5.5": {
        "creator": "openai",
        "evidence_grade": "user_asserted_not_publicly_disclosed",
        "max_reasoning_slugs": [
            "gpt-5",
            "gpt-5-1",
            "gpt-5-2",
            "gpt-5-3-codex",
            "gpt-5-4",
            "gpt-5-5",
        ],
        "nonreasoning_slugs": [
            "gpt-5-minimal",
            "gpt-5-1-non-reasoning",
            "gpt-5-2-non-reasoning",
            "gpt-5-4-non-reasoning",
            "gpt-5-5-non-reasoning",
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def mode_stripped_name(value: Any) -> str:
    text = str(value).lower().replace("non-reasoning", "non reasoning")
    text = re.sub(
        r"\([^)]*\b(max|xhigh|high|medium|low|minimal|reasoning)\b[^)]*\)",
        "",
        text,
    )
    text = re.sub(
        r"\b(non reasoning|reasoning|thinking|instruct|vision|preview|it)\b",
        "",
        text,
    )
    return normalize_name(text)


def years(value: str) -> float:
    return (date.fromisoformat(value) - DATE_ORIGIN).days / 365.25


def number(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if not np.isfinite(parsed) else float(parsed)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty audit output: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def parameter_metrics(errors: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(errors), dtype=float)
    if not len(values):
        return {"n": 0}
    absolute = np.abs(values)
    return {
        "n": int(len(values)),
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "mean_absolute_log10_error": float(np.mean(absolute)),
        "geomean_multiplicative_error": float(10 ** np.mean(absolute)),
        "p80_multiplicative_error": float(10 ** np.quantile(absolute, 0.8)),
        "signed_bias_factor": float(10 ** np.mean(values)),
        "within_2x": float(np.mean(absolute <= math.log10(2))),
    }


def equal_cluster_bootstrap(
    rows: list[dict[str, Any]], candidate: str, baseline: str, seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["base_cluster_id"]].append(row)
    effects = np.asarray(
        [
            np.mean(
                [abs(item[candidate]) - abs(item[baseline]) for item in items]
            )
            for _, items in sorted(grouped.items())
        ],
        dtype=float,
    )
    if len(effects) < 2:
        return {
            "metric": "equal-base mean absolute log10 error; candidate minus baseline",
            "observed_delta": float(np.mean(effects)) if len(effects) else None,
            "ci_90": [None, None],
            "bootstrap_probability_candidate_better": None,
            "bases": int(len(effects)),
            "samples": 0,
        }
    rng = np.random.default_rng(seed)
    draws = np.mean(
        effects[rng.integers(0, len(effects), size=(BOOTSTRAPS, len(effects)))],
        axis=1,
    )
    return {
        "metric": "equal-base mean absolute log10 error; candidate minus baseline",
        "observed_delta": float(np.mean(effects)),
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "bootstrap_probability_candidate_better": float(np.mean(draws < 0)),
        "bases": int(len(effects)),
        "samples": BOOTSTRAPS,
    }


def build_epoch_edges(
    epoch: pd.DataFrame, unified: pd.DataFrame
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_normalized: dict[str, list[int]] = defaultdict(list)
    for index, row in epoch.iterrows():
        by_normalized[normalize_name(row["Model"])].append(index)

    epoch_observations = unified[
        (unified["source"] == "Epoch") & (unified["record_type"] == "model")
    ]
    checkpoint_by_name: dict[str, str] = {}
    for name, group in epoch_observations.groupby("source_model_name"):
        checkpoints = group["canonical_checkpoint_id"].dropna().unique()
        if len(checkpoints) == 1:
            checkpoint_by_name[str(name)] = str(checkpoints[0])

    inventory = Counter()
    inventory["epoch_rows"] = len(epoch)
    inventory["base_model_links"] = int(epoch["Base model"].notna().sum())
    output = []
    for child_index, child in epoch[epoch["Base model"].notna()].iterrows():
        candidates = by_normalized.get(normalize_name(child["Base model"]), [])
        if len(candidates) != 1:
            continue
        inventory["unique_exact_parent_matches"] += 1
        parent_index = candidates[0]
        parent = epoch.loc[parent_index]
        child_parameters = number(child["Parameters"])
        parent_parameters = number(parent["Parameters"])
        if child_parameters is None or parent_parameters is None:
            continue
        if abs(child_parameters / parent_parameters - 1) > 0.01:
            continue
        inventory["same_parameter_links_1pct"] += 1
        if child["Open model weights?"] != "Yes" or parent["Open model weights?"] != "Yes":
            continue
        inventory["same_parameter_both_open_links"] += 1
        if "Language" not in str(child["Domain"]):
            continue
        if str(child["Publication date"]) < str(parent["Publication date"]):
            continue
        child_checkpoint = checkpoint_by_name.get(str(child["Model"]))
        parent_checkpoint = checkpoint_by_name.get(str(parent["Model"]))
        if not child_checkpoint or not parent_checkpoint:
            continue
        output.append(
            {
                "edge_id": f"{parent_checkpoint}->{child_checkpoint}",
                "base_cluster_id": parent_checkpoint,
                "child_model": str(child["Model"]),
                "child_checkpoint_id": child_checkpoint,
                "child_release_date": str(child["Publication date"]),
                "child_epoch_row": int(child_index) + 2,
                "child_organization": str(child["Organization"]),
                "child_domain": str(child["Domain"]),
                "child_task": str(child["Task"]),
                "child_parameters_b": child_parameters / 1e9,
                "child_active_parameters_b": "",
                "child_link": str(child["Link"]),
                "child_parameter_notes": str(child["Parameters notes"]),
                "finetune_compute_flop": number(child["Finetune compute (FLOP)"]) or "",
                "finetune_compute_notes": str(child["Finetune compute notes"]),
                "parent_model": str(parent["Model"]),
                "parent_checkpoint_id": parent_checkpoint,
                "parent_release_date": str(parent["Publication date"]),
                "parent_epoch_row": int(parent_index) + 2,
                "parent_organization": str(parent["Organization"]),
                "parent_parameters_b": parent_parameters / 1e9,
                "parent_link": str(parent["Link"]),
                "parameter_ratio_child_over_parent": child_parameters / parent_parameters,
                "parameter_tolerance": "within_1pct",
                "identity_evidence": "Epoch structured Base model field + unique normalized parent name",
                "admission_status": "candidate_measurement_overlap_required",
            }
        )
    return output, dict(inventory)


def measurement_maps(unified: pd.DataFrame) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for checkpoint, group in unified[unified["canonical_checkpoint_id"].notna()].groupby(
        "canonical_checkpoint_id"
    ):
        item: dict[str, Any] = {}
        for column in (
            "aa_intelligence_index",
            "eci_score",
            "nocot_time_horizon_minutes",
            "metr_p50_horizon_minutes",
        ):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            item[column] = float(values.max()) if len(values) else None
        components = group[
            (group["source"] == "ECI Component")
            & group["benchmark_name"].notna()
            & group["eci_component_performance"].notna()
        ]
        item["components"] = {
            str(row.benchmark_name): float(row.eci_component_performance)
            for row in components.itertuples()
        }
        output[str(checkpoint)] = item
    return output


def benchmark_difficulties() -> dict[str, tuple[float, float]]:
    with zipfile.ZipFile(ECI_BENCHMARK_ZIP) as archive:
        with archive.open(
            "additional_eci_data/eci_benchmark_difficulties_and_slopes.csv"
        ) as handle:
            frame = pd.read_csv(handle)
    return {
        str(row.benchmark_name): (float(row.edi), float(row.estimated_slope_scaled))
        for row in frame.itertuples()
    }


def component_implied_eci(
    performance: float, difficulty: float, slope: float
) -> float:
    clipped = min(0.999, max(0.001, performance))
    return difficulty + math.log(clipped / (1 - clipped)) / slope


def attach_measurements(
    edges: list[dict[str, Any]], maps: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    difficulties = benchmark_difficulties()
    admitted = []
    measurements = []
    for edge in edges:
        child = maps.get(edge["child_checkpoint_id"], {})
        parent = maps.get(edge["parent_checkpoint_id"], {})
        edge_measurements = []
        for column, metric, unit in (
            ("aa_intelligence_index", "AA Intelligence Index", "index points"),
            ("eci_score", "ECI aggregate", "ECI points"),
            ("nocot_time_horizon_minutes", "No-CoT horizon", "minutes"),
            ("metr_p50_horizon_minutes", "METR p50 horizon", "minutes"),
        ):
            child_value = child.get(column)
            parent_value = parent.get(column)
            if child_value is None or parent_value is None:
                continue
            edge_measurements.append(
                {
                    "edge_id": edge["edge_id"],
                    "base_cluster_id": edge["base_cluster_id"],
                    "child_model": edge["child_model"],
                    "parent_model": edge["parent_model"],
                    "metric": metric,
                    "benchmark": "",
                    "benchmark_category": "aggregate",
                    "child_value": child_value,
                    "parent_value": parent_value,
                    "difference_child_minus_parent": child_value - parent_value,
                    "ratio_child_over_parent": (
                        child_value / parent_value if parent_value != 0 else ""
                    ),
                    "child_component_implied_eci": "",
                    "parent_component_implied_eci": "",
                    "component_implied_eci_delta": "",
                    "unit": unit,
                }
            )
        child_components = child.get("components", {})
        parent_components = parent.get("components", {})
        for benchmark in sorted(set(child_components) & set(parent_components)):
            if benchmark not in difficulties:
                continue
            difficulty, slope = difficulties[benchmark]
            child_value = child_components[benchmark]
            parent_value = parent_components[benchmark]
            child_eci = component_implied_eci(child_value, difficulty, slope)
            parent_eci = component_implied_eci(parent_value, difficulty, slope)
            if benchmark in KNOWLEDGE_BENCHMARKS:
                category = "knowledge"
            elif benchmark in PRETRAINING_LIKE_BENCHMARKS:
                category = "pretraining_like_nonknowledge"
            else:
                category = "other"
            edge_measurements.append(
                {
                    "edge_id": edge["edge_id"],
                    "base_cluster_id": edge["base_cluster_id"],
                    "child_model": edge["child_model"],
                    "parent_model": edge["parent_model"],
                    "metric": "ECI component",
                    "benchmark": benchmark,
                    "benchmark_category": category,
                    "child_value": child_value,
                    "parent_value": parent_value,
                    "difference_child_minus_parent": child_value - parent_value,
                    "ratio_child_over_parent": (
                        child_value / parent_value if parent_value != 0 else ""
                    ),
                    "child_component_implied_eci": child_eci,
                    "parent_component_implied_eci": parent_eci,
                    "component_implied_eci_delta": child_eci - parent_eci,
                    "unit": "raw performance / ECI-equivalent points",
                }
            )
        if not edge_measurements:
            continue
        admitted_edge = dict(edge)
        admitted_edge["admission_status"] = "admitted_exact_open_same_parameter_lineage"
        admitted_edge["overlapping_measurements"] = len(edge_measurements)
        admitted_edge["overlapping_component_benchmarks"] = sum(
            row["metric"] == "ECI component" for row in edge_measurements
        )
        admitted.append(admitted_edge)
        measurements.extend(edge_measurements)
    return admitted, measurements


def family_weights(rows: pd.DataFrame, field: str) -> np.ndarray:
    counts = rows[field].value_counts()
    weights = np.asarray([1.0 / counts[value] for value in rows[field]], dtype=float)
    return weights / weights.mean()


def fit_eci(train: pd.DataFrame, include_date: bool) -> np.ndarray:
    x = np.asarray(
        [
            [
                1.0,
                float(row.eci_score),
                *([years(str(row.release_date))] if include_date else []),
            ]
            for row in train.itertuples()
        ],
        dtype=float,
    )
    y = np.log10(train["total_parameters_b"].to_numpy(dtype=float))
    root = np.sqrt(train["wls_weight"].to_numpy(dtype=float))
    return np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]


def predict_eci(
    score_only: np.ndarray, score_date: np.ndarray, score: float, release_date: str
) -> float:
    score_prediction = float(np.asarray([1.0, score]) @ score_only)
    date_prediction = float(
        np.asarray([1.0, score, years(release_date)]) @ score_date
    )
    return 0.60 * score_prediction + 0.40 * date_prediction


def build_eci_predictions(
    edges: list[dict[str, Any]], measurements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    panel = pd.read_csv(ECI_PANEL)
    aggregate = {
        row["edge_id"]: row
        for row in measurements
        if row["metric"] == "ECI aggregate"
    }
    output = []
    for edge in edges:
        observed = aggregate.get(edge["edge_id"])
        if not observed:
            continue
        family_candidates = panel[
            panel["model"].isin([edge["child_model"], edge["parent_model"]])
        ]["family"].unique()
        if len(family_candidates) != 1:
            continue
        family = str(family_candidates[0])
        train = panel[
            (panel["release_date"] < edge["parent_release_date"])
            & (panel["family"] != family)
        ].copy()
        if len(train) < MIN_ECI_TRAIN or train["family"].nunique() < MIN_ECI_FAMILIES:
            continue
        score_only = fit_eci(train, False)
        score_date = fit_eci(train, True)
        child_score = float(observed["child_value"])
        parent_score = float(observed["parent_value"])
        actual = math.log10(float(edge["child_parameters_b"]))
        baseline = predict_eci(
            score_only, score_date, child_score, edge["child_release_date"]
        )
        collapsed = predict_eci(
            score_only,
            score_date,
            (child_score + parent_score) / 2,
            edge["child_release_date"],
        )
        parent_only = predict_eci(
            score_only, score_date, parent_score, edge["child_release_date"]
        )
        parent_at_parent_date = predict_eci(
            score_only, score_date, parent_score, edge["parent_release_date"]
        )
        output.append(
            {
                "signal": "ECI aggregate",
                "edge_id": edge["edge_id"],
                "base_cluster_id": edge["base_cluster_id"],
                "child_model": edge["child_model"],
                "parent_model": edge["parent_model"],
                "child_release_date": edge["child_release_date"],
                "parent_release_date": edge["parent_release_date"],
                "actual_parameters_b": edge["child_parameters_b"],
                "child_score": child_score,
                "parent_score": parent_score,
                "score_delta": child_score - parent_score,
                "implied_child_over_parent_parameter_ratio": 10
                ** (baseline - parent_at_parent_date),
                "baseline_predicted_b": 10**baseline,
                "collapsed_predicted_b": 10**collapsed,
                "parent_only_predicted_b": 10**parent_only,
                "baseline_log10_error": baseline - actual,
                "collapsed_log10_error": collapsed - actual,
                "parent_only_log10_error": parent_only - actual,
                "train_n": len(train),
                "train_groups": int(train["family"].nunique()),
                "train_max_date": str(train["release_date"].max()),
                "test_group_excluded": True,
                "training_rule": "strictly earlier than parent; endpoint family excluded",
                "candidate_rule": "arithmetic parent/child score mean at child date",
            }
        )
    return output


def aa_endpoint(panel: pd.DataFrame, epoch_name: str) -> pd.Series | None:
    normalized = mode_stripped_name(epoch_name)
    candidates = panel[
        panel["selected_name"].map(mode_stripped_name) == normalized
    ]
    if candidates.empty:
        return None
    return candidates.sort_values("intelligence_index").iloc[-1]


def fit_aa(train: pd.DataFrame) -> np.ndarray:
    x = np.asarray(
        [
            [1.0, float(row.intelligence_index), years(str(row.release_date))]
            for row in train.itertuples()
        ],
        dtype=float,
    )
    y = np.log10(train["parameters_b"].to_numpy(dtype=float))
    root = np.sqrt(family_weights(train, "creator_slug"))
    return np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]


def predict_aa(beta: np.ndarray, score: float, release_date: str) -> float:
    return float(np.asarray([1.0, score, years(release_date)]) @ beta)


def build_aa_predictions(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    panel = pd.read_csv(AA_PANEL)
    output = []
    for edge in edges:
        child = aa_endpoint(panel, edge["child_model"])
        parent = aa_endpoint(panel, edge["parent_model"])
        if child is None or parent is None:
            continue
        if child["creator_slug"] != parent["creator_slug"]:
            continue
        train = panel[
            (panel["release_date"] < str(parent["release_date"]))
            & (panel["creator_slug"] != child["creator_slug"])
        ].copy()
        if len(train) < MIN_AA_TRAIN or train["creator_slug"].nunique() < MIN_AA_CREATORS:
            continue
        beta = fit_aa(train)
        child_score = float(child["intelligence_index"])
        parent_score = float(parent["intelligence_index"])
        actual = math.log10(float(edge["child_parameters_b"]))
        baseline = predict_aa(beta, child_score, str(child["release_date"]))
        collapsed = predict_aa(
            beta,
            (child_score + parent_score) / 2,
            str(child["release_date"]),
        )
        parent_only = predict_aa(beta, parent_score, str(child["release_date"]))
        parent_at_parent_date = predict_aa(
            beta, parent_score, str(parent["release_date"])
        )
        output.append(
            {
                "signal": "AA Intelligence Index",
                "edge_id": edge["edge_id"],
                "base_cluster_id": edge["base_cluster_id"],
                "child_model": edge["child_model"],
                "parent_model": edge["parent_model"],
                "child_release_date": str(child["release_date"]),
                "parent_release_date": str(parent["release_date"]),
                "actual_parameters_b": edge["child_parameters_b"],
                "child_score": child_score,
                "parent_score": parent_score,
                "score_delta": child_score - parent_score,
                "implied_child_over_parent_parameter_ratio": 10
                ** (baseline - parent_at_parent_date),
                "baseline_predicted_b": 10**baseline,
                "collapsed_predicted_b": 10**collapsed,
                "parent_only_predicted_b": 10**parent_only,
                "baseline_log10_error": baseline - actual,
                "collapsed_log10_error": collapsed - actual,
                "parent_only_log10_error": parent_only - actual,
                "train_n": len(train),
                "train_groups": int(train["creator_slug"].nunique()),
                "train_max_date": str(train["release_date"].max()),
                "test_group_excluded": True,
                "training_rule": "strictly earlier than parent; endpoint creator excluded",
                "candidate_rule": "arithmetic parent/child score mean at child date",
            }
        )
    return output


def summarize_predictions(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "bases": len({row["base_cluster_id"] for row in rows}),
        "baseline": parameter_metrics(row["baseline_log10_error"] for row in rows),
        "collapsed": parameter_metrics(row["collapsed_log10_error"] for row in rows),
        "parent_only": parameter_metrics(row["parent_only_log10_error"] for row in rows),
        "collapsed_vs_baseline": equal_cluster_bootstrap(
            rows, "collapsed_log10_error", "baseline_log10_error", seed
        ),
        "parent_only_vs_baseline": equal_cluster_bootstrap(
            rows, "parent_only_log10_error", "baseline_log10_error", seed + 1
        ),
        "median_implied_child_over_parent_parameter_ratio": (
            float(
                np.median(
                    [row["implied_child_over_parent_parameter_ratio"] for row in rows]
                )
            )
            if rows
            else None
        ),
    }


def component_category_summary(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in measurements if row["metric"] == "ECI component"]
    output = {}
    for category in ("knowledge", "pretraining_like_nonknowledge", "other"):
        selected = [row for row in rows if row["benchmark_category"] == category]
        deltas = np.asarray(
            [float(row["component_implied_eci_delta"]) for row in selected], dtype=float
        )
        output[category] = {
            "measurements": len(selected),
            "bases": len({row["base_cluster_id"] for row in selected}),
            "benchmarks": len({row["benchmark"] for row in selected}),
            "median_component_implied_eci_delta": (
                float(np.median(deltas)) if len(deltas) else None
            ),
            "mean_component_implied_eci_delta": (
                float(np.mean(deltas)) if len(deltas) else None
            ),
            "fraction_positive": float(np.mean(deltas > 0)) if len(deltas) else None,
        }
    return output


def frontier_sensitivity_rows() -> list[dict[str, Any]]:
    detail = pd.read_csv(AA_DETAIL)
    panel = pd.read_csv(AA_PANEL)
    output = []
    for chain, specification in FRONTIER_CHAINS.items():
        for mode, slug_field in (
            ("max_reasoning", "max_reasoning_slugs"),
            ("nonreasoning", "nonreasoning_slugs"),
        ):
            slugs = specification[slug_field]
            chain_rows = detail[detail["slug"].isin(slugs)].copy()
            if len(chain_rows) != len(slugs):
                missing = sorted(set(slugs) - set(chain_rows["slug"]))
                raise ValueError(f"Missing frontier chain AA rows: {chain}: {missing}")
            chain_rows = chain_rows.sort_values(["release_date", "slug"])
            first = chain_rows.iloc[0]
            train = panel[
                (panel["release_date"] < str(first["release_date"]))
                & (panel["creator_slug"] != specification["creator"])
            ].copy()
            if len(train) < MIN_AA_TRAIN or train["creator_slug"].nunique() < MIN_AA_CREATORS:
                raise ValueError(f"Insufficient calibration history for frontier chain {chain}")
            beta = fit_aa(train)
            first_implied = predict_aa(
                beta, float(first["intelligence_index"]), str(first["release_date"])
            )
            previous_implied = None
            previous_score = None
            for sequence, source in enumerate(chain_rows.itertuples(), start=1):
                implied = predict_aa(
                    beta, float(source.intelligence_index), str(source.release_date)
                )
                output.append(
                    {
                        "chain": chain,
                        "mode": mode,
                        "sequence": sequence,
                        "slug": str(source.slug),
                        "model": str(source.name),
                        "release_date": str(source.release_date),
                        "aa_intelligence_index": float(source.intelligence_index),
                        "aa_score_change_from_previous": (
                            ""
                            if previous_score is None
                            else float(source.intelligence_index) - previous_score
                        ),
                        "date_adjusted_implied_parameter_ratio_vs_first": 10
                        ** (implied - first_implied),
                        "date_adjusted_implied_parameter_ratio_vs_previous": (
                            ""
                            if previous_implied is None
                            else 10 ** (implied - previous_implied)
                        ),
                        "calibration_train_n": len(train),
                        "calibration_train_creators": int(
                            train["creator_slug"].nunique()
                        ),
                        "calibration_train_max_date": str(train["release_date"].max()),
                        "calibration_score_min": float(
                            train["intelligence_index"].min()
                        ),
                        "calibration_score_max": float(
                            train["intelligence_index"].max()
                        ),
                        "score_over_calibration_max": float(source.intelligence_index)
                        / float(train["intelligence_index"].max()),
                        "score_extrapolates_above_calibration_max": bool(
                            float(source.intelligence_index)
                            > float(train["intelligence_index"].max())
                        ),
                        "target_creator_excluded": True,
                        "evidence_grade": specification["evidence_grade"],
                        "live_weight": 0.0,
                        "interpretation": "counterfactual sensitivity if the user-supplied same-base claim is true",
                    }
                )
                previous_implied = implied
                previous_score = float(source.intelligence_index)
    return output


def evidence_rows() -> list[dict[str, Any]]:
    return [
        {
            "lineage": "Claude Opus 4.5-4.8",
            "claim": "same pretrained base across releases",
            "evidence_grade": "user_asserted_not_publicly_disclosed",
            "public_source_finding": "Anthropic publishes separate system cards describing pretraining followed by substantial post-training for these releases; no same-underlying-model statement was found.",
            "model_treatment": "sensitivity and shared-lineage collapse per user instruction; not a factual disclosed parameter identity",
            "source_url": "https://www.anthropic.com/system-cards",
        },
        {
            "lineage": "GPT-5 through GPT-5.5",
            "claim": "same pretrained base across releases",
            "evidence_grade": "user_asserted_not_publicly_disclosed",
            "public_source_finding": "OpenAI calls GPT-5.5 a new model and describes reasoning models as trained through reinforcement learning; it does not disclose a shared pretrained base with GPT-5.",
            "model_treatment": "sensitivity and shared-lineage collapse per user instruction; not a factual disclosed parameter identity",
            "source_url": "https://openai.com/index/gpt-5-5-system-card/",
        },
        {
            "lineage": "GPT-5.5 and GPT-5.5 Pro",
            "claim": "same underlying model; Pro uses parallel test-time compute",
            "evidence_grade": "primary_source_explicit",
            "public_source_finding": "OpenAI explicitly states that GPT-5.5 Pro is the same underlying model as GPT-5.5 with parallel test-time compute.",
            "model_treatment": "one base model; service mode is not a separate parameter count",
            "source_url": "https://openai.com/index/gpt-5-5-system-card/",
        },
        {
            "lineage": "GPT-5.5 to GPT-5.6 Sol/Terra",
            "claim": "new pretraining improvement",
            "evidence_grade": "primary_source_explicit_direction_not_size",
            "public_source_finding": "OpenAI states that Sol and Terra improve substantially over GPT-5.5 on small-scale pretraining optimization; parameter identity and size remain undisclosed.",
            "model_treatment": "do not extend the GPT-5 through 5.5 shared-base assumption to GPT-5.6",
            "source_url": "https://deploymentsafety.openai.com/gpt-5-6",
        },
    ]


def main() -> None:
    epoch = pd.read_csv(EPOCH, low_memory=False)
    unified = pd.read_csv(UNIFIED, low_memory=False)
    candidate_edges, epoch_inventory = build_epoch_edges(epoch, unified)
    maps = measurement_maps(unified)
    edges, measurements = attach_measurements(candidate_edges, maps)
    eci_predictions = build_eci_predictions(edges, measurements)
    aa_predictions = build_aa_predictions(edges)
    predictions = eci_predictions + aa_predictions
    frontier = frontier_sensitivity_rows()
    evidence = evidence_rows()

    aa_audit = json.loads(AA_AUDIT.read_text(encoding="utf-8"))
    eci_summary = summarize_predictions(eci_predictions, 2026071811)
    aa_summary = summarize_predictions(aa_predictions, 2026071813)
    component_summary = component_category_summary(measurements)

    verified_bases = len({row["base_cluster_id"] for row in edges})
    eci_ci = eci_summary["collapsed_vs_baseline"]["ci_90"]
    aa_ci = aa_summary["collapsed_vs_baseline"]["ci_90"]
    gates = {
        "verified_open_lineage_bases_at_least_8": verified_bases
        >= MIN_VERIFIED_BASES_FOR_PROMOTION,
        "eci_signal_bases_at_least_6": eci_summary["bases"]
        >= MIN_SIGNAL_BASES_FOR_PROMOTION,
        "aa_signal_bases_at_least_6": aa_summary["bases"]
        >= MIN_SIGNAL_BASES_FOR_PROMOTION,
        "eci_collapse_ci_wholly_favorable": bool(
            eci_ci[1] is not None and eci_ci[1] < 0
        ),
        "aa_collapse_ci_wholly_favorable": bool(
            aa_ci[1] is not None and aa_ci[1] < 0
        ),
        "proprietary_shared_base_claims_publicly_verified": False,
    }
    promote = all(gates.values())

    result = {
        "metadata": {
            "generated_on": "2026-07-31",
            "objective": "measure parameter-inference distortion from post-training, inference budget, and asserted shared-base lineages",
            "admitted_lineage_evidence": "Epoch structured Base model field, unique exact normalized parent match, both open weights, total parameters within 1%, language-domain child, nonnegative date order, and at least one matched measurement",
            "outer_rule": "calibration rows strictly earlier than the parent; endpoint family/developer excluded",
            "candidate_rule": "replace descendant score with arithmetic parent/child mean at descendant date",
            "duplicate_rule": "highest AA score per exact/mode-stripped checkpoint, matching the user-specified AA duplicate policy",
            "causal_caveat": "Epoch Base model links identify lineage and unchanged parameter count, not a randomized estimate of RL compute; descendants may also include continued pretraining, data changes, or architecture-preserving training.",
        },
        "inventory": {
            **epoch_inventory,
            "candidate_open_language_same_parameter_links": len(candidate_edges),
            "admitted_measured_lineage_edges": len(edges),
            "admitted_measured_lineage_bases": verified_bases,
            "admitted_developers": len({row["child_organization"] for row in edges}),
            "edges_with_finetune_compute": sum(
                row["finetune_compute_flop"] != "" for row in edges
            ),
            "matched_measurements": len(measurements),
            "matched_component_measurements": sum(
                row["metric"] == "ECI component" for row in measurements
            ),
            "eci_prediction_edges": len(eci_predictions),
            "aa_prediction_edges": len(aa_predictions),
            "nocot_lineage_edges": len(
                {row["edge_id"] for row in measurements if row["metric"] == "No-CoT horizon"}
            ),
            "metr_lineage_edges": len(
                {row["edge_id"] for row in measurements if row["metric"] == "METR p50 horizon"}
            ),
        },
        "hard_same_checkpoint_control": {
            "open_weight_reasoning_pairs": aa_audit["same_weight_reasoning_pairs"],
            "interpretation": "Changing reasoning configuration on the same public weights shifts AA by a creator-balanced median of about 5.5 points. This is an inference-compute control, not an RL-compute estimate.",
        },
        "lineage_backtests": {
            "eci": eci_summary,
            "aa": aa_summary,
        },
        "component_posttraining_sensitivity": {
            "categories": component_summary,
            "interpretation": "Knowledge components show smaller median within-lineage uplift than other components, but only a handful of bases contribute and the pretraining-like non-knowledge group is dominated by one Llama lineage.",
            "incremental_live_weight": 0.0,
        },
        "frontier_asserted_same_base_sensitivity": {
            "rows": len(frontier),
            "chains": len({row["chain"] for row in frontier}),
            "evidence_grade": "user_asserted_not_publicly_disclosed",
            "interpretation": "Large counterfactual ratios show that treating successive proprietary capability scores as independent size observations would be untenable if the same-base claims are true. The live model therefore preserves the user-requested lineage collapse but does not treat the identity as publicly disclosed fact.",
            "incremental_live_weight": 0.0,
        },
        "promotion_gates": gates,
        "decision": {
            "promote_posttraining_correction": promote,
            "incremental_live_weight": 0.0,
            "change_headline_forecasts": False,
            "reason": (
                "The audit establishes a real distortion floor, but has only "
                f"{verified_bases} measured open bases, only {len(aa_predictions)} AA "
                "prediction edges, no METR edges, one No-CoT edge, no decisive ECI "
                "collapse result, and no public verification of the proprietary "
                "same-base claims."
            ),
        },
        "source_files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                EPOCH,
                ECI_COMPONENTS,
                ECI_BENCHMARK_ZIP,
                AA_DETAIL,
                UNIFIED,
                ECI_PANEL,
                AA_PANEL,
                AA_AUDIT,
            )
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(EDGES, edges)
    write_csv(MEASUREMENTS, measurements)
    write_csv(PREDICTIONS, predictions)
    write_csv(FRONTIER_SENSITIVITY, frontier)
    write_csv(EVIDENCE, evidence)
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
