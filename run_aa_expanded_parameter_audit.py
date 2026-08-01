#!/usr/bin/env python3
"""Audit an exact AA↔Epoch extension before changing the live AA branch.

The live AA calibration contains 50 manually curated recent open-weight
checkpoints.  The canonical megafile independently contains exact
Artificial-Analysis-to-Epoch checkpoint links with parameter provenance.  This
script admits only unique, high-confidence, open-weight checkpoint links,
collapses duplicate AA configurations to the highest score (the user's stated
rule), reconciles overlaps with the live panel, and tests the resulting union.

Every comparison uses the same test rows.  Training checkpoints must be
strictly earlier than the test checkpoint and the entire test developer is
removed.  The main specification predicts log10 total parameters from AA score
and exact release date, matching the current leakage audit's predeclared AA
specification.  The live branch is changed only if frontier-like held-out rows
support the extension, not merely because a broader panel reduces tail errors.
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
from typing import Any, Iterable

import numpy as np

from aa_calibration_overrides import (
    parameter_label_available_before,
    parameter_training_eligibility_date,
)
from aa_parameter_label_availability import (
    LEDGER_PATH as AA_PARAMETER_LABEL_AVAILABILITY_PATH,
    load_parameter_label_availability,
)
from aa_score_availability import (
    LEDGER_PATH as AA_SCORE_AVAILABILITY_PATH,
    RAW_PATH as AA_CHANGELOG_PATH,
    aa_prediction_information_date,
    aa_score_availability_verified,
    aa_score_available_date,
)

from k3_primary_evidence import K3_EVIDENCE_PATH, K3_PARAMETER_SOURCE, K3_TOTAL_B, K3_TOTAL_T
from open_model_parameter_truth import (
    LEDGER_PATH as OPEN_MODEL_PARAMETER_TRUTH_PATH,
    apply_parameter_truth,
)


def canonical_total_parameters_b(model: str, value: float) -> float:
    return float(
        apply_parameter_truth(
            {"model": model, "total_parameters_b": value},
            total_fields=("total_parameters_b",),
            active_fields=(),
        )["total_parameters_b"]
    )


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
REGRESSION = ROOT / "regression_results.json"
UNIFIED = OUT / "unified_model_observations_compute_enriched_2026-07-17.csv"
EPOCH = ROOT / "sources/epoch_all_ai_models_2026-07-31.csv"

RESULT = OUT / "aa_expanded_parameter_audit_2026-07-18.json"
PANEL = OUT / "aa_expanded_parameter_panel_2026-07-18.csv"
PREDICTIONS = OUT / "aa_expanded_parameter_predictions_2026-07-18.csv"
OVERLAPS = OUT / "aa_expanded_parameter_overlap_audit_2026-07-18.csv"

DATE_ORIGIN = date(2023, 1, 1)
MIN_TRAIN_ROWS = 16
MIN_TRAIN_DEVELOPERS = 6
BOOTSTRAP_SAMPLES = 20_000
COEFFICIENT_BOOTSTRAPS = 5_000

K3_SCORE = 57.1123394372091
K3_DATE = "2026-07-16"
LIVE_SCORE_SLOPE = 0.04520595908212863
LIVE_DATE_SLOPE = -0.35092795129752774

CURRENT_FAMILY_DEVELOPER = {
    "deepseek_v3": "deepseek",
    "deepseek_v4": "deepseek",
    "glm": "zai",
    "gpt_oss": "openai",
    "hy3": "tencent",
    "inkling": "thinking_machines",
    "k_exaone": "lg",
    "kimi_k2": "moonshot",
    "longcat": "meituan",
    "mimo_v2": "xiaomi",
    "minimax_m2": "minimax",
    "minimax_m3": "minimax",
    "mistral_large3": "mistral",
    "nemotron3": "nvidia",
    "north": "cohere",
    "olmo3": "allenai",
    "qwen35": "alibaba",
    "qwen36": "alibaba",
    "qwen3_235": "alibaba",
    "qwen3_30": "alibaba",
    "qwen3_coder": "alibaba",
    "seed_oss": "bytedance",
}

MANUAL_OVERLAPS = {
    # The exact AA label omits the disclosed total/active suffix used by the
    # live panel; release date, developer, total parameters, and score agree.
    "Nemotron 3 Ultra 550B A55B": "Nemotron 3 Ultra",
}

FRONTIER_TARGET_NAMES = (
    "Claude Opus 5",
    "Claude Fable 5",
    "GPT-5.6 Sol",
    "GPT-5.5",
    "Claude Opus 4.8",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def years(value: str) -> float:
    return (date.fromisoformat(value) - DATE_ORIGIN).days / 365.25


def normalize_name(value: str) -> str:
    text = value.lower()
    text = re.sub(r"\b(reasoning|non reasoning|instruct|thinking)\b", "", text)
    text = re.sub(r"\((high|max|low)\)", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_developer(organization: str) -> str:
    lower = organization.lower()
    mappings = (
        ("allen institute", "allenai"),
        ("meta ai", "meta"),
        ("microsoft", "microsoft"),
        ("perplexity", "perplexity"),
        ("reka", "reka"),
        ("cohere", "cohere"),
        ("nvidia", "nvidia"),
        ("nous", "nous"),
        ("baidu", "baidu"),
        ("lg ai", "lg"),
        ("openai", "openai"),
        ("google", "google"),
        ("alibaba", "alibaba"),
        ("mistral", "mistral"),
        ("ibm", "ibm"),
        ("ant group", "ant"),
        ("moonshot", "moonshot"),
        ("prime intellect", "prime_intellect"),
        ("xiaomi", "xiaomi"),
        ("deepseek", "deepseek"),
        ("tencent", "tencent"),
        ("z.ai", "zai"),
    )
    for needle, normalized in mappings:
        if needle in lower:
            return normalized
    fallback = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")
    if not fallback:
        raise ValueError("Exact AA↔Epoch row lacks a developer organization")
    return fallback


def load_current_panel() -> list[dict[str, Any]]:
    regression = json.loads(REGRESSION.read_text(encoding="utf-8"))
    output = []
    for row in regression["open_models"]:
        developer = (
            "nex"
            if row["model"] == "Nex-N2-Pro"
            else CURRENT_FAMILY_DEVELOPER[row["family"]]
        )
        panel_row = {
            "panel_id": f"current:{row['model']}",
            "model": row["model"],
            "release_date": row["release_date"],
            "aa_score": float(row["score"]),
            "total_parameters_b": float(row["total_b"]),
            "developer": developer,
            "estimated_score": int(row["estimated"]),
            "panel_source": "current_manual_only",
            "also_in_current_panel": True,
            "overlap_current_model": row["model"],
            "canonical_checkpoint_id": "",
            "matched_epoch_model": "",
            "epoch_accessibility": "",
            "aa_source": row["source"],
            "parameter_source": row["parameter_source"],
            "epoch_parameter_notes": "",
            "epoch_total_parameters_b": "",
            "parameter_reconciliation": "current_manual_panel",
            "parameter_truth_id": row.get("parameter_truth_id", ""),
            "raw_parameter_total_b": row.get("raw_parameter_total_b", ""),
            "parameter_label_available_date": parameter_training_eligibility_date(
                row
            ),
            "aa_slug": row.get("aa_slug", ""),
            "aa_score_available_date": aa_score_available_date(row),
            "aa_score_availability_verified": aa_score_availability_verified(row),
        }
        output.append(panel_row)
    if len(output) != 50 or len({row["panel_id"] for row in output}) != 50:
        raise ValueError("Current AA panel must contain 50 unique rows")
    return output


def load_exact_panel() -> tuple[list[dict[str, Any]], int]:
    unified = [
        row
        for row in read_csv(UNIFIED)
        if row["source"] == "AA" and row["model_level_include"] == "true"
    ]
    epoch_rows = read_csv(EPOCH)
    epoch_counts = Counter(row["Model"] for row in epoch_rows)
    epoch_by_name = {
        row["Model"]: row for row in epoch_rows if epoch_counts[row["Model"]] == 1
    }

    candidates: list[tuple[dict[str, str], dict[str, str]]] = []
    for row in unified:
        if not (
            row["epoch_link_level"] == "checkpoint"
            and row["epoch_match_confidence"] == "high"
            and row["epoch_candidate_count"] == "1"
            and row["epoch_match_status"] == "matched_checkpoint"
            and row["total_parameters_b"]
            and row["aa_intelligence_index"]
        ):
            continue
        epoch = epoch_by_name.get(row["matched_epoch_model"])
        if epoch is None or not epoch["Model accessibility"].startswith("Open weights"):
            continue
        if row["canonical_release_date"] != epoch["Publication date"]:
            raise ValueError(f"AA/Epoch release-date disagreement for {row['source_model_name']}")
        source_b = canonical_total_parameters_b(
            row["source_model_name"], float(row["total_parameters_b"])
        )
        epoch_b = canonical_total_parameters_b(
            row["matched_epoch_model"], float(epoch["Parameters"]) / 1e9
        )
        primary_k3_supersedes_epoch_rounding = (
            row["source_model_name"] == "Kimi K3"
            and math.isclose(
                source_b, K3_TOTAL_B, rel_tol=0, abs_tol=1e-9
            )
            and math.isclose(epoch_b, 2800.0, rel_tol=0, abs_tol=1e-9)
        )
        exact_epoch_parameter_match = math.isclose(
            source_b, epoch_b, rel_tol=0, abs_tol=1e-9
        )
        if not primary_k3_supersedes_epoch_rounding and not exact_epoch_parameter_match:
            raise ValueError(f"AA/Epoch parameter disagreement for {row['source_model_name']}")
        candidates.append((row, epoch))

    highest: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
    for row, epoch in candidates:
        checkpoint = row["canonical_checkpoint_id"]
        previous = highest.get(checkpoint)
        if previous is None or float(row["aa_intelligence_index"]) > float(
            previous[0]["aa_intelligence_index"]
        ):
            highest[checkpoint] = (row, epoch)

    output = []
    for checkpoint, (row, epoch) in highest.items():
        source_b = canonical_total_parameters_b(
            row["source_model_name"], float(row["total_parameters_b"])
        )
        epoch_b = canonical_total_parameters_b(
            row["matched_epoch_model"], float(epoch["Parameters"]) / 1e9
        )
        primary_k3_supersedes_epoch_rounding = (
            checkpoint == "checkpoint:moonshot:kimi-k3"
            and math.isclose(
                source_b, K3_TOTAL_B, rel_tol=0, abs_tol=1e-9
            )
            and math.isclose(epoch_b, 2800.0, rel_tol=0, abs_tol=1e-9)
        )
        organization = epoch["Organization"] or row["source_organization"]
        timing_row = {
            "model": row["source_model_name"],
            "release_date": row["canonical_release_date"],
            "aa_score": float(row["aa_intelligence_index"]),
            "total_parameters_b": source_b,
            "raw_total_parameters_b": float(
                row.get("raw_total_parameters_b") or row["total_parameters_b"]
            ),
            "parameter_source": row.get("source_url", ""),
        }
        output.append(
            {
                "panel_id": checkpoint,
                "model": row["source_model_name"],
                "release_date": row["canonical_release_date"],
                "aa_score": float(row["aa_intelligence_index"]),
                "total_parameters_b": source_b,
                "raw_total_parameters_b": timing_row["raw_total_parameters_b"],
                "developer": normalize_developer(organization),
                "estimated_score": int(row["aa_score_qualifier"] == "asterisk"),
                "panel_source": "exact_epoch_checkpoint",
                "also_in_current_panel": False,
                "overlap_current_model": "",
                "canonical_checkpoint_id": checkpoint,
                "matched_epoch_model": row["matched_epoch_model"],
                "epoch_accessibility": epoch["Model accessibility"],
                "aa_source": row["source_url"],
                "parameter_source": (
                    K3_PARAMETER_SOURCE
                    if row["source_model_name"] == "Kimi K3"
                    else epoch["Link"]
                    or f"sources/epoch_all_ai_models_2026-07-31.csv#Model={row['matched_epoch_model']}"
                ),
                "epoch_parameter_notes": epoch["Parameters notes"],
                "epoch_total_parameters_b": epoch_b,
                "parameter_reconciliation": (
                    "primary_k3_exact_2780b_supersedes_epoch_rounded_2800b"
                    if primary_k3_supersedes_epoch_rounding
                    else "exact_epoch_parameter_match"
                ),
                "parameter_label_available_date": parameter_training_eligibility_date(timing_row),
                "aa_slug": "",
                "aa_score_available_date": aa_score_available_date(timing_row),
                "aa_score_availability_verified": aa_score_availability_verified(timing_row),
            }
        )
    output.sort(key=lambda row: (row["release_date"], row["model"]))
    if len(output) != 63 or len({row["canonical_checkpoint_id"] for row in output}) != 63:
        raise ValueError(
            "Exact AA↔Epoch panel must contain 63 unique checkpoints; "
            f"found {len(output)} rows / "
            f"{len({row['canonical_checkpoint_id'] for row in output})} IDs"
        )
    discarded_configurations = len(candidates) - len(output)
    if discarded_configurations != 9:
        raise ValueError(
            f"Expected nine lower-scoring duplicate configurations; found {discarded_configurations}"
        )
    return output, discarded_configurations


def reconcile_union(
    current: list[dict[str, Any]], exact: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exact_by_normalized: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(exact):
        exact_by_normalized[normalize_name(row["model"])].append(index)

    matches: dict[int, int] = {}
    match_method: dict[int, str] = {}
    for current_index, row in enumerate(current):
        candidates = exact_by_normalized[normalize_name(row["model"])]
        if len(candidates) == 1:
            matches[current_index] = candidates[0]
            match_method[current_index] = "normalized exact model label"
        elif len(candidates) > 1:
            raise ValueError(f"Ambiguous normalized AA overlap for {row['model']}")

    for current_name, exact_name in MANUAL_OVERLAPS.items():
        current_index = next(
            index for index, row in enumerate(current) if row["model"] == current_name
        )
        exact_index = next(
            index for index, row in enumerate(exact) if row["model"] == exact_name
        )
        if current_index in matches and matches[current_index] != exact_index:
            raise ValueError(f"Manual overlap conflicts with normalized match: {current_name}")
        matches[current_index] = exact_index
        match_method[current_index] = "manual exact checkpoint alias"

    if len(matches) != 19 or len(set(matches.values())) != 19:
        raise ValueError(f"Expected 19 one-to-one current/exact overlaps; found {len(matches)}")

    overlap_rows = []
    for current_index, exact_index in sorted(matches.items()):
        old = current[current_index]
        new = exact[exact_index]
        new["also_in_current_panel"] = True
        new["overlap_current_model"] = old["model"]
        overlap_rows.append(
            {
                "current_model": old["model"],
                "exact_model": new["model"],
                "match_method": match_method[current_index],
                "current_release_date": old["release_date"],
                "exact_release_date": new["release_date"],
                "date_delta_days": (
                    date.fromisoformat(new["release_date"])
                    - date.fromisoformat(old["release_date"])
                ).days,
                "current_aa_score": old["aa_score"],
                "exact_highest_aa_score": new["aa_score"],
                "current_total_parameters_b": old["total_parameters_b"],
                "exact_canonical_total_parameters_b": new["total_parameters_b"],
                "epoch_total_parameters_b": new["epoch_total_parameters_b"],
                "parameter_reconciliation": new["parameter_reconciliation"],
                "developer": new["developer"],
                "canonical_checkpoint_id": new["canonical_checkpoint_id"],
                "matched_epoch_model": new["matched_epoch_model"],
                "parameter_source": new["parameter_source"],
            }
        )

    matched_current = set(matches)
    union = [row for index, row in enumerate(current) if index not in matched_current]
    union.extend(exact)
    union.sort(key=lambda row: (row["release_date"], row["model"]))
    if len(union) != 94 or len({row["panel_id"] for row in union}) != 94:
        raise ValueError("Expanded AA union must contain 94 unique panel IDs")
    if len({row["model"] for row in union}) != 94:
        raise ValueError("Expanded AA union contains duplicate model labels")
    return union, overlap_rows


def developer_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = Counter(row["developer"] for row in rows)
    values = np.asarray(
        [
            (0.5 if int(row["estimated_score"]) else 1.0)
            / counts[row["developer"]]
            for row in rows
        ],
        dtype=float,
    )
    return values / values.mean()


def fit_coefficients(rows: list[dict[str, Any]]) -> np.ndarray:
    x = np.asarray(
        [[1.0, row["aa_score"], years(row["release_date"])] for row in rows],
        dtype=float,
    )
    y = np.log10(
        np.asarray([row["total_parameters_b"] for row in rows], dtype=float)
    )
    root = np.sqrt(developer_weights(rows))
    beta, *_ = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)
    return beta


def predict_log10(rows: list[dict[str, Any]], test: dict[str, Any]) -> tuple[float, list[float]]:
    beta = fit_coefficients(rows)
    x = np.asarray([1.0, test["aa_score"], years(test["release_date"])])
    return float(x @ beta), [float(value) for value in beta]


def make_predictions(
    current: list[dict[str, Any]], expanded: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    output = []
    for test in expanded:
        prediction_date = aa_prediction_information_date(test)
        current_train = [
            row
            for row in current
            if parameter_label_available_before(row, prediction_date)
            and row["developer"] != test["developer"]
        ]
        expanded_train = [
            row
            for row in expanded
            if parameter_label_available_before(row, prediction_date)
            and row["developer"] != test["developer"]
        ]
        if (
            min(len(current_train), len(expanded_train)) < MIN_TRAIN_ROWS
            or min(
                len({row["developer"] for row in current_train}),
                len({row["developer"] for row in expanded_train}),
            )
            < MIN_TRAIN_DEVELOPERS
        ):
            continue
        current_prediction, current_beta = predict_log10(current_train, test)
        expanded_prediction, expanded_beta = predict_log10(expanded_train, test)
        actual = math.log10(test["total_parameters_b"])
        frontier_rank = sum(
            row["aa_score"] <= test["aa_score"] for row in current_train
        ) / len(current_train)
        output.append(
            {
                "release_date": test["release_date"],
                "prediction_information_date": prediction_date,
                "model": test["model"],
                "developer": test["developer"],
                "panel_source": test["panel_source"],
                "also_in_current_panel": test["also_in_current_panel"],
                "aa_score": test["aa_score"],
                "actual_parameters_b": test["total_parameters_b"],
                "frontier_score_rank": frontier_rank,
                "current_50_predicted_b": 10**current_prediction,
                "expanded_panel_predicted_b": 10**expanded_prediction,
                "current_50_log10_error": current_prediction - actual,
                "expanded_panel_log10_error": expanded_prediction - actual,
                "current_train_n": len(current_train),
                "current_train_developer_n": len(
                    {row["developer"] for row in current_train}
                ),
                "current_train_max_date": max(
                    parameter_training_eligibility_date(row)
                    for row in current_train
                ),
                "expanded_train_n": len(expanded_train),
                "expanded_train_developer_n": len(
                    {row["developer"] for row in expanded_train}
                ),
                "expanded_train_max_date": max(
                    parameter_training_eligibility_date(row)
                    for row in expanded_train
                ),
                "test_developer_excluded": True,
                "current_coefficients": json.dumps(current_beta, separators=(",", ":")),
                "expanded_coefficients": json.dumps(
                    expanded_beta, separators=(",", ":")
                ),
            }
        )
    return output


def parameter_metrics(errors: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(errors), dtype=float)
    absolute = np.abs(values)
    return {
        "n": int(len(values)),
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "mean_absolute_log10_error": float(np.mean(absolute)),
        "geomean_multiplicative_error": float(10 ** np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(values**2))),
        "p80_multiplicative_error": float(10 ** np.quantile(absolute, 0.80)),
        "within_2x": float(np.mean(absolute <= math.log10(2.0))),
        "signed_bias_factor": float(10 ** np.mean(values)),
    }


def paired_developer_bootstrap(
    rows: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    by_developer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_developer[row["developer"]].append(row)
    effects = np.asarray(
        [
            np.mean(
                [
                    abs(item["expanded_panel_log10_error"])
                    - abs(item["current_50_log10_error"])
                    for item in developer_rows
                ]
            )
            for _, developer_rows in sorted(by_developer.items())
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    draws = np.mean(
        effects[
            rng.integers(0, len(effects), size=(BOOTSTRAP_SAMPLES, len(effects)))
        ],
        axis=1,
    )
    return {
        "metric": "equal-developer mean absolute log10 error; expanded minus current",
        "observed_delta": float(np.mean(effects)),
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "bootstrap_probability_expanded_better": float(np.mean(draws < 0)),
        "samples": BOOTSTRAP_SAMPLES,
        "developer_clusters": int(len(effects)),
    }


def comparison_scope(
    predictions: list[dict[str, Any]], scope: str, seed: int
) -> dict[str, Any]:
    if scope == "all":
        selected = predictions
    elif scope == "current_panel":
        selected = [row for row in predictions if row["also_in_current_panel"]]
    elif scope == "exact_additions":
        selected = [row for row in predictions if not row["also_in_current_panel"]]
    elif scope == "frontier_like":
        selected = [row for row in predictions if row["frontier_score_rank"] >= 0.90]
    else:
        raise ValueError(scope)
    return {
        "scope": scope,
        "n": len(selected),
        "developers": len({row["developer"] for row in selected}),
        "current_50": parameter_metrics(
            row["current_50_log10_error"] for row in selected
        ),
        "expanded_panel": parameter_metrics(
            row["expanded_panel_log10_error"] for row in selected
        ),
        "paired_developer_bootstrap": paired_developer_bootstrap(selected, seed),
    }


def coefficient_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["developer"]].append(row)
    developers = sorted(grouped)
    rng = np.random.default_rng(20260720)
    draws = []
    attempts = 0
    while len(draws) < COEFFICIENT_BOOTSTRAPS and attempts < COEFFICIENT_BOOTSTRAPS * 3:
        attempts += 1
        sample = []
        for index, developer in enumerate(
            rng.choice(developers, size=len(developers), replace=True)
        ):
            for row in grouped[developer]:
                sample.append({**row, "developer": f"{developer}__{index}"})
        beta = fit_coefficients(sample)
        if beta[1] <= 0:
            continue
        draws.append(beta)
    if len(draws) != COEFFICIENT_BOOTSTRAPS:
        raise ValueError("Could not complete AA coefficient bootstrap")
    array = np.asarray(draws)
    labels = ("intercept", "score_slope", "date_slope")
    return {
        "samples": len(draws),
        **{
            label: {
                "p05": float(np.quantile(array[:, index], 0.05)),
                "median": float(np.quantile(array[:, index], 0.50)),
                "p95": float(np.quantile(array[:, index], 0.95)),
            }
            for index, label in enumerate(labels)
        },
    }


def k3_anchored_total_t(score: float, release_date: str, score_slope: float, date_slope: float) -> float:
    return K3_TOTAL_T * 10 ** (
        score_slope * (score - K3_SCORE)
        + date_slope * (years(release_date) - years(K3_DATE))
    )


def frontier_stability(expanded_beta: np.ndarray) -> list[dict[str, Any]]:
    regression = json.loads(REGRESSION.read_text(encoding="utf-8"))
    predictions = {
        row["model"]: row for row in regression["frontier_predictions"]
    }
    if missing := sorted(set(FRONTIER_TARGET_NAMES) - set(predictions)):
        raise ValueError(f"Missing frontier targets in current regression: {missing}")
    output = []
    for model in FRONTIER_TARGET_NAMES:
        target = predictions[model]
        score = float(target["aa_score"])
        release_date = str(target["release_date"])
        current = k3_anchored_total_t(
            score, release_date, LIVE_SCORE_SLOPE, LIVE_DATE_SLOPE
        )
        expanded = k3_anchored_total_t(
            score, release_date, float(expanded_beta[1]), float(expanded_beta[2])
        )
        output.append(
            {
                "model": model,
                "release_date": release_date,
                "aa_score": score,
                "current_live_aa_t": current,
                "expanded_panel_aa_t": expanded,
                "expanded_over_current": expanded / current,
            }
        )
    return output


def pre_anchor_checks(
    current: list[dict[str, Any]], expanded: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    k3 = {"aa_score": K3_SCORE, "release_date": K3_DATE}
    checks = []
    for label, panel in (("current_50", current), ("expanded_panel", expanded)):
        for holdout_moonshot in (False, True):
            train = [
                row
                for row in panel
                if parameter_label_available_before(row, K3_DATE)
                and (not holdout_moonshot or row["developer"] != "moonshot")
            ]
            prediction, beta = predict_log10(train, k3)
            predicted_b = 10**prediction
            checks.append(
                {
                    "panel": label,
                    "moonshot_held_out": holdout_moonshot,
                    "train_n": len(train),
                    "train_developer_n": len({row["developer"] for row in train}),
                    "predicted_k3_b": predicted_b,
                    "multiplicative_error": max(
                        predicted_b / K3_TOTAL_B, K3_TOTAL_B / predicted_b
                    ),
                    "coefficients": [float(value) for value in beta],
                }
            )
    return checks


def main() -> None:
    label_timing_ledger = load_parameter_label_availability()
    label_timing_evidence_paths = tuple(
        ROOT / evidence["path"]
        for record in label_timing_ledger["records"]
        for evidence in record["local_evidence"]
    )
    current = load_current_panel()
    exact, discarded_configurations = load_exact_panel()
    expanded, overlaps = reconcile_union(current, exact)
    predictions = make_predictions(current, expanded)

    scopes = {
        scope: comparison_scope(predictions, scope, 20260718 + index)
        for index, scope in enumerate(
            ("all", "current_panel", "exact_additions", "frontier_like")
        )
    }
    current_beta = fit_coefficients(current)
    expanded_beta = fit_coefficients(expanded)
    stability = frontier_stability(expanded_beta)
    coefficient_uncertainty = coefficient_bootstrap(expanded)

    result = {
        "generated_on": "2026-07-31",
        "question": "Does the exact AA↔Epoch checkpoint extension improve frontier parameter inference enough to replace the live AA calibration?",
        "data_audit": {
            "current_panel_models": len(current),
            "exact_open_epoch_checkpoints": len(exact),
            "lower_scoring_duplicate_configurations_discarded": discarded_configurations,
            "current_exact_overlaps": len(overlaps),
            "expanded_unique_models": len(expanded),
            "expanded_developers": len({row["developer"] for row in expanded}),
            "primary_parameter_overrides_of_epoch_rounded_values": sum(
                row["parameter_reconciliation"]
                == "primary_k3_exact_2780b_supersedes_epoch_rounded_2800b"
                for row in exact
            ),
            "matching_policy": "Unique high-confidence checkpoint link, unique Epoch label, exact Epoch release date, open weights, then highest AA score per checkpoint. Total parameters must exactly match Epoch except Kimi K3, where the official technical report's exact 2.780T value supersedes Epoch's rounded 2.800T entry.",
            "overlap_policy": "Normalized exact AA model labels plus one manually verified Nemotron alias; the exact Epoch-backed row supersedes the current duplicate.",
            "parameter_label_timing_records": len(label_timing_ledger["records"]),
        },
        "backtest": {
            "target": "log10 total parameters in billions",
            "specification": "log10(total parameters) ~ AA score + exact release date",
            "outer_split": "strictly earlier parameter-training eligibility date; entire test developer removed",
            "training_weights": "equal total weight per developer; AA scores marked with an asterisk receive 0.5 quality weight",
            "minimum_training_rows": MIN_TRAIN_ROWS,
            "minimum_training_developers": MIN_TRAIN_DEVELOPERS,
            "eligible_predictions": len(predictions),
            "scopes": scopes,
        },
        "full_fit": {
            "current_50_developer_balanced_coefficients": {
                "intercept": float(current_beta[0]),
                "score_slope": float(current_beta[1]),
                "date_slope": float(current_beta[2]),
            },
            "expanded_panel_developer_balanced_coefficients": {
                "intercept": float(expanded_beta[0]),
                "score_slope": float(expanded_beta[1]),
                "date_slope": float(expanded_beta[2]),
            },
            "current_live_k3_anchored_coefficients": {
                "score_slope": LIVE_SCORE_SLOPE,
                "date_slope": LIVE_DATE_SLOPE,
            },
            "expanded_coefficient_developer_bootstrap": coefficient_uncertainty,
        },
        "pre_anchor_k3_checks": pre_anchor_checks(current, expanded),
        "frontier_aa_stability": stability,
        "external_grok_check": {
            "actual_t": 1.5,
            "current_live_aa_t": next(
                row["current_live_aa_t"] for row in stability if row["model"] == "Grok 4.5"
            ),
            "expanded_panel_aa_t": next(
                row["expanded_panel_aa_t"] for row in stability if row["model"] == "Grok 4.5"
            ),
        },
        "decision": {
            "change_live_aa_branch": False,
            "incremental_expanded_aa_weight": 0.0,
            "reason": "The extension robustly reduces equal-developer mean error on the broad and current-panel tests, but it does not improve the frontier-like subset: that interval crosses zero and the score/date coefficients move frontier AA centers materially. Retain the current K3-anchored AA branch until new high-score disclosures prospectively distinguish the slopes.",
        },
        "k3_anchor": {
            "total_parameters_b": K3_TOTAL_B,
            "source": K3_PARAMETER_SOURCE,
        },
        "limitations": [
            "AA scores come from the current leaderboard snapshot, not historical score vintages.",
            "AA does not expose a machine-readable reasoning-budget field for every row; the main comparison therefore uses the predeclared score+date specification without a subjective reasoning label.",
            "Developer holdout is conservative but cannot remove distillation or algorithm diffusion across developers.",
            "The extension adds many small and mid-scale models; frontier-like held-out coverage remains only 15 checkpoints across seven developers.",
        ],
        "files": {
            "expanded_panel": str(PANEL.relative_to(ROOT)),
            "predictions": str(PREDICTIONS.relative_to(ROOT)),
            "overlap_audit": str(OVERLAPS.relative_to(ROOT)),
        },
        "source_manifest": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                REGRESSION,
                UNIFIED,
                EPOCH,
                K3_EVIDENCE_PATH,
                AA_PARAMETER_LABEL_AVAILABILITY_PATH,
                AA_SCORE_AVAILABILITY_PATH,
                AA_CHANGELOG_PATH,
                OPEN_MODEL_PARAMETER_TRUTH_PATH,
                *label_timing_evidence_paths,
            )
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(PANEL, expanded)
    write_csv(PREDICTIONS, predictions)
    write_csv(OVERLAPS, overlaps)
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(RESULT),
                "current_models": len(current),
                "exact_checkpoints": len(exact),
                "expanded_models": len(expanded),
                "eligible_predictions": len(predictions),
                "frontier_like_predictions": scopes["frontier_like"]["n"],
                "change_live_aa_branch": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
