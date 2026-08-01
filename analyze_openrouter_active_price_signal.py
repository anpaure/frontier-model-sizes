#!/usr/bin/env python3
"""Test whether current API price predicts active parameters better than totals.

Inference economics are more directly tied to parameters activated per token
than to the full parameter inventory of a sparse MoE.  This audit therefore
joins the frozen OpenRouter price snapshot to the disclosed active-parameter
panel through only two exact identity routes:

* identical Hugging Face repository IDs; or
* the existing manually audited AA-to-Epoch checkpoint crosswalk.

The evaluation uses release-ordered developer-family holdouts, but the price
feature is explicitly a *current* snapshot.  It is therefore not a genuinely
prospective price backtest.  No signal is promoted unless the total-parameter
transport beats the direct-total price model, its developer-cluster interval is
wholly favorable, and at least 20 tests from eight developers qualify.
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

from frontier_target_signals import AA_TARGET_SIGNALS

from k3_primary_evidence import K3_EVIDENCE_PATH, K3_PARAMETER_SOURCE, K3_TOTAL_T


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
COMPATIBILITY_FILE_DATE = "2026-07-18"
SNAPSHOT_DATE = COMPATIBILITY_FILE_DATE

OPENROUTER_CALIBRATION = OUT / f"openrouter_parameter_calibration_{SNAPSHOT_DATE}.csv"
OPENROUTER_MODELS = ROOT / f"sources/openrouter_model_signals_{SNAPSHOT_DATE}.csv"
AA_PANEL = OUT / f"aa_detailed_parameter_panel_{SNAPSHOT_DATE}.csv"
AA_EPOCH_CROSSCHECK = OUT / f"aa_detailed_epoch_crosscheck_{SNAPSHOT_DATE}.csv"
HF_CONFIG_SIGNALS = ROOT / f"sources/huggingface_architecture_config_signals_{SNAPSHOT_DATE}.csv"
HF_CONFIG_SNAPSHOT = ROOT / f"sources/huggingface_architecture_config_snapshot_{SNAPSHOT_DATE}.json.gz"
HF_CONFIG_AUDIT = OUT / f"huggingface_architecture_config_collection_audit_{SNAPSHOT_DATE}.json"

MATCH_AUDIT = OUT / f"openrouter_active_parameter_match_audit_{SNAPSHOT_DATE}.csv"
PREDICTIONS = OUT / f"openrouter_active_price_predictions_{SNAPSHOT_DATE}.csv"
TARGETS = OUT / f"openrouter_active_price_targets_{SNAPSHOT_DATE}.csv"
RESULT = OUT / f"openrouter_active_price_audit_{SNAPSHOT_DATE}.json"

DATE_ORIGIN = date(2023, 1, 1)
MIN_TRAIN_ROWS = 12
MIN_TRAIN_DEVELOPERS = 5
HIGH_SPARSITY_THRESHOLD = 15.0
MIN_PROMOTION_TESTS = 20
MIN_PROMOTION_DEVELOPERS = 8
BOOTSTRAP_SAMPLES = 20_000
SEED = 20_260_718

FEATURES = {
    "date_price": ("date", "price"),
    "score_date": ("score", "date"),
    "score_date_price": ("score", "date", "price"),
}

# The frozen 2026-07-18 HF-config evidence predates the canonical-ID migration
# in the July 31 unified snapshot.  This is the same exact Gemma checkpoint;
# remap the identity key while preserving the original config record verbatim.
HF_CHECKPOINT_ALIASES = {
    "checkpoint:epoch:gemma-4-31b-it": "checkpoint:google:gemma-4-31b-it",
}

TARGET_SPECS = (
    {
        "model": "Claude Fable 5",
        "developer": "anthropic",
        "release_date": "2026-06-09",
        "score": AA_TARGET_SIGNALS["Claude Fable 5"]["score"],
        "openrouter_model_id": "anthropic/claude-fable-5",
    },
    {
        "model": "GPT-5.6 Sol",
        "developer": "openai",
        "release_date": "2026-07-09",
        "score": AA_TARGET_SIGNALS["GPT-5.6 Sol"]["score"],
        "openrouter_model_id": "openai/gpt-5.6-sol",
    },
    {
        "model": "Kimi K3",
        "developer": "kimi",
        "release_date": "2026-07-16",
        "score": AA_TARGET_SIGNALS["Kimi K3"]["score"],
        "openrouter_model_id": "moonshotai/kimi-k3",
    },
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    fields = list(rows[0])
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


def hf_repo(value: str | None) -> str:
    normalized = (value or "").strip().lower().rstrip("/")
    return re.sub(r"^https?://huggingface\.co/", "", normalized)


def years(value: str) -> float:
    return (date.fromisoformat(value) - DATE_ORIGIN).days / 365.25


def metric_summary(errors: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(errors), dtype=float)
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


def design(rows: list[dict[str, Any]], features: tuple[str, ...]) -> np.ndarray:
    matrix: list[list[float]] = []
    for row in rows:
        values = [1.0]
        for feature in features:
            if feature == "score":
                values.append(float(row["score"]))
            elif feature == "date":
                values.append(years(row["release_date"]))
            elif feature == "price":
                values.append(math.log10(float(row["price"])))
            else:
                raise ValueError(f"Unknown feature: {feature}")
        matrix.append(values)
    return np.asarray(matrix, dtype=float)


def fit(rows: list[dict[str, Any]], target: str, features: tuple[str, ...]) -> np.ndarray:
    matrix = design(rows, features)
    values = np.log10(np.asarray([row[target] for row in rows], dtype=float))
    root_weight = np.sqrt(family_weights(rows))
    beta, *_ = np.linalg.lstsq(
        matrix * root_weight[:, None], values * root_weight, rcond=None
    )
    return beta


def predict(beta: np.ndarray, row: dict[str, Any], features: tuple[str, ...]) -> float:
    return float(10 ** (design([row], features) @ beta).item())


def paired_bootstrap(
    rows: list[dict[str, Any]], left_error: str, right_error: str
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["developer"]].append(
            abs(float(row[left_error])) - abs(float(row[right_error]))
        )
    developer_delta = {
        developer: float(np.mean(values)) for developer, values in grouped.items()
    }
    developers = sorted(developer_delta)
    rng = np.random.default_rng(SEED)
    draws = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for index in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(developers, size=len(developers), replace=True)
        draws[index] = np.mean([developer_delta[name] for name in sampled])
    return {
        "metric": f"equal-developer absolute log10 error; {left_error} minus {right_error}",
        "observed_delta": float(np.mean(list(developer_delta.values()))),
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "probability_left_better": float(np.mean(draws < 0)),
        "samples": BOOTSTRAP_SAMPLES,
        "developers": len(developers),
    }


def build_exact_panel() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    calibration = read_csv(OPENROUTER_CALIBRATION)
    models = {row["openrouter_model_id"]: row for row in read_csv(OPENROUTER_MODELS)}
    aa_rows = read_csv(AA_PANEL)
    config_rows = read_csv(HF_CONFIG_SIGNALS)
    aa_by_group = {row["checkpoint_group_id"]: row for row in aa_rows}
    crosscheck = {
        row["epoch_checkpoint_id"]: row for row in read_csv(AA_EPOCH_CROSSCHECK)
    }

    aa_by_hf: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in aa_rows:
        repo = hf_repo(row["model_weights_source_url"])
        if repo:
            aa_by_hf[repo].append(row)

    configs_by_checkpoint: dict[str, list[dict[str, str]]] = defaultdict(list)
    for config in config_rows:
        for checkpoint_id in config["checkpoint_ids"].split("|"):
            if checkpoint_id:
                configs_by_checkpoint[
                    HF_CHECKPOINT_ALIASES.get(checkpoint_id, checkpoint_id)
                ].append(config)

    audits: list[dict[str, Any]] = []
    panel: list[dict[str, Any]] = []
    conflicts: list[str] = []
    ambiguous_hf = 0
    unresolved_active_ambiguities = 0
    for row in calibration:
        hf_candidates: dict[str, dict[str, str]] = {}
        router_hf_repos: set[str] = set()
        for model_id in row["openrouter_model_ids"].split("|"):
            repo = hf_repo(models[model_id]["hugging_face_id"])
            if not repo:
                continue
            router_hf_repos.add(repo)
            for candidate in aa_by_hf.get(repo, []):
                hf_candidates[candidate["checkpoint_group_id"]] = candidate
        cross_group = crosscheck.get(row["canonical_checkpoint_id"], {}).get(
            "aa_detailed_group_id", ""
        )
        if len(hf_candidates) > 1:
            ambiguous_hf += 1
        if len(hf_candidates) == 1:
            hf_group = next(iter(hf_candidates))
        elif cross_group and cross_group in hf_candidates:
            hf_group = cross_group
        else:
            hf_group = ""
            if any(candidate["active_parameters_b"] for candidate in hf_candidates.values()):
                unresolved_active_ambiguities += 1
        if hf_group and cross_group and hf_group != cross_group:
            conflicts.append(row["canonical_checkpoint_id"])
        selected_group = hf_group or cross_group
        selected = aa_by_group.get(selected_group)
        if len(hf_candidates) > 1 and hf_group and cross_group:
            method = "audited_crosswalk_resolves_multiple_exact_hf_configs"
        elif hf_group and cross_group:
            method = "exact_hf_repo_and_audited_crosswalk"
        elif hf_group:
            method = "exact_hf_repo"
        elif cross_group:
            method = "audited_aa_epoch_crosswalk"
        elif len(hf_candidates) > 1:
            method = "ambiguous_hf_repo"
        else:
            method = "no_exact_aa_identity"

        has_active = bool(selected and selected["active_parameters_b"])
        checkpoint_configs = configs_by_checkpoint.get(
            row["canonical_checkpoint_id"], []
        )
        config_classifications = sorted(
            {config["architecture_classification"] for config in checkpoint_configs}
        )
        dense_config_control = bool(
            selected
            and not has_active
            and checkpoint_configs
            and all(
                config["architecture_classification"] == "dense_config"
                for config in checkpoint_configs
            )
        )
        epoch_total = float(row["total_parameters_b"])
        aa_total = float(selected["parameters_b"]) if selected else None
        total_ratio = (
            max(epoch_total / aa_total, aa_total / epoch_total)
            if aa_total is not None
            else None
        )
        status = (
            "active_parameter_match"
            if has_active
            else "dense_config_active_equals_total"
            if dense_config_control
            else "exact_identity_without_active_parameter"
            if selected
            else method
        )
        active_parameter_source = (
            "Artificial Analysis disclosed active parameters"
            if has_active
            else "Primary Hugging Face config is dense; active equals Epoch total"
            if dense_config_control
            else ""
        )
        audits.append(
            {
                "canonical_checkpoint_id": row["canonical_checkpoint_id"],
                "epoch_model": row["canonical_model_name"],
                "developer": row["family"],
                "epoch_release_date": row["release_date"],
                "epoch_total_parameters_b": epoch_total,
                "openrouter_model_ids": row["openrouter_model_ids"],
                "openrouter_hugging_face_repos": "|".join(sorted(router_hf_repos)),
                "aa_checkpoint_group_id": selected_group,
                "aa_model": selected["selected_name"] if selected else "",
                "aa_creator": selected["creator_slug"] if selected else "",
                "aa_release_date": selected["release_date"] if selected else "",
                "aa_total_parameters_b": aa_total,
                "aa_active_parameters_b": (
                    float(selected["active_parameters_b"]) if has_active else None
                ),
                "hf_config_repositories": "|".join(
                    sorted(config["hugging_face_repo"] for config in checkpoint_configs)
                ),
                "hf_config_classifications": "|".join(config_classifications),
                "active_parameter_source": active_parameter_source,
                "max_total_parameter_ratio": total_ratio,
                "match_method": method,
                "status": status,
            }
        )
        if not has_active and not dense_config_control:
            continue
        active_b = (
            float(selected["active_parameters_b"])
            if has_active
            else epoch_total
        )
        if active_b > epoch_total:
            raise ValueError(f"Invalid active/total pair: {row['canonical_checkpoint_id']}")
        panel.append(
            {
                "canonical_checkpoint_id": row["canonical_checkpoint_id"],
                "model": row["canonical_model_name"],
                "developer": selected["creator_slug"],
                "release_date": row["release_date"],
                "score": float(selected["intelligence_index"]),
                "estimated_score": selected["intelligence_index_estimated"].lower()
                == "true",
                "price": float(row["blended_price_usd_per_mtoken"]),
                "active_b": active_b,
                "total_b": epoch_total,
                "total_to_active_ratio": epoch_total / active_b,
                "match_method": (
                    method
                    if has_active
                    else "primary_hf_config_dense_active_equals_total"
                ),
                "active_parameter_source": active_parameter_source,
                "aa_checkpoint_group_id": selected_group,
                "openrouter_model_ids": row["openrouter_model_ids"],
            }
        )

    if conflicts:
        raise ValueError(f"Exact identity routes disagree: {conflicts}")
    if unresolved_active_ambiguities:
        raise ValueError(
            f"Unresolved active-parameter Hugging Face joins: {unresolved_active_ambiguities}"
        )
    if len(calibration) != 93 or len(audits) != 93:
        raise ValueError("OpenRouter active-price audit failed to preserve all 93 calibration rows")
    disclosed_active_matches = sum(
        row["status"] == "active_parameter_match" for row in audits
    )
    dense_config_controls = sum(
        row["status"] == "dense_config_active_equals_total" for row in audits
    )
    if disclosed_active_matches != 45 or dense_config_controls != 18:
        raise ValueError(
            "Expected 45 disclosed active labels and 18 dense-config controls; "
            f"found {disclosed_active_matches} and {dense_config_controls}"
        )
    if len({row["canonical_checkpoint_id"] for row in panel}) != len(panel):
        raise ValueError("Duplicate canonical checkpoint in active-price panel")
    if len({row["aa_checkpoint_group_id"] for row in panel}) != len(panel):
        raise ValueError("One AA checkpoint was assigned to multiple OpenRouter checkpoints")
    max_disagreement = max(
        row["max_total_parameter_ratio"]
        for row in audits
        if row["status"] == "active_parameter_match"
    )
    max_dense_disagreement = max(
        row["max_total_parameter_ratio"]
        for row in audits
        if row["status"] == "dense_config_active_equals_total"
    )
    if max_disagreement > 1.07:
        raise ValueError(f"Unexpected exact-join parameter disagreement: {max_disagreement}")
    if max_dense_disagreement > 1.08:
        raise ValueError(
            "Unexpected dense-control parameter disagreement: "
            f"{max_dense_disagreement}"
        )
    return panel, audits, {
        "calibration_rows_audited": len(audits),
        "active_parameter_matches": len(panel),
        "aa_disclosed_active_parameter_matches": disclosed_active_matches,
        "dense_config_active_equals_total_controls": dense_config_controls,
        "developers": len({row["developer"] for row in panel}),
        "match_method_counts": dict(Counter(row["match_method"] for row in panel)),
        "max_total_parameter_ratio": max_disagreement,
        "max_dense_control_total_parameter_ratio": max_dense_disagreement,
        "identity_conflicts": len(conflicts),
        "ambiguous_hf_repositories": ambiguous_hf,
        "unresolved_active_hf_ambiguities": unresolved_active_ambiguities,
    }


def chronological_backtest(panel: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for test in sorted(panel, key=lambda row: (row["release_date"], row["model"])):
        train = [
            row
            for row in panel
            if row["release_date"] < test["release_date"]
            and row["developer"] != test["developer"]
        ]
        if (
            len(train) < MIN_TRAIN_ROWS
            or len({row["developer"] for row in train}) < MIN_TRAIN_DEVELOPERS
        ):
            continue
        record: dict[str, Any] = {
            "canonical_checkpoint_id": test["canonical_checkpoint_id"],
            "release_date": test["release_date"],
            "model": test["model"],
            "developer": test["developer"],
            "aa_score": test["score"],
            "blended_price_usd_per_mtoken": test["price"],
            "actual_active_b": test["active_b"],
            "actual_total_b": test["total_b"],
            "actual_total_to_active_ratio": test["total_to_active_ratio"],
            "match_method": test["match_method"],
        }
        for feature_name, features in FEATURES.items():
            for target, label in (("active_b", "active"), ("total_b", "total")):
                prediction = predict(fit(train, target, features), test, features)
                record[f"predicted_{label}_{feature_name}_b"] = prediction
                record[f"{label}_{feature_name}_log10_error"] = math.log10(
                    prediction / test[target]
                )

        high_sparsity = [
            row
            for row in train
            if row["total_to_active_ratio"] >= HIGH_SPARSITY_THRESHOLD
        ]
        ratios_by_developer: dict[str, list[float]] = defaultdict(list)
        for row in high_sparsity:
            ratios_by_developer[row["developer"]].append(
                math.log10(row["total_to_active_ratio"])
            )
        reference_ratio = None
        if len(high_sparsity) >= 8 and len(ratios_by_developer) >= 4:
            reference_ratio = float(
                10
                ** np.mean(
                    [np.mean(values) for values in ratios_by_developer.values()]
                )
            )
        record["high_sparsity_reference_ratio"] = reference_ratio
        for feature_name in ("date_price", "score_date_price"):
            converted = (
                record[f"predicted_active_{feature_name}_b"] * reference_ratio
                if reference_ratio is not None
                else None
            )
            record[f"converted_total_{feature_name}_b"] = converted
            record[f"converted_total_{feature_name}_log10_error"] = (
                math.log10(converted / test["total_b"])
                if converted is not None
                else None
            )
        record.update(
            {
                "train_n": len(train),
                "train_developers": len({row["developer"] for row in train}),
                "train_max_date": max(row["release_date"] for row in train),
                "test_developer_excluded": not any(
                    row["developer"] == test["developer"] for row in train
                ),
                "current_price_snapshot_not_historical": True,
            }
        )
        output.append(record)
    if len({row["canonical_checkpoint_id"] for row in output}) != len(output):
        raise ValueError("Duplicate release-ordered active-price fold")
    return output


def target_sensitivity(
    panel: list[dict[str, Any]], model_signals: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {row["openrouter_model_id"]: row for row in model_signals}
    earliest_target_date = min(row["release_date"] for row in TARGET_SPECS)
    excluded_target_families = {"anthropic", "openai", "kimi", "moonshot"}
    training = [
        row
        for row in panel
        if row["release_date"] < earliest_target_date
        and row["developer"] not in excluded_target_families
    ]
    target_rows: list[dict[str, Any]] = []
    for target in TARGET_SPECS:
        model = by_id[target["openrouter_model_id"]]
        price = float(model["blended_price_geomean_median_usd_per_mtoken"])
        target_rows.append(
            {
                **target,
                "price": price,
                "estimated_score": False,
            }
        )
    predictions: dict[str, dict[str, float]] = defaultdict(dict)
    coefficients: dict[str, list[float]] = {}
    for feature_name in ("date_price", "score_date_price"):
        beta = fit(training, "active_b", FEATURES[feature_name])
        coefficients[feature_name] = [float(value) for value in beta]
        for target in target_rows:
            predictions[target["model"]][feature_name] = predict(
                beta, target, FEATURES[feature_name]
            )
    k3_total_t = K3_TOTAL_T
    output: list[dict[str, Any]] = []
    for target in target_rows:
        name = target["model"]
        date_price_active = predictions[name]["date_price"]
        score_date_price_active = predictions[name]["score_date_price"]
        output.append(
            {
                "model": name,
                "release_date": target["release_date"],
                "aa_score": target["score"],
                "blended_price_usd_per_mtoken": target["price"],
                "price_over_training_max": target["price"]
                / max(row["price"] for row in training),
                "predicted_active_date_price_b": date_price_active,
                "predicted_active_score_date_price_b": score_date_price_active,
                "k3_anchored_total_date_price_t": k3_total_t
                * date_price_active
                / predictions["Kimi K3"]["date_price"],
                "k3_anchored_total_score_date_price_t": k3_total_t
                * score_date_price_active
                / predictions["Kimi K3"]["score_date_price"],
                "k3_anchor_source": K3_PARAMETER_SOURCE,
                "status": "disclosed total anchor" if name == "Kimi K3" else "0%-weight extrapolative sensitivity",
            }
        )
    return output, {
        "common_training_cutoff": earliest_target_date,
        "excluded_target_families": sorted(excluded_target_families),
        "training_rows": len(training),
        "training_developers": len({row["developer"] for row in training}),
        "coefficients": coefficients,
        "price_support_min": min(row["price"] for row in training),
        "price_support_max": max(row["price"] for row in training),
    }


def main() -> None:
    openrouter_models = read_csv(OPENROUTER_MODELS)
    source_dates = {row["snapshot_date"] for row in openrouter_models}
    if len(source_dates) != 1:
        raise ValueError(
            f"Expected one OpenRouter observation date, found {source_dates}"
        )
    observation_date = next(iter(source_dates))
    panel, audits, inventory = build_exact_panel()
    predictions = chronological_backtest(panel)
    write_csv(MATCH_AUDIT, audits)
    write_csv(PREDICTIONS, predictions)

    comparison: dict[str, Any] = {}
    for feature_name in FEATURES:
        comparison[feature_name] = {
            "active": metric_summary(
                row[f"active_{feature_name}_log10_error"] for row in predictions
            ),
            "total": metric_summary(
                row[f"total_{feature_name}_log10_error"] for row in predictions
            ),
            "paired_active_vs_total": paired_bootstrap(
                predictions,
                f"active_{feature_name}_log10_error",
                f"total_{feature_name}_log10_error",
            ),
        }
    comparison["price_incremental_to_score_date"] = {
        "active": paired_bootstrap(
            predictions,
            "active_score_date_price_log10_error",
            "active_score_date_log10_error",
        ),
        "total": paired_bootstrap(
            predictions,
            "total_score_date_price_log10_error",
            "total_score_date_log10_error",
        ),
    }

    transport_rows = [
        row
        for row in predictions
        if row["actual_total_to_active_ratio"] >= HIGH_SPARSITY_THRESHOLD
        and row["converted_total_score_date_price_log10_error"] is not None
    ]
    transport = {
        "scope": f"actual total/active ratio >= {HIGH_SPARSITY_THRESHOLD:g}x",
        "candidate": metric_summary(
            row["converted_total_score_date_price_log10_error"]
            for row in transport_rows
        ),
        "direct_total_baseline": metric_summary(
            row["total_score_date_price_log10_error"] for row in transport_rows
        ),
        "date_price_candidate": metric_summary(
            row["converted_total_date_price_log10_error"] for row in transport_rows
        ),
        "date_price_direct_total_baseline": metric_summary(
            row["total_date_price_log10_error"] for row in transport_rows
        ),
        "paired_cluster_bootstrap": paired_bootstrap(
            transport_rows,
            "converted_total_score_date_price_log10_error",
            "total_score_date_price_log10_error",
        ),
    }
    bootstrap = transport["paired_cluster_bootstrap"]
    candidate = transport["candidate"]
    baseline = transport["direct_total_baseline"]
    performance_gate = (
        candidate["median_multiplicative_error"]
        < baseline["median_multiplicative_error"]
        and candidate["mean_absolute_log10_error"]
        < baseline["mean_absolute_log10_error"]
        and candidate["p80_multiplicative_error"]
        < baseline["p80_multiplicative_error"]
        and bootstrap["ci_90"][1] < 0
    )
    coverage_gate = (
        candidate["n"] >= MIN_PROMOTION_TESTS
        and bootstrap["developers"] >= MIN_PROMOTION_DEVELOPERS
    )
    promote = performance_gate and coverage_gate

    targets, target_fit = target_sensitivity(panel, openrouter_models)
    write_csv(TARGETS, targets)
    result = {
        "metadata": {
            "generated_on": observation_date,
            "compatibility_filename_date": COMPATIBILITY_FILE_DATE,
            "question": "Does current API price predict active parameters better than total parameters, and does active-to-total transport improve total-size recovery?",
            "evaluation_split": "release-ordered developer-family holdout",
            "critical_nonprospective_caveat": f"All prices come from the {observation_date} snapshot. Older test rows may have been repriced after release, so this is not a genuinely historical prospective backtest.",
            "features": FEATURES,
            "training_weights": "equal total weight per developer; estimated AA scores receive half weight",
            "minimum_train_rows": MIN_TRAIN_ROWS,
            "minimum_train_developers": MIN_TRAIN_DEVELOPERS,
            "promotion_minimum_tests": MIN_PROMOTION_TESTS,
            "promotion_minimum_developers": MIN_PROMOTION_DEVELOPERS,
        },
        "inventory": {
            **inventory,
            "release_ordered_predictions": len(predictions),
            "prediction_developers": len({row["developer"] for row in predictions}),
            "high_sparsity_transport_predictions": len(transport_rows),
            "high_sparsity_transport_developers": len(
                {row["developer"] for row in transport_rows}
            ),
        },
        "active_vs_total_predictability": comparison,
        "high_sparsity_total_transport": transport,
        "target_sensitivity": targets,
        "target_fit": target_fit,
        "promotion_gates": {
            "performance_gate_passed": performance_gate,
            "coverage_gate_passed": coverage_gate,
            "required_tests": MIN_PROMOTION_TESTS,
            "observed_tests": candidate["n"],
            "required_developers": MIN_PROMOTION_DEVELOPERS,
            "observed_developers": bootstrap["developers"],
        },
        "decision": {
            "promote_active_price_transport": promote,
            "incremental_live_weight": 0.0,
            "replace_existing_price_branch": False,
            "change_headline_forecasts": False,
            "reason": "Active-price transport is a credible mechanism and beats direct total-price prediction on the sparse subset, but only 16 tests from 7 developers qualify versus the 20/8 promotion gate. Target prices are also far outside the common training support and the feature overlaps existing AA/date and price branches. Retain as a zero-weight diagnostic.",
        },
        "limitations": [
            "Current OpenRouter prices are not launch-vintage historical prices.",
            "Hosted-model availability and active-parameter disclosure are non-random.",
            "The high-sparsity transport comparison has only seven developer clusters.",
            "Fable and Sol prices strongly extrapolate beyond the common target-fit training range.",
            "The proposed feature is correlated with the existing AA/date and API-price branches and must not be added as independent evidence.",
        ],
        "source_files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                OPENROUTER_CALIBRATION,
                OPENROUTER_MODELS,
                AA_PANEL,
                AA_EPOCH_CROSSCHECK,
                HF_CONFIG_SIGNALS,
                HF_CONFIG_SNAPSHOT,
                HF_CONFIG_AUDIT,
                K3_EVIDENCE_PATH,
            )
        },
        "outputs": {
            "match_audit": str(MATCH_AUDIT.relative_to(ROOT)),
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
                "transport": transport,
                "targets": targets,
                "decision": result["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
