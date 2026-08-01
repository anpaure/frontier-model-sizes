#!/usr/bin/env python3
"""Test OpenRouter price incrementally against the existing evidence ensemble."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from k3_primary_evidence import K3_EVIDENCE_PATH, K3_PARAMETER_SOURCE, K3_TOTAL_B


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
BASE_BACKTEST = OUT / "frontier_parameter_chronological_backtest_2026-07-17.json"
OPENROUTER_PREDICTIONS = OUT / "openrouter_parameter_backtest_predictions_2026-07-18.csv"
OPENROUTER_FRONTIER = OUT / "openrouter_frontier_operational_estimates_2026-07-18.csv"
RESULT = OUT / "openrouter_incremental_price_backtest_2026-07-18.json"
PREDICTIONS = OUT / "openrouter_incremental_price_backtest_predictions_2026-07-18.csv"


# Explicit crosswalk only. Entries without a strictly chronological OpenRouter
# price prediction remain documented but cannot enter the paired test.
EXPLICIT_OVERLAP_MAP = {
    "deepseekv4flash": "checkpoint:epoch:deepseek-v4-flash",
    "deepseekv4pro": "checkpoint:deepseek:deepseek-v4-pro",
    "gemma227b": "checkpoint:google:gemma-2-27b",
    "glm46": "checkpoint:z-ai-zhipu-ai-tsinghua-university:glm-4-6",
    "glm5": "checkpoint:z-ai-zhipu-ai:glm-5",
    "glm52": "checkpoint:z-ai-zhipu-ai:glm-5-2",
    "gptoss120b": "checkpoint:openai:gpt-oss-120b",
    "kimik2": "checkpoint:epoch:kimi-k2",
    "llama3170b": "checkpoint:meta:llama-3-1-70b",
    "llama318b": "checkpoint:meta:llama-3-1-8b",
    "llama3370b": "checkpoint:meta:llama-3-3-70b",
    "llama4maverick": "checkpoint:meta:llama-4-maverick",
    "llama4scout": "checkpoint:meta:llama-4-scout",
    "mixtral8x22b": "checkpoint:mistral:mixtral-8x22b",
    "nemotron3nano30ba3b": "checkpoint:epoch:nemotron-3-nano-30b-a3b",
    "phi4": "checkpoint:microsoft-research:phi-4",
    "qwen3next80ba3b": "checkpoint:epoch:qwen3-next-80b-a3b",
}

FIXED_WEIGHTS = (0.0, 0.03375, 0.0675, 0.10, 0.20, 0.30, 0.50, 1.0)
BOOTSTRAP_SEED = 20260718
BOOTSTRAP_PROMOTION_PROBABILITY = 0.90


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def external_check_prediction(
    checks: list[dict[str, Any]], anchor: str, method_marker: str
) -> float:
    """Resolve one semantic external check without depending on display copy.

    The backtest producer may clarify a method label (for example by adding
    ``direct``) without changing the underlying check.  Anchor identity plus a
    stable method marker is the contract; ambiguity or absence remains fatal.
    """

    matches = [
        row
        for row in checks
        if row.get("anchor") == anchor and method_marker in row.get("method", "")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {anchor!r} external check containing "
            f"{method_marker!r}; found {len(matches)}"
        )
    return float(matches[0]["predicted_b"])


def blend_log(row: dict[str, Any], price_weight: float) -> float:
    return (1 - price_weight) * row["evidence_log"] + price_weight * row["price_log"]


def metrics(rows: list[dict[str, Any]], price_weight: float) -> dict[str, Any]:
    errors = np.asarray([blend_log(row, price_weight) - row["actual_log"] for row in rows])
    absolute = np.abs(errors)
    return {
        "n": len(rows),
        "price_weight": price_weight,
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "mean_absolute_log10_error": float(np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(errors**2))),
        "p80_multiplicative_error": float(10 ** np.quantile(absolute, 0.80)),
        "within_2x": float(np.mean(absolute <= math.log10(2))),
        "signed_bias_factor": float(10 ** np.mean(errors)),
    }


def paired_family_bootstrap(
    rows: list[dict[str, Any]], price_weight: float, samples: int = 20_000
) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    families = sorted(by_family)

    def delta(chosen: list[str]) -> float:
        selected = [row for family in chosen for row in by_family[family]]
        return float(
            np.mean(
                [
                    abs(blend_log(row, price_weight) - row["actual_log"])
                    - abs(row["evidence_log"] - row["actual_log"])
                    for row in selected
                ]
            )
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.asarray(
        [delta(list(rng.choice(families, size=len(families), replace=True))) for _ in range(samples)]
    )
    return {
        "price_weight": price_weight,
        "paired_models": len(rows),
        "developer_families": len(families),
        "metric": "mean absolute log10 error; blended minus evidence-only",
        "observed_delta": delta(families),
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "bootstrap_probability_blend_better": float(np.mean(draws < 0)),
        "samples": samples,
        "random_seed": BOOTSTRAP_SEED,
    }


def nested_weight_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    grid = np.linspace(0, 1, 21)
    for test in sorted(rows, key=lambda row: (row["release_date"], row["model"])):
        train = [
            row
            for row in rows
            if row["release_date"] < test["release_date"] and row["family"] != test["family"]
        ]
        if len(train) < 5 or len({row["family"] for row in train}) < 4:
            continue
        losses = [
            np.mean([(blend_log(row, float(weight)) - row["actual_log"]) ** 2 for row in train])
            for weight in grid
        ]
        weight = float(grid[int(np.argmin(losses))])
        predicted = blend_log(test, weight)
        output.append(
            {
                "model": test["model"],
                "family": test["family"],
                "release_date": test["release_date"],
                "learned_price_weight": weight,
                "actual_b": 10 ** test["actual_log"],
                "predicted_b": 10**predicted,
                "log10_error": predicted - test["actual_log"],
                "training_rows": len(train),
                "training_families": len({row["family"] for row in train}),
                "training_max_date": max(row["release_date"] for row in train),
                "test_family_excluded": all(row["family"] != test["family"] for row in train),
            }
        )
    return output


def main() -> None:
    base = json.loads(BASE_BACKTEST.read_text(encoding="utf-8"))
    base_by_model = {row["normalized_model"]: row for row in base["ensemble_predictions"]}
    price_predictions = [
        row
        for row in read_csv(OPENROUTER_PREDICTIONS)
        if row["mode"] == "chronological_family" and row["specification"] == "date_price"
    ]
    price_by_checkpoint = {row["canonical_checkpoint_id"]: row for row in price_predictions}

    rows: list[dict[str, Any]] = []
    unavailable = []
    for normalized_model, checkpoint_id in EXPLICIT_OVERLAP_MAP.items():
        evidence = base_by_model.get(normalized_model)
        price = price_by_checkpoint.get(checkpoint_id)
        if not evidence or not price:
            unavailable.append(
                {
                    "normalized_model": normalized_model,
                    "canonical_checkpoint_id": checkpoint_id,
                    "reason": "no strictly chronological prediction from one or both branches",
                }
            )
            continue
        actual_evidence = float(evidence["actual_b"])
        actual_price = float(price["actual_parameters_b"])
        if abs(math.log10(actual_evidence / actual_price)) > 0.03:
            raise ValueError(
                f"Ground-truth mismatch for {normalized_model}: {actual_evidence} vs {actual_price}"
            )
        rows.append(
            {
                "normalized_model": normalized_model,
                "canonical_checkpoint_id": checkpoint_id,
                "model": evidence["model"],
                "family": price["family"],
                "release_date": evidence["release_date"],
                "actual_log": math.log10(actual_evidence),
                "evidence_log": math.log10(float(evidence["predicted_b"])),
                "price_log": math.log10(float(price["predicted_parameters_b"])),
                "evidence_training_rows": evidence["train_n"],
                "price_training_rows": price["training_rows"],
            }
        )
    if len({row["canonical_checkpoint_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate checkpoint in incremental overlap")

    prediction_rows = []
    for row in rows:
        record = {
            "canonical_checkpoint_id": row["canonical_checkpoint_id"],
            "model": row["model"],
            "family": row["family"],
            "release_date": row["release_date"],
            "actual_b": 10 ** row["actual_log"],
            "evidence_predicted_b": 10 ** row["evidence_log"],
            "price_predicted_b": 10 ** row["price_log"],
        }
        for weight in FIXED_WEIGHTS:
            record[f"blend_w{weight:g}_b"] = 10 ** blend_log(row, weight)
        prediction_rows.append(record)
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(prediction_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    fixed_metrics = {f"{weight:g}": metrics(rows, weight) for weight in FIXED_WEIGHTS}
    bootstraps = {
        f"{weight:g}": paired_family_bootstrap(rows, weight)
        for weight in FIXED_WEIGHTS
        if weight > 0
    }
    nested = nested_weight_predictions(rows)
    nested_metrics = None
    if nested:
        errors = [float(row["log10_error"]) for row in nested]
        nested_metrics = {
            "n": len(nested),
            "median_multiplicative_error": float(10 ** np.median(np.abs(errors))),
            "mean_absolute_log10_error": float(np.mean(np.abs(errors))),
            "rmse_log10": float(np.sqrt(np.mean(np.asarray(errors) ** 2))),
            "median_learned_price_weight": float(np.median([row["learned_price_weight"] for row in nested])),
            "status": "small-sample diagnostic only",
        }

    # External anchors are not used to fit or select the weight.
    frontier = {row["openrouter_model_id"]: row for row in read_csv(OPENROUTER_FRONTIER)}
    external = []
    anchor_inputs = {
        "moonshotai/kimi-k3": {
            "actual_b": K3_TOTAL_B,
            "evidence_b": external_check_prediction(
                base["external_checks"], "Kimi K3", "all Kimi held out"
            ),
        },
        "x-ai/grok-4.5": {
            "actual_b": 1500.0,
            "evidence_b": external_check_prediction(
                base["external_checks"],
                "Grok 4.5",
                "geometric ensemble before anchor lock",
            ),
        },
    }
    for model_id, anchor in anchor_inputs.items():
        price_b = float(frontier[model_id]["operational_model_central_b"])
        evidence_log = math.log10(float(anchor["evidence_b"]))
        price_log = math.log10(price_b)
        current_weight = 0.0675
        blended_b = 10 ** ((1 - current_weight) * evidence_log + current_weight * price_log)
        external.append(
            {
                "model_id": model_id,
                "actual_b": anchor["actual_b"],
                "actual_parameter_source": (
                    K3_PARAMETER_SOURCE
                    if model_id == "moonshotai/kimi-k3"
                    else "Grok 4.5 first-party 1.5T disclosure"
                ),
                "evidence_b": anchor["evidence_b"],
                "price_b": price_b,
                "current_evidence_level_price_weight": current_weight,
                "blended_b": blended_b,
                "evidence_error_factor": max(anchor["actual_b"] / anchor["evidence_b"], anchor["evidence_b"] / anchor["actual_b"]),
                "blended_error_factor": max(anchor["actual_b"] / blended_b, blended_b / anchor["actual_b"]),
            }
        )

    current = bootstraps["0.0675"]
    result = {
        "generated_on": "2026-07-18",
        "question": "Does a chronological family-held-out OpenRouter price prediction improve the existing chronological family-held-out AA/ECI/no-CoT/compute ensemble on the same checkpoints?",
        "coverage": {
            "explicit_crosswalk_entries": len(EXPLICIT_OVERLAP_MAP),
            "strictly_chronological_paired_models": len(rows),
            "developer_families": len({row["family"] for row in rows}),
            "unavailable_entries": unavailable,
        },
        "fixed_weight_metrics": fixed_metrics,
        "paired_developer_family_bootstraps": bootstraps,
        "nested_chronological_weight_learning": {
            "predictions": nested,
            "metrics": nested_metrics,
            "rule": "For each test checkpoint, choose price weight on only earlier paired rows and exclude the test developer family.",
        },
        "external_disclosed_checks": external,
        "decision": {
            "current_evidence_level_price_weight": 0.0675,
            "current_final_weight_for_fable_sol": 0.03375,
            "bootstrap_promotion_probability": BOOTSTRAP_PROMOTION_PROBABILITY,
            "current_weight_has_favorable_paired_bootstrap": current["bootstrap_probability_blend_better"] >= BOOTSTRAP_PROMOTION_PROBABILITY and current["ci_90"][1] < 0,
            "change_live_weight": False,
            "reason": (
                "The current small price branch improves the clean overlap, but only "
                f"{len(rows)} models/{len({row['family'] for row in rows})} developers "
                "qualify and current frontier prices exceed calibration support. This "
                "validates the sign and conservative scale, not a larger optimized weight."
            ),
        },
        "limitations": [
            "Only checkpoints with two independently chronological, family-held-out predictions enter the paired test.",
            "The OpenRouter snapshot contains current prices, not historical launch-vintage prices.",
            "The clean overlap is selected by hosted-model and benchmark availability and is not a random sample.",
            "Frontier Anthropic/OpenAI prices are partially outside the open-model calibration range.",
        ],
        "source_manifest": {
            str(BASE_BACKTEST.relative_to(ROOT)): sha256(BASE_BACKTEST),
            str(OPENROUTER_PREDICTIONS.relative_to(ROOT)): sha256(OPENROUTER_PREDICTIONS),
            str(OPENROUTER_FRONTIER.relative_to(ROOT)): sha256(OPENROUTER_FRONTIER),
            str(K3_EVIDENCE_PATH.relative_to(ROOT)): sha256(K3_EVIDENCE_PATH),
        },
        "prediction_csv": str(PREDICTIONS.relative_to(ROOT)),
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(RESULT), "paired_models": len(rows), "families": len({row['family'] for row in rows}), "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
