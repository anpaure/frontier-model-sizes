#!/usr/bin/env python3
"""Retrospective aggregate ECI validation beyond the frozen tournament."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import analyze_eci_fit_tournament as tournament
import collect_eci_validation_extension as collector
import fit_eci_validation_extension as extension_fit


ROOT = Path(__file__).resolve().parent
DATE = "2026-07-31"
OUT = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
TARGET_LEDGER = ROOT / "sources" / f"eci_historical_validation_targets_{DATE}.csv"
EXTENSION_SCORES = ROOT / "sources" / f"epoch_eci_validation_extension_scores_{DATE}.csv"
PREDICTIONS = OUT / f"eci_historical_validation_extension_predictions_{DATE}.csv"
RESULT = OUT / f"eci_historical_validation_extension_{DATE}.json"
AUDIT = ROOT / "ECI_HISTORICAL_VALIDATION_EXTENSION.md"

EXPECTED_TARGETS = {
    "Kimi K2.5",
    "Kimi K2.7 Code",
    "Grok 4.5",
    "GLM-5.2",
    "Kimi K3",
}
INTERVAL_TARGETS = EXPECTED_TARGETS - {"Kimi K3"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_targets() -> pd.DataFrame:
    targets = pd.read_csv(TARGET_LEDGER, keep_default_na=False)
    if set(targets["model"]) != EXPECTED_TARGETS or targets["model"].duplicated().any():
        raise ValueError("Historical validation target inventory changed")
    targets["project_prospective"] = (
        targets["project_prospective"].astype(str).str.lower() == "true"
    )
    if targets["project_prospective"].any():
        raise ValueError("Retrospective targets cannot be project-prospective")
    interval = set(
        targets.loc[
            targets["validation_class"].str.startswith("historical_interval"), "model"
        ]
    )
    if interval != INTERVAL_TARGETS:
        raise ValueError("Historical interval classification changed")
    k3 = targets.loc[targets["model"] == "Kimi K3"].iloc[0]
    if k3["validation_class"] != "score_vintage_only_not_project_prospective":
        raise ValueError("Kimi K3 must remain score-vintage-only")
    glm = targets.loc[targets["model"] == "GLM-5.2"].iloc[0]
    if not (
        parse_timestamp(glm["prior_capture_timestamp_utc"])
        < parse_timestamp(glm["release_timestamp_utc"])
        < parse_timestamp(glm["first_score_capture_timestamp_utc"])
    ):
        raise ValueError("GLM-5.2 exact timestamps do not establish interval ordering")
    return targets


def compact_timestamp(value: str) -> str:
    return (
        value.replace("-", "")
        .replace(":", "")
        .replace("T", "")
        .replace("Z", "")
    )


def score_vintage(
    historical: pd.DataFrame, extension: pd.DataFrame, target: Any
) -> pd.Series:
    first = compact_timestamp(target.first_score_capture_timestamp_utc)
    if first == collector.CAPTURE_TIMESTAMP:
        frame = extension
    else:
        frame = historical[
            historical["snapshot_timestamp"].astype("int64") == int(first)
        ]
    if frame.empty or frame["Model"].duplicated().any():
        raise ValueError(f"Missing or duplicate score vintage for {target.model}")
    return frame.set_index("Model")["eci"]


def target_frame(target: Any, panel: pd.DataFrame, score: float) -> pd.DataFrame:
    matched = panel.loc[panel["model"] == target.model]
    if len(matched) > 1:
        raise ValueError(f"Duplicate parameter-map target: {target.model}")
    record = matched.iloc[0].to_dict() if len(matched) else {}
    record.update(
        {
            "model": target.model,
            "release_date": target.release_date,
            "total_b": float(target.total_b),
            "family": target.family,
            "reasoning": int(target.reasoning),
            "moe": int(target.moe),
            "coder": int(target.coder),
            "multimodal": int(target.multimodal),
            "score": score,
        }
    )
    return pd.DataFrame([record])


def build_predictions() -> list[dict[str, Any]]:
    collector.verify_existing()
    extension_fit.verify_existing()
    targets = load_targets()
    panel, historical = tournament.load_panel()
    extension = pd.read_csv(EXTENSION_SCORES)
    rows: list[dict[str, Any]] = []
    for target in targets.itertuples(index=False):
        scores = score_vintage(historical, extension, target)
        if target.model not in scores.index:
            raise ValueError(f"Target absent from its first score vintage: {target.model}")
        train = panel[
            panel["model"].isin(scores.index)
            & (panel["release_date"] < target.release_date)
            & (panel["family"] != target.family)
        ].copy()
        train["score"] = train["model"].map(scores)
        if (
            len(train) < tournament.MIN_TRAIN_ROWS
            or train["family"].nunique() < tournament.MIN_TRAIN_FAMILIES
        ):
            raise ValueError(f"Insufficient outer-fold training data for {target.model}")
        test = target_frame(target, panel, float(scores[target.model]))
        actual_log = math.log10(float(target.total_b))
        for mode in tournament.WEIGHT_MODES:
            for candidate in tournament.CANDIDATES:
                predicted_log = float(
                    tournament.predict(train, test, candidate, mode)[0]
                )
                rows.append(
                    {
                        "model": target.model,
                        "family": target.family,
                        "release_date": target.release_date,
                        "prior_capture_timestamp_utc": target.prior_capture_timestamp_utc,
                        "first_score_capture_timestamp_utc": target.first_score_capture_timestamp_utc,
                        "validation_class": target.validation_class,
                        "historical_interval_prospective": target.model
                        in INTERVAL_TARGETS,
                        "score_vintage_holdout": target.model == "Kimi K3",
                        "project_prospective": bool(target.project_prospective),
                        "timestamp_rescued": target.model == "GLM-5.2",
                        "candidate": candidate,
                        "weight_mode": mode,
                        "train_rows": len(train),
                        "train_families": int(train["family"].nunique()),
                        "target_eci": float(scores[target.model]),
                        "actual_b": float(target.total_b),
                        "predicted_b": float(10**predicted_log),
                        "log10_error": predicted_log - actual_log,
                    }
                )
    expected = (
        len(EXPECTED_TARGETS)
        * len(tournament.CANDIDATES)
        * len(tournament.WEIGHT_MODES)
    )
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} prediction rows; found {len(rows)}")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in tournament.WEIGHT_MODES:
        result[mode] = {}
        for candidate in tournament.CANDIDATES:
            selected = [
                row
                for row in rows
                if row["weight_mode"] == mode and row["candidate"] == candidate
            ]
            result[mode][candidate] = {
                "four_target_historical_interval": tournament.metric_summary(
                    [
                        row
                        for row in selected
                        if row["historical_interval_prospective"]
                    ]
                ),
                "k3_score_vintage_only": tournament.metric_summary(
                    [row for row in selected if row["score_vintage_holdout"]]
                ),
            }
    return result


def write_predictions(rows: list[dict[str, Any]]) -> None:
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown(result: dict[str, Any]) -> str:
    primary = result["summary"]["live_inverse_eci_ci"][tournament.BASELINE]
    interval = primary["four_target_historical_interval"]
    baseline = result["baseline_predictions"]
    detail = [
        f"- {row['model']}: {row['predicted_b']/1000:.3f}T predicted vs "
        f"{row['actual_b']/1000:.3f}T disclosed "
        f"({10 ** abs(row['log10_error']):.2f}x error)."
        for row in baseline
        if row["historical_interval_prospective"]
    ]
    k3 = next(row for row in baseline if row["model"] == "Kimi K3")
    lines = [
        "# ECI historical validation extension",
        "",
        "This is a validation-only extension. It cannot tune candidates, select weights, or change the live forecast.",
        "",
        "## Four historical interval targets",
        "",
        "The expanded retrospective interval set spans Moonshot, xAI, and Z.ai. GLM-5.2 is admitted by exact timestamps: its public Hugging Face repository appeared at 07:20:33Z, after the prior Epoch capture at 02:44:13Z on the same day.",
        "",
        f"For the frozen live 60/40 form under inverse-ECI-CI weighting, median multiplicative error is {interval['median_multiplicative_error']:.2f}x and {interval['within_2x']:.0%} fall within 2x.",
        "",
        *detail,
        "",
        "## Kimi K3",
        "",
        f"K3 is a score-vintage holdout only: {k3['predicted_b']/1000:.3f}T predicted vs 2.780T disclosed. It is not project-prospective because the project already used its disclosed size before incorporating the July 22 score vintage.",
        "",
        "## Decision",
        "",
        "No live-weight or functional-form change is permitted from this small retrospective sample. The four-model GPQA/MATH/AIME check is implemented as a separate zero-weight audit and is never mixed into this aggregate result.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    rows = build_predictions()
    write_predictions(rows)
    baseline = [
        row
        for row in rows
        if row["weight_mode"] == "live_inverse_eci_ci"
        and row["candidate"] == tournament.BASELINE
    ]
    result = {
        "generated_on": DATE,
        "role": "external validation only; zero live-model weight",
        "method": {
            "outer_training": "same score vintage, earlier releases only, entire target family excluded",
            "candidate_and_weight_laws": "identical to analyze_eci_fit_tournament.py",
            "selection_or_tuning_on_extension": False,
            "live_weight_change_allowed": False,
        },
        "inventory": {
            "historical_interval_targets": sorted(INTERVAL_TARGETS),
            "score_vintage_only_targets": ["Kimi K3"],
            "project_prospective_targets": [],
            "prediction_rows": len(rows),
        },
        "summary": summarize(rows),
        "baseline_predictions": baseline,
        "decision": {
            "change_live_weights": False,
            "change_live_functional_form": False,
            "reason": "Small retrospective extension; K3 is not project-prospective and no selection occurs here.",
        },
        "sources": {
            "target_ledger": str(TARGET_LEDGER.relative_to(ROOT)),
            "target_ledger_sha256": sha256(TARGET_LEDGER),
            "july_22_capture": str(collector.SOURCE.relative_to(ROOT)),
            "july_22_capture_sha256": sha256(collector.SOURCE),
            "july_22_collection_metadata": str(collector.METADATA.relative_to(ROOT)),
            "july_22_collection_metadata_sha256": sha256(collector.METADATA),
            "july_22_scores": str(EXTENSION_SCORES.relative_to(ROOT)),
            "july_22_scores_sha256": sha256(EXTENSION_SCORES),
            "july_22_fit_metadata": str(extension_fit.METADATA.relative_to(ROOT)),
            "july_22_fit_metadata_sha256": sha256(extension_fit.METADATA),
            "frozen_historical_scores": str(
                tournament.HISTORICAL_SCORES.relative_to(ROOT)
            ),
            "frozen_historical_scores_sha256": sha256(
                tournament.HISTORICAL_SCORES
            ),
        },
        "outputs": {
            "predictions": str(PREDICTIONS.relative_to(ROOT)),
            "predictions_sha256": sha256(PREDICTIONS),
        },
        "linked_component_panel": {
            "component_panel": "Deterministic four-model common GPQA/MATH/AIME rolling-origin panel: Gemma 3 27B, Mistral Small 3.1, Llama 4 Scout, and Llama 4 Maverick.",
            "status": "implemented separately by analyze_eci_historical_common_components.py",
            "live_weight": 0,
        },
        "caveats": [
            "All targets are retrospective, not preregistered project forecasts.",
            "Grok 4.5 uses the user-accepted external 1.5T disclosure and is not added to training.",
            "GLM-5.2 relies on exact timestamp ordering because release and prior capture share a date.",
            "Kimi K3 validates a later score vintage, not an unknown parameter count.",
        ],
    }
    RESULT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    AUDIT.write_text(markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "change_live_weights": False,
                "baseline_predictions_b": {
                    row["model"]: round(row["predicted_b"], 3) for row in baseline
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
