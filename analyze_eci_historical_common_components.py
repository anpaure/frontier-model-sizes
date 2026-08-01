#!/usr/bin/env python3
"""Four-model rolling-origin backtest on locked Epoch benchmark components."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import tarfile
import zipfile
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATE = "2026-07-31"
OUT = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
ARCHIVE = ROOT / "sources" / "epoch_eci_historical_snapshots_2026-07-18.tar.gz"
ARCHIVE_METADATA = (
    ROOT / "sources" / "epoch_eci_historical_collection_metadata_2026-07-18.json"
)
FOLD_LEDGER = ROOT / "sources" / f"eci_historical_common_component_folds_{DATE}.csv"
PREDICTIONS = OUT / f"eci_historical_common_component_predictions_{DATE}.csv"
TRAINING = OUT / f"eci_historical_common_component_training_panel_{DATE}.csv"
RESULT = OUT / f"eci_historical_common_component_audit_{DATE}.json"
AUDIT = ROOT / "ECI_HISTORICAL_COMMON_COMPONENT_AUDIT.md"

LOCKED_BENCHMARKS = (
    "GPQA diamond",
    "MATH level 5",
    "OTIS Mock AIME 2024-2025",
)
LOCKED_BENCHMARK_STRING = "|".join(LOCKED_BENCHMARKS)
WEIGHT_MODES = ("equal_developer", "equal_checkpoint")
PRIMARY_WEIGHT_MODE = "equal_developer"
ORIGIN = date(2023, 1, 1)

EXPECTED_SNAPSHOT_HASHES = {
    "20250305190317": "b1b89260bd87fc0f84561046d88b035526f2d8f66b849c4613ec75ebe9a565cf",
    "20250403051524": "d74d26b56e0201ebbbd990b67c45f0eafe234d9a9d32a556e36ead1e14c416be",
    "20250510183121": "ad0e0a70b60b6e1ec7b8df60ec52903a029fe58ed21c2c31e385fef53f720f45",
}
EXPECTED_FOLDS = {
    "Gemma 3 27B": {
        "target_model_version": "gemma-3-27b-it",
        "target_release_date": "2025-03-12",
        "target_developer_token": "Google",
        "target_family_token": "Gemma",
        "training_snapshot_timestamp": "20250305190317",
        "target_score_snapshot_timestamp": "20250403051524",
        "expected_total_b": 27.0,
    },
    "Mistral Small 3.1": {
        "target_model_version": "mistral-small-2503",
        "target_release_date": "2025-03-17",
        "target_developer_token": "Mistral",
        "target_family_token": "Mistral",
        "training_snapshot_timestamp": "20250305190317",
        "target_score_snapshot_timestamp": "20250403051524",
        "expected_total_b": 24.0,
    },
    "Llama 4 Scout": {
        "target_model_version": "Llama-4-Scout-17B-16E-Instruct",
        "target_release_date": "2025-04-05",
        "target_developer_token": "Meta",
        "target_family_token": "Llama",
        "training_snapshot_timestamp": "20250403051524",
        "target_score_snapshot_timestamp": "20250510183121",
        "expected_total_b": 109.0,
    },
    "Llama 4 Maverick": {
        "target_model_version": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        "target_release_date": "2025-04-05",
        "target_developer_token": "Meta",
        "target_family_token": "Llama",
        "training_snapshot_timestamp": "20250403051524",
        "target_score_snapshot_timestamp": "20250510183121",
        "expected_total_b": 400.0,
    },
}
PROMOTION_THRESHOLDS = {
    "minimum_targets": 8,
    "minimum_target_developers": 4,
    "minimum_within_2x": 0.75,
    "maximum_median_multiplicative_error": 2.0,
    "require_project_preregistered_targets": True,
    "maximum_primary_to_sensitivity_median_error_ratio": 1.25,
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def date_years(value: str) -> float:
    return (date.fromisoformat(value) - ORIGIN).days / 365.25


def load_folds() -> pd.DataFrame:
    folds = pd.read_csv(FOLD_LEDGER)
    if folds["target_model"].duplicated().any():
        raise ValueError("Common-component fold ledger has duplicate targets")
    if set(folds["target_model"]) != set(EXPECTED_FOLDS):
        raise ValueError("Common-component fold target inventory changed")
    if set(folds["locked_benchmarks"]) != {LOCKED_BENCHMARK_STRING}:
        raise ValueError("Benchmark selection is not locked to GPQA/MATH/AIME")
    for row in folds.itertuples(index=False):
        expected = EXPECTED_FOLDS[row.target_model]
        for field, value in expected.items():
            observed = getattr(row, field)
            if field == "expected_total_b":
                if float(observed) != value:
                    raise ValueError(f"Target truth changed for {row.target_model}")
            elif str(observed) != value:
                raise ValueError(
                    f"Fold declaration changed for {row.target_model}: {field}"
                )
    return folds


def validate_capture_adjacency(folds: pd.DataFrame) -> None:
    """Prove that each target uses the first pinned ZIP after its release."""
    collection = json.loads(ARCHIVE_METADATA.read_text(encoding="utf-8"))
    ordered = sorted(
        row["timestamp"]
        for row in collection["captures"]
        if row["kind"] == "benchmark_zip"
    )
    for fold in folds.itertuples(index=False):
        training = str(fold.training_snapshot_timestamp)
        target = str(fold.target_score_snapshot_timestamp)
        index = ordered.index(training)
        if index + 1 >= len(ordered) or ordered[index + 1] != target:
            raise ValueError(
                f"Target score snapshot is not the next pinned ZIP: {fold.target_model}"
            )
        before = datetime.strptime(training[:8], "%Y%m%d").date()
        after = datetime.strptime(target[:8], "%Y%m%d").date()
        released = date.fromisoformat(fold.target_release_date)
        if not before < released <= after:
            raise ValueError(
                f"Target release does not fall between adjacent ZIPs: {fold.target_model}"
            )


def snapshot_payloads(
    timestamps: set[str],
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, Any]]]:
    collection = json.loads(ARCHIVE_METADATA.read_text(encoding="utf-8"))
    if collection["archive_sha256"] != sha256(ARCHIVE):
        raise ValueError("Historical archive hash differs from collection metadata")
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    inventory: dict[str, dict[str, Any]] = {}
    required = ("model_versions.csv", "benchmarks_runs.csv", "ml_models.csv")
    with tarfile.open(ARCHIVE, mode="r:gz") as archive:
        for timestamp in sorted(timestamps):
            member_name = f"benchmark_zip/benchmark_data_{timestamp}.zip"
            handle = archive.extractfile(member_name)
            if handle is None:
                raise ValueError(f"Historical snapshot is missing: {member_name}")
            payload = handle.read()
            observed_hash = sha256_bytes(payload)
            if observed_hash != EXPECTED_SNAPSHOT_HASHES[timestamp]:
                raise ValueError(f"Historical snapshot hash changed: {timestamp}")
            with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
                if not set(required) <= set(zipped.namelist()):
                    raise ValueError(f"Required tables missing from {timestamp}")
                inner = {name: zipped.read(name) for name in required}
            frames[timestamp] = {
                name: pd.read_csv(io.BytesIO(payload_bytes))
                for name, payload_bytes in inner.items()
            }
            inventory[timestamp] = {
                "archive_member": member_name,
                "sha256": observed_hash,
                "bytes": len(payload),
                "required_member_sha256": {
                    name: sha256_bytes(payload_bytes)
                    for name, payload_bytes in inner.items()
                },
                "required_member_rows": {
                    name: len(frames[timestamp][name]) for name in required
                },
            }
    return frames, inventory


def component_panel(snapshot: dict[str, pd.DataFrame]) -> pd.DataFrame:
    versions = snapshot["model_versions.csv"].rename(columns={"id": "model_version"})
    if versions["model_version"].duplicated().any():
        raise ValueError("Snapshot model-version IDs are not unique")
    runs = snapshot["benchmarks_runs.csv"].copy()
    runs["Best score (across scorers)"] = pd.to_numeric(
        runs["Best score (across scorers)"], errors="coerce"
    )
    selected = runs[runs["task"].isin(LOCKED_BENCHMARKS)].copy()
    if not bool(
        selected["Best score (across scorers)"].dropna().between(0, 1).all()
    ):
        raise ValueError("Locked benchmark score outside [0, 1]")
    joined = selected.merge(
        versions[["model_version", "Model", "Version release date"]],
        left_on="model",
        right_on="model_version",
        how="inner",
        validate="many_to_one",
    )
    pivot = (
        joined.pivot_table(
            index=["model_version", "Model", "Version release date"],
            columns="task",
            values="Best score (across scorers)",
            aggfunc="max",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    models = snapshot["ml_models.csv"][
        [
            "Model",
            "Parameters",
            "Parameters notes",
            "Organization",
            "Publication date",
            "Confidence",
            "Reference",
            "Link",
            "Base model",
        ]
    ].copy()
    scored_models = set(pivot["Model"])
    models = models[models["Model"].isin(scored_models)]
    duplicated = models[models["Model"].duplicated(keep=False)]
    if not duplicated.empty:
        details = duplicated["Model"].unique().tolist()
        raise ValueError(f"Scored model parameter identities are duplicated: {details}")
    panel = pivot.merge(models, on="Model", how="left", validate="many_to_one")
    panel["Parameters"] = pd.to_numeric(panel["Parameters"], errors="coerce")
    return panel


def complete_training_panel(panel: pd.DataFrame, fold: Any) -> pd.DataFrame:
    release = pd.Timestamp(fold.target_release_date)
    frame = panel.copy()
    frame = frame[
        frame[list(LOCKED_BENCHMARKS)].notna().all(axis=1)
        & frame["Parameters"].notna()
        & (frame["Parameters"] > 0)
        & frame["Organization"].notna()
        & frame["Organization"].astype(str).str.strip().ne("")
    ]
    frame["Version release date"] = pd.to_datetime(
        frame["Version release date"], errors="coerce"
    )
    frame = frame[
        frame["Version release date"].notna()
        & (frame["Version release date"] < release)
    ]
    developer_match = frame["Organization"].fillna("").str.contains(
        fold.target_developer_token, case=False, regex=False
    )
    family_match = frame["Model"].fillna("").str.contains(
        fold.target_family_token, case=False, regex=False
    ) | frame["model_version"].fillna("").str.contains(
        fold.target_family_token, case=False, regex=False
    )
    frame = frame[~developer_match & ~family_match]
    frame = (
        frame.sort_values(["Model", "Version release date", "model_version"])
        .groupby("Model", as_index=False, group_keys=False)
        .tail(1)
        .reset_index(drop=True)
    )
    if frame["Model"].duplicated().any():
        raise ValueError(f"Training bases are duplicated for {fold.target_model}")
    if len(frame) < 10 or frame["Organization"].nunique() < 4:
        raise ValueError(f"Training coverage is too small for {fold.target_model}")
    return frame


def target_measurement(panel: pd.DataFrame, fold: Any) -> pd.Series:
    rows = panel[panel["model_version"] == fold.target_model_version]
    if len(rows) != 1:
        raise ValueError(f"Target model-version is not one-to-one: {fold.target_model}")
    row = rows.iloc[0]
    if row["Model"] != fold.target_model:
        raise ValueError(f"Target model identity mismatch: {fold.target_model}")
    if str(row["Version release date"])[:10] != fold.target_release_date:
        raise ValueError(f"Target score release date mismatch: {fold.target_model}")
    if row[list(LOCKED_BENCHMARKS)].isna().any():
        raise ValueError(f"Target lacks a locked component: {fold.target_model}")
    actual_b = float(row["Parameters"]) / 1e9
    if not math.isclose(actual_b, float(fold.expected_total_b), abs_tol=1e-9):
        raise ValueError(f"Snapshot parameter truth mismatch: {fold.target_model}")
    return row


def normalized_weights(frame: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "equal_checkpoint":
        values = np.ones(len(frame))
    elif mode == "equal_developer":
        counts = Counter(frame["Organization"])
        values = np.asarray([1.0 / counts[value] for value in frame["Organization"]])
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    return values / values.mean()


def weighted_prediction(
    x: np.ndarray, y: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> float:
    root = np.sqrt(weights)
    if np.linalg.matrix_rank(x * root[:, None]) != x.shape[1]:
        raise ValueError("Common-component design matrix is rank deficient")
    beta = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]
    return float(target @ beta)


def fold_predictions(
    folds: pd.DataFrame, snapshots: dict[str, dict[str, pd.DataFrame]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    panels = {timestamp: component_panel(snapshot) for timestamp, snapshot in snapshots.items()}
    predictions: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for fold in folds.itertuples(index=False):
        if fold.target_model_version in set(
            panels[str(fold.training_snapshot_timestamp)]["model_version"]
        ):
            raise ValueError(
                f"Target was already present in its prior snapshot: {fold.target_model}"
            )
        train = complete_training_panel(
            panels[str(fold.training_snapshot_timestamp)], fold
        )
        target = target_measurement(
            panels[str(fold.target_score_snapshot_timestamp)], fold
        )
        means = train[list(LOCKED_BENCHMARKS)].mean()
        standard_deviations = train[list(LOCKED_BENCHMARKS)].std(ddof=0)
        if (standard_deviations <= 0).any():
            raise ValueError(f"Zero component variance for {fold.target_model}")
        train_index = (
            (train[list(LOCKED_BENCHMARKS)] - means) / standard_deviations
        ).mean(axis=1)
        target_index = float(
            ((target[list(LOCKED_BENCHMARKS)] - means) / standard_deviations).mean()
        )
        train_dates = np.asarray(
            [
                date_years(value.date().isoformat())
                for value in train["Version release date"]
            ]
        )
        target_date = date_years(fold.target_release_date)
        y = np.log10(train["Parameters"].to_numpy(float) / 1e9)
        design_score = np.column_stack([np.ones(len(train)), train_index])
        design_date = np.column_stack(
            [np.ones(len(train)), train_index, train_dates]
        )
        target_score = np.asarray([1.0, target_index])
        target_dated = np.asarray([1.0, target_index, target_date])
        equal_developer = normalized_weights(train, "equal_developer")
        for index, (_, row) in enumerate(train.iterrows()):
            training_rows.append(
                {
                    "target_model": fold.target_model,
                    "training_snapshot_timestamp": str(
                        fold.training_snapshot_timestamp
                    ),
                    "training_model": row["Model"],
                    "training_model_version": row["model_version"],
                    "training_release_date": row["Version release date"]
                    .date()
                    .isoformat(),
                    "training_organization": row["Organization"],
                    "training_total_b": float(row["Parameters"]) / 1e9,
                    "gpqa_diamond": float(row["GPQA diamond"]),
                    "math_level_5": float(row["MATH level 5"]),
                    "otis_mock_aime": float(row["OTIS Mock AIME 2024-2025"]),
                    "component_index": float(train_index.iloc[index]),
                    "equal_developer_weight": float(equal_developer[index]),
                    "developer_exclusion_token": fold.target_developer_token,
                    "family_exclusion_token": fold.target_family_token,
                }
            )
        for mode in WEIGHT_MODES:
            weights = normalized_weights(train, mode)
            score_log = weighted_prediction(
                design_score, y, target_score, weights
            )
            dated_log = weighted_prediction(
                design_date, y, target_dated, weights
            )
            blended_log = 0.60 * score_log + 0.40 * dated_log
            predicted_b = float(10**blended_log)
            actual_b = float(target["Parameters"]) / 1e9
            error = blended_log - math.log10(actual_b)
            predictions.append(
                {
                    "target_model": fold.target_model,
                    "target_model_version": fold.target_model_version,
                    "target_release_date": fold.target_release_date,
                    "target_organization": target["Organization"],
                    "training_snapshot_timestamp": str(
                        fold.training_snapshot_timestamp
                    ),
                    "target_score_snapshot_timestamp": str(
                        fold.target_score_snapshot_timestamp
                    ),
                    "locked_benchmarks": LOCKED_BENCHMARK_STRING,
                    "weight_mode": mode,
                    "train_rows": len(train),
                    "train_developers": int(train["Organization"].nunique()),
                    "target_gpqa_diamond": float(target["GPQA diamond"]),
                    "target_math_level_5": float(target["MATH level 5"]),
                    "target_otis_mock_aime": float(
                        target["OTIS Mock AIME 2024-2025"]
                    ),
                    "target_component_index": target_index,
                    "actual_b": actual_b,
                    "parameter_confidence": target["Confidence"],
                    "parameter_source": target["Link"],
                    "score_only_predicted_b": float(10**score_log),
                    "score_date_predicted_b": float(10**dated_log),
                    "predicted_b": predicted_b,
                    "log10_error": error,
                    "multiplicative_error": float(10 ** abs(error)),
                    "within_2x": abs(error) <= math.log10(2),
                    "project_preregistered": False,
                }
            )
    return predictions, training_rows


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([float(row["log10_error"]) for row in rows])
    absolute = np.abs(errors)
    return {
        "n": len(rows),
        "target_developers": len({row["target_organization"] for row in rows}),
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "mean_absolute_log10_error": float(absolute.mean()),
        "rmse_log10": float(np.sqrt(np.mean(np.square(errors)))),
        "within_2x": float(np.mean(absolute <= math.log10(2))),
        "maximum_multiplicative_error": float(10 ** absolute.max()),
    }


def promotion_decision(summary: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary = summary[PRIMARY_WEIGHT_MODE]
    sensitivity = summary["equal_checkpoint"]
    ratio = max(
        primary["median_multiplicative_error"],
        sensitivity["median_multiplicative_error"],
    ) / min(
        primary["median_multiplicative_error"],
        sensitivity["median_multiplicative_error"],
    )
    gates = {
        "target_count": {
            "observed": primary["n"],
            "required": PROMOTION_THRESHOLDS["minimum_targets"],
            "passes": primary["n"] >= PROMOTION_THRESHOLDS["minimum_targets"],
        },
        "target_developer_count": {
            "observed": primary["target_developers"],
            "required": PROMOTION_THRESHOLDS["minimum_target_developers"],
            "passes": primary["target_developers"]
            >= PROMOTION_THRESHOLDS["minimum_target_developers"],
        },
        "within_2x": {
            "observed": primary["within_2x"],
            "required": PROMOTION_THRESHOLDS["minimum_within_2x"],
            "passes": primary["within_2x"]
            >= PROMOTION_THRESHOLDS["minimum_within_2x"],
        },
        "median_error": {
            "observed": primary["median_multiplicative_error"],
            "required_maximum": PROMOTION_THRESHOLDS[
                "maximum_median_multiplicative_error"
            ],
            "passes": primary["median_multiplicative_error"]
            <= PROMOTION_THRESHOLDS["maximum_median_multiplicative_error"],
        },
        "project_preregistration": {
            "observed": False,
            "required": PROMOTION_THRESHOLDS[
                "require_project_preregistered_targets"
            ],
            "passes": False,
        },
        "weight_sensitivity": {
            "observed_max_over_min_median_error": ratio,
            "required_maximum": PROMOTION_THRESHOLDS[
                "maximum_primary_to_sensitivity_median_error_ratio"
            ],
            "passes": ratio
            <= PROMOTION_THRESHOLDS[
                "maximum_primary_to_sensitivity_median_error_ratio"
            ],
        },
    }
    promote = all(row["passes"] for row in gates.values())
    return {
        "gates": gates,
        "promote_to_live_model": promote,
        "live_weight": 0.0 if not promote else None,
        "reason": (
            "All fixed coverage, accuracy, preregistration, and sensitivity gates pass."
            if promote
            else "At least one fixed gate fails; retain as a zero-weight historical diagnostic."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown(result: dict[str, Any]) -> str:
    primary = result["summary"][PRIMARY_WEIGHT_MODE]
    rows = [
        row
        for row in result["predictions"]
        if row["weight_mode"] == PRIMARY_WEIGHT_MODE
    ]
    detail = [
        f"- {row['target_model']}: {row['predicted_b']:.1f}B predicted vs "
        f"{row['actual_b']:.1f}B snapshot truth "
        f"({row['multiplicative_error']:.2f}x error)."
        for row in rows
    ]
    failed = [
        name
        for name, gate in result["decision"]["gates"].items()
        if not gate["passes"]
    ]
    lines = [
        "# ECI historical common-component audit",
        "",
        "This rolling-origin panel uses exactly GPQA diamond, MATH level 5, and OTIS Mock AIME 2024-2025. The fold ledger fixes all targets, versions, snapshots, developer exclusions, family exclusions, and benchmark selection before fitting.",
        "",
        "Each target is scored in its first later pinned snapshot. Training parameters and scores come only from the prior snapshot, training releases are strictly earlier, and any row matching the target developer or base-family token is removed. Canonical model names are then collapsed to the latest eligible scored version.",
        "",
        f"The primary equal-developer 60/40 component/date blend has {primary['median_multiplicative_error']:.2f}x median error and {primary['within_2x']:.0%} within 2x.",
        "",
        *detail,
        "",
        "## Promotion decision",
        "",
        f"Live weight remains 0%. Failed fixed gates: {', '.join(failed)}.",
        "",
        "These four retrospective observations are a useful failure-mode check, not independent evidence precise enough to tighten frontier parameter forecasts.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    folds = load_folds()
    validate_capture_adjacency(folds)
    timestamps = {
        str(value)
        for column in (
            "training_snapshot_timestamp",
            "target_score_snapshot_timestamp",
        )
        for value in folds[column]
    }
    snapshots, source_inventory = snapshot_payloads(timestamps)
    predictions, training_rows = fold_predictions(folds, snapshots)
    write_csv(PREDICTIONS, predictions)
    write_csv(TRAINING, training_rows)
    summary = {
        mode: metric_summary(
            [row for row in predictions if row["weight_mode"] == mode]
        )
        for mode in WEIGHT_MODES
    }
    decision = promotion_decision(summary)
    result = {
        "generated_on": DATE,
        "role": "zero-weight historical common-component validation",
        "method": {
            "locked_benchmarks": list(LOCKED_BENCHMARKS),
            "component_index": "equal mean of prior-snapshot z-scores",
            "prediction_law": "60% component-only plus 40% component-and-exact-date direct log10(total parameters) regression",
            "primary_weight_mode": PRIMARY_WEIGHT_MODE,
            "weight_sensitivity": "equal_checkpoint",
            "training_time_rule": "training version release date strictly earlier than target release date",
            "identity_rule": "exclude target developer token and target base-family token, then retain latest complete scored version per canonical Model",
            "parameter_rule": "training and target truths are read from ml_models.csv inside their declared snapshot ZIPs",
            "target_score_rule": "target component scores are read from the declared first later snapshot only",
            "project_preregistered": False,
            "promotion_thresholds": PROMOTION_THRESHOLDS,
        },
        "inventory": {
            "targets": sorted(EXPECTED_FOLDS),
            "target_count": len(EXPECTED_FOLDS),
            "target_developers": int(
                folds["target_developer_token"].nunique()
            ),
            "prediction_rows": len(predictions),
            "training_rows": len(training_rows),
            "distinct_snapshots": len(source_inventory),
        },
        "summary": summary,
        "decision": decision,
        "predictions": predictions,
        "sources": {
            "fold_ledger": str(FOLD_LEDGER.relative_to(ROOT)),
            "fold_ledger_sha256": sha256(FOLD_LEDGER),
            "historical_archive": str(ARCHIVE.relative_to(ROOT)),
            "historical_archive_sha256": sha256(ARCHIVE),
            "historical_archive_metadata": str(
                ARCHIVE_METADATA.relative_to(ROOT)
            ),
            "historical_archive_metadata_sha256": sha256(ARCHIVE_METADATA),
            "snapshot_inventory": source_inventory,
        },
        "outputs": {
            "predictions": str(PREDICTIONS.relative_to(ROOT)),
            "predictions_sha256": sha256(PREDICTIONS),
            "training_panel": str(TRAINING.relative_to(ROOT)),
            "training_panel_sha256": sha256(TRAINING),
        },
        "caveats": [
            "All four checks are retrospective and were not preregistered before parameter truth was public.",
            "The two Llama targets share one developer, release date, and training fold and are not independent.",
            "Target benchmark measurements arrive after release and therefore test archive ordering, not a real-time forecast.",
            "Maverick uses Epoch's scored FP8 model-version as the measurement identity while parameter truth is the canonical 400B base.",
        ],
    }
    RESULT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    AUDIT.write_text(markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "primary_summary": summary[PRIMARY_WEIGHT_MODE],
                "failed_gates": [
                    name
                    for name, gate in decision["gates"].items()
                    if not gate["passes"]
                ],
                "live_weight": decision["live_weight"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
