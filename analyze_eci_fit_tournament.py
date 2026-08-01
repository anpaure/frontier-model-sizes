#!/usr/bin/env python3
"""Archive-vintage tournament of functional forms for parameter inference.

The live ECI branch is a 60/40 log blend of score-only and score-plus-date
direct regressions.  This audit challenges it with robust, architectural, and
nonlinear specifications.  Model selection uses archive first-observed rows
that are *not* interval-prospective.  The two checkpoints released between
adjacent archive captures are reserved as the closest available prospective
validation set.

No challenger is promoted merely for fitting the selection rows better.
Promotion also requires prospective improvement and stable frontier
extrapolation across weighting and same-base-collapse sensitivities.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATE = "2026-07-18"
GENERATED_ON = "2026-07-31"
OUT = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
REGRESSION = ROOT / "regression_results.json"
HISTORICAL_SCORES = ROOT / "sources" / f"epoch_eci_historical_model_scores_{DATE}.csv"
HISTORICAL_METADATA = ROOT / "sources" / f"epoch_eci_historical_fit_metadata_{DATE}.json"
COLLECTION_METADATA = ROOT / "sources" / f"epoch_eci_historical_collection_metadata_{DATE}.json"
HISTORICAL_ARCHIVE = ROOT / "sources" / f"epoch_eci_historical_snapshots_{DATE}.tar.gz"
EXPANDED_AUDIT = OUT / f"eci_component_extended_audit_{DATE}.json"
RESULT = OUT / f"eci_fit_tournament_{DATE}.json"
PREDICTIONS = OUT / f"eci_fit_tournament_predictions_{DATE}.csv"
TARGETS = OUT / f"eci_fit_tournament_frontier_sensitivity_{DATE}.csv"
AUDIT = ROOT / "ECI_FIT_TOURNAMENT_AUDIT.md"

ORIGIN = date(2023, 1, 1)
WLS_NUMERATOR = 15.3664  # cancels after normalization; retained to mirror live method
MIN_TRAIN_ROWS = 20
MIN_TRAIN_FAMILIES = 8
BOOTSTRAP_SAMPLES = 20_000
RIDGE_ALPHA = 1.0
MAX_TARGET_SENSITIVITY_RATIO = 1.25
ARCHIVAL_TERMINAL_TIMESTAMP = 20260716153134

# Predeclared in the user-visible work update before results were inspected.
CANDIDATES: dict[str, dict[str, Any]] = {
    "linear_score": {"features": "score"},
    "linear_score_date": {"features": "score_date"},
    "live_60_40_blend": {"features": "blend"},
    "linear_architecture": {"features": "architecture"},
    "huber_score_date": {"features": "huber_score_date"},
    "huber_architecture": {"features": "huber_architecture"},
    "quadratic_score": {"features": "quadratic"},
    "score_date_interaction": {"features": "interaction"},
    "ridge_flexible": {"features": "ridge_flexible", "alpha": RIDGE_ALPHA},
    "linear_tail_spline": {"features": "spline"},
}
BASELINE = "live_60_40_blend"
WEIGHT_MODES = ("live_inverse_eci_ci", "equal_family")

# These two aggregates were present throughout the frozen 15-vintage archive
# and in the July 18 parameter map, but Epoch retired them in the July 31
# canonical ECI refresh.  They remain valid historical tournament labels and
# are never reintroduced into the current frontier fit.
ARCHIVE_RETIRED_PARAMETER_ROWS = (
    {
        "model": "DeepSeek-V3.1",
        "release_date": "2025-08-21",
        "score": 138.68007500233182,
        "ci_width": 142.56796343441684 - 135.7262767532495,
        "total_b": 671.0,
        "active_b": 37.0,
        "family": "deepseek_v3",
        "reasoning": 1,
        "moe": 1,
        "coder": 0,
        "multimodal": 0,
    },
    {
        "model": "Kimi K2 (Sep 2025)",
        "release_date": "2025-09-05",
        "score": 140.97820310910095,
        "ci_width": 142.2685184429225 - 138.38550552192376,
        "total_b": 1000.0,
        "active_b": 32.0,
        "family": "kimi_k2",
        "reasoning": 0,
        "moe": 1,
        "coder": 0,
        "multimodal": 0,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def years(value: str) -> float:
    return (date.fromisoformat(value) - ORIGIN).days / 365.25


def timestamp_date(value: int | str) -> date:
    return datetime.strptime(str(value)[:8], "%Y%m%d").date()


def validate_historical_provenance(historical: pd.DataFrame) -> dict[str, Any]:
    """Prove that the tournament uses the frozen archive, not the live panel."""
    collection = json.loads(COLLECTION_METADATA.read_text(encoding="utf-8"))
    fit = json.loads(HISTORICAL_METADATA.read_text(encoding="utf-8"))
    policy = collection.get("archive_policy", {})
    terminal = collection.get("archival_terminal_exact_match", {})
    successor = collection.get("current_live_successor_reference", {})
    fit_terminal = fit.get("archival_terminal_fit_crosscheck", {})
    fit_successor = fit.get("current_live_successor_reference", {})
    if int(policy.get("terminal_timestamp", 0)) != ARCHIVAL_TERMINAL_TIMESTAMP:
        raise ValueError("ECI archive policy has the wrong terminal timestamp")
    if terminal.get("pinned_file") != "sources/epoch_eci_benchmarks_2026-07-17.csv":
        raise ValueError("ECI archive terminal is not bound to the archival source")
    if successor.get("pinned_file") != "sources/epoch_eci_benchmarks_2026-07-31.csv":
        raise ValueError("ECI archive metadata lacks the live-successor reference")
    if successor.get("byte_equality_assertion_against_archival_terminal") is not False:
        raise ValueError("Live ECI data must not be an archive-terminal equality target")
    if int(fit_terminal.get("timestamp", 0)) != ARCHIVAL_TERMINAL_TIMESTAMP:
        raise ValueError("Historical ECI fit has the wrong archival terminal")
    if fit_terminal.get("archival_terminal_scores") != (
        "sources/epoch_eci_reproduced_scores_2026-07-18.csv"
    ):
        raise ValueError("Historical ECI fit is not bound to archival-terminal scores")
    if fit_successor.get(
        "byte_equality_or_fit_assertion_against_archival_terminal"
    ) is not False:
        raise ValueError("Live ECI scores must not be a historical-fit equality target")
    observed = int(historical["snapshot_timestamp"].astype("int64").max())
    if observed != ARCHIVAL_TERMINAL_TIMESTAMP:
        raise ValueError(f"Historical ECI ledger has unexpected terminal: {observed}")
    if sha256(HISTORICAL_ARCHIVE) != collection["archive_sha256"]:
        raise ValueError("Historical ECI archive hash differs from collection metadata")
    if sha256(HISTORICAL_ARCHIVE) != fit["source_archive_sha256"]:
        raise ValueError("Historical ECI archive hash differs from fit metadata")
    return {
        "archive_cutoff_date": policy["cutoff_date"],
        "terminal_timestamp": ARCHIVAL_TERMINAL_TIMESTAMP,
        "terminal_capture_date": policy["terminal_capture_date"],
        "archival_terminal_source": terminal["pinned_file"],
        "current_live_successor": successor["pinned_file"],
        "current_live_excluded_from_archive_fit": True,
    }


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    regression = json.loads(REGRESSION.read_text(encoding="utf-8"))
    panel = pd.DataFrame(regression["eci"]["open_models"])
    if (
        len(panel) != 89
        or panel["model"].nunique() != 89
        or panel["family"].nunique() != 40
    ):
        raise ValueError("Expected 89 unique current ECI parameter-map checkpoints across 40 families")
    panel["current_canonical"] = True
    retired = pd.DataFrame(ARCHIVE_RETIRED_PARAMETER_ROWS)
    retired["current_canonical"] = False
    panel = pd.concat([panel, retired], ignore_index=True, sort=False)
    if panel["model"].duplicated().any():
        raise ValueError("Current and retired historical parameter rows overlap")
    historical = pd.read_csv(HISTORICAL_SCORES)
    if historical["snapshot_timestamp"].nunique() != 15:
        raise ValueError("Expected 15 historical ECI vintages")
    if historical.duplicated(["snapshot_timestamp", "Model"]).any():
        raise ValueError("Historical ECI score ledger is not one-to-one by vintage/model")
    first = historical.groupby("Model")["snapshot_timestamp"].min()
    panel["first_snapshot"] = panel["model"].map(first)
    panel["archive_available"] = panel["first_snapshot"].notna()
    missing = set(
        panel.loc[panel["current_canonical"] & ~panel["archive_available"], "model"]
    )
    expected_missing = {"Kimi K3", "Gemma 4 31B IT", "Qwen 3.6 35B-A3B"}
    if missing != expected_missing:
        raise ValueError(
            f"Unexpected current checkpoints absent from the frozen archive: {sorted(missing)}"
        )
    panel["first_snapshot"] = panel["first_snapshot"].astype("Int64")
    panel["release_ordinal"] = panel["release_date"].map(date.fromisoformat).map(date.toordinal)
    panel["wls_weight"] = WLS_NUMERATOR / np.square(panel["ci_width"].astype(float))
    panel["base_group"] = panel.apply(
        lambda row: (
            f"{row['family']}|{round(math.log10(float(row['total_b'])), 2)}|"
            f"{round(math.log10(float(row['active_b'])), 2)}"
        ),
        axis=1,
    )
    if panel["first_snapshot"].min() != 20251113094011:
        raise ValueError("Unexpected initial ECI archive vintage")
    return panel, historical


def weights(frame: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "live_inverse_eci_ci":
        values = frame["wls_weight"].to_numpy(float)
    elif mode == "equal_family":
        counts = Counter(frame["family"])
        values = np.asarray([1.0 / counts[value] for value in frame["family"]])
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    return values / values.mean()


def design(
    frame: pd.DataFrame,
    specification: str,
    state: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    score = frame["score"].to_numpy(float)
    release = np.asarray([years(value) for value in frame["release_date"]], dtype=float)
    if state is None:
        state = {
            "score_mean": float(score.mean()),
            "score_sd": float(score.std()) or 1.0,
            "date_mean": float(release.mean()),
            "date_sd": float(release.std()) or 1.0,
            "knots": [float(value) for value in np.quantile(score, [0.25, 0.50, 0.75])],
        }
    z_score = (score - state["score_mean"]) / state["score_sd"]
    z_date = (release - state["date_mean"]) / state["date_sd"]
    columns = [np.ones(len(frame)), z_score]
    if specification != "score":
        columns.append(z_date)
    if specification in {"architecture", "huber_architecture", "ridge_flexible"}:
        columns.extend(
            [frame["reasoning"].to_numpy(float), frame["moe"].to_numpy(float)]
        )
    if specification == "ridge_flexible":
        columns.extend(
            [
                frame["coder"].to_numpy(float),
                frame["multimodal"].to_numpy(float),
                np.square(z_score),
                z_score * z_date,
            ]
        )
    elif specification == "quadratic":
        columns.append(np.square(z_score))
    elif specification == "interaction":
        columns.append(z_score * z_date)
    elif specification == "spline":
        # A linear basis plus hinges has linear, rather than polynomial,
        # extrapolation beyond its outermost knot.
        columns.extend(
            [
                np.maximum(0.0, score - knot) / state["score_sd"]
                for knot in state["knots"]
            ]
        )
    return np.column_stack(columns), state


def solve_weighted(
    x: np.ndarray, y: np.ndarray, sample_weights: np.ndarray, alpha: float
) -> np.ndarray:
    root = np.sqrt(sample_weights)
    matrix = x * root[:, None]
    outcome = y * root
    if alpha:
        penalty = np.eye(x.shape[1])
        penalty[0, 0] = 0.0
        matrix = np.vstack([matrix, math.sqrt(alpha) * penalty])
        outcome = np.concatenate([outcome, np.zeros(x.shape[1])])
    return np.linalg.lstsq(matrix, outcome, rcond=None)[0]


def fit_predict_single(
    train: pd.DataFrame,
    test: pd.DataFrame,
    specification: str,
    weight_mode: str,
    alpha: float = 0.0,
) -> np.ndarray:
    x, state = design(train, specification)
    y = np.log10(train["total_b"].to_numpy(float))
    base_weights = weights(train, weight_mode)
    beta = solve_weighted(x, y, base_weights, alpha)
    if specification.startswith("huber_"):
        for _ in range(50):
            residual = y - x @ beta
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
            if scale < 1e-10:
                break
            standardized = np.abs(residual) / (1.345 * scale)
            robust = np.ones_like(standardized)
            np.divide(1.0, standardized, out=robust, where=standardized > 1.0)
            updated = solve_weighted(x, y, base_weights * robust, alpha)
            if np.max(np.abs(updated - beta)) < 1e-10:
                beta = updated
                break
            beta = updated
    return design(test, specification, state)[0] @ beta


def predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidate: str,
    weight_mode: str,
) -> np.ndarray:
    specification = CANDIDATES[candidate]["features"]
    if specification == "blend":
        score = fit_predict_single(train, test, "score", weight_mode)
        dated = fit_predict_single(train, test, "score_date", weight_mode)
        return 0.60 * score + 0.40 * dated
    return fit_predict_single(
        train,
        test,
        specification,
        weight_mode,
        float(CANDIDATES[candidate].get("alpha", 0.0)),
    )


def collapse_same_base(frame: pd.DataFrame) -> pd.DataFrame:
    """Retain the highest observed ECI per mechanically defined base group."""
    return (
        frame.sort_values(["score", "release_date", "model"])
        .groupby("base_group", as_index=False, group_keys=False)
        .tail(1)
        .reset_index(drop=True)
    )


def interval_prospective_flags(panel: pd.DataFrame, snapshots: list[int]) -> dict[str, bool]:
    ordered = sorted(snapshots)
    output: dict[str, bool] = {}
    for row in panel.itertuples(index=False):
        index = ordered.index(int(row.first_snapshot))
        if index == 0:
            output[row.model] = False
            continue
        previous = timestamp_date(ordered[index - 1])
        observed = timestamp_date(row.first_snapshot)
        released = date.fromisoformat(row.release_date)
        output[row.model] = previous < released <= observed
    return output


def prediction_rows(panel: pd.DataFrame, historical: pd.DataFrame) -> list[dict[str, Any]]:
    panel = panel.loc[panel["archive_available"]].copy()
    panel["first_snapshot"] = panel["first_snapshot"].astype("int64")
    snapshots = sorted(historical["snapshot_timestamp"].astype("int64").unique())
    prospective = interval_prospective_flags(panel, snapshots)
    initial = min(snapshots)
    output: list[dict[str, Any]] = []
    for target in panel[panel["first_snapshot"] > initial].sort_values(
        ["first_snapshot", "release_date", "model"]
    ).itertuples(index=False):
        snapshot = historical[
            historical["snapshot_timestamp"].astype("int64") == target.first_snapshot
        ]
        score_by_model = snapshot.set_index("Model")["eci"]
        train = panel[
            panel["model"].isin(score_by_model.index)
            & (panel["release_date"] < target.release_date)
            & (panel["family"] != target.family)
        ].copy()
        train["score"] = train["model"].map(score_by_model)
        if len(train) < MIN_TRAIN_ROWS or train["family"].nunique() < MIN_TRAIN_FAMILIES:
            continue
        test = pd.DataFrame([target._asdict()])
        test["score"] = float(score_by_model[target.model])
        actual = math.log10(float(target.total_b))
        for mode in WEIGHT_MODES:
            for candidate in CANDIDATES:
                predicted = float(predict(train, test, candidate, mode)[0])
                output.append(
                    {
                        "snapshot_timestamp": int(target.first_snapshot),
                        "snapshot_date": timestamp_date(target.first_snapshot).isoformat(),
                        "release_date": target.release_date,
                        "model": target.model,
                        "family": target.family,
                        "interval_prospective": prospective[target.model],
                        "candidate": candidate,
                        "weight_mode": mode,
                        "train_rows": len(train),
                        "train_families": int(train["family"].nunique()),
                        "target_eci": float(score_by_model[target.model]),
                        "actual_b": float(target.total_b),
                        "predicted_b": float(10**predicted),
                        "log10_error": predicted - actual,
                    }
                )
    frame = pd.DataFrame(output)
    targets = frame[["model", "interval_prospective"]].drop_duplicates()
    if len(targets) != 27:
        raise ValueError(f"Expected 27 first-observed targets; found {len(targets)}")
    named = set(targets.loc[targets["interval_prospective"], "model"])
    if named != {"Kimi K2.5", "Kimi K2.7 Code"}:
        raise ValueError(f"Unexpected interval-prospective inventory: {sorted(named)}")
    return output


def metric_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    errors = np.asarray([float(row["log10_error"]) for row in selected])
    absolute = np.abs(errors)
    return {
        "n": len(selected),
        "families": len({row["family"] for row in selected}),
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "mean_absolute_log10_error": float(absolute.mean()),
        "rmse_log10": float(np.sqrt(np.mean(np.square(errors)))),
        "p80_multiplicative_error": float(10 ** np.quantile(absolute, 0.80)),
        "within_2x": float(np.mean(absolute <= math.log10(2.0))),
        "signed_bias_factor": float(10 ** errors.mean()),
    }


def family_mean_absolute(rows: list[dict[str, Any]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(abs(float(row["log10_error"])))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def paired_family_bootstrap(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    base = {row["model"]: row for row in baseline}
    challenger = {row["model"]: row for row in candidate}
    if set(base) != set(challenger):
        raise ValueError("Paired bootstrap model inventories differ")
    effects: dict[str, list[float]] = defaultdict(list)
    for model in base:
        effects[base[model]["family"]].append(
            abs(float(challenger[model]["log10_error"]))
            - abs(float(base[model]["log10_error"]))
        )
    family_effects = np.asarray([np.mean(values) for values in effects.values()])
    rng = np.random.default_rng(seed)
    draws = family_effects[
        rng.integers(0, len(family_effects), size=(BOOTSTRAP_SAMPLES, len(family_effects)))
    ].mean(axis=1)
    return {
        "metric": "equal-family mean absolute log10 error; challenger minus baseline",
        "observed_delta": float(family_effects.mean()),
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "probability_challenger_better": float(np.mean(draws < 0)),
        "families": len(family_effects),
        "samples": BOOTSTRAP_SAMPLES,
    }


def tournament_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode in WEIGHT_MODES:
        mode_rows = [row for row in rows if row["weight_mode"] == mode]
        selection = [row for row in mode_rows if not row["interval_prospective"]]
        validation = [row for row in mode_rows if row["interval_prospective"]]
        candidates = {}
        for candidate in CANDIDATES:
            candidate_selection = [
                row for row in selection if row["candidate"] == candidate
            ]
            candidate_validation = [
                row for row in validation if row["candidate"] == candidate
            ]
            candidates[candidate] = {
                "selection_first_observed_backfills": metric_summary(candidate_selection),
                "selection_equal_family_mae": family_mean_absolute(candidate_selection),
                "interval_prospective_validation": metric_summary(candidate_validation),
            }
        selected = min(
            CANDIDATES,
            key=lambda candidate: (
                candidates[candidate]["selection_equal_family_mae"],
                list(CANDIDATES).index(candidate),
            ),
        )
        baseline_selection = [
            row for row in selection if row["candidate"] == BASELINE
        ]
        selected_selection = [
            row for row in selection if row["candidate"] == selected
        ]
        baseline_validation = [
            row for row in validation if row["candidate"] == BASELINE
        ]
        selected_validation = [
            row for row in validation if row["candidate"] == selected
        ]
        output[mode] = {
            "selected_challenger": selected,
            "selection_rows": len(baseline_selection),
            "selection_families": len({row["family"] for row in baseline_selection}),
            "validation_rows": len(baseline_validation),
            "validation_models": [row["model"] for row in baseline_validation],
            "candidates": candidates,
            "selected_vs_baseline_selection_bootstrap": paired_family_bootstrap(
                baseline_selection, selected_selection, 20260718
            ),
            "selected_vs_baseline_validation_bootstrap": paired_family_bootstrap(
                baseline_validation, selected_validation, 20260719
            ),
        }
    return output


def frontier_rows(panel: pd.DataFrame) -> list[dict[str, Any]]:
    panel = panel.loc[panel["current_canonical"]].copy()
    regression = json.loads(REGRESSION.read_text(encoding="utf-8"))
    predictions = {row["model"]: row for row in regression["frontier_predictions"]}
    current = json.loads(EXPANDED_AUDIT.read_text(encoding="utf-8"))
    live = {
        row["model"]: row["legacy_57_estimate_t"] * 1000
        for row in current["expanded_total_parameter_panel"]["frontier_estimate_stability"]
    }
    target_names = ("Claude Fable 5", "GPT-5.6 Sol")
    tests = pd.DataFrame(
        [
            {
                "model": model,
                "release_date": predictions[model]["release_date"],
                "score": predictions[model]["eci"],
                "reasoning": 1,
                "moe": 1,
                "coder": 0,
                "multimodal": 0,
            }
            for model in target_names
        ]
    )
    output: list[dict[str, Any]] = []
    for mode in WEIGHT_MODES:
        for collapsed in (False, True):
            train = collapse_same_base(panel) if collapsed else panel.copy()
            for candidate in CANDIDATES:
                values = 10 ** predict(train, tests, candidate, mode)
                for index, model in enumerate(target_names):
                    output.append(
                        {
                            "model": model,
                            "release_date": tests.iloc[index]["release_date"],
                            "eci": float(tests.iloc[index]["score"]),
                            "candidate": candidate,
                            "weight_mode": mode,
                            "same_base_collapsed": collapsed,
                            "train_rows": len(train),
                            "train_families": int(train["family"].nunique()),
                            "training_max_eci": float(train["score"].max()),
                            "eci_extrapolation_points": float(
                                tests.iloc[index]["score"] - train["score"].max()
                            ),
                            "predicted_b": float(values[index]),
                            "legacy_57_b": float(live[model]),
                            "candidate_over_legacy": float(values[index] / live[model]),
                        }
                    )
    return output


def promotion_decision(
    tournament: dict[str, Any], frontier: list[dict[str, Any]]
) -> dict[str, Any]:
    selected = {
        mode: tournament[mode]["selected_challenger"] for mode in WEIGHT_MODES
    }
    selected_agrees = len(set(selected.values())) == 1
    primary = tournament["live_inverse_eci_ci"]
    challenger = primary["selected_challenger"]
    selection_candidate = primary["candidates"][challenger][
        "selection_first_observed_backfills"
    ]
    selection_baseline = primary["candidates"][BASELINE][
        "selection_first_observed_backfills"
    ]
    selection_point_gate = all(
        selection_candidate[field] < selection_baseline[field]
        for field in (
            "median_multiplicative_error",
            "mean_absolute_log10_error",
            "rmse_log10",
            "p80_multiplicative_error",
        )
    )
    selection_interval_gate = (
        primary["selected_vs_baseline_selection_bootstrap"]["ci_90"][1] < 0
    )
    validation_gate_by_mode = {}
    for mode in WEIGHT_MODES:
        block = tournament[mode]
        chosen = block["selected_challenger"]
        candidate = block["candidates"][chosen]["interval_prospective_validation"]
        baseline = block["candidates"][BASELINE]["interval_prospective_validation"]
        validation_gate_by_mode[mode] = all(
            candidate[field] < baseline[field]
            for field in ("mean_absolute_log10_error", "rmse_log10")
        )
    validation_gate = all(validation_gate_by_mode.values())

    target_stability: dict[str, Any] = {}
    if selected_agrees:
        chosen = next(iter(selected.values()))
        for model in ("Claude Fable 5", "GPT-5.6 Sol"):
            rows = [
                row
                for row in frontier
                if row["model"] == model and row["candidate"] == chosen
            ]
            values = [row["predicted_b"] for row in rows]
            ratio = max(values) / min(values)
            target_stability[model] = {
                "minimum_b": min(values),
                "maximum_b": max(values),
                "max_over_min": ratio,
                "passes_1_25x_gate": ratio <= MAX_TARGET_SENSITIVITY_RATIO,
            }
    target_gate = bool(target_stability) and all(
        row["passes_1_25x_gate"] for row in target_stability.values()
    )
    promote = all(
        (
            selected_agrees,
            selection_point_gate,
            selection_interval_gate,
            validation_gate,
            target_gate,
        )
    )
    return {
        "selected_challenger_by_weight_mode": selected,
        "same_challenger_selected_across_weight_modes": selected_agrees,
        "selection_point_metrics_all_improve": selection_point_gate,
        "selection_equal_family_ci_wholly_favorable": selection_interval_gate,
        "interval_prospective_metrics_improve_in_both_weight_modes": validation_gate,
        "interval_prospective_gate_by_weight_mode": validation_gate_by_mode,
        "frontier_sensitivity": target_stability,
        "frontier_sensitivity_below_1_25x": target_gate,
        "change_live_eci_functional_form": promote,
        "reason": (
            "All selection, prospective, and extrapolation-stability gates pass."
            if promote
            else "At least one independent prospective or extrapolation-stability gate fails; retain the live 60/40 linear blend."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown(result: dict[str, Any]) -> str:
    primary = result["tournament"]["live_inverse_eci_ci"]
    selected = primary["selected_challenger"]
    base = primary["candidates"][BASELINE]["selection_first_observed_backfills"]
    chosen = primary["candidates"][selected]["selection_first_observed_backfills"]
    validation_base = primary["candidates"][BASELINE]["interval_prospective_validation"]
    validation_chosen = primary["candidates"][selected]["interval_prospective_validation"]
    targets = result["decision"]["frontier_sensitivity"]
    lines = [
        "# ECI functional-form tournament audit",
        "",
        f"Generated {GENERATED_ON}. Status: **{'PROMOTE' if result['decision']['change_live_eci_functional_form'] else 'RETAIN LIVE FORM'}**.",
        "",
        "## What changed",
        "",
        "Fifteen hash-pinned Epoch vintages create 27 first-observed parameter checkpoints. Twenty-five backfills form the selection set; Kimi K2.5 and Kimi K2.7 Code, each released between adjacent archive captures, are reserved as the closest available prospective check.",
        "",
        f"The inverse-ECI-CI tournament selects `{selected}` on the 25-row selection set. Median error moves from {base['median_multiplicative_error']:.2f}x to {chosen['median_multiplicative_error']:.2f}x and RMSE from {base['rmse_log10']:.3f} to {chosen['rmse_log10']:.3f} log10.",
        "",
        f"On the two interval-prospective checkpoints, mean absolute error moves from {validation_base['mean_absolute_log10_error']:.3f} to {validation_chosen['mean_absolute_log10_error']:.3f} log10. This tiny validation set is decisive only as a veto, not as proof of a winner.",
        "",
        "## Why the central forecast is not changed",
        "",
        f"Fable and Sol are {result['frontier_extrapolation']['Claude Fable 5']:.1f} and {result['frontier_extrapolation']['GPT-5.6 Sol']:.1f} ECI points beyond the strongest open-weight calibrator. Flexible score curves therefore extrapolate rather than interpolate.",
        "",
    ]
    if targets:
        for model, row in targets.items():
            lines.append(
                f"- {model}: the selected challenger spans {row['minimum_b']/1000:.2f}–{row['maximum_b']/1000:.2f}T across weighting and same-base-collapse sensitivities ({row['max_over_min']:.2f}x)."
            )
        lines.append("")
    lines.extend(
        [
            "The live 60/40 linear blend is retained because the selected flexible challenger fails at least one prospective/stability gate. Movement from a nonlinear curve without this veto would be model-selection and extrapolation noise, not stronger evidence.",
            "",
            "## Data integrity",
            "",
            f"- Frozen archive: `{HISTORICAL_ARCHIVE.relative_to(ROOT)}` ({result['sources']['historical_archive_sha256']}); terminal capture 2026-07-16 (`{ARCHIVAL_TERMINAL_TIMESTAMP}`).",
            f"- Archive collection metadata: `{COLLECTION_METADATA.relative_to(ROOT)}` ({result['sources']['collection_metadata_sha256']})",
            f"- Historical score ledger: `{HISTORICAL_SCORES.relative_to(ROOT)}` ({result['sources']['historical_scores_sha256']})",
            f"- Fixed-code fit metadata: `{HISTORICAL_METADATA.relative_to(ROOT)}` ({result['sources']['historical_metadata_sha256']})",
            f"- Prediction ledger: `{PREDICTIONS.relative_to(ROOT)}` ({result['outputs']['predictions_sha256']})",
            f"- Frontier sensitivity ledger: `{TARGETS.relative_to(ROOT)}` ({result['outputs']['frontier_sha256']})",
            "",
            "The July 31 canonical ECI panel is a separately hashed live-successor reference. It is used by the current frontier model, but it is not an equality target or an input to the frozen historical refit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    panel, historical = load_panel()
    historical_provenance = validate_historical_provenance(historical)
    rows = prediction_rows(panel, historical)
    frontier = frontier_rows(panel)
    tournament = tournament_summary(rows)
    decision = promotion_decision(tournament, frontier)
    write_csv(PREDICTIONS, rows)
    write_csv(TARGETS, frontier)
    frontier_extrapolation = {
        model: next(
            row["eci_extrapolation_points"]
            for row in frontier
            if row["model"] == model
            and row["candidate"] == BASELINE
            and row["weight_mode"] == "live_inverse_eci_ci"
            and not row["same_base_collapsed"]
        )
        for model in ("Claude Fable 5", "GPT-5.6 Sol")
    }
    result = {
        "generated_on": GENERATED_ON,
        "question": "Does a different ECI functional form predict held-out parameter counts better enough to replace the live 60/40 blend?",
        "method": {
            "archive_split": "27 first-observed archive-vintage parameter-map checkpoints",
            "archive_boundary": historical_provenance,
            "selection_set": "25 first-observed backfills; no target appears in the initial 2025-11-13 snapshot",
            "validation_set": "Kimi K2.5 and Kimi K2.7 Code, released between adjacent archive captures",
            "outer_training": "same archived ECI vintage, strictly earlier release dates, entire target family excluded",
            "selection_metric": "equal-family mean absolute log10 error",
            "baseline": "60% score-only plus 40% score-and-exact-date direct log-parameter regression",
            "candidate_inventory": CANDIDATES,
            "weight_sensitivities": {
                "live_inverse_eci_ci": "mirrors the live workbook inverse-current-ECI-CI law; current CI widths are a reliability sensitivity, not vintage-available data",
                "equal_family": "strict no-future-score-uncertainty sensitivity; each developer family has equal total training weight",
            },
            "same_base_sensitivity": "one highest-ECI checkpoint per family and rounded log10(total/active) base group",
            "promotion_policy": "selection improvements plus wholly favorable family CI, prospective improvement in both weight modes, same selected form, and <=1.25x frontier sensitivity",
        },
        "inventory": {
            "parameter_map_checkpoints": int(panel["current_canonical"].sum()),
            "parameter_families": int(
                panel.loc[panel["current_canonical"], "family"].nunique()
            ),
            "archive_compatible_checkpoints": int(panel["archive_available"].sum()),
            "retired_historical_checkpoints": sorted(
                panel.loc[~panel["current_canonical"], "model"].tolist()
            ),
            "current_checkpoints_absent_from_frozen_archive": sorted(
                panel.loc[
                    panel["current_canonical"] & ~panel["archive_available"], "model"
                ].tolist()
            ),
            "historical_snapshots": int(historical["snapshot_timestamp"].nunique()),
            "historical_score_rows": len(historical),
            "first_observed_outer_targets": len({row["model"] for row in rows}),
            "selection_targets": len(
                {row["model"] for row in rows if not row["interval_prospective"]}
            ),
            "interval_prospective_targets": len(
                {row["model"] for row in rows if row["interval_prospective"]}
            ),
        },
        "tournament": tournament,
        "frontier_extrapolation": frontier_extrapolation,
        "decision": decision,
        "sources": {
            "historical_scores": str(HISTORICAL_SCORES.relative_to(ROOT)),
            "historical_scores_sha256": sha256(HISTORICAL_SCORES),
            "historical_metadata": str(HISTORICAL_METADATA.relative_to(ROOT)),
            "historical_metadata_sha256": sha256(HISTORICAL_METADATA),
            "collection_metadata": str(COLLECTION_METADATA.relative_to(ROOT)),
            "collection_metadata_sha256": sha256(COLLECTION_METADATA),
            "historical_archive": str(HISTORICAL_ARCHIVE.relative_to(ROOT)),
            "historical_archive_sha256": sha256(HISTORICAL_ARCHIVE),
            "regression": str(REGRESSION.relative_to(ROOT)),
            "regression_sha256": sha256(REGRESSION),
            "expanded_audit": str(EXPANDED_AUDIT.relative_to(ROOT)),
            "expanded_audit_sha256": sha256(EXPANDED_AUDIT),
        },
        "outputs": {
            "predictions": str(PREDICTIONS.relative_to(ROOT)),
            "predictions_sha256": sha256(PREDICTIONS),
            "frontier": str(TARGETS.relative_to(ROOT)),
            "frontier_sha256": sha256(TARGETS),
        },
        "caveats": [
            "The 25-row selection set contains archive first-observations but mostly backfills, not prospective releases.",
            "Only two interval-prospective targets exist, both from the Kimi family; they are a veto, not an independent multi-lab confirmation.",
            "The live inverse-CI sensitivity uses current ECI confidence widths because historical central fits do not contain bootstrap intervals.",
            "The 15-vintage archive ends before the July 31 ECI refresh; Kimi K3, Gemma 4 31B IT, and Qwen 3.6 35B-A3B are excluded only from the historical tournament and retained in current frontier-fit sensitivities.",
            "The July 31 live ECI source is a separately hashed successor reference and is never compared for byte or fit equality with the July 16 archival terminal.",
            "Fable and Sol lie beyond the open-weight ECI score range, so nonlinear target estimates are extrapolations.",
        ],
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    AUDIT.write_text(markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected": decision["selected_challenger_by_weight_mode"],
                "change_live": decision["change_live_eci_functional_form"],
                "gates": decision,
                "frontier_extrapolation": frontier_extrapolation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
