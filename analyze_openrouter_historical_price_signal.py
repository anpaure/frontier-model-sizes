#!/usr/bin/env python3
"""Prospectively audit launch-vintage OpenRouter prices as size signals.

Unlike the current-price audits, every held-out model receives only prices
observed near its own OpenRouter onboarding date.  Price windows are considered
available only after the complete window has elapsed, training models must have
become available strictly before the test model, and the test developer is
always excluded.  Current 2026-07-18 prices are evaluated only as an explicitly
non-prospective comparison.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from frontier_target_signals import AA_TARGET_SIGNALS

import analyze_openrouter_active_price_signal as active_audit
from k3_primary_evidence import K3_EVIDENCE_PATH, K3_PARAMETER_SOURCE, K3_TOTAL_T


ROOT = Path(__file__).resolve().parent
COMPATIBILITY_FILE_DATE = "2026-07-18"
DATE = COMPATIBILITY_FILE_DATE
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RAW = ROOT / f"sources/openrouter_historical_price_ledger_{DATE}.json.gz"
CHANGE_POINTS = ROOT / f"sources/openrouter_historical_price_change_points_{DATE}.csv"
COLLECTION_METADATA = ROOT / f"sources/openrouter_historical_price_collection_metadata_{DATE}.json"
CALIBRATION = OUT / f"openrouter_parameter_calibration_{DATE}.csv"
CURRENT_MODELS = ROOT / f"sources/openrouter_model_signals_{DATE}.csv"

MATCH_AUDIT = OUT / f"openrouter_historical_price_match_audit_{DATE}.csv"
PREDICTIONS = OUT / f"openrouter_historical_price_backtest_predictions_{DATE}.csv"
TARGETS = OUT / f"openrouter_historical_price_frontier_targets_{DATE}.csv"
RESULT = OUT / f"openrouter_historical_price_audit_{DATE}.json"

HISTORY_FLOOR = date(2024, 9, 21)
MAX_ABS_ONBOARDING_LAG_DAYS = 30
WINDOWS = (1, 3, 7, 14, 30, 60, 90)
MIN_TOTAL_TRAIN = 20
MIN_ACTIVE_TRAIN = 12
MIN_TRAIN_DEVELOPERS = 5
BOOTSTRAP_SAMPLES = 20_000
SEED = 20_260_718
DATE_ORIGIN = date(2023, 1, 1)

TOTAL_SPECS = {
    "date_only": ("date",),
    "date_historical_price": ("date", "historical_price"),
    "date_current_price_nonprospective": ("date", "current_price"),
}
ACTIVE_SPECS = {
    "date_historical_price": ("date", "historical_price"),
    "score_date": ("score", "date"),
    "score_date_historical_price": ("score", "date", "historical_price"),
    "score_date_current_price_nonprospective": ("score", "date", "current_price"),
}

FRONTIER_TARGETS = (
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
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ledger() -> dict[str, Any]:
    with gzip.open(RAW, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload["model_count"] != len(payload["models"]):
        raise ValueError("Historical price ledger model_count does not reconcile")
    return payload


def years(value: str) -> float:
    return (date.fromisoformat(value) - DATE_ORIGIN).days / 365.25


def daily_price_window(
    record: dict[str, Any], days: int, *, allow_partial: bool = False
) -> dict[str, Any]:
    """Return an equal-day-weighted price window from ordered change points.

    Multiple changes on one date are collapsed to the last observed price that
    date.  Free/unavailable days remain explicit but are excluded from log-price
    medians rather than being silently replaced with an earlier paid price.
    """

    first = date.fromisoformat(record["first_seen"])
    last = date.fromisoformat(record["last_seen"])
    requested_end = first + timedelta(days=days - 1)
    complete = requested_end <= last
    if not complete and not allow_partial:
        return {
            "complete": False,
            "requested_days": days,
            "observed_days": max(0, (last - first).days + 1),
        }
    end = min(requested_end, last)
    changes = {
        date.fromisoformat(effective): (prompt, completion)
        for effective, prompt, completion in record["points"]
    }
    current: tuple[float | None, float | None] = (None, None)
    prompt_values: list[float] = []
    completion_values: list[float] = []
    blended_values: list[float] = []
    free_days = 0
    unavailable_days = 0
    observed_days = (end - first).days + 1
    for offset in range(observed_days):
        day = first + timedelta(days=offset)
        if day in changes:
            current = changes[day]
        prompt, completion = current
        if prompt is None or completion is None:
            unavailable_days += 1
        elif prompt <= 0 or completion <= 0:
            free_days += 1
        else:
            prompt_values.append(float(prompt))
            completion_values.append(float(completion))
            blended_values.append(math.sqrt(float(prompt) * float(completion)))
    return {
        "complete": complete,
        "requested_days": days,
        "observed_days": observed_days,
        "availability_date": end.isoformat(),
        "positive_price_days": len(blended_values),
        "free_or_partly_free_days": free_days,
        "unavailable_days": unavailable_days,
        "prompt_median": statistics.median(prompt_values) if prompt_values else None,
        "completion_median": statistics.median(completion_values)
        if completion_values
        else None,
        "blended_median": statistics.median(blended_values) if blended_values else None,
    }


def checkpoint_window(
    calibration: dict[str, str],
    models: dict[str, dict[str, Any]],
    days: int,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    release = date.fromisoformat(calibration["release_date"])
    aliases: list[dict[str, Any]] = []
    for model_id in calibration["openrouter_model_ids"].split("|"):
        record = models[model_id]
        first = date.fromisoformat(record["first_seen"])
        lag = (first - release).days
        window = daily_price_window(record, days, allow_partial=allow_partial)
        eligible_identity = (
            release >= HISTORY_FLOOR and abs(lag) <= MAX_ABS_ONBOARDING_LAG_DAYS
        )
        aliases.append(
            {
                "model_id": model_id,
                "first_seen": record["first_seen"],
                "last_seen": record["last_seen"],
                "onboarding_lag_days": lag,
                "eligible_identity": eligible_identity,
                **window,
            }
        )
    usable = [
        alias
        for alias in aliases
        if alias["eligible_identity"]
        and (alias.get("complete") or allow_partial)
        and alias.get("blended_median") is not None
    ]
    if not usable:
        return {"eligible": False, "aliases": aliases}
    return {
        "eligible": True,
        "aliases": aliases,
        "usable_aliases": [alias["model_id"] for alias in usable],
        "availability_date": max(alias["availability_date"] for alias in usable),
        "blended_median": statistics.median(
            alias["blended_median"] for alias in usable
        ),
        "prompt_median": statistics.median(alias["prompt_median"] for alias in usable),
        "completion_median": statistics.median(
            alias["completion_median"] for alias in usable
        ),
        "positive_price_days": sum(alias["positive_price_days"] for alias in usable),
        "complete": all(alias["complete"] for alias in usable),
    }


def build_match_audit(
    calibration: list[dict[str, str]], models: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    audits: list[dict[str, Any]] = []
    window_index: dict[tuple[str, int], dict[str, Any]] = {}
    for row in calibration:
        ids = row["openrouter_model_ids"].split("|")
        missing = [model_id for model_id in ids if model_id not in models]
        if missing:
            raise ValueError(f"Historical ledger missing exact aliases: {missing}")
        release = date.fromisoformat(row["release_date"])
        lags = [
            (date.fromisoformat(models[model_id]["first_seen"]) - release).days
            for model_id in ids
        ]
        audit: dict[str, Any] = {
            "canonical_checkpoint_id": row["canonical_checkpoint_id"],
            "canonical_model_name": row["canonical_model_name"],
            "developer": row["family"],
            "release_date": row["release_date"],
            "total_parameters_b": row["total_parameters_b"],
            "openrouter_model_ids": row["openrouter_model_ids"],
            "all_aliases_exactly_matched": True,
            "alias_count": len(ids),
            "history_floor": HISTORY_FLOOR.isoformat(),
            "release_after_history_floor": release >= HISTORY_FLOOR,
            "first_seen_dates": "|".join(models[model_id]["first_seen"] for model_id in ids),
            "last_seen_dates": "|".join(models[model_id]["last_seen"] for model_id in ids),
            "onboarding_lag_days": "|".join(str(value) for value in lags),
            "identity_lag_within_30_days": all(
                abs(value) <= MAX_ABS_ONBOARDING_LAG_DAYS for value in lags
            ),
        }
        for days in WINDOWS:
            signal = checkpoint_window(row, models, days)
            window_index[(row["canonical_checkpoint_id"], days)] = signal
            prefix = f"window_{days}d"
            audit[f"{prefix}_eligible"] = signal["eligible"]
            audit[f"{prefix}_usable_aliases"] = "|".join(
                signal.get("usable_aliases", [])
            )
            audit[f"{prefix}_availability_date"] = signal.get(
                "availability_date", ""
            )
            audit[f"{prefix}_blended_price_usd_per_mtoken"] = signal.get(
                "blended_median", ""
            )
            audit[f"{prefix}_positive_alias_days"] = signal.get(
                "positive_price_days", ""
            )
        audits.append(audit)
    if len(audits) != 93 or len({row["canonical_checkpoint_id"] for row in audits}) != 93:
        raise ValueError("Historical price audit must preserve exactly 93 calibration checkpoints")
    return audits, window_index


def family_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = Counter(row["developer"] for row in rows)
    weights = np.asarray(
        [
            (0.5 if row.get("estimated_score") else 1.0)
            / counts[row["developer"]]
            for row in rows
        ],
        dtype=float,
    )
    return weights / weights.mean()


def design(rows: list[dict[str, Any]], features: tuple[str, ...]) -> np.ndarray:
    output: list[list[float]] = []
    for row in rows:
        values = [1.0]
        for feature in features:
            if feature == "date":
                values.append(years(row["release_date"]))
            elif feature == "score":
                values.append(float(row["score"]))
            elif feature == "historical_price":
                values.append(math.log10(float(row["historical_price"])))
            elif feature == "current_price":
                values.append(math.log10(float(row["current_price"])))
            else:
                raise ValueError(feature)
        output.append(values)
    return np.asarray(output, dtype=float)


def fit(
    rows: list[dict[str, Any]], target: str, features: tuple[str, ...]
) -> np.ndarray:
    matrix = design(rows, features)
    values = np.log10(np.asarray([float(row[target]) for row in rows]))
    root_weight = np.sqrt(family_weights(rows))
    beta, *_ = np.linalg.lstsq(
        matrix * root_weight[:, None], values * root_weight, rcond=None
    )
    return beta


def predict(beta: np.ndarray, row: dict[str, Any], features: tuple[str, ...]) -> float:
    return float(10 ** (design([row], features) @ beta).item())


def metric_summary(errors: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(errors), dtype=float)
    absolute = np.abs(values)
    return {
        "n": int(len(values)),
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "mean_absolute_log10_error": float(np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(values**2))),
        "p80_multiplicative_error": float(10 ** np.quantile(absolute, 0.80)),
        "within_2x": float(np.mean(absolute <= math.log10(2))),
        "signed_bias_factor": float(10 ** np.mean(values)),
    }


def build_panels(
    calibration: list[dict[str, str]],
    window_index: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    active_panel, _, _ = active_audit.build_exact_panel()
    active_by_checkpoint = {
        row["canonical_checkpoint_id"]: row for row in active_panel
    }
    total: dict[int, list[dict[str, Any]]] = defaultdict(list)
    active: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source in calibration:
        checkpoint = source["canonical_checkpoint_id"]
        for days in WINDOWS:
            signal = window_index[(checkpoint, days)]
            if not signal["eligible"]:
                continue
            common = {
                "canonical_checkpoint_id": checkpoint,
                "model": source["canonical_model_name"],
                "release_date": source["release_date"],
                "price_availability_date": signal["availability_date"],
                "historical_price": float(signal["blended_median"]),
                "current_price": float(source["blended_price_usd_per_mtoken"]),
                "total_b": float(source["total_parameters_b"]),
                "window_days": days,
                "estimated_score": False,
            }
            total[days].append({**common, "developer": source["family"]})
            active_source = active_by_checkpoint.get(checkpoint)
            if active_source:
                active[days].append(
                    {
                        **common,
                        "developer": active_source["developer"],
                        "score": float(active_source["score"]),
                        "estimated_score": bool(active_source["estimated_score"]),
                        "active_b": float(active_source["active_b"]),
                        "total_b": float(active_source["total_b"]),
                    }
                )
    return total, active


def chronological_predictions(
    panel: list[dict[str, Any]],
    panel_name: str,
    targets: tuple[str, ...],
    specs: dict[str, tuple[str, ...]],
    minimum_train: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for test in sorted(
        panel, key=lambda row: (row["price_availability_date"], row["model"])
    ):
        train = [
            row
            for row in panel
            if row["price_availability_date"] < test["price_availability_date"]
            and row["developer"] != test["developer"]
        ]
        if (
            len(train) < minimum_train
            or len({row["developer"] for row in train}) < MIN_TRAIN_DEVELOPERS
        ):
            continue
        for target in targets:
            for spec, features in specs.items():
                estimate = predict(fit(train, target, features), test, features)
                output.append(
                    {
                        "panel": panel_name,
                        "window_days": test["window_days"],
                        "canonical_checkpoint_id": test["canonical_checkpoint_id"],
                        "model": test["model"],
                        "developer": test["developer"],
                        "release_date": test["release_date"],
                        "price_availability_date": test["price_availability_date"],
                        "historical_blended_price_usd_per_mtoken": test[
                            "historical_price"
                        ],
                        "current_blended_price_usd_per_mtoken": test["current_price"],
                        "target": target,
                        "specification": spec,
                        "actual_parameters_b": test[target],
                        "predicted_parameters_b": estimate,
                        "log10_error": math.log10(estimate / test[target]),
                        "train_n": len(train),
                        "train_developers": len(
                            {row["developer"] for row in train}
                        ),
                        "train_max_price_availability_date": max(
                            row["price_availability_date"] for row in train
                        ),
                        "test_developer_excluded": True,
                        "historical_price_is_prospective_at_fold_date": True,
                        "current_price_is_nonprospective_comparison": "current_price"
                        in features,
                    }
                )
    return output


def select_predictions(
    predictions: list[dict[str, Any]],
    *,
    panel: str,
    window: int,
    target: str,
    specification: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in predictions
        if row["panel"] == panel
        and row["window_days"] == window
        and row["target"] == target
        and row["specification"] == specification
    ]


def paired_developer_bootstrap(
    predictions: list[dict[str, Any]],
    *,
    panel: str,
    window: int,
    target: str,
    candidate: str,
    baseline: str,
) -> dict[str, Any]:
    selected = [
        row
        for row in predictions
        if row["panel"] == panel
        and row["window_days"] == window
        and row["target"] == target
        and row["specification"] in {candidate, baseline}
    ]
    keyed: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        keyed[(row["canonical_checkpoint_id"], row["developer"])][
            row["specification"]
        ] = row
    pairs = [value for value in keyed.values() if candidate in value and baseline in value]
    grouped: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        developer = pair[candidate]["developer"]
        grouped[developer].append(
            abs(pair[candidate]["log10_error"])
            - abs(pair[baseline]["log10_error"])
        )
    developer_delta = {
        developer: float(np.mean(values)) for developer, values in grouped.items()
    }
    developers = sorted(developer_delta)
    rng = np.random.default_rng(SEED + window)
    draws = np.empty(BOOTSTRAP_SAMPLES)
    for index in range(BOOTSTRAP_SAMPLES):
        sampled = rng.choice(developers, len(developers), replace=True)
        draws[index] = np.mean([developer_delta[item] for item in sampled])
    return {
        "panel": panel,
        "window_days": window,
        "target": target,
        "candidate": candidate,
        "baseline": baseline,
        "metric": "equal-developer absolute log10 error; candidate minus baseline",
        "paired_checkpoints": len(pairs),
        "developers": len(developers),
        "observed_delta": float(np.mean(list(developer_delta.values()))),
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "probability_candidate_better": float(np.mean(draws < 0)),
        "samples": BOOTSTRAP_SAMPLES,
    }


def frontier_sensitivity(
    active_panels: dict[int, list[dict[str, Any]]],
    models: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    days = 1
    cutoff = min(target["release_date"] for target in FRONTIER_TARGETS)
    training = [
        row
        for row in active_panels[days]
        if row["price_availability_date"] < cutoff
        and row["developer"] not in {"anthropic", "openai", "kimi"}
    ]
    targets: list[dict[str, Any]] = []
    for source in FRONTIER_TARGETS:
        record = models[source["openrouter_model_id"]]
        window = daily_price_window(record, days)
        if not window["complete"] or window["blended_median"] is None:
            raise ValueError(f"No complete first-day price for {source['model']}")
        targets.append(
            {
                **source,
                "historical_price": float(window["blended_median"]),
                "prompt_price": float(window["prompt_median"]),
                "completion_price": float(window["completion_median"]),
                "price_availability_date": window["availability_date"],
                "estimated_score": False,
            }
        )
    specs = {
        "date_historical_price": ("date", "historical_price"),
        "score_date_historical_price": ("score", "date", "historical_price"),
    }
    predicted: dict[tuple[str, str], float] = {}
    coefficients: dict[str, list[float]] = {}
    for spec, features in specs.items():
        beta = fit(training, "active_b", features)
        coefficients[spec] = [float(value) for value in beta]
        for target in targets:
            predicted[(target["model"], spec)] = predict(beta, target, features)
    output: list[dict[str, Any]] = []
    for target in targets:
        row: dict[str, Any] = {
            "model": target["model"],
            "release_date": target["release_date"],
            "openrouter_first_seen": models[target["openrouter_model_id"]][
                "first_seen"
            ],
            "aa_score": target["score"],
            "first_day_prompt_price_usd_per_mtoken": target["prompt_price"],
            "first_day_completion_price_usd_per_mtoken": target["completion_price"],
            "first_day_blended_price_usd_per_mtoken": target["historical_price"],
            "status": "disclosed total anchor"
            if target["model"] == "Kimi K3"
            else "zero-weight historical-price sensitivity",
        }
        for spec in specs:
            active_prediction = predicted[(target["model"], spec)]
            k3_active = predicted[("Kimi K3", spec)]
            row[f"predicted_active_{spec}_b"] = active_prediction
            row[f"k3_anchored_total_{spec}_t"] = (
                K3_TOTAL_T * active_prediction / k3_active
            )
        output.append(row)
    return output, {
        "price_window_days": days,
        "common_training_cutoff": cutoff,
        "training_rows": len(training),
        "training_developers": len({row["developer"] for row in training}),
        "excluded_target_developers": ["anthropic", "openai", "kimi"],
        "coefficients": coefficients,
        "interpretation": "K3-anchored zero-weight sensitivity; not an independent forecast branch",
    }


def main() -> None:
    collection_metadata = json.loads(COLLECTION_METADATA.read_text(encoding="utf-8"))
    observation_date = str(collection_metadata["generated_on"])
    ledger = load_ledger()
    models = ledger["models"]
    calibration = read_csv(CALIBRATION)
    audits, window_index = build_match_audit(calibration, models)
    total_panels, active_panels = build_panels(calibration, window_index)
    write_csv(MATCH_AUDIT, audits)

    predictions: list[dict[str, Any]] = []
    for days in WINDOWS:
        predictions.extend(
            chronological_predictions(
                total_panels[days],
                "total_calibration",
                ("total_b",),
                TOTAL_SPECS,
                MIN_TOTAL_TRAIN,
            )
        )
        predictions.extend(
            chronological_predictions(
                active_panels[days],
                "active_label_common_panel",
                ("active_b", "total_b"),
                ACTIVE_SPECS,
                MIN_ACTIVE_TRAIN,
            )
        )
    write_csv(PREDICTIONS, predictions)

    metrics: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []
    for days in WINDOWS:
        metrics[str(days)] = {
            "total_panel_rows": len(total_panels[days]),
            "active_panel_rows": len(active_panels[days]),
            "total": {
                spec: metric_summary(
                    row["log10_error"]
                    for row in select_predictions(
                        predictions,
                        panel="total_calibration",
                        window=days,
                        target="total_b",
                        specification=spec,
                    )
                )
                for spec in TOTAL_SPECS
            },
            "active_common_panel": {
                target: {
                    spec: metric_summary(
                        row["log10_error"]
                        for row in select_predictions(
                            predictions,
                            panel="active_label_common_panel",
                            window=days,
                            target=target,
                            specification=spec,
                        )
                    )
                    for spec in ACTIVE_SPECS
                }
                for target in ("active_b", "total_b")
            },
        }
        comparisons.extend(
            [
                paired_developer_bootstrap(
                    predictions,
                    panel="total_calibration",
                    window=days,
                    target="total_b",
                    candidate="date_historical_price",
                    baseline="date_only",
                ),
                paired_developer_bootstrap(
                    predictions,
                    panel="active_label_common_panel",
                    window=days,
                    target="active_b",
                    candidate="score_date_historical_price",
                    baseline="score_date",
                ),
                paired_developer_bootstrap(
                    predictions,
                    panel="active_label_common_panel",
                    window=days,
                    target="total_b",
                    candidate="score_date_historical_price",
                    baseline="score_date",
                ),
            ]
        )

    frontier_rows, frontier_method = frontier_sensitivity(active_panels, models)
    write_csv(TARGETS, frontier_rows)
    total_comparisons = [
        item for item in comparisons if item["panel"] == "total_calibration"
    ]
    active_incremental = [
        item
        for item in comparisons
        if item["panel"] == "active_label_common_panel"
        and item["target"] == "active_b"
    ]
    total_price_robust_across_windows = all(
        item["observed_delta"] < 0 and item["ci_90"][1] < 0
        for item in total_comparisons
    )
    active_price_incremental_robust = any(
        item["observed_delta"] < 0 and item["ci_90"][1] < 0
        for item in active_incremental
    )
    target_by_model = {row["model"]: row for row in frontier_rows}
    result = {
        "metadata": {
            "generated_on": observation_date,
            "compatibility_filename_date": COMPATIBILITY_FILE_DATE,
            "question": "Do launch-vintage OpenRouter prices prospectively recover model size, and do they add information beyond score plus date?",
            "history_floor": HISTORY_FLOOR.isoformat(),
            "price_windows_days": WINDOWS,
            "price_window_definition": "equal calendar-day median of positive geometric-mean prompt/completion price; complete window required; aliases receive equal weight",
            "identity_eligibility": f"exact audited OpenRouter alias, release on/after history floor, absolute onboarding lag <= {MAX_ABS_ONBOARDING_LAG_DAYS} days",
            "evaluation_split": "strict price-availability-ordered developer-held-out folds",
            "current_price_comparator": f"{observation_date} price, explicitly labeled non-prospective",
        },
        "inventory": {
            "historical_ledger_models": ledger["model_count"],
            "historical_change_points": sum(
                len(record["points"]) for record in models.values()
            ),
            "calibration_checkpoints_audited": len(audits),
            "calibration_aliases_missing_from_history": 0,
            "duplicate_calibration_checkpoints": len(audits)
            - len({row["canonical_checkpoint_id"] for row in audits}),
            "eligible_total_rows_by_window": {
                str(days): len(total_panels[days]) for days in WINDOWS
            },
            "eligible_active_rows_by_window": {
                str(days): len(active_panels[days]) for days in WINDOWS
            },
            "prediction_rows": len(predictions),
        },
        "heldout_metrics": metrics,
        "paired_developer_bootstraps": comparisons,
        "frontier_first_day_price_sensitivity": frontier_rows,
        "frontier_sensitivity_method": frontier_method,
        "decision": {
            "launch_vintage_price_predicts_total_beyond_date_robustly_across_all_windows": total_price_robust_across_windows,
            "launch_vintage_price_adds_robust_information_beyond_score_date_for_active_parameters": active_price_incremental_robust,
            "change_existing_live_price_weight": False,
            "incremental_live_weight_from_this_audit": 0.0,
            "headline_forecasts_changed": False,
            "reason": "Launch-vintage price robustly beats date alone for total parameters, removing the old temporal-leakage objection. But on the exact active-parameter/AA-score common panel, no tested window has a developer-bootstrap interval wholly below zero beyond score plus date. This validates price as a correlated mechanism, not a new independent evidence branch.",
        },
        "headline_crosscheck": {
            "fable_k3_anchored_score_date_first_day_price_t": target_by_model[
                "Claude Fable 5"
            ]["k3_anchored_total_score_date_historical_price_t"],
            "sol_k3_anchored_score_date_first_day_price_t": target_by_model[
                "GPT-5.6 Sol"
            ]["k3_anchored_total_score_date_historical_price_t"],
            "k3_disclosed_t": K3_TOTAL_T,
            "k3_parameter_source": K3_PARAMETER_SOURCE,
            "status": "zero-weight crosscheck",
        },
        "limitations": [
            "The upstream ledger begins 2024-09-21, so earlier releases are explicitly excluded rather than assigned later prices.",
            "OpenRouter model-level price is a market/provider routing price, not a direct inference-cost measurement.",
            "Thirty-day onboarding eligibility limits identity drift but cannot prove an unversioned alias was never repointed.",
            "Hosted availability and active-parameter disclosure remain non-random.",
            "Historical throughput is unavailable from this source; current tokens/second remains a separate zero-weight operational audit.",
        ],
        "source_files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                RAW,
                CHANGE_POINTS,
                COLLECTION_METADATA,
                CALIBRATION,
                CURRENT_MODELS,
                active_audit.AA_PANEL,
                active_audit.AA_EPOCH_CROSSCHECK,
                active_audit.HF_CONFIG_SIGNALS,
                K3_EVIDENCE_PATH,
            )
        },
        "outputs": {
            "match_audit": str(MATCH_AUDIT.relative_to(ROOT)),
            "predictions": str(PREDICTIONS.relative_to(ROOT)),
            "frontier_targets": str(TARGETS.relative_to(ROOT)),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "inventory": result["inventory"],
                "decision": result["decision"],
                "headline_crosscheck": result["headline_crosscheck"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
