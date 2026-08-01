#!/usr/bin/env python3
"""Audit the Epoch employee calibration feedback and lean architecture alternatives.

The externally supplied sheet applies the live K3-anchored AA formula and the
live 60/40 ECI formula to eight known-size open-weight checkpoints, then takes
their geometric mean.  This script reproduces every displayed value from the
canonical local sources, quantifies the residual structure, and tests simpler
architecture-aware alternatives without treating the hand-selected sheet as an
independent random holdout.

Candidate promotion remains strict: train rows must precede the test date, the
entire test family is removed, comparisons use identical rows, and uncertainty
is clustered by family.  The disclosed Grok 4.5 total is an external frontier
scale veto.  A candidate that improves broad held-out error but loses the
frontier anchor or extrapolates unstably cannot change the live forecast.
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

from open_model_parameter_truth import LEDGER_PATH as PARAMETER_TRUTH_PATH
from open_model_parameter_truth import resolve_parameter_truth


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
FEEDBACK = ROOT / "sources/epoch_employee_calibration_feedback_2026-07-21.csv"
FEEDBACK_EXACT = ROOT / "sources/epoch_employee_calibration_feedback_exact_inputs_2026-07-21.csv"
EPOCH = ROOT / "sources/epoch_all_ai_models_2026-07-31.csv"
AA_PANEL = OUT / "aa_detailed_parameter_panel_2026-07-18.csv"
REGRESSION = ROOT / "regression_results.json"
BACKTEST = OUT / "frontier_parameter_chronological_backtest_2026-07-17.json"
ECI_MULTIVARIATE = OUT / "eci_multivariate_component_audit_2026-07-18.json"

RESULT = OUT / "epoch_feedback_lean_architecture_audit_2026-07-21.json"
CRITIQUE_PANEL = OUT / "epoch_feedback_critique_panel_2026-07-21.csv"
PREDICTIONS = OUT / "lean_architecture_predictions_2026-07-21.csv"
TARGETS = OUT / "lean_architecture_target_sensitivity_2026-07-21.csv"

DATE_ORIGIN = date(2023, 1, 1)
K3_DATE = "2026-07-16"
K3_AA = 57.1123394372091
# The employee sheet rounded K3 to 2.8T.  This constant exists only to
# reproduce that dated spreadsheet; the live model uses the primary 2.78T fact.
FEEDBACK_SHEET_K3_TOTAL_B = 2800.0
AA_SCORE_SLOPE = 0.04520595908212863
AA_DATE_SLOPE = -0.35092795129752774
ECI_DATED_INTERCEPT = 3.4585141554296785
ECI_DATED_SCORE = 0.06429752658522685
ECI_DATED_DATE = -0.35092795129752774
ECI_NODATE_INTERCEPT = 5.704969833163
ECI_NODATE_SCORE = 0.041808992552

MIN_TRAIN_ROWS = 20
MIN_TRAIN_FAMILIES = 6
MIN_MOE_RATIO_ROWS = 5
MIN_MOE_RATIO_FAMILIES = 3
BOOTSTRAPS = 20_000
SEED = 20_260_721

FEEDBACK_TO_AA = {
    "Kimi K2.6": "Kimi K2.6",
    "Kimi K2.7 Code": "Kimi K2.7 Code",
    "Kimi K2 Thinking": "Kimi K2 Thinking",
    "DeepSeek-V3.1": "DeepSeek V3.1 (Non-reasoning)",
    "Llama 4 Maverick": "Llama 4 Maverick",
    "Llama 3.1-405B": "Llama 3.1 Instruct 405B",
    "Mistral Small 3.1": "Mistral Small 3.1",
    "Phi-4": "Phi-4",
}

FEEDBACK_TO_ECI = {
    "Kimi K2.6": "Kimi K2.6",
    "Kimi K2.7 Code": "Kimi K2.7 Code",
    "Kimi K2 Thinking": "Kimi K2 Thinking",
    "DeepSeek-V3.1": "DeepSeek-V3.1",
    "Llama 4 Maverick": "Llama 4 Maverick",
    "Llama 3.1-405B": "Llama 3.1-405B",
    "Mistral Small 3.1": "Mistral Small 3.1",
    "Phi-4": "Phi-4",
}

FEEDBACK_TO_EPOCH = {
    "Kimi K2.6": "Kimi K2.6",
    "Kimi K2.7 Code": "Kimi K2.7 Code",
    "Kimi K2 Thinking": "Kimi K2 Thinking",
    "DeepSeek-V3.1": "DeepSeek-V3.1",
    "Llama 4 Maverick": "Llama 4 Maverick",
    "Llama 3.1-405B": "Llama 3.1-405B",
    "Mistral Small 3.1": "Mistral Small 3.1",
    "Phi-4": "Phi-4",
}

ENSEMBLE_NORMALIZED = {
    "Llama 3.1-405B": "llama31405b",
    "Llama 4 Maverick": "llama4maverick",
    "Phi-4": "phi4",
    "Mistral Small 3.1": "mistralsmall31",
    "DeepSeek-V3.1": "deepseekv31",
    "Kimi K2 Thinking": "kimik2thinking",
    "Kimi K2.6": "kimik26",
    "Kimi K2.7 Code": "kimik27code",
}

TARGET_SPECS = (
    {
        "model": "Claude Fable 5",
        "release_date": "2026-06-09",
        "eci_score": 160.72794930035383,
        "family": "anthropic",
        "moe": 1,
        "reasoning": 1,
    },
    {
        "model": "GPT-5.6 Sol",
        "release_date": "2026-07-09",
        "eci_score": 161.7210971187139,
        "family": "openai",
        "moe": 1,
        "reasoning": 1,
    },
    {
        "model": "Grok 4.5",
        "release_date": "2026-07-08",
        "eci_score": 153.60935822710232,
        "family": "xai",
        "moe": 1,
        "reasoning": 1,
        "disclosed_total_b": 1500.0,
    },
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty audit output: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def years(value: str) -> float:
    return (date.fromisoformat(value) - DATE_ORIGIN).days / 365.25


def normalize_name(value: str) -> str:
    text = value.lower().replace("non-reasoning", "").replace("reasoning", "")
    text = re.sub(r"\b(instruct|thinking|preview|latest|max|high|medium|low)\b", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def is_moe(value: Any) -> int:
    return int("moe" in str(value or "").lower())


def aa_live_prediction_b(score: float, release_date: str) -> float:
    return FEEDBACK_SHEET_K3_TOTAL_B * 10 ** (
        AA_SCORE_SLOPE * (score - K3_AA)
        + AA_DATE_SLOPE
        * ((date.fromisoformat(release_date) - date.fromisoformat(K3_DATE)).days / 365.25)
    )


def eci_live_prediction_b(score: float, release_date: str) -> float:
    dated = 10 ** (
        ECI_DATED_INTERCEPT
        + ECI_DATED_SCORE * score
        + ECI_DATED_DATE * years(release_date)
    ) / 1e9
    no_date = 10 ** (ECI_NODATE_INTERCEPT + ECI_NODATE_SCORE * score) / 1e9
    return float(no_date**0.60 * dated**0.40)


def metric_summary(errors: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(errors), dtype=float)
    if not len(values):
        return {"n": 0}
    absolute = np.abs(values)
    return {
        "n": int(len(values)),
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "geomean_multiplicative_error": float(10 ** np.mean(absolute)),
        "mean_absolute_log10_error": float(np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(values**2))),
        "p80_multiplicative_error": float(10 ** np.quantile(absolute, 0.8)),
        "within_2x": float(np.mean(absolute <= math.log10(2.0))),
        "signed_bias_factor": float(10 ** np.mean(values)),
    }


def critique_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_architecture = {}
    for architecture in ("MoE", "Dense"):
        selected = [row for row in rows if row["architecture"] == architecture]
        by_architecture[architecture] = metric_summary(
            row["benchmark_log10_error"] for row in selected
        )

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["base_cluster"]].append(abs(row["benchmark_log10_error"]))
    base_errors = [float(np.mean(values)) for values in grouped.values()]
    return {
        "all_rows": metric_summary(row["benchmark_log10_error"] for row in rows),
        "by_architecture": by_architecture,
        "independent_base_clusters": len(grouped),
        "equal_base_mean_absolute_log10_error": float(np.mean(base_errors)),
        "equal_base_geomean_multiplicative_error": float(10 ** np.mean(base_errors)),
        "selection_caveat": (
            "The eight rows were selected after the live formulas were known; three Kimi rows "
            "share one lineage. Treat this as a structural model check, not an iid holdout."
        ),
    }


def family_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = Counter(row["family"] for row in rows)
    values = np.asarray(
        [
            (0.5 if int(row.get("estimated", 0) or 0) else 1.0)
            / counts[row["family"]]
            for row in rows
        ],
        dtype=float,
    )
    return values / values.mean()


def feature_value(row: dict[str, Any], feature: str) -> float:
    if feature == "date":
        return years(row["release_date"])
    return float(row.get(feature, 0.0) or 0.0)


def design(rows: list[dict[str, Any]], features: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[1.0, *[feature_value(row, feature) for feature in features]] for row in rows],
        dtype=float,
    )


def fit_predict_log10(
    train: list[dict[str, Any]],
    test: dict[str, Any],
    features: tuple[str, ...],
    target: str,
) -> tuple[float, list[float]]:
    matrix = design(train, features)
    values = np.log10(np.asarray([row[target] for row in train], dtype=float))
    weights = family_weights(train)
    sqrt_weight = np.sqrt(weights)
    beta, *_ = np.linalg.lstsq(
        matrix * sqrt_weight[:, None], values * sqrt_weight, rcond=None
    )
    prediction = float((design([test], features) @ beta).item())
    return prediction, [float(value) for value in beta]


def moe_ratio_prediction_log10(
    train: list[dict[str, Any]], test: dict[str, Any]
) -> tuple[float | None, list[float]]:
    if not test["moe"]:
        return 0.0, []
    moe_train = [row for row in train if row["moe"]]
    if (
        len(moe_train) < MIN_MOE_RATIO_ROWS
        or len({row["family"] for row in moe_train}) < MIN_MOE_RATIO_FAMILIES
    ):
        return None, []
    return fit_predict_log10(moe_train, test, ("date",), "sparsity_ratio")


def chronological_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda row: (row["release_date"], row["model"]))
    for test in ordered:
        prior = [row for row in ordered if row["release_date"] < test["release_date"]]
        if not prior:
            continue
        frontier_rank = sum(row["score"] <= test["score"] for row in prior) / len(prior)
        train = [row for row in prior if row["family"] != test["family"]]
        if len(train) < MIN_TRAIN_ROWS or len({row["family"] for row in train}) < MIN_TRAIN_FAMILIES:
            continue

        score_only, _ = fit_predict_log10(train, test, ("score",), "total_b")
        score_date, _ = fit_predict_log10(train, test, ("score", "date"), "total_b")
        direct_arch, direct_arch_beta = fit_predict_log10(
            train, test, ("score", "date", "moe"), "total_b"
        )
        direct_arch_reasoning, direct_arch_reasoning_beta = fit_predict_log10(
            train, test, ("score", "date", "moe", "reasoning"), "total_b"
        )
        active_log, active_beta = fit_predict_log10(
            train, test, ("score", "date"), "active_b"
        )
        ratio_log, ratio_beta = moe_ratio_prediction_log10(train, test)
        active_transport = None if ratio_log is None else active_log + ratio_log
        live_blend = 0.60 * score_only + 0.40 * score_date
        actual = math.log10(float(test["total_b"]))

        specifications = {
            "live_eci_60_40": (live_blend, []),
            "lean_score_only": (score_only, []),
            "architecture_score_date_moe": (direct_arch, direct_arch_beta),
            "architecture_score_date_moe_reasoning": (
                direct_arch_reasoning,
                direct_arch_reasoning_beta,
            ),
            "active_then_date_sparsity": (
                active_transport,
                [*active_beta, *ratio_beta],
            ),
        }
        for specification, (prediction, coefficients) in specifications.items():
            if prediction is None:
                continue
            error = prediction - actual
            output.append(
                {
                    "specification": specification,
                    "model": test["model"],
                    "release_date": test["release_date"],
                    "family": test["family"],
                    "architecture": "MoE" if test["moe"] else "Dense",
                    "reasoning": test["reasoning"],
                    "actual_b": test["total_b"],
                    "predicted_b": float(10**prediction),
                    "log10_error": float(error),
                    "multiplicative_error": float(10 ** abs(error)),
                    "train_n": len(train),
                    "train_families": len({row["family"] for row in train}),
                    "train_max_date": max(row["release_date"] for row in train),
                    "test_family_excluded": all(
                        row["family"] != test["family"] for row in train
                    ),
                    "frontier_signal_rank": float(frontier_rank),
                    "coefficients_json": json.dumps(coefficients),
                }
            )
    return output


def paired_family_bootstrap(
    rows: list[dict[str, Any]], candidate: str, baseline: str, seed: int
) -> dict[str, Any]:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(row["release_date"], row["model"])][row["specification"]] = row
    paired = [
        (items[candidate], items[baseline])
        for items in by_key.values()
        if candidate in items and baseline in items
    ]
    grouped: dict[str, list[float]] = defaultdict(list)
    for candidate_row, baseline_row in paired:
        grouped[candidate_row["family"]].append(
            abs(candidate_row["log10_error"]) - abs(baseline_row["log10_error"])
        )
    effects = np.asarray(
        [float(np.mean(values)) for _, values in sorted(grouped.items())], dtype=float
    )
    if len(effects) < 2:
        return {
            "paired_rows": len(paired),
            "families": len(effects),
            "observed_delta": float(np.mean(effects)) if len(effects) else None,
            "ci_90": [None, None],
            "probability_candidate_better": None,
            "samples": 0,
        }
    rng = np.random.default_rng(seed)
    draws = effects[
        rng.integers(0, len(effects), size=(BOOTSTRAPS, len(effects)))
    ].mean(axis=1)
    return {
        "metric": "equal-family mean absolute log10 error; candidate minus live ECI",
        "paired_rows": len(paired),
        "families": len(effects),
        "observed_delta": float(np.mean(effects)),
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "probability_candidate_better": float(np.mean(draws < 0)),
        "samples": BOOTSTRAPS,
        "random_seed": seed,
    }


def model_comparison(
    predictions: list[dict[str, Any]], specification: str
) -> dict[str, Any]:
    selected = [row for row in predictions if row["specification"] == specification]
    scopes = {
        "all": selected,
        "moe": [row for row in selected if row["architecture"] == "MoE"],
        "dense": [row for row in selected if row["architecture"] == "Dense"],
        "frontier_like": [row for row in selected if row["frontier_signal_rank"] >= 0.90],
    }
    return {
        scope: metric_summary(row["log10_error"] for row in rows)
        for scope, rows in scopes.items()
    }


def fit_target(
    rows: list[dict[str, Any]], target: dict[str, Any], specification: str
) -> dict[str, Any]:
    test = {
        "model": target["model"],
        "release_date": target["release_date"],
        "score": target["eci_score"],
        "family": target["family"],
        "moe": target["moe"],
        "reasoning": target["reasoning"],
    }
    train = [
        row
        for row in rows
        if row["release_date"] < test["release_date"]
        and row["family"] != test["family"]
    ]
    if specification == "architecture_score_date_moe":
        prediction, coefficients = fit_predict_log10(
            train, test, ("score", "date", "moe"), "total_b"
        )
    elif specification == "architecture_score_date_moe_reasoning":
        prediction, coefficients = fit_predict_log10(
            train, test, ("score", "date", "moe", "reasoning"), "total_b"
        )
    elif specification == "active_then_date_sparsity":
        active_log, active_beta = fit_predict_log10(
            train, test, ("score", "date"), "active_b"
        )
        ratio_log, ratio_beta = moe_ratio_prediction_log10(train, test)
        if ratio_log is None:
            raise ValueError(f"No MoE ratio training support for {target['model']}")
        prediction = active_log + ratio_log
        coefficients = [*active_beta, *ratio_beta]
    else:
        raise ValueError(specification)
    return {
        "model": target["model"],
        "release_date": target["release_date"],
        "specification": specification,
        "predicted_t": float(10**prediction / 1000),
        "train_n": len(train),
        "train_families": len({row["family"] for row in train}),
        "train_max_date": max(row["release_date"] for row in train),
        "target_family_excluded": not any(
            row["family"] == target["family"] for row in train
        ),
        "coefficients_json": json.dumps(coefficients),
        "disclosed_total_t": (
            float(target["disclosed_total_b"] / 1000)
            if target.get("disclosed_total_b")
            else None
        ),
    }


def main() -> None:
    feedback = read_csv(FEEDBACK)
    feedback_exact = read_csv(FEEDBACK_EXACT)
    epoch = read_csv(EPOCH)
    aa_panel = read_csv(AA_PANEL)
    regression = json.loads(REGRESSION.read_text(encoding="utf-8"))
    backtest = json.loads(BACKTEST.read_text(encoding="utf-8"))
    eci_multivariate = json.loads(ECI_MULTIVARIATE.read_text(encoding="utf-8"))

    if len(feedback) != 8 or len({row["feedback_row_id"] for row in feedback}) != 8:
        raise ValueError("Expected eight unique Epoch feedback rows")
    if len(feedback_exact) != 8 or len({row["feedback_row_id"] for row in feedback_exact}) != 8:
        raise ValueError("Expected eight unique frozen exact-input rows")
    exact_by_id = {row["feedback_row_id"]: row for row in feedback_exact}

    aa_by_name = {row["selected_name"]: row for row in aa_panel}
    eci_by_name = {row["model"]: row for row in regression["eci"]["open_models"]}
    epoch_by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in epoch:
        epoch_by_name[row["Model"]].append(row)
    ensemble_by_name = {
        row["normalized_model"]: row for row in backtest["ensemble_predictions"]
    }

    critique_rows: list[dict[str, Any]] = []
    for source in feedback:
        model = source["model"]
        frozen = exact_by_id.get(source["feedback_row_id"])
        if frozen is None or frozen["model"] != model:
            raise ValueError(f"Frozen feedback identity failed: {model}")
        aa = aa_by_name.get(FEEDBACK_TO_AA[model])
        eci = eci_by_name.get(FEEDBACK_TO_ECI[model])
        epoch_rows = epoch_by_name.get(FEEDBACK_TO_EPOCH[model], [])
        if aa is None or len(epoch_rows) != 1:
            raise ValueError(
                f"Feedback identity failed: {model}; aa={aa is not None}; "
                f"eci={eci is not None}; epoch_rows={len(epoch_rows)}"
            )
        epoch_row = epoch_rows[0]
        current_aa = float(aa["intelligence_index"])
        current_eci = None if eci is None else float(eci["score"])
        frozen_aa = float(frozen["aa_score_exact"])
        frozen_eci = float(frozen["eci_score_exact"])
        sheet_aa = float(source["aa_score"])
        sheet_eci = float(source["eci_score"])
        release_date = str(frozen["release_date"])
        current_release_date = (
            str(eci["release_date"])
            if eci is not None
            else str(epoch_row["Publication date"])
        )
        if current_release_date != release_date:
            raise ValueError(
                f"Current release date drift for {model}: {current_release_date} != {release_date}"
            )
        sheet_total_b = float(source["epoch_total_b"])
        canonical_total_b = float(eci["total_b"]) if eci is not None else sheet_total_b
        truth = resolve_parameter_truth(model)
        parameter_truth_id = "" if eci is None else (eci.get("parameter_truth_id") or "")
        if eci is not None:
            if math.isclose(canonical_total_b, sheet_total_b, rel_tol=0, abs_tol=1e-9):
                pass
            elif (
                truth is None
                or parameter_truth_id != truth["truth_id"]
                or not any(
                    math.isclose(sheet_total_b, float(value), rel_tol=0, abs_tol=1e-9)
                    for value in truth["accepted_raw_total_parameters_b"]
                )
                or not math.isclose(
                    canonical_total_b,
                    float(truth["canonical_total_parameters_b"]),
                    rel_tol=0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(f"Parameter feedback mismatch for {model}")
        if epoch_row["Parameters"] and not math.isclose(
            float(epoch_row["Parameters"]) / 1e9, sheet_total_b, rel_tol=0.02
        ):
            raise ValueError(f"Current Epoch parameter mismatch for {model}")
        if epoch_row["Confidence"] != source["epoch_confidence"]:
            raise ValueError(
                f"Epoch confidence mismatch for {model}: {epoch_row['Confidence']}"
            )
        architecture = (
            "MoE" if eci is not None and is_moe(eci["architecture"]) else "Dense"
        ) if eci is not None else source["architecture"]
        if eci is not None and architecture != source["architecture"]:
            raise ValueError(f"Architecture mismatch for {model}: {architecture}")

        # Reproduce the dated employee sheet with its frozen exact inputs.  The
        # visible sheet scores are rounded to one decimal, so substituting them
        # here would falsely make a correct historical formula appear to drift.
        # Current AA/ECI values are retained alongside them as a freshness audit.
        aa_b = aa_live_prediction_b(frozen_aa, release_date)
        eci_b = eci_live_prediction_b(frozen_eci, release_date)
        benchmark_b = math.sqrt(aa_b * eci_b)
        if abs(round(aa_b) - float(source["sheet_p_aa_b"])) > 1:
            raise ValueError(f"Failed to reproduce sheet AA result for {model}: {aa_b}")
        if abs(round(eci_b) - float(source["sheet_p_eci_b"])) > 1:
            raise ValueError(f"Failed to reproduce sheet ECI result for {model}: {eci_b}")
        if abs(round(benchmark_b) - float(source["sheet_model_estimate_b"])) > 1:
            raise ValueError(
                f"Failed to reproduce sheet benchmark result for {model}: {benchmark_b}"
            )
        sheet_ratio = benchmark_b / sheet_total_b
        canonical_ratio = benchmark_b / canonical_total_b
        if abs(round(sheet_ratio, 2) - float(source["sheet_ratio"])) > 0.01:
            raise ValueError(f"Failed to reproduce sheet ratio for {model}: {sheet_ratio}")
        for field, reproduced in (
            ("reproduced_p_aa_b", aa_b),
            ("reproduced_p_eci_b", eci_b),
            ("reproduced_model_estimate_b", benchmark_b),
            ("reproduced_ratio", sheet_ratio),
        ):
            if not math.isclose(reproduced, float(frozen[field]), rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"Frozen exact-input reproduction drift for {model}: {field}")
        strict = ensemble_by_name.get(ENSEMBLE_NORMALIZED[model])
        log_error = math.log10(canonical_ratio)
        critique_rows.append(
            {
                "feedback_row_id": source["feedback_row_id"],
                "model": model,
                "release_date": release_date,
                "epoch_confidence": source["epoch_confidence"],
                "architecture": architecture,
                "base_cluster": source["base_cluster"],
                "aa_score_sheet": sheet_aa,
                "aa_score_frozen_exact": frozen_aa,
                "aa_score_current_exact": current_aa,
                "aa_score_current_minus_frozen": current_aa - frozen_aa,
                "aa_score_current_minus_sheet": current_aa - sheet_aa,
                "eci_score_sheet": sheet_eci,
                "eci_score_frozen_exact": frozen_eci,
                "eci_score_current_exact": current_eci,
                "eci_score_current_minus_frozen": None if current_eci is None else current_eci - frozen_eci,
                "eci_score_current_minus_sheet": None if current_eci is None else current_eci - sheet_eci,
                "eci_current_aggregate_available": eci is not None,
                "epoch_total_b": sheet_total_b,
                "canonical_total_b": canonical_total_b,
                "parameter_truth_id": parameter_truth_id,
                "sheet_p_aa_b": float(source["sheet_p_aa_b"]),
                "reproduced_p_aa_b": aa_b,
                "sheet_p_eci_b": float(source["sheet_p_eci_b"]),
                "reproduced_p_eci_b": eci_b,
                "sheet_model_estimate_b": float(source["sheet_model_estimate_b"]),
                "reproduced_model_estimate_b": benchmark_b,
                "sheet_ratio": float(source["sheet_ratio"]),
                "reproduced_ratio": sheet_ratio,
                "canonical_ratio": canonical_ratio,
                "benchmark_log10_error": log_error,
                "benchmark_multiplicative_error": 10 ** abs(log_error),
                "strict_heldout_ensemble_b": strict["predicted_b"] if strict else None,
                "strict_heldout_multiplicative_error": (
                    strict["multiplicative_error"] if strict else None
                ),
                "source_url": source["source_url"],
                "notes": source["notes"],
            }
        )

    eci_rows: list[dict[str, Any]] = []
    for row in regression["eci"]["open_models"]:
        eci_rows.append(
            {
                "model": row["model"],
                "release_date": row["release_date"],
                "score": float(row["score"]),
                "total_b": float(row["total_b"]),
                "active_b": float(row["active_b"]),
                "sparsity_ratio": float(row["total_b"] / row["active_b"]),
                "family": row["family"],
                "moe": is_moe(row["architecture"]),
                "reasoning": int(row.get("reasoning", 0) or 0),
                "estimated": int(row.get("estimated", 0) or 0),
            }
        )
    if len(eci_rows) != 89 or len({row["family"] for row in eci_rows}) != 40:
        raise ValueError("Unexpected ECI architecture panel inventory")

    predictions = chronological_predictions(eci_rows)
    specifications = sorted({row["specification"] for row in predictions})
    comparison = {
        specification: model_comparison(predictions, specification)
        for specification in specifications
    }
    paired = {
        specification: paired_family_bootstrap(
            predictions,
            specification,
            "live_eci_60_40",
            SEED + index,
        )
        for index, specification in enumerate(specifications)
        if specification != "live_eci_60_40"
    }
    paired_moe = {}
    moe_predictions = [row for row in predictions if row["architecture"] == "MoE"]
    for index, specification in enumerate(specifications):
        if specification == "live_eci_60_40":
            continue
        paired_moe[specification] = paired_family_bootstrap(
            moe_predictions,
            specification,
            "live_eci_60_40",
            SEED + 100 + index,
        )

    target_rows = [
        fit_target(eci_rows, target, specification)
        for target in TARGET_SPECS
        for specification in (
            "architecture_score_date_moe",
            "architecture_score_date_moe_reasoning",
            "active_then_date_sparsity",
        )
    ]
    for row in target_rows:
        # Recompute the live aggregate-ECI branch from its upstream score/date
        # contract.  This audit must not read the final site: the workbook hashes
        # this audit and the site is generated from that workbook, so doing so
        # creates an irreducible self-reference and permits stale hashes.
        row["current_live_eci_t"] = eci_live_prediction_b(
            float(next(target["eci_score"] for target in TARGET_SPECS if target["model"] == row["model"])),
            row["release_date"],
        ) / 1000
        if row["disclosed_total_t"]:
            row["candidate_anchor_error_x"] = max(
                row["predicted_t"] / row["disclosed_total_t"],
                row["disclosed_total_t"] / row["predicted_t"],
            )
            row["current_eci_anchor_error_x"] = max(
                row["current_live_eci_t"] / row["disclosed_total_t"],
                row["disclosed_total_t"] / row["current_live_eci_t"],
            )

    grok_rows = {row["specification"]: row for row in target_rows if row["model"] == "Grok 4.5"}
    architecture_gate = paired["architecture_score_date_moe"]
    architecture_moe_gate = paired_moe["architecture_score_date_moe"]
    architecture_grok = grok_rows["architecture_score_date_moe"]
    active_gate = paired["active_then_date_sparsity"]
    active_moe_gate = paired_moe["active_then_date_sparsity"]
    active_grok = grok_rows["active_then_date_sparsity"]
    promotion = {
        "minimum_paired_rows": 60,
        "minimum_paired_families": 25,
        "minimum_moe_paired_rows": 20,
        "minimum_moe_paired_families": 8,
        "maximum_ci90_upper": 0.0,
        "maximum_grok_error_relative_to_current": 1.10,
        "architecture_candidate": {
            "coverage_pass": architecture_gate["paired_rows"] >= 60
            and architecture_gate["families"] >= 25,
            "all_family_interval_pass": architecture_gate["ci_90"][1] < 0,
            "moe_coverage_pass": architecture_moe_gate["paired_rows"] >= 20
            and architecture_moe_gate["families"] >= 8,
            "moe_family_interval_pass": architecture_moe_gate["ci_90"][1] < 0,
            "grok_anchor_pass": architecture_grok["candidate_anchor_error_x"]
            <= 1.10 * architecture_grok["current_eci_anchor_error_x"],
        },
        "active_sparsity_candidate": {
            "coverage_pass": active_gate["paired_rows"] >= 60
            and active_gate["families"] >= 25,
            "all_family_interval_pass": active_gate["ci_90"][1] < 0,
            "moe_coverage_pass": active_moe_gate["paired_rows"] >= 20
            and active_moe_gate["families"] >= 8,
            "moe_family_interval_pass": active_moe_gate["ci_90"][1] < 0,
            "grok_anchor_pass": active_grok["candidate_anchor_error_x"]
            <= 1.10 * active_grok["current_eci_anchor_error_x"],
        },
    }
    for candidate in promotion.values():
        if isinstance(candidate, dict):
            candidate["all_gates_pass"] = all(candidate.values())

    critique = critique_metrics(critique_rows)
    eci_component_summary = {
        "all_total_baseline_median_error_x": eci_multivariate["backtest"]["total"]["all"]["baseline"]["median_multiplicative_error"],
        "all_total_candidate_median_error_x": eci_multivariate["backtest"]["total"]["all"]["candidate"]["median_multiplicative_error"],
        "all_total_family_ci90": eci_multivariate["backtest"]["total"]["all"]["paired_family_bootstrap"]["ci_90"],
        "frontier_total_baseline_median_error_x": eci_multivariate["backtest"]["total"]["frontier_like"]["baseline"]["median_multiplicative_error"],
        "frontier_total_candidate_median_error_x": eci_multivariate["backtest"]["total"]["frontier_like"]["candidate"]["median_multiplicative_error"],
        "frontier_total_family_ci90": eci_multivariate["backtest"]["total"]["frontier_like"]["paired_family_bootstrap"]["ci_90"],
        "selected_target_feature_sets": {
            row["model"]: row["full_panel_policy"]["feature_set"]
            for row in eci_multivariate["target_fit"]["target_models"]
        },
        "promotion_decision": eci_multivariate["decision"],
    }

    change_live = any(
        candidate["all_gates_pass"]
        for candidate in promotion.values()
        if isinstance(candidate, dict)
    )
    result = {
        "generated_on": "2026-07-31",
        "question": (
            "Does the Epoch employee's eight-row calibration critique justify replacing the "
            "live aggregate benchmark branch with a lean architecture-aware model?"
        ),
        "feedback_reproduction": {
            "rows": len(critique_rows),
            "historical_sheet_scores_preserved": True,
            "current_aa_scores_within_0_06": all(
                abs(row["aa_score_current_minus_sheet"]) <= 0.06
                for row in critique_rows
            ),
            "current_eci_aggregate_coverage": sum(
                bool(row["eci_current_aggregate_available"])
                for row in critique_rows
            ),
            "current_eci_scores_within_0_06_where_available": all(
                row["eci_score_current_minus_sheet"] is None
                or abs(row["eci_score_current_minus_sheet"]) <= 0.06
                for row in critique_rows
            ),
            "all_parameter_labels_match_current_epoch_or_eci": True,
            "all_displayed_formula_outputs_reproduced": True,
            "metrics": critique,
        },
        "architecture_panel": {
            "rows": len(eci_rows),
            "families": len({row["family"] for row in eci_rows}),
            "moe_rows": sum(row["moe"] for row in eci_rows),
            "dense_rows": sum(not row["moe"] for row in eci_rows),
            "chronological_predictions": len(
                [row for row in predictions if row["specification"] == "live_eci_60_40"]
            ),
        },
        "heldout_metrics": comparison,
        "paired_family_bootstraps_vs_live_eci": paired,
        "paired_moe_family_bootstraps_vs_live_eci": paired_moe,
        "target_sensitivity": target_rows,
        "promotion_gates": promotion,
        "less_is_more_component_evidence": eci_component_summary,
        "presentation_layer_separation": {
            "final_site_consumed": False,
            "reason": (
                "This upstream architecture audit intentionally excludes the final site/workbook. "
                "Price-removal display sensitivity belongs to a downstream presentation audit and "
                "cannot be an input to a workbook that hashes this result."
            ),
        },
        "decision": {
            "change_live_model": change_live,
            "promote_architecture_candidate": promotion["architecture_candidate"]["all_gates_pass"],
            "promote_active_sparsity_candidate": promotion["active_sparsity_candidate"]["all_gates_pass"],
            "incremental_live_weight": 0.0,
            "reason": (
                "The critique correctly identifies architecture- and efficiency-structured residuals, "
                "and MoE-aware candidates improve several point metrics. Neither candidate passes the "
                "complete coverage, family-bootstrap, MoE-subset, and disclosed-Grok frontier-anchor gates. "
                "The knowledge-only component branch remains promising but lacks stable target coverage."
            ),
            "next_model": (
                "Prioritize a preregistered active-parameter plus sparsity model and a small knowledge-only "
                "benchmark set; collect prospective target coverage before changing central forecasts."
            ),
        },
        "limitations": [
            "The employee sheet is a hand-selected structural check, not a random or prospective holdout.",
            "Three of five MoE rows share the Kimi K2 lineage and must not be counted independently.",
            "Dense rows include both a very large inefficient model and small efficient models; a binary MoE flag is insufficient.",
            "Target MoE/reasoning flags are working assumptions rather than public architecture disclosures.",
            "Current benchmark snapshots are not release-vintage historical measurements.",
        ],
        "source_files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                FEEDBACK,
                FEEDBACK_EXACT,
                EPOCH,
                AA_PANEL,
                REGRESSION,
                BACKTEST,
                ECI_MULTIVARIATE,
                PARAMETER_TRUTH_PATH,
            )
        },
        "outputs": {
            "critique_panel": str(CRITIQUE_PANEL.relative_to(ROOT)),
            "predictions": str(PREDICTIONS.relative_to(ROOT)),
            "target_sensitivity": str(TARGETS.relative_to(ROOT)),
        },
    }

    write_csv(CRITIQUE_PANEL, critique_rows)
    write_csv(PREDICTIONS, predictions)
    write_csv(TARGETS, target_rows)
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "feedback_rows": len(critique_rows),
                "feedback_median_error_x": critique["all_rows"]["median_multiplicative_error"],
                "chronological_predictions": result["architecture_panel"]["chronological_predictions"],
                "change_live_model": change_live,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
