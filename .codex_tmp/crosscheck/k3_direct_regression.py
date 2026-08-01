import json
import math
from datetime import date
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = json.loads((ROOT / "regression_results.json").read_text())
ROWS = DATA["open_models"]
K3 = {
    "release_date": "2026-07-16",
    "model": "Kimi K3",
    "total_b": 2800.0,
    "score": 57.1123,
    "reasoning": 1,
    "moe": 1,
    "coder": 0,
    "multimodal": 1,
    "estimated": 0,
    "family": "kimi_k3",
}
DATE_ORIGIN = date(2026, 1, 1)
WORKBOOK_DATE_COEF = -0.35092795129752774


def parse_date(value):
    return date.fromisoformat(value)


def date_years(row):
    return (parse_date(row["release_date"]) - DATE_ORIGIN).days / 365.25


def design(rows, extras):
    values = []
    for row in rows:
        x = [1.0, float(row["score"]), date_years(row)]
        x.extend(float(row[name]) for name in extras)
        values.append(x)
    return np.asarray(values, dtype=float)


def wls(rows, extras, fixed_date=None):
    x = design(rows, extras)
    y = np.log10(np.asarray([r["total_b"] * 1e9 for r in rows], dtype=float))
    weights = np.asarray([0.5 if r["estimated"] else 1.0 for r in rows], dtype=float)
    sw = np.sqrt(weights)
    if fixed_date is None:
        beta, *_ = np.linalg.lstsq(x * sw[:, None], y * sw, rcond=None)
    else:
        reduced = np.delete(x, 2, axis=1)
        target = y - fixed_date * x[:, 2]
        reduced_beta, *_ = np.linalg.lstsq(reduced * sw[:, None], target * sw, rcond=None)
        beta = np.insert(reduced_beta, 2, fixed_date)
    pred = x @ beta
    return beta, pred, y, weights


def lofo(rows, extras, fixed_date=None):
    families = sorted({r["family"] for r in rows})
    errors = []
    for family in families:
        train = [r for r in rows if r["family"] != family]
        test = [r for r in rows if r["family"] == family]
        beta, *_ = wls(train, extras, fixed_date=fixed_date)
        errors.extend((design(test, extras) @ beta - np.log10([r["total_b"] * 1e9 for r in test])).tolist())
    errors = np.asarray(errors)
    return {
        "rmse_log10": float(np.sqrt(np.mean(errors ** 2))),
        "mae_log10": float(np.mean(np.abs(errors))),
        "median_abs_log10": float(np.median(np.abs(errors))),
        "p80_abs_log10": float(np.quantile(np.abs(errors), 0.8)),
    }


def summarize(rows, extras, fixed_date=None):
    beta, pred, y, weights = wls(rows, extras, fixed_date=fixed_date)
    k3_log10 = float(design([K3], extras) @ beta)
    k3_pred_b = 10 ** k3_log10 / 1e9
    residual = y - pred
    weighted_rmse = math.sqrt(float(np.sum(weights * residual ** 2) / np.sum(weights)))
    return {
        "extras": list(extras),
        "date_constraint": fixed_date,
        "beta": beta.tolist(),
        "weighted_rmse_log10": weighted_rmse,
        "lofo": lofo(rows, extras, fixed_date=fixed_date),
        "k3_pred_b": k3_pred_b,
        "k3_ratio_pred_to_actual": k3_pred_b / K3["total_b"],
    }


specs = [(), ("reasoning",), ("moe",), ("reasoning", "moe"), ("coder", "multimodal"), ("reasoning", "moe", "coder", "multimodal")]
results = []
for spec in specs:
    results.append(summarize(ROWS, spec, fixed_date=None))
    results.append(summarize(ROWS, spec, fixed_date=WORKBOOK_DATE_COEF))

results.sort(key=lambda item: (item["lofo"]["rmse_log10"], len(item["extras"])))
print(json.dumps(results, indent=2))

selected_extras = ("reasoning", "moe")
selected_beta, *_ = wls(ROWS, selected_extras, fixed_date=WORKBOOK_DATE_COEF)
k3_raw_b = 10 ** float((design([K3], selected_extras) @ selected_beta).item()) / 1e9
k3_anchor_shift = math.log10(K3["total_b"] / k3_raw_b)
frontier_rows = []
for row in DATA["frontier_predictions"]:
    target = {
        "release_date": row["release_date"],
        "score": row["aa_score"],
        "reasoning": 1,
        "moe": 1,
    }
    raw_b = 10 ** float((design([target], selected_extras) @ selected_beta).item()) / 1e9
    frontier_rows.append({
        "model": row["model"],
        "aa_score": row["aa_score"],
        "aa_direct_raw_b": raw_b,
        "aa_direct_k3_calibrated_b": raw_b * 10 ** k3_anchor_shift,
    })
print("K3_CALIBRATED_FRONTIER")
print(json.dumps({
    "selected_beta": selected_beta.tolist(),
    "k3_raw_b": k3_raw_b,
    "k3_anchor_shift_log10": k3_anchor_shift,
    "k3_anchor_factor": 10 ** k3_anchor_shift,
    "frontier": frontier_rows,
}, indent=2))
