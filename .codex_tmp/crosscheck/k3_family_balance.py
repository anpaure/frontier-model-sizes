import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ROWS = json.loads((ROOT / "regression_results.json").read_text())["open_models"]
DATE_ORIGIN = date(2026, 1, 1)
DATE_COEF = -0.35092795129752774
EXTRAS = ("reasoning", "moe")
K3 = {"release_date": "2026-07-16", "score": 57.1123, "reasoning": 1, "moe": 1}


def x(rows):
    result = []
    for row in rows:
        dy = (date.fromisoformat(row["release_date"]) - DATE_ORIGIN).days / 365.25
        result.append([1.0, row["score"], dy, row["reasoning"], row["moe"]])
    return np.asarray(result, dtype=float)


def fit(rows, family_balance):
    design = x(rows)
    y = np.log10([row["total_b"] * 1e9 for row in rows])
    base = np.asarray([0.5 if row["estimated"] else 1.0 for row in rows])
    if family_balance:
        totals = defaultdict(float)
        for row, weight in zip(rows, base):
            totals[row["family"]] += weight
        weights = np.asarray([weight / totals[row["family"]] for row, weight in zip(rows, base)])
    else:
        weights = base
    reduced = np.delete(design, 2, axis=1)
    target = y - DATE_COEF * design[:, 2]
    sw = np.sqrt(weights)
    reduced_beta, *_ = np.linalg.lstsq(reduced * sw[:, None], target * sw, rcond=None)
    return np.insert(reduced_beta, 2, DATE_COEF)


def predict(beta, row):
    return 10 ** float((x([row]) @ beta).item()) / 1e9


output = {}
for balanced in (False, True):
    label = "family_balanced" if balanced else "row_weighted"
    beta = fit(ROWS, balanced)
    without_kimi = [row for row in ROWS if row["family"] != "kimi_k2"]
    beta_without = fit(without_kimi, balanced)
    output[label] = {
        "beta": beta.tolist(),
        "k3_prediction_b": predict(beta, K3),
        "leave_kimi_out_beta": beta_without.tolist(),
        "leave_kimi_out_k3_prediction_b": predict(beta_without, K3),
    }

print(json.dumps(output, indent=2))
