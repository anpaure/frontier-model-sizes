from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "eci_component_chronological_backtest_2026-07-18.json"
PREDICTIONS = OUT / "eci_component_backtest_predictions_2026-07-18.csv"
FRONTIER = OUT / "eci_component_frontier_estimates_2026-07-18.csv"
COMPONENT = ROOT / "sources/epoch_eci_benchmarks_2026-07-31.csv"


def rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


result = json.loads(RESULT.read_text(encoding="utf-8"))
inventory = result["inventory"]
assert inventory["component_rows_all"] == 2059
assert inventory["component_models_all"] == 213
assert inventory["component_benchmarks_all"] == 54
assert inventory["open_ground_truth_models"] == 89
assert inventory["matched_component_rows"] == 723
assert inventory["matched_component_models"] == 89
assert inventory["matched_component_benchmarks"] == 50
assert result["materially_better_than_eci_after_family_cluster_bootstrap"] == []
assert result["compute_coverage"]["training_and_post_training_models"] == 1
assert result["aggregate_eci_exact_live_backtest"]["eci_n"] == 77
assert result["aggregate_eci_exact_live_backtest"]["eci_median_factor"] < 2.5
assert result["source_files"][COMPONENT.relative_to(ROOT).as_posix()] == hashlib.sha256(COMPONENT.read_bytes()).hexdigest()

predictions = rows(PREDICTIONS)
assert predictions
assert len({(row["panel"], row["model"], row["release_date"]) for row in predictions}) == len(predictions)
assert all(row["train_max_date"] < row["release_date"] for row in predictions)
assert all(int(row["train_n"]) >= 12 and int(row["train_family_n"]) >= 5 for row in predictions)

for comparison in result["benchmark_comparisons"]:
    assert comparison["component_n"] == comparison["eci_n"]

frontier = rows(FRONTIER)
assert frontier
assert all(row["exploratory_only"] == "True" for row in frontier)
print({"predictions": len(predictions), "panels": len(result["benchmark_comparisons"]), "status": "PASS"})
