from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "factor_weight_optimization_2026-07-18.json"
PREDICTIONS = OUT / "factor_weight_optimization_predictions_2026-07-18.csv"
IKP_AUDIT = OUT / "ikp_parameter_signal_audit_2026-07-18.json"


data = json.loads(RESULT.read_text(encoding="utf-8"))
ikp_audit = json.loads(IKP_AUDIT.read_text(encoding="utf-8"))
with PREDICTIONS.open(newline="", encoding="utf-8") as handle:
    predictions = list(csv.DictReader(handle))

patterns = data["coverage"]["availability_patterns"]
assert data["coverage"]["matched_checkpoints"] == sum(patterns.values())
for factor, count in data["coverage"]["rows_by_factor"].items():
    assert count == sum(
        rows for pattern, rows in patterns.items() if factor in pattern.split("|")
    )
assert data["nested_outer_evaluation"]["eligible_predictions"] == len(predictions)
assert len(predictions) == len(
    {(row["release_date"], row["model"], row["family"]) for row in predictions}
)
assert data["decision"]["update_live_weights"] is False
price = data["separate_api_price_validation"]
assert price["coverage"]["strictly_chronological_paired_models"] == price[
    "current_price_weight_metrics"
]["n"]
assert price["coverage"]["developer_families"] == price[
    "paired_developer_family_bootstrap"
]["developer_families"]
assert price["current_price_weight_metrics"]["mean_absolute_log10_error"] < price["evidence_only_metrics"]["mean_absolute_log10_error"]
assert price["paired_developer_family_bootstrap"]["ci_90"][1] < 0
assert price["decision"]["change_live_weight"] is False
ikp = data["separate_ikp_validation"]
assert ikp["coverage"] == {
    "models": ikp_audit["incremental_overlap"]["models"],
    "families": ikp_audit["incremental_overlap"]["families"],
    "chronological_models": ikp_audit["incremental_overlap"]["chronological_fixed_weight_subset"]["models"],
    "chronological_families": ikp_audit["incremental_overlap"]["chronological_fixed_weight_subset"]["families"],
}
assert ikp["fixed_10pct_blend_metrics"]["mean_absolute_log10_error"] < ikp["existing_metrics"]["mean_absolute_log10_error"]
assert ikp["full_overlap_family_bootstrap"]["ci_90"][1] < 0
assert ikp["chronological_family_bootstrap"]["ci_90"][1] < 0
assert ikp["decision"] == ikp_audit["decision"]
assert ikp["decision"]["promote_incremental_ikp_weight"] is False
assert (
    ikp["coverage"]["chronological_models"]
    < ikp["decision"]["evidence_gates"]["minimum_chronological_subset_models"]
)
assert ikp["decision"]["incremental_evidence_weight"] == 0
assert data["live_final_weights_percent"]["ikp"] == 100 * ikp["decision"]["incremental_final_weight_when_crowd_is_50pct"] == 0
assert data["live_final_weights_percent"] == {
    "aa": 9.5625,
    "eci": 9.5625,
    "price": 3.375,
    "horizon": 25.0,
    "compute": 2.5,
    "ikp": 0.0,
    "crowd": 50.0,
}
assert abs(sum(data["live_final_weights_percent"].values()) - 100) < 1e-12

for row in predictions:
    assert row["meta_train_max_date"] < row["release_date"]
    assert int(row["meta_train_n"]) >= 12
    assert int(row["meta_train_family_n"]) >= 5
    for prefix in ("optimized_mse_weight_", "optimized_mae_weight_"):
        total = sum(float(row[prefix + factor]) for factor in ("AA", "ECI", "No-CoT", "Compute"))
        assert abs(total - 1.0) < 1e-12

metrics = data["nested_outer_evaluation"]["metrics"]
assert metrics["optimized_mse"]["rmse_log10"] > metrics["current_weights"]["rmse_log10"]
assert metrics["optimized_mae"]["median_multiplicative_error"] > metrics["current_weights"]["median_multiplicative_error"]
paired = data["nested_outer_evaluation"]["paired_family_bootstrap"]
assert paired["ci_90"][1] >= 0
assert paired["bootstrap_probability_optimized_better"] < 0.9
print({"nested_predictions": len(predictions), "status": "PASS"})
