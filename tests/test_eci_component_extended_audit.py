from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "eci_component_extended_audit_2026-07-18.json"
PANEL = OUT / "eci_component_expanded_parameter_panel_2026-07-18.csv"
AGGREGATE = OUT / "eci_component_expanded_aggregate_predictions_2026-07-18.csv"
COMPARISON = OUT / "eci_component_active_incremental_comparison_2026-07-18.csv"
PREDICTIONS = OUT / "eci_component_active_incremental_predictions_2026-07-18.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


data = json.loads(RESULT.read_text(encoding="utf-8"))
panel = rows(PANEL)
aggregate = rows(AGGREGATE)
comparison = rows(COMPARISON)
predictions = rows(PREDICTIONS)

assert len(panel) == 84
assert len({row["canonical_checkpoint_id"] for row in panel}) == 84
assert Counter(row["panel_source"] for row in panel) == {
    "legacy_workbook_parameter_row": 57,
    "exact_epoch_open_extension": 26,
    "primary_exact_open_extension": 1,
}
canonicalized = [row for row in panel if "canonicalized by" in row["parameter_value_source"]]
assert len(canonicalized) == 6
assert {row["model"] for row in canonicalized} == {
    "Kimi K2 Thinking",
    "Kimi K2.5",
    "Kimi K2.6",
    "Kimi K2.7 Code",
    "MiniMax-M2.5",
    "MiniMax-M2.7",
}
for row in panel:
    expected = 15.3664 / float(row["eci_ci_width"]) ** 2
    assert math.isclose(float(row["wls_weight"]), expected, rel_tol=1e-12)
for row in panel:
    if row["panel_source"] == "exact_epoch_open_extension":
        assert row["epoch_match_method"] == "alphanumeric exact"
        assert row["epoch_match_confidence"] == "high"
        assert row["epoch_accessibility"].startswith("Open weights")

assert len(aggregate) == 64
assert all(row["current_train_max_date"] < row["release_date"] for row in aggregate)
assert all(row["expanded_train_max_date"] < row["release_date"] for row in aggregate)
assert all(int(row["current_train_n"]) >= 12 for row in aggregate)
assert all(int(row["expanded_train_family_n"]) >= 5 for row in aggregate)

primary_k3 = next(row for row in panel if row["panel_source"] == "primary_exact_open_extension")
assert primary_k3["model"] == "Kimi K3"
assert math.isclose(float(primary_k3["total_parameters_b"]), 2780.0)

expanded = data["expanded_total_parameter_panel"]
assert expanded["models"] == 84
assert expanded["decision"]["change_live_eci_center"] is False
live_test = expanded["aggregate_backtest"]["legacy_57_tests"]
all_test = expanded["aggregate_backtest"]["all_84_tests"]
assert live_test["n"] == 45
assert live_test["expanded_84"]["mean_absolute_log10_error"] < live_test["legacy_57"]["mean_absolute_log10_error"]
assert live_test["paired_family_bootstrap"]["ci_90"][1] > 0
assert all_test["n"] == 64
assert all_test["expanded_84"]["mean_absolute_log10_error"] < all_test["legacy_57"]["mean_absolute_log10_error"]
assert all_test["paired_family_bootstrap"]["ci_90"][1] > 0
assert max(
    abs(math.log(float(row["expanded_over_legacy"])))
    for row in expanded["frontier_estimate_stability"]
) < math.log(1.04)

active = data["active_parameter_component_audit"]
assert active["parameter_map_models"] == 89
assert active["parameter_map_families"] == 40
assert active["component_rows"] == 723
assert active["component_benchmarks"] == 50
assert active["eligible_comparisons"] == len(comparison) == 13
assert active["supported_after_familywise_correction"] == []
assert active["decision"]["add_component_branch"] is False
assert active["decision"]["incremental_component_weight"] == 0

otis = next(row for row in comparison if row["benchmark"] == "OTIS Mock AIME 2024-2025")
assert float(otis["augmented_active_median_error_x"]) < float(otis["baseline_active_median_error_x"])
assert float(otis["active_delta_ci_90_high"]) < 0
assert float(otis["active_familywise_one_sided_p"]) > 0.05
assert len(predictions) == 291
assert len({(row["benchmark"], row["model"], row["release_date"]) for row in predictions}) == len(predictions)
assert all(row["train_max_date"] < row["release_date"] for row in predictions)
assert all(int(row["train_n"]) >= 12 and int(row["train_family_n"]) >= 5 for row in predictions)
assert sum(int(row["heldout_predictions"]) for row in comparison) == 281
# Nine valid folds are retained for completeness but belong to component panels
# that do not clear the predeclared benchmark-level coverage gate.
assert len(predictions) - sum(int(row["heldout_predictions"]) for row in comparison) == 10

for relative, digest in data["source_manifest"].items():
    path = ROOT / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

print(
    {
        "expanded_models": len(panel),
        "active_predictions": len(predictions),
        "familywise_supported_components": 0,
        "status": "PASS",
    }
)
