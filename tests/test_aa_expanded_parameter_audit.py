from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "aa_expanded_parameter_audit_2026-07-18.json"
PANEL = OUT / "aa_expanded_parameter_panel_2026-07-18.csv"
PREDICTIONS = OUT / "aa_expanded_parameter_predictions_2026-07-18.csv"
OVERLAPS = OUT / "aa_expanded_parameter_overlap_audit_2026-07-18.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


data = json.loads(RESULT.read_text(encoding="utf-8"))
panel = rows(PANEL)
predictions = rows(PREDICTIONS)
overlaps = rows(OVERLAPS)

assert len(panel) == 94
assert len({row["panel_id"] for row in panel}) == 94
assert len({row["model"] for row in panel}) == 94
assert Counter(row["panel_source"] for row in panel) == {
    "exact_epoch_checkpoint": 63,
    "current_manual_only": 31,
}
assert Counter(row["also_in_current_panel"] for row in panel) == {
    "True": 50,
    "False": 44,
}
exact = [row for row in panel if row["panel_source"] == "exact_epoch_checkpoint"]
assert len({row["canonical_checkpoint_id"] for row in exact}) == 63
assert all(row["epoch_accessibility"].startswith("Open weights") for row in exact)
assert all(row["matched_epoch_model"] and row["parameter_source"] for row in exact)
assert Counter(row["parameter_reconciliation"] for row in exact) == {
    "exact_epoch_parameter_match": 62,
    "primary_k3_exact_2780b_supersedes_epoch_rounded_2800b": 1,
}
k3 = next(row for row in exact if row["model"] == "Kimi K3")
assert float(k3["total_parameters_b"]) == 2780.0
assert float(k3["epoch_total_parameters_b"]) == 2800.0
assert k3["parameter_source"] == "Kimi K3 official technical report Table 1"
assert all(
    math.isclose(
        float(row["total_parameters_b"]),
        float(row["epoch_total_parameters_b"]),
        rel_tol=0,
        abs_tol=1e-9,
    )
    for row in exact
    if row["model"] != "Kimi K3"
)

assert len(overlaps) == 19
assert len({row["current_model"] for row in overlaps}) == 19
assert len({row["canonical_checkpoint_id"] for row in overlaps}) == 19
assert Counter(row["match_method"] for row in overlaps) == {
    "normalized exact model label": 18,
    "manual exact checkpoint alias": 1,
}
assert next(
    row for row in overlaps if row["current_model"] == "Nemotron 3 Ultra 550B A55B"
)["exact_model"] == "Nemotron 3 Ultra"

assert len(predictions) == 46
assert len({(row["release_date"], row["model"]) for row in predictions}) == 46
assert all(
    row["current_train_max_date"] < row["prediction_information_date"]
    for row in predictions
)
assert all(
    row["expanded_train_max_date"] < row["prediction_information_date"]
    for row in predictions
)
assert all(row["test_developer_excluded"] == "True" for row in predictions)
assert all(int(row["current_train_n"]) >= 16 for row in predictions)
assert all(int(row["expanded_train_developer_n"]) >= 6 for row in predictions)

audit = data["data_audit"]
assert audit["current_panel_models"] == 50
assert audit["exact_open_epoch_checkpoints"] == 63
assert audit["lower_scoring_duplicate_configurations_discarded"] == 9
assert audit["current_exact_overlaps"] == 19
assert audit["expanded_unique_models"] == 94
assert audit["expanded_developers"] == 27
assert audit["primary_parameter_overrides_of_epoch_rounded_values"] == 1

scopes = data["backtest"]["scopes"]
assert scopes["all"]["n"] == 46
assert scopes["current_panel"]["n"] == 34
assert scopes["exact_additions"]["n"] == 12
assert scopes["frontier_like"]["n"] == 15
assert scopes["all"]["expanded_panel"]["median_multiplicative_error"] < scopes["all"]["current_50"]["median_multiplicative_error"]
assert scopes["all"]["paired_developer_bootstrap"]["ci_90"][1] < 0
assert scopes["current_panel"]["expanded_panel"]["mean_absolute_log10_error"] < scopes["current_panel"]["current_50"]["mean_absolute_log10_error"]
# The refreshed exact AA scores preserve the point improvement on the current
# panel, but its developer-clustered 90% interval now narrowly crosses zero.
# Keep that uncertainty explicit rather than freezing the prior significance.
assert scopes["current_panel"]["paired_developer_bootstrap"]["observed_delta"] < 0
assert scopes["current_panel"]["paired_developer_bootstrap"]["ci_90"][0] < 0
assert scopes["current_panel"]["paired_developer_bootstrap"]["ci_90"][1] > 0
assert scopes["frontier_like"]["paired_developer_bootstrap"]["ci_90"][0] < 0
assert scopes["frontier_like"]["paired_developer_bootstrap"]["ci_90"][1] > 0

fit = data["full_fit"]
assert fit["expanded_panel_developer_balanced_coefficients"]["score_slope"] > fit["current_live_k3_anchored_coefficients"]["score_slope"]
assert fit["expanded_panel_developer_balanced_coefficients"]["date_slope"] < fit["current_live_k3_anchored_coefficients"]["date_slope"]
assert fit["expanded_coefficient_developer_bootstrap"]["samples"] == 5000

stability = data["frontier_aa_stability"]
assert len(stability) == 9
assert {row["model"] for row in stability} >= {"Claude Opus 5", "Claude Fable 5", "GPT-5.6 Sol"}
assert max(abs(math.log(row["expanded_over_current"])) for row in stability) > math.log(1.15)
assert data["decision"]["change_live_aa_branch"] is False
assert data["decision"]["incremental_expanded_aa_weight"] == 0
assert data["k3_anchor"]["total_parameters_b"] == 2780
assert data["k3_anchor"]["source"] == "Kimi K3 official technical report Table 1"

for relative, digest in data["source_manifest"].items():
    path = ROOT / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

print(
    {
        "expanded_models": len(panel),
        "overlaps": len(overlaps),
        "eligible_predictions": len(predictions),
        "frontier_like_predictions": scopes["frontier_like"]["n"],
        "live_branch_changed": False,
        "status": "PASS",
    }
)
