from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "epoch_feedback_lean_architecture_audit_2026-07-21.json"
CRITIQUE = OUT / "epoch_feedback_critique_panel_2026-07-21.csv"
PREDICTIONS = OUT / "lean_architecture_predictions_2026-07-21.csv"
TARGETS = OUT / "lean_architecture_target_sensitivity_2026-07-21.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_epoch_feedback_reproduction_and_decision() -> None:
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    critique = read_csv(CRITIQUE)
    predictions = read_csv(PREDICTIONS)
    targets = read_csv(TARGETS)

    reproduction = data["feedback_reproduction"]
    assert reproduction["rows"] == len(critique) == 8
    assert reproduction["historical_sheet_scores_preserved"] is True
    assert reproduction["current_aa_scores_within_0_06"] is True
    assert reproduction["current_eci_aggregate_coverage"] == 7
    assert reproduction["current_eci_scores_within_0_06_where_available"] is False
    assert reproduction["all_parameter_labels_match_current_epoch_or_eci"] is True
    assert reproduction["all_displayed_formula_outputs_reproduced"] is True
    assert reproduction["metrics"]["independent_base_clusters"] == 6
    assert reproduction["metrics"]["by_architecture"]["MoE"]["n"] == 5
    assert reproduction["metrics"]["by_architecture"]["Dense"]["n"] == 3
    assert 3.22 < reproduction["metrics"]["all_rows"]["median_multiplicative_error"] < 3.24
    assert reproduction["metrics"]["by_architecture"]["MoE"]["signed_bias_factor"] < 0.51

    kimi_clusters = {
        row["base_cluster"] for row in critique if row["model"].startswith("Kimi K2")
    }
    assert kimi_clusters == {"kimi_k2_shared"}
    kimi_thinking = next(row for row in critique if row["model"] == "Kimi K2 Thinking")
    assert float(kimi_thinking["epoch_total_b"]) == 1000.0
    assert float(kimi_thinking["canonical_total_b"]) == 1040.0
    assert (
        kimi_thinking["parameter_truth_id"]
        == "moonshot-kimi-k2-family-report-table-1"
    )
    assert float(kimi_thinking["canonical_ratio"]) < float(
        kimi_thinking["reproduced_ratio"]
    )
    for row in critique:
        assert row["release_date"]
        assert row["aa_score_frozen_exact"]
        assert row["aa_score_current_exact"]
        assert row["eci_score_frozen_exact"]
        assert abs(float(row["reproduced_p_aa_b"]) - float(row["sheet_p_aa_b"])) < 1
        assert abs(float(row["reproduced_p_eci_b"]) - float(row["sheet_p_eci_b"])) < 1
        assert abs(float(row["reproduced_model_estimate_b"]) - float(row["sheet_model_estimate_b"])) < 1
        assert abs(float(row["reproduced_ratio"]) - float(row["sheet_ratio"])) < 0.01

    assert data["architecture_panel"] == {
        "rows": 89,
        "families": 40,
        "moe_rows": 29,
        "dense_rows": 60,
        "chronological_predictions": 69,
    }
    assert len(predictions) == 339
    assert len(targets) == 9
    assert all(row["train_max_date"] < row["release_date"] for row in predictions)
    assert all(row["test_family_excluded"] == "True" for row in predictions)
    assert min(int(row["train_n"]) for row in predictions) >= 20
    assert min(int(row["train_families"]) for row in predictions) >= 6

    active = data["heldout_metrics"]["active_then_date_sparsity"]
    live = data["heldout_metrics"]["live_eci_60_40"]
    assert active["all"]["median_multiplicative_error"] < live["all"]["median_multiplicative_error"]
    assert active["all"]["within_2x"] > live["all"]["within_2x"]
    assert data["paired_family_bootstraps_vs_live_eci"]["active_then_date_sparsity"]["ci_90"][1] > 0
    assert data["paired_moe_family_bootstraps_vs_live_eci"]["architecture_score_date_moe"]["ci_90"][1] > 0

    gates = data["promotion_gates"]
    assert gates["architecture_candidate"]["coverage_pass"] is True
    assert gates["architecture_candidate"]["all_gates_pass"] is False
    assert gates["active_sparsity_candidate"]["coverage_pass"] is True
    assert gates["active_sparsity_candidate"]["all_gates_pass"] is False
    assert data["decision"]["change_live_model"] is False
    assert data["decision"]["incremental_live_weight"] == 0.0

    less = data["less_is_more_component_evidence"]
    assert less["selected_target_feature_sets"] == {
        "Claude Fable 5": "knowledge_only",
        "GPT-5.6 Sol": "knowledge_only",
    }
    assert less["all_total_candidate_median_error_x"] < less["all_total_baseline_median_error_x"]
    assert less["promotion_decision"]["promote_multivariate_component_branch"] is False

    separation = data["presentation_layer_separation"]
    assert separation["final_site_consumed"] is False

    for relative, expected_hash in data["source_files"].items():
        assert sha256(ROOT / relative) == expected_hash


if __name__ == "__main__":
    test_epoch_feedback_reproduction_and_decision()
    print({"feedback_rows": 8, "predictions": 339, "status": "PASS"})
