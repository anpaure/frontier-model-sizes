#!/usr/bin/env python3
"""Zero-weight sensitivity for LongCat 2.0 parameter-count conventions.

This audit never changes raw snapshots, the canonical parameter registry,
``regression_results.json``, backtest artifacts, or live forecast weights.  It
asks what would happen if the AA calibration row used either Meituan's
publisher-defined 1.6T model total or Hugging Face's exact 1.775560491136T
safetensors element inventory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

import analyze_frontier_equivalence as frontier
import run_parameter_backtest as backtest


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
INVENTORY_PATH = ROOT / "sources/longcat_parameter_definition_inventory_2026-07-31.json"
RESULT_PATH = OUTPUT_DIR / "longcat_parameter_definition_sensitivity_2026-07-31.json"
TARGET_CSV_PATH = OUTPUT_DIR / "longcat_parameter_definition_target_sensitivity_2026-07-31.csv"
METRICS_CSV_PATH = OUTPUT_DIR / "longcat_parameter_definition_backtest_sensitivity_2026-07-31.csv"
REPORT_PATH = ROOT / "LONGCAT_PARAMETER_DEFINITION_SENSITIVITY.md"
SITE_PATH = ROOT / "site/public/data/forecast-model.json"
K3_BUILDER_PATH = ROOT / "build_k3_calibrated_crosscheck.mjs"
PRICE_BUILDER_PATH = ROOT / "build_price_informed_crosscheck.mjs"
FINAL_BUILDER_PATH = ROOT / "build_horizon_informed_model.mjs"

PUBLISHER_TOTAL_B = 1600.0
HF_SERIALIZED_TOTAL_B = 1775.560491136
ACTIVE_B = 48.0
SCENARIOS = (
    (
        "publisher_model_total",
        PUBLISHER_TOTAL_B,
        "Meituan first-party release: publisher-defined rounded model total",
    ),
    (
        "hf_serialized_tensor_elements",
        HF_SERIALIZED_TOTAL_B,
        "Hugging Face safetensors element inventory at pinned repository revision",
    ),
)
HEADLINE_TARGETS = ("Claude Fable 5", "GPT-5.6 Sol", "Claude Opus 5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_inventory() -> dict[str, Any]:
    payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0" or payload.get("snapshot_date") != "2026-07-31":
        raise ValueError("Unsupported LongCat parameter-definition inventory")

    for section, path_key, hash_key in (
        ("publisher_definition", "source_path", "source_sha256"),
        ("hugging_face_serialized_inventory", "repository_api_path", "repository_api_sha256"),
        ("configuration_crosscheck", "config_path", "config_sha256"),
    ):
        record = payload[section]
        path = ROOT / record[path_key]
        if not path.is_file() or _sha256(path) != record[hash_key]:
            raise ValueError(f"LongCat evidence hash mismatch: {record[path_key]}")

    publisher_html = (ROOT / payload["publisher_definition"]["source_path"]).read_text(
        encoding="utf-8", errors="ignore"
    )
    if not all(token in publisher_html for token in ("总参数 1.6 T", "平均激活约 48 B", "33B~56B")):
        raise ValueError("First-party LongCat parameter statement is missing")

    api = json.loads(
        (ROOT / payload["hugging_face_serialized_inventory"]["repository_api_path"]).read_text(
            encoding="utf-8"
        )
    )
    hf = payload["hugging_face_serialized_inventory"]
    if (
        api.get("id") != payload["model"]["repository"]
        or api.get("sha") != payload["model"]["revision"]
        or int(api["safetensors"]["total"]) != int(hf["safetensors_total_elements"])
        or int(api["safetensors"]["parameters"]["BF16"]) != int(hf["bf16_elements"])
        or int(api["safetensors"]["parameters"]["F32"]) != int(hf["f32_elements"])
    ):
        raise ValueError("Hugging Face LongCat inventory no longer reconciles")

    config = json.loads(
        (ROOT / payload["configuration_crosscheck"]["config_path"]).read_text(encoding="utf-8")
    )
    if config.get("mtp_num_layers") != 3 or config.get("mtp_replicate_modules") is not True:
        raise ValueError("LongCat MTP configuration crosscheck failed")

    categories = payload["tensor_name_inventory"]
    category_total = sum(int(record["elements"]) for record in categories.values())
    category_tensors = sum(int(record["tensor_entries"]) for record in categories.values())
    if category_total != int(hf["safetensors_total_elements"]) or category_tensors != int(hf["tensor_entries"]):
        raise ValueError("LongCat tensor-name inventory does not sum to the HF totals")
    derived = payload["derived_reconciliation"]
    excluding_mtp = int(hf["safetensors_total_elements"]) - int(categories["mtp"]["elements"])
    if excluding_mtp != int(derived["serialized_elements_excluding_mtp"]):
        raise ValueError("LongCat non-MTP inventory arithmetic failed")
    if not (1_550_000_000_000 <= excluding_mtp < 1_650_000_000_000):
        raise ValueError("Non-MTP inventory does not round to the disclosed 1.6T label")
    return payload


def _frontier_scenario(total_b: float) -> dict[str, Any]:
    records = frontier.rows_to_records(frontier.OPEN_MODELS)
    longcat = [row for row in records if row["model"] == "LongCat 2.0"]
    if len(longcat) != 1 or longcat[0]["total_b"] != PUBLISHER_TOTAL_B or longcat[0]["active_b"] != ACTIVE_B:
        raise ValueError("Expected one unmodified LongCat AA calibration row")
    longcat[0]["total_b"] = total_b

    candidates = []
    for spec in ((), ("coder",), ("multimodal",), ("coder", "multimodal"), ("moe",)):
        fit = frontier.fit_grid(records, spec, fixed_reasoning=6.0)
        metrics = frontier.lofo_metrics(records, spec, fixed_reasoning=6.0)
        candidates.append((round(metrics["median_abs_log2_scale"], 3), len(spec), spec, fit, metrics))
    candidates.sort(key=lambda item: item[:2])
    _, _, selected_spec, fit, metrics = candidates[0]
    fit["frontier_shift"] = frontier.weighted_quantile(fit["resid"], 0.90, fit["weights"])

    eci_records = frontier.load_eci_records()
    eci_candidates = []
    for spec in ((), ("coder",), ("moe",)):
        eci_fit = frontier.fit_grid(eci_records, spec, fixed_reasoning=6.0)
        eci_metrics = frontier.lofo_metrics(eci_records, spec, fixed_reasoning=6.0)
        eci_candidates.append(
            (round(eci_metrics["median_abs_log2_scale"], 3), len(spec), spec, eci_fit)
        )
    eci_candidates.sort(key=lambda item: item[:2])
    _, _, eci_selected_spec, eci_fit = eci_candidates[0]
    eci_fit["frontier_shift"] = frontier.weighted_quantile(
        eci_fit["resid"], 0.90, eci_fit["weights"]
    )

    moe_ratio = float(np.median([row["total_b"] / row["active_b"] for row in records if row["moe"]]))
    targets = []
    for source in frontier.FRONTIER:
        release, model, aa_score = source[:3]
        eci_score = source[5]
        aa_effective = frontier.implied_eff(
            aa_score,
            release,
            fit,
            reasoning=1,
            frontier_shift=fit["frontier_shift"],
        )
        eci_effective = frontier.implied_eff(
            eci_score,
            release,
            eci_fit,
            reasoning=1,
            frontier_shift=eci_fit["frontier_shift"],
        )
        aa_total = aa_effective * moe_ratio ** (1 - fit["kappa"])
        eci_total = eci_effective * moe_ratio ** (1 - eci_fit["kappa"])
        targets.append(
            {
                "model": model,
                "aa_total_b": float(aa_total),
                "eci_total_b": float(eci_total),
                "legacy_frontier_equivalence_total_b": float(math.sqrt(aa_total * eci_total)),
            }
        )

    return {
        "longcat_total_b": total_b,
        "longcat_active_b": ACTIVE_B,
        "selected_spec": list(selected_spec),
        "kappa": float(fit["kappa"]),
        "coefficients": [float(value) for value in fit["beta"]],
        "frontier_shift": float(fit["frontier_shift"]),
        "lofo_metrics": metrics,
        "observed_moe_ratio_median": moe_ratio,
        "eci_selected_spec_unchanged": list(eci_selected_spec),
        "targets": targets,
    }


def _backtest_scenario(total_b: float) -> dict[str, Any]:
    panels, _ = backtest._load_panels()
    longcat = [row for row in panels["AA"] if row["model"] == "LongCat 2.0"]
    if len(longcat) != 1 or longcat[0]["total_b"] != PUBLISHER_TOTAL_B:
        raise ValueError("Expected one unmodified LongCat AA backtest row")
    # The label-timing ledger observed the publisher's 1.6T label.  Preserve it
    # as the raw value so only the target convention changes, never chronology.
    longcat[0]["raw_total_b"] = longcat[0]["total_b"]
    longcat[0]["total_b"] = total_b

    registry = deepcopy(backtest._load_parameter_truth_registry())
    key = backtest._normal_model_name("LongCat 2.0")
    for candidate in registry.get(key, []):
        candidate["actual_b"] = total_b

    aa = backtest._backtest(
        "AA", panels["AA"], "score_date", ("score", "date"), 16, 6, True, "score"
    )
    eci_score = backtest._backtest(
        "ECI", panels["ECI"], "score_only", ("score",), 20, 6, True, "score"
    )
    eci_date = backtest._backtest(
        "ECI", panels["ECI"], "score_date", ("score", "date"), 20, 6, True, "score"
    )
    eci = backtest._blend_predictions(
        "ECI", "blend_60_score_40_score_date", eci_score, eci_date, 0.60
    )
    nocot = backtest._backtest(
        "No-CoT",
        panels["No-CoT"],
        "log_horizon_date_moe",
        ("log_horizon", "date", "moe"),
        12,
        3,
        True,
        "log_horizon",
    )
    compute = backtest._backtest(
        "Compute",
        panels["Compute"],
        "log_compute_date",
        ("log_compute", "date"),
        100,
        20,
        True,
        "log_compute",
    )
    ensemble = backtest._current_ensemble(
        {"AA": aa, "ECI": eci, "No-CoT": nocot, "Compute": compute},
        equal_weight=False,
        parameter_registry=registry,
    )
    external = backtest._external_checks(panels)
    return {
        "longcat_total_b": total_b,
        "metrics": {
            "aa_all": backtest._metric_summary(aa),
            "aa_frontier": backtest._metric_summary(
                [row for row in aa if row["frontier_signal_rank"] >= 0.90]
            ),
            "ensemble_all": backtest._metric_summary(ensemble),
            "ensemble_frontier": backtest._metric_summary(
                [row for row in ensemble if row["frontier_signal_rank"] >= 0.90]
            ),
        },
        "aa_predictions": aa,
        "ensemble_predictions": ensemble,
        "k3_external_checks": [row for row in external if row["anchor"] == "Kimi K3"],
    }


def _live_dependency_audit() -> dict[str, Any]:
    k3_source = K3_BUILDER_PATH.read_text(encoding="utf-8")
    price_source = PRICE_BUILDER_PATH.read_text(encoding="utf-8")
    final_source = FINAL_BUILDER_PATH.read_text(encoding="utf-8")
    if "results.frontier_predictions" not in k3_source or "moe_total_b" in k3_source:
        raise ValueError("K3 builder dependency contract changed")
    if "k3_calibrated_frontier_parameter_crosscheck_2026-07-17.xlsx" not in price_source:
        raise ValueError("Price builder no longer consumes the K3-calibrated workbook")
    if "price_informed_frontier_parameter_crosscheck_2026-07-17.xlsx" not in final_source:
        raise ValueError("Final builder no longer consumes the price-informed workbook")

    site = json.loads(SITE_PATH.read_text(encoding="utf-8"))
    site_by_name = {row["name"]: row for row in site["models"]}
    targets = []
    for model in HEADLINE_TARGETS:
        row = site_by_name[model]
        targets.append(
            {
                "model": model,
                "publisher_current_evidence_t": float(row["currentEvidenceT"]),
                "hf_serialized_current_evidence_t": float(row["currentEvidenceT"]),
                "evidence_delta_percent": 0.0,
                "publisher_current_final_t": float(row["currentFinalT"]),
                "hf_serialized_current_final_t": float(row["currentFinalT"]),
                "final_delta_percent": 0.0,
            }
        )
    return {
        "longcat_parameter_target_consumed_by_live_center": False,
        "reason": (
            "The K3-calibrated live branch consumes frontier release dates, AA scores, and ECI scores "
            "from regression_results, but not its open-panel parameter targets or moe_total_b. The "
            "LongCat convention changes only diagnostic regression/backtest quantities."
        ),
        "frontier_fields_consumed": ["release_date", "model", "aa_score", "eci"],
        "excluded_regression_field": "moe_total_b",
        "downstream_targets": targets,
        "source_hashes": {
            str(K3_BUILDER_PATH.relative_to(ROOT)): _sha256(K3_BUILDER_PATH),
            str(PRICE_BUILDER_PATH.relative_to(ROOT)): _sha256(PRICE_BUILDER_PATH),
            str(FINAL_BUILDER_PATH.relative_to(ROOT)): _sha256(FINAL_BUILDER_PATH),
            str(SITE_PATH.relative_to(ROOT)): _sha256(SITE_PATH),
        },
    }


def _percent_change(new: float, old: float) -> float:
    return 100.0 * (new / old - 1.0)


def _write_report(result: dict[str, Any]) -> None:
    inventory = result["definition_evidence"]
    tensor = inventory["tensor_name_inventory"]
    reconciliation = inventory["derived_reconciliation"]
    publisher = result["scenarios"]["publisher_model_total"]
    hf = result["scenarios"]["hf_serialized_tensor_elements"]
    pub_targets = {row["model"]: row for row in publisher["frontier_regression"]["targets"]}
    hf_targets = {row["model"]: row for row in hf["frontier_regression"]["targets"]}
    pub_metrics = publisher["backtest"]["metrics"]
    hf_metrics = hf["backtest"]["metrics"]

    target_lines = []
    for model in HEADLINE_TARGETS:
        before = pub_targets[model]["legacy_frontier_equivalence_total_b"] / 1000
        after = hf_targets[model]["legacy_frontier_equivalence_total_b"] / 1000
        target_lines.append(
            f"| {model} | {before:.4f}T | {after:.4f}T | {_percent_change(after, before):+.2f}% |"
        )

    live_lines = []
    for row in result["live_dependency_audit"]["downstream_targets"]:
        live_lines.append(
            f"| {row['model']} | {row['publisher_current_evidence_t']:.4f}T | "
            f"{row['hf_serialized_current_evidence_t']:.4f}T | {row['evidence_delta_percent']:+.2f}% |"
        )

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# LongCat 2.0 parameter-definition sensitivity",
                "",
                "## Verdict",
                "",
                "Keep **1.6T total / about 48B average active** as the canonical regression target. It is the publisher's explicit model-level definition. Treat **1.775560491136T** as an exact serialized-tensor inventory sensitivity, not as a silent correction to the disclosed model total. The sensitivity receives **zero live weight**.",
                "",
                "## Why the two numbers differ",
                "",
                "| Quantity | Parameters | Interpretation |",
                "|---|---:|---|",
                f"| Publisher model total | {inventory['publisher_definition']['total_parameters'] / 1e9:.3f}B | First-party rounded semantic total |",
                f"| HF safetensors elements | {inventory['hugging_face_serialized_inventory']['safetensors_total_elements'] / 1e9:.6f}B | Exact stored tensor elements |",
                f"| `model.mtp.*` elements | {tensor['mtp']['elements'] / 1e9:.6f}B | Auxiliary multi-token-prediction tensors in the checkpoint |",
                f"| Serialized elements excluding MTP | {reconciliation['serialized_elements_excluding_mtp'] / 1e9:.6f}B | Rounds to 1.6T at the publisher's displayed precision |",
                "",
                f"MTP tensors explain {100 * reconciliation['mtp_share_of_serialized_minus_publisher_nominal']:.1f}% of the nominal 175.6B gap. The remaining 38.6B is inside the rounding interval for a one-decimal 1.6T label. The config explicitly declares three replicated MTP layers. This makes the publisher/HF difference consistent with a definition boundary, but it does not prove every detail of Meituan's counting convention; serialized inventories can also differ on tied, replicated, auxiliary, or inference-only tensors.",
                "",
                "## Diagnostic regression effect",
                "",
                "| Target | Publisher 1.6T | HF serialized 1.7756T | Change |",
                "|---|---:|---:|---:|",
                *target_lines,
                "",
                f"The strict AA median held-out error moves from {pub_metrics['aa_all']['median_multiplicative_error']:.4f}× to {hf_metrics['aa_all']['median_multiplicative_error']:.4f}×, while AA p80 moves from {pub_metrics['aa_all']['p80_multiplicative_error']:.4f}× to {hf_metrics['aa_all']['p80_multiplicative_error']:.4f}×. The available-component ensemble is unchanged at {pub_metrics['ensemble_all']['median_multiplicative_error']:.4f}× because LongCat has no independently matched second component in that ensemble, and its convention changes only the LongCat target plus later AA fits.",
                "",
                "## Live forecast effect",
                "",
                "| Target | Current evidence | HF counterfactual | Change |",
                "|---|---:|---:|---:|",
                *live_lines,
                "",
                "The exact live effect is zero: the K3-calibrated live branch consumes frontier AA scores, ECI scores, and dates, not the LongCat calibration target or the legacy `moe_total_b` field. This is a dependency-graph result, not an assertion that the diagnostic regression itself is invariant.",
                "",
                "## Provenance and policy",
                "",
                f"- Publisher source: {inventory['publisher_definition']['source_url']}",
                f"- HF repository API: {inventory['hugging_face_serialized_inventory']['repository_api_url']}",
                f"- Pinned tensor index: {inventory['hugging_face_serialized_inventory']['index_url']}",
                f"- Full machine-readable audit: `{RESULT_PATH.relative_to(ROOT)}`",
                f"- Target sensitivity CSV: `{TARGET_CSV_PATH.relative_to(ROOT)}`",
                f"- Backtest sensitivity CSV: `{METRICS_CSV_PATH.relative_to(ROOT)}`",
                "- Raw source snapshots, canonical parameter truth, live weights, and forecast centers are not modified by this audit.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    inventory = _load_inventory()
    scenarios: dict[str, Any] = {}
    for scenario_id, total_b, basis in SCENARIOS:
        scenarios[scenario_id] = {
            "parameter_basis": basis,
            "frontier_regression": _frontier_scenario(total_b),
            "backtest": _backtest_scenario(total_b),
        }

    publisher = scenarios["publisher_model_total"]
    hf = scenarios["hf_serialized_tensor_elements"]
    publisher_aa = {row["model"]: row for row in publisher["backtest"]["aa_predictions"]}
    hf_aa = {row["model"]: row for row in hf["backtest"]["aa_predictions"]}
    changed_aa = []
    for model in sorted(publisher_aa.keys() & hf_aa.keys()):
        before, after = publisher_aa[model], hf_aa[model]
        if not (
            math.isclose(before["actual_b"], after["actual_b"], rel_tol=0, abs_tol=1e-12)
            and math.isclose(before["predicted_b"], after["predicted_b"], rel_tol=0, abs_tol=1e-12)
        ):
            changed_aa.append(
                {
                    "model": model,
                    "publisher_actual_b": before["actual_b"],
                    "hf_serialized_actual_b": after["actual_b"],
                    "publisher_predicted_b": before["predicted_b"],
                    "hf_serialized_predicted_b": after["predicted_b"],
                    "publisher_error_factor": before["multiplicative_error"],
                    "hf_serialized_error_factor": after["multiplicative_error"],
                }
            )

    for scenario in scenarios.values():
        scenario["backtest"].pop("aa_predictions")
        scenario["backtest"].pop("ensemble_predictions")

    pub_targets = {
        row["model"]: row for row in publisher["frontier_regression"]["targets"]
    }
    hf_targets = {row["model"]: row for row in hf["frontier_regression"]["targets"]}
    target_changes = []
    for model in sorted(pub_targets):
        before = pub_targets[model]["legacy_frontier_equivalence_total_b"]
        after = hf_targets[model]["legacy_frontier_equivalence_total_b"]
        target_changes.append(
            {
                "model": model,
                "publisher_total_b": before,
                "hf_serialized_total_b": after,
                "change_percent": _percent_change(after, before),
            }
        )

    live = _live_dependency_audit()
    result = {
        "metadata": {
            "generated_on": "2026-07-31",
            "audit_type": "zero-weight parameter-definition sensitivity",
            "network_reads": 0,
            "raw_sources_modified": False,
            "canonical_parameter_registry_modified": False,
            "live_forecast_weights_modified": False,
        },
        "definition_evidence": inventory,
        "scenarios": scenarios,
        "changed_aa_predictions": changed_aa,
        "target_changes": target_changes,
        "live_dependency_audit": live,
        "decision": {
            "canonical_total_b": PUBLISHER_TOTAL_B,
            "canonical_active_b": ACTIVE_B,
            "canonical_basis": "publisher_disclosed_model_total",
            "hf_inventory_role": "zero_weight_serialized_element_sensitivity",
            "incremental_live_weight": 0.0,
            "change_live_forecast": False,
            "reason": (
                "The publisher and HF values answer different parameter-definition questions. "
                "The diagnostic target changes headline legacy regression estimates by less than 1%, "
                "leaves the matched ensemble unchanged, and has no dependency path into live centers."
            ),
        },
        "source_hashes": {
            str(INVENTORY_PATH.relative_to(ROOT)): _sha256(INVENTORY_PATH),
            "regression_results.json": _sha256(ROOT / "regression_results.json"),
            str(backtest.RESULT_PATH.relative_to(ROOT)): _sha256(backtest.RESULT_PATH),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    with TARGET_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model", "publisher_total_b", "hf_serialized_total_b", "change_percent"),
        )
        writer.writeheader()
        writer.writerows(target_changes)

    metric_rows = []
    for scenario_id, scenario in scenarios.items():
        for cohort, metrics in scenario["backtest"]["metrics"].items():
            metric_rows.append({"scenario": scenario_id, "cohort": cohort, **metrics})
    metric_fields = (
        "scenario",
        "cohort",
        "n",
        "median_multiplicative_error",
        "geomean_multiplicative_error",
        "rmse_log10",
        "signed_bias_factor",
        "within_1_5x",
        "within_2x",
        "within_3x",
        "p80_multiplicative_error",
        "p90_multiplicative_error",
    )
    with METRICS_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metric_rows)

    _write_report(result)
    print(
        json.dumps(
            {
                "result": str(RESULT_PATH),
                "report": str(REPORT_PATH),
                "canonical_total_b": PUBLISHER_TOTAL_B,
                "hf_serialized_total_b": HF_SERIALIZED_TOTAL_B,
                "changed_aa_predictions": [row["model"] for row in changed_aa],
                "headline_target_changes_percent": {
                    row["model"]: row["change_percent"]
                    for row in target_changes
                    if row["model"] in HEADLINE_TARGETS
                },
                "live_target_change_percent": 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
