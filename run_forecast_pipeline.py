#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get("FORECAST_PYTHON", sys.executable))
NODE = Path(os.environ.get("FORECAST_NODE", shutil.which("node") or "node"))
NPM = Path(shutil.which("npm") or "npm")
SITE = ROOT / "site"


def watched_state():
    """Return a cheap content-change proxy for every editable pipeline input.

    The old watcher observed only the human forecast ledger, so changes to
    benchmark snapshots, matching tables, collectors, or tests were silently
    ignored.  Generated outputs are excluded to avoid self-triggered rebuilds;
    after each successful build the state is sampled again so collector-written
    frozen sources become the new baseline.
    """

    paths = list((ROOT / "sources").rglob("*"))
    for suffix in ("*.py", "*.mjs"):
        paths.extend(ROOT.glob(suffix))
        paths.extend((ROOT / "tests").glob(suffix))
    paths.extend((ROOT / "site" / "app").rglob("*.tsx"))
    paths.extend((ROOT / "site" / "tests").rglob("*.mjs"))
    return tuple(
        (str(path.relative_to(ROOT)), path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(set(paths))
        if path.is_file()
    )


def run(command, cwd=ROOT):
    environment = dict(os.environ)
    environment["FORECAST_PYTHON"] = str(PYTHON)
    # Release builds should not leave bytecode snapshots of earlier source
    # revisions in the project tree.  This also keeps privacy scans focused on
    # authoritative source and generated artifacts.
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([str(part) for part in command], cwd=cwd, check=True, env=environment)


def build(
    full_audit=False,
    refresh_openrouter=False,
    refresh_openrouter_history=False,
    refresh_aa=False,
    refresh_hf_configs=False,
    refresh_no_cot_dates=False,
    refresh_frontier_primary=False,
    refresh_metr=False,
    refresh_ikp=False,
    refresh_parameter_truth=False,
):
    # Validate the pre-disclosure commitment as a self-contained artifact on
    # every build.  Deliberately do not use --check-repository here: future
    # legitimate source/code changes must not mutate or invalidate the locked
    # historical forecast payload.
    run([PYTHON, ROOT / "verify_prospective_forecast_freeze.py"])
    if refresh_openrouter:
        run([PYTHON, ROOT / "collect_openrouter_signals.py"])
        run([PYTHON, ROOT / "audit_openrouter_official_endpoints.py", "--refresh"])
    if refresh_aa:
        run([PYTHON, ROOT / "collect_aa_detailed_signals.py", "--refresh"])
    historical_price_command = [
        PYTHON,
        ROOT / "collect_openrouter_historical_prices.py",
    ]
    if refresh_openrouter_history:
        historical_price_command.append("--refresh")
    run(historical_price_command)
    run([PYTHON, ROOT / "manage_epoch_snapshot.py"])
    run([PYTHON, ROOT / "audit_epoch_eci_reproduction.py"])
    run([PYTHON, ROOT / "tests/test_epoch_eci_reproduction.py"])
    # Historical captures terminate at the pinned July 16 archive.  Verify and
    # refit them offline; the July 31 current snapshot is a successor reference,
    # never a byte-equality target for the archival series.
    run([PYTHON, ROOT / "collect_epoch_eci_historical_snapshots.py", "--verify-existing"])
    run([PYTHON, ROOT / "fit_epoch_eci_historical_snapshots.py", "--verify-existing"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_epoch_eci_historical_pipeline"])
    # Keep the later July 22 capture outside the frozen selection archive. It
    # is verified and fit only as a score-vintage validation extension.
    run([PYTHON, ROOT / "collect_eci_validation_extension.py"])
    run([PYTHON, ROOT / "fit_eci_validation_extension.py", "--verify-existing"])
    run([PYTHON, ROOT / "collect_kimi_k3_release_evidence.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_kimi_k3_release_evidence"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_k3_primary_evidence"])
    frontier_primary_command = [PYTHON, ROOT / "collect_frontier_primary_evidence.py"]
    if refresh_frontier_primary:
        frontier_primary_command.append("--refresh")
    run(frontier_primary_command)
    run([PYTHON, "-m", "unittest", "-v", "tests.test_frontier_primary_evidence"])
    metr_command = [PYTHON, ROOT / "collect_metr_primary_source.py"]
    if refresh_metr:
        metr_command.append("--refresh")
    run(metr_command)
    run([PYTHON, "-m", "unittest", "-v", "tests.test_metr_primary_source"])
    ikp_command = [PYTHON, ROOT / "collect_ikp_source.py"]
    if refresh_ikp:
        ikp_command.append("--refresh")
    run(ikp_command)
    no_cot_date_command = [PYTHON, ROOT / "collect_no_cot_exact_dates.py"]
    if refresh_no_cot_dates:
        no_cot_date_command.append("--refresh")
    run(no_cot_date_command)
    run([PYTHON, "-m", "unittest", "-v", "tests.test_no_cot_exact_dates"])
    # Validate the frozen detailed AA source before it enters the canonical
    # megafile.  A live refresh, when requested, already happened above.
    run([PYTHON, ROOT / "collect_aa_detailed_signals.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_aa_detailed_signals"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_aa_calibration_overrides"])
    run([PYTHON, ROOT / "collect_aa_parameter_label_availability.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_aa_parameter_label_availability",
        ]
    )
    run([PYTHON, ROOT / "collect_aa_score_availability.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_aa_score_availability",
        ]
    )
    parameter_truth_command = [
        PYTHON,
        ROOT / "collect_open_model_parameter_truth.py",
    ]
    if refresh_parameter_truth:
        parameter_truth_command.append("--refresh")
    run(parameter_truth_command)
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_open_model_parameter_truth",
        ]
    )
    run([PYTHON, "-m", "unittest", "-v", "tests.test_frontier_target_signals"])
    run([NODE, ROOT / "build_unified_model_data.mjs"])
    run([NODE, ROOT / "tests/test_unified_model_data.mjs"])
    # Opus 5's negative No-CoT coverage check consumes the canonical base
    # megafile, never the later operational extension.  This makes a clean
    # build acyclic while retaining a hash-pinned absence assertion.
    run([PYTHON, ROOT / "collect_claude_opus5_evidence.py", "--rebuild-summary"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_claude_opus5_evidence"])
    run([PYTHON, ROOT / "analyze_frontier_equivalence.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_frontier_equivalence_exact_inputs"])
    # This audit depends only on the canonical compute-enriched megafile and
    # regression_results.json.  Build it before the K3 workbook so that the
    # workbook consumes current held-out checks instead of copied constants.
    run([PYTHON, ROOT / "run_parameter_backtest.py"])
    run([NODE, ROOT / "build_parameter_backtest_workbook.mjs"])
    run([PYTHON, ROOT / "test_parameter_backtest.py"])
    # These generated workbooks and registry are direct inputs to the final
    # workbook and must follow the current frontier evidence normalization.
    run([NODE, ROOT / "build_k3_calibrated_crosscheck.mjs"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_k3_crosscheck_current_data"])
    run([NODE, ROOT / "build_price_informed_crosscheck.mjs"])
    run([PYTHON, ROOT / "build_prediction_registry.py"])
    run([PYTHON, ROOT / "analyze_no_cot_exact_date_signal.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_no_cot_exact_date_signal"])
    run([PYTHON, ROOT / "analyze_no_cot_architecture_elasticity.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_no_cot_architecture_elasticity",
        ]
    )
    run([PYTHON, ROOT / "run_aa_expanded_parameter_audit.py"])
    run([PYTHON, ROOT / "tests/test_aa_expanded_parameter_audit.py"])
    run([PYTHON, ROOT / "analyze_aa_inference_budget_signal.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_aa_inference_budget_signal"])
    run([PYTHON, ROOT / "analyze_aa_score_availability_timing.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_aa_score_availability_timing_audit",
        ]
    )
    run([PYTHON, ROOT / "analyze_active_parameter_transport.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_active_parameter_transport"])
    run([PYTHON, ROOT / "analyze_active_parameter_shrinkage_challenger.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_active_parameter_shrinkage_challenger",
        ]
    )
    run([PYTHON, ROOT / "build_openrouter_snapshot_history.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_openrouter_signals"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_openrouter_snapshot_history"])
    run([PYTHON, ROOT / "audit_openrouter_official_endpoints.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_openrouter_official_endpoints"])
    run([PYTHON, ROOT / "analyze_openrouter_parameter_signal.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_openrouter_parameter_signal"])
    run([PYTHON, ROOT / "analyze_openrouter_temporal_stability.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_openrouter_temporal_stability"])
    run([PYTHON, ROOT / "analyze_openrouter_request_weighted_signal.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_openrouter_request_weighted_signal"])
    hf_config_command = [PYTHON, ROOT / "collect_huggingface_architecture_configs.py"]
    if refresh_hf_configs or refresh_openrouter:
        hf_config_command.append("--refresh")
    run(hf_config_command)
    run([PYTHON, "-m", "unittest", "-v", "tests.test_huggingface_architecture_configs"])
    run([PYTHON, ROOT / "analyze_openrouter_active_price_signal.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_openrouter_active_price_signal"])
    run([PYTHON, ROOT / "analyze_openrouter_historical_price_signal.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_openrouter_historical_price_signal",
        ]
    )
    run([PYTHON, ROOT / "analyze_aa_operational_signal.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_aa_operational_signal"])
    run([PYTHON, ROOT / "run_eci_component_backtest.py"])
    run([PYTHON, ROOT / "tests/test_eci_component_backtest.py"])
    run([PYTHON, ROOT / "run_eci_component_extended_audit.py"])
    run([PYTHON, ROOT / "tests/test_eci_component_extended_audit.py"])
    run([PYTHON, ROOT / "analyze_eci_fit_tournament.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_eci_fit_tournament"])
    run([PYTHON, ROOT / "analyze_eci_architecture_blend_challenger.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_eci_architecture_blend_challenger",
        ]
    )
    run([PYTHON, ROOT / "analyze_eci_historical_validation_extension.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_eci_historical_validation_extension",
        ]
    )
    run([PYTHON, ROOT / "analyze_eci_historical_common_components.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_eci_historical_common_components",
        ]
    )
    run([PYTHON, ROOT / "analyze_eci_multivariate_components.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_eci_multivariate_components"])
    run([PYTHON, ROOT / "analyze_posttraining_lineage_signal.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_posttraining_lineage_signal"])
    run([PYTHON, ROOT / "analyze_frontier_primary_evidence.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_frontier_primary_evidence_signal"])
    run([PYTHON, ROOT / "build_openrouter_unified_extension.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_openrouter_unified_extension"])
    # Rebuild the release-vintage/developer-holdout stress test from the same
    # current backtest and canonical megafile.  Keeping this in the DAG avoids
    # stale precision diagnostics after identities or parameter truths change.
    run([PYTHON, ROOT / "analyze_parameter_vintage_sensitivity.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_parameter_vintage_sensitivity",
        ]
    )
    # Evaluate the promising knowledge-component residual branch using only
    # snapshot-vintage Epoch inputs.  Both outer and inner folds are strictly
    # chronological whole-developer holdouts, and the branch remains at zero
    # weight unless every predeclared coverage/stability gate passes.
    run([PYTHON, ROOT / "analyze_eci_vintage_knowledge_residual.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_eci_vintage_knowledge_residual",
        ]
    )
    # Reproduce the external Epoch-employee calibration sheet exactly, then
    # rerun the lean architecture/active-parameter alternatives under the same
    # strict chronological, test-family-excluded policy as the main backtest.
    # This is a structural challenge set, not an iid holdout, and it can change
    # the live model only by passing the predeclared family-bootstrap, MoE, and
    # disclosed-frontier-anchor gates encoded in the audit.
    run([PYTHON, ROOT / "analyze_epoch_feedback_signal.py"])
    run([PYTHON, ROOT / "tests/test_epoch_feedback_signal.py"])
    run([PYTHON, ROOT / "analyze_ikp_parameter_signal.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_ikp_parameter_signal"])
    run([PYTHON, ROOT / "analyze_ikp_conditional_benchmark_signal.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_ikp_conditional_benchmark_signal",
        ]
    )
    # Build presentation artifacts once, only after every statistical decision
    # that they contain.  No upstream audit may read these final artifacts.
    run([NODE, ROOT / "build_horizon_informed_model.mjs"])
    run([PYTHON, ROOT / "generate_forecast_site_data.py"])
    # Keep the publisher-defined LongCat 1.6T model total canonical while
    # quantifying the exact 1.7756T serialized-tensor convention downstream.
    # This diagnostic depends on the rebuilt site centers but has no path back
    # into model fitting or live weights.
    run([PYTHON, ROOT / "analyze_longcat_parameter_definition_sensitivity.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_longcat_parameter_definition_sensitivity",
        ]
    )
    run([PYTHON, ROOT / "analyze_crowd_robustness.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_crowd_robustness"])
    run([PYTHON, ROOT / "analyze_parameter_predictive_uncertainty.py"])
    run(
        [
            PYTHON,
            "-m",
            "unittest",
            "-v",
            "tests.test_parameter_predictive_uncertainty",
        ]
    )
    run([PYTHON, ROOT / "run_openrouter_incremental_backtest.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_openrouter_incremental_backtest"])
    run([PYTHON, ROOT / "run_factor_weight_optimization.py"])
    run([PYTHON, ROOT / "tests/test_factor_weight_optimization.py"])
    # Generate the human-readable readiness statement from the fully rebuilt
    # forecasts and validation artifacts.  This report is descriptive only;
    # no upstream model stage reads it.
    run([PYTHON, ROOT / "generate_model_readiness_report.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_model_readiness_report"])
    # The independent reaudit and workbook tests are the final data gates. Run
    # them after every downstream statistical artifact has been regenerated so
    # a successful build certifies one coherent dependency snapshot.
    run([PYTHON, ROOT / "run_codex_independent_reaudit.py"])
    run([PYTHON, ROOT / "audit_declared_input_hashes.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_declared_input_hashes"])
    run([PYTHON, ROOT / "tests/test_forecast_pipeline.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_ooxml_determinism"])
    run([PYTHON, ROOT / "tests/test_poll_anonymization.py"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_local_path_privacy"])
    run([PYTHON, "-m", "unittest", "-v", "tests.test_image_privacy"])
    run([NPM, "run", "build"], cwd=SITE)
    run([PYTHON, ROOT / "package_forecast_site.py"])
    run([NODE, "--test", "tests/rendered-html.test.mjs"], cwd=SITE)
    run([PYTHON, "-m", "unittest", "-v", "tests.test_site_package_privacy"])


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the prediction registry and final workbook from the normalized human-forecast ledger."
    )
    parser.add_argument("--watch", action="store_true", help="Rebuild whenever the ledger changes.")
    parser.add_argument(
        "--full-audit",
        action="store_true",
        help="Deprecated compatibility flag; the independent model-data reaudit is now a mandatory build gate.",
    )
    parser.add_argument(
        "--refresh-openrouter",
        action="store_true",
        help="Refresh the live OpenRouter catalog/provider snapshot before rebuilding; otherwise use the frozen audited snapshot.",
    )
    parser.add_argument(
        "--refresh-aa",
        action="store_true",
        help="Refresh the live Artificial Analysis React-Flight snapshot before rebuilding; otherwise use the frozen audited snapshot.",
    )
    parser.add_argument(
        "--refresh-openrouter-history",
        action="store_true",
        help="Re-fetch the hash-pinned historical OpenRouter price ledger; otherwise rebuild from the frozen audited source.",
    )
    parser.add_argument(
        "--refresh-hf-configs",
        action="store_true",
        help="Refresh primary Hugging Face architecture configs before rebuilding; otherwise use the frozen audited snapshot.",
    )
    parser.add_argument(
        "--refresh-no-cot-dates",
        action="store_true",
        help="Refresh the first-party Qwen checkpoint history used by the no-CoT exact-date ledger; otherwise use the frozen audited snapshot.",
    )
    parser.add_argument(
        "--refresh-frontier-primary",
        action="store_true",
        help="Refresh the official OpenAI GPT-5.6 and Anthropic Fable/Mythos sources; otherwise rebuild from frozen audited artifacts.",
    )
    parser.add_argument(
        "--refresh-metr",
        action="store_true",
        help="Refresh the official METR-Horizon-v1.1 YAML before rebuilding; otherwise use the frozen audited asset.",
    )
    parser.add_argument(
        "--refresh-ikp",
        action="store_true",
        help="Refresh immutable IKP benchmark and independent-replication files before rebuilding; otherwise verify the frozen snapshots.",
    )
    parser.add_argument(
        "--refresh-parameter-truth",
        action="store_true",
        help="Refresh the two official MiniMax Hugging Face API records used by the narrow parameter-truth overlay; otherwise verify the frozen evidence offline.",
    )
    args = parser.parse_args()
    build(
        args.full_audit,
        args.refresh_openrouter,
        args.refresh_openrouter_history,
        args.refresh_aa,
        args.refresh_hf_configs,
        args.refresh_no_cot_dates,
        args.refresh_frontier_primary,
        args.refresh_metr,
        args.refresh_ikp,
        args.refresh_parameter_truth,
    )
    if not args.watch:
        return
    previous_state = watched_state()
    print("Watching all source, model, audit, and site-test inputs")
    while True:
        time.sleep(1)
        current_state = watched_state()
        if current_state == previous_state:
            continue
        try:
            build(
                args.full_audit,
                args.refresh_openrouter,
                args.refresh_openrouter_history,
                args.refresh_aa,
                args.refresh_hf_configs,
                args.refresh_no_cot_dates,
                args.refresh_frontier_primary,
                args.refresh_metr,
                args.refresh_ikp,
                args.refresh_parameter_truth,
            )
            previous_state = watched_state()
        except subprocess.CalledProcessError as error:
            print(f"Build failed with exit code {error.returncode}; waiting for the next ledger edit.")
            previous_state = watched_state()


if __name__ == "__main__":
    main()
