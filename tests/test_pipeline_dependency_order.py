from __future__ import annotations

import unittest
from pathlib import Path

import run_forecast_pipeline as pipeline


class PipelineDependencyOrderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calls: list[tuple[list[str], str]] = []
        self.original_run = pipeline.run

        def record(command, cwd=pipeline.ROOT):
            self.calls.append(([str(part) for part in command], str(cwd)))

        pipeline.run = record
        pipeline.build()

    def tearDown(self) -> None:
        pipeline.run = self.original_run

    @property
    def scripts(self) -> list[str]:
        return [Path(command[1]).name if len(command) > 1 else command[0] for command, _ in self.calls]

    def index(self, script: str, *, last: bool = False) -> int:
        positions = [index for index, value in enumerate(self.scripts) if value == script]
        self.assertTrue(positions, f"missing pipeline stage: {script}")
        return positions[-1] if last else positions[0]

    def test_source_fidelity_precedes_derived_artifacts(self) -> None:
        order = [
            "audit_epoch_eci_reproduction.py",
            "collect_epoch_eci_historical_snapshots.py",
            "fit_epoch_eci_historical_snapshots.py",
            "collect_no_cot_exact_dates.py",
            "collect_aa_parameter_label_availability.py",
            "collect_aa_score_availability.py",
            "collect_open_model_parameter_truth.py",
            "build_unified_model_data.mjs",
            "collect_claude_opus5_evidence.py",
            "analyze_frontier_equivalence.py",
            "analyze_no_cot_exact_date_signal.py",
            "analyze_frontier_primary_evidence.py",
            "build_openrouter_unified_extension.py",
        ]
        self.assertEqual([self.index(script) for script in order], sorted(self.index(script) for script in order))

    def test_statistical_contract_precedes_single_presentation_build(self) -> None:
        self.assertEqual(self.scripts.count("build_horizon_informed_model.mjs"), 1)
        self.assertEqual(self.scripts.count("generate_forecast_site_data.py"), 1)
        self.assertEqual(self.scripts.count("run_parameter_backtest.py"), 1)
        backtest = self.index("run_parameter_backtest.py")
        label_availability = self.index("collect_aa_parameter_label_availability.py")
        score_availability = self.index("collect_aa_score_availability.py")
        parameter_truth = self.index("collect_open_model_parameter_truth.py")
        score_timing_audit = self.index("analyze_aa_score_availability_timing.py")
        vintage_knowledge = self.index("analyze_eci_vintage_knowledge_residual.py")
        active_transport = self.index("analyze_active_parameter_transport.py")
        active_shrinkage = self.index(
            "analyze_active_parameter_shrinkage_challenger.py"
        )
        epoch_feedback = self.index("analyze_epoch_feedback_signal.py")
        architecture_blend = self.index(
            "analyze_eci_architecture_blend_challenger.py"
        )
        ikp = self.index("analyze_ikp_parameter_signal.py")
        workbook = self.index("build_horizon_informed_model.mjs")
        site = self.index("generate_forecast_site_data.py")
        longcat_definition = self.index(
            "analyze_longcat_parameter_definition_sensitivity.py"
        )
        crowd_robustness = self.index("analyze_crowd_robustness.py")
        uncertainty = self.index("analyze_parameter_predictive_uncertainty.py")
        openrouter_increment = self.index("run_openrouter_incremental_backtest.py")
        weight_optimization = self.index("run_factor_weight_optimization.py")
        readiness = self.index("generate_model_readiness_report.py")
        self.assertLess(backtest, epoch_feedback)
        self.assertLess(label_availability, backtest)
        self.assertLess(score_availability, backtest)
        self.assertLess(parameter_truth, backtest)
        self.assertLess(score_timing_audit, uncertainty)
        self.assertLess(score_timing_audit, readiness)
        self.assertLess(backtest, vintage_knowledge)
        self.assertLess(vintage_knowledge, epoch_feedback)
        self.assertLess(vintage_knowledge, workbook)
        self.assertLess(active_transport, active_shrinkage)
        self.assertLess(active_shrinkage, workbook)
        self.assertLess(architecture_blend, workbook)
        self.assertLess(backtest, ikp)
        self.assertLess(epoch_feedback, workbook)
        self.assertLess(ikp, workbook)
        self.assertLess(workbook, site)
        self.assertLess(site, longcat_definition)
        self.assertLess(longcat_definition, readiness)
        self.assertLess(site, crowd_robustness)
        self.assertLess(crowd_robustness, readiness)
        self.assertLess(backtest, uncertainty)
        self.assertLess(backtest, openrouter_increment)
        self.assertLess(backtest, weight_optimization)
        self.assertLess(weight_optimization, readiness)

    def test_prospective_freeze_is_verified_but_never_rebuilt(self) -> None:
        self.assertEqual(self.scripts.count("verify_prospective_forecast_freeze.py"), 1)
        self.assertEqual(self.scripts.count("build_prospective_forecast_freeze.py"), 0)
        self.assertLess(
            self.index("verify_prospective_forecast_freeze.py"),
            self.index("manage_epoch_snapshot.py"),
        )

    def test_release_gates_run_after_statistical_regeneration(self) -> None:
        final_statistical_stage = max(
            self.index("analyze_parameter_predictive_uncertainty.py"),
            self.index("run_openrouter_incremental_backtest.py"),
            self.index("run_factor_weight_optimization.py"),
            self.index("generate_model_readiness_report.py"),
        )
        reaudit = self.index("run_codex_independent_reaudit.py")
        hash_gate = self.index("audit_declared_input_hashes.py")
        workbook_test = self.index("test_forecast_pipeline.py")
        privacy_test = self.index("test_poll_anonymization.py")
        site_build = next(
            index
            for index, (command, cwd) in enumerate(self.calls)
            if Path(cwd) == pipeline.SITE and len(command) >= 3 and command[-2:] == ["run", "build"]
        )
        rendered_test = next(
            index
            for index, (command, cwd) in enumerate(self.calls)
            if Path(cwd) == pipeline.SITE
            and any(Path(part).name == "rendered-html.test.mjs" for part in command)
        )
        site_package = self.index("package_forecast_site.py")
        package_privacy_test = next(
            index
            for index, (command, _) in enumerate(self.calls)
            if "tests.test_site_package_privacy" in command
        )
        self.assertLess(final_statistical_stage, reaudit)
        self.assertLess(reaudit, hash_gate)
        self.assertLess(hash_gate, workbook_test)
        self.assertLess(workbook_test, privacy_test)
        self.assertLess(privacy_test, site_build)
        self.assertLess(workbook_test, site_build)
        self.assertLess(site_build, site_package)
        self.assertLess(site_package, rendered_test)
        self.assertLess(site_package, package_privacy_test)
        self.assertLess(site_build, rendered_test)


if __name__ == "__main__":
    unittest.main()
