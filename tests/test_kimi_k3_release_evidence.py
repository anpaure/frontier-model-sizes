#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "sources/kimi_k3_release_evidence_2026-07-31.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class KimiK3ReleaseEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.record = json.loads(SUMMARY.read_text(encoding="utf-8"))

    def test_source_identity_is_commit_pinned(self) -> None:
        policy = self.record["source_policy"]
        self.assertEqual(
            policy["pinned_commit"],
            "7c5be9599120d7993748de66a76128614f15f210",
        )
        self.assertTrue(policy["ordinary_build_is_offline"])
        self.assertTrue(policy["network_refresh_is_explicit"])
        sources = self.record["source_files"]
        self.assertIn(policy["pinned_commit"], sources["official_model_card"]["url"])
        self.assertIn(
            policy["pinned_commit"], sources["official_technical_report"]["url"]
        )
        config = sources["official_huggingface_config"]
        self.assertEqual(
            config["revision"], "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
        )
        self.assertIn(config["revision"], config["url"])

    def test_every_frozen_source_hash_reconciles(self) -> None:
        for source in self.record["source_files"].values():
            path = ROOT / source["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_size, source["bytes"])
            self.assertEqual(sha256(path), source["sha256"])

    def test_exact_total_and_activated_parameters_replace_old_placeholder(self) -> None:
        k3 = self.record["kimi_k3"]
        self.assertEqual(k3["total_parameters_b_exact"], 2780.0)
        self.assertEqual(k3["total_parameters_t_display"], 2.8)
        self.assertEqual(k3["activated_parameters_b_exact"], 104.2)
        self.assertTrue(k3["parameter_count_disclosed"])
        self.assertTrue(k3["activated_parameter_count_disclosed"])
        self.assertTrue(k3["weights_released"])
        self.assertEqual(k3["initial_model_release_date"], "2026-07-16")
        self.assertEqual(k3["weights_release_date"], "2026-07-27")
        self.assertEqual(
            self.record["validation"]["hf_initial_weights_commit"],
            "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721",
        )
        self.assertEqual(
            self.record["validation"]["hf_initial_weights_commit_utc"],
            "2026-07-27T13:31:26.000Z",
        )
        self.assertAlmostEqual(
            self.record["derived_quantities"]["total_to_activated_parameter_ratio"],
            2780 / 104.2,
            places=14,
        )
        self.assertAlmostEqual(
            self.record["derived_quantities"]["activated_parameter_fraction"],
            104.2 / 2780,
            places=14,
        )

    def test_architecture_table_is_complete_and_internally_consistent(self) -> None:
        k3 = self.record["kimi_k3"]
        self.assertEqual(k3["architecture"], "Mixture-of-Experts (MoE)")
        self.assertEqual(k3["layers"], 93)
        self.assertEqual(k3["attention_layer_composition"], {"kda": 69, "gated_mla": 24})
        self.assertEqual(k3["routed_experts"], 896)
        self.assertEqual(k3["selected_routed_experts_per_token"], 16)
        self.assertEqual(k3["shared_experts"], 2)
        self.assertAlmostEqual(
            self.record["derived_quantities"]["selected_routed_expert_fraction"],
            16 / 896,
            places=15,
        )
        self.assertGreater(
            self.record["derived_quantities"]["activated_parameter_fraction"],
            self.record["derived_quantities"]["selected_routed_expert_fraction"],
        )
        self.assertEqual(k3["released_config_next_token_prediction_layers"], 0)
        self.assertEqual(
            self.record["validation"]["hf_config_architecture_fields_verified"], 11
        )

    def test_k2_comparator_uses_the_report_exact_values(self) -> None:
        k2 = self.record["kimi_k2_comparator"]
        self.assertEqual(k2["total_parameters_b_exact"], 1040.0)
        self.assertEqual(k2["activated_parameters_b_exact"], 32.6)
        self.assertAlmostEqual(
            self.record["derived_quantities"]["k2_total_to_activated_parameter_ratio"],
            1040 / 32.6,
            places=14,
        )
        self.assertAlmostEqual(
            self.record["derived_quantities"]["k3_over_k2_activated_parameter_ratio"],
            104.2 / 32.6,
            places=14,
        )

    def test_training_compute_absences_are_explicit(self) -> None:
        training = self.record["training_disclosures"]
        self.assertFalse(training["exact_pretraining_tokens_disclosed"])
        self.assertFalse(training["exact_pretraining_flops_disclosed"])
        self.assertFalse(training["exact_posttraining_flops_disclosed"])
        self.assertEqual(training["reported_scaling_efficiency_vs_k2"], 2.5)
        self.assertEqual(training["posttraining_stages"], [
            "supervised fine-tuning",
            "reinforcement learning",
            "multi-teacher on-policy distillation",
        ])
        self.assertEqual(training["rl_teacher_experts"], 9)

    def test_report_is_visually_auditable_pdf(self) -> None:
        source = self.record["source_files"]["official_technical_report"]
        reader = PdfReader(ROOT / source["path"])
        self.assertEqual(len(reader.pages), 47)
        first_page = reader.pages[0].extract_text() or ""
        table_page = reader.pages[10].extract_text() or ""
        self.assertIn("104 billion activated", first_page)
        self.assertIn("Activated Parameters 32.6B 104.2B", table_page)


if __name__ == "__main__":
    unittest.main()
