from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collect_claude_opus5_evidence import SUMMARY, build_summary, canonical_json

EXPECTED_HASHES = {
    "anthropic_model_overview": "27d045cd02ff58b131087db93d3b04940cfa5b573de8c9fd9cddbadca96ac2ad",
    "anthropic_news": "ebec0339c1111b0a08bf206d3947a13763523aeff1e662e218893bf3f8a983ad",
    "anthropic_system_card": "897768f0f6f1724f3109279ab3f6458c9fbf496b56d5d2be14cab3a4f91ca472",
    "anthropic_whats_new": "5ded904d8ea2acbfd209b413ab33bf347de049bdb8f3a7988293708bc807d3d6",
    "artificial_analysis_launch_article": "f7f4be299c0569d8ca27be8cb882738c4628c8c9753002df808e151c43366e28",
    "artificial_analysis_model_page": "ebda24a4d59535f3499f5cce56bbd038f2293118fa97678fbaa31d0411183983",
    "epoch_all_ai_models": "0d1fcfc497cccc1079068a1ec3031e30e97ad5c8e6f5b5d43baceac3778ba579",
    "epoch_benchmark_data": "cc844ca094e4372ff81eb636f3503317d1d495d4f4c493bd6b7a37dd5f83049c",
    "epoch_eci_benchmarks": "d7cad7a8595347a62a2f832205aae579e371f05afe6ab08bc9506631e38c70d1",
    "metr_horizon": "aae31902b0519a4da73e16643915e5e8aca13cd3315c3aac893ce3d6dfe92ad9",
    "openrouter_catalog": "f3083683f2af01788379db7e349d01ef805b2d13b0d020b7f945f691f1f7dfbd",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ClaudeOpus5EvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.committed = json.loads(SUMMARY.read_text(encoding="utf-8"))
        cls.rebuilt = build_summary()

    def test_offline_rebuild_is_byte_exact(self) -> None:
        self.assertEqual(self.rebuilt, self.committed)
        self.assertEqual(canonical_json(self.rebuilt), SUMMARY.read_bytes())
        self.assertEqual(
            set(self.committed),
            {
                "identity",
                "artificial_analysis",
                "epoch",
                "anthropic_system_card",
                "api",
                "openrouter",
                "availability",
                "coverage_source_files",
                "coverage_source_hashes",
                "source_files",
                "source_hashes",
            },
        )

    def test_every_raw_source_is_immutable_and_hash_pinned(self) -> None:
        self.assertEqual(self.committed["source_hashes"], EXPECTED_HASHES)
        self.assertEqual(set(self.committed["source_files"]), set(EXPECTED_HASHES))
        for key, relative in self.committed["source_files"].items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), key)
            self.assertEqual(sha256(path), EXPECTED_HASHES[key], key)

    def test_negative_coverage_panels_are_hash_pinned(self) -> None:
        files = self.committed["coverage_source_files"]
        hashes = self.committed["coverage_source_hashes"]
        self.assertEqual(set(files), {"no_cot_model_panel", "ikp_model_inventory"})
        self.assertEqual(set(files), set(hashes))
        for key, relative in files.items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), key)
            self.assertEqual(sha256(path), hashes[key], key)

    def test_first_party_identity_and_api_contract(self) -> None:
        identity = self.committed["identity"]
        self.assertEqual(identity["canonical_name"], "Claude Opus 5")
        self.assertEqual(identity["release_date"], "2026-07-24")
        self.assertEqual(identity["api_model_id"], "claude-opus-5")
        self.assertFalse(identity["parameter_disclosed"])
        self.assertFalse(identity["same_weight_identity_disclosed"])
        self.assertEqual(identity["base_identity_policy"], "unique_base")
        self.assertIn("modeling policy", identity["same_weight_identity_status"])

        api = self.committed["api"]
        self.assertEqual(api["input_usd_per_mtok"], 5)
        self.assertEqual(api["output_usd_per_mtok"], 25)
        self.assertEqual(api["context_window_tokens"], 1_000_000)
        self.assertEqual(api["max_output_tokens"], 128_000)
        self.assertEqual(api["knowledge_cutoff"], "2026-05")
        self.assertEqual(api["training_data_cutoff"], "2026-05")

    def test_system_card_pretraining_and_aeci_are_exact(self) -> None:
        card = self.committed["anthropic_system_card"]
        self.assertEqual(card["training_pdf_page"], 10)
        self.assertEqual(card["aeci_pdf_page"], 29)
        self.assertEqual(
            card["pretraining_statement"],
            "After the pretraining process, Opus 5 underwent rigorous post-training and fine-tuning",
        )
        self.assertEqual(card["knowledge_cutoff"], "2026-05")
        self.assertEqual(
            card["aeci"],
            {
                "point_estimate": 162.1,
                "ci_low": 158.0,
                "ci_high": 167.3,
                "n_benchmarks": 40,
            },
        )

    def test_aa_preserves_all_five_efforts_and_selects_exact_max(self) -> None:
        aa = self.committed["artificial_analysis"]
        rows = aa["effort_rows"]
        self.assertEqual([row["effort"] for row in rows], ["low", "medium", "high", "xhigh", "max"])
        self.assertEqual(
            [row["score"] for row in rows],
            [
                50.6136789545909,
                56.2806227158121,
                58.8641803200348,
                60.0681611916281,
                60.6918740157091,
            ],
        )
        selected = aa["selected"]
        self.assertEqual(selected["name"], "Claude Opus 5 (Adaptive Reasoning, Max Effort)")
        self.assertEqual(selected["score"], 60.6918740157091)
        self.assertEqual(selected["output_tokens_total"], 101_161_732)
        self.assertEqual(selected["configuration"], "adaptive_reasoning_max_effort")
        self.assertEqual(selected["fallback_model"], "Claude Opus 4.8")

    def test_epoch_score_and_non_disclosures_are_not_rounded_away(self) -> None:
        epoch = self.committed["epoch"]
        self.assertEqual(epoch["eci_exact"], 159.3778667882398)
        self.assertEqual(epoch["eci_ci_low"], 157.24933114170264)
        self.assertEqual(epoch["eci_ci_high"], 162.20640578425878)
        self.assertEqual(epoch["displayed_eci"], 159.38)
        self.assertEqual(epoch["canonical_component_benchmarks"], 12)
        self.assertEqual(epoch["published_configuration_rows"], 7)
        self.assertIsNone(epoch["parameters"])
        self.assertIsNone(epoch["training_compute_flop"])

    def test_openrouter_tiers_and_negative_panels_are_explicit(self) -> None:
        tiers = self.committed["openrouter"]["tiers"]
        self.assertEqual(set(tiers), {"standard", "fast"})
        self.assertEqual(tiers["standard"]["id"], "anthropic/claude-opus-5")
        self.assertEqual(tiers["fast"]["id"], "anthropic/claude-opus-5-fast")
        self.assertEqual(
            (tiers["standard"]["input_usd_per_mtok"], tiers["standard"]["output_usd_per_mtok"]),
            (5, 25),
        )
        self.assertEqual(
            (tiers["fast"]["input_usd_per_mtok"], tiers["fast"]["output_usd_per_mtok"]),
            (10, 50),
        )
        for tier in tiers.values():
            self.assertEqual(tier["context_window_tokens"], 1_000_000)
            self.assertEqual(tier["max_output_tokens"], 128_000)

        availability = self.committed["availability"]
        self.assertFalse(availability["metr"])
        self.assertFalse(availability["no_cot"])
        self.assertFalse(availability["ikp"])


if __name__ == "__main__":
    unittest.main()
