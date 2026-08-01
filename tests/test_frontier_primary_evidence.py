import csv
import gzip
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "sources/openai_gpt_5_6_system_card_2026-07-18.html.gz"
CLAIMS = ROOT / "sources/anthropic_fable_mythos_primary_claims_2026-07-18.json"
LEDGER = ROOT / "sources/frontier_primary_evidence_2026-07-18.csv"
METADATA = ROOT / "sources/frontier_primary_evidence_collection_metadata_2026-07-18.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrontierPrimaryEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with LEDGER.open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.by_id = {row["evidence_id"]: row for row in cls.rows}
        cls.claims = json.loads(CLAIMS.read_text(encoding="utf-8"))
        cls.metadata = json.loads(METADATA.read_text(encoding="utf-8"))

    def test_official_direct_horizons_are_exact(self):
        self.assertEqual(
            float(self.by_id["openai_gpt_5_6_sol_nocot_horizon"]["value"]), 3.6
        )
        self.assertEqual(
            float(self.by_id["openai_gpt_5_5_nocot_comparator"]["value"]), 2.3
        )
        self.assertEqual(self.metadata["inventory"]["numeric_measurements"], 2)
        with gzip.open(RAW, "rb") as handle:
            self.assertGreater(len(handle.read()), 100_000)

    def test_fable_identity_and_fallback_are_not_conflated(self):
        identity = self.by_id["anthropic_fable_mythos_shared_weights"]
        fallback = self.by_id["anthropic_fable_fallback_scope"]
        self.assertEqual(identity["comparator_model"], "Claude Mythos 5")
        self.assertEqual(
            identity["parameter_identity_policy"],
            "same_underlying_weights_single_parameter_target",
        )
        self.assertEqual(fallback["comparator_model"], "Claude Opus 4.8")
        self.assertEqual(
            fallback["parameter_identity_policy"],
            "fallback_is_serving_behavior_not_shared_base",
        )

    def test_anthropic_claim_locations_and_gpqa_caveat_are_pinned(self):
        expected_pages = {
            "shared_underlying_weights": 12,
            "client_fallback_to_opus_4_8": 14,
            "messages_api_no_default_fallback": 14,
            "no_fallback_means_underlying_mythos": 131,
            "gpqa_saturated": 258,
        }
        self.assertEqual(
            {key: value["page"] for key, value in self.claims["claims"].items()},
            expected_pages,
        )
        caveat = self.by_id["anthropic_gpqa_saturation_caveat"]
        self.assertEqual(caveat["live_weight_policy"], "exclude_from_new_incremental_size_signal")

    def test_metadata_and_file_hashes_reconcile(self):
        self.assertEqual(len(self.rows), 5)
        self.assertEqual(len(self.by_id), len(self.rows))
        for relative, record in self.metadata["files"].items():
            self.assertEqual(sha256(ROOT / relative), record["sha256"])
        policy = self.metadata["integrity_policy"]
        self.assertTrue(policy["official_sources_only"])
        self.assertTrue(policy["fable_opus_fallback_is_not_a_parameter_identity"])
        self.assertTrue(policy["direct_horizon_is_not_mapped_to_size_without_backtest"])


if __name__ == "__main__":
    unittest.main()
