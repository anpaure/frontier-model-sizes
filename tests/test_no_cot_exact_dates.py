import csv
import gzip
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OVERRIDES = ROOT / "sources/no_cot_exact_date_overrides_2026-07-18.csv"
RAW = ROOT / "sources/qwen3_30b_a3b_instruct_2507_hf_commits_2026-07-18.json.gz"
METADATA = ROOT / "sources/no_cot_exact_date_collection_metadata_2026-07-18.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NoCotExactDateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with OVERRIDES.open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.by_model = {row["paper_model"]: row for row in cls.rows}
        cls.metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        with gzip.open(RAW, "rb") as handle:
            cls.commits = json.loads(handle.read())

    def test_all_four_previously_month_only_rows_are_resolved(self):
        self.assertEqual(
            set(self.by_model),
            {"GPT-2", "GPT-3", "GPT-3.5", "Qwen 3 30B-A3B (2507)"},
        )
        self.assertEqual(self.by_model["GPT-2"]["exact_release_date"], "2019-02-14")
        self.assertEqual(self.by_model["GPT-3"]["exact_release_date"], "2020-05-28")
        self.assertEqual(self.by_model["GPT-3.5"]["exact_release_date"], "2022-03-15")
        self.assertEqual(
            self.by_model["Qwen 3 30B-A3B (2507)"]["exact_release_date"],
            "2025-07-28",
        )

    def test_date_overrides_never_authorize_parameter_joining(self):
        self.assertTrue(
            all(
                row["parameter_join_policy"] == "date_only_no_epoch_parameter_join"
                for row in self.rows
            )
        )
        policy = self.metadata["integrity_policy"]
        self.assertTrue(policy["legacy_epoch_parameter_join_exclusion_preserved"])
        self.assertTrue(policy["qwen_epoch_parameter_join_not_inferred"])

    def test_qwen_initial_commit_is_first_party_and_immutable(self):
        source = self.metadata["qwen_source"]
        initial = next(row for row in self.commits if row["id"] == source["initial_commit"])
        self.assertEqual(initial["title"], "initial commit")
        self.assertEqual(initial["date"], source["initial_timestamp"])
        self.assertEqual(initial["date"][:10], "2025-07-28")
        self.assertEqual(len({row["id"] for row in self.commits}), len(self.commits))

    def test_metadata_hashes_reconcile(self):
        for relative, record in self.metadata["files"].items():
            self.assertEqual(sha256(ROOT / relative), record["sha256"])
        self.assertEqual(self.metadata["inventory"]["paper_rows_overridden"], 4)
        self.assertTrue(self.metadata["integrity_policy"]["all_four_month_only_rows_resolved"])


if __name__ == "__main__":
    unittest.main()
