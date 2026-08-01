import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "sources/metr_benchmark_results_1_1_2026-07-18.yaml"
SIGNALS = ROOT / "sources/metr_horizon_official_signals_2026-07-18.csv"
METADATA = ROOT / "sources/metr_horizon_official_metadata_2026-07-18.json"
AUDIT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/metr_primary_source_audit_2026-07-18.json"
LEGACY = ROOT / "sources/metr_horizon_user_snapshot_2026-07-17.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MetrPrimarySourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with SIGNALS.open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.by_id = {row["source_id"]: row for row in cls.rows}
        cls.metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        cls.audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    def test_official_asset_and_inventory_are_pinned(self):
        self.assertEqual(
            sha256(RAW),
            "aae31902b0519a4da73e16643915e5e8aca13cd3315c3aac893ce3d6dfe92ad9",
        )
        self.assertEqual(len(self.rows), 26)
        self.assertEqual(len(self.by_id), 26)
        self.assertEqual(self.metadata["inventory"]["full_scaffold_entries"], 114)
        self.assertEqual(
            self.metadata["source"]["url"],
            "https://metr.org/assets/benchmark_results_1_1.yaml",
        )

    def test_trend_law_is_read_from_official_document(self):
        trend = self.metadata["trend"]
        self.assertEqual(trend["all_time_stitched_point_estimate_days"], "187.778")
        self.assertEqual(trend["from_2023_on_point_estimate_days"], "128.744")
        self.assertEqual(trend["from_2023_on_ci_low_days"], "104.428")
        self.assertEqual(trend["from_2023_on_ci_high_days"], "158.012")

    def test_every_full_scaffold_array_survives_normalization(self):
        scaffold_arrays = [json.loads(row["scaffolds_json"]) for row in self.rows]
        self.assertTrue(all(scaffolds for scaffolds in scaffold_arrays))
        self.assertEqual(sum(map(len, scaffold_arrays)), 114)
        mythos = self.by_id["claude_mythos_preview_early_inspect"]
        self.assertEqual(len(json.loads(mythos["scaffolds_json"])), 5)
        self.assertEqual(mythos["p50_estimate_minutes"], "1044.780145")

    def test_legacy_snapshot_is_an_exact_non_authoritative_crosscheck(self):
        crosscheck = self.audit["legacy_exact_crosscheck"]
        self.assertEqual(crosscheck["official_rows"], 26)
        self.assertEqual(crosscheck["legacy_rows"], 26)
        self.assertEqual(crosscheck["exact_rows"], 26)
        self.assertEqual(crosscheck["mismatch_count"], 0)
        self.assertEqual(crosscheck["mismatches"], [])
        self.assertTrue(
            self.metadata["integrity_policy"][
                "legacy_snapshot_used_only_as_exact_crosscheck"
            ]
        )
        self.assertEqual(
            self.audit["files"][str(LEGACY.relative_to(ROOT))]["sha256"],
            sha256(LEGACY),
        )

    def test_all_declared_file_hashes_reconcile(self):
        for relative, record in self.audit["files"].items():
            self.assertEqual(sha256(ROOT / relative), record["sha256"])


if __name__ == "__main__":
    unittest.main()
