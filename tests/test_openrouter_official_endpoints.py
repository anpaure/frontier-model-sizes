from __future__ import annotations

import csv
import gzip
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-07-18"
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
AUDIT = OUT / f"openrouter_official_endpoint_audit_{DATE}.json"
OFFICIAL = ROOT / f"sources/openrouter_official_endpoint_prices_{DATE}.csv"
RAW = ROOT / f"sources/openrouter_official_endpoint_snapshot_{DATE}.json.gz"
COMPARISON = OUT / f"openrouter_official_endpoint_crosscheck_{DATE}.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if len(fields) != len(set(fields)):
            raise AssertionError(f"Duplicate header in {path}")
        return list(reader)


class OpenRouterOfficialEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        cls.official = rows(OFFICIAL)
        cls.comparison = rows(COMPARISON)

    def test_official_snapshot_is_complete_and_frozen(self) -> None:
        self.assertEqual(self.audit["failure_count"], 0)
        self.assertEqual(self.audit["model_count_requested"], 334)
        self.assertEqual(self.audit["model_count_succeeded"], 334)
        self.assertEqual(self.audit["official_endpoint_rows"], 1051)
        self.assertEqual(self.audit["frontend_endpoint_tier_rows"], 1172)
        self.assertEqual(len(self.official), self.audit["official_endpoint_rows"])
        with gzip.open(RAW, "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
        self.assertEqual(len(raw["models"]), self.audit["model_count_requested"])
        self.assertTrue(all(not row["error"] for row in raw["models"]))
        for relative, expected in self.audit["source_hashes"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected
            )

    def test_price_crosscheck_is_nearly_exact_and_mismatches_are_retained(self) -> None:
        self.assertEqual(self.audit["official_rows_in_exact_groups"], 995)
        self.assertAlmostEqual(
            self.audit["official_price_row_exact_share"], 995 / 1051
        )
        self.assertEqual(self.audit["comparison_group_counts"]["exact"], 993)
        self.assertGreater(self.audit["comparison_group_counts"]["official_only"], 0)
        self.assertGreater(self.audit["comparison_group_counts"]["frontend_only"], 0)
        self.assertTrue(
            all(row["official_signatures_json"] and row["frontend_signatures_json"] for row in self.comparison)
        )

    def test_frontier_prices_crosscheck(self) -> None:
        focal = self.audit["focal_models"]
        self.assertEqual(focal["anthropic/claude-fable-5"]["non_exact_groups"], [])
        self.assertEqual(focal["moonshotai/kimi-k3"]["non_exact_groups"], [])
        self.assertEqual(
            focal["openai/gpt-5.6-sol"]["non_exact_groups"],
            ["azure/priority"],
        )
        exact = {
            (row["openrouter_model_id"], row["provider_tag"])
            for row in self.comparison
            if row["status"] == "exact"
        }
        for key in (
            ("anthropic/claude-fable-5", "anthropic"),
            ("openai/gpt-5.6-sol", "openai"),
            ("openai/gpt-5.6-sol", "openai/flex"),
            ("openai/gpt-5.6-sol", "openai/priority"),
            ("moonshotai/kimi-k3", "moonshotai/mxfp4"),
        ):
            self.assertIn(key, exact)

    def test_units_and_uptime_are_valid(self) -> None:
        for row in self.official:
            for field in (
                "prompt_usd_per_mtoken",
                "completion_usd_per_mtoken",
                "cache_read_usd_per_mtoken",
                "cache_write_usd_per_mtoken",
            ):
                if row[field]:
                    self.assertGreaterEqual(float(row[field]), 0)
            for field in ("uptime_last_30m", "uptime_last_5m", "uptime_last_1d"):
                if row[field]:
                    self.assertGreaterEqual(float(row[field]), 0)
                    self.assertLessEqual(float(row[field]), 100)


if __name__ == "__main__":
    unittest.main()
