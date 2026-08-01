from __future__ import annotations

import csv
import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-07-18"
SOURCES = ROOT / "sources"
MANIFEST = SOURCES / f"openrouter_snapshot_history_manifest_{DATE}.csv"
MODEL_HISTORY = SOURCES / f"openrouter_model_snapshot_history_{DATE}.csv"
PROVIDER_HISTORY = SOURCES / f"openrouter_provider_snapshot_history_{DATE}.csv"
TIER_HISTORY = SOURCES / f"openrouter_endpoint_tier_snapshot_history_{DATE}.csv"
DAILY_HISTORY = SOURCES / f"openrouter_throughput_daily_history_{DATE}.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OpenRouterSnapshotHistoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = rows(MANIFEST)
        cls.models = rows(MODEL_HISTORY)
        cls.providers = rows(PROVIDER_HISTORY)
        cls.tiers = rows(TIER_HISTORY)
        cls.daily = rows(DAILY_HISTORY)

    def test_every_refresh_is_immutable_and_hash_verified(self) -> None:
        self.assertGreaterEqual(len(self.manifest), 8)
        self.assertEqual(
            len({row["history_snapshot_id"] for row in self.manifest}),
            len(self.manifest),
        )
        self.assertEqual(
            [row["fetched_at_utc"] for row in self.manifest],
            sorted(row["fetched_at_utc"] for row in self.manifest),
        )
        for row in self.manifest:
            archive = ROOT / row["archive_directory"]
            self.assertEqual(
                sha256(archive / f"openrouter_operational_snapshot_{DATE}.json.gz"),
                row["raw_sha256"],
            )
            self.assertEqual(
                sha256(archive / f"openrouter_model_signals_{DATE}.csv"),
                row["model_sha256"],
            )
            self.assertEqual(
                sha256(archive / f"openrouter_provider_signals_{DATE}.csv"),
                row["provider_sha256"],
            )
            self.assertEqual(
                sha256(archive / f"openrouter_collection_audit_{DATE}.json"),
                row["audit_sha256"],
            )

    def test_latest_refresh_has_current_observation_provenance(self) -> None:
        latest = self.manifest[-1]
        self.assertEqual(latest["snapshot_date"], "2026-07-31")
        self.assertEqual(latest["compatibility_filename_date"], DATE)
        self.assertEqual(int(latest["catalog_model_count"]), 364)
        self.assertEqual(int(latest["eligible_text_model_count"]), 334)
        for table in (self.models, self.providers, self.tiers, self.daily):
            current = [
                row
                for row in table
                if row["history_snapshot_id"] == latest["history_snapshot_id"]
            ]
            self.assertTrue(current)
            self.assertTrue(
                all(row["snapshot_date"] == "2026-07-31" for row in current)
            )

    def test_history_row_counts_reconcile_exactly(self) -> None:
        self.assertEqual(
            len(self.models),
            sum(int(row["eligible_text_model_count"]) for row in self.manifest),
        )
        self.assertEqual(
            len(self.providers),
            sum(int(row["provider_endpoint_row_count"]) for row in self.manifest),
        )
        self.assertEqual(
            len(self.tiers),
            sum(int(row["endpoint_tier_row_count"]) for row in self.manifest),
        )
        self.assertEqual(
            len(self.daily),
            sum(int(row["daily_throughput_row_count"]) for row in self.manifest),
        )
        for table in (self.models, self.providers, self.tiers, self.daily):
            self.assertEqual(
                {row["history_snapshot_id"] for row in table},
                {row["history_snapshot_id"] for row in self.manifest},
            )

    def test_snapshot_provenance_is_consistent_across_every_history_table(self) -> None:
        manifest_revision = {
            row["history_snapshot_id"]: row["source_revision"]
            for row in self.manifest
        }
        for table in (self.models, self.providers, self.tiers, self.daily):
            self.assertTrue(
                all(
                    row["history_source_revision"]
                    == manifest_revision[row["history_snapshot_id"]]
                    for row in table
                )
            )

    def test_daily_history_is_lossless_and_unique(self) -> None:
        keys = [
            (
                row["history_snapshot_id"],
                row["openrouter_model_id"],
                row["observation_time_raw"],
                row["endpoint_tier_key"],
            )
            for row in self.daily
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            {row["service_tier"] for row in self.daily},
            {"default", "priority", "flex"},
        )
        self.assertTrue(all(float(row["throughput_tps"]) >= 0 for row in self.daily))
        self.assertGreaterEqual(
            len({row["observation_date"] for row in self.daily}), 8
        )

    def test_endpoint_tier_history_is_lossless_and_unique(self) -> None:
        keys = [
            (
                row["history_snapshot_id"],
                row["openrouter_model_id"],
                row["endpoint_id"],
                row["service_tier"],
            )
            for row in self.tiers
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            {row["service_tier"] for row in self.tiers},
            {"default", "priority", "flex"},
        )
        self.assertGreater(
            sum(bool(row["high_context_min_prompt_tokens"]) for row in self.tiers),
            0,
        )


if __name__ == "__main__":
    unittest.main()
