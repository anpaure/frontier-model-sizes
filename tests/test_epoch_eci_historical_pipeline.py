from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tarfile
import unittest
from pathlib import Path

import collect_epoch_eci_historical_snapshots as collector


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources"
ARCHIVE = SOURCES / "epoch_eci_historical_snapshots_2026-07-18.tar.gz"
COLLECTION = SOURCES / "epoch_eci_historical_collection_metadata_2026-07-18.json"
FIT = SOURCES / "epoch_eci_historical_fit_metadata_2026-07-18.json"
HISTORICAL = SOURCES / "epoch_eci_historical_model_scores_2026-07-18.csv"
ARCHIVAL_CANONICAL = SOURCES / "epoch_eci_benchmarks_2026-07-17.csv"
LIVE_CANONICAL = SOURCES / "epoch_eci_benchmarks_2026-07-31.csv"
ARCHIVAL_SCORES = SOURCES / "epoch_eci_reproduced_scores_2026-07-18.csv"
LIVE_SCORES = SOURCES / "epoch_eci_reproduced_scores_2026-07-31.csv"
TERMINAL = "20260716153134"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EpochEciHistoricalPipelineTest(unittest.TestCase):
    def test_network_free_fit_verifier_accepts_frozen_outputs(self) -> None:
        watched = [
            FIT,
            HISTORICAL,
            SOURCES / "epoch_eci_historical_benchmark_parameters_2026-07-18.csv",
            SOURCES / "epoch_eci_historical_reconstructed_inputs_2026-07-18.csv.gz",
        ]
        before = {path: sha256(path) for path in watched}
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "fit_epoch_eci_historical_snapshots.py"),
                "--verify-existing",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        after = {path: sha256(path) for path in watched}
        self.assertEqual(before, after)
        result = json.loads(completed.stdout)
        self.assertEqual(result["mode"], "verify-existing")
        self.assertEqual(result["snapshots"], 15)
        self.assertEqual(result["outputs"], 3)
        self.assertEqual(result["terminal"], TERMINAL)

    @classmethod
    def setUpClass(cls) -> None:
        cls.collection = json.loads(COLLECTION.read_text(encoding="utf-8"))
        cls.fit = json.loads(FIT.read_text(encoding="utf-8"))

    def test_archive_terminal_and_live_successor_are_separate_vintages(self) -> None:
        policy = self.collection["archive_policy"]
        self.assertEqual(policy["terminal_timestamp"], TERMINAL)
        self.assertEqual(policy["terminal_capture_date"], "2026-07-16")
        terminal = self.collection["archival_terminal_exact_match"]
        successor = self.collection["current_live_successor_reference"]
        self.assertEqual(terminal["pinned_file"], "sources/epoch_eci_benchmarks_2026-07-17.csv")
        self.assertEqual(successor["pinned_file"], "sources/epoch_eci_benchmarks_2026-07-31.csv")
        self.assertTrue(terminal["exact"])
        self.assertFalse(successor["byte_equality_assertion_against_archival_terminal"])
        self.assertEqual(terminal["sha256"], sha256(ARCHIVAL_CANONICAL))
        self.assertEqual(successor["sha256"], sha256(LIVE_CANONICAL))
        self.assertNotEqual(sha256(ARCHIVAL_CANONICAL), sha256(LIVE_CANONICAL))

    def test_archive_terminal_bytes_match_only_the_archival_pin(self) -> None:
        member = f"canonical_csv/eci_benchmarks_{TERMINAL}.csv"
        with tarfile.open(ARCHIVE, mode="r:gz") as archive:
            handle = archive.extractfile(member)
            self.assertIsNotNone(handle)
            terminal_bytes = handle.read()
        self.assertEqual(terminal_bytes, ARCHIVAL_CANONICAL.read_bytes())
        self.assertNotEqual(terminal_bytes, LIVE_CANONICAL.read_bytes())

    def test_network_free_collector_verification_is_complete(self) -> None:
        before = {path: sha256(path) for path in (ARCHIVE, COLLECTION)}
        result = collector.verify_existing()
        after = {path: sha256(path) for path in (ARCHIVE, COLLECTION)}
        self.assertEqual(before, after)
        self.assertEqual(result["mode"], "verify-existing")
        self.assertEqual(result["archive_members"], 20)
        self.assertEqual(result["archival_terminal"], TERMINAL)
        self.assertEqual(result["current_live_source"], "sources/epoch_eci_benchmarks_2026-07-31.csv")

    def test_fit_crosschecks_archival_scores_and_only_references_live_scores(self) -> None:
        terminal = self.fit["archival_terminal_fit_crosscheck"]
        successor = self.fit["current_live_successor_reference"]
        self.assertEqual(terminal["timestamp"], TERMINAL)
        self.assertEqual(
            terminal["archival_terminal_scores"],
            "sources/epoch_eci_reproduced_scores_2026-07-18.csv",
        )
        self.assertEqual(
            terminal["archival_terminal_scores_sha256"], sha256(ARCHIVAL_SCORES)
        )
        self.assertTrue(terminal["exact_within_1e_8"])
        self.assertLess(terminal["maximum_absolute_eci_difference"], 1e-12)
        self.assertEqual(successor["scores"], "sources/epoch_eci_reproduced_scores_2026-07-31.csv")
        self.assertEqual(successor["scores_sha256"], sha256(LIVE_SCORES))
        self.assertFalse(
            successor["byte_equality_or_fit_assertion_against_archival_terminal"]
        )

    def test_historical_score_ledger_ends_before_live_refresh(self) -> None:
        with HISTORICAL.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        timestamps = {row["snapshot_timestamp"] for row in rows}
        self.assertEqual(max(timestamps), TERMINAL)
        self.assertLess(int(max(timestamps)), 20260731000000)
        self.assertEqual(len(timestamps), 15)


if __name__ == "__main__":
    unittest.main()
