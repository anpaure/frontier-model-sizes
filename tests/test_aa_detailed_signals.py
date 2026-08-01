from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

from collect_aa_detailed_signals import extract_models, require_record


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "sources/aa_detailed_snapshot_2026-07-31.html.gz"
MODELS = ROOT / "sources/aa_detailed_model_signals_2026-07-31.csv"
METADATA = ROOT / "sources/aa_detailed_collection_metadata_2026-07-31.json"
MANIFEST = ROOT / "sources/aa_detailed_snapshot_manifest_2026-07-31.json"
DELTA = (
    ROOT
    / "sources/aa_detailed_snapshot_delta_2026-07-18_to_2026-07-31.json"
)


def rows() -> list[dict[str, str]]:
    with MODELS.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class AADetailedSignalsTest(unittest.TestCase):
    def test_frozen_snapshot_hashes_and_counts(self) -> None:
        metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        raw_gzip = RAW.read_bytes()
        raw_html = gzip.decompress(raw_gzip)
        model_rows = rows()

        self.assertEqual(manifest["snapshot_date"], "2026-07-31")
        self.assertEqual(
            manifest["source_url"],
            "https://artificialanalysis.ai/models/claude-opus-5",
        )
        self.assertEqual(digest_bytes(raw_gzip), metadata["raw_gzip_sha256"])
        self.assertEqual(
            digest_bytes(raw_html), metadata["raw_html_uncompressed_sha256"]
        )
        self.assertEqual(digest_bytes(MODELS.read_bytes()), metadata["model_csv_sha256"])
        self.assertEqual(len(raw_html), metadata["raw_html_uncompressed_bytes"])
        self.assertEqual(len(model_rows), metadata["models"])
        self.assertEqual(len(model_rows), 587)
        self.assertEqual(len({row["model_id"] for row in model_rows}), 587)
        self.assertEqual(len({row["slug"] for row in model_rows}), 587)
        self.assertTrue(
            all(
                row["snapshot_html_sha256"]
                == metadata["raw_html_uncompressed_sha256"]
                for row in model_rows
            )
        )

        eligible = [
            row
            for row in model_rows
            if row["is_open_weights"] == "True"
            and row["parameters_b"]
            and row["intelligence_index"]
            and row["release_date"]
        ]
        token_rows = [
            row for row in eligible if row["intelligence_output_tokens_per_task"]
        ]
        self.assertEqual(len(eligible), 333)
        self.assertEqual(len(token_rows), 88)
        self.assertEqual(len({row["creator_slug"] for row in eligible}), 47)
        self.assertEqual(metadata["open_weight_models"], 335)
        self.assertEqual(metadata["models_with_total_parameters"], 347)
        self.assertEqual(metadata["models_with_active_parameters"], 167)

        for key in ("raw", "models", "metadata", "delta_from_prior", "prior_raw"):
            record = manifest["files"][key]
            path = ROOT / record["path"]
            payload = path.read_bytes()
            self.assertEqual(len(payload), record["bytes"])
            self.assertEqual(digest_bytes(payload), record["sha256"])

    def test_offline_rebuild_is_byte_deterministic(self) -> None:
        tracked = (RAW, MODELS, METADATA, MANIFEST, DELTA)
        before = {path: digest_bytes(path.read_bytes()) for path in tracked}
        subprocess.run(
            [sys.executable, str(ROOT / "collect_aa_detailed_signals.py")],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        after = {path: digest_bytes(path.read_bytes()) for path in tracked}
        self.assertEqual(after, before)

    def test_manifest_verifier_rejects_tampered_bytes(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "byte count changed"):
            require_record(b"tampered", manifest["files"]["raw"], "raw gzip")

    def test_flat_rows_reconcile_to_preserved_raw_records(self) -> None:
        model_rows = rows()
        raw_records = extract_models(gzip.decompress(RAW.read_bytes()).decode("utf-8"))
        self.assertEqual(set(raw_records), {row["slug"] for row in model_rows})
        for row in model_rows:
            preserved = json.loads(row["source_record_json"])
            raw = raw_records[row["slug"]]
            self.assertEqual(preserved, raw)
            self.assertEqual(row["model_id"], str(raw["id"]))
            self.assertEqual(row["name"], str(raw["name"]))
            self.assertEqual(row["release_date"], str(raw.get("releaseDate") or ""))
            self.assertEqual(
                row["intelligence_index"], str(raw.get("intelligenceIndex") or "")
            )
            self.assertTrue(row["source_page_url"].endswith("/" + row["slug"]))

    def test_frontier_records_include_exact_budget_fields(self) -> None:
        by_slug = {row["slug"]: row for row in rows()}
        for slug in (
            "claude-fable-5",
            "claude-opus-5",
            "gpt-5-6-sol",
            "kimi-k3",
            "claude-opus-4-8",
            "gpt-5-5",
            "grok-4-5",
        ):
            self.assertIn(slug, by_slug)
            row = by_slug[slug]
            self.assertTrue(row["intelligence_index"])
            self.assertTrue(row["intelligence_output_tokens_per_task"])
            self.assertGreater(float(row["intelligence_output_tokens_per_task"]), 0)
            self.assertGreater(float(row["median_output_speed_tps"]), 0)

        kimi = by_slug["kimi-k3"]
        self.assertEqual(float(kimi["parameters_b"]), 2800.0)
        self.assertEqual(float(kimi["active_parameters_b"]), 104.0)
        self.assertEqual(kimi["is_open_weights"], "True")
        self.assertEqual(kimi["open_source_categorization"], "commercial-license")
        self.assertEqual(
            kimi["model_weights_source_url"],
            "https://huggingface.co/moonshotai/Kimi-K3",
        )
        self.assertEqual(kimi["is_reasoning"], "True")
        self.assertAlmostEqual(float(kimi["intelligence_index"]), 57.1123394372091)

        opus = by_slug["claude-opus-5"]
        self.assertAlmostEqual(float(opus["intelligence_index"]), 60.6918740157091)
        self.assertEqual(opus["release_date"], "2026-07-24")
        self.assertFalse(opus["parameters_b"])

    def test_july_18_to_31_delta_is_explicit_and_exact(self) -> None:
        delta = json.loads(DELTA.read_text(encoding="utf-8"))
        expected_added = {
            "agnes-2-5-pro-alpha",
            "claude-opus-5",
            "claude-opus-5-high",
            "claude-opus-5-low",
            "claude-opus-5-medium",
            "claude-opus-5-xhigh",
            "g9v3-3b",
            "gemini-3-5-flash-lite",
            "gemini-3-6-flash",
            "inkling-small",
            "motif-0714",
        }
        self.assertEqual(delta["inventory"]["prior_models"], 576)
        self.assertEqual(delta["inventory"]["current_models"], 587)
        self.assertEqual(delta["inventory"]["shared_slugs"], 576)
        self.assertEqual(set(delta["inventory"]["added_slugs"]), expected_added)
        self.assertEqual(delta["inventory"]["removed_slugs"], [])
        self.assertEqual(delta["identity_changes"], [])
        self.assertEqual(delta["counts"]["shared_models_with_score_changes"], 170)
        self.assertEqual(
            delta["counts"]["shared_models_with_intelligence_index_changes"], 12
        )
        self.assertEqual(
            delta["counts"]["shared_models_with_architecture_changes"], 15
        )

        intelligence_changes = {
            row["slug"]: row["changes"]["intelligence_index"]
            for row in delta["score_changes"]
            if "intelligence_index" in row["changes"]
        }
        self.assertEqual(
            set(intelligence_changes),
            {
                "deepseek-v4-flash-high",
                "deepseek-v4-pro-high",
                "exaone-4-5-33b",
                "gemma-4-12b",
                "gemma-4-e2b",
                "gemma-4-e4b",
                "k-exaone",
                "mercury-2",
                "nemotron-cascade-2-30b-a3b",
                "qwen3-5-0-8b-non-reasoning",
                "qwen3-5-35b-a3b-non-reasoning",
                "trinity-large-thinking",
            },
        )
        self.assertEqual(
            intelligence_changes["deepseek-v4-pro-high"],
            {"before": 40.8255694978728, "after": 43.1117111950581},
        )
        self.assertEqual(
            intelligence_changes["trinity-large-thinking"],
            {"before": 24.4651372124901, "after": 18.1600095755902},
        )

        architecture = {
            row["slug"]: row["changes"] for row in delta["architecture_changes"]
        }
        self.assertEqual(
            architecture["kimi-k3"]["active_parameters_b"],
            {"before": None, "after": 104},
        )
        self.assertEqual(
            architecture["kimi-k3"]["is_open_weights"],
            {"before": False, "after": True},
        )
        self.assertEqual(
            architecture["jt-4-1-flash-236b-a21b"]["parameters_b"],
            {"before": None, "after": 236},
        )
        self.assertEqual(
            architecture["jt-4-1-flash-236b-a21b"]["active_parameters_b"],
            {"before": None, "after": 21},
        )

        added = {row["slug"]: row for row in delta["inventory"]["added_models"]}
        self.assertEqual(added["g9v3-3b"]["parameters_b"], 3)
        self.assertEqual(added["inkling-small"]["parameters_b"], 266)
        self.assertEqual(added["inkling-small"]["active_parameters_b"], 12)
        self.assertEqual(added["motif-0714"]["parameters_b"], 314)
        self.assertEqual(added["motif-0714"]["active_parameters_b"], 13)


if __name__ == "__main__":
    unittest.main()
