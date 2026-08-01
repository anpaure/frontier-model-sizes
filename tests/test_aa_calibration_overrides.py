from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from aa_calibration_overrides import (
    OVERRIDES_PATH,
    apply_calibration_overrides,
    load_calibration_overrides,
    parameter_label_available_before,
    parameter_training_eligibility_date,
)


MOTIF_RAW = {
    "model_id": "a3043d4f-9b92-4261-8ea2-a2d391b09cf4",
    "slug": "motif-0714",
    "name": "Motif 3 (Beta)",
    "release_date": "2026-07-14",
    "is_open_weights": False,
    "open_source_categorization": "proprietary",
    "parameters_b": 314.0,
    "active_parameters_b": 13.0,
    "model_weights_source_url": "",
    "license_name": "",
    "license_url": "",
    "commercial_allowed": None,
}

MOTIF2_RAW = {
    "model_id": "666eb13f-0d22-4438-8eb0-01876e1a8604",
    "slug": "motif-2-12-7b",
    "name": "Motif-2-12.7B-Reasoning",
    "release_date": "2025-12-04",
    "is_open_weights": False,
    "open_source_categorization": "proprietary",
    "parameters_b": 12.7,
    "active_parameters_b": None,
    "model_weights_source_url": "",
    "license_name": "",
    "license_url": "",
    "commercial_allowed": None,
}


class AACalibrationOverridesTest(unittest.TestCase):
    def test_motif_overrides_are_primary_sourced_and_lineage_explicit(self) -> None:
        payload = load_calibration_overrides()
        self.assertEqual(len(payload["overrides"]), 2)
        overrides = {
            row["identity"]["aa_slug"]: row for row in payload["overrides"]
        }
        override = overrides["motif-0714"]
        self.assertEqual(
            override["primary_source"]["repository_commit"],
            "eee28175b11eada7e98142fef20afd5136ae92de",
        )
        self.assertEqual(
            override["primary_source"]["hugging_face_safetensors_total_parameters"],
            314841775750,
        )
        self.assertEqual(override["lineage"]["lineage_class"], "independent_pretrain")
        self.assertIsNone(override["lineage"]["base_model_id"])

        motif2 = overrides["motif-2-12-7b"]
        self.assertEqual(
            motif2["primary_source"]["repository_commit"],
            "02d060835415f43d57dc6bee72b1e2779d8e57e5",
        )
        self.assertEqual(
            motif2["primary_source"]["weights_commit"],
            "a259d49eca076d37c62c0a9eb7dca7893c763ceb",
        )
        self.assertEqual(
            motif2["primary_source"]["hugging_face_safetensors_total_parameters"],
            12_703_860_896,
        )
        self.assertEqual(motif2["primary_source"]["architecture_class"], "dense")
        self.assertEqual(motif2["replacement"]["license_name"], "Apache 2.0")
        self.assertEqual(motif2["lineage"]["lineage_class"], "same_base_posttrain")
        self.assertEqual(
            motif2["lineage"]["base_model_id"],
            "Motif-Technologies/Motif-2-12.7B-Base",
        )

    def test_overlay_corrects_eligibility_without_mutating_raw_row(self) -> None:
        original = copy.deepcopy(MOTIF_RAW)
        original2 = copy.deepcopy(MOTIF2_RAW)
        rows, audit = apply_calibration_overrides([MOTIF_RAW, MOTIF2_RAW])
        self.assertEqual(MOTIF_RAW, original)
        self.assertEqual(MOTIF2_RAW, original2)
        self.assertEqual(len(audit), 2)
        by_slug = {row["slug"]: row for row in rows}
        motif = by_slug["motif-0714"]
        self.assertIs(motif["is_open_weights"], True)
        self.assertEqual(motif["parameters_b"], 314.0)
        self.assertEqual(motif["active_parameters_b"], 13.0)
        self.assertEqual(
            motif["model_weights_source_url"],
            "https://huggingface.co/Motif-Technologies/Motif-3-Beta",
        )
        self.assertEqual(motif["lineage_class"], "independent_pretrain")
        self.assertEqual(motif["base_model_id"], "")
        self.assertEqual(motif["release_date"], "2026-07-14")
        self.assertEqual(motif["parameter_label_available_date"], "2026-07-20")
        self.assertEqual(
            parameter_training_eligibility_date(motif), "2026-07-20"
        )
        for prediction_date in (
            "2026-07-15",
            "2026-07-16",
            "2026-07-17",
            "2026-07-18",
            "2026-07-19",
            "2026-07-20",
        ):
            self.assertFalse(parameter_label_available_before(motif, prediction_date))
        self.assertTrue(parameter_label_available_before(motif, "2026-07-21"))
        audit_by_slug = {row["slug"]: row for row in audit}
        self.assertIn("is_open_weights", audit_by_slug["motif-0714"]["changed_fields"])
        self.assertEqual(
            len(audit_by_slug["motif-0714"]["source_record_sha256"]), 64
        )

        motif2 = by_slug["motif-2-12-7b"]
        self.assertIs(motif2["is_open_weights"], True)
        self.assertEqual(motif2["parameters_b"], 12.703860896)
        self.assertIsNone(motif2["active_parameters_b"])
        self.assertEqual(motif2["architecture_class"], "dense")
        self.assertEqual(motif2["exact_tensor_parameters"], 12_703_860_896)
        self.assertEqual(motif2["license_name"], "Apache 2.0")
        self.assertEqual(motif2["lineage_class"], "same_base_posttrain")
        self.assertEqual(
            motif2["base_model_id"], "Motif-Technologies/Motif-2-12.7B-Base"
        )
        self.assertEqual(motif2["parameter_label_available_date"], "2025-12-01")
        self.assertEqual(
            parameter_training_eligibility_date(motif2), "2025-12-04"
        )
        for prediction_date in ("2025-12-02", "2025-12-03", "2025-12-04"):
            self.assertFalse(parameter_label_available_before(motif2, prediction_date))
        self.assertTrue(parameter_label_available_before(motif2, "2025-12-05"))

    def test_stale_expected_value_fails_closed(self) -> None:
        stale = copy.deepcopy(MOTIF_RAW)
        stale["parameters_b"] = 315.0
        with self.assertRaisesRegex(ValueError, "stale expected value for parameters_b"):
            apply_calibration_overrides([stale, MOTIF2_RAW])

    def test_duplicate_override_identity_is_rejected(self) -> None:
        payload = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
        payload["overrides"].append(copy.deepcopy(payload["overrides"][0]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicates.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate identity"):
                load_calibration_overrides(path)

    def test_vendored_primary_evidence_is_required_and_hash_pinned(self) -> None:
        payload = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
        motif2 = next(
            row
            for row in payload["overrides"]
            if row["identity"]["aa_slug"] == "motif-2-12-7b"
        )
        self.assertEqual(
            {row["kind"] for row in motif2["primary_source"]["local_evidence"]},
            {
                "commit_pinned_model_card",
                "commit_pinned_config",
                "commit_pinned_hugging_face_model_api",
                "hugging_face_commit_history",
            },
        )
        motif2["primary_source"]["local_evidence"][0]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad-evidence-hash.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence hash mismatch"):
                load_calibration_overrides(path)


if __name__ == "__main__":
    unittest.main()
