from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import build_prospective_forecast_freeze as builder
from verify_prospective_forecast_freeze import verify


class ProspectiveForecastFreezeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact = json.loads(builder.ARTIFACT.read_text(encoding="utf-8"))

    def test_builder_is_deterministic_and_cannot_rewrite_the_freeze(self) -> None:
        first = builder.render_artifact()
        second = builder.render_artifact()
        self.assertEqual(first, second)
        # The repository has intentionally gained post-freeze diagnostics. A
        # rebuild from today's inputs must differ, while the one-shot writer
        # must refuse to replace the committed pre-disclosure bytes.
        self.assertNotEqual(first[0], builder.ARTIFACT.read_bytes())
        with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
            builder.write_once(builder.ARTIFACT, first[0])

    def test_artifact_and_repository_snapshot_verify(self) -> None:
        self.assertEqual(verify(), [])
        repository_drift = verify(check_repository=True)
        self.assertTrue(repository_drift)
        self.assertTrue(
            all("repository hash mismatch" in issue for issue in repository_drift)
        )

    def test_exact_locked_targets_centers_and_intervals(self) -> None:
        targets = {
            row["identity"]["model_id"]: row for row in self.artifact["targets"]
        }
        self.assertEqual(set(targets), set(builder.TARGET_IDS))
        self.assertEqual(
            targets["claude-fable-5"]["forecast"]["evidence_center_t"],
            4.731857984602975,
        )
        self.assertEqual(
            targets["claude-fable-5"]["forecast"]["final_center_t"],
            4.548292155336898,
        )
        self.assertEqual(
            targets["gpt-56-sol"]["forecast"]["evidence_center_t"],
            3.081003239637193,
        )
        self.assertEqual(
            targets["gpt-56-sol"]["forecast"]["final_center_t"],
            3.145668582841,
        )
        self.assertEqual(
            targets["claude-opus-5"]["forecast"]["evidence_center_t"],
            3.003794890426257,
        )
        self.assertEqual(
            targets["claude-opus-5"]["forecast"]["final_center_t"],
            3.003794890426257,
        )
        for target in targets.values():
            self.assertEqual(set(target["forecast"]["empirical_intervals"]), {"50", "80", "90"})
            self.assertIsNone(target["identity"]["outcome_at_freeze"])
            self.assertEqual(target["identity"]["parameter_status_at_freeze"], "undisclosed")

    def test_crowd_pools_and_effective_weights_are_frozen(self) -> None:
        targets = {
            row["identity"]["model_id"]: row for row in self.artifact["targets"]
        }
        self.assertEqual(targets["claude-fable-5"]["crowd_pool"]["n"], 20)
        self.assertEqual(targets["gpt-56-sol"]["crowd_pool"]["n"], 19)
        self.assertEqual(targets["claude-opus-5"]["crowd_pool"]["n"], 0)
        self.assertEqual(
            targets["claude-fable-5"]["crowd_pool"]["geometric_center_t"],
            4.371847506331046,
        )
        self.assertEqual(
            targets["gpt-56-sol"]["crowd_pool"]["geometric_center_t"],
            3.2116911484449235,
        )
        fable_weights = targets["claude-fable-5"]["forecast"]["weights"][
            "final_effective_weights_fraction"
        ]
        opus_weights = targets["claude-opus-5"]["forecast"]["weights"][
            "final_effective_weights_fraction"
        ]
        self.assertEqual(fable_weights["crowd"], 0.5)
        self.assertNotIn("crowd", opus_weights)
        self.assertEqual(opus_weights["horizon"], 0.5)

    def test_policy_forbids_post_outcome_refitting(self) -> None:
        policy = self.artifact["evaluation_policy"]
        self.assertEqual(policy["post_outcome_refitting"], "FORBIDDEN")
        self.assertIn("no forecast", policy["amendment_rule"])
        self.assertIn("Never regenerate", policy["evaluation_record_rule"])
        self.assertEqual(policy["primary_point_forecast"], "final_center_t")

    def test_privacy_redaction_preserves_anonymous_crowd_linkage(self) -> None:
        privacy = self.artifact["privacy_redaction"]
        self.assertFalse(privacy["name_to_id_mapping_retained"])
        self.assertFalse(privacy["numerical_values_changed"])
        self.assertFalse(privacy["original_name_bearing_bytes_retained_in_current_tree"])
        records = [
            record
            for target in self.artifact["targets"]
            for record in target["crowd_pool"]["records"]
        ]
        self.assertTrue(records)
        self.assertTrue(
            all(record["contributor"].startswith("Respondent R") for record in records)
        )
        self.assertTrue(all(record["forecast_id"].startswith("r") for record in records))

    def test_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / builder.ARTIFACT.name
            detached_path = root / builder.DETACHED_DIGEST.name
            payload = json.loads(builder.ARTIFACT.read_text(encoding="utf-8"))
            payload["targets"][0]["forecast"]["final_center_t"] = 999.0
            artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            detached_path.write_bytes(builder.DETACHED_DIGEST.read_bytes())
            issues = verify(
                artifact_path,
                detached_path=detached_path,
                check_repository=False,
            )
            self.assertTrue(any("digest mismatch" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main(verbosity=2)
