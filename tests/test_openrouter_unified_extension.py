from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
BASE_OBS = OUT / "unified_model_observations_compute_enriched_2026-07-17.csv"
BASE_MEAS = OUT / "unified_model_measurements_long_compute_enriched_2026-07-17.csv"
BASE_MANIFEST = OUT / "unified_model_source_manifest_compute_enriched_2026-07-17.csv"
OBS = OUT / "unified_model_observations_operational_enriched_2026-07-18.csv"
MEAS = OUT / "unified_model_measurements_long_operational_enriched_2026-07-18.csv"
MANIFEST = OUT / "unified_model_source_manifest_operational_enriched_2026-07-18.csv"
SUMMARY = OUT / "unified_model_data_summary_operational_enriched_2026-07-18.json"
AUDIT = OUT / "openrouter_epoch_match_audit_2026-07-18.csv"
COLLECTION_AUDIT = OUT / "openrouter_collection_audit_2026-07-18.json"
TRUTH = ROOT / "sources/open_model_parameter_truth_reconciliation_2026-07-31.json"


def load(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


class OpenRouterUnifiedExtensionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_obs_fields, cls.base_obs = load(BASE_OBS)
        cls.obs_fields, cls.obs = load(OBS)
        cls.base_meas_fields, cls.base_meas = load(BASE_MEAS)
        cls.meas_fields, cls.meas = load(MEAS)
        cls.base_manifest_fields, cls.base_manifest = load(BASE_MANIFEST)
        cls.manifest_fields, cls.manifest = load(MANIFEST)
        _, cls.audit = load(AUDIT)
        cls.summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        cls.collection_audit = json.loads(
            COLLECTION_AUDIT.read_text(encoding="utf-8")
        )
        cls.truth_ids = {
            row["truth_id"]
            for row in json.loads(TRUTH.read_text(encoding="utf-8"))["records"]
        }

    def assert_epoch_parameter_source(self, value: str) -> None:
        prefix = "Epoch Parameters"
        self.assertTrue(value == prefix or value.startswith(prefix + "; canonicalized by "))
        if value != prefix:
            self.assertIn(value.removeprefix(prefix + "; canonicalized by "), self.truth_ids)

    def test_every_original_observation_is_preserved_field_for_field(self) -> None:
        self.assertEqual(self.obs_fields[: len(self.base_obs_fields)], self.base_obs_fields)
        self.assertEqual(len(self.base_obs), self.summary["base_observations"])
        extended_by_id = {row["observation_id"]: row for row in self.obs}
        for original in self.base_obs:
            preserved = extended_by_id[original["observation_id"]]
            self.assertEqual(
                {field: preserved[field] for field in self.base_obs_fields},
                original,
            )

    def test_every_original_measurement_and_manifest_row_is_preserved(self) -> None:
        self.assertEqual(self.meas_fields, self.base_meas_fields)
        extended_by_id = {row["measurement_id"]: row for row in self.meas}
        for original in self.base_meas:
            self.assertEqual(extended_by_id[original["measurement_id"]], original)
        self.assertEqual(self.manifest_fields, self.base_manifest_fields)
        self.assertEqual(self.manifest[: len(self.base_manifest)], self.base_manifest)

    def test_new_row_counts_and_ids_reconcile(self) -> None:
        model_rows = [row for row in self.obs if row["source"] == "OpenRouter"]
        provider_rows = [row for row in self.obs if row["source"] == "OpenRouter Provider Stats"]
        tier_rows = [
            row for row in self.obs if row["source"] == "OpenRouter Provider Tier Stats"
        ]
        daily_rows = [
            row
            for row in self.obs
            if row["source"] == "OpenRouter Provider Daily Stats"
        ]
        historical_price_rows = [
            row
            for row in self.obs
            if row["source"] == "OpenRouter Historical Prices"
        ]
        no_cot_date_rows = [
            row for row in self.obs if row["source"] == "No-CoT Exact-Date Audit"
        ]
        frontier_primary_rows = [
            row
            for row in self.obs
            if row["source"]
            in {"Frontier Primary Evidence", "Frontier Primary Evidence Audit"}
        ]
        opus5_rows = [
            row for row in self.obs if row["source"] == "Claude Opus 5 Evidence"
        ]
        self.assertEqual(
            len(model_rows), self.collection_audit["eligible_text_model_count"]
        )
        self.assertEqual(
            len(provider_rows), self.collection_audit["provider_endpoint_row_count"]
        )
        self.assertEqual(
            len(model_rows), self.summary["openrouter_model_observations"]
        )
        self.assertEqual(
            len(provider_rows), self.summary["openrouter_provider_observations"]
        )
        self.assertEqual(
            len(tier_rows), self.collection_audit["endpoint_tier_row_count"]
        )
        self.assertEqual(len(tier_rows), self.summary["openrouter_tier_observations"])
        self.assertEqual(
            len(daily_rows), self.collection_audit["daily_throughput_row_count"]
        )
        self.assertEqual(
            len(daily_rows), self.summary["openrouter_daily_observations"]
        )
        self.assertEqual(
            len(historical_price_rows),
            self.summary["openrouter_historical_price_observations"],
        )
        self.assertEqual(len(no_cot_date_rows), 50)
        self.assertEqual(
            len(no_cot_date_rows), self.summary["no_cot_exact_date_observations"]
        )
        self.assertEqual(len(frontier_primary_rows), 39)
        self.assertEqual(
            len(frontier_primary_rows),
            self.summary["frontier_primary_evidence_observations"],
        )
        self.assertEqual(len(opus5_rows), 7)
        self.assertEqual(len(opus5_rows), self.summary["opus5_evidence_observations"])
        self.assertTrue(all(row["model_level_include"] == "no" for row in opus5_rows))
        self.assertEqual(
            len(self.obs),
            self.summary["base_observations"]
            + self.summary["openrouter_model_observations"]
            + self.summary["openrouter_provider_observations"]
            + self.summary["openrouter_tier_observations"]
            + self.summary["openrouter_daily_observations"]
            + self.summary["openrouter_historical_price_observations"]
            + self.summary["no_cot_exact_date_observations"]
            + self.summary["frontier_primary_evidence_observations"]
            + self.summary["opus5_evidence_observations"],
        )
        observation_ids = [row["observation_id"] for row in self.obs]
        measurement_ids = [row["measurement_id"] for row in self.meas]
        self.assertEqual(len(observation_ids), len(set(observation_ids)))
        self.assertEqual(len(measurement_ids), len(set(measurement_ids)))
        self.assertEqual(
            len(self.meas) - len(self.base_meas),
            self.summary["openrouter_measurements"]
            + self.summary["no_cot_exact_date_measurements"]
            + self.summary["frontier_primary_evidence_measurements"]
            + self.summary["opus5_evidence_measurements"],
        )
        historical_measurements = [
            row
            for row in self.meas
            if row["source"] == "OpenRouter Historical Prices"
        ]
        self.assertEqual(
            len(historical_measurements),
            self.summary["openrouter_historical_price_measurements"],
        )

    def test_no_cot_exact_date_audit_is_complete_and_date_only(self) -> None:
        rows = [row for row in self.obs if row["source"] == "No-CoT Exact-Date Audit"]
        model_rows = [row for row in rows if row["record_type"] == "release_date_audit"]
        law_rows = [row for row in rows if row["record_type"] == "scaling_law_date_sensitivity"]
        self.assertEqual(len(model_rows), 49)
        self.assertEqual(len(law_rows), 1)
        self.assertEqual(sum(row["epoch_match_status"] == "date_only_override" for row in model_rows), 4)
        base_by_model = {
            row["source_model_name"]: row
            for row in self.base_obs
            if row["source"] == "No-CoT" and row["source_model_name"] != "__scaling_law__"
        }
        for row in model_rows:
            base = base_by_model[row["source_model_name"]]
            self.assertEqual(row["canonical_checkpoint_id"], base["canonical_checkpoint_id"])
            self.assertEqual(row["canonical_release_date"], base["canonical_release_date"])
            self.assertEqual(row["total_parameters_b"], "")
            self.assertIn("Date-only evidence", row["notes"])
        measurements = [
            row for row in self.meas if row["source"] == "No-CoT Exact-Date Audit"
        ]
        self.assertEqual(len(measurements), 256)
        self.assertEqual(len(measurements), self.summary["no_cot_exact_date_measurements"])
        for row in measurements:
            float(row["value"])
            self.assertTrue(row["metric_name"].startswith("nocot.date_audit."))

    def test_frontier_primary_evidence_is_lossless_and_nonduplicative(self) -> None:
        evidence = [
            row for row in self.obs if row["source"] == "Frontier Primary Evidence"
        ]
        audit = [
            row
            for row in self.obs
            if row["source"] == "Frontier Primary Evidence Audit"
        ]
        self.assertEqual(len(evidence), 5)
        self.assertEqual(len(audit), 34)
        by_type = {row["record_type"]: row for row in evidence}
        identity = by_type["base_model_identity"]
        fallback = by_type["serving_system_caveat"]
        self.assertEqual(
            identity["canonical_base_id"], "base:anthropic-claude-fable-mythos-5"
        )
        fallback_raw = json.loads(fallback["source_record_json"])
        self.assertEqual(fallback_raw["comparator_model"], "Claude Opus 4.8")
        self.assertEqual(
            fallback_raw["parameter_identity_policy"],
            "fallback_is_serving_behavior_not_shared_base",
        )
        opus = next(
            row
            for row in self.base_obs
            if row["source_model_name"] == "Claude Opus 4.8"
            and row["source"] == "ECI"
        )
        self.assertNotEqual(identity["canonical_base_id"], opus["canonical_base_id"])
        measurements = [
            row
            for row in self.meas
            if row["source"]
            in {"Frontier Primary Evidence", "Frontier Primary Evidence Audit"}
        ]
        self.assertEqual(
            len(measurements), self.summary["frontier_primary_evidence_measurements"]
        )
        direct = [
            row
            for row in measurements
            if row["source"] == "Frontier Primary Evidence"
        ]
        self.assertEqual(len(direct), 2)
        self.assertEqual(
            sorted(float(row["value"]) for row in direct), [2.3, 3.6]
        )
        self.assertTrue(all(row["model_level_include"] == "no" for row in evidence + audit))

    def test_opus5_evidence_bundle_is_lossless_and_nonduplicative(self) -> None:
        rows = [row for row in self.obs if row["source"] == "Claude Opus 5 Evidence"]
        self.assertEqual(
            {row["source_configuration"] for row in rows},
            {
                "identity",
                "anthropic_system_card",
                "api",
                "artificial_analysis",
                "epoch",
                "openrouter",
                "availability",
            },
        )
        measurements = [
            row for row in self.meas if row["source"] == "Claude Opus 5 Evidence"
        ]
        self.assertEqual(len(measurements), self.summary["opus5_evidence_measurements"])
        self.assertGreaterEqual(len(measurements), 40)
        self.assertTrue(all(row["canonical_checkpoint_id"] for row in measurements))

    def test_audited_epoch_matches_and_unmatched_rows_are_not_conflated(self) -> None:
        model_by_id = {
            row["or_openrouter_model_id"]: row
            for row in self.obs
            if row["source"] == "OpenRouter"
        }
        for audit in self.audit:
            row = model_by_id[audit["openrouter_model_id"]]
            if audit["match_status"] == "matched_epoch_manual":
                self.assertEqual(row["canonical_checkpoint_id"], audit["canonical_checkpoint_id"])
                self.assertEqual(row["total_parameters_b"], audit["total_parameters_b"])
                self.assert_epoch_parameter_source(row["parameter_value_source"])
            elif audit["match_status"] == "unmatched":
                self.assertTrue(row["canonical_checkpoint_id"].startswith("openrouter:"))
                self.assertEqual(row["total_parameters_b"], "")
                self.assertEqual(row["parameter_value_source"], "")

    def test_provider_rows_have_parent_models_and_complete_raw_records(self) -> None:
        model_ids = {
            row["or_openrouter_model_id"]
            for row in self.obs
            if row["source"] == "OpenRouter"
        }
        provider_rows = [row for row in self.obs if row["source"] == "OpenRouter Provider Stats"]
        for row in provider_rows:
            self.assertIn(row["or_openrouter_model_id"], model_ids)
            self.assertTrue(row["or_endpoint_id"])
            raw = json.loads(row["source_record_json"])
            self.assertEqual(raw["endpoint_id"], row["or_endpoint_id"])
            self.assertEqual(raw["openrouter_model_id"], row["or_openrouter_model_id"])

        tier_rows = [
            row for row in self.obs if row["source"] == "OpenRouter Provider Tier Stats"
        ]
        tier_keys = []
        for row in tier_rows:
            self.assertIn(row["or_openrouter_model_id"], model_ids)
            self.assertIn(row["or_service_tier"], {"default", "priority", "flex"})
            self.assertTrue(row["or_endpoint_id"])
            tier_keys.append(
                (
                    row["or_openrouter_model_id"],
                    row["or_endpoint_id"],
                    row["or_service_tier"],
                )
            )
        self.assertEqual(len(tier_keys), len(set(tier_keys)))
        sol_openai = {
            row["or_service_tier"]: row
            for row in tier_rows
            if row["or_openrouter_model_id"] == "openai/gpt-5.6-sol"
            and row["or_provider_slug"] == "openai"
        }
        self.assertEqual(set(sol_openai), {"default", "flex", "priority"})
        self.assertEqual(sol_openai["default"]["or_high_context_min_prompt_tokens"], "272000")

        daily_rows = [
            row
            for row in self.obs
            if row["source"] == "OpenRouter Provider Daily Stats"
        ]
        daily_keys = []
        for row in daily_rows:
            self.assertIn(row["or_openrouter_model_id"], model_ids)
            self.assertTrue(row["or_observation_time_raw"])
            self.assertTrue(row["or_endpoint_tier_key"])
            self.assertIn(row["or_service_tier"], {"default", "priority", "flex"})
            raw = json.loads(row["source_record_json"])
            self.assertEqual(raw["throughput_tps"], row["or_throughput_tps"])
            daily_keys.append(
                (
                    row["or_openrouter_model_id"],
                    row["or_observation_time_raw"],
                    row["or_endpoint_tier_key"],
                )
            )
        self.assertEqual(len(daily_keys), len(set(daily_keys)))

    def test_long_measurements_are_numeric_and_link_to_observations(self) -> None:
        observation_ids = {row["observation_id"] for row in self.obs}
        new_measurements = [row for row in self.meas if row["source"].startswith("OpenRouter")]
        self.assertEqual(len(new_measurements), self.summary["openrouter_measurements"])
        for row in new_measurements:
            self.assertIn(row["observation_id"], observation_ids)
            float(row["value"])
            self.assertTrue(row["metric_name"].startswith("openrouter."))
            self.assertTrue(row["unit"])

    def test_output_hashes_and_manifest_hashes(self) -> None:
        for record in self.summary["files"].values():
            path = ROOT / record["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"])
        for row in self.manifest:
            path = Path(row["path"])
            if not path.is_absolute():
                path = ROOT / path
            self.assertTrue(path.exists())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
        sources = {row["source"] for row in self.manifest}
        self.assertIn("ECI", sources)
        current_eci = next(row for row in self.manifest if row["source"] == "ECI")
        self.assertTrue(current_eci["path"].endswith("epoch_eci_reproduced_scores_2026-07-31.csv"))
        self.assertNotIn("Official ECI reproduced scores", sources)
        self.assertIn("OpenRouter endpoint service-tier observations", sources)
        self.assertIn("OpenRouter official endpoint audit", sources)
        self.assertIn("OpenRouter request-weighted operational audit", sources)
        self.assertIn("OpenRouter request-weighted predictions", sources)
        self.assertIn("OpenRouter active-price audit", sources)
        self.assertIn("OpenRouter active-parameter identity ledger", sources)
        self.assertIn("Primary Hugging Face architecture config snapshot", sources)
        self.assertIn("Hugging Face architecture config signals", sources)
        self.assertIn("Hugging Face architecture config audit", sources)
        self.assertIn("OpenRouter historical price raw ledger", sources)
        self.assertIn("OpenRouter historical price change points", sources)
        self.assertIn("OpenRouter historical price backtest", sources)
        self.assertIn("OpenRouter historical price predictions", sources)
        self.assertIn("No-CoT exact-date audit", sources)
        self.assertIn("No-CoT exact-date model ledger", sources)
        self.assertIn("Frontier primary evidence", sources)
        self.assertIn("Frontier primary-evidence audit", sources)
        self.assertIn("Frontier primary-evidence controls", sources)
        self.assertIn("Claude Opus 5 normalized evidence bundle", sources)
        self.assertEqual(self.summary["source_manifest_rows"], len(self.manifest))


if __name__ == "__main__":
    unittest.main()
