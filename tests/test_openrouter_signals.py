from __future__ import annotations

import csv
import gzip
import json
import statistics
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILE_DATE = "2026-07-18"
OBSERVATION_DATE = "2026-07-31"
SOURCES = ROOT / "sources"
OUTPUTS = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.reader(handle))


class OpenRouterSignalsTest(unittest.TestCase):
    def test_collection_integrity(self) -> None:
        audit = json.loads((OUTPUTS / f"openrouter_collection_audit_{FILE_DATE}.json").read_text())
        models = _rows(SOURCES / f"openrouter_model_signals_{FILE_DATE}.csv")
        providers = _rows(SOURCES / f"openrouter_provider_signals_{FILE_DATE}.csv")
        tiers = _rows(SOURCES / f"openrouter_endpoint_tier_signals_{FILE_DATE}.csv")
        daily = _rows(SOURCES / f"openrouter_throughput_daily_{FILE_DATE}.csv")

        self.assertEqual(audit["snapshot_date"], OBSERVATION_DATE)
        self.assertEqual(audit["compatibility_filename_date"], FILE_DATE)
        for table in (models, providers, tiers, daily):
            self.assertTrue(
                all(row["snapshot_date"] == OBSERVATION_DATE for row in table)
            )
            self.assertTrue(
                all(row["fetched_at_utc"] == audit["fetched_at_utc"] for row in table)
            )

        self.assertIs(audit["unique_model_ids"], True)
        self.assertIs(audit["unique_provider_observation_keys"], True)
        self.assertEqual(len(models), audit["eligible_text_model_count"])
        self.assertEqual(len(providers), audit["provider_endpoint_row_count"])
        self.assertEqual(len(tiers), audit["endpoint_tier_row_count"])
        self.assertEqual(len(daily), audit["daily_throughput_row_count"])
        self.assertEqual(len({row["openrouter_model_id"] for row in models}), len(models))
        observation_keys = [
            (row["openrouter_model_id"], row["endpoint_id"], row["variant"])
            for row in providers
            if row["endpoint_id"]
        ]
        self.assertEqual(len(observation_keys), len(set(observation_keys)))
        daily_keys = [
            (row["openrouter_model_id"], row["observation_time_raw"], row["endpoint_tier_key"])
            for row in daily
        ]
        self.assertEqual(len(daily_keys), len(set(daily_keys)))
        self.assertIs(audit["unique_daily_throughput_keys"], True)
        tier_keys = [
            (row["openrouter_model_id"], row["endpoint_id"], row["service_tier"])
            for row in tiers
        ]
        self.assertEqual(len(tier_keys), len(set(tier_keys)))
        self.assertIs(audit["unique_endpoint_tier_keys"], True)
        self.assertEqual(
            len({row["openrouter_model_id"] for row in daily}),
            audit["daily_throughput_model_count"],
        )
        self.assertEqual(
            len({(row["openrouter_model_id"], row["endpoint_tier_key"]) for row in daily}),
            audit["daily_throughput_endpoint_tier_count"],
        )
        self.assertEqual(audit["failure_count"], 0)
        self.assertGreaterEqual(audit["models_with_price"], 250)
        self.assertGreaterEqual(audit["models_with_throughput_1w"], 150)
        self.assertGreater(audit["endpoint_tier_rows_with_high_context_price"], 0)

        for filename in (
            f"openrouter_provider_signals_{FILE_DATE}.csv",
            f"openrouter_endpoint_tier_signals_{FILE_DATE}.csv",
            f"openrouter_throughput_daily_{FILE_DATE}.csv",
            f"openrouter_model_signals_{FILE_DATE}.csv",
        ):
            header = _header(SOURCES / filename)
            self.assertEqual(len(header), len(set(header)), filename)

        raw_path = SOURCES / f"openrouter_operational_snapshot_{FILE_DATE}.json.gz"
        self.assertTrue(raw_path.exists())
        with gzip.open(raw_path, "rt", encoding="utf-8") as handle:
            raw = json.load(handle)
        self.assertEqual(len(raw["catalog_payload"]["data"]), audit["catalog_model_count"])
        self.assertEqual(len(raw["models"]), audit["eligible_text_model_count"])
        self.assertEqual(raw["snapshot_date"], OBSERVATION_DATE)
        self.assertEqual(raw["compatibility_filename_date"], FILE_DATE)
        self.assertEqual(audit["files"]["raw_snapshot"], str(raw_path.relative_to(ROOT)))

        raw_daily_count = sum(
            1
            for model in raw["models"]
            for point in (model["throughput_payload"].get("data") or [])
            for value in (point.get("y") or {}).values()
            if isinstance(value, (int, float))
        )
        self.assertEqual(raw_daily_count, len(daily))

    def test_units_and_aggregates(self) -> None:
        models = _rows(SOURCES / f"openrouter_model_signals_{FILE_DATE}.csv")
        providers = _rows(SOURCES / f"openrouter_provider_signals_{FILE_DATE}.csv")
        by_model = {row["openrouter_model_id"]: row for row in models}

        for row in providers:
            if row["prompt_usd_per_mtoken"]:
                self.assertGreaterEqual(float(row["prompt_usd_per_mtoken"]), 0)
            if row["completion_usd_per_mtoken"]:
                self.assertGreaterEqual(float(row["completion_usd_per_mtoken"]), 0)
            if row["throughput_median_tps_1w"]:
                self.assertGreater(float(row["throughput_median_tps_1w"]), 0)
                self.assertGreater(int(row["throughput_daily_observations_1w"]), 0)

        for model_id in ("moonshotai/kimi-k3", "openai/gpt-5.6-sol", "anthropic/claude-fable-5"):
            self.assertIn(model_id, by_model)
            row = by_model[model_id]
            self.assertGreater(float(row["prompt_price_median_usd_per_mtoken"]), 0)
            self.assertGreater(float(row["completion_price_median_usd_per_mtoken"]), 0)
            self.assertGreater(float(row["throughput_best_provider_median_tps_1w"]), 0)
            self.assertTrue(row["catalog_source_url"].startswith("https://openrouter.ai/"))
            self.assertTrue(row["endpoint_source_url"].startswith("https://openrouter.ai/"))
            self.assertTrue(row["throughput_source_url"].startswith("https://openrouter.ai/"))

        expected_frontier_prices = {
            "anthropic/claude-fable-5": (10.0, 50.0),
            "openai/gpt-5.6-sol": (5.25, 31.5),
            "moonshotai/kimi-k3": (3.0, 15.0),
            "anthropic/claude-opus-4.8": (5.0, 25.0),
            "openai/gpt-5.5": (5.25, 31.5),
            "openai/gpt-5.6-terra": (2.35, 14.1),
            "anthropic/claude-sonnet-5": (2.0, 10.0),
            "openai/gpt-5.6-luna": (0.61, 3.66),
            "x-ai/grok-4.5": (2.0, 6.0),
        }
        for model_id, (prompt, completion) in expected_frontier_prices.items():
            self.assertAlmostEqual(float(by_model[model_id]["prompt_price_median_usd_per_mtoken"]), prompt)
            self.assertAlmostEqual(float(by_model[model_id]["completion_price_median_usd_per_mtoken"]), completion)

        # The official endpoints now include sharply cheaper OpenAI serving
        # options for Terra and Luna.  Keep both the minimum and the robust
        # provider median so a future refresh cannot silently erase either.
        expected_openai_minima = {
            "openai/gpt-5.6-sol": (5.0, 30.0),
            "openai/gpt-5.6-terra": (1.0, 6.0),
            "openai/gpt-5.6-luna": (0.1, 0.6),
        }
        for model_id, (prompt, completion) in expected_openai_minima.items():
            self.assertAlmostEqual(
                float(by_model[model_id]["prompt_price_min_usd_per_mtoken"]),
                prompt,
            )
            self.assertAlmostEqual(
                float(by_model[model_id]["completion_price_min_usd_per_mtoken"]),
                completion,
            )

    def test_model_counts_reconcile_to_provider_rows(self) -> None:
        models = _rows(SOURCES / f"openrouter_model_signals_{FILE_DATE}.csv")
        providers = _rows(SOURCES / f"openrouter_provider_signals_{FILE_DATE}.csv")
        counts: dict[str, int] = {}
        for row in providers:
            counts[row["openrouter_model_id"]] = counts.get(row["openrouter_model_id"], 0) + 1
        for row in models:
            self.assertEqual(int(row["endpoint_count"]), counts.get(row["openrouter_model_id"], 0))

    def test_one_week_aggregates_use_default_service_tier_only(self) -> None:
        providers = _rows(SOURCES / f"openrouter_provider_signals_{FILE_DATE}.csv")
        daily = _rows(SOURCES / f"openrouter_throughput_daily_{FILE_DATE}.csv")
        default_values: dict[tuple[str, str], list[float]] = defaultdict(list)
        tiers: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in daily:
            key = (row["openrouter_model_id"], row["endpoint_id"])
            tiers[key].add(row["service_tier"])
            if row["service_tier"] == "default":
                default_values[key].append(float(row["throughput_tps"]))

        self.assertGreaterEqual(sum(len(values) > 1 for values in tiers.values()), 40)
        for row in providers:
            self.assertEqual(row["throughput_service_tier_1w"], "default")
            key = (row["openrouter_model_id"], row["endpoint_id"])
            values = default_values.get(key, [])
            self.assertEqual(int(row["throughput_daily_observations_1w"]), len(values))
            if not values:
                self.assertEqual(row["throughput_median_tps_1w"], "")
                continue
            self.assertAlmostEqual(
                float(row["throughput_median_tps_1w"]), statistics.median(values)
            )
            self.assertAlmostEqual(
                float(row["throughput_mean_tps_1w"]), statistics.fmean(values)
            )
            self.assertAlmostEqual(float(row["throughput_min_tps_1w"]), min(values))
            self.assertAlmostEqual(float(row["throughput_max_tps_1w"]), max(values))

    def test_endpoint_tier_prices_and_stats_are_not_pooled(self) -> None:
        tiers = _rows(
            SOURCES / f"openrouter_endpoint_tier_signals_{FILE_DATE}.csv"
        )
        sol = [row for row in tiers if row["openrouter_model_id"] == "openai/gpt-5.6-sol"]
        self.assertGreaterEqual(len(sol), 5)
        openai_rows = {
            row["service_tier"]: row
            for row in sol
            if row["provider_slug"] == "openai"
        }
        self.assertEqual(set(openai_rows), {"default", "flex", "priority"})
        self.assertAlmostEqual(float(openai_rows["default"]["prompt_usd_per_mtoken"]), 5.0)
        self.assertAlmostEqual(float(openai_rows["flex"]["prompt_usd_per_mtoken"]), 2.5)
        self.assertAlmostEqual(float(openai_rows["priority"]["prompt_usd_per_mtoken"]), 10.0)
        self.assertEqual(
            int(float(openai_rows["default"]["high_context_min_prompt_tokens"])),
            272000,
        )
        self.assertAlmostEqual(
            float(openai_rows["default"]["high_context_prompt_usd_per_mtoken"]),
            10.0,
        )
        self.assertTrue(openai_rows["default"]["p50_throughput_tps_30m"])
        self.assertTrue(openai_rows["flex"]["p50_throughput_tps_30m"])
        self.assertEqual(
            openai_rows["priority"]["stats_source"],
            "endpoint.statsByTier.priority",
        )


if __name__ == "__main__":
    unittest.main()
