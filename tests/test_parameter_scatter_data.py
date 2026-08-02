from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "site" / "public" / "data" / "parameter-scatter.json"


class ParameterScatterDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(DATA.read_text())

    def test_universe_is_unique_and_complete(self) -> None:
        self.assertEqual(self.data["counts"], {"models": 10, "frontier": 10, "calibration": 0, "disclosedAnchors": 2})
        ids = [model["id"] for model in self.data["models"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids.count("checkpoint:moonshot:kimi-k3"), 1)
        self.assertEqual(sum("opus-4-7-4-8" in value for value in ids), 1)

    def test_every_point_has_valid_project_data(self) -> None:
        for model in self.data["models"]:
            self.assertRegex(model["releaseDate"], r"^20\d\d-\d\d-\d\d$")
            self.assertTrue(math.isfinite(model["parameterT"]))
            self.assertGreater(model["parameterT"], 0)
            self.assertIn(model["organizationGroup"], {"OpenAI", "Anthropic", "xAI", "Moonshot AI"})
            self.assertIsNotNone(model["forecastModelId"])

    def test_live_centers_and_identity_collapses_are_exact(self) -> None:
        by_forecast = {model["forecastModelId"]: model for model in self.data["models"] if model["forecastModelId"]}
        self.assertAlmostEqual(by_forecast["claude-fable-5"]["parameterT"], 4.548292155336898)
        self.assertEqual(by_forecast["claude-fable-5"]["shortName"], "Claude Fable 5")
        self.assertAlmostEqual(by_forecast["gpt-56-sol"]["parameterT"], 3.145668582841)
        self.assertEqual(by_forecast["claude-opus-5"]["shortName"], "Claude Opus 5")
        self.assertAlmostEqual(by_forecast["kimi-k3"]["parameterT"], 2.78)
        self.assertTrue(by_forecast["kimi-k3"]["lockedAnchor"])
        self.assertTrue(by_forecast["grok-45"]["lockedAnchor"])
        self.assertEqual(by_forecast["kimi-k3"]["organizationGroup"], "Moonshot AI")
        self.assertAlmostEqual(by_forecast["claude-fable-5"]["publicEstimateT"], 4.371847506331046)
        self.assertAlmostEqual(by_forecast["gpt-56-sol"]["publicEstimateT"], 3.2116911484449235)
        self.assertEqual(sum(model["publicEstimateT"] is not None for model in by_forecast.values()), 2)
        self.assertEqual(by_forecast["claude-opus-47-48-shared-base"]["name"], "Claude Opus 4.7 / 4.8 shared base")

    def test_every_model_retains_its_published_factor_inputs(self) -> None:
        factor_ids = {factor["id"] for factor in self.data["factors"]}
        self.assertEqual(factor_ids, {"aa", "eci", "price", "horizon", "compute", "ikp", "crowd"})
        self.assertTrue(all(set(model["factors"]) == factor_ids for model in self.data["models"]))

    def test_declared_source_hashes_reconcile(self) -> None:
        lookup = {"forecast": ROOT / "site" / "public" / "data" / "forecast-model.json"}
        for key, path in lookup.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), self.data["sources"][key]["sha256"])


if __name__ == "__main__":
    unittest.main()
