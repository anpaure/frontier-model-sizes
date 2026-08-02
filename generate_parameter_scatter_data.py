#!/usr/bin/env python3
"""Build the Epoch-layout scatter contract from the ten published site models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FORECAST = ROOT / "site" / "public" / "data" / "forecast-model.json"
OUTPUT = ROOT / "site" / "public" / "data" / "parameter-scatter.json"

CHECKPOINTS = {
    "claude-fable-5": "checkpoint:anthropic:claude-fable-5",
    "gpt-56-sol": "checkpoint:openai:gpt-5-6-sol",
    "kimi-k3": "checkpoint:moonshot:kimi-k3",
    "claude-opus-47-48-shared-base": "weight-identity:anthropic:claude-opus-4-7-4-8",
    "gpt-55": "checkpoint:openai:gpt-5-5",
    "gpt-56-terra": "checkpoint:openai:gpt-5-6-terra",
    "claude-sonnet-5": "checkpoint:anthropic:claude-sonnet-5",
    "gpt-56-luna": "checkpoint:openai:gpt-5-6-luna",
    "grok-45": "checkpoint:xai:grok-4-5",
    "claude-opus-5": "checkpoint:anthropic:claude-opus-5",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def organization_group(value: str) -> str:
    lowered = value.lower()
    if "openai" in lowered:
        return "OpenAI"
    if "anthropic" in lowered:
        return "Anthropic"
    if "xai" in lowered or "spacex" in lowered:
        return "xAI"
    if "moonshot" in lowered:
        return "Moonshot AI"
    return "Other"


def build() -> dict:
    forecast = json.loads(FORECAST.read_text())
    models = []
    for model in forecast["models"]:
        models.append(
            {
                "id": CHECKPOINTS[model["id"]],
                "forecastModelId": model["id"],
                "name": model["name"],
                "shortName": model["shortName"],
                "organization": model["provider"],
                "organizationGroup": organization_group(model["provider"]),
                "releaseDate": model["releaseDate"],
                "parameterT": model["currentFinalT"],
                "parameterKind": "disclosed" if model["lockedAnchor"] else "forecast",
                "parameterSource": "Project disclosure ledger" if model["lockedAnchor"] else "Audited final forecast",
                "parameterSourceUrl": None,
                "lockedAnchor": model["lockedAnchor"],
                "currentFinalT": model["currentFinalT"],
                "publicEstimateT": model["factors"].get("crowd"),
                "factors": model["factors"],
                "factorWeights": forecast["defaultWeights"],
                "signals": {
                    "aaScore": model["aaScore"],
                    "eciScore": model["eciScore"],
                    "noCotMinutes": None,
                    "metrP50Minutes": None,
                    "trainingComputeFlop": None,
                    "activeParametersB": 104.2 if model["id"] == "kimi-k3" else None,
                },
                "architecture": {"moe": True, "reasoning": True},
                "backtest": None,
                "labelPriority": 100 + (10 if model["lockedAnchor"] else 0),
                "roles": ["disclosed anchor" if model["lockedAnchor"] else "frontier forecast"],
            }
        )

    models.sort(key=lambda item: (item["releaseDate"], item["name"]))
    if len(models) != 10 or len({model["id"] for model in models}) != 10:
        raise AssertionError("The published scatter must contain exactly ten unique frontier/base models")

    payload = {
        "schemaVersion": 2,
        "snapshotDate": forecast["snapshotDate"],
        "title": "Frontier estimates",
        "unit": "trillion parameters",
        "parameterPolicy": "Every plotted value, identity, date, and contribution comes from the project's published forecast data. External visual references contribute layout only.",
        "counts": {
            "models": 10,
            "frontier": 10,
            "calibration": 0,
            "disclosedAnchors": sum(model["parameterKind"] == "disclosed" for model in models),
        },
        "organizationGroups": [
            {"id": "OpenAI", "color": "#E03D90"},
            {"id": "Anthropic", "color": "#6A3ECB"},
            {"id": "xAI", "color": "#0058DC"},
            {"id": "Moonshot AI", "color": "#2AA74B"},
        ],
        "factors": forecast["factors"],
        "defaultWeights": forecast["defaultWeights"],
        "presets": forecast["presets"],
        "models": models,
        "sources": {
            "forecast": {"name": FORECAST.name, "sha256": sha256(FORECAST)},
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "models": 10, "anchors": payload["counts"]["disclosedAnchors"]}, indent=2))
    return payload


if __name__ == "__main__":
    build()
