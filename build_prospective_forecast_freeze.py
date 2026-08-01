#!/usr/bin/env python3
"""Create a one-shot prospective commitment for hidden parameter forecasts.

The output is intentionally outside the ordinary regenerated-output directory.
Once created, this script refuses to overwrite it with different bytes.  A
byte-identical rebuild is allowed so tests can prove determinism.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
FREEZE_DIR = ROOT / "forecast_freezes/2026-07-31-frontier-parameters-v1"
ARTIFACT = FREEZE_DIR / "forecast_freeze.json"
DETACHED_DIGEST = FREEZE_DIR / "forecast_freeze.sha256"

LOCKED_AT_UTC = "2026-07-31T03:22:39Z"
REPOSITORY_HEAD_AT_LOCK = "abc33edca3ba1c0e1662cebe7f6c43bae0119ddf"
REPOSITORY_BRANCH_AT_LOCK = "agent/publish-frontier-parameter-model"

FORECAST_DATA = ROOT / "site/public/data/forecast-model.json"
UNCERTAINTY_DATA = ROOT / "site/public/data/predictive-uncertainty.json"
CROWD_LEDGER = ROOT / "sources/human_parameter_forecasts_2026-07-17.csv"

TARGET_IDS = ("claude-fable-5", "gpt-56-sol", "claude-opus-5")
TARGET_SCOPE = {
    "claude-fable-5": {
        "canonical_name": "Claude Fable 5",
        "provider": "Anthropic",
        "serving_fallback_excluded": "Claude Opus 4.8",
        "identity_note": "Score the named Fable 5 base checkpoint, not its fallback-enabled serving cascade or Mythos 5.",
    },
    "gpt-56-sol": {
        "canonical_name": "GPT-5.6 Sol",
        "provider": "OpenAI",
        "serving_fallback_excluded": None,
        "identity_note": "Score the named GPT-5.6 Sol base checkpoint; do not substitute Terra, Luna, GPT-5.5, or a later Sol revision.",
    },
    "claude-opus-5": {
        "canonical_name": "Claude Opus 5",
        "provider": "Anthropic",
        "serving_fallback_excluded": "Claude Opus 4.8",
        "identity_note": "Score the distinct Opus 5 base checkpoint, not the Opus 4.8 fallback or another Claude 5 checkpoint.",
    },
}

SOURCE_FILES = (
    "sources/human_parameter_forecasts_2026-07-17.csv",
    "sources/frontier_primary_evidence_2026-07-18.csv",
    "sources/claude_opus_5_evidence_2026-07-31.json",
    "sources/kimi_k3_release_evidence_2026-07-31.json",
    "sources/epoch_eci_benchmarks_2026-07-31.csv",
)

OUTPUT_FILES = (
    "site/public/data/forecast-model.json",
    "site/public/data/predictive-uncertainty.json",
    "regression_results.json",
    "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/frontier_parameter_model_crowd_50pct_2026-07-17.xlsx",
    "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/frontier_parameter_predictive_uncertainty_2026-07-18.json",
    "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/frontier_parameter_chronological_backtest_2026-07-17.json",
    "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/parameter_developer_vintage_sensitivity_2026-07-31.json",
    "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/eci_vintage_knowledge_residual_audit_2026-07-31.json",
)

CODE_FILES = (
    "build_horizon_informed_model.mjs",
    "generate_forecast_site_data.py",
    "analyze_parameter_predictive_uncertainty.py",
    "run_parameter_backtest.py",
    "frontier_target_signals.py",
    "site/app/page.tsx",
    "build_prospective_forecast_freeze.py",
    "verify_prospective_forecast_freeze.py",
    "tests/test_prospective_forecast_freeze.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("Geometric means require at least one positive value")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def active_crowd_rows(model: str) -> list[dict[str, Any]]:
    ledger = read_csv(CROWD_LEDGER)
    superseded = {row["supersedes"] for row in ledger if row["supersedes"]}
    rows = [
        row
        for row in ledger
        if row["model"] == model and row["forecast_id"] not in superseded
    ]
    output = []
    for row in rows:
        low = float(row["low_t"])
        high = float(row["high_t"])
        central = float(row["central_t"]) if row["central_t"] else None
        point = central if central is not None else math.sqrt(low * high)
        output.append(
            {
                "forecast_id": row["forecast_id"],
                "contributor": row["contributor"],
                "forecast_date": row["date"],
                "forecast_text": row["forecast_text"],
                "low_t": low,
                "high_t": high,
                "stated_central_t": central,
                "pool_point_t": float(point),
                "pool_point_rule": "stated central if supplied; otherwise geometric midpoint of bounds",
                "confidence": row["confidence"],
                "provenance": row["provenance"],
                "notes": row["notes"],
            }
        )
    return output


def weight_records(model: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    evidence_ids = [key for key in weights if key != "crowd"]
    available = [
        key
        for key in evidence_ids
        if model["factors"].get(key) is not None and weights[key] > 0
    ]
    requested_evidence = sum(max(0.0, weights[key]) for key in evidence_ids)
    available_evidence = sum(weights[key] for key in available)
    raw = {
        key: requested_evidence * weights[key] / available_evidence
        for key in available
    }
    if model["factors"].get("crowd") is not None and weights["crowd"] > 0:
        raw["crowd"] = weights["crowd"]
    total = sum(raw.values())
    effective = {key: value / total for key, value in raw.items()}
    evidence_total = sum(effective.get(key, 0.0) for key in evidence_ids)
    evidence_effective = {
        key: effective[key] / evidence_total for key in available
    }
    return {
        "requested_default_weights_percent": dict(weights),
        "model_raw_weights_after_missing_signal_reallocation": raw,
        "final_effective_weights_fraction": effective,
        "evidence_only_effective_weights_fraction": evidence_effective,
    }


def path_manifest(paths: tuple[str, ...], role: str) -> list[dict[str, str]]:
    output = []
    for relative in paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        output.append({"path": relative, "sha256": sha256(path), "role": role})
    return output


def target_record(
    model: dict[str, Any], uncertainty: dict[str, Any], weights: dict[str, float]
) -> dict[str, Any]:
    identity = TARGET_SCOPE[model["id"]]
    crowd_rows = active_crowd_rows(model["name"])
    crowd_center = (
        geometric_mean([row["pool_point_t"] for row in crowd_rows])
        if crowd_rows
        else None
    )
    if crowd_center is not None and not math.isclose(
        crowd_center, model["factors"]["crowd"], rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"Crowd center mismatch for {model['name']}")
    weight_block = weight_records(model, weights)
    intervals = {
        level: {
            **uncertainty["intervals"][level],
            "center_basis": "evidence_center_t",
        }
        for level in ("50", "80", "90")
    }
    return {
        "identity": {
            "model_id": model["id"],
            "canonical_name": identity["canonical_name"],
            "provider": identity["provider"],
            "release_date": model["releaseDate"],
            "checkpoint_scope": "the named base checkpoint only",
            "serving_fallback_excluded": identity["serving_fallback_excluded"],
            "identity_note": identity["identity_note"],
            "parameter_status_at_freeze": "undisclosed",
            "outcome_at_freeze": None,
        },
        "forecast": {
            "unit": "trillion total parameters",
            "evidence_center_t": model["currentEvidenceT"],
            "final_center_t": model["currentFinalT"],
            "one_decimal_display_t": float(f"{model['currentFinalT']:.1f}"),
            "factor_values_t": dict(model["factors"]),
            "weights": weight_block,
            "empirical_intervals": intervals,
            "interval_calibration": {
                "cohort": uncertainty["calibration_cohort"],
                "calibration_rows": uncertainty["calibration_rows"],
                "calibration_families": uncertainty["calibration_families"],
                "calibration_developers": uncertainty["calibration_developers"],
                "target_chronological_calibration_rows": uncertainty[
                    "target_chronological_calibration_rows"
                ],
                "target_chronological_calibration_developers": uncertainty[
                    "target_chronological_calibration_developers"
                ],
            },
        },
        "crowd_pool": {
            "pooled_into_final": bool(model["crowd"]["pooled"]),
            "records": crowd_rows,
            "n": len(crowd_rows),
            "geometric_center_t": crowd_center,
            "policy": "equal weight per active contributor record in log space",
        },
    }


def build_payload() -> dict[str, Any]:
    forecast = json.loads(FORECAST_DATA.read_text(encoding="utf-8"))
    uncertainty = json.loads(UNCERTAINTY_DATA.read_text(encoding="utf-8"))
    models = {row["id"]: row for row in forecast["models"]}
    intervals = {row["model_id"]: row for row in uncertainty["targets"]}
    if set(TARGET_IDS) - models.keys() or set(TARGET_IDS) - intervals.keys():
        raise ValueError("One or more locked targets are missing from generated data")

    path_hashes = {
        "source_files": path_manifest(SOURCE_FILES, "source"),
        "generated_outputs": path_manifest(OUTPUT_FILES, "generated output"),
        "forecast_code": path_manifest(CODE_FILES, "code"),
    }
    targets = [
        target_record(models[model_id], intervals[model_id], forecast["defaultWeights"])
        for model_id in TARGET_IDS
    ]
    return {
        "schema": "frontier-parameter-prospective-freeze/v1",
        "freeze_id": "frontier-parameters-2026-07-31-v1",
        "status": "LOCKED_PRE_DISCLOSURE",
        "locked_at_utc": LOCKED_AT_UTC,
        "forecast_snapshot_date": forecast["snapshotDate"],
        "repository_state": {
            "git_head": REPOSITORY_HEAD_AT_LOCK,
            "branch": REPOSITORY_BRANCH_AT_LOCK,
            "worktree_dirty_at_lock": True,
            "note": "File-level hashes, not the Git head alone, identify the exact dirty-worktree forecast state.",
        },
        "targets": targets,
        "live_policy": {
            "combination": forecast["method"]["combination"],
            "anchors": forecast["method"]["anchors"],
            "crowd_calibration": uncertainty["method"]["crowd_policy"],
            "interval_method": uncertainty["method"],
            "zero_weight_diagnostic": {
                "name": "archive-vintage knowledge-residual challenger",
                "incremental_live_weight": 0.0,
                "reason": "predeclared archive-vintage coverage and target-coverage gates failed",
            },
        },
        "evaluation_policy": {
            "post_outcome_refitting": "FORBIDDEN",
            "locked_fields": [
                "target identities",
                "evidence and final centers",
                "factor values and weights",
                "crowd membership and pool points",
                "empirical interval factors and bounds",
                "scoring and disclosure-resolution rules",
            ],
            "primary_outcome": (
                "officially disclosed unique trainable total parameter count of the "
                "named base checkpoint, expressed in trillions"
            ),
            "moe_rule": (
                "Use total parameters, not per-token active parameters, for the primary "
                "score. Record active parameters only as a secondary fact."
            ),
            "fallback_and_cascade_rule": (
                "Exclude serving fallbacks, routers, tool models, and cascade components. "
                "If the provider discloses only a multi-model system total and does not "
                "identify the named base checkpoint, leave the target unresolved."
            ),
            "source_priority": [
                "provider technical report or model/system card",
                "provider model repository or architecture configuration",
                "provider-authored announcement with an explicit exact count",
            ],
            "unacceptable_outcome_sources": [
                "rumor or anonymous claim",
                "third-party estimate",
                "benchmark-implied estimate",
                "API price or throughput inference",
            ],
            "rounded_or_range_disclosure": (
                "Do not assign an exact point score from an approximate or bounded "
                "disclosure. Report range compatibility separately and wait for an exact "
                "first-party total before point scoring."
            ),
            "primary_point_forecast": "final_center_t",
            "secondary_point_forecast": "evidence_center_t",
            "point_metrics": {
                "absolute_log10_error": "abs(log10(forecast_t / actual_t))",
                "multiplicative_error": "max(forecast_t / actual_t, actual_t / forecast_t)",
            },
            "interval_metrics": (
                "For each frozen 50%, 80%, and 90% interval, report whether the exact "
                "actual total lies inside the inclusive [low_t, high_t] bounds."
            ),
            "multi_target_summary": {
                "mean_absolute_log10_error": "arithmetic mean across resolved targets",
                "geometric_mean_multiplicative_error": (
                    "10 ** mean_absolute_log10_error"
                ),
                "no_target_weighting": "each resolved target receives equal weight",
            },
            "amendment_rule": (
                "Before any qualifying outcome is known, a factual-transcription correction "
                "may be recorded only in a new append-only amendment that cites this freeze "
                "digest and preserves the original. After any qualifying outcome is known, "
                "no forecast, identity, weight, crowd record, interval, or scoring change is permitted."
            ),
            "evaluation_record_rule": (
                "Evaluation must be a new append-only artifact that cites the frozen payload "
                "digest and hashes the qualifying first-party disclosure. Never regenerate "
                "or replace this freeze with outcomes included."
            ),
        },
        "path_hashes": path_hashes,
        "commitment_note": (
            "SHA-256 detects mutation but is not an external timestamp by itself. Commit and "
            "push this freeze and detached digest before any qualifying disclosure."
        ),
    }


def render_artifact() -> tuple[bytes, bytes]:
    payload = build_payload()
    payload_digest = sha256_bytes(canonical_json_bytes(payload))
    artifact = {
        **payload,
        "artifact_integrity": {
            "algorithm": "sha256",
            "canonicalization": (
                "UTF-8 JSON of the artifact excluding artifact_integrity; keys sorted; "
                "separators ',' and ':'; ensure_ascii=false; allow_nan=false"
            ),
            "canonical_payload_sha256": payload_digest,
        },
    }
    artifact_bytes = (json.dumps(artifact, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    detached = f"{sha256_bytes(artifact_bytes)}  {ARTIFACT.name}\n".encode("ascii")
    return artifact_bytes, detached


def write_once(path: Path, value: bytes) -> None:
    if path.exists() and path.read_bytes() != value:
        raise RuntimeError(
            f"Refusing to overwrite immutable forecast freeze with different bytes: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def main() -> None:
    artifact, detached = render_artifact()
    write_once(ARTIFACT, artifact)
    write_once(DETACHED_DIGEST, detached)
    print(f"Frozen {ARTIFACT}")
    print(f"Detached digest {DETACHED_DIGEST}")


if __name__ == "__main__":
    main()
