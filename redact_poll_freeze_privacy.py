#!/usr/bin/env python3
"""Create a privacy-redacted derivative of the locked forecast artifact.

The transformation removes poll identities only. Forecast values, pool points,
weights, intervals, target identities, and evaluation policy are required to be
byte-equivalent after respondent-only fields are normalized to sentinels. The
repository retains hashes of the prior artifact, but no name-to-ID mapping.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FREEZE_DIR = ROOT / "forecast_freezes/2026-07-31-frontier-parameters-v1"
ARTIFACT = FREEZE_DIR / "forecast_freeze.json"
DETACHED_DIGEST = FREEZE_DIR / "forecast_freeze.sha256"
REDACTION_RECORD = FREEZE_DIR / "privacy_redaction_2026-08-01.json"
LEDGER = ROOT / "sources/human_parameter_forecasts_2026-07-17.csv"
REDACTED_AT_UTC = "2026-08-01T00:00:00Z"
EXPECTED_PRIOR_ARTIFACT_SHA256 = (
    "0ed93f398ad2f80f8c7b76ce7d7add4b017ab45b53a0f14303d47613ae9ac785"
)
EXPECTED_PRIOR_CANONICAL_SHA256 = (
    "1d5cbe253bf441578a6f6a504ef52c10a4c6b18834201c8fb3e374637cf6ddc9"
)
ANONYMOUS_RESPONDENT = re.compile(r"^Respondent R\d{2}$")
ANONYMOUS_FORECAST_ID = re.compile(r"^r\d{2}-[a-z0-9-]+-\d{8}$")
REDACTED_FIELDS = ("forecast_id", "contributor", "provenance", "notes")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def read_ledger() -> list[dict[str, str]]:
    with LEDGER.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if not ANONYMOUS_RESPONDENT.fullmatch(row["contributor"]):
            raise ValueError(f"Non-anonymous respondent label: {row['contributor']!r}")
        if not ANONYMOUS_FORECAST_ID.fullmatch(row["forecast_id"]):
            raise ValueError(f"Non-anonymous forecast ID: {row['forecast_id']!r}")
    return rows


def signature(
    model: str,
    date: str,
    forecast_text: str,
    low_t: float | str,
    high_t: float | str,
    central_t: float | str | None,
) -> tuple[Any, ...]:
    central = None if central_t in (None, "") else float(central_t)
    return (
        model,
        date,
        forecast_text,
        float(low_t),
        float(high_t),
        central,
    )


def normalized_for_privacy_comparison(value: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(value)
    normalized.pop("artifact_integrity", None)
    normalized.pop("privacy_redaction", None)
    for target in normalized["targets"]:
        for record in target["crowd_pool"]["records"]:
            for field in REDACTED_FIELDS:
                record[field] = f"<{field}>"
    return normalized


def numerical_projection(value: dict[str, Any]) -> dict[str, Any]:
    output = []
    for target in value["targets"]:
        forecast = target["forecast"]
        crowd = target["crowd_pool"]
        output.append(
            {
                "model_id": target["identity"]["model_id"],
                "evidence_center_t": forecast["evidence_center_t"],
                "final_center_t": forecast["final_center_t"],
                "one_decimal_display_t": forecast["one_decimal_display_t"],
                "factor_values_t": forecast["factor_values_t"],
                "weights": forecast["weights"],
                "empirical_intervals": forecast["empirical_intervals"],
                "interval_calibration": forecast["interval_calibration"],
                "crowd_n": crowd["n"],
                "crowd_geometric_center_t": crowd["geometric_center_t"],
                "crowd_pool_points_t": [
                    record["pool_point_t"] for record in crowd["records"]
                ],
            }
        )
    return {"targets": output}


def validate_redacted(artifact: dict[str, Any]) -> None:
    redaction = artifact.get("privacy_redaction", {})
    if redaction.get("prior_artifact_sha256") != EXPECTED_PRIOR_ARTIFACT_SHA256:
        raise ValueError("Privacy chain does not cite the expected prior artifact")
    if redaction.get("name_to_id_mapping_retained") is not False:
        raise ValueError("Privacy metadata must state that no mapping is retained")
    for target in artifact["targets"]:
        records = target["crowd_pool"]["records"]
        for record in records:
            if not ANONYMOUS_RESPONDENT.fullmatch(record["contributor"]):
                raise ValueError("Freeze still contains a non-anonymous respondent")
            if not ANONYMOUS_FORECAST_ID.fullmatch(record["forecast_id"]):
                raise ValueError("Freeze still contains a name-bearing forecast ID")
        points = [record["pool_point_t"] for record in records]
        if len(points) != target["crowd_pool"]["n"]:
            raise ValueError("Freeze crowd count changed during privacy redaction")
        if points:
            center = math.exp(sum(math.log(point) for point in points) / len(points))
            if not math.isclose(
                center,
                target["crowd_pool"]["geometric_center_t"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("Freeze crowd center changed during privacy redaction")


def redact() -> tuple[dict[str, Any], dict[str, Any]]:
    original_bytes = ARTIFACT.read_bytes()
    artifact = json.loads(original_bytes)
    if "privacy_redaction" in artifact:
        validate_redacted(artifact)
        record = json.loads(REDACTION_RECORD.read_text(encoding="utf-8"))
        return artifact, record

    if sha256_bytes(original_bytes) != EXPECTED_PRIOR_ARTIFACT_SHA256:
        raise ValueError("Refusing to redact an unrecognized freeze artifact")
    prior_integrity = artifact.get("artifact_integrity", {})
    if (
        prior_integrity.get("canonical_payload_sha256")
        != EXPECTED_PRIOR_CANONICAL_SHA256
    ):
        raise ValueError("Prior canonical payload digest mismatch")

    ledger = read_ledger()
    ledger_index: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for row in ledger:
        key = signature(
            row["model"],
            row["date"],
            row["forecast_text"],
            row["low_t"],
            row["high_t"],
            row["central_t"],
        )
        ledger_index.setdefault(key, []).append(row)

    before_normalized = normalized_for_privacy_comparison(artifact)
    numerical_digest = sha256_bytes(canonical_json_bytes(numerical_projection(artifact)))
    for target in artifact["targets"]:
        model = target["identity"]["canonical_name"]
        for record in target["crowd_pool"]["records"]:
            key = signature(
                model,
                record["forecast_date"],
                record["forecast_text"],
                record["low_t"],
                record["high_t"],
                record["stated_central_t"],
            )
            matches = ledger_index.get(key, [])
            if not matches:
                raise ValueError("Expected an anonymous ledger match; found none")
            # Identical point forecasts can legitimately share every public
            # field in the signature. Both the frozen pool and source ledger
            # preserve submission order, so consume the matching queue once.
            row = matches.pop(0)
            record["forecast_id"] = row["forecast_id"]
            record["contributor"] = row["contributor"]
            record["provenance"] = row["provenance"]
            record["notes"] = row["notes"]

    artifact["privacy_redaction"] = {
        "schema": "poll-respondent-privacy-redaction/v1",
        "redacted_at_utc": REDACTED_AT_UTC,
        "reason": "User-requested removal of poll respondent identities",
        "prior_artifact_sha256": EXPECTED_PRIOR_ARTIFACT_SHA256,
        "prior_canonical_payload_sha256": EXPECTED_PRIOR_CANONICAL_SHA256,
        "redacted_fields": list(REDACTED_FIELDS),
        "stable_anonymous_ids_preserve_cross_target_pairing": True,
        "name_to_id_mapping_retained": False,
        "numerical_values_changed": False,
        "numerical_projection_sha256": numerical_digest,
        "original_name_bearing_bytes_retained_in_current_tree": False,
    }
    if normalized_for_privacy_comparison(artifact) != before_normalized:
        raise ValueError("Privacy redaction changed a non-identity field")
    if sha256_bytes(canonical_json_bytes(numerical_projection(artifact))) != numerical_digest:
        raise ValueError("Privacy redaction changed the numerical projection")

    artifact.pop("artifact_integrity", None)
    canonical_digest = sha256_bytes(canonical_json_bytes(artifact))
    artifact["artifact_integrity"] = {
        "algorithm": "sha256",
        "canonicalization": (
            "UTF-8 JSON of the artifact excluding artifact_integrity; keys sorted; "
            "separators ',' and ':'; ensure_ascii=false; allow_nan=false"
        ),
        "canonical_payload_sha256": canonical_digest,
    }
    validate_redacted(artifact)
    artifact_bytes = (json.dumps(artifact, indent=2, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    artifact_sha = sha256_bytes(artifact_bytes)
    record = {
        "schema": "poll-respondent-privacy-chain/v1",
        "redacted_at_utc": REDACTED_AT_UTC,
        "prior_artifact_sha256": EXPECTED_PRIOR_ARTIFACT_SHA256,
        "prior_canonical_payload_sha256": EXPECTED_PRIOR_CANONICAL_SHA256,
        "redacted_artifact_path": str(ARTIFACT.relative_to(ROOT)),
        "redacted_artifact_sha256": artifact_sha,
        "redacted_canonical_payload_sha256": canonical_digest,
        "numerical_projection_sha256": numerical_digest,
        "respondent_count": len({row["contributor"] for row in ledger}),
        "active_record_count": len(ledger),
        "name_to_id_mapping_retained": False,
        "transformation": (
            "Replaced respondent labels, name-bearing forecast IDs, personal provenance, "
            "and personal notes only; preserved all numerical and scoring fields."
        ),
    }
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_bytes(artifact_bytes)
    DETACHED_DIGEST.write_text(
        f"{artifact_sha}  {ARTIFACT.name}\n", encoding="ascii"
    )
    REDACTION_RECORD.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artifact, record


def main() -> None:
    artifact, record = redact()
    print(
        json.dumps(
            {
                "status": "PASS",
                "artifact": str(ARTIFACT.relative_to(ROOT)),
                "redaction_record": str(REDACTION_RECORD.relative_to(ROOT)),
                "targets": len(artifact["targets"]),
                "respondents": record["respondent_count"],
                "numerical_values_changed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
