#!/usr/bin/env python3
"""Verify the immutable prospective frontier-parameter forecast commitment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from build_prospective_forecast_freeze import (
    ARTIFACT,
    DETACHED_DIGEST,
    ROOT,
    TARGET_IDS,
    canonical_json_bytes,
    geometric_mean,
)


ANONYMOUS_RESPONDENT = re.compile(r"^Respondent R\d{2}$")
ANONYMOUS_FORECAST_ID = re.compile(r"^r\d{2}-[a-z0-9-]+-\d{8}$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify(
    artifact_path: Path = ARTIFACT,
    *,
    detached_path: Path | None = None,
    check_repository: bool = False,
) -> list[str]:
    issues: list[str] = []
    detached_path = detached_path or artifact_path.with_name(DETACHED_DIGEST.name)
    if not artifact_path.is_file():
        return [f"missing artifact: {artifact_path}"]
    raw = artifact_path.read_bytes()
    try:
        artifact = json.loads(raw)
    except json.JSONDecodeError as error:
        return [f"invalid JSON: {error}"]

    integrity = artifact.get("artifact_integrity")
    if not isinstance(integrity, dict):
        issues.append("missing artifact_integrity")
    else:
        payload = dict(artifact)
        payload.pop("artifact_integrity", None)
        actual = sha256_bytes(canonical_json_bytes(payload))
        if actual != integrity.get("canonical_payload_sha256"):
            issues.append("canonical payload digest mismatch")

    if not detached_path.is_file():
        issues.append(f"missing detached digest: {detached_path}")
    else:
        parts = detached_path.read_text(encoding="ascii").strip().split()
        if len(parts) != 2 or parts[1] != artifact_path.name:
            issues.append("malformed detached digest")
        elif parts[0] != sha256_bytes(raw):
            issues.append("detached artifact digest mismatch")

    if artifact.get("status") != "LOCKED_PRE_DISCLOSURE":
        issues.append("freeze status is not LOCKED_PRE_DISCLOSURE")
    privacy = artifact.get("privacy_redaction", {})
    if privacy.get("name_to_id_mapping_retained") is not False:
        issues.append("privacy redaction does not forbid retaining the name-to-ID mapping")
    if privacy.get("numerical_values_changed") is not False:
        issues.append("privacy redaction does not preserve numerical values")
    if privacy.get("original_name_bearing_bytes_retained_in_current_tree") is not False:
        issues.append("privacy redaction does not declare removal of original named bytes")
    policy = artifact.get("evaluation_policy", {})
    if policy.get("post_outcome_refitting") != "FORBIDDEN":
        issues.append("post-outcome refitting is not explicitly forbidden")

    targets = artifact.get("targets", [])
    ids = [row.get("identity", {}).get("model_id") for row in targets]
    if ids != list(TARGET_IDS):
        issues.append(f"target identity/order mismatch: {ids}")
    for target in targets:
        identity = target.get("identity", {})
        forecast = target.get("forecast", {})
        factors = forecast.get("factor_values_t", {})
        weights = forecast.get("weights", {})
        evidence_weights = weights.get("evidence_only_effective_weights_fraction", {})
        final_weights = weights.get("final_effective_weights_fraction", {})
        if identity.get("outcome_at_freeze") is not None:
            issues.append(f"outcome leaked into freeze for {identity.get('model_id')}")
        try:
            evidence = math.exp(
                sum(evidence_weights[key] * math.log(factors[key]) for key in evidence_weights)
            )
            final = math.exp(
                sum(final_weights[key] * math.log(factors[key]) for key in final_weights)
            )
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"weight arithmetic unavailable for {identity.get('model_id')}: {error}")
            continue
        if not math.isclose(
            evidence, forecast.get("evidence_center_t", math.nan), rel_tol=1e-12, abs_tol=1e-12
        ):
            issues.append(f"evidence-center arithmetic mismatch for {identity.get('model_id')}")
        if not math.isclose(
            final, forecast.get("final_center_t", math.nan), rel_tol=1e-12, abs_tol=1e-12
        ):
            issues.append(f"final-center arithmetic mismatch for {identity.get('model_id')}")
        if not math.isclose(sum(final_weights.values()), 1.0, abs_tol=1e-12):
            issues.append(f"effective weights do not sum to one for {identity.get('model_id')}")
        for level, interval in forecast.get("empirical_intervals", {}).items():
            factor = interval["multiplicative_factor"]
            center = forecast["evidence_center_t"]
            if not math.isclose(interval["low_t"], center / factor, rel_tol=1e-12):
                issues.append(f"{level}% interval low mismatch for {identity.get('model_id')}")
            if not math.isclose(interval["high_t"], center * factor, rel_tol=1e-12):
                issues.append(f"{level}% interval high mismatch for {identity.get('model_id')}")

        crowd = target.get("crowd_pool", {})
        for record in crowd.get("records", []):
            if not ANONYMOUS_RESPONDENT.fullmatch(str(record.get("contributor", ""))):
                issues.append(
                    f"non-anonymous crowd respondent for {identity.get('model_id')}"
                )
            if not ANONYMOUS_FORECAST_ID.fullmatch(str(record.get("forecast_id", ""))):
                issues.append(
                    f"non-anonymous crowd forecast ID for {identity.get('model_id')}"
                )
        points = [row["pool_point_t"] for row in crowd.get("records", [])]
        if len(points) != crowd.get("n"):
            issues.append(f"crowd count mismatch for {identity.get('model_id')}")
        expected_crowd = geometric_mean(points) if points else None
        actual_crowd = crowd.get("geometric_center_t")
        if expected_crowd is None:
            if actual_crowd is not None:
                issues.append(f"unexpected crowd center for {identity.get('model_id')}")
        elif not math.isclose(expected_crowd, actual_crowd, rel_tol=1e-12):
            issues.append(f"crowd center mismatch for {identity.get('model_id')}")

    if check_repository:
        for category, records in artifact.get("path_hashes", {}).items():
            for record in records:
                path = ROOT / record["path"]
                if not path.is_file():
                    issues.append(f"{category}: missing repository file {record['path']}")
                    continue
                actual = sha256_bytes(path.read_bytes())
                if actual != record["sha256"]:
                    issues.append(f"{category}: repository hash mismatch {record['path']}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=ARTIFACT)
    parser.add_argument(
        "--check-repository",
        action="store_true",
        help="Also require every frozen source/output/code path to match the locked bytes.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    issues = verify(args.artifact, check_repository=args.check_repository)
    result: dict[str, Any] = {
        "artifact": str(args.artifact),
        "repository_checked": args.check_repository,
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['status']}: {len(issues)} issue(s)")
        for issue in issues:
            print(f"- {issue}")
    raise SystemExit(0 if not issues else 1)


if __name__ == "__main__":
    main()
