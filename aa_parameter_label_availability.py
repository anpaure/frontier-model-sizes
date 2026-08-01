#!/usr/bin/env python3
"""Fail-closed parameter-label timing for AA-backed checkpoints.

Model release, API availability, model-card publication, and downloadable
weights are different events.  Chronological parameter regressions may train
on a checkpoint only after the parameter label used as the regression target
was public.  This module validates a small, primary-source timing ledger and
resolves it against the heterogeneous row schemas used by the AA audits.

The ledger affects chronological validation only.  It does not remove rows
from the current-as-of-2026-07-31 calibration fit, where every recorded label
was already public.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LEDGER_PATH = ROOT / "sources/aa_parameter_label_availability_2026-07-31.json"
SNAPSHOT_DATE = "2026-07-31"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_url(value: Any) -> str:
    text = str(value or "").strip().lower().rstrip("/")
    return re.sub(r"/(?:tree|blob)/(?:main|master)$", "", text)


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _row_name(row: dict[str, Any]) -> str:
    for field in (
        "model",
        "selected_name",
        "name",
        "canonical_model_name",
        "canonical_display_name",
        "source_model_name",
    ):
        if row.get(field) not in (None, ""):
            return str(row[field])
    return ""


def _row_release_date(row: dict[str, Any]) -> str:
    for field in ("release_date", "canonical_release_date"):
        if row.get(field) not in (None, ""):
            value = str(row[field])
            date.fromisoformat(value)
            return value
    raise ValueError("Parameter row lacks a release date")


def _row_parameters_b(row: dict[str, Any]) -> float | None:
    for field in (
        # Narrow canonical parameter-truth overlays preserve the source label
        # under one of these keys. Timing identity must validate the value
        # that the timing ledger actually observed, not the later canonical
        # reconciliation.
        "raw_parameter_total_b",
        "raw_parameters_b",
        "raw_total_b",
        "raw_total_parameters_b",
        "parameters_b",
        "total_b",
        "total_parameters_b",
        "actual_parameters_b",
    ):
        if row.get(field) not in (None, ""):
            return float(row[field])
    return None


def _row_weights_url(row: dict[str, Any]) -> str:
    for field in (
        "model_weights_source_url",
        "parameter_source",
        "parameter_value_source_url",
    ):
        value = _normalize_url(row.get(field))
        if value.startswith("https://huggingface.co/"):
            return value
    return ""


@lru_cache(maxsize=4)
def load_parameter_label_availability(
    path: Path = LEDGER_PATH,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported AA parameter-label timing schema")
    if payload.get("snapshot_date") != SNAPSHOT_DATE:
        raise ValueError("AA parameter-label timing snapshot is not pinned")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("AA parameter-label timing ledger is empty")

    ids: set[str] = set()
    names: set[str] = set()
    urls: set[str] = set()
    for record in records:
        record_id = str(record.get("record_id") or "")
        identity = record.get("identity") or {}
        timing = record.get("timing") or {}
        name = _normalize_name(identity.get("canonical_model_name"))
        url = _normalize_url(identity.get("model_weights_source_url"))
        if not record_id or not name or not url:
            raise ValueError("AA parameter-label timing record lacks stable identity")
        if record_id in ids or name in names or url in urls:
            raise ValueError("AA parameter-label timing ledger has duplicate identity")
        ids.add(record_id)
        names.add(name)
        urls.add(url)

        release = str(identity.get("aa_release_date") or "")
        label = str(timing.get("parameter_label_available_date") or "")
        weights = str(timing.get("weights_available_date") or "")
        try:
            release_date = date.fromisoformat(release)
            label_date = date.fromisoformat(label)
            weights_date = date.fromisoformat(weights)
        except ValueError as error:
            raise ValueError(
                f"AA parameter-label timing record {record_id} has malformed dates"
            ) from error
        if label_date < release_date:
            raise ValueError(
                f"AA parameter-label timing record {record_id} predates release"
            )
        if weights_date < label_date:
            raise ValueError(
                f"AA parameter-label timing record {record_id} has weights before label"
            )
        parameters = identity.get("total_parameters_b")
        if not isinstance(parameters, (int, float)) or parameters <= 0:
            raise ValueError(
                f"AA parameter-label timing record {record_id} lacks parameters"
            )
        accepted_parameters = identity.get(
            "accepted_total_parameters_b", [parameters]
        )
        if (
            not isinstance(accepted_parameters, list)
            or not accepted_parameters
            or any(
                not isinstance(value, (int, float)) or value <= 0
                for value in accepted_parameters
            )
        ):
            raise ValueError(
                f"AA parameter-label timing record {record_id} has invalid accepted parameters"
            )
        if not any(
            math.isclose(
                float(parameters), float(value), rel_tol=0.0, abs_tol=1e-9
            )
            for value in accepted_parameters
        ):
            raise ValueError(
                f"AA parameter-label timing record {record_id} excludes its canonical parameters"
            )
        if timing.get("parameter_label_basis") not in {
            "first_party_launch_model_card",
            "first_party_technical_report",
            "official_hugging_face_config_and_weights",
            "first_party_release_article",
        }:
            raise ValueError(
                f"AA parameter-label timing record {record_id} lacks a valid basis"
            )
        commit = str(timing.get("weights_commit") or "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(
                f"AA parameter-label timing record {record_id} lacks a weight commit"
            )

        evidence = record.get("local_evidence")
        if not isinstance(evidence, list) or len(evidence) < 3:
            raise ValueError(
                f"AA parameter-label timing record {record_id} lacks local evidence"
            )
        kinds: set[str] = set()
        evidence_by_kind: dict[str, Path] = {}
        for item in evidence:
            evidence_path = ROOT / str(item.get("path") or "")
            expected_hash = str(item.get("sha256") or "")
            kind = str(item.get("kind") or "")
            if kind in kinds:
                raise ValueError(
                    f"AA parameter-label timing record {record_id} duplicates evidence kind"
                )
            kinds.add(kind)
            evidence_by_kind[kind] = evidence_path
            if not evidence_path.is_file():
                raise ValueError(f"Missing parameter-label evidence {evidence_path}")
            if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                raise ValueError(f"Malformed evidence hash for {evidence_path}")
            if _sha256(evidence_path) != expected_hash:
                raise ValueError(f"Parameter-label evidence hash mismatch: {evidence_path}")

        required_kinds = {
            "hugging_face_model_api",
            "hugging_face_commit_history",
            "commit_pinned_config",
            "commit_pinned_model_card",
        }
        if not required_kinds.issubset(kinds):
            raise ValueError(
                f"AA parameter-label timing record {record_id} lacks required evidence kinds"
            )
        model_api = json.loads(
            evidence_by_kind["hugging_face_model_api"].read_text(encoding="utf-8")
        )
        expected_repo = str(identity["model_weights_source_url"]).split(
            "huggingface.co/", 1
        )[1]
        if model_api.get("id") != expected_repo:
            raise ValueError(f"Hugging Face repository mismatch for {record_id}")
        if model_api.get("createdAt") != timing.get("repository_created_at_utc"):
            raise ValueError(f"Hugging Face creation time mismatch for {record_id}")
        exact_total = timing.get("hugging_face_safetensors_total_parameters")
        if (
            not isinstance(exact_total, int)
            or model_api.get("safetensors", {}).get("total") != exact_total
        ):
            raise ValueError(f"Hugging Face tensor total mismatch for {record_id}")
        commits = json.loads(
            evidence_by_kind["hugging_face_commit_history"].read_text(
                encoding="utf-8"
            )
        )
        weight_commit = next(
            (item for item in commits if item.get("id") == commit), None
        )
        if weight_commit is None or weight_commit.get("date") != timing.get(
            "weights_available_at_utc"
        ):
            raise ValueError(f"Hugging Face weight timing mismatch for {record_id}")
        config = json.loads(
            evidence_by_kind["commit_pinned_config"].read_text(encoding="utf-8")
        )
        if not isinstance(config, dict) or not config:
            raise ValueError(f"Invalid commit-pinned config for {record_id}")
        match_tokens = timing.get("parameter_label_match_tokens")
        if not isinstance(match_tokens, list):
            raise ValueError(f"Missing parameter-label match tokens for {record_id}")
        if timing["parameter_label_basis"] in {
            "first_party_launch_model_card",
            "first_party_technical_report",
            "first_party_release_article",
        }:
            label_kinds = {
                "first_party_label_source",
                "first_party_launch_page",
            } & kinds
        else:
            label_kinds = {"commit_pinned_model_card"}
        label_text = "\n".join(
            evidence_by_kind[kind].read_text(encoding="utf-8", errors="ignore")
            for kind in sorted(label_kinds)
        ).casefold()
        if any(str(token).casefold() not in label_text for token in match_tokens):
            raise ValueError(f"Parameter-label text mismatch for {record_id}")
    return payload


def resolve_parameter_label_available_date(row: dict[str, Any]) -> str:
    """Return a provenance-aware label date, rejecting partial identity matches."""

    release = _row_release_date(row)
    explicit = str(row.get("parameter_label_available_date") or release)
    date.fromisoformat(explicit)
    normalized_name = _normalize_name(_row_name(row))
    weights_url = _row_weights_url(row)
    parameters = _row_parameters_b(row)

    matches = []
    for record in load_parameter_label_availability()["records"]:
        identity = record["identity"]
        aliases = {
            _normalize_name(identity["canonical_model_name"]),
            *(_normalize_name(value) for value in identity.get("aliases", [])),
        }
        record_url = _normalize_url(identity["model_weights_source_url"])
        if normalized_name in aliases or (weights_url and weights_url == record_url):
            matches.append(record)
    if not matches:
        return explicit
    if len(matches) != 1:
        raise ValueError(f"Ambiguous parameter-label timing match for {_row_name(row)!r}")

    record = matches[0]
    identity = record["identity"]
    record_id = record["record_id"]
    aliases = {
        _normalize_name(identity["canonical_model_name"]),
        *(_normalize_name(value) for value in identity.get("aliases", [])),
    }
    if normalized_name not in aliases:
        raise ValueError(f"Parameter-label timing name mismatch for {record_id}")
    if release != identity["aa_release_date"]:
        raise ValueError(f"Parameter-label timing release mismatch for {record_id}")
    accepted_parameters = identity.get(
        "accepted_total_parameters_b", [identity["total_parameters_b"]]
    )
    if parameters is None or not any(
        math.isclose(parameters, float(value), rel_tol=0.0, abs_tol=1e-9)
        for value in accepted_parameters
    ):
        raise ValueError(f"Parameter-label timing parameter mismatch for {record_id}")
    record_url = _normalize_url(identity["model_weights_source_url"])
    if weights_url and weights_url != record_url:
        raise ValueError(f"Parameter-label timing weights URL mismatch for {record_id}")

    label = str(record["timing"]["parameter_label_available_date"])
    return max(explicit, label)
