#!/usr/bin/env python3
"""Validate and apply primary-source corrections to AA calibration metadata.

The frozen Artificial Analysis snapshot is a source record and must never be
silently rewritten when a model publisher releases newer metadata.  This
module applies a small, hash-pinned overlay only when every expected AA value
and identity field still matches.  Downstream calibration tables therefore
gain the corrected eligibility while retaining both the raw AA record and the
explicit primary-source correction.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any

from aa_parameter_label_availability import (
    resolve_parameter_label_available_date,
)
from aa_score_availability import aa_score_available_date


ROOT = Path(__file__).resolve().parent
OVERRIDES_PATH = (
    ROOT / "sources/aa_calibration_primary_overrides_2026-07-31.json"
)
SNAPSHOT_DATE = "2026-07-31"

ALLOWED_REPLACEMENT_FIELDS = {
    "is_open_weights",
    "open_source_categorization",
    "parameters_b",
    "active_parameters_b",
    "model_weights_source_url",
    "license_name",
    "license_url",
    "commercial_allowed",
}


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def load_calibration_overrides(
    path: Path = OVERRIDES_PATH,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported AA calibration override schema")
    if payload.get("snapshot_date") != SNAPSHOT_DATE:
        raise ValueError("AA calibration override snapshot date is not pinned")
    overrides = payload.get("overrides")
    if not isinstance(overrides, list) or not overrides:
        raise ValueError("AA calibration override inventory is empty")

    ids: set[str] = set()
    slugs: set[str] = set()
    model_ids: set[str] = set()
    for override in overrides:
        override_id = override.get("override_id")
        identity = override.get("identity") or {}
        expected = override.get("expected_raw") or {}
        replacement = override.get("replacement") or {}
        primary = override.get("primary_source") or {}
        reconciliation = override.get("identity_reconciliation") or {}
        lineage = override.get("lineage") or {}
        slug = identity.get("aa_slug")
        model_id = identity.get("aa_model_id")
        if not override_id or not slug or not model_id:
            raise ValueError("AA calibration override lacks a stable identity")
        if override_id in ids or slug in slugs or model_id in model_ids:
            raise ValueError("AA calibration overrides contain a duplicate identity")
        ids.add(override_id)
        slugs.add(slug)
        model_ids.add(model_id)
        if not expected or not replacement:
            raise ValueError(f"Override {override_id} lacks expected/replacement fields")
        unexpected = set(replacement) - ALLOWED_REPLACEMENT_FIELDS
        if unexpected:
            raise ValueError(
                f"Override {override_id} changes unsupported fields: {sorted(unexpected)}"
            )
        if not set(replacement).issuperset(expected):
            raise ValueError(
                f"Override {override_id} must explicitly replace every expected field"
            )
        if primary.get("source_type") != "official_open_weight_model_repository":
            raise ValueError(f"Override {override_id} is not backed by an official repo")
        if not str(primary.get("source_url", "")).startswith("https://huggingface.co/"):
            raise ValueError(f"Override {override_id} lacks an official HF source URL")
        if primary.get("model_card_total_parameters_b") != expected.get("parameters_b"):
            raise ValueError(f"Override {override_id} total-parameter evidence disagrees")
        if primary.get("model_card_active_parameters_b") != expected.get(
            "active_parameters_b"
        ):
            raise ValueError(f"Override {override_id} active-parameter evidence disagrees")
        if primary.get("weights_publicly_downloadable") is not True:
            raise ValueError(f"Override {override_id} does not establish public weights")
        local_evidence = primary.get("local_evidence")
        if not isinstance(local_evidence, list) or not local_evidence:
            raise ValueError(f"Override {override_id} lacks vendored source evidence")
        evidence_by_kind: dict[str, tuple[Path, bytes]] = {}
        for evidence in local_evidence:
            kind = str(evidence.get("kind", ""))
            relative_path = str(evidence.get("path", ""))
            expected_hash = str(evidence.get("sha256", ""))
            evidence_path = (ROOT / relative_path).resolve()
            evidence_root = (ROOT / "sources/aa_calibration_evidence").resolve()
            if not kind or kind in evidence_by_kind:
                raise ValueError(f"Override {override_id} has duplicate/empty evidence kind")
            if evidence_root not in evidence_path.parents or not evidence_path.is_file():
                raise ValueError(
                    f"Override {override_id} evidence is missing or outside its frozen directory"
                )
            raw_evidence = evidence_path.read_bytes()
            actual_hash = hashlib.sha256(raw_evidence).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Override {override_id} evidence hash mismatch for {relative_path}"
                )
            evidence_by_kind[kind] = (evidence_path, raw_evidence)
        required_evidence = {
            "commit_pinned_model_card",
            "commit_pinned_hugging_face_model_api",
        }
        if not required_evidence.issubset(evidence_by_kind):
            raise ValueError(f"Override {override_id} lacks required frozen HF evidence")
        repository_commit = str(primary.get("repository_commit", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", repository_commit):
            raise ValueError(f"Override {override_id} lacks a pinned repository commit")
        if repository_commit not in str(primary.get("source_url", "")):
            raise ValueError(f"Override {override_id} source URL is not commit-pinned")
        exact_tensor_count = primary.get("hugging_face_safetensors_total_parameters")
        if not isinstance(exact_tensor_count, int) or exact_tensor_count <= 0:
            raise ValueError(f"Override {override_id} lacks an exact tensor count")
        calibration_total_b = primary.get("calibration_total_parameters_b")
        if not isinstance(calibration_total_b, (int, float)) or calibration_total_b <= 0:
            raise ValueError(f"Override {override_id} lacks a calibration parameter value")
        if not math.isclose(
            float(replacement["parameters_b"]),
            float(calibration_total_b),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Override {override_id} replacement disagrees with its calibration value"
            )
        value_basis = primary.get("parameter_value_basis")
        if value_basis == "exact_safetensors_tensor_count":
            if not math.isclose(
                float(calibration_total_b),
                exact_tensor_count / 1_000_000_000,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Override {override_id} exact tensor count does not reproduce parameters_b"
                )
        elif value_basis == "publisher_disclosed_rounded_total":
            if not math.isclose(
                float(calibration_total_b),
                float(primary["model_card_total_parameters_b"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"Override {override_id} disclosed total does not reproduce parameters_b"
                )
        else:
            raise ValueError(f"Override {override_id} lacks a parameter-value basis")
        architecture_class = primary.get("architecture_class")
        if architecture_class not in {"dense", "sparse_moe"}:
            raise ValueError(f"Override {override_id} lacks an architecture classification")
        if architecture_class == "dense" and replacement.get("active_parameters_b") is not None:
            raise ValueError(
                f"Override {override_id} must not invent a separate dense active count"
            )
        model_card_raw = evidence_by_kind["commit_pinned_model_card"][1]
        if hashlib.sha256(model_card_raw).hexdigest() != str(
            primary.get("model_card_sha256", "")
        ):
            raise ValueError(f"Override {override_id} local model-card hash disagrees")
        # Hugging Face's raw Markdown response omits a final newline. Keep its
        # body hash as an explicitly named alternate, while model_card_sha256
        # always means the exact vendored bytes used by offline builds.
        if hashlib.sha256(model_card_raw.removesuffix(b"\n")).hexdigest() != str(
            primary.get("remote_model_card_body_sha256_without_trailing_newline", "")
        ):
            raise ValueError(
                f"Override {override_id} remote model-card body hash disagrees"
            )
        api_payload = json.loads(
            evidence_by_kind["commit_pinned_hugging_face_model_api"][1]
        )
        if (
            api_payload.get("id") != identity.get("hugging_face_repository_id")
            or api_payload.get("sha") != repository_commit
            or api_payload.get("createdAt") != primary.get("repository_created_at_utc")
            or api_payload.get("lastModified")
            != primary.get("repository_last_modified_at_utc")
            or api_payload.get("gated") is not False
            or (api_payload.get("safetensors") or {}).get("total")
            != exact_tensor_count
        ):
            raise ValueError(f"Override {override_id} frozen HF metadata disagrees")
        if architecture_class == "dense":
            if "commit_pinned_config" not in evidence_by_kind:
                raise ValueError(f"Override {override_id} lacks its frozen dense config")
            config_path, config_raw = evidence_by_kind["commit_pinned_config"]
            if hashlib.sha256(config_raw).hexdigest() != primary.get("config_sha256"):
                raise ValueError(f"Override {override_id} config hash disagrees")
            config = json.loads(config_raw)
            expert_fields = {
                key
                for key in config
                if any(token in key.lower() for token in ("expert", "router", "moe"))
            }
            if expert_fields:
                raise ValueError(
                    f"Override {override_id} dense config has expert fields: {expert_fields}"
                )
            card_data = api_payload.get("cardData") or {}
            if card_data.get("license") != "apache-2.0":
                raise ValueError(f"Override {override_id} Apache license is not reproduced")
            if lineage.get("base_model_id") not in (card_data.get("base_model") or []):
                raise ValueError(f"Override {override_id} base lineage is not reproduced")
        else:
            card_tags = set((api_payload.get("cardData") or {}).get("tags") or [])
            if not {"mixture-of-experts", "moe"}.intersection(card_tags):
                raise ValueError(f"Override {override_id} sparse-MoE tag is not reproduced")
        if reconciliation.get("status") != "exact_checkpoint_identity":
            raise ValueError(f"Override {override_id} has unresolved checkpoint identity")
        label_date = reconciliation.get("parameter_label_available_date")
        if not label_date:
            raise ValueError(f"Override {override_id} lacks parameter-label availability")
        try:
            date.fromisoformat(label_date)
            date.fromisoformat(identity["aa_release_date"])
        except ValueError as error:
            raise ValueError(
                f"Override {override_id} has a malformed availability date"
            ) from error
        if not str(primary.get("weights_available_at_utc", "")).startswith(label_date):
            raise ValueError(
                f"Override {override_id} availability date is not pinned to weight evidence"
            )
        availability_basis = primary.get("weights_availability_basis")
        if availability_basis not in {"repository_creation", "weights_upload_commit"}:
            raise ValueError(f"Override {override_id} lacks a weight-availability basis")
        if availability_basis == "repository_creation" and not str(
            primary.get("repository_created_at_utc", "")
        ).startswith(label_date):
            raise ValueError(
                f"Override {override_id} repository-creation availability is inconsistent"
            )
        if availability_basis == "weights_upload_commit" and not re.fullmatch(
            r"[0-9a-f]{40}", str(primary.get("weights_commit", ""))
        ):
            raise ValueError(f"Override {override_id} lacks a pinned weight-upload commit")
        if availability_basis == "weights_upload_commit":
            if "hugging_face_commit_history" not in evidence_by_kind:
                raise ValueError(f"Override {override_id} lacks frozen commit history")
            commits = json.loads(evidence_by_kind["hugging_face_commit_history"][1])
            weight_commit = next(
                (
                    commit
                    for commit in commits
                    if commit.get("id") == primary.get("weights_commit")
                ),
                None,
            )
            if (
                weight_commit is None
                or weight_commit.get("date") != primary.get("weights_available_at_utc")
                or "weight" not in str(weight_commit.get("title", "")).lower()
            ):
                raise ValueError(f"Override {override_id} weight-upload date is not reproduced")
        if lineage.get("lineage_class") not in {
            "independent_pretrain",
            "same_base_posttrain",
        }:
            raise ValueError(f"Override {override_id} lacks a lineage classification")
        if lineage.get("lineage_class") == "same_base_posttrain" and not lineage.get(
            "base_model_id"
        ):
            raise ValueError(f"Override {override_id} lacks its post-training base")
    return payload


def apply_calibration_overrides(
    rows: list[dict[str, Any]],
    path: Path = OVERRIDES_PATH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return copied rows plus a deterministic audit record for each override."""

    payload = load_calibration_overrides(path)
    output = [dict(row) for row in rows]
    by_slug: dict[str, list[dict[str, Any]]] = {}
    for row in output:
        by_slug.setdefault(str(row.get("slug", "")), []).append(row)

    audit: list[dict[str, Any]] = []
    for override in payload["overrides"]:
        identity = override["identity"]
        matches = by_slug.get(identity["aa_slug"], [])
        if len(matches) != 1:
            raise ValueError(
                f"AA override {override['override_id']} resolved to {len(matches)} rows"
            )
        row = matches[0]
        identity_expectations = {
            "model_id": identity["aa_model_id"],
            "slug": identity["aa_slug"],
            "name": identity["aa_name"],
            "release_date": identity["aa_release_date"],
        }
        for field, expected in identity_expectations.items():
            if row.get(field) != expected:
                raise ValueError(
                    f"AA override {override['override_id']} identity mismatch for {field}: "
                    f"{row.get(field)!r} != {expected!r}"
                )
        for field, expected in override["expected_raw"].items():
            if row.get(field) != expected:
                raise ValueError(
                    f"AA override {override['override_id']} stale expected value for {field}: "
                    f"{row.get(field)!r} != {expected!r}"
                )

        replacement = override["replacement"]
        changed_fields = sorted(
            field for field, value in replacement.items() if row.get(field) != value
        )
        row.update(replacement)
        source_record_sha256 = hashlib.sha256(canonical_json(override)).hexdigest()
        row.update(
            {
                "calibration_override_id": override["override_id"],
                "calibration_override_fields": " | ".join(changed_fields),
                "calibration_override_source_url": override["primary_source"][
                    "source_url"
                ],
                "calibration_override_source_record_sha256": source_record_sha256,
                "parameter_label_available_date": override[
                    "identity_reconciliation"
                ]["parameter_label_available_date"],
                "architecture_class": override["primary_source"][
                    "architecture_class"
                ],
                "exact_tensor_parameters": override["primary_source"][
                    "hugging_face_safetensors_total_parameters"
                ],
                "lineage_class": override["lineage"]["lineage_class"],
                "base_model_id": override["lineage"].get("base_model_id") or "",
                "lineage_family_id": override["lineage"].get("family_balance_key")
                or override["lineage"].get("base_model_id")
                or identity["hugging_face_repository_id"],
            }
        )
        audit.append(
            {
                "override_id": override["override_id"],
                "slug": row["slug"],
                "model_id": row["model_id"],
                "changed_fields": changed_fields,
                "source_url": override["primary_source"]["source_url"],
                "source_record_sha256": source_record_sha256,
                "parameter_label_available_date": override[
                    "identity_reconciliation"
                ]["parameter_label_available_date"],
                "architecture_class": override["primary_source"][
                    "architecture_class"
                ],
                "exact_tensor_parameters": override["primary_source"][
                    "hugging_face_safetensors_total_parameters"
                ],
                "lineage_class": override["lineage"]["lineage_class"],
                "base_model_id": override["lineage"].get("base_model_id"),
                "lineage_family_id": override["lineage"].get("family_balance_key")
                or override["lineage"].get("base_model_id")
                or identity["hugging_face_repository_id"],
            }
        )
    return output, audit


def parameter_training_eligibility_date(row: dict[str, Any]) -> str:
    """Earliest date an AA score/parameter pair may enter a training fold."""

    release_date = str(row["release_date"])
    label_date = resolve_parameter_label_available_date(row)
    is_aa_row = any(
        field in row
        for field in (
            "aa_slug",
            "selected_slug",
            "aa_score_available_date",
            "aa_score_availability_verified",
            "aa_score",
            "intelligence_index",
        )
    )
    score_date = aa_score_available_date(row) if is_aa_row else release_date
    # ISO day dates compare lexicographically, but parse both so malformed
    # provenance cannot silently pass a chronological gate.
    date.fromisoformat(release_date)
    date.fromisoformat(label_date)
    date.fromisoformat(score_date)
    return max(release_date, label_date, score_date)


def parameter_label_available_before(
    row: dict[str, Any], prediction_date: str
) -> bool:
    """True only when both the AA score and parameter label were public."""

    date.fromisoformat(prediction_date)
    return parameter_training_eligibility_date(row) < prediction_date
