#!/usr/bin/env python3
"""Apply the narrow, hash-verified open-model parameter truth overlay."""

from __future__ import annotations

import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LEDGER_PATH = ROOT / "sources/open_model_parameter_truth_reconciliation_2026-07-31.json"


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


@lru_cache(maxsize=1)
def load_parameter_truth() -> dict[str, Any]:
    payload = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0" or payload.get("snapshot_date") != "2026-07-31":
        raise ValueError("Unsupported open-model parameter-truth ledger")
    aliases: dict[str, str] = {}
    for source in payload.get("source_files", []):
        path = ROOT / source["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Parameter-truth source hash mismatch: {source['path']}")
    for record in payload.get("records", []):
        if record["canonical_total_parameters_b"] <= 0:
            raise ValueError("Canonical parameter truth must be positive")
        for alias in record["aliases"]:
            key = normalize_name(alias)
            if not key:
                raise ValueError(f"Empty parameter-truth alias: {alias}")
            existing = aliases.get(key)
            if existing is not None and existing != record["truth_id"]:
                raise ValueError(f"Duplicate parameter-truth alias: {alias}")
            aliases[key] = record["truth_id"]
    return payload


@lru_cache(maxsize=1)
def _by_alias() -> dict[str, dict[str, Any]]:
    return {
        normalize_name(alias): record
        for record in load_parameter_truth()["records"]
        for alias in record["aliases"]
    }


def resolve_parameter_truth(model_name: Any) -> dict[str, Any] | None:
    return _by_alias().get(normalize_name(model_name))


def apply_parameter_truth(
    row: dict[str, Any],
    *,
    name_fields: tuple[str, ...] = ("model", "name", "canonical_model_name"),
    total_fields: tuple[str, ...] = ("total_b", "parameters_b", "total_parameters_b"),
    active_fields: tuple[str, ...] = ("active_b", "active_parameters_b"),
) -> dict[str, Any]:
    """Return a copied row with canonical values and raw values retained."""

    output = dict(row)
    record = next(
        (
            resolve_parameter_truth(output.get(field))
            for field in name_fields
            if output.get(field) not in (None, "")
            and resolve_parameter_truth(output.get(field)) is not None
        ),
        None,
    )
    if record is None:
        return output

    total_field = next((field for field in total_fields if output.get(field) not in (None, "")), None)
    active_field = next((field for field in active_fields if output.get(field) not in (None, "")), None)
    if total_field is None:
        return output
    raw_total = float(output[total_field])
    accepted_total = [float(value) for value in record["accepted_raw_total_parameters_b"]]
    if not any(math.isclose(raw_total, value, rel_tol=0, abs_tol=1e-9) for value in accepted_total):
        raise ValueError(
            f"Unexpected raw total for {record['truth_id']}: {raw_total} not in {accepted_total}"
        )
    raw_active = None if active_field is None else float(output[active_field])
    if raw_active is not None:
        accepted_active = [float(value) for value in record["accepted_raw_active_parameters_b"]]
        if not any(math.isclose(raw_active, value, rel_tol=0, abs_tol=1e-9) for value in accepted_active):
            raise ValueError(
                f"Unexpected raw active count for {record['truth_id']}: {raw_active} not in {accepted_active}"
            )

    output[f"raw_{total_field}"] = output[total_field]
    output[total_field] = float(record["canonical_total_parameters_b"])
    if active_field is not None:
        output[f"raw_{active_field}"] = output[active_field]
        output[active_field] = float(record["canonical_active_parameters_b"])
    output.update(
        {
            "parameter_truth_id": record["truth_id"],
            "parameter_truth_basis": record["parameter_value_basis"],
            "parameter_truth_ledger_path": str(LEDGER_PATH.relative_to(ROOT)),
            "raw_parameter_total_b": raw_total,
            "raw_parameter_active_b": raw_active,
        }
    )
    return output


def apply_parameter_truth_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [apply_parameter_truth(row) for row in rows]
