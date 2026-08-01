#!/usr/bin/env python3
"""Collect primary Hugging Face architecture configs for OpenRouter calibration.

The frozen snapshot is the source of truth for ordinary builds.  Network access
occurs only with ``--refresh``.  Configs are fetched from each model repository's
raw ``config.json`` and retained verbatim inside a gzip JSON snapshot.  The
derived CSV exposes nested expert-routing fields without inferring active
parameter counts from names.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATE = "2026-07-18"
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
CALIBRATION = OUT / f"openrouter_parameter_calibration_{DATE}.csv"
MODELS = ROOT / f"sources/openrouter_model_signals_{DATE}.csv"
RAW = ROOT / f"sources/huggingface_architecture_config_snapshot_{DATE}.json.gz"
SIGNALS = ROOT / f"sources/huggingface_architecture_config_signals_{DATE}.csv"
AUDIT = OUT / f"huggingface_architecture_config_collection_audit_{DATE}.json"

EXPERT_KEY_PATTERN = re.compile(
    r"(?:^|_)(?:num|n)?_?(?:local_)?(?:routed_)?experts?(?:_per_tok(?:en)?)?$|"
    r"experts?_per_tok(?:en)?|moe_intermediate_size|shared_expert",
    re.IGNORECASE,
)
ARCHITECTURE_MOE_PATTERN = re.compile(
    r"moe|mixtral|jamba|dbrx|switch|deepseekv2|deepseekv3|glm4moe|qwen3moe|"
    r"qwen3next|minimax|ernie4_5_moe|nemotronh",
    re.IGNORECASE,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_repo(value: str) -> str:
    normalized = value.strip().lower().rstrip("/")
    return re.sub(r"^https?://huggingface\.co/", "", normalized)


def repository_inventory() -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    models = {row["openrouter_model_id"]: row for row in read_csv(MODELS)}
    inventory: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"checkpoint_ids": [], "openrouter_model_ids": []}
    )
    missing_repo_models: list[str] = []
    for checkpoint in read_csv(CALIBRATION):
        for model_id in checkpoint["openrouter_model_ids"].split("|"):
            repo = normalize_repo(models[model_id]["hugging_face_id"])
            if not repo:
                missing_repo_models.append(model_id)
                continue
            inventory[repo]["checkpoint_ids"].append(
                checkpoint["canonical_checkpoint_id"]
            )
            inventory[repo]["openrouter_model_ids"].append(model_id)
    for record in inventory.values():
        for key in record:
            record[key] = sorted(set(record[key]))
    return dict(sorted(inventory.items())), sorted(set(missing_repo_models))


def fetch_config(repo: str) -> dict[str, Any]:
    url = f"https://huggingface.co/{repo}/raw/main/config.json"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "frontier-parameter-model/1.0 architecture-audit",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
            status = int(response.status)
            final_url = response.geturl()
    except urllib.error.HTTPError as error:
        content = error.read()
        status = int(error.code)
        final_url = error.geturl()
        return {
            "repo": repo,
            "requested_url": url,
            "final_url": final_url,
            "status_code": status,
            "content_sha256": sha256_bytes(content),
            "content_bytes": len(content),
            "error": f"HTTP {status}",
            "config": None,
        }
    except Exception as error:  # network failures are preserved, not hidden
        return {
            "repo": repo,
            "requested_url": url,
            "final_url": "",
            "status_code": 0,
            "content_sha256": "",
            "content_bytes": 0,
            "error": f"{type(error).__name__}: {error}",
            "config": None,
        }
    try:
        config = json.loads(content)
    except json.JSONDecodeError as error:
        config = None
        parse_error = f"JSONDecodeError: {error}"
    else:
        parse_error = ""
    return {
        "repo": repo,
        "requested_url": url,
        "final_url": final_url,
        "status_code": status,
        "content_sha256": sha256_bytes(content),
        "content_bytes": len(content),
        "error": parse_error,
        "config": config,
    }


def walk_config(value: Any, path: str = "") -> list[tuple[str, Any]]:
    output: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            output.append((child_path, child))
            output.extend(walk_config(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            output.extend(walk_config(child, f"{path}[{index}]"))
    return output


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def config_signals(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {
            "model_type": "",
            "architectures": [],
            "expert_fields": {},
            "routed_expert_count": None,
            "experts_per_token": None,
            "shared_expert_count": None,
            "classification": "unavailable",
        }
    flattened = walk_config(config)
    expert_fields = {
        path: value
        for path, value in flattened
        if EXPERT_KEY_PATTERN.search(path.rsplit(".", 1)[-1])
        and isinstance(value, (str, int, float, bool, type(None)))
    }
    routed_candidates: list[float] = []
    active_candidates: list[float] = []
    shared_candidates: list[float] = []
    for path, value in expert_fields.items():
        numeric = number(value)
        if numeric is None:
            continue
        key = path.lower()
        if "per_tok" in key or "experts_per_token" in key:
            active_candidates.append(numeric)
        elif "shared" in key and "intermediate" not in key:
            shared_candidates.append(numeric)
        elif "intermediate" not in key:
            routed_candidates.append(numeric)
    architectures = config.get("architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    model_types = [
        str(value)
        for path, value in flattened
        if path.endswith("model_type") and isinstance(value, str)
    ]
    architecture_text = "|".join(
        [str(value) for value in architectures] + model_types
    )
    has_expert_routing = any(
        value > 1 for value in routed_candidates + active_candidates
    ) or bool(ARCHITECTURE_MOE_PATTERN.search(architecture_text))
    classification = "moe_config" if has_expert_routing else "dense_config"
    return {
        "model_type": str(config.get("model_type") or ""),
        "architectures": [str(value) for value in architectures],
        "expert_fields": expert_fields,
        "routed_expert_count": max(routed_candidates) if routed_candidates else None,
        "experts_per_token": min(active_candidates) if active_candidates else None,
        "shared_expert_count": max(shared_candidates) if shared_candidates else None,
        "classification": classification,
    }


def write_outputs(snapshot: dict[str, Any]) -> None:
    inventory, missing_repo_models = repository_inventory()
    entries = {entry["repo"]: entry for entry in snapshot["entries"]}
    if set(entries) != set(inventory):
        raise ValueError(
            "Frozen Hugging Face config inventory is stale: "
            f"missing={sorted(set(inventory) - set(entries))}, "
            f"extra={sorted(set(entries) - set(inventory))}"
        )
    rows: list[dict[str, Any]] = []
    for repo, identity in inventory.items():
        entry = entries[repo]
        signals = config_signals(entry.get("config"))
        rows.append(
            {
                "snapshot_date": DATE,
                "fetched_at_utc": snapshot["fetched_at_utc"],
                "hugging_face_repo": repo,
                "checkpoint_count": len(identity["checkpoint_ids"]),
                "checkpoint_ids": "|".join(identity["checkpoint_ids"]),
                "openrouter_model_count": len(identity["openrouter_model_ids"]),
                "openrouter_model_ids": "|".join(identity["openrouter_model_ids"]),
                "requested_url": entry["requested_url"],
                "final_url": entry["final_url"],
                "status_code": entry["status_code"],
                "content_bytes": entry["content_bytes"],
                "content_sha256": entry["content_sha256"],
                "error": entry["error"],
                "model_type": signals["model_type"],
                "architectures_json": json.dumps(
                    signals["architectures"], sort_keys=True
                ),
                "expert_fields_json": json.dumps(
                    signals["expert_fields"], sort_keys=True
                ),
                "routed_expert_count": signals["routed_expert_count"],
                "experts_per_token": signals["experts_per_token"],
                "shared_expert_count": signals["shared_expert_count"],
                "architecture_classification": signals["classification"],
            }
        )
    fields = list(rows[0])
    with SIGNALS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    classifications = defaultdict(int)
    status_counts = defaultdict(int)
    for row in rows:
        classifications[row["architecture_classification"]] += 1
        status_counts[str(row["status_code"])] += 1
    audit = {
        "schema_version": "1.0",
        "snapshot_date": DATE,
        "fetched_at_utc": snapshot["fetched_at_utc"],
        "network_refresh": snapshot["network_refresh"],
        "calibration_checkpoints": len(read_csv(CALIBRATION)),
        "unique_repositories": len(rows),
        "openrouter_models_without_hugging_face_repo": len(missing_repo_models),
        "missing_repo_model_ids": missing_repo_models,
        "http_status_counts": dict(sorted(status_counts.items())),
        "architecture_classification_counts": dict(sorted(classifications.items())),
        "successful_json_configs": sum(
            row["architecture_classification"] != "unavailable" for row in rows
        ),
        "method_notes": [
            "Repository identity comes from the frozen OpenRouter catalog and exact calibration crosswalk.",
            "Dense/MoE classification uses nested primary config fields and architecture names; it does not infer active parameter counts.",
            "HTTP errors and gated repositories are retained explicitly and never converted to dense labels.",
        ],
        "files": {
            "raw_snapshot": str(RAW.relative_to(ROOT)),
            "signals": str(SIGNALS.relative_to(ROOT)),
        },
        "source_hashes": {
            str(CALIBRATION.relative_to(ROOT)): sha256(CALIBRATION),
            str(MODELS.relative_to(ROOT)): sha256(MODELS),
            str(RAW.relative_to(ROOT)): sha256(RAW),
            str(SIGNALS.relative_to(ROOT)): sha256(SIGNALS),
        },
    }
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch every primary Hugging Face config instead of using the frozen snapshot.",
    )
    args = parser.parse_args()
    inventory, _ = repository_inventory()
    if args.refresh:
        entries: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(fetch_config, repo): repo for repo in inventory}
            for future in as_completed(futures):
                entries.append(future.result())
        snapshot = {
            "schema_version": "1.0",
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "network_refresh": True,
            "entries": sorted(entries, key=lambda row: row["repo"]),
        }
        with gzip.open(RAW, "wt", encoding="utf-8", compresslevel=9) as handle:
            json.dump(snapshot, handle, sort_keys=True, separators=(",", ":"))
    else:
        if not RAW.exists():
            raise FileNotFoundError(
                f"No frozen config snapshot at {RAW}; run once with --refresh"
            )
        with gzip.open(RAW, "rt", encoding="utf-8") as handle:
            snapshot = json.load(handle)
        snapshot["network_refresh"] = False
    write_outputs(snapshot)


if __name__ == "__main__":
    main()
