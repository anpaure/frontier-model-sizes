#!/usr/bin/env python3
"""Collect and verify narrow primary-source parameter-truth reconciliations.

Ordinary pipeline runs are offline and only validate the pinned evidence.  The
explicit ``--refresh`` mode fetches the two official MiniMax Hugging Face API
records whose safetensors inventories resolve the 229B/230B display conflict.
Kimi K2 uses Moonshot's already-vendored technical-report table; the frozen
official configs are used only to prove equal parameter-bearing shapes across
K2.5, K2.6, and K2.7 Code, never identical weights.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATE = "2026-07-31"
EVIDENCE_DIR = ROOT / "sources/open_model_parameter_truth_evidence_2026-07-31"
LEDGER = ROOT / "sources/open_model_parameter_truth_reconciliation_2026-07-31.json"
K3_EVIDENCE = ROOT / "sources/kimi_k3_release_evidence_2026-07-31.json"
ARCHITECTURE_SNAPSHOT = (
    ROOT / "sources/huggingface_architecture_config_snapshot_2026-07-18.json.gz"
)
M27_CONFIG = ROOT / (
    "sources/aa_parameter_label_availability_evidence_2026-07-31/"
    "aa-label-minimax-m2-7-2026-04-09__commit_pinned_config.json"
)

HF_TARGETS = {
    "minimax-m2-5": "https://huggingface.co/api/models/MiniMaxAI/MiniMax-M2.5",
    "minimax-m2-7": "https://huggingface.co/api/models/MiniMaxAI/MiniMax-M2.7",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def refresh() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    for key, url in HF_TARGETS.items():
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "frontier-parameter-audit/1.0"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
        (EVIDENCE_DIR / f"{key}__hugging_face_model_api.json").write_bytes(
            canonical_json(payload)
        )
    LEDGER.write_bytes(canonical_json(build_ledger()))


def normalized_parameter_shape(config: Any) -> Any:
    """Remove only token IDs and serialization/quantization metadata."""

    if isinstance(config, dict):
        return {
            key: normalized_parameter_shape(value)
            for key, value in config.items()
            if key not in {"eos_token_id", "quantization_config"}
        }
    if isinstance(config, list):
        return [normalized_parameter_shape(value) for value in config]
    return config


def k2_shape_evidence() -> dict[str, Any]:
    with gzip.open(ARCHITECTURE_SNAPSHOT, "rt", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    repos = (
        "moonshotai/kimi-k2.5",
        "moonshotai/kimi-k2.6",
        "moonshotai/kimi-k2.7-code",
    )
    entries = {entry["repo"]: entry for entry in snapshot["entries"]}
    missing = set(repos) - set(entries)
    if missing:
        raise ValueError(f"Missing K2 config evidence: {sorted(missing)}")
    hashes: dict[str, str] = {}
    raw_hashes: dict[str, str] = {}
    for repo in repos:
        entry = entries[repo]
        if entry["status_code"] != 200 or not isinstance(entry["config"], dict):
            raise ValueError(f"Invalid K2 config snapshot entry: {repo}")
        normalized = json.dumps(
            normalized_parameter_shape(entry["config"]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        hashes[repo] = hashlib.sha256(normalized).hexdigest()
        raw_hashes[repo] = entry["content_sha256"]
    if len(set(hashes.values())) != 1:
        raise ValueError(f"K2 parameter-bearing config shapes disagree: {hashes}")
    return {
        "repositories": list(repos),
        "raw_config_sha256": raw_hashes,
        "normalized_parameter_shape_sha256": next(iter(hashes.values())),
        "normalization": "remove eos_token_id and quantization_config only",
        "same_parameter_shape": True,
        "same_weight_identity": "unproven",
    }


def hf_api_record(key: str) -> tuple[Path, dict[str, Any]]:
    path = EVIDENCE_DIR / f"{key}__hugging_face_model_api.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run {Path(__file__).name} --refresh once"
        )
    return path, json.loads(path.read_text(encoding="utf-8"))


def build_ledger() -> dict[str, Any]:
    k3 = json.loads(K3_EVIDENCE.read_text(encoding="utf-8"))
    k2 = k3["kimi_k2_comparator"]
    if (
        float(k2["total_parameters_b_exact"]) != 1040.0
        or float(k2["activated_parameters_b_exact"]) != 32.6
    ):
        raise ValueError("Pinned Moonshot K2 comparator no longer reconciles")

    m25_path, m25 = hf_api_record("minimax-m2-5")
    m27_path, m27 = hf_api_record("minimax-m2-7")
    expected_total = 228_703_644_928
    for expected_id, payload in (
        ("MiniMaxAI/MiniMax-M2.5", m25),
        ("MiniMaxAI/MiniMax-M2.7", m27),
    ):
        if payload.get("id") != expected_id:
            raise ValueError(f"Unexpected Hugging Face model identity: {payload.get('id')}")
        if (payload.get("safetensors") or {}).get("total") != expected_total:
            raise ValueError(f"Unexpected safetensors inventory for {expected_id}")
    if sha256(M27_CONFIG) != "3372a31ecbdba614b309cfa5b3dfec33ceea3c580cb9612ff35345d92ced8618":
        raise ValueError("Pinned MiniMax M2.7 config hash changed")

    sources = []
    for path, role, url in (
        (K3_EVIDENCE, "Moonshot K3 report extraction containing K2 Table 1", "https://arxiv.org/abs/2607.24653"),
        (ARCHITECTURE_SNAPSHOT, "frozen official Hugging Face config snapshot", "https://huggingface.co/"),
        (M27_CONFIG, "commit-pinned official MiniMax M2.7 config", "https://huggingface.co/MiniMaxAI/MiniMax-M2.7"),
        (m25_path, "official MiniMax M2.5 Hugging Face model API", HF_TARGETS["minimax-m2-5"]),
        (m27_path, "official MiniMax M2.7 Hugging Face model API", HF_TARGETS["minimax-m2-7"]),
    ):
        sources.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "role": role,
                "url": url,
            }
        )

    return {
        "schema_version": "1.0",
        "snapshot_date": DATE,
        "purpose": "Resolve only evidenced coarse parameter-label conflicts without changing checkpoint identity or widening global match tolerances.",
        "records": [
            {
                "truth_id": "moonshot-kimi-k2-family-report-table-1",
                "aliases": [
                    "Kimi K2",
                    "Kimi K2 (Jul 2025)",
                    "Kimi K2 Instruct",
                    "Kimi K2-0905",
                    "Kimi K2 (Sep 2025)",
                    "Kimi K2 Thinking",
                    "Kimi K2.5",
                    "Kimi K2.5 Reasoning",
                    "Kimi K2.5 (Reasoning)",
                    "Kimi K2.5 (Non-reasoning)",
                    "Kimi K2.6",
                    "Kimi K2.6 Reasoning",
                    "Kimi K2.6 (Non-reasoning)",
                    "Kimi K2.7 Code",
                ],
                "accepted_raw_total_parameters_b": [1000.0, 1040.0],
                "accepted_raw_active_parameters_b": [32.0, 32.6],
                "canonical_total_parameters_b": 1040.0,
                "canonical_active_parameters_b": 32.6,
                "parameter_value_basis": "publisher_disclosed_model_total; Moonshot K3 technical report page 11 Table 1",
                "reported_precision_t": 0.01,
                "parameter_shape": k2_shape_evidence(),
                "checkpoint_policy": "retain every release as a distinct checkpoint in one Kimi K2 lineage; equal parameter shape is not a same-weight claim",
                "modality_caveat": "32.6B is the reported K2 activated count; multimodal-path activation can depend on the request modality.",
            },
            {
                "truth_id": "minimax-m2-5-official-safetensors",
                "aliases": ["MiniMax-M2.5", "MiniMax M2.5"],
                "accepted_raw_total_parameters_b": [228.703644928, 229.0, 230.0],
                "accepted_raw_active_parameters_b": [10.0],
                "canonical_total_parameters_b": 228.703644928,
                "canonical_active_parameters_b": 10.0,
                "parameter_value_basis": "official Hugging Face safetensors tensor inventory",
                "exact_tensor_parameters": expected_total,
                "repository_revision": m25["sha"],
                "config_shape_crosscheck": "M2.5 and M2.7 official configs have identical SHA-256 3372a31ecbdba614b309cfa5b3dfec33ceea3c580cb9612ff35345d92ced8618",
            },
            {
                "truth_id": "minimax-m2-7-official-safetensors",
                "aliases": ["MiniMax-M2.7", "MiniMax M2.7"],
                "accepted_raw_total_parameters_b": [228.703644928, 229.0, 230.0],
                "accepted_raw_active_parameters_b": [10.0],
                "canonical_total_parameters_b": 228.703644928,
                "canonical_active_parameters_b": 10.0,
                "parameter_value_basis": "official Hugging Face safetensors tensor inventory",
                "exact_tensor_parameters": expected_total,
                "repository_revision": m27["sha"],
                "config_shape_crosscheck": "M2.5 and M2.7 official configs have identical SHA-256 3372a31ecbdba614b309cfa5b3dfec33ceea3c580cb9612ff35345d92ced8618",
            },
        ],
        "source_files": sources,
        "policy": {
            "preserve_raw_values": True,
            "deduplicate_checkpoints": False,
            "same_weight_claim_from_equal_shape": False,
            "expand_global_parameter_match_tolerance": False,
            "ordinary_build_network_reads": 0,
        },
    }


def verify() -> dict[str, Any]:
    committed = json.loads(LEDGER.read_text(encoding="utf-8"))
    rebuilt = build_ledger()
    if committed != rebuilt:
        raise ValueError("Open-model parameter-truth ledger is stale")
    return committed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.refresh:
        refresh()
    payload = verify()
    print(
        json.dumps(
            {
                "ledger": str(LEDGER.relative_to(ROOT)),
                "records": len(payload["records"]),
                "sources": len(payload["source_files"]),
                "network_reads": len(HF_TARGETS) if args.refresh else 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
