#!/usr/bin/env python3
"""Install or verify the frozen Epoch snapshot used by the forecast pipeline.

The three upstream Epoch files form one atomic snapshot.  The fitted ECI/EDI
files are generated with Epoch's pinned public implementation and installed
with ``--fit-output-dir``.  Ordinary pipeline runs use the default verification
mode: no network access and no writes, with byte hashes, inventories, identity
sets, and fitted-ledger coverage checked together.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"
AS_OF = "2026-07-31"
MANIFEST = SOURCES / f"epoch_snapshot_manifest_{AS_OF}.json"

RAW_FILES = {
    "all_models": SOURCES / f"epoch_all_ai_models_{AS_OF}.csv",
    "eci_components": SOURCES / f"epoch_eci_benchmarks_{AS_OF}.csv",
    "published_archive": SOURCES / f"epoch_benchmark_data_{AS_OF}.zip",
}
FIT_FILES = {
    "eci_scores": SOURCES / f"epoch_eci_reproduced_scores_{AS_OF}.csv",
    "edi_scores": SOURCES / f"epoch_edi_reproduced_scores_{AS_OF}.csv",
}
METADATA = SOURCES / f"epoch_eci_reproduction_metadata_{AS_OF}.json"

PINNED_RAW_HASHES = {
    "all_models": "0d1fcfc497cccc1079068a1ec3031e30e97ad5c8e6f5b5d43baceac3778ba579",
    "eci_components": "d7cad7a8595347a62a2f832205aae579e371f05afe6ab08bc9506631e38c70d1",
    "published_archive": "cc844ca094e4372ff81eb636f3503317d1d495d4f4c493bd6b7a37dd5f83049c",
}
OFFICIAL_REPOSITORY = "https://github.com/epoch-research/eci-public"
OFFICIAL_COMMIT = "542567e72a415b72624e5bbd12603cfd3f485179"
EXPECTED = {
    "all_model_rows": 3574,
    "all_model_unique_names": 3569,
    "component_rows": 2059,
    "models": 213,
    "benchmarks": 54,
    "published_rows": 768,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def published_rows(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        raw = archive.read("epoch_capabilities_index.csv").decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(raw)))


def digest_lines(values: set[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def build_inventory() -> dict[str, Any]:
    for label, path in RAW_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing Epoch {label} source: {path}")
        actual = sha256(path)
        if actual != PINNED_RAW_HASHES[label]:
            raise RuntimeError(f"Epoch {label} hash mismatch: {actual}")

    all_models = csv_rows(RAW_FILES["all_models"])
    components = csv_rows(RAW_FILES["eci_components"])
    published = published_rows(RAW_FILES["published_archive"])
    model_names = {row["model"] for row in components}
    model_versions = {row["model_version"] for row in components}
    model_ids = {row["model_id"] for row in components}
    benchmarks = {row["benchmark"] for row in components}
    benchmark_ids = {row["benchmark_id"] for row in components}
    inventory = {
        "all_model_rows": len(all_models),
        "all_model_unique_names": len({row["Model"] for row in all_models}),
        "component_rows": len(components),
        "models": len(model_names),
        "model_versions": len(model_versions),
        "model_ids": len(model_ids),
        "benchmarks": len(benchmarks),
        "benchmark_ids": len(benchmark_ids),
        "published_rows": len(published),
    }
    for key, expected in EXPECTED.items():
        if inventory[key] != expected:
            raise RuntimeError(
                f"Epoch snapshot inventory mismatch for {key}: "
                f"{inventory[key]} != {expected}"
            )
    if inventory["models"] != inventory["model_ids"]:
        raise RuntimeError("Epoch canonical model name/id identity is not one-to-one")
    if inventory["benchmarks"] != inventory["benchmark_ids"]:
        raise RuntimeError("Epoch canonical benchmark identity is not one-to-one")
    duplicate_pairs = len(components) - len({(row["model"], row["benchmark"]) for row in components})
    if duplicate_pairs:
        raise RuntimeError(f"Epoch canonical component panel has {duplicate_pairs} duplicate name pairs")
    inventory.update(
        {
            "duplicate_model_benchmark_name_pairs": duplicate_pairs,
            "model_name_set_sha256": digest_lines(model_names),
            "model_version_set_sha256": digest_lines(model_versions),
            "benchmark_name_set_sha256": digest_lines(benchmarks),
            "identity_join_policy": "within_snapshot_only; join canonical names across snapshots, never bare model_id or benchmark_id",
        }
    )
    return inventory


def validate_fit_files() -> dict[str, Any]:
    for label, path in FIT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing fitted Epoch {label}: {path}")
    components = csv_rows(RAW_FILES["eci_components"])
    eci = csv_rows(FIT_FILES["eci_scores"])
    edi = csv_rows(FIT_FILES["edi_scores"])
    expected_models = {row["model"] for row in components}
    versions_by_model: dict[str, set[str]] = {}
    for row in components:
        versions_by_model.setdefault(row["model"], set()).add(row["model_version"])
    expected_benchmarks = {row["benchmark"] for row in components}
    if len(eci) != EXPECTED["models"]:
        raise RuntimeError(f"Expected {EXPECTED['models']} fitted ECI rows; found {len(eci)}")
    if {row["Model"] for row in eci} != expected_models:
        raise RuntimeError("Fitted ECI model set does not exactly match the canonical component input")
    if any(row["model_version"] not in versions_by_model[row["Model"]] for row in eci):
        raise RuntimeError("A fitted ECI selected version is absent from its canonical model rows")
    if len(edi) != EXPECTED["benchmarks"] or {row["benchmark"] for row in edi} != expected_benchmarks:
        raise RuntimeError("Fitted EDI benchmark set does not exactly match the canonical component input")
    anchors = {row["Model"]: float(row["eci"]) for row in eci if row["Model"] in {"Claude 3.5 Sonnet", "GPT-5"}}
    if anchors != {"Claude 3.5 Sonnet": 130.0, "GPT-5": 150.0}:
        raise RuntimeError(f"Unexpected fitted ECI anchors: {anchors}")
    return {
        "eci_rows": len(eci),
        "edi_rows": len(edi),
        "anchors": anchors,
        "eci_model_set_sha256": digest_lines({row["Model"] for row in eci}),
        "eci_selected_version_set_sha256": digest_lines({row["model_version"] for row in eci}),
        "edi_benchmark_set_sha256": digest_lines({row["benchmark"] for row in edi}),
    }


def install_fit(fit_output_dir: Path) -> None:
    source_eci = fit_output_dir / "eci_scores.csv"
    source_edi = fit_output_dir / "edi_scores.csv"
    if not source_eci.exists() or not source_edi.exists():
        raise FileNotFoundError("Official fit output must contain eci_scores.csv and edi_scores.csv")
    SOURCES.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_eci, FIT_FILES["eci_scores"])
    shutil.copyfile(source_edi, FIT_FILES["edi_scores"])


def write_metadata(inventory: dict[str, Any], fit_inventory: dict[str, Any]) -> None:
    metadata = {
        "schema_version": "2.0",
        "generated_at_utc": "2026-07-31T00:00:00Z",
        "snapshot_as_of": AS_OF,
        "source": "Epoch AI official ECI implementation",
        "source_repository": OFFICIAL_REPOSITORY,
        "source_commit": OFFICIAL_COMMIT,
        "input_url": "https://epoch.ai/data/eci_benchmarks.csv",
        "input_file": str(RAW_FILES["eci_components"].relative_to(ROOT)),
        "input_sha256": sha256(RAW_FILES["eci_components"]),
        "output_file": str(FIT_FILES["eci_scores"].relative_to(ROOT)),
        "output_sha256": sha256(FIT_FILES["eci_scores"]),
        "edi_output_file": str(FIT_FILES["edi_scores"].relative_to(ROOT)),
        "edi_output_sha256": sha256(FIT_FILES["edi_scores"]),
        "bootstrap_samples": 500,
        "bootstrap_seed": 12345,
        "anchor_models": fit_inventory["anchors"],
        "command": "PYTHONPATH=<eci-public>/src python <eci-public>/scripts/fit_eci.py --input sources/epoch_eci_benchmarks_2026-07-31.csv --bootstrap-samples 500 --output-dir <output-dir>",
        "input_inventory": inventory,
        "fit_inventory": fit_inventory,
    }
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_manifest(inventory: dict[str, Any], fit_inventory: dict[str, Any]) -> dict[str, Any]:
    files = {**RAW_FILES, **FIT_FILES, "reproduction_metadata": METADATA}
    return {
        "schema_version": "1.0",
        "snapshot_as_of": AS_OF,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "files": {
            label: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for label, path in files.items()
        },
        "inventory": inventory,
        "fit_inventory": fit_inventory,
    }


def verify_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing Epoch snapshot manifest: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["snapshot_as_of"] != AS_OF or manifest["official_commit"] != OFFICIAL_COMMIT:
        raise RuntimeError("Epoch snapshot manifest version/commit mismatch")
    inventory = build_inventory()
    fit_inventory = validate_fit_files()
    if manifest["inventory"] != inventory or manifest["fit_inventory"] != fit_inventory:
        raise RuntimeError("Epoch snapshot manifest inventory no longer matches its files")
    for label, record in manifest["files"].items():
        path = ROOT / record["path"]
        if sha256(path) != record["sha256"]:
            raise RuntimeError(f"Epoch manifest hash mismatch for {label}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-output-dir",
        type=Path,
        help="Install eci_scores.csv and edi_scores.csv from a pinned official fit, then write the manifest",
    )
    args = parser.parse_args()
    if args.fit_output_dir:
        build_inventory()
        install_fit(args.fit_output_dir)
        inventory = build_inventory()
        fit_inventory = validate_fit_files()
        write_metadata(inventory, fit_inventory)
        MANIFEST.write_text(
            json.dumps(build_manifest(inventory, fit_inventory), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest = verify_manifest()
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "snapshot_as_of": manifest["snapshot_as_of"],
                "models": manifest["inventory"]["models"],
                "benchmarks": manifest["inventory"]["benchmarks"],
                "component_rows": manifest["inventory"]["component_rows"],
                "status": "PASS",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
