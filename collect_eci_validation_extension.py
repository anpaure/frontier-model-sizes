#!/usr/bin/env python3
"""Pin and verify the post-freeze 2026-07-22 Epoch ECI capture.

The main historical archive intentionally ends at 2026-07-16.  This capture
is kept as a separate score-vintage holdout so adding it cannot rewrite the
selection sample or masquerade as data that were available to the original
project.  Ordinary runs are network-free; ``--refresh`` is explicit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sources" / "epoch_eci_benchmarks_2026-07-22.csv"
METADATA = ROOT / "sources" / "epoch_eci_validation_extension_2026-07-31.json"
CAPTURE_TIMESTAMP = "20260722005716"
ORIGINAL_URL = "https://epoch.ai/data/eci_benchmarks.csv"
WAYBACK_URL = (
    "https://web.archive.org/web/20260722005716id_/"
    "https://epoch.ai/data/eci_benchmarks.csv"
)
EXPECTED_SHA256 = "85eb265713ebd29068b5cc5430c33596d5896c7b98a03b5f18f8d0e1e66615f8"
EXPECTED_ROWS = 2034
EXPECTED_MODELS = 212
EXPECTED_COARSE_MODEL_FAMILIES = 184
EXPECTED_BENCHMARKS = 53
EXPECTED_K3_COMPONENTS = 8


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def validate(payload: bytes) -> dict[str, object]:
    if sha256_bytes(payload) != EXPECTED_SHA256:
        raise ValueError("July 22 ECI capture differs from the pinned SHA-256")
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    required = {
        "model_id",
        "benchmark_id",
        "performance",
        "benchmark",
        "model",
        "model_version",
        "date",
    }
    if not rows or not required <= set(rows[0]):
        raise ValueError("July 22 ECI capture has an unexpected schema")
    keys = [(row["model_id"], row["benchmark_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("July 22 ECI capture contains duplicate model/benchmark rows")
    if any(not 0 <= float(row["performance"]) <= 1 for row in rows):
        raise ValueError("July 22 ECI capture has performance outside [0, 1]")
    models = {row["model"] for row in rows}
    coarse_families = {row["Model"] for row in rows}
    benchmarks = {row["benchmark"] for row in rows}
    k3 = [row for row in rows if row["model"] == "Kimi K3"]
    observed = (
        len(rows),
        len(models),
        len(coarse_families),
        len(benchmarks),
        len(k3),
    )
    expected = (
        EXPECTED_ROWS,
        EXPECTED_MODELS,
        EXPECTED_COARSE_MODEL_FAMILIES,
        EXPECTED_BENCHMARKS,
        EXPECTED_K3_COMPONENTS,
    )
    if observed != expected:
        raise ValueError(f"July 22 ECI inventory changed: {observed} != {expected}")
    if {row["model_version"] for row in k3} != {"kimi-k3_max"}:
        raise ValueError("July 22 Kimi K3 rows do not resolve uniquely to kimi-k3_max")
    return {
        "rows": len(rows),
        "models": len(models),
        "coarse_model_families": len(coarse_families),
        "benchmarks": len(benchmarks),
        "k3_component_rows": len(k3),
        "k3_model_version": "kimi-k3_max",
    }


def metadata_document(payload: bytes) -> dict[str, object]:
    inventory = validate(payload)
    return {
        "generated_on": "2026-07-31",
        "role": "post-freeze score-vintage holdout; excluded from historical model selection and live forecast weighting",
        "capture_timestamp_utc": CAPTURE_TIMESTAMP,
        "capture_date": "2026-07-22",
        "original_url": ORIGINAL_URL,
        "wayback_url": WAYBACK_URL,
        "pinned_file": str(SOURCE.relative_to(ROOT)),
        "sha256": EXPECTED_SHA256,
        "bytes": len(payload),
        "inventory": inventory,
        "classification": {
            "Kimi K3": {
                "score_vintage_holdout": True,
                "project_prospective": False,
                "reason": "The project had already used K3's disclosed 2.8T size before this score vintage was incorporated.",
            }
        },
    }


def verify_existing() -> dict[str, object]:
    payload = SOURCE.read_bytes()
    expected = metadata_document(payload)
    observed = json.loads(METADATA.read_text(encoding="utf-8"))
    if observed != expected:
        raise ValueError("July 22 ECI extension metadata is stale")
    return {
        "mode": "verify-existing",
        "source": str(SOURCE.relative_to(ROOT)),
        "sha256": EXPECTED_SHA256,
        **expected["inventory"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="explicitly fetch the pinned Wayback payload; default is offline verification",
    )
    args = parser.parse_args()
    if not args.refresh:
        print(json.dumps(verify_existing(), indent=2, sort_keys=True))
        return
    request = urllib.request.Request(
        WAYBACK_URL, headers={"User-Agent": "frontier-parameter-model/1.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    metadata = metadata_document(payload)
    SOURCE.write_bytes(payload)
    METADATA.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"mode": "refresh", **metadata["inventory"]}, indent=2))


if __name__ == "__main__":
    main()
