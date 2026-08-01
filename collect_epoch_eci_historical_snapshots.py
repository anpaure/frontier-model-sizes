#!/usr/bin/env python3
"""Collect or verify the frozen Epoch ECI archive ending on 2026-07-16.

The Internet Archive stored some payloads with their original gzip or Brotli
content encoding.  This collector normalizes only that transport layer and
then places the decoded official resources in a deterministic tarball.  Both
the CDX digest and decoded SHA-256 are retained for every capture.

The five pre-ECI benchmark archives are preserved losslessly even though the
statistical audit starts with the first archive containing
``epoch_capabilities_index.csv``.  Nothing fetched is silently discarded.

The archive is a historical data product, not an alias for the latest live
Epoch panel.  Its terminal 2026-07-16 capture is pinned against the archival
2026-07-17 canonical file.  The 2026-07-31 live source is recorded only as a
newer successor and is deliberately never used for byte-equality validation.

``--verify-existing`` is the deterministic, network-free pipeline mode.
``--reindex-existing`` upgrades metadata from the already pinned archive
without fetching or changing any archived payload bytes.  A no-flag run is an
explicit Wayback refresh and may fail if the remote capture inventory changes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATE = "2026-07-18"
SOURCES = ROOT / "sources"
ARCHIVE = SOURCES / f"epoch_eci_historical_snapshots_{DATE}.tar.gz"
CDX_OUTPUT = SOURCES / f"epoch_eci_historical_cdx_{DATE}.json"
METADATA = SOURCES / f"epoch_eci_historical_collection_metadata_{DATE}.json"
ARCHIVAL_TERMINAL_TIMESTAMP = "20260716153134"
ARCHIVAL_TERMINAL_CANONICAL = SOURCES / "epoch_eci_benchmarks_2026-07-17.csv"
CURRENT_LIVE_CANONICAL = SOURCES / "epoch_eci_benchmarks_2026-07-31.csv"

CANONICAL_URL = "https://epoch.ai/data/eci_benchmarks.csv"
BENCHMARK_URL = "https://epoch.ai/data/benchmark_data.zip"
CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
WAYBACK_ENDPOINT = "https://web.archive.org/web/{timestamp}id_/{url}"
USER_AGENT = "frontier-parameter-model/1.0 historical source audit"

EXPECTED_CANONICAL = {
    "20260213005335",
    "20260505224414",
    "20260516072913",
    "20260616024413",
    "20260716153134",
}
EXPECTED_BENCHMARK = {
    "20250305190317",
    "20250403051524",
    "20250510183121",
    "20250612030853",
    "20250720150710",
    "20251113094011",
    "20251125180202",
    "20251214160510",
    "20260101232245",
    "20260204115444",
    "20260204224101",
    "20260205070603",
    "20260320171809",
    "20260427185318",
    "20260610165052",
}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def request_bytes(url: str, timeout: int = 180) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_cdx(url: str) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "url": url,
            "output": "json",
            "fl": "timestamp,original,statuscode,mimetype,digest,length",
            "filter": "statuscode:200",
            "collapse": "digest",
            "from": "2024",
            "to": "2026",
        }
    )
    payload = json.loads(request_bytes(f"{CDX_ENDPOINT}?{query}"))
    if not payload or payload[0][0] != "timestamp":
        raise ValueError(f"Unexpected CDX response for {url}")
    header = payload[0]
    records = [dict(zip(header, row, strict=True)) for row in payload[1:]]
    return {"query_url": f"{CDX_ENDPOINT}?{query}", "records": records}


def decode_transport(data: bytes, expected: str) -> tuple[bytes, list[str]]:
    layers: list[str] = []
    for _ in range(4):
        if expected == "zip" and zipfile.is_zipfile(io.BytesIO(data)):
            return data, layers
        if expected == "csv":
            try:
                text = data.decode("utf-8-sig")
                if text.startswith("model_id,benchmark_id,performance,"):
                    return data, layers
            except UnicodeDecodeError:
                pass
        if data.startswith(b"\x1f\x8b"):
            data = gzip.decompress(data)
            layers.append("gzip")
            continue
        brotli = shutil.which("brotli")
        if not brotli:
            raise RuntimeError(
                "Archived payload is not plain/gzip and the brotli CLI is unavailable"
            )
        decoded = subprocess.run(
            [brotli, "--decompress"],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        ).stdout
        if decoded == data:
            break
        data = decoded
        layers.append("brotli")
    raise ValueError(f"Could not decode archived {expected} payload")


def deterministic_tar_gz(members: list[tuple[str, bytes]]) -> bytes:
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        for name, data in sorted(members):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    return gzip.compress(tar_buffer.getvalue(), compresslevel=9, mtime=0)


def cached_or_fetch(
    record: dict[str, str],
    url: str,
    cache: Path | None,
    suffix: str,
) -> tuple[bytes, str]:
    candidate = cache / f"{record['timestamp']}{suffix}" if cache else None
    if candidate and candidate.exists():
        return candidate.read_bytes(), str(candidate)
    source = WAYBACK_ENDPOINT.format(timestamp=record["timestamp"], url=url)
    return request_bytes(source), source


def validate_csv(data: bytes) -> dict[str, Any]:
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    required = {"model_id", "benchmark_id", "performance", "benchmark", "model"}
    if not rows or not required <= set(rows[0]):
        raise ValueError("Historical canonical ECI CSV lacks required columns")
    keys = [(row["model_id"], row["benchmark_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate model_id/benchmark_id in historical canonical CSV")
    performance = [float(row["performance"]) for row in rows]
    if min(performance) < 0 or max(performance) > 1:
        raise ValueError("Historical canonical ECI performance outside [0, 1]")
    return {
        "rows": len(rows),
        "models": len({row["model"] for row in rows}),
        "benchmarks": len({row["benchmark"] for row in rows}),
        "columns": list(rows[0]),
    }


def validate_zip(data: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if not names or any(name.startswith("/") or ".." in Path(name).parts for name in names):
            raise ValueError("Unsafe or empty historical benchmark ZIP")
        has_eci = "epoch_capabilities_index.csv" in names
        eci_rows = 0
        eci_scored_rows = 0
        if has_eci:
            rows = list(
                csv.DictReader(
                    io.StringIO(
                        archive.read("epoch_capabilities_index.csv").decode("utf-8-sig")
                    )
                )
            )
            eci_rows = len(rows)
            eci_scored_rows = sum(bool(row.get("ECI Score")) for row in rows)
        return {
            "members": len(names),
            "has_eci_table": has_eci,
            "eci_table_rows": eci_rows,
            "eci_scored_rows": eci_scored_rows,
            "member_names": names,
        }


def read_archive_members(path: Path = ARCHIVE) -> dict[str, bytes]:
    """Read the normalized historical payloads without mutating the archive."""
    with tarfile.open(path, mode="r:gz") as archive:
        members: dict[str, bytes] = {}
        for member in archive.getmembers():
            if not member.isfile():
                raise ValueError(f"Unexpected non-file historical member: {member.name}")
            if member.name.startswith("/") or ".." in Path(member.name).parts:
                raise ValueError(f"Unsafe historical archive member: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Could not read historical archive member: {member.name}")
            members[member.name] = handle.read()
    return members


def capture_inventory(captures: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "canonical_csv_captures": sum(
            row["kind"] == "canonical_eci_csv" for row in captures
        ),
        "benchmark_zip_captures": sum(row["kind"] == "benchmark_zip" for row in captures),
        "benchmark_zip_eci_era": sum(
            row["kind"] == "benchmark_zip"
            and row["inventory"].get("has_eci_table", False)
            for row in captures
        ),
        "benchmark_zip_pre_eci_preserved": sum(
            row["kind"] == "benchmark_zip"
            and not row["inventory"].get("has_eci_table", False)
            for row in captures
        ),
        "archive_members": len(captures),
        "earliest_capture": min(row["timestamp"] for row in captures),
        "latest_capture": max(row["timestamp"] for row in captures),
    }


def validate_capture_inventory(captures: list[dict[str, Any]]) -> None:
    canonical = {
        row["timestamp"] for row in captures if row["kind"] == "canonical_eci_csv"
    }
    benchmark = {
        row["timestamp"] for row in captures if row["kind"] == "benchmark_zip"
    }
    if canonical != EXPECTED_CANONICAL:
        raise ValueError(f"Canonical ECI capture inventory changed: {canonical}")
    if benchmark != EXPECTED_BENCHMARK:
        raise ValueError(f"Benchmark ZIP capture inventory changed: {benchmark}")


def validate_cdx_against_captures(
    cdx: dict[str, Any], captures: list[dict[str, Any]]
) -> None:
    expected_by_kind = {
        "canonical_eci_csv": cdx.get("canonical_eci_csv", {}).get("records", []),
        "benchmark_zip": cdx.get("benchmark_zip", {}).get("records", []),
    }
    for kind, records in expected_by_kind.items():
        by_timestamp = {row["timestamp"]: row for row in records}
        observed = [row for row in captures if row["kind"] == kind]
        if set(by_timestamp) != {row["timestamp"] for row in observed}:
            raise ValueError(f"Historical CDX and capture metadata differ for {kind}")
        for row in observed:
            cdx_row = by_timestamp[row["timestamp"]]
            for field in ("original", "digest", "length"):
                if str(cdx_row[field]) != str(row[f"cdx_{field}"] if field in {"digest", "length"} else row["original_url"]):
                    raise ValueError(
                        f"Historical CDX {field} differs for {kind} {row['timestamp']}"
                    )


def validate_members_against_captures(
    members: dict[str, bytes], captures: list[dict[str, Any]]
) -> None:
    if set(members) != {row["archive_member"] for row in captures}:
        raise ValueError("Historical archive members differ from collection metadata")
    for row in captures:
        data = members[row["archive_member"]]
        if sha256_bytes(data) != row["decoded_bytes_sha256"]:
            raise ValueError(f"Decoded hash mismatch for {row['archive_member']}")
        if len(data) != int(row["decoded_bytes"]):
            raise ValueError(f"Decoded byte count mismatch for {row['archive_member']}")
        observed = validate_csv(data) if row["kind"] == "canonical_eci_csv" else validate_zip(data)
        if observed != row["inventory"]:
            raise ValueError(f"Decoded inventory mismatch for {row['archive_member']}")


def terminal_member(members: dict[str, bytes]) -> bytes:
    name = f"canonical_csv/eci_benchmarks_{ARCHIVAL_TERMINAL_TIMESTAMP}.csv"
    try:
        return members[name]
    except KeyError as error:
        raise ValueError(f"Historical terminal member is missing: {name}") from error


def metadata_document(cdx: dict[str, Any], captures: list[dict[str, Any]]) -> dict[str, Any]:
    members = read_archive_members()
    validate_capture_inventory(captures)
    validate_cdx_against_captures(cdx, captures)
    validate_members_against_captures(members, captures)
    terminal = terminal_member(members)
    if terminal != ARCHIVAL_TERMINAL_CANONICAL.read_bytes():
        raise ValueError(
            "Terminal archived canonical ECI CSV differs from the pinned archival source"
        )
    return {
        "generated_on": DATE,
        "source_authority": "Epoch AI official data endpoints captured by the Internet Archive",
        "canonical_url": CANONICAL_URL,
        "benchmark_url": BENCHMARK_URL,
        "archive_policy": {
            "role": "frozen historical archive used for archive-vintage backtests",
            "cutoff_date": DATE,
            "terminal_timestamp": ARCHIVAL_TERMINAL_TIMESTAMP,
            "terminal_capture_date": "2026-07-16",
            "current_live_sources_are_appended_elsewhere": True,
        },
        "archive": relative(ARCHIVE),
        "archive_sha256": sha256(ARCHIVE),
        "cdx": relative(CDX_OUTPUT),
        "cdx_sha256": sha256(CDX_OUTPUT),
        "captures": captures,
        "inventory": capture_inventory(captures),
        "archival_terminal_exact_match": {
            "timestamp": ARCHIVAL_TERMINAL_TIMESTAMP,
            "pinned_file": relative(ARCHIVAL_TERMINAL_CANONICAL),
            "sha256": sha256(ARCHIVAL_TERMINAL_CANONICAL),
            "exact": True,
        },
        "current_live_successor_reference": {
            "pinned_file": relative(CURRENT_LIVE_CANONICAL),
            "sha256": sha256(CURRENT_LIVE_CANONICAL),
            "role": "newer live source; excluded from the frozen historical archive",
            "byte_equality_assertion_against_archival_terminal": False,
        },
        "normalization_policy": "Only gzip/Brotli HTTP transport encoding is removed; decoded official resource bytes are otherwise unchanged.",
    }


def verify_existing() -> dict[str, Any]:
    metadata = json.loads(METADATA.read_text())
    cdx = json.loads(CDX_OUTPUT.read_text())
    if sha256(ARCHIVE) != metadata["archive_sha256"]:
        raise ValueError("Historical archive hash differs from collection metadata")
    if sha256(CDX_OUTPUT) != metadata["cdx_sha256"]:
        raise ValueError("Historical CDX hash differs from collection metadata")
    validate_capture_inventory(metadata["captures"])
    validate_cdx_against_captures(cdx, metadata["captures"])
    members = read_archive_members()
    validate_members_against_captures(members, metadata["captures"])
    terminal = metadata.get("archival_terminal_exact_match", {})
    if terminal.get("timestamp") != ARCHIVAL_TERMINAL_TIMESTAMP:
        raise ValueError("Historical metadata has the wrong terminal timestamp")
    if terminal.get("pinned_file") != relative(ARCHIVAL_TERMINAL_CANONICAL):
        raise ValueError("Historical metadata points at a non-archival terminal source")
    if terminal.get("sha256") != sha256(ARCHIVAL_TERMINAL_CANONICAL):
        raise ValueError("Historical terminal source hash differs from metadata")
    if terminal_member(members) != ARCHIVAL_TERMINAL_CANONICAL.read_bytes():
        raise ValueError("Historical terminal bytes differ from the archival pin")
    successor = metadata.get("current_live_successor_reference", {})
    if successor.get("pinned_file") != relative(CURRENT_LIVE_CANONICAL):
        raise ValueError("Historical metadata lacks the current-live successor reference")
    if successor.get("sha256") != sha256(CURRENT_LIVE_CANONICAL):
        raise ValueError("Current-live successor hash differs from historical metadata")
    if successor.get("byte_equality_assertion_against_archival_terminal") is not False:
        raise ValueError("Current live source must not be an archival byte-equality target")
    return {
        "mode": "verify-existing",
        "archive": relative(ARCHIVE),
        **capture_inventory(metadata["captures"]),
        "archival_terminal": ARCHIVAL_TERMINAL_TIMESTAMP,
        "current_live_source": relative(CURRENT_LIVE_CANONICAL),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-cache", type=Path)
    parser.add_argument("--benchmark-cache", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify-existing",
        action="store_true",
        help="validate the pinned archive and metadata without network access or writes",
    )
    mode.add_argument(
        "--reindex-existing",
        action="store_true",
        help="rewrite metadata from the pinned archive/CDX without fetching payloads",
    )
    args = parser.parse_args()

    if args.verify_existing:
        print(json.dumps(verify_existing(), indent=2))
        return
    if args.reindex_existing:
        existing = json.loads(METADATA.read_text())
        cdx = json.loads(CDX_OUTPUT.read_text())
        metadata = metadata_document(cdx, existing["captures"])
        METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        print(json.dumps(verify_existing(), indent=2))
        return

    cdx = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_eci_csv": fetch_cdx(CANONICAL_URL),
        "benchmark_zip": fetch_cdx(BENCHMARK_URL),
    }
    canonical_records = cdx["canonical_eci_csv"]["records"]
    benchmark_records = cdx["benchmark_zip"]["records"]
    validate_capture_inventory(
        [
            *({"kind": "canonical_eci_csv", **row} for row in canonical_records),
            *({"kind": "benchmark_zip", **row} for row in benchmark_records),
        ]
    )

    members: list[tuple[str, bytes]] = []
    captures: list[dict[str, Any]] = []
    for kind, records, url, cache, suffix, expected in (
        (
            "canonical_eci_csv",
            canonical_records,
            CANONICAL_URL,
            args.canonical_cache,
            ".csv",
            "csv",
        ),
        (
            "benchmark_zip",
            benchmark_records,
            BENCHMARK_URL,
            args.benchmark_cache,
            ".zip",
            "zip",
        ),
    ):
        for record in records:
            raw, retrieval_source = cached_or_fetch(record, url, cache, suffix)
            decoded, layers = decode_transport(raw, expected)
            inventory = validate_csv(decoded) if expected == "csv" else validate_zip(decoded)
            timestamp = record["timestamp"]
            member_name = (
                f"canonical_csv/eci_benchmarks_{timestamp}.csv"
                if kind == "canonical_eci_csv"
                else f"benchmark_zip/benchmark_data_{timestamp}.zip"
            )
            members.append((member_name, decoded))
            captures.append(
                {
                    "kind": kind,
                    "timestamp": timestamp,
                    "original_url": record["original"],
                    "wayback_url": WAYBACK_ENDPOINT.format(timestamp=timestamp, url=url),
                    "cdx_digest": record["digest"],
                    "cdx_length": int(record["length"]),
                    "retrieval_source": retrieval_source,
                    "transport_layers_removed": layers,
                    "retrieved_bytes_sha256": sha256_bytes(raw),
                    "decoded_bytes_sha256": sha256_bytes(decoded),
                    "decoded_bytes": len(decoded),
                    "archive_member": member_name,
                    "inventory": inventory,
                    "statistical_use": (
                        "fixed-code ECI reconstruction"
                        if kind == "canonical_eci_csv"
                        or inventory.get("has_eci_table")
                        else "preserved pre-ECI source; excluded from ECI regression"
                    ),
                }
            )

    terminal = next(
        data
        for name, data in members
        if name
        == f"canonical_csv/eci_benchmarks_{ARCHIVAL_TERMINAL_TIMESTAMP}.csv"
    )
    if terminal != ARCHIVAL_TERMINAL_CANONICAL.read_bytes():
        raise ValueError(
            "Terminal archived canonical ECI CSV differs from the pinned archival source"
        )

    archive_bytes = deterministic_tar_gz(members)
    ARCHIVE.write_bytes(archive_bytes)
    CDX_OUTPUT.write_text(json.dumps(cdx, indent=2, sort_keys=True) + "\n")
    metadata = metadata_document(cdx, captures)
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"archive": str(ARCHIVE), **metadata["inventory"]}, indent=2))


if __name__ == "__main__":
    main()
