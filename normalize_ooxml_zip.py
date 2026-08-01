#!/usr/bin/env python3
"""Make DOCX/XLSX ZIP containers byte-stable without changing package content."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
RELATIONSHIP_ID = re.compile(rb"R[0-9A-Fa-f]{16}")
GUID = re.compile(
    rb"\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}"
)
THREADED_COMMENT_TIMESTAMP = re.compile(rb'\bdT="[^"]+"')
FIXED_THREADED_COMMENT_TIMESTAMP = b'dT="2026-07-18T00:00:00.000"'


def _canonical_tokens(
    entries: list[tuple[ZipInfo, bytes]],
) -> list[tuple[ZipInfo, bytes]]:
    """Canonicalize random artifact-tool relationship and comment identifiers."""

    relationship_map: dict[bytes, bytes] = {}
    guid_map: dict[bytes, bytes] = {}
    xml_entries = sorted(
        (
            (info.filename, payload)
            for info, payload in entries
            if info.filename.endswith((".xml", ".rels"))
        ),
        key=lambda item: item[0],
    )
    for _, payload in xml_entries:
        for match in RELATIONSHIP_ID.finditer(payload):
            token = match.group(0)
            relationship_map.setdefault(
                token, f"R{len(relationship_map) + 1:016x}".encode("ascii")
            )

    comment_entries = [
        (name, payload)
        for name, payload in xml_entries
        if name.startswith(("xl/comments", "xl/persons/", "xl/threadedcomments/"))
    ]
    for _, payload in comment_entries:
        for match in GUID.finditer(payload):
            token = match.group(0)
            guid_map.setdefault(
                token,
                (
                    "{00000000-0000-4000-8000-"
                    f"{len(guid_map) + 1:012x}"
                    "}"
                ).encode("ascii"),
            )

    output = []
    for info, payload in entries:
        if info.filename.endswith((".xml", ".rels")):
            for token, replacement in relationship_map.items():
                payload = payload.replace(token, replacement)
            if info.filename.startswith(
                ("xl/comments", "xl/persons/", "xl/threadedcomments/")
            ):
                for token, replacement in guid_map.items():
                    payload = payload.replace(token, replacement)
            if info.filename.startswith("xl/threadedcomments/"):
                payload = THREADED_COMMENT_TIMESTAMP.sub(
                    FIXED_THREADED_COMMENT_TIMESTAMP, payload
                )
        output.append((info, payload))
    return output


def normalize_ooxml_zip(path: Path) -> None:
    """Rewrite an OOXML ZIP with fixed entry metadata and deterministic deflate."""

    path = Path(path)
    if path.suffix.lower() not in {".docx", ".xlsx", ".pptx"}:
        raise ValueError(f"Expected an OOXML package, got: {path}")

    with ZipFile(path, "r") as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
        archive_comment = source.comment

    if path.suffix.lower() == ".xlsx":
        entries = _canonical_tokens(entries)

    names = [info.filename for info, _ in entries]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate ZIP member names in {path}")

    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with ZipFile(
            temporary,
            "w",
            compression=ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=True,
        ) as target:
            target.comment = archive_comment
            for original, payload in entries:
                normalized = ZipInfo(original.filename, FIXED_ZIP_TIMESTAMP)
                normalized.compress_type = ZIP_DEFLATED
                normalized.comment = original.comment
                normalized.extra = original.extra
                normalized.create_system = original.create_system
                normalized.create_version = original.create_version
                normalized.extract_version = original.extract_version
                normalized.internal_attr = original.internal_attr
                normalized.external_attr = original.external_attr
                normalized.volume = original.volume
                target.writestr(normalized, payload, compress_type=ZIP_DEFLATED, compresslevel=9)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize DOCX/XLSX/PPTX ZIP timestamps for reproducible hashes."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        normalize_ooxml_zip(path)


if __name__ == "__main__":
    main()
