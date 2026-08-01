#!/usr/bin/env python3
"""Canonicalize ephemeral artifact-tool IDs in workbook inspection NDJSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


RUNTIME_HANDLE = re.compile(r"^(?P<prefix>wb|ws|ch|sh|img|tbl|th)/[^/]+$")
GUID = re.compile(
    r"\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}"
)
IDENTIFIER_KEYS = {
    "id",
    "authorId",
    "commentId",
    "drawingId",
    "personId",
    "sheetId",
    "threadId",
    "workbookId",
}
FIXED_COMMENT_TIMESTAMP = "2026-07-18T00:00:00.000Z"


def _canonical_guid(index: int) -> str:
    return "{00000000-0000-4000-8000-" f"{index:012x}" "}"


def normalize_inspect_ndjson(path: Path) -> None:
    """Rewrite a diagnostic ledger while retaining every non-ID value exactly."""

    path = Path(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    handle_maps: dict[str, dict[str, str]] = {}
    thread_token_map: dict[str, str] = {}
    guid_map: dict[str, str] = {}

    def canonicalize_identifier(value: str, key: str) -> str:
        handle = RUNTIME_HANDLE.fullmatch(value)
        if handle:
            prefix = handle.group("prefix")
            mapping = handle_maps.setdefault(prefix, {})
            mapping.setdefault(value, f"{prefix}/{len(mapping) + 1:06d}")
            if prefix == "th":
                thread_token_map.setdefault(
                    value.split("/", 1)[1], mapping[value].split("/", 1)[1]
                )
            return mapping[value]
        if key == "id" and value in thread_token_map:
            return thread_token_map[value]

        def replace_guid(match: re.Match[str]) -> str:
            token = match.group(0)
            guid_map.setdefault(token, _canonical_guid(len(guid_map) + 1))
            return guid_map[token]

        return GUID.sub(replace_guid, value)

    def walk(value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {child_key: walk(child, child_key) for child_key, child in value.items()}
        if isinstance(value, list):
            return [walk(child) for child in value]
        if isinstance(value, str) and key == "createdAt":
            return FIXED_COMMENT_TIMESTAMP
        if isinstance(value, str) and key in IDENTIFIER_KEYS:
            return canonicalize_identifier(value, key)
        return value

    normalized = "\n".join(
        json.dumps(walk(record), ensure_ascii=False, separators=(",", ":"))
        for record in records
    ) + "\n"
    path.write_text(normalized, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize ephemeral identifiers in artifact-tool inspection NDJSON."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        normalize_inspect_ndjson(path)


if __name__ == "__main__":
    main()
