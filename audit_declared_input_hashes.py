#!/usr/bin/env python3
"""Verify path/hash contracts declared inside generated JSON artifacts.

The forecast pipeline's audit files use several equivalent provenance shapes:

* ``{"path/to/input": "<sha256>"}``
* ``{"path": "path/to/input", "sha256": "<sha256>"}``
* ``{"input_file": "path/to/input", "input_sha256": "<sha256>"}``
* sibling ``source_files`` and ``source_hashes`` maps keyed by source label

This checker recognizes those shapes recursively and verifies the referenced
bytes.  It deliberately does not infer hashes for paths that an artifact did
not declare; its job is to make existing freshness contracts enforceable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = (
    ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DeclaredHash:
    artifact: str
    json_pointer: str
    declared_path: str
    expected_sha256: str


@dataclass(frozen=True)
class HashIssue:
    artifact: str
    json_pointer: str
    declared_path: str
    resolved_path: str
    expected_sha256: str
    actual_sha256: str | None
    issue: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def is_pathlike(value: Any) -> bool:
    if not isinstance(value, str) or not value or "://" in value:
        return False
    path = Path(value)
    return (
        path.is_absolute()
        or "/" in value
        or "\\" in value
        or path.suffix.lower()
        in {
            ".csv",
            ".docx",
            ".gz",
            ".html",
            ".json",
            ".md",
            ".ndjson",
            ".pdf",
            ".txt",
            ".xlsx",
            ".yaml",
            ".yml",
            ".zip",
        }
    )


def pointer(parts: Iterable[str]) -> str:
    encoded = [part.replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(encoded)


def declared_hashes(artifact: Path, payload: Any) -> list[DeclaredHash]:
    """Return every unambiguous local path/SHA-256 pair in ``payload``."""

    records: dict[tuple[str, str], DeclaredHash] = {}

    def add(parts: tuple[str, ...], raw_path: Any, expected: Any) -> None:
        if not is_pathlike(raw_path) or not is_sha256(expected):
            return
        record = DeclaredHash(
            artifact=str(artifact),
            json_pointer=pointer(parts),
            declared_path=str(raw_path),
            expected_sha256=str(expected),
        )
        records[(record.declared_path, record.expected_sha256)] = record

    def walk(value: Any, parts: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            # Direct path -> digest maps, common in source_files/source_manifest.
            # Maps explicitly describing archive-member hashes are semantic
            # checks on bytes inside a parent ZIP/TAR, not local filesystem
            # paths.  Their producing audit verifies extraction; resolving an
            # inner name such as ``ml_models.csv`` against ROOT is a false
            # missing-file error.
            archive_member_hash_map = bool(
                parts
                and re.search(
                    r"(?:^|_)(?:member|members)_sha256$", parts[-1]
                )
            )
            if not archive_member_hash_map:
                for key, child in value.items():
                    add(parts + (str(key),), key, child)

            # Record-shaped declarations.
            expected = value.get("sha256")
            # A selector/value contract hashes the selected JSON value rather
            # than the complete file at ``path``.  Its producer-specific test
            # must validate that semantic hash; treating it as a file digest
            # would be a false freshness failure.
            hashes_selected_value = "selector" in value and "value" in value
            if is_sha256(expected) and not hashes_selected_value:
                for path_key in (
                    "path",
                    "local_path",
                    "file",
                    "source_path",
                    "source_file",
                ):
                    if path_key in value:
                        add(parts + (path_key,), value[path_key], expected)

            # Sibling foo / foo_sha256 declarations.
            for key, child in value.items():
                if not (isinstance(key, str) and key.endswith("_sha256")):
                    continue
                stem = key[: -len("_sha256")]
                candidates = (stem, f"{stem}_path", f"{stem}_file")
                for candidate in candidates:
                    if candidate in value:
                        add(parts + (key,), value[candidate], child)

            # A few manifests use input_file/input_sha256 rather than a shared
            # stem, so make those pairs explicit.
            for path_key, hash_key in (
                ("input_file", "input_sha256"),
                ("output_file", "output_sha256"),
                ("edi_output_file", "edi_output_sha256"),
            ):
                if path_key in value and hash_key in value:
                    add(parts + (hash_key,), value[path_key], value[hash_key])

            # Opus-style evidence bundles keep label -> path and label -> hash
            # in separate sibling maps.
            # Evidence bundles may have several parallel path/hash maps, e.g.
            # source_files/source_hashes and coverage_source_files/
            # coverage_source_hashes.  Pair every unambiguous ``*_files`` map
            # with its sibling ``*_hashes`` map instead of special-casing one.
            for files_key, source_files in value.items():
                if not (
                    isinstance(files_key, str)
                    and files_key.endswith("_files")
                    and isinstance(source_files, dict)
                ):
                    continue
                hashes_key = files_key[: -len("_files")] + "_hashes"
                source_hashes = value.get(hashes_key)
                if not isinstance(source_hashes, dict):
                    continue
                for label, raw_path in source_files.items():
                    add(
                        parts + (hashes_key, str(label)),
                        raw_path,
                        source_hashes.get(label),
                    )

            for key, child in value.items():
                walk(child, parts + (str(key),))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, parts + (str(index),))

    walk(payload)
    return sorted(
        records.values(),
        key=lambda row: (row.artifact, row.declared_path, row.json_pointer),
    )


def resolve_declared_path(raw_path: str, root: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else root / path


def audit_artifacts(
    artifacts: Iterable[Path], *, root: Path = ROOT
) -> tuple[list[DeclaredHash], list[HashIssue]]:
    declarations: list[DeclaredHash] = []
    issues: list[HashIssue] = []
    for artifact in sorted(artifacts):
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        for record in declared_hashes(artifact, payload):
            declarations.append(record)
            resolved = resolve_declared_path(record.declared_path, root)
            if not resolved.is_file():
                issues.append(
                    HashIssue(
                        **asdict(record),
                        resolved_path=str(resolved),
                        actual_sha256=None,
                        issue="missing",
                    )
                )
                continue
            actual = sha256(resolved)
            if actual != record.expected_sha256:
                issues.append(
                    HashIssue(
                        **asdict(record),
                        resolved_path=str(resolved),
                        actual_sha256=actual,
                        issue="sha256_mismatch",
                    )
                )
    return declarations, issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify every local path/SHA-256 contract declared by generated JSON audits."
    )
    parser.add_argument(
        "artifact_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing generated JSON audit artifacts.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()
    artifacts = sorted(args.artifact_dir.glob("*.json"))
    declarations, issues = audit_artifacts(artifacts)
    result = {
        "artifact_directory": str(args.artifact_dir),
        "json_artifacts": len(artifacts),
        "declared_local_hashes": len(declarations),
        "issues": [asdict(issue) for issue in issues],
        "status": "PASS" if not issues else "FAIL",
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"{result['status']}: {len(declarations)} declared local hashes across "
            f"{len(artifacts)} JSON artifacts; {len(issues)} issue(s)."
        )
        for issue in issues:
            print(
                f"{issue.issue}: {issue.artifact}{issue.json_pointer} -> "
                f"{issue.declared_path}"
            )
    raise SystemExit(0 if not issues else 1)


if __name__ == "__main__":
    main()
