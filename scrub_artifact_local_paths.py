#!/usr/bin/env python3
"""Replace local absolute paths with portable forms in publishable artifacts."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path


def scrub_bytes(value: bytes, root: Path) -> bytes:
    output = value
    # Apply the more-specific repository root first.  Any remaining paths under
    # the current user's home are made portable without preserving the account
    # name in a publishable package.
    replacements = (
        (root.resolve(), b"./", b".\\"),
        (Path.home().resolve(), b"~/", b"~\\"),
    )
    for prefix_path, forward_replacement, backward_replacement in replacements:
        forward = str(prefix_path).encode("utf-8")
        backward = str(prefix_path).replace("/", "\\").encode("utf-8")
        output = output.replace(forward + b"/", forward_replacement)
        output = output.replace(backward + b"\\", backward_replacement)
    return output


def scrub(path: Path, root: Path) -> int:
    replacements = 0
    if zipfile.is_zipfile(path):
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED
            ) as target:
                for info in source.infolist():
                    original = source.read(info.filename)
                    redacted = scrub_bytes(original, root)
                    replacements += original != redacted
                    target.writestr(info, redacted)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        original = path.read_bytes()
        redacted = scrub_bytes(original, root)
        replacements = int(original != redacted)
        path.write_bytes(redacted)
    return replacements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if not args.artifact.is_file():
        raise FileNotFoundError(args.artifact)
    changed_members = scrub(args.artifact, args.root)
    print(f"Scrubbed {args.artifact}: {changed_members} member(s)/file changed")


if __name__ == "__main__":
    main()
