from __future__ import annotations

import gzip
import re
import tarfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
# macOS and Windows workstation homes are never valid committed provenance in
# this project.  Linux ``/home/...`` strings are excluded here because several
# immutable upstream model datasets legitimately contain container paths.
POSIX_HOME = re.compile(rb"/Users/[A-Za-z0-9._-]+/")
WINDOWS_HOME = re.compile(rb"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+\\")
SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    ".next",
    ".vinext",
    ".wrangler",
}


def has_private_home_path(data: bytes) -> bool:
    variants = (data, data.replace(b"\x00", b""))
    return any(POSIX_HOME.search(value) or WINDOWS_HOME.search(value) for value in variants)


def publishable_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {"tmp"}:
            continue
        files.append(path)
    return sorted(files)


class LocalPathPrivacyTest(unittest.TestCase):
    def test_text_and_archive_layers_do_not_expose_user_home_paths(self) -> None:
        leaks: list[str] = []
        for path in publishable_files():
            relative = path.relative_to(ROOT).as_posix()
            raw = path.read_bytes()
            if has_private_home_path(raw):
                leaks.append(relative)
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        if member.is_dir():
                            continue
                        if has_private_home_path(archive.read(member)):
                            leaks.append(f"{relative}::{member.filename}")
            elif path.suffix.lower() == ".gz":
                try:
                    payload = gzip.decompress(raw)
                except (gzip.BadGzipFile, EOFError, OSError):
                    continue
                if has_private_home_path(payload):
                    leaks.append(f"{relative}::gzip-payload")
            elif path.suffix.lower() == ".tar" and tarfile.is_tarfile(path):
                with tarfile.open(path) as archive:
                    for member in archive.getmembers():
                        extracted = archive.extractfile(member) if member.isfile() else None
                        if extracted is not None and has_private_home_path(extracted.read()):
                            leaks.append(f"{relative}::{member.name}")
        self.assertEqual(leaks, [], "Private local paths remain:\n" + "\n".join(leaks))


if __name__ == "__main__":
    unittest.main(verbosity=2)
