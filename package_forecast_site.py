#!/usr/bin/env python3
"""Create the tracked production-site archive deterministically."""

from __future__ import annotations

import gzip
import hashlib
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
DIST = SITE / "dist"
OUTPUT = ROOT / "outputs" / "frontier-parameter-lab-site.tar.gz"


def normalized_info(archive: tarfile.TarFile, path: Path) -> tarfile.TarInfo:
    info = archive.gettarinfo(str(path), arcname=path.relative_to(SITE).as_posix())
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.pax_headers = {}
    return info


def build() -> None:
    if not DIST.is_dir():
        raise FileNotFoundError(f"Build the production site first: {DIST}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{OUTPUT.name}.", suffix=".tmp", dir=OUTPUT.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                ) as archive:
                    paths = [DIST, *sorted(DIST.rglob("*"), key=lambda p: p.as_posix())]
                    for path in paths:
                        info = normalized_info(archive, path)
                        if info.isfile():
                            with path.open("rb") as source:
                                archive.addfile(info, source)
                        else:
                            archive.addfile(info)
        temporary.replace(OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print({"archive": str(OUTPUT), "sha256": digest})


if __name__ == "__main__":
    build()
