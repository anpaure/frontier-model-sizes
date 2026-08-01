"""Portable path serialization for generated project artifacts.

Runtime code may use absolute paths internally, but committed provenance must
not expose the local account name or workstation layout.  Paths inside this
project are serialized relative to the repository root; paths elsewhere in the
current home directory use ``~/``.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def portable_path(value: str | Path, *, root: Path = PROJECT_ROOT) -> str:
    resolved = Path(value).expanduser().resolve()
    root = root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        return f"~/{resolved.relative_to(Path.home().resolve()).as_posix()}"
    except ValueError:
        return resolved.as_posix()

