#!/usr/bin/env python3
"""One-time, hash-checked vendoring of the original user-supplied inputs.

The forecast build must never depend on a particular user's Downloads or
Codex-attachment directory.  This helper exists only to install the immutable
original bytes into ``sources/``; ordinary builds read the vendored files
directly and do not invoke this script.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
HOME = Path.home()
ATTACHMENTS = HOME / ".codex/attachments"

INPUTS = {
    ROOT / "sources/input_eci_parameter_regression_workbook_2026-07-17.xlsx": (
        HOME / "Downloads/eci_parameter_regression_workbook.xlsx",
        "e085ec57793c1bed7231df03035f880000e41581700eb868152d41fe6406a3dc",
    ),
    ROOT / "sources/input_epoch_ai_models_archive_2026-07-17.zip": (
        HOME / "Downloads/ai_models.zip",
        "835d88548f1db982983d2f919131bfceae8ecb5f232eb82572372b05547b3b8a",
    ),
    ROOT / "sources/input_no_cot_arxiv_2606.07157v3_source.tar.gz": (
        HOME / "Downloads/arXiv-2606.07157v3.tar.gz",
        "c1c38d4ac51dda5061a3da8d0322b984409bf77e6c91e114d03b4d534b2f5bb8",
    ),
    ROOT / "sources/input_artificial_analysis_leaderboard_2026-07-17.txt": (
        ATTACHMENTS / "238ade1d-cc43-4361-b8c1-9d9660b798ea/pasted-text.txt",
        "56da88142181cfe20e053a8b614010c7a25519d9f79cb4a755ba4961669cd478",
    ),
    ROOT / "sources/input_lesswrong_no_cot_article_2026-07-17.txt": (
        ATTACHMENTS / "0f45b0c5-4b8c-4916-b6d7-b6bc53dcc26f/pasted-text.txt",
        "0d29897af81f98ba6f8cf76c076b7568793dc98a2cfe74ee22c3e2765aea4944",
    ),
    ROOT / "sources/input_openai_api_pricing_docs_2026-07-17.txt": (
        ATTACHMENTS / "7fd90f78-7cc0-4a40-bac0-2132738152d1/pasted-text.txt",
        "5a9febe195200f7e8ec0ce5339da983af0e11a2a94c45263305bfeb77e57f796",
    ),
    ROOT / "sources/input_anthropic_api_pricing_docs_2026-07-17.txt": (
        ATTACHMENTS / "d61abc72-44c9-4f25-a5d4-d08de62acff7/pasted-text.txt",
        "f006686d30af7ba6613abfc28864f7c09c92d1bea5e2dc354e558c828cd16070",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for destination, (source, expected) in INPUTS.items():
        if destination.exists():
            actual = sha256(destination)
            if actual != expected:
                raise ValueError(f"Vendored input changed: {destination}: {actual} != {expected}")
            continue
        if not source.is_file():
            raise FileNotFoundError(f"Original input missing for one-time vendoring: {source}")
        actual = sha256(source)
        if actual != expected:
            raise ValueError(f"Original input hash drifted: {source}: {actual} != {expected}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256(destination) != expected:
            raise ValueError(f"Copy verification failed: {destination}")
        print(f"vendored {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
