"""Exact current Artificial Analysis signals for forecast targets.

Every consumer reads the same frozen detailed-AA rows by explicit slug.  This
prevents integer display values (for example 60 or 59) from silently replacing
the exact model inputs used in regressions and sensitivity analyses.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
AA_DETAILED_PATH = ROOT / "sources/aa_detailed_model_signals_2026-07-31.csv"
TARGET_SLUGS = {
    "Claude Fable 5": "claude-fable-5",
    "GPT-5.6 Sol": "gpt-5-6-sol",
    "Kimi K3": "kimi-k3",
    "Claude Opus 4.8": "claude-opus-4-8",
    "GPT-5.5": "gpt-5-5",
    "GPT-5.6 Terra": "gpt-5-6-terra",
    "Claude Sonnet 5": "claude-sonnet-5",
    "GPT-5.6 Luna": "gpt-5-6-luna",
    "Grok 4.5": "grok-4-5",
    "Claude Opus 5": "claude-opus-5",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_target_signals() -> dict[str, dict[str, Any]]:
    with AA_DETAILED_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 587 or len({row["model_id"] for row in rows}) != 587:
        raise ValueError("Detailed AA source inventory is not the pinned 587-record snapshot")
    by_slug = {row["slug"]: row for row in rows}
    output: dict[str, dict[str, Any]] = {}
    for model, slug in TARGET_SLUGS.items():
        row = by_slug.get(slug)
        if row is None or not row["intelligence_index"] or not row["release_date"]:
            raise ValueError(f"Missing exact AA target signal: {model} -> {slug}")
        output[model] = {
            "model": model,
            "slug": slug,
            "model_id": row["model_id"],
            "source_name": row["name"],
            "release_date": row["release_date"],
            "score": float(row["intelligence_index"]),
            "output_tokens_total": (
                int(float(row["intelligence_output_tokens_total"]))
                if row["intelligence_output_tokens_total"]
                else None
            ),
            "source_path": str(AA_DETAILED_PATH.relative_to(ROOT)),
            "source_sha256": sha256(AA_DETAILED_PATH),
        }
    return output


AA_TARGET_SIGNALS = load_target_signals()
