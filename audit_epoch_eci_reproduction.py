#!/usr/bin/env python3
"""Audit the frozen official ECI reproduction and derive release-date policy.

Epoch's canonical ``eci_benchmarks.csv`` is the correct input for reproducing
ECI scores, but its ``date`` field is not always the release date of the exact
checkpoint named by ``model_version``.  The published capabilities archive has
a checkpoint-level ``Release date`` column.  This audit preserves both fields
and defines the chronological-regression date as the published release date,
falling back to the canonical ECI-input date only when the published date is
blank.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from manage_epoch_snapshot import MANIFEST, verify_manifest


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
ECI_INPUT = ROOT / "sources/epoch_eci_benchmarks_2026-07-31.csv"
PUBLISHED_ARCHIVE = ROOT / "sources/epoch_benchmark_data_2026-07-31.zip"
REPRODUCED = ROOT / "sources/epoch_eci_reproduced_scores_2026-07-31.csv"
METADATA = ROOT / "sources/epoch_eci_reproduction_metadata_2026-07-31.json"
CROSSCHECK = OUT / "epoch_eci_reproduction_crosscheck_2026-07-31.csv"
AUDIT = OUT / "epoch_eci_reproduction_audit_2026-07-31.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty ECI reproduction crosscheck")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_published() -> list[dict[str, str]]:
    with zipfile.ZipFile(PUBLISHED_ARCHIVE) as archive:
        raw = archive.read("epoch_capabilities_index.csv").decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(raw)))


def optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def main() -> None:
    manifest = verify_manifest()
    eci_input = read_csv(ECI_INPUT)
    reproduced = read_csv(REPRODUCED)
    published = load_published()
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))

    input_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in eci_input:
        input_by_model[row["model"]].append(row)
    reproduced_by_version = {row["model_version"]: row for row in reproduced}
    reproduced_by_model = {row["Model"]: row for row in reproduced}
    published_by_version = {row["Model version"]: row for row in published}

    if len(eci_input) != 2059 or len(input_by_model) != 213:
        raise RuntimeError("Canonical ECI input inventory changed")
    if len(reproduced) != 213 or len(reproduced_by_version) != 213 or len(reproduced_by_model) != 213:
        raise RuntimeError("Official ECI reproduction is not one row per input checkpoint")
    if set(reproduced_by_model) != set(input_by_model):
        raise RuntimeError("Official ECI reproduction does not exactly cover the canonical input models")
    if not set(reproduced_by_version) <= set(published_by_version):
        raise RuntimeError("Published Epoch archive is missing reproduced ECI model versions")

    if sha256(ECI_INPUT) != metadata["input_sha256"]:
        raise RuntimeError("ECI input hash does not match reproduction metadata")
    if sha256(REPRODUCED) != metadata["output_sha256"]:
        raise RuntimeError("Reproduced ECI hash does not match reproduction metadata")

    crosscheck: list[dict[str, Any]] = []
    score_differences: list[float] = []
    published_score_matches = 0
    published_blank_scores = 0
    date_disagreements = 0
    date_fallbacks = 0

    for version in sorted(reproduced_by_version):
        row = reproduced_by_version[version]
        canonical_rows = input_by_model[row["Model"]]
        canonical_models = {item["model"] for item in canonical_rows}
        canonical_dates = {item["date"] for item in canonical_rows}
        canonical_versions = {item["model_version"] for item in canonical_rows}
        if canonical_models != {row["Model"]} or canonical_dates != {row["date"]} or version not in canonical_versions:
            raise RuntimeError(f"Canonical identity/date mismatch for {version}")

        published_row = published_by_version[version]
        published_date = published_row["Release date"]
        if published_date:
            regression_date = published_date
            date_policy = "published capabilities release date"
            delta_days = (date.fromisoformat(published_date) - date.fromisoformat(row["date"])).days
            if delta_days:
                date_disagreements += 1
        else:
            regression_date = row["date"]
            date_policy = "canonical ECI-input date fallback"
            delta_days = None
            date_fallbacks += 1

        published_score = optional_float(published_row["ECI Score"])
        if published_score is None:
            published_blank_scores += 1
            score_difference = None
        else:
            published_score_matches += 1
            score_difference = abs(float(row["eci"]) - published_score)
            score_differences.append(score_difference)
            if score_difference > 0.01 + 1e-12:
                raise RuntimeError(f"Published ECI score mismatch for {version}: {score_difference}")

        is_anchor = row["Model"] in {"Claude 3.5 Sonnet", "GPT-5"}
        ci_low = optional_float(row["eci_ci_low"])
        ci_high = optional_float(row["eci_ci_high"])
        if is_anchor:
            if ci_low is not None or ci_high is not None:
                raise RuntimeError(f"Anchor interval should be blank for {row['Model']}")
        else:
            if ci_low is None or ci_high is None:
                raise RuntimeError(f"Non-anchor interval missing for {row['Model']}")
            if not (ci_low <= float(row["eci"]) <= ci_high):
                raise RuntimeError(f"Interval does not contain score for {row['Model']}")

        crosscheck.append(
            {
                "model": row["Model"],
                "model_version": version,
                "eci_score_reproduced": row["eci"],
                "eci_ci_low_reproduced": row["eci_ci_low"],
                "eci_ci_high_reproduced": row["eci_ci_high"],
                "eci_input_date": row["date"],
                "published_release_date": published_date,
                "regression_release_date": regression_date,
                "published_minus_input_days": "" if delta_days is None else delta_days,
                "release_date_policy": date_policy,
                "published_eci_score": "" if published_score is None else published_score,
                "absolute_published_score_difference": "" if score_difference is None else score_difference,
                "eci_input_source": row["source"],
            }
        )

    anchors = {row["Model"]: float(row["eci"]) for row in reproduced if row["Model"] in {"Claude 3.5 Sonnet", "GPT-5"}}
    if anchors != {"Claude 3.5 Sonnet": 130.0, "GPT-5": 150.0}:
        raise RuntimeError(f"Unexpected ECI anchors: {anchors}")

    write_csv(CROSSCHECK, crosscheck)
    audit = {
        "schema_version": "2.0",
        "snapshot_as_of": manifest["snapshot_as_of"],
        "reproduction": {
            "official_repository": metadata["source_repository"],
            "official_commit": metadata["source_commit"],
            "bootstrap_samples": metadata["bootstrap_samples"],
            "bootstrap_seed": metadata["bootstrap_seed"],
            "input_rows": len(eci_input),
            "input_models": len(input_by_model),
            "input_benchmarks": len({row["benchmark"] for row in eci_input}),
            "reproduced_models": len(reproduced),
            "anchors": anchors,
            "all_models_exactly_covered": set(reproduced_by_model) == set(input_by_model),
        },
        "published_score_crosscheck": {
            "published_nonblank_scores": published_score_matches,
            "published_blank_scores": published_blank_scores,
            "maximum_absolute_score_difference": max(score_differences),
            "tolerance": 0.01,
            "all_nonblank_scores_match_within_display_rounding": all(value <= 0.01 + 1e-12 for value in score_differences),
        },
        "release_date_crosscheck": {
            "published_release_dates_used": len(reproduced) - date_fallbacks,
            "canonical_input_date_fallbacks": date_fallbacks,
            "published_vs_input_date_disagreements": date_disagreements,
            "policy": "Use the published checkpoint-level Release date for chronological regression; preserve and use the canonical ECI-input date only when the published date is blank.",
            "known_fallback_models": [row["model"] for row in crosscheck if row["release_date_policy"].endswith("fallback")],
        },
        "upstream_test_context": {
            "fit_implementation": "Pinned official Epoch eci-public code",
            "snapshot_manifest_verified": True,
            "manifest": str(MANIFEST.relative_to(ROOT)),
        },
        "files": {
            "eci_input": str(ECI_INPUT.relative_to(ROOT)),
            "published_archive": str(PUBLISHED_ARCHIVE.relative_to(ROOT)),
            "reproduced_scores": str(REPRODUCED.relative_to(ROOT)),
            "reproduction_metadata": str(METADATA.relative_to(ROOT)),
            "crosscheck": str(CROSSCHECK.relative_to(ROOT)),
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (ECI_INPUT, PUBLISHED_ARCHIVE, REPRODUCED, METADATA, MANIFEST, CROSSCHECK)
        },
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(AUDIT),
                "reproduced_models": len(reproduced),
                "published_score_matches": published_score_matches,
                "release_date_disagreements_preserved": date_disagreements,
                "release_date_fallbacks": date_fallbacks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
