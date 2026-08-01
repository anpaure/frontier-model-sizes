#!/usr/bin/env python3
"""Reconstruct and fit every ECI-era Epoch archive with one pinned codebase.

This script deliberately uses the current official ``eci-public`` code for
every vintage.  That holds the estimator fixed while the archived data vary,
which is the comparison needed for leakage-free archive-vintage backtests.
Five older benchmark ZIPs are retained in the source archive but are not fit
because they predate ``epoch_capabilities_index.csv``.

The official repository is not vendored.  Pass a checkout at the pinned
commit with ``--eci-public-dir`` (or let the script clone it into a temporary
directory).  SciPy and tqdm must be available to the invoking interpreter.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


# Imported lazily for an explicit numerical refit.  The normal
# ``--verify-existing`` pipeline gate intentionally requires only the standard
# library, so it can validate frozen hashes in the bundled workspace Python.
pd: Any


ROOT = Path(__file__).resolve().parent
DATE = "2026-07-18"
SOURCES = ROOT / "sources"
ARCHIVE = SOURCES / f"epoch_eci_historical_snapshots_{DATE}.tar.gz"
COLLECTION_METADATA = SOURCES / f"epoch_eci_historical_collection_metadata_{DATE}.json"
ARCHIVAL_TERMINAL_SCORES = SOURCES / f"epoch_eci_reproduced_scores_{DATE}.csv"
CURRENT_LIVE_SCORES = SOURCES / "epoch_eci_reproduced_scores_2026-07-31.csv"
OUTPUT_SCORES = SOURCES / f"epoch_eci_historical_model_scores_{DATE}.csv"
OUTPUT_BENCHMARKS = SOURCES / f"epoch_eci_historical_benchmark_parameters_{DATE}.csv"
OUTPUT_INPUTS = SOURCES / f"epoch_eci_historical_reconstructed_inputs_{DATE}.csv.gz"
OUTPUT_METADATA = SOURCES / f"epoch_eci_historical_fit_metadata_{DATE}.json"

OFFICIAL_REPOSITORY = "https://github.com/epoch-research/eci-public.git"
OFFICIAL_COMMIT = "542567e72a415b72624e5bbd12603cfd3f485179"
EXPECTED_SNAPSHOTS = 15
EXPECTED_ZIP_SNAPSHOTS = 10
EXPECTED_CANONICAL_SNAPSHOTS = 5
LATEST_TIMESTAMP = "20260716153134"
ARCHIVAL_TERMINAL_CANONICAL = SOURCES / "epoch_eci_benchmarks_2026-07-17.csv"
CURRENT_LIVE_CANONICAL = SOURCES / "epoch_eci_benchmarks_2026-07-31.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def official_checkout(requested: Path | None, work: Path) -> Path:
    if requested is None:
        checkout = work / "eci-public"
        git("clone", "--quiet", OFFICIAL_REPOSITORY, str(checkout))
        git("checkout", "--quiet", OFFICIAL_COMMIT, cwd=checkout)
    else:
        checkout = requested.resolve()
    head = git("rev-parse", "HEAD", cwd=checkout)
    if head != OFFICIAL_COMMIT:
        raise ValueError(f"eci-public commit mismatch: {head} != {OFFICIAL_COMMIT}")
    if git("status", "--porcelain", cwd=checkout):
        raise ValueError("eci-public checkout has uncommitted changes")
    return checkout


def load_official_modules(checkout: Path):
    sys.path.insert(0, str(checkout / "src"))
    try:
        dataloader = importlib.import_module("eci.dataloader")
        fitting = importlib.import_module("eci.fitting")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Official ECI dependencies are missing; install scipy and tqdm in "
            "the invoking Python environment"
        ) from error
    return dataloader, fitting


def zip_frames(zip_path: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if name.endswith(".csv") and not name.startswith("additional_eci_data/"):
                with archive.open(name) as handle:
                    frames[Path(name).name] = pd.read_csv(handle)
    return frames


def prepare_zip_snapshot(
    zip_path: Path, dataloader: Any
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Run the official data-loader with one explicit old-schema shim."""
    frames = zip_frames(zip_path)
    shims: list[dict[str, str]] = []
    weird = frames.get("weirdml_external.csv")
    original_score_col = dataloader.EXTERNAL_BENCHMARKS["weirdml_external.csv"][
        "score_col"
    ]
    if weird is not None and original_score_col not in weird.columns:
        if "Average" not in weird.columns:
            raise ValueError(
                f"Unknown WeirdML schema in {zip_path.name}: {list(weird.columns)}"
            )
        dataloader.EXTERNAL_BENCHMARKS["weirdml_external.csv"]["score_col"] = "Average"
        shims.append(
            {
                "file": "weirdml_external.csv",
                "current_column": original_score_col,
                "archived_column": "Average",
                "operation": "column-name compatibility only; values unchanged",
            }
        )
    try:
        versions = dataloader.load_model_versions(frames)
        internal = dataloader.load_internal_benchmarks(frames)
        external = dataloader.load_external_benchmarks(frames)
        scores = pd.concat([internal, external], ignore_index=True)
        scores = dataloader.apply_random_baseline_correction(scores)
        scores = scores[
            (scores["performance"] >= 0)
            & (scores["performance"] <= 1)
            & scores["performance"].notna()
        ]
        scores = scores.merge(versions, on="model_version", how="inner")
        minimum = pd.Timestamp("2023-01-01")
        scores = scores[(scores["date"] >= minimum) | scores["date"].isna()]
        scores = scores.sort_values(["Model", "date"], ascending=[True, False])
        counts = scores.groupby("Model")["benchmark"].nunique()
        scores = scores[scores["Model"].isin(counts[counts >= 4].index)]
        scores = dataloader.add_benchmark_metadata(scores)
        benchmark_ids = {
            name: f"b{index + 1}"
            for index, name in enumerate(scores["benchmark"].unique())
        }
        model_ids = {
            name: f"m{index + 1}"
            for index, name in enumerate(scores["Model"].unique())
        }
        scores["benchmark_id"] = scores["benchmark"].map(benchmark_ids)
        scores["model_id"] = scores["Model"].map(model_ids)
        result = (
            scores.groupby(["model_id", "benchmark_id"])
            .agg(
                performance=("performance", "max"),
                benchmark=("benchmark", "first"),
                benchmark_release_date=("benchmark_release_date", "first"),
                model=("Model", "first"),
                model_version=("model_version", "first"),
                Model=("Model", "first"),
                date=("date", "max"),
                source=("source", "first"),
            )
            .reset_index()
        )
        result = result[
            [
                "model_id",
                "benchmark_id",
                "performance",
                "benchmark",
                "benchmark_release_date",
                "model",
                "model_version",
                "Model",
                "date",
                "source",
            ]
        ]
    finally:
        dataloader.EXTERNAL_BENCHMARKS["weirdml_external.csv"][
            "score_col"
        ] = original_score_col
    return result, shims


def snapshot_date(timestamp: str) -> str:
    return datetime.strptime(timestamp[:8], "%Y%m%d").date().isoformat()


def annotate(frame: pd.DataFrame, timestamp: str, kind: str) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "snapshot_timestamp", timestamp)
    result.insert(1, "snapshot_date", snapshot_date(timestamp))
    result.insert(2, "snapshot_kind", kind)
    return result


def deterministic_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    payload = frame.to_csv(index=False, lineterminator="\n").encode()
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))


def validate_collection_boundary(collection: dict[str, Any], archive: Path) -> None:
    """Keep the frozen archive and the newer live panel in separate vintages."""
    if sha256(archive) != collection["archive_sha256"]:
        raise ValueError("Historical archive hash differs from collection metadata")
    policy = collection.get("archive_policy", {})
    if policy.get("terminal_timestamp") != LATEST_TIMESTAMP:
        raise ValueError("Historical collection has the wrong terminal timestamp")
    terminal = collection.get("archival_terminal_exact_match", {})
    if terminal.get("pinned_file") != str(ARCHIVAL_TERMINAL_CANONICAL.relative_to(ROOT)):
        raise ValueError("Historical collection does not pin the archival terminal source")
    if terminal.get("sha256") != sha256(ARCHIVAL_TERMINAL_CANONICAL):
        raise ValueError("Archival terminal source hash differs from collection metadata")
    successor = collection.get("current_live_successor_reference", {})
    if successor.get("pinned_file") != str(CURRENT_LIVE_CANONICAL.relative_to(ROOT)):
        raise ValueError("Historical collection lacks the current-live successor reference")
    if successor.get("byte_equality_assertion_against_archival_terminal") is not False:
        raise ValueError("Current live data cannot be an archival byte-equality target")


def verify_existing_outputs(archive: Path) -> dict[str, Any]:
    """Verify the frozen refit without importing SciPy or cloning code.

    A normal reproducible build should validate immutable bytes, not require a
    numerical stack and network checkout merely to rewrite identical outputs.
    ``--verify-existing`` is therefore the default-pipeline contract; an
    explicit refit still exercises the pinned official implementation.
    """

    metadata = json.loads(OUTPUT_METADATA.read_text(encoding="utf-8"))
    if metadata["official_commit"] != OFFICIAL_COMMIT:
        raise ValueError("Historical fit metadata commit mismatch")
    if metadata["source_archive"] != str(archive.relative_to(ROOT)):
        raise ValueError("Historical fit metadata archive path mismatch")
    if metadata["source_archive_sha256"] != sha256(archive):
        raise ValueError("Historical fit metadata archive hash mismatch")
    inventory = metadata["inventory"]
    if len(inventory) != EXPECTED_SNAPSHOTS:
        raise ValueError("Historical fit metadata snapshot inventory mismatch")
    if max(row["snapshot_timestamp"] for row in inventory) != LATEST_TIMESTAMP:
        raise ValueError("Historical fit metadata terminal timestamp mismatch")
    terminal = metadata["archival_terminal_fit_crosscheck"]
    if (
        terminal["timestamp"] != LATEST_TIMESTAMP
        or terminal["archival_terminal_scores"]
        != str(ARCHIVAL_TERMINAL_SCORES.relative_to(ROOT))
        or not terminal["exact_within_1e_8"]
        or terminal["archival_terminal_scores_sha256"]
        != sha256(ARCHIVAL_TERMINAL_SCORES)
    ):
        raise ValueError("Historical fit terminal crosscheck is stale")
    successor = metadata["current_live_successor_reference"]
    if (
        successor["scores"] != str(CURRENT_LIVE_SCORES.relative_to(ROOT))
        or successor["scores_sha256"] != sha256(CURRENT_LIVE_SCORES)
        or successor["byte_equality_or_fit_assertion_against_archival_terminal"]
        is not False
    ):
        raise ValueError("Historical fit live-successor reference is stale")
    expected_outputs = {
        str(OUTPUT_SCORES.relative_to(ROOT)),
        str(OUTPUT_BENCHMARKS.relative_to(ROOT)),
        str(OUTPUT_INPUTS.relative_to(ROOT)),
    }
    if set(metadata["outputs"]) != expected_outputs:
        raise ValueError("Historical fit output inventory mismatch")
    for relative, record in metadata["outputs"].items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Historical fit output changed: {relative}")
    return {
        "mode": "verify-existing",
        "snapshots": len(inventory),
        "terminal": LATEST_TIMESTAMP,
        "outputs": len(metadata["outputs"]),
        "official_commit": OFFICIAL_COMMIT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--eci-public-dir", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()

    collection = json.loads(COLLECTION_METADATA.read_text())
    validate_collection_boundary(collection, args.archive)
    if args.verify_existing:
        print(json.dumps(verify_existing_outputs(args.archive), indent=2))
        return

    global pd
    pd = importlib.import_module("pandas")

    score_frames: list[pd.DataFrame] = []
    benchmark_frames: list[pd.DataFrame] = []
    input_frames: list[pd.DataFrame] = []
    inventory: list[dict[str, Any]] = []
    shims: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="eci-historical-") as temporary:
        work = Path(temporary)
        checkout = official_checkout(args.eci_public_dir, work)
        dataloader, fitting = load_official_modules(checkout)
        extracted = work / "snapshots"
        extracted.mkdir()
        with tarfile.open(args.archive) as archive:
            archive.extractall(extracted, filter="data")

        members = sorted(extracted.rglob("*"))
        members = [path for path in members if path.is_file()]
        for path in members:
            timestamp = path.stem.rsplit("_", 1)[-1]
            if path.suffix == ".zip":
                with zipfile.ZipFile(path) as archive:
                    if "epoch_capabilities_index.csv" not in archive.namelist():
                        continue
                frame, applied = prepare_zip_snapshot(path, dataloader)
                kind = "benchmark_zip_fixed_code_reconstruction"
                for shim in applied:
                    shims.append({"snapshot_timestamp": timestamp, **shim})
            elif path.suffix == ".csv":
                frame = pd.read_csv(path)
                kind = "canonical_eci_csv"
            else:
                raise ValueError(f"Unexpected archive member: {path}")

            required = {
                "model_id",
                "benchmark_id",
                "performance",
                "benchmark",
                "Model",
                "date",
            }
            if not required <= set(frame.columns):
                raise ValueError(f"Malformed reconstructed frame for {timestamp}")
            if frame.duplicated(["model_id", "benchmark_id"]).any():
                raise ValueError(f"Duplicate model/benchmark rows for {timestamp}")
            # Mirror official ``load_benchmark_data`` exactly: the CSV's
            # capitalized column is a coarser family label, while ``model``
            # is the release-dated aggregation that the ECI fit identifies.
            frame["Model"] = frame["model"]
            eci, edi, _ = fitting.fit_eci_model(frame, bootstrap_samples=0)
            metadata = (
                frame.sort_values("date")
                .drop_duplicates("model_id", keep="last")
                .set_index("model_id")
            )
            eci["date"] = eci["model_id"].map(metadata["date"])
            eci["model_version"] = eci["model_id"].map(metadata["model_version"])
            eci["source"] = eci["model_id"].map(metadata["source"])
            score_frames.append(annotate(eci, timestamp, kind))
            benchmark_frames.append(annotate(edi, timestamp, kind))
            input_frames.append(annotate(frame, timestamp, kind))
            inventory.append(
                {
                    "snapshot_timestamp": timestamp,
                    "snapshot_date": snapshot_date(timestamp),
                    "kind": kind,
                    "input_rows": int(len(frame)),
                    "models": int(frame["Model"].nunique()),
                    "benchmarks": int(frame["benchmark"].nunique()),
                    "applied_schema_shims": len(applied) if path.suffix == ".zip" else 0,
                }
            )

        scores = pd.concat(score_frames, ignore_index=True)
        benchmarks = pd.concat(benchmark_frames, ignore_index=True)
        inputs = pd.concat(input_frames, ignore_index=True)
        scores = scores.sort_values(["snapshot_timestamp", "eci", "Model"], ascending=[True, False, True])
        benchmarks = benchmarks.sort_values(["snapshot_timestamp", "edi", "benchmark"])
        inputs = inputs.sort_values(["snapshot_timestamp", "model_id", "benchmark_id"])

        if len(inventory) != EXPECTED_SNAPSHOTS:
            raise ValueError(f"Expected {EXPECTED_SNAPSHOTS} ECI vintages; got {len(inventory)}")
        kinds = pd.Series([row["kind"] for row in inventory]).value_counts().to_dict()
        if kinds.get("benchmark_zip_fixed_code_reconstruction") != EXPECTED_ZIP_SNAPSHOTS:
            raise ValueError("Historical ZIP snapshot count mismatch")
        if kinds.get("canonical_eci_csv") != EXPECTED_CANONICAL_SNAPSHOTS:
            raise ValueError("Historical canonical snapshot count mismatch")

        latest = scores[scores["snapshot_timestamp"] == LATEST_TIMESTAMP].set_index("Model")
        archival_terminal = pd.read_csv(ARCHIVAL_TERMINAL_SCORES).set_index("Model")
        if set(latest.index) != set(archival_terminal.index):
            raise ValueError(
                "Latest historical fit model inventory differs from archival-terminal reproduction"
            )
        maximum_difference = float(
            (
                latest.loc[archival_terminal.index, "eci"]
                - archival_terminal["eci"]
            )
            .abs()
            .max()
        )
        if maximum_difference > 1e-8:
            raise ValueError(
                "Latest historical fit differs from archival-terminal reproduction: "
                f"{maximum_difference}"
            )

        scores.to_csv(OUTPUT_SCORES, index=False, lineterminator="\n")
        benchmarks.to_csv(OUTPUT_BENCHMARKS, index=False, lineterminator="\n")
        deterministic_csv_gzip(inputs, OUTPUT_INPUTS)

        code_files = [
            checkout / "src/eci/dataloader.py",
            checkout / "src/eci/fitting.py",
        ]
        metadata = {
            "generated_on": DATE,
            "method": "all vintages fit with one pinned current official estimator",
            "official_repository": OFFICIAL_REPOSITORY,
            "official_commit": OFFICIAL_COMMIT,
            "official_code_sha256": {
                str(path.relative_to(checkout)): sha256(path) for path in code_files
            },
            "source_archive": str(args.archive.relative_to(ROOT)),
            "source_archive_sha256": sha256(args.archive),
            "inventory": inventory,
            "schema_compatibility_shims": shims,
            "pre_eci_zip_snapshots_preserved_but_not_fit": 5,
            "archival_terminal_fit_crosscheck": {
                "timestamp": LATEST_TIMESTAMP,
                "models": int(len(latest)),
                "maximum_absolute_eci_difference": maximum_difference,
                "exact_within_1e_8": True,
                "archival_terminal_scores": str(
                    ARCHIVAL_TERMINAL_SCORES.relative_to(ROOT)
                ),
                "archival_terminal_scores_sha256": sha256(ARCHIVAL_TERMINAL_SCORES),
            },
            "current_live_successor_reference": {
                "scores": str(CURRENT_LIVE_SCORES.relative_to(ROOT)),
                "scores_sha256": sha256(CURRENT_LIVE_SCORES),
                "role": "newer live source; not compared to the frozen archive terminal",
                "byte_equality_or_fit_assertion_against_archival_terminal": False,
            },
            "outputs": {},
        }
        for path in (OUTPUT_SCORES, OUTPUT_BENCHMARKS, OUTPUT_INPUTS):
            metadata["outputs"][str(path.relative_to(ROOT))] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        OUTPUT_METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "snapshots": len(inventory),
                "score_rows": len(scores),
                "input_rows": len(inputs),
                "latest_maximum_difference": maximum_difference,
                "schema_shims": shims,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
