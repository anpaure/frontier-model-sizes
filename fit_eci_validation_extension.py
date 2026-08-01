#!/usr/bin/env python3
"""Fit the pinned July 22 ECI score-vintage holdout with official code."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import collect_eci_validation_extension as collector


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sources" / "epoch_eci_benchmarks_2026-07-22.csv"
OUTPUT = ROOT / "sources" / "epoch_eci_validation_extension_scores_2026-07-31.csv"
METADATA = ROOT / "sources" / "epoch_eci_validation_extension_fit_2026-07-31.json"
OFFICIAL_REPOSITORY = "https://github.com/epoch-research/eci-public.git"
OFFICIAL_COMMIT = "542567e72a415b72624e5bbd12603cfd3f485179"
EXPECTED_MODELS = 212
EXPECTED_K3_ECI = 155.61602792683064


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def checkout(requested: Path | None, work: Path) -> Path:
    if requested is None:
        target = work / "eci-public"
        git("clone", "--quiet", OFFICIAL_REPOSITORY, str(target))
        git("checkout", "--quiet", OFFICIAL_COMMIT, cwd=target)
    else:
        target = requested.resolve()
    if git("rev-parse", "HEAD", cwd=target) != OFFICIAL_COMMIT:
        raise ValueError("eci-public checkout is not at the pinned commit")
    if git("status", "--porcelain", cwd=target):
        raise ValueError("eci-public checkout has uncommitted changes")
    return target


def metadata_document(code: dict[str, str]) -> dict[str, Any]:
    import pandas as pd

    scores = pd.read_csv(OUTPUT)
    if len(scores) != EXPECTED_MODELS or scores["Model"].nunique() != EXPECTED_MODELS:
        raise ValueError("July 22 fitted score inventory is incomplete")
    if scores["Model"].duplicated().any():
        raise ValueError("July 22 fitted scores are not unique by model")
    k3 = scores.loc[scores["Model"] == "Kimi K3", "eci"]
    if len(k3) != 1 or abs(float(k3.iloc[0]) - EXPECTED_K3_ECI) > 1e-10:
        raise ValueError("July 22 Kimi K3 ECI differs from the pinned official fit")
    return {
        "generated_on": "2026-07-31",
        "method": "official ECI fit with bootstrap_samples=0; central estimate only",
        "role": "score-vintage validation extension; excluded from live weights and historical model selection",
        "official_repository": OFFICIAL_REPOSITORY,
        "official_commit": OFFICIAL_COMMIT,
        "official_code_sha256": code,
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256(OUTPUT),
        "models": len(scores),
        "k3_eci": float(k3.iloc[0]),
        "k3_classification": {
            "score_vintage_holdout": True,
            "project_prospective": False,
        },
    }


def verify_existing() -> dict[str, Any]:
    collector.verify_existing()
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if metadata["official_commit"] != OFFICIAL_COMMIT:
        raise ValueError("July 22 fit metadata has the wrong official commit")
    if metadata["source_sha256"] != sha256(SOURCE):
        raise ValueError("July 22 fit metadata has a stale source hash")
    if metadata["output_sha256"] != sha256(OUTPUT):
        raise ValueError("July 22 fitted scores differ from metadata")
    observed = metadata_document(metadata["official_code_sha256"])
    if observed != metadata:
        raise ValueError("July 22 fit metadata is stale")
    return {
        "mode": "verify-existing",
        "models": metadata["models"],
        "k3_eci": metadata["k3_eci"],
        "official_commit": OFFICIAL_COMMIT,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eci-public-dir", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.verify_existing:
        print(json.dumps(verify_existing(), indent=2, sort_keys=True))
        return

    collector.verify_existing()
    import pandas as pd

    with tempfile.TemporaryDirectory(prefix="eci-validation-extension-") as temporary:
        official = checkout(args.eci_public_dir, Path(temporary))
        sys.path.insert(0, str(official / "src"))
        fitting = importlib.import_module("eci.fitting")
        frame = pd.read_csv(SOURCE)
        frame["Model"] = frame["model"]
        scores, _, _ = fitting.fit_eci_model(frame, bootstrap_samples=0)
        model_metadata = (
            frame.sort_values("date")
            .drop_duplicates("model_id", keep="last")
            .set_index("model_id")
        )
        for field in ("date", "model_version", "source"):
            scores[field] = scores["model_id"].map(model_metadata[field])
        scores = scores.sort_values(["eci", "Model"], ascending=[False, True])
        scores.to_csv(OUTPUT, index=False, lineterminator="\n")
        code = {
            str(path.relative_to(official)): sha256(path)
            for path in (official / "src/eci/dataloader.py", official / "src/eci/fitting.py")
        }
        metadata = metadata_document(code)
        METADATA.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"mode": "fit", "models": len(scores), "k3_eci": metadata["k3_eci"]}, indent=2))


if __name__ == "__main__":
    main()
