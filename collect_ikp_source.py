#!/usr/bin/env python3
"""Pin the IKP parameter-probe dataset and its independent replication.

The upstream and replication repositories evolve quickly.  This collector uses
immutable Git commit URLs and checks the exact expected SHA-256 of every file.
Without ``--refresh`` it only verifies the already-frozen local snapshots, so
the normal forecast pipeline never depends on live network state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "sources"
METADATA = SOURCE_DIR / "ikp_source_metadata_2026-07-18.json"

UPSTREAM_REPOSITORY = "https://github.com/19PINE-AI/ikp"
UPSTREAM_COMMIT = "e5c4231985048bb2db5dc2611b6eb659b891791d"
REPLICATION_REPOSITORY = "https://github.com/BenSturgeon/ikp-replication"
REPLICATION_COMMIT = "c44e4dc82132e268dc9a2c86350863f59282fddb"


@dataclass(frozen=True)
class SourceFile:
    label: str
    repository: str
    commit: str
    source_path: str
    output_name: str
    sha256: str

    @property
    def url(self) -> str:
        owner_repo = self.repository.removeprefix("https://github.com/")
        return (
            f"https://raw.githubusercontent.com/{owner_repo}/"
            f"{self.commit}/{self.source_path}"
        )

    @property
    def output(self) -> Path:
        return SOURCE_DIR / self.output_name


FILES = (
    SourceFile(
        "upstream_calibration_contract",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "website/public/data/calibration.json",
        "ikp_upstream_calibration_2026-07-18.json",
        "87745c4cb8d704edccd7be98b3605591dcc12e9305c7fb96bd40da1c899b37c8",
    ),
    SourceFile(
        "upstream_evaluation_summary",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "data/results/evaluation_summary.json",
        "ikp_upstream_evaluation_summary_2026-07-18.json",
        "ae420ee241dd401142b115c2068484b98e4a7355f06ef9c3ce7aef8cec9a31da",
    ),
    SourceFile(
        "upstream_model_config",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "configs/all_models.json",
        "ikp_upstream_models_2026-07-18.json",
        "73f998fbbd6c2fd3e1fe3b68b6e30a33c5c785d70313b81d473d25bff3653799",
    ),
    SourceFile(
        "upstream_method_note",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "IKP_V2.md",
        "ikp_upstream_method_2026-07-18.md",
        "0a007d46747db4e101f23d77c6260bd056c382bc2a1c611e75ec7ca634b24364",
    ),
    SourceFile(
        "upstream_sensitivity",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "website/public/data/sensitivity.json",
        "ikp_upstream_sensitivity_2026-07-18.json",
        "8a190f8f117c58203d9704d2698f3238c458c8cb0dd627a881cf6fe0d57e06c6",
    ),
    SourceFile(
        "upstream_v2_validation",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "data/results/ikp_v2_validation.json",
        "ikp_upstream_v2_validation_2026-07-18.json",
        "cfa8d60048634765dd002305df1ef202201e0fe96c50dde8710e4a1f80bccb43",
    ),
    SourceFile(
        "upstream_densing_analysis_panel",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "data/densing_analysis_data.csv",
        "ikp_upstream_densing_analysis_2026-07-18.csv",
        "2f51325f47f4f51708666ff3912efa691fe0b14f5d3bf4a5739d87c3239521a3",
    ),
    SourceFile(
        "upstream_benchmark_comparison_script",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "scripts/16_benchmark_comparison.py",
        "ikp_upstream_benchmark_comparison_script_2026-07-18.py",
        "ef30ab480f7946f852941af8f6d708ca9806c6963b3e2abf383b8f5d27582386",
    ),
    SourceFile(
        "upstream_benchmark_scores",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "data/benchmarks/benchmark_scores.csv",
        "ikp_upstream_benchmark_scores_2026-07-18.csv",
        "163d8a72fd7cc0fa0dc387f1590c10a4703407ae8198c67d165629e4fd6a2b29",
    ),
    SourceFile(
        "upstream_benchmark_joined_panel",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "data/benchmarks/joined_per_model.csv",
        "ikp_upstream_benchmark_joined_2026-07-18.csv",
        "707273dacdbb76e43825bc491016f93798159b50ba510dc87eea722bd863cb5b",
    ),
    SourceFile(
        "upstream_benchmark_regression_summary",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "data/benchmarks/regression_summary.csv",
        "ikp_upstream_benchmark_regression_summary_2026-07-18.csv",
        "c8c615e3dfb23f9b62b10245884564403bcd418a4466cbbcb30897bd7a25fe04",
    ),
    SourceFile(
        "upstream_benchmark_time_coefficients",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "data/benchmarks/time_coefficients.csv",
        "ikp_upstream_benchmark_time_coefficients_2026-07-18.csv",
        "4cf721af4def0b6d123d7ecde1c776310c492eb0581846acead9628c4d390be2",
    ),
    SourceFile(
        "upstream_benchmark_narrative_summary",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "data/benchmarks/SUMMARY.md",
        "ikp_upstream_benchmark_summary_2026-07-18.md",
        "79bc705a437e12d8474ac33377068c2ec9600091a19de59e5e7775f27acc6b7b",
    ),
    SourceFile(
        "upstream_benchmark_raw_anthropic",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "data/benchmarks/raw_anthropic.md",
        "ikp_upstream_benchmark_raw_anthropic_2026-07-18.md",
        "182594023dcaba8cb59f77160cbb6787b90d3097e47b558e93c605934b3fa367",
    ),
    SourceFile(
        "upstream_benchmark_raw_deepseek_qwen_kimi_glm",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "data/benchmarks/raw_deepseek_qwen_kimi_glm.md",
        "ikp_upstream_benchmark_raw_deepseek_qwen_kimi_glm_2026-07-18.md",
        "fdd8fa7ee60d5f7ba581a0c920c92c42fb5e4decfaf93ce7271d5551adb878a6",
    ),
    SourceFile(
        "upstream_benchmark_raw_google_meta",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "data/benchmarks/raw_google_meta.md",
        "ikp_upstream_benchmark_raw_google_meta_2026-07-18.md",
        "fe2359ba6d141610db2dd179c1c85b954b4aa7a75b0fe58456700f247d6d39cf",
    ),
    SourceFile(
        "upstream_benchmark_raw_openai",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "data/benchmarks/raw_openai.md",
        "ikp_upstream_benchmark_raw_openai_2026-07-18.md",
        "65761bb7e5c0b0e58b7bdc82ce347b25c7f776c8e8e61a44ff6ac78b60799cab",
    ),
    SourceFile(
        "upstream_benchmark_raw_others",
        UPSTREAM_REPOSITORY,
        UPSTREAM_COMMIT,
        "data/benchmarks/raw_others.md",
        "ikp_upstream_benchmark_raw_others_2026-07-18.md",
        "cd26b9f931cd1eb375e6d1e961900d935fe92173c756d2df2b1d6680c28010a8",
    ),
    SourceFile(
        "replication_readme",
        REPLICATION_REPOSITORY,
        REPLICATION_COMMIT,
        "README.md",
        "ikp_replication_readme_2026-07-18.md",
        "db9417c3b169578544a16998d37be2fa8ddac455ae6d9d98e604817729a21d2b",
    ),
    SourceFile(
        "replication_calibration_errors",
        REPLICATION_REPOSITORY,
        REPLICATION_COMMIT,
        "runs/calibration_error_per_model.csv",
        "ikp_replication_calibration_errors_2026-07-18.csv",
        "1b88c2d9f218dffb06ad4f03091eb43d64aeaa1035b16a56afcd8e85fa6b1501",
    ),
    SourceFile(
        "replication_wikidata_dominance",
        REPLICATION_REPOSITORY,
        REPLICATION_COMMIT,
        "data/wikidata_dominance_check.json",
        "ikp_replication_wikidata_dominance_2026-07-18.json",
        "13273f555e7e3dc6f4505c6f0acfbb3a12996dfe10e8faa56c86b86b5f43ba12",
    ),
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect(source: SourceFile, refresh: bool) -> dict[str, object]:
    if refresh or not source.output.exists():
        request = urllib.request.Request(
            source.url,
            headers={"User-Agent": "frontier-parameter-model/ikp-audit"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        actual = digest(data)
        if actual != source.sha256:
            raise ValueError(
                f"Pinned IKP source changed for {source.label}: "
                f"expected {source.sha256}, got {actual}"
            )
        source.output.write_bytes(data)
    data = source.output.read_bytes()
    actual = digest(data)
    if actual != source.sha256:
        raise ValueError(
            f"Local IKP snapshot mismatch for {source.label}: "
            f"expected {source.sha256}, got {actual}"
        )
    return {
        "label": source.label,
        "repository": source.repository,
        "commit": source.commit,
        "source_path": source.source_path,
        "raw_url": source.url,
        "local_path": str(source.output.relative_to(ROOT)),
        "sha256": actual,
        "bytes": len(data),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch the immutable raw GitHub objects instead of only checking local copies.",
    )
    args = parser.parse_args()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    records = [collect(source, args.refresh) for source in FILES]
    metadata = {
        "collected_on": "2026-07-18",
        "collection_policy": "immutable Git commit URLs plus exact expected SHA-256",
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "role": "primary benchmark data and current v2 methodology",
        },
        "replication": {
            "repository": REPLICATION_REPOSITORY,
            "commit": REPLICATION_COMMIT,
            "role": "independent methodological and data-quality audit",
        },
        "files": records,
        "scope_note": (
            "The snapshots preserve the published aggregate model measurements and "
            "validation contracts, not all raw per-probe API responses. Claims about "
            "raw-response reproduction therefore remain upstream validation claims."
        ),
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metadata": str(METADATA), "files": len(records)}, indent=2))


if __name__ == "__main__":
    main()
