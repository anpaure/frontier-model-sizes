#!/usr/bin/env python3
"""Vendor primary evidence for AA parameter-label availability.

Network refresh is explicit.  Normal pipeline runs consume only the generated,
hash-pinned ledger and evidence files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from aa_parameter_label_availability import load_parameter_label_availability


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "sources/aa_parameter_label_availability_evidence_2026-07-31"
LEDGER = ROOT / "sources/aa_parameter_label_availability_2026-07-31.json"


SPECS: tuple[dict[str, Any], ...] = (
    {
        "record_id": "aa-label-longcat-flash-lite-2026-01-29",
        "name": "LongCat Flash Lite",
        "aliases": ["LongCat-Flash-Lite"],
        "slug": "longcat-flash-lite",
        "release": "2026-01-28",
        "parameters_b": 68.5,
        "repo": "meituan-longcat/LongCat-Flash-Lite",
        "label_date": "2026-01-29",
        "label_basis": "first_party_technical_report",
        "label_url": "https://arxiv.org/abs/2601.21204",
        "weights_date": "2026-01-30",
        "weights_timestamp": "2026-01-30T07:20:40.000Z",
        "weights_commit": "ef1bd778242f0abcf6d8536bace504c658c37c99",
        "repo_created": "2026-01-27T08:03:21.000Z",
        "safetensors_total": 69_073_335_552,
        "label_match_tokens": ["68.5B", "parameter"],
    },
    {
        "record_id": "aa-label-minimax-m2-7-2026-04-09",
        "name": "MiniMax-M2.7",
        "aliases": ["MiniMax M2.7"],
        "slug": "minimax-m2-7",
        "release": "2026-03-18",
        "parameters_b": 230.0,
        "accepted_parameters_b": [229.0, 230.0],
        "repo": "MiniMaxAI/MiniMax-M2.7",
        "label_date": "2026-04-09",
        "label_basis": "official_hugging_face_config_and_weights",
        "label_url": "https://huggingface.co/MiniMaxAI/MiniMax-M2.7",
        "launch_url": "https://www.minimax.io/news/minimax-m27-en",
        "weights_date": "2026-04-09",
        "weights_timestamp": "2026-04-09T11:34:39.000Z",
        "weights_commit": "a0ca8c65a93219ea256d47fd0851c49d13fca5fc",
        "model_card_commit": "1eece56934dda5c770bdb7fec67021222b705629",
        "repo_created": "2026-04-09T03:37:12.000Z",
        "safetensors_total": 228_703_644_928,
        "label_match_tokens": [],
    },
    {
        "record_id": "aa-label-mimo-v2-5-2026-04-22",
        "name": "MiMo-V2.5",
        "aliases": ["MiMo V2.5"],
        "slug": "mimo-v2-5-0424",
        "release": "2026-04-22",
        "parameters_b": 310.0,
        "repo": "XiaomiMiMo/MiMo-V2.5",
        "label_date": "2026-04-22",
        "label_basis": "first_party_launch_model_card",
        "label_url": "https://mimo.xiaomi.com/mimo-v2-5/",
        "weights_date": "2026-04-27",
        "weights_timestamp": "2026-04-27T15:10:09.000Z",
        "weights_commit": "c8dea8ddc91add71ed7624bef73391aaf671c419",
        "config_commit": "4f78a4d18f532779ecc6cf3d33ce398a20f78328",
        "model_card_commit": "4f78a4d18f532779ecc6cf3d33ce398a20f78328",
        "repo_created": "2026-04-27T13:37:38.000Z",
        "safetensors_total": 310_775_040_000,
        "label_match_tokens": ["April 22nd, 2026", "310B-parameter", "15B active"],
    },
    {
        "record_id": "aa-label-minimax-m3-2026-06-12",
        "name": "MiniMax-M3",
        "aliases": ["MiniMax M3"],
        "slug": "minimax-m3",
        "release": "2026-06-01",
        "parameters_b": 428.0,
        "repo": "MiniMaxAI/MiniMax-M3",
        "label_date": "2026-06-12",
        "label_basis": "official_hugging_face_config_and_weights",
        "label_url": "https://huggingface.co/MiniMaxAI/MiniMax-M3",
        "launch_url": "https://www.minimax.io/blog/minimax-m3",
        "weights_date": "2026-06-12",
        "weights_timestamp": "2026-06-12T12:52:11.000Z",
        "weights_commit": "3a41b311ffa5719cef48fed3974ccf2cc03733ea",
        "repo_created": "2026-06-02T07:49:31.000Z",
        "safetensors_total": 427_040_140_160,
        "label_match_tokens": ["428B parameters", "23B activated"],
    },
    {
        "record_id": "aa-label-nex-n2-pro-2026-06-03",
        "name": "Nex-N2-Pro",
        "aliases": ["Nex N2 Pro"],
        "slug": "nex-n2-pro",
        "release": "2026-06-02",
        "parameters_b": 397.0,
        "repo": "nex-agi/Nex-N2-Pro",
        "label_date": "2026-06-03",
        "label_basis": "official_hugging_face_config_and_weights",
        "label_url": "https://huggingface.co/nex-agi/Nex-N2-Pro",
        "weights_date": "2026-06-03",
        "weights_timestamp": "2026-06-03T17:24:18.000Z",
        "weights_commit": "a85b319a367e85c544a79e882638964be8238941",
        "repo_created": "2026-06-03T03:15:13.000Z",
        "safetensors_total": 396_802_360_816,
        "label_match_tokens": ["397B", "17B"],
    },
    {
        "record_id": "aa-label-longcat-2-0-2026-06-30",
        "name": "LongCat 2.0",
        "aliases": ["LongCat-2.0"],
        "slug": "longcat-2-0",
        "release": "2026-06-29",
        "parameters_b": 1600.0,
        "repo": "meituan-longcat/LongCat-2.0",
        "label_date": "2026-06-30",
        "label_basis": "first_party_release_article",
        "label_url": "https://www.meituan.com/news/NN260630164005904",
        "weights_date": "2026-07-05",
        "weights_timestamp": "2026-07-05T01:48:21.000Z",
        "weights_commit": "373726b7bb09d29076e85c87384ba718ba153d07",
        "repo_created": "2026-07-05T00:07:44.000Z",
        "safetensors_total": 1_775_560_491_136,
        "label_match_tokens": ["2026-06-30", "1.6 T", "48 B"],
    },
)


def fetch(url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(8):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "frontier-parameter-label-audit/1.0"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            time.sleep(min(15.0, 1.5 * (attempt + 1)))
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_evidence(record_id: str, kind: str, extension: str, url: str) -> dict[str, str]:
    data = fetch(url)
    relative = Path("sources/aa_parameter_label_availability_evidence_2026-07-31") / (
        f"{record_id}__{kind}.{extension}"
    )
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"kind": kind, "path": str(relative), "sha256": sha256(data), "url": url}


def refresh() -> None:
    records = []
    for spec in SPECS:
        repo = spec["repo"]
        commit = spec["weights_commit"]
        config_commit = spec.get("config_commit", commit)
        model_card_commit = spec.get("model_card_commit", commit)
        evidence = [
            write_evidence(
                spec["record_id"],
                "hugging_face_model_api",
                "json",
                f"https://huggingface.co/api/models/{repo}",
            ),
            write_evidence(
                spec["record_id"],
                "hugging_face_commit_history",
                "json",
                f"https://huggingface.co/api/models/{repo}/commits/main?limit=1000",
            ),
            write_evidence(
                spec["record_id"],
                "commit_pinned_config",
                "json",
                f"https://huggingface.co/{repo}/raw/{config_commit}/config.json",
            ),
            write_evidence(
                spec["record_id"],
                "commit_pinned_model_card",
                "md",
                f"https://huggingface.co/{repo}/raw/{model_card_commit}/README.md",
            ),
        ]
        if spec.get("launch_url"):
            evidence.append(
                write_evidence(
                    spec["record_id"],
                    "first_party_launch_page",
                    "html",
                    spec["launch_url"],
                )
            )
        elif not spec["label_url"].startswith("https://huggingface.co/"):
            evidence.append(
                write_evidence(
                    spec["record_id"],
                    "first_party_label_source",
                    "html",
                    spec["label_url"],
                )
            )
        records.append(
            {
                "record_id": spec["record_id"],
                "identity": {
                    "canonical_model_name": spec["name"],
                    "aliases": spec["aliases"],
                    "aa_slug": spec["slug"],
                    "aa_release_date": spec["release"],
                    "total_parameters_b": spec["parameters_b"],
                    "accepted_total_parameters_b": spec.get(
                        "accepted_parameters_b", [spec["parameters_b"]]
                    ),
                    "model_weights_source_url": f"https://huggingface.co/{repo}",
                },
                "timing": {
                    "parameter_label_available_date": spec["label_date"],
                    "parameter_label_basis": spec["label_basis"],
                    "parameter_label_source_url": spec["label_url"],
                    "repository_created_at_utc": spec["repo_created"],
                    "weights_available_date": spec["weights_date"],
                    "weights_available_at_utc": spec["weights_timestamp"],
                    "weights_commit": commit,
                    "config_commit": config_commit,
                    "model_card_commit": model_card_commit,
                    "hugging_face_safetensors_total_parameters": spec[
                        "safetensors_total"
                    ],
                    "parameter_label_match_tokens": spec["label_match_tokens"],
                    "parameter_value_basis": (
                        "official_safetensors_total_rounded_to_aa_value"
                        if spec["label_basis"]
                        == "official_hugging_face_config_and_weights"
                        else "publisher_disclosed_rounded_total"
                    ),
                },
                "local_evidence": evidence,
            }
        )
    payload = {
        "schema_version": "1.0",
        "snapshot_date": "2026-07-31",
        "purpose": (
            "Separate checkpoint release from public parameter-label and weight "
            "availability in chronological parameter regressions."
        ),
        "policy": (
            "Training eligibility is max(checkpoint release date, public parameter-label "
            "date). Weight-release dates are preserved separately and are not substituted "
            "when an earlier first-party model card or technical report states the count."
        ),
        "records": records,
    }
    LEDGER.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.refresh:
        refresh()
        # Revalidate the newly written ledger in a fresh cache slot.
        load_parameter_label_availability.cache_clear()
    payload = load_parameter_label_availability()
    print(
        json.dumps(
            {
                "ledger": str(LEDGER),
                "records": len(payload["records"]),
                "mode": "network_refresh" if args.refresh else "offline_verify",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
