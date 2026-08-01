#!/usr/bin/env python3
"""Freeze and normalize first-party evidence for the frontier targets.

The default path is deliberately offline. It reconstructs the evidence ledger
from a frozen OpenAI system-card HTML response and a compact Anthropic claim
manifest. ``--refresh`` downloads both first-party sources, verifies the exact
claims against the source text, and rewrites the frozen artifacts.

The 27 MB Anthropic PDF is not committed. Its byte hash, page count, page-text
hashes, and verified claim locations are preserved in the compact claim
manifest so a refresh is auditable without putting a large binary in Git.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
DATE = "2026-07-18"
OPENAI_URL = "https://deploymentsafety.openai.com/gpt-5-6"
ANTHROPIC_URL = (
    "https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/"
    "Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf"
)
OPENAI_RAW = ROOT / f"sources/openai_gpt_5_6_system_card_{DATE}.html.gz"
ANTHROPIC_CLAIMS = ROOT / f"sources/anthropic_fable_mythos_primary_claims_{DATE}.json"
LEDGER = ROOT / f"sources/frontier_primary_evidence_{DATE}.csv"
METADATA = ROOT / f"sources/frontier_primary_evidence_collection_metadata_{DATE}.json"


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalize_text(text: str) -> str:
    return " ".join(text.replace("ﬁ", "fi").replace("ﬂ", "fl").split())


def html_text(payload: bytes) -> str:
    parser = TextExtractor()
    parser.feed(payload.decode("utf-8", errors="replace"))
    return normalize_text(" ".join(parser.parts))


def deterministic_gzip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        handle.write(payload)
    return buffer.getvalue()


def fetch(url: str, accept: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": "frontier-parameter-model primary-evidence audit",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def load_openai() -> bytes:
    if not OPENAI_RAW.exists():
        raise FileNotFoundError(f"Missing frozen source {OPENAI_RAW}; run with --refresh once")
    with gzip.open(OPENAI_RAW, "rb") as handle:
        return handle.read()


def validate_openai(payload: bytes) -> dict[str, Any]:
    text = html_text(payload)
    pattern = re.compile(
        r"GPT-5\.6 Sol performed at a (?P<sol>[0-9.]+) minute time horizon for "
        r"[\u201c\"]no-CoT[\u201d\"] math reasoning tasks.*?higher than GPT[ -]5\.5 at "
        r"(?P<gpt55>[0-9.]+) minutes",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        raise ValueError("Pinned GPT-5.6 / GPT-5.5 no-CoT statement was not found")
    sol = float(match.group("sol"))
    gpt55 = float(match.group("gpt55"))
    if (sol, gpt55) != (3.6, 2.3):
        raise ValueError(f"Official no-CoT values changed: {(sol, gpt55)}")
    return {
        "sol_minutes": sol,
        "gpt_5_5_minutes": gpt55,
        "normalized_text_sha256": sha256_bytes(text.encode("utf-8")),
    }


def page_text(reader: PdfReader, page_number: int) -> str:
    return normalize_text(reader.pages[page_number - 1].extract_text() or "")


def validate_anthropic_pdf(payload: bytes) -> dict[str, Any]:
    reader = PdfReader(io.BytesIO(payload))
    if len(reader.pages) != 317:
        raise ValueError(f"Anthropic system-card page count changed: {len(reader.pages)}")
    pages = {number: page_text(reader, number) for number in (12, 14, 131, 258)}
    checks = {
        "shared_underlying_weights": (
            12,
            "same underlying model weights as Mythos 5",
        ),
        "client_fallback_to_opus_4_8": (
            14,
            "automatically falls back to the most recent Claude Opus model (at the time of release, Claude Opus 4.8)",
        ),
        "messages_api_no_default_fallback": (
            14,
            "In the Messages API, there is no automatic fallback by default",
        ),
        "no_fallback_means_underlying_mythos": (
            131,
            "these evaluations entirely test the underlying Mythos 5 model",
        ),
        "gpqa_saturated": (
            258,
            "We consider GPQA Diamond to be a saturated evaluation",
        ),
    }
    for claim, (page, phrase) in checks.items():
        if phrase not in pages[page]:
            raise ValueError(f"Anthropic claim {claim!r} was not found on page {page}")
    return {
        "source_url": ANTHROPIC_URL,
        "pdf_sha256": sha256_bytes(payload),
        "pdf_bytes": len(payload),
        "page_count": len(reader.pages),
        "page_text_sha256": {
            str(page): sha256_bytes(text.encode("utf-8")) for page, text in pages.items()
        },
        "claims": {
            claim: {"verified": True, "page": page} for claim, (page, _) in checks.items()
        },
    }


def load_anthropic_claims() -> dict[str, Any]:
    if not ANTHROPIC_CLAIMS.exists():
        raise FileNotFoundError(
            f"Missing frozen source {ANTHROPIC_CLAIMS}; run with --refresh once"
        )
    claims = json.loads(ANTHROPIC_CLAIMS.read_text(encoding="utf-8"))
    required = {
        "shared_underlying_weights",
        "client_fallback_to_opus_4_8",
        "messages_api_no_default_fallback",
        "no_fallback_means_underlying_mythos",
        "gpqa_saturated",
    }
    if set(claims.get("claims", {})) != required:
        raise ValueError("Frozen Anthropic claim inventory changed")
    if not all(record.get("verified") for record in claims["claims"].values()):
        raise ValueError("Frozen Anthropic manifest contains an unverified claim")
    return claims


def build_rows(openai: dict[str, Any], anthropic: dict[str, Any]) -> list[dict[str, Any]]:
    def row(**values: Any) -> dict[str, Any]:
        return {
            "evidence_id": values["evidence_id"],
            "developer": values["developer"],
            "model": values["model"],
            "comparator_model": values.get("comparator_model", ""),
            "evidence_type": values["evidence_type"],
            "metric_name": values.get("metric_name", ""),
            "value": values.get("value", ""),
            "unit": values.get("unit", ""),
            "source_url": values["source_url"],
            "source_date": values["source_date"],
            "page_or_section": values["page_or_section"],
            "claim_summary": values["claim_summary"],
            "parameter_identity_policy": values["parameter_identity_policy"],
            "live_weight_policy": values["live_weight_policy"],
        }

    return [
        row(
            evidence_id="openai_gpt_5_6_sol_nocot_horizon",
            developer="OpenAI",
            model="GPT-5.6 Sol",
            evidence_type="direct_capability_measurement",
            metric_name="nocot_time_horizon_minutes",
            value=openai["sol_minutes"],
            unit="minutes",
            source_url=OPENAI_URL,
            source_date="2026-07-09",
            page_or_section="9.2.3 external evaluations / UK AISI",
            claim_summary="UK AISI reports a 3.6-minute no-CoT math time horizon for GPT-5.6 Sol.",
            parameter_identity_policy="target_measurement_only_no_parameter_disclosure",
            live_weight_policy="diagnostic_until_heldout_mapping_gate_passes",
        ),
        row(
            evidence_id="openai_gpt_5_5_nocot_comparator",
            developer="OpenAI",
            model="GPT-5.5",
            comparator_model="GPT-5.6 Sol",
            evidence_type="direct_capability_comparator",
            metric_name="nocot_time_horizon_minutes",
            value=openai["gpt_5_5_minutes"],
            unit="minutes",
            source_url=OPENAI_URL,
            source_date="2026-07-09",
            page_or_section="9.2.3 external evaluations / UK AISI",
            claim_summary="The same UK AISI statement reports 2.3 minutes for GPT-5.5.",
            parameter_identity_policy="comparator_measurement_only_no_parameter_disclosure",
            live_weight_policy="diagnostic_until_cross_suite_reconciliation_passes",
        ),
        row(
            evidence_id="anthropic_fable_mythos_shared_weights",
            developer="Anthropic",
            model="Claude Fable 5",
            comparator_model="Claude Mythos 5",
            evidence_type="base_model_identity",
            source_url=ANTHROPIC_URL,
            source_date="2026-07-14",
            page_or_section=f"PDF p.{anthropic['claims']['shared_underlying_weights']['page']}",
            claim_summary="Anthropic states that Fable 5 and Mythos 5 use the same underlying model weights; Fable adds safeguards.",
            parameter_identity_policy="same_underlying_weights_single_parameter_target",
            live_weight_policy="identity_constraint_not_numeric_likelihood",
        ),
        row(
            evidence_id="anthropic_fable_fallback_scope",
            developer="Anthropic",
            model="Claude Fable 5",
            comparator_model="Claude Opus 4.8",
            evidence_type="serving_system_caveat",
            source_url=ANTHROPIC_URL,
            source_date="2026-07-14",
            page_or_section=f"PDF p.{anthropic['claims']['client_fallback_to_opus_4_8']['page']}",
            claim_summary="Client apps can route classifier-triggering Fable requests to Opus 4.8; the Messages API has no automatic fallback by default.",
            parameter_identity_policy="fallback_is_serving_behavior_not_shared_base",
            live_weight_policy="flag_potential_benchmark_contamination_only",
        ),
        row(
            evidence_id="anthropic_gpqa_saturation_caveat",
            developer="Anthropic",
            model="Claude Mythos 5",
            evidence_type="benchmark_caveat",
            metric_name="gpqa_diamond",
            source_url=ANTHROPIC_URL,
            source_date="2026-07-14",
            page_or_section=f"PDF p.{anthropic['claims']['gpqa_saturated']['page']}",
            claim_summary="Anthropic classifies GPQA Diamond as saturated and plans to stop reporting it for future models.",
            parameter_identity_policy="no_parameter_identity_effect",
            live_weight_policy="exclude_from_new_incremental_size_signal",
        ),
    ]


def write_csv(rows: list[dict[str, Any]]) -> None:
    with LEDGER.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh and verify both official sources; default is a frozen offline rebuild.",
    )
    args = parser.parse_args()

    if args.refresh:
        openai_raw = fetch(OPENAI_URL, "text/html")
        anthropic_raw = fetch(ANTHROPIC_URL, "application/pdf")
        openai = validate_openai(openai_raw)
        anthropic = validate_anthropic_pdf(anthropic_raw)
        OPENAI_RAW.parent.mkdir(parents=True, exist_ok=True)
        OPENAI_RAW.write_bytes(deterministic_gzip(openai_raw))
        ANTHROPIC_CLAIMS.write_text(
            json.dumps(anthropic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        mode = "first-party refresh"
    else:
        openai_raw = load_openai()
        openai = validate_openai(openai_raw)
        anthropic = load_anthropic_claims()
        mode = "frozen offline rebuild"

    rows = build_rows(openai, anthropic)
    write_csv(rows)
    metadata = {
        "generated_on": DATE,
        "collection_mode": mode,
        "inventory": {
            "normalized_evidence_rows": len(rows),
            "numeric_measurements": sum(bool(row["value"] != "") for row in rows),
            "identity_constraints": sum(row["evidence_type"] == "base_model_identity" for row in rows),
            "serving_system_caveats": sum(row["evidence_type"] == "serving_system_caveat" for row in rows),
            "benchmark_caveats": sum(row["evidence_type"] == "benchmark_caveat" for row in rows),
        },
        "openai_source": {
            "url": OPENAI_URL,
            "published_date": "2026-07-09",
            "raw_uncompressed_sha256": sha256_bytes(openai_raw),
            **openai,
        },
        "anthropic_source": {
            "url": ANTHROPIC_URL,
            "published_date": "2026-07-14",
            "pdf_sha256": anthropic["pdf_sha256"],
            "pdf_bytes": anthropic["pdf_bytes"],
            "page_count": anthropic["page_count"],
            "claims": anthropic["claims"],
        },
        "integrity_policy": {
            "network_in_frozen_pipeline": False,
            "official_sources_only": True,
            "fable_mythos_identity_is_not_a_numeric_size_observation": True,
            "fable_opus_fallback_is_not_a_parameter_identity": True,
            "direct_horizon_is_not_mapped_to_size_without_backtest": True,
            "large_anthropic_pdf_reproducible_by_url_and_hash": True,
        },
        "files": {
            str(OPENAI_RAW.relative_to(ROOT)): {"sha256": sha256(OPENAI_RAW)},
            str(ANTHROPIC_CLAIMS.relative_to(ROOT)): {"sha256": sha256(ANTHROPIC_CLAIMS)},
            str(LEDGER.relative_to(ROOT)): {"sha256": sha256(LEDGER)},
        },
    }
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ledger": str(LEDGER), **metadata["inventory"]}, indent=2))


if __name__ == "__main__":
    main()
