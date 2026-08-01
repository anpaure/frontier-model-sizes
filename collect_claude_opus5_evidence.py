#!/usr/bin/env python3
"""Freeze and normalize primary evidence for Claude Opus 5.

Network access is opt-in.  ``--refresh`` replaces the dated raw snapshots only
after all downloads pass structural validation.  An ordinary invocation is
offline: it rebuilds the normalized record from the frozen bytes and requires
byte-for-byte agreement with the committed JSON summary.

The collector deliberately distinguishes three things that are easy to blur:

* a disclosed fact (release date, API identifier, price, benchmark score),
* an explicit absence in a frozen dataset (METR, Epoch parameter/compute), and
* a modeling policy.  Anthropic has not disclosed an Opus 5 parameter count or
  an exact same-weight relationship to an earlier checkpoint.  Consequently,
  the normalized record marks the count unknown and assigns a unique base ID;
  the latter is a conservative join policy, not an architecture disclosure.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import io
import json
import os
import re
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from collect_aa_detailed_signals import extract_models


ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"
DATE = "2026-07-31"
SUMMARY = SOURCES / f"claude_opus_5_evidence_{DATE}.json"
NO_COT_PANEL = (
    ROOT
    / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
    / "unified_model_measurements_long_compute_enriched_2026-07-17.csv"
)
IKP_MODEL_PANEL = SOURCES / "ikp_upstream_models_2026-07-18.json"

ECI_EXACT = 159.3778667882398
ECI_CI_LOW = 157.24933114170264
ECI_CI_HIGH = 162.20640578425878


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    path: Path
    wire_format: str


SOURCE_SPECS = (
    Source(
        "anthropic_news",
        "https://www.anthropic.com/news/claude-opus-5",
        SOURCES / f"claude_opus_5_anthropic_news_{DATE}.html.gz",
        "html_gzip",
    ),
    Source(
        "anthropic_model_overview",
        "https://platform.claude.com/docs/en/about-claude/models/overview",
        SOURCES / f"claude_opus_5_anthropic_model_overview_{DATE}.html.gz",
        "html_gzip",
    ),
    Source(
        "anthropic_whats_new",
        "https://platform.claude.com/docs/en/about-claude/models/whats-new-opus-5",
        SOURCES / f"claude_opus_5_anthropic_whats_new_{DATE}.html.gz",
        "html_gzip",
    ),
    Source(
        "anthropic_system_card",
        "https://www.anthropic.com/claude-opus-5-system-card",
        SOURCES / f"claude_opus_5_system_card_{DATE}.pdf",
        "pdf",
    ),
    Source(
        "artificial_analysis_model_page",
        "https://artificialanalysis.ai/models/claude-opus-5",
        SOURCES / f"claude_opus_5_artificial_analysis_{DATE}.html.gz",
        "html_gzip",
    ),
    Source(
        "artificial_analysis_launch_article",
        "https://artificialanalysis.ai/articles/opus-5",
        SOURCES / f"claude_opus_5_artificial_analysis_article_{DATE}.html.gz",
        "html_gzip",
    ),
    Source(
        "epoch_all_ai_models",
        "https://epoch.ai/data/all_ai_models.csv",
        SOURCES / f"epoch_all_ai_models_{DATE}.csv",
        "csv",
    ),
    Source(
        "epoch_eci_benchmarks",
        "https://epoch.ai/data/eci_benchmarks.csv",
        SOURCES / f"epoch_eci_benchmarks_{DATE}.csv",
        "csv",
    ),
    Source(
        "epoch_benchmark_data",
        "https://epoch.ai/data/benchmark_data.zip",
        SOURCES / f"epoch_benchmark_data_{DATE}.zip",
        "zip",
    ),
    Source(
        "openrouter_catalog",
        "https://openrouter.ai/api/v1/models",
        SOURCES / f"claude_opus_5_openrouter_catalog_{DATE}.json.gz",
        "json_gzip",
    ),
    Source(
        "metr_horizon",
        "https://metr.org/assets/benchmark_results_1_1.yaml",
        SOURCES / f"metr_benchmark_results_1_1_{DATE}.yaml",
        "yaml",
    ),
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "frontier-parameter-model/1.0 Claude-Opus-5 evidence audit",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise ValueError(f"Empty response from {url}")
    return payload


def stored_bytes(spec: Source, payload: bytes) -> bytes:
    if spec.wire_format.endswith("_gzip"):
        return gzip.compress(payload, compresslevel=9, mtime=0)
    return payload


def payload_bytes(spec: Source) -> bytes:
    payload = spec.path.read_bytes()
    if spec.wire_format.endswith("_gzip"):
        return gzip.decompress(payload)
    return payload


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def csv_rows(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))


def normalized_text(payload: bytes) -> str:
    text = html.unescape(payload.decode("utf-8", "replace"))
    text = re.sub(r"<[^>]*>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalized_pdf_text(value: str) -> str:
    # Preserve wording while undoing presentation ligatures emitted by the PDF
    # font.  This makes extraction deterministic across pypdf/Poppler versions.
    for ligature, expanded in (
        ("\ufb00", "ff"),
        ("\ufb01", "fi"),
        ("\ufb02", "fl"),
        ("\ufb03", "ffi"),
        ("\ufb04", "ffl"),
    ):
        value = value.replace(ligature, expanded)
    return re.sub(r"\s+", " ", value).strip()


def validate_download(spec: Source, payload: bytes) -> None:
    if spec.wire_format == "pdf":
        if not payload.startswith(b"%PDF-") or len(payload) < 1_000_000:
            raise ValueError("Anthropic system card is not a substantial PDF")
        return
    if spec.wire_format == "zip":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if "epoch_capabilities_index.csv" not in archive.namelist():
                raise ValueError("Epoch benchmark ZIP lacks the public ECI table")
        return
    if spec.key == "epoch_all_ai_models":
        rows = csv_rows(payload)
        if len(rows) < 3_000 or "Parameters" not in rows[0]:
            raise ValueError("Epoch all-model CSV has an unexpected schema")
        return
    if spec.key == "epoch_eci_benchmarks":
        rows = csv_rows(payload)
        if len(rows) < 2_000 or "benchmark_id" not in rows[0]:
            raise ValueError("Epoch ECI CSV has an unexpected schema")
        return
    if spec.key == "openrouter_catalog":
        records = json.loads(payload)["data"]
        identifiers = {row["id"] for row in records}
        expected = {"anthropic/claude-opus-5", "anthropic/claude-opus-5-fast"}
        if not expected <= identifiers:
            raise ValueError("OpenRouter catalog lacks both Opus 5 tiers")
        return
    if spec.key == "metr_horizon":
        if not payload.startswith(b"benchmark_name: METR-Horizon-v1.1\n"):
            raise ValueError("METR result asset has an unexpected header")
        return
    text = payload.decode("utf-8", "replace")
    if "Claude Opus 5" not in text and "claude-opus-5" not in text:
        raise ValueError(f"{spec.key} does not mention Claude Opus 5")
    if spec.key == "artificial_analysis_model_page" and b"self.__next_f.push" not in payload:
        raise ValueError("AA page lacks its machine-readable React Flight payload")


def refresh_sources() -> None:
    staged: list[tuple[Source, bytes]] = []
    for spec in SOURCE_SPECS:
        payload = fetch(spec.url)
        validate_download(spec, payload)
        staged.append((spec, stored_bytes(spec, payload)))
    for spec, payload in staged:
        atomic_write(spec.path, payload)


def require_sources() -> None:
    missing = [relative(spec.path) for spec in SOURCE_SPECS if not spec.path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Frozen Opus 5 sources are missing; run with --refresh: " + ", ".join(missing)
        )
    for spec in SOURCE_SPECS:
        validate_download(spec, payload_bytes(spec))


def parse_anthropic() -> tuple[dict[str, Any], dict[str, Any]]:
    raw = {spec.key: payload_bytes(spec) for spec in SOURCE_SPECS}
    news = normalized_text(raw["anthropic_news"])
    overview = normalized_text(raw["anthropic_model_overview"])
    whats_new = normalized_text(raw["anthropic_whats_new"])

    required_news = (
        "Jul 24, 2026",
        "$5 per million input tokens and $25 per million output tokens",
        "claude-opus-5",
        "Fast mode",
    )
    for value in required_news:
        if value not in news:
            raise ValueError(f"Anthropic news page lacks exact text: {value!r}")
    required_overview = ("claude-opus-5", "1M tokens", "May 2026")
    for value in required_overview:
        if value not in overview:
            raise ValueError(f"Anthropic model overview lacks exact text: {value!r}")
    required_whats_new = (
        "1M token context window",
        "128k max output tokens",
        "$5 per million input tokens and $25 per million output tokens",
        "$10 per million input tokens and $50 per million output tokens",
    )
    for value in required_whats_new:
        if value not in whats_new:
            raise ValueError(f"Anthropic Opus 5 docs lack exact text: {value!r}")

    reader = PdfReader(io.BytesIO(raw["anthropic_system_card"]))
    # All relevant model-identity and AECI sections are in the first 40 pages.
    page_text = [normalized_pdf_text(page.extract_text() or "") for page in reader.pages[:40]]
    card = " ".join(page_text)
    pretraining = (
        "After the pretraining process, Opus 5 underwent rigorous post-training and "
        "fine-tuning"
    )
    cutoff = "Claude Opus 5’s knowledge cutoff date is May 2026."
    aeci = (
        "Claude Opus 5’s AECI point estimate is 162.1 (95% CI [158.0, 167.3], "
        "n=40)"
    )
    for value in (pretraining, cutoff, aeci):
        if value not in card:
            raise ValueError(f"Anthropic system card lacks exact extracted text: {value!r}")

    training_page = next(index + 1 for index, text in enumerate(page_text) if pretraining in text)
    aeci_page = next(index + 1 for index, text in enumerate(page_text) if aeci in text)
    system_card = {
        "aeci": {
            "ci_high": 167.3,
            "ci_low": 158.0,
            "n_benchmarks": 40,
            "point_estimate": 162.1,
        },
        "aeci_pdf_page": aeci_page,
        "knowledge_cutoff": "2026-05",
        "pretraining_statement": pretraining,
        "training_pdf_page": training_page,
    }
    api = {
        "api_model_id": "claude-opus-5",
        "context_window_tokens": 1_000_000,
        "fast_input_usd_per_mtok": 10,
        "fast_output_usd_per_mtok": 50,
        "input_usd_per_mtok": 5,
        "knowledge_cutoff": "2026-05",
        "max_output_tokens": 128_000,
        "output_usd_per_mtok": 25,
        "training_data_cutoff": "2026-05",
    }
    return system_card, api


def parse_artificial_analysis() -> dict[str, Any]:
    model_page = payload_bytes(
        next(spec for spec in SOURCE_SPECS if spec.key == "artificial_analysis_model_page")
    ).decode("utf-8")
    models = extract_models(model_page)
    effort_slugs = {
        "low": "claude-opus-5-low",
        "medium": "claude-opus-5-medium",
        "high": "claude-opus-5-high",
        "xhigh": "claude-opus-5-xhigh",
        "max": "claude-opus-5",
    }
    rows: list[dict[str, Any]] = []
    for effort, slug in effort_slugs.items():
        record = models[slug]
        if record["releaseDate"] != "2026-07-24":
            raise ValueError(f"AA release date mismatch for {slug}")
        if record.get("parameters") is not None:
            raise ValueError(f"AA unexpectedly contains an Opus 5 parameter count: {slug}")
        token_count = record["canonicalIntelligenceIndexTokenCount"]
        per_task = record["intelligenceIndexOutputTokensPerTask"]
        rows.append(
            {
                "answer_tokens_total": token_count["answer"],
                "configuration": f"adaptive_reasoning_{effort}_effort",
                "effort": effort,
                "name": record["name"],
                "output_tokens_per_task": per_task["output"],
                "output_tokens_total": token_count["output"],
                "reasoning_tokens_total": token_count["reasoning"],
                "score": record["intelligenceIndex"],
                "slug": slug,
            }
        )
    if len({row["slug"] for row in rows}) != 5:
        raise ValueError("AA does not contain exactly five distinct Opus 5 effort rows")
    selected = next(row for row in rows if row["effort"] == "max")
    if selected["score"] != 60.6918740157091:
        raise ValueError(f"Unexpected AA max score: {selected['score']}")

    article = normalized_text(
        payload_bytes(
            next(
                spec
                for spec in SOURCE_SPECS
                if spec.key == "artificial_analysis_launch_article"
            )
        )
    )
    fallback_claim = "Intelligence Index evaluations were run with Opus 4.8 fallback enabled"
    if fallback_claim not in article:
        raise ValueError("AA launch article no longer contains the fallback disclosure")
    return {
        "effort_rows": rows,
        "selected": {
            "configuration": selected["configuration"],
            "fallback_model": "Claude Opus 4.8",
            "name": selected["name"],
            "output_tokens_total": selected["output_tokens_total"],
            "score": selected["score"],
            "slug": selected["slug"],
        },
    }


def parse_epoch() -> dict[str, Any]:
    all_models = csv_rows(
        payload_bytes(next(spec for spec in SOURCE_SPECS if spec.key == "epoch_all_ai_models"))
    )
    model_rows = [row for row in all_models if row["Model"] == "Claude Opus 5"]
    if len(model_rows) != 1:
        raise ValueError(f"Expected one Epoch all-model row for Opus 5; found {len(model_rows)}")
    model = model_rows[0]
    if model["Publication date"] != "2026-07-24":
        raise ValueError("Epoch Opus 5 publication date mismatch")
    if model["Parameters"] or model["Training compute (FLOP)"]:
        raise ValueError("Epoch unexpectedly disclosed Opus 5 parameters or compute")

    components = csv_rows(
        payload_bytes(next(spec for spec in SOURCE_SPECS if spec.key == "epoch_eci_benchmarks"))
    )
    component_rows = [row for row in components if row["model"] == "Claude Opus 5"]
    if len(component_rows) != 12 or {row["model_id"] for row in component_rows} != {"m25"}:
        raise ValueError("Epoch canonical ECI component coverage for Opus 5 changed")

    archive_payload = payload_bytes(
        next(spec for spec in SOURCE_SPECS if spec.key == "epoch_benchmark_data")
    )
    with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
        published_rows = csv_rows(archive.read("epoch_capabilities_index.csv"))
    opus_rows = [row for row in published_rows if row["Model name"] == "Claude Opus 5"]
    if len(opus_rows) != 7:
        raise ValueError(f"Expected seven public Epoch Opus 5 variants; found {len(opus_rows)}")
    if {row["ECI Score"] for row in opus_rows} != {"159.38"}:
        raise ValueError("Epoch public ECI variants do not share displayed score 159.38")
    if any(row["Training compute (FLOP)"] for row in opus_rows):
        raise ValueError("Epoch public ECI table unexpectedly contains Opus 5 compute")
    if round(ECI_EXACT, 2) != 159.38 or not (ECI_CI_LOW <= ECI_EXACT <= ECI_CI_HIGH):
        raise ValueError("Pinned exact Epoch reproduction does not reconcile with display/CI")

    return {
        "canonical_component_benchmarks": len(component_rows),
        "displayed_eci": 159.38,
        "eci_ci_high": ECI_CI_HIGH,
        "eci_ci_low": ECI_CI_LOW,
        "eci_exact": ECI_EXACT,
        "eci_reproduction_method": "official Epoch ECI fit on the frozen canonical component rows",
        "model_id": "m25",
        "parameters": None,
        "published_configuration_rows": len(opus_rows),
        "training_compute_flop": None,
    }


def price_per_million(value: str) -> float | int:
    converted = Decimal(value) * Decimal(1_000_000)
    integral = converted.to_integral_value()
    return int(integral) if converted == integral else float(converted)


def parse_openrouter() -> dict[str, Any]:
    payload = payload_bytes(next(spec for spec in SOURCE_SPECS if spec.key == "openrouter_catalog"))
    records = {row["id"]: row for row in json.loads(payload)["data"]}
    tiers: dict[str, Any] = {}
    for tier, identifier in (
        ("standard", "anthropic/claude-opus-5"),
        ("fast", "anthropic/claude-opus-5-fast"),
    ):
        record = records[identifier]
        top = record["top_provider"]
        row = {
            "context_window_tokens": record["context_length"],
            "id": identifier,
            "input_usd_per_mtok": price_per_million(record["pricing"]["prompt"]),
            "max_output_tokens": top["max_completion_tokens"],
            "name": record["name"],
            "output_usd_per_mtok": price_per_million(record["pricing"]["completion"]),
        }
        if row["context_window_tokens"] != 1_000_000 or row["max_output_tokens"] != 128_000:
            raise ValueError(f"OpenRouter Opus 5 {tier} limits changed")
        tiers[tier] = row
    if (tiers["standard"]["input_usd_per_mtok"], tiers["standard"]["output_usd_per_mtok"]) != (5, 25):
        raise ValueError("OpenRouter standard price differs from first-party price")
    if (tiers["fast"]["input_usd_per_mtok"], tiers["fast"]["output_usd_per_mtok"]) != (10, 50):
        raise ValueError("OpenRouter fast price differs from first-party price")
    return {"tiers": tiers}


def parse_availability() -> dict[str, Any]:
    metr = payload_bytes(next(spec for spec in SOURCE_SPECS if spec.key == "metr_horizon"))
    folded = metr.decode("utf-8").casefold()
    aliases = ("claude_opus_5", "claude-opus-5", "claude opus 5", "opus_5")
    if any(alias in folded for alias in aliases):
        raise ValueError("Official METR YAML now contains Opus 5; update normalization")

    if not NO_COT_PANEL.is_file() or not IKP_MODEL_PANEL.is_file():
        raise FileNotFoundError("Frozen No-CoT/IKP coverage panels are missing")
    no_cot_rows = [row for row in csv_rows(NO_COT_PANEL.read_bytes()) if row["source"] == "No-CoT"]
    if len(no_cot_rows) < 100:
        raise ValueError("Frozen No-CoT panel is unexpectedly small")
    no_cot_names = {row["source_model_name"].casefold() for row in no_cot_rows}
    if "claude opus 5" in no_cot_names:
        raise ValueError("No-CoT panel now contains Opus 5; update normalization")

    ikp_models = json.loads(IKP_MODEL_PANEL.read_text(encoding="utf-8"))["models"]
    if len(ikp_models) < 100:
        raise ValueError("Frozen IKP replication inventory is unexpectedly small")
    ikp_text = json.dumps(ikp_models, ensure_ascii=False).casefold()
    if "claude opus 5" in ikp_text or "claude-opus-5" in ikp_text:
        raise ValueError("IKP replication inventory now contains Opus 5")
    return {
        "ikp": False,
        "ikp_basis": "absent from the frozen IKP replication model inventory",
        "metr": False,
        "metr_basis": "zero Opus 5 aliases in the frozen official METR-Horizon-v1.1 YAML",
        "no_cot": False,
        "no_cot_basis": "released after the frozen No-CoT paper model panel",
    }


def build_summary() -> dict[str, Any]:
    require_sources()
    anthropic_system_card, api = parse_anthropic()
    artificial_analysis = parse_artificial_analysis()
    epoch = parse_epoch()
    openrouter = parse_openrouter()
    availability = parse_availability()

    source_files = {spec.key: relative(spec.path) for spec in SOURCE_SPECS}
    source_hashes = {
        spec.key: sha256_bytes(spec.path.read_bytes()) for spec in SOURCE_SPECS
    }
    coverage_source_files = {
        "no_cot_model_panel": relative(NO_COT_PANEL),
        "ikp_model_inventory": relative(IKP_MODEL_PANEL),
    }
    coverage_source_hashes = {
        key: sha256_bytes((ROOT / path).read_bytes())
        for key, path in coverage_source_files.items()
    }
    return {
        "anthropic_system_card": anthropic_system_card,
        "api": api,
        "artificial_analysis": artificial_analysis,
        "availability": availability,
        "epoch": epoch,
        "identity": {
            "api_model_id": "claude-opus-5",
            "base_identity_policy": "unique_base",
            "canonical_name": "Claude Opus 5",
            "parameter_disclosed": False,
            "parameter_status": "unknown; no first-party count and AA/Epoch parameter fields are blank",
            "release_date": "2026-07-24",
            "same_weight_identity_disclosed": False,
            "same_weight_identity_status": "no first-party same-weight identity claim; unique_base is a modeling policy",
        },
        "openrouter": openrouter,
        "coverage_source_files": coverage_source_files,
        "coverage_source_hashes": coverage_source_hashes,
        "source_files": source_files,
        "source_hashes": source_hashes,
    }


def write_or_verify_summary(summary: dict[str, Any], *, replace: bool) -> None:
    expected = canonical_json(summary)
    if replace or not SUMMARY.exists():
        atomic_write(SUMMARY, expected)
    actual = SUMMARY.read_bytes()
    if actual != expected:
        raise ValueError(
            f"{relative(SUMMARY)} is stale or hand-edited; run with --refresh to replace it"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch, validate, and replace the dated frozen source bundle.",
    )
    parser.add_argument(
        "--rebuild-summary",
        action="store_true",
        help="Rebuild only the normalized summary from frozen local inputs; never access the network.",
    )
    args = parser.parse_args()
    if args.refresh:
        refresh_sources()
    summary = build_summary()
    write_or_verify_summary(summary, replace=args.refresh or args.rebuild_summary)
    print(
        json.dumps(
            {
                "summary": relative(SUMMARY),
                "source_files": len(SOURCE_SPECS),
                "release_date": summary["identity"]["release_date"],
                "aa_score": summary["artificial_analysis"]["selected"]["score"],
                "epoch_eci": summary["epoch"]["eci_exact"],
                "parameter_disclosed": summary["identity"]["parameter_disclosed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
