#!/usr/bin/env python3
"""Freeze and normalize the post-release Kimi K3 architecture evidence.

Kimi's launch page disclosed a rounded 2.8T total but initially omitted an
activated-parameter count.  The subsequently released technical report and
official model card provide the exact architectural table: 2.78T total and
104.2B activated parameters.  This collector pins both artifacts to one
MoonshotAI/Kimi-K3 commit so an ordinary forecast build is fully offline and
cannot silently move with the repository's main branch.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"
DATE = "2026-07-31"
PINNED_COMMIT = "7c5be9599120d7993748de66a76128614f15f210"
REPOSITORY = "https://github.com/MoonshotAI/Kimi-K3"
MODEL_CARD = SOURCES / f"kimi_k3_official_model_card_{DATE}.md"
TECHNICAL_REPORT = SOURCES / f"kimi_k3_technical_report_{DATE}.pdf"
HF_CONFIG = SOURCES / f"kimi_k3_hf_config_{DATE}.json"
LAUNCH_HTML = SOURCES / f"kimi_k3_official_launch_{DATE}.html"
HF_COMMIT_HISTORY = SOURCES / f"kimi_k3_hf_commit_history_{DATE}.json"
SUMMARY = SOURCES / f"kimi_k3_release_evidence_{DATE}.json"
MODEL_CARD_URL = (
    "https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/"
    f"{PINNED_COMMIT}/README.md"
)
TECHNICAL_REPORT_URL = (
    "https://raw.githubusercontent.com/MoonshotAI/Kimi-K3/"
    f"{PINNED_COMMIT}/k3_tech_report.pdf"
)
HF_CONFIG_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
HF_CONFIG_URL = (
    "https://huggingface.co/moonshotai/Kimi-K3/raw/"
    f"{HF_CONFIG_REVISION}/config.json"
)
LAUNCH_URL = "https://www.kimi.com/blog/kimi-k3"
HF_COMMIT_HISTORY_URL = (
    "https://huggingface.co/api/models/moonshotai/Kimi-K3/commits/main?limit=100"
)
HF_INITIAL_WEIGHTS_COMMIT = "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", delete=False
    )
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


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "frontier-parameter-model/1.0 Kimi-K3 evidence audit",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise ValueError(f"Empty response from {url}")
    return payload


def normalized_pdf_text(payload: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(payload))
    text = " ".join((page.extract_text() or "") for page in reader.pages)
    text = re.sub(r"\s+", " ", text).strip()
    return text, len(reader.pages)


def require(pattern: str, text: str, label: str) -> None:
    if not re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
        raise ValueError(f"Kimi K3 source is missing {label}: {pattern}")


def validate_sources(
    model_card: bytes,
    report: bytes,
    hf_config: bytes,
    launch_html: bytes,
    hf_commit_history: bytes,
) -> dict[str, Any]:
    if len(model_card) < 20_000:
        raise ValueError("Kimi K3 model card is unexpectedly small")
    if not report.startswith(b"%PDF-") or len(report) < 1_000_000:
        raise ValueError("Kimi K3 technical report is not a substantial PDF")

    card = model_card.decode("utf-8")
    card_plain = re.sub(r"<[^>]+>", " ", card)
    card_plain = re.sub(r"[*_`#|]+", " ", card_plain)
    card_plain = re.sub(r"\s+", " ", card_plain).strip()
    report_text, pages = normalized_pdf_text(report)
    if pages != 47:
        raise ValueError(f"Expected a 47-page Kimi K3 report; found {pages}")

    card_checks = {
        "official_open_weights": r"Open Frontier Weights.*release the full Kimi K3 model weights",
        "rounded_total": r"Total Parameters\s+2\.8T",
        "rounded_active": r"Activated Parameters\s+104B",
        "layers": r"Number of Layers\s+93",
        "routed_experts": r"Number of Experts\s+896",
        "selected_experts": r"Selected Experts per Token\s+16",
        "shared_experts": r"Number of Shared Experts\s+2",
        "vision_parameters": r"Parameters of Vision Encoder\s+401M",
    }
    for label, pattern in card_checks.items():
        require(pattern, card_plain, label)

    report_checks = {
        "exact_total": r"Total Parameters 1\.04T 2\.78T",
        "exact_active": r"Activated Parameters 32\.6B 104\.2B",
        "scaling_efficiency": r"approximately\s*2\.5.?\s*improvement in overall scaling efficiency",
        "pretraining_context": r"pre-training begins with a context length of 8k tokens.*extended to 64k",
        "cooldown_context": r"from 256K to 1M tokens during the cooldown phase",
        "posttraining_stages": r"three-stage paradigm:.*supervised fine-tuning.*Reinforcement Learning.*Multi-Teacher On-Policy Distillation",
        "nine_rl_experts": r"three domain experts with three reasoning effort levels.*yields a total of nine expert models",
    }
    for label, pattern in report_checks.items():
        require(pattern, report_text, label)

    config = json.loads(hf_config)
    text = config["text_config"]
    expected = {
        "hidden_size": 7168,
        "num_hidden_layers": 93,
        "num_attention_heads": 96,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_shared_experts": 2,
        "first_k_dense_replace": 1,
        "moe_intermediate_size": 3072,
        "routed_expert_hidden_size": 3584,
        "max_position_embeddings": 1_048_576,
        "num_nextn_predict_layers": 0,
    }
    mismatches = {
        key: {"expected": value, "observed": text.get(key)}
        for key, value in expected.items()
        if text.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Kimi K3 HF config mismatch: {mismatches}")
    quantization = text["quantization_config"]
    if (
        quantization.get("format") != "mxfp4-pack-quantized"
        or quantization["config_groups"]["group_0"]["weights"].get("group_size")
        != 32
    ):
        raise ValueError("Kimi K3 HF config lacks the expected group-32 MXFP4 format")

    launch = launch_html.decode("utf-8", "replace")
    require(r"2026-07-16", launch, "official launch date")
    require(r"2\.8T", launch, "official rounded launch scale")

    commits = json.loads(hf_commit_history)
    initial = next(
        (row for row in commits if row.get("id") == HF_INITIAL_WEIGHTS_COMMIT), None
    )
    if initial is None:
        raise ValueError("Kimi K3 HF history lacks the pinned initial weights commit")
    if initial.get("date") != "2026-07-27T13:31:26.000Z":
        raise ValueError(f"Unexpected Kimi K3 initial commit date: {initial.get('date')}")

    return {
        "technical_report_pages": pages,
        "hf_config_revision": HF_CONFIG_REVISION,
        "hf_config_architecture_fields_verified": len(expected),
        "hf_initial_weights_commit": HF_INITIAL_WEIGHTS_COMMIT,
        "hf_initial_weights_commit_utc": initial["date"],
    }


def build_record(
    model_card: bytes,
    report: bytes,
    hf_config: bytes,
    launch_html: bytes,
    hf_commit_history: bytes,
) -> dict[str, Any]:
    validation = validate_sources(
        model_card, report, hf_config, launch_html, hf_commit_history
    )
    total_b = 2780.0
    active_b = 104.2
    k2_total_b = 1040.0
    k2_active_b = 32.6
    return {
        "schema_version": "1.0",
        "snapshot_date": DATE,
        "source_policy": {
            "repository": REPOSITORY,
            "pinned_commit": PINNED_COMMIT,
            "network_refresh_is_explicit": True,
            "ordinary_build_is_offline": True,
        },
        "kimi_k3": {
            "initial_model_release_date": "2026-07-16",
            "weights_release_date": "2026-07-27",
            "architecture": "Mixture-of-Experts (MoE)",
            "total_parameters_b_exact": total_b,
            "total_parameters_t_display": 2.8,
            "activated_parameters_b_exact": active_b,
            "parameter_count_disclosed": True,
            "activated_parameter_count_disclosed": True,
            "weights_released": True,
            "layers": 93,
            "dense_layers": 1,
            "attention_layer_composition": {"kda": 69, "gated_mla": 24},
            "attention_hidden_dimension": 7168,
            "attention_heads": 96,
            "latent_moe_dimension": 3584,
            "moe_hidden_dimension_per_expert": 3072,
            "routed_experts": 896,
            "selected_routed_experts_per_token": 16,
            "shared_experts": 2,
            "vocabulary_size": 160_000,
            "maximum_context_tokens": 1_048_576,
            "vision_encoder": "MoonViT-V2",
            "vision_encoder_parameters_b": 0.401,
            "vision_encoder_layers": 27,
            "quantization": "MXFP4 expert weights / MXFP8 activations from SFT onward",
            "released_config_next_token_prediction_layers": 0,
        },
        "kimi_k2_comparator": {
            "total_parameters_b_exact": k2_total_b,
            "activated_parameters_b_exact": k2_active_b,
            "layers": 61,
            "routed_experts": 384,
            "selected_routed_experts_per_token": 8,
            "shared_experts": 1,
            "training_context_tokens": 131_072,
        },
        "training_disclosures": {
            "pretraining_domains": [
                "Web Text",
                "Code",
                "Mathematics",
                "Knowledge",
                "Vision",
            ],
            "pretraining_context_curriculum": [8192, 65_536],
            "cooldown_context_curriculum": [262_144, 1_048_576],
            "posttraining_stages": [
                "supervised fine-tuning",
                "reinforcement learning",
                "multi-teacher on-policy distillation",
            ],
            "rl_domain_experts": 3,
            "rl_reasoning_effort_levels": 3,
            "rl_teacher_experts": 9,
            "exact_pretraining_tokens_disclosed": False,
            "exact_pretraining_flops_disclosed": False,
            "exact_posttraining_flops_disclosed": False,
            "reported_scaling_efficiency_vs_k2": 2.5,
        },
        "derived_quantities": {
            "total_to_activated_parameter_ratio": total_b / active_b,
            "activated_parameter_fraction": active_b / total_b,
            "selected_routed_expert_fraction": 16 / 896,
            "k2_total_to_activated_parameter_ratio": k2_total_b / k2_active_b,
            "k3_over_k2_total_parameter_ratio": total_b / k2_total_b,
            "k3_over_k2_activated_parameter_ratio": active_b / k2_active_b,
        },
        "interpretation_limits": [
            "Activated parameters include attention, shared experts, routers, embeddings, and other non-routed components; 16/896 is not the active-parameter fraction.",
            "The reported 2.5x scaling-efficiency improvement combines architecture, data, and training-recipe changes and is not a parameter-count multiplier.",
            "The report plots scaling against FLOPs but does not disclose K3's exact pretraining or post-training FLOP total.",
            "The report describes data domains and context curricula but does not disclose an exact pretraining-token total.",
        ],
        "validation": validation,
        "source_files": {
            "official_model_card": {
                "path": str(MODEL_CARD.relative_to(ROOT)),
                "url": MODEL_CARD_URL,
                "bytes": len(model_card),
                "sha256": sha256_bytes(model_card),
            },
            "official_technical_report": {
                "path": str(TECHNICAL_REPORT.relative_to(ROOT)),
                "url": TECHNICAL_REPORT_URL,
                "arxiv_url": "https://arxiv.org/abs/2607.24653",
                "bytes": len(report),
                "sha256": sha256_bytes(report),
            },
            "official_huggingface_config": {
                "path": str(HF_CONFIG.relative_to(ROOT)),
                "url": HF_CONFIG_URL,
                "revision": HF_CONFIG_REVISION,
                "bytes": len(hf_config),
                "sha256": sha256_bytes(hf_config),
            },
            "official_launch_page": {
                "path": str(LAUNCH_HTML.relative_to(ROOT)),
                "url": LAUNCH_URL,
                "bytes": len(launch_html),
                "sha256": sha256_bytes(launch_html),
            },
            "official_huggingface_commit_history": {
                "path": str(HF_COMMIT_HISTORY.relative_to(ROOT)),
                "url": HF_COMMIT_HISTORY_URL,
                "initial_weights_commit": HF_INITIAL_WEIGHTS_COMMIT,
                "bytes": len(hf_commit_history),
                "sha256": sha256_bytes(hf_commit_history),
            },
        },
    }


def refresh() -> None:
    model_card = fetch(MODEL_CARD_URL)
    report = fetch(TECHNICAL_REPORT_URL)
    hf_config = fetch(HF_CONFIG_URL)
    launch_html = fetch(LAUNCH_URL)
    hf_commit_history = fetch(HF_COMMIT_HISTORY_URL)
    record = build_record(
        model_card, report, hf_config, launch_html, hf_commit_history
    )
    atomic_write(MODEL_CARD, model_card)
    atomic_write(TECHNICAL_REPORT, report)
    atomic_write(HF_CONFIG, hf_config)
    atomic_write(LAUNCH_HTML, launch_html)
    atomic_write(HF_COMMIT_HISTORY, hf_commit_history)
    atomic_write(SUMMARY, canonical_json(record))


def verify_offline() -> dict[str, Any]:
    if (
        not MODEL_CARD.is_file()
        or not TECHNICAL_REPORT.is_file()
        or not HF_CONFIG.is_file()
        or not LAUNCH_HTML.is_file()
        or not HF_COMMIT_HISTORY.is_file()
        or not SUMMARY.is_file()
    ):
        raise FileNotFoundError(
            "Frozen Kimi K3 release evidence is incomplete; run with --refresh"
        )
    rebuilt = build_record(
        MODEL_CARD.read_bytes(),
        TECHNICAL_REPORT.read_bytes(),
        HF_CONFIG.read_bytes(),
        LAUNCH_HTML.read_bytes(),
        HF_COMMIT_HISTORY.read_bytes(),
    )
    expected = canonical_json(rebuilt)
    actual = SUMMARY.read_bytes()
    if actual != expected:
        raise ValueError(
            "Kimi K3 normalized evidence does not reproduce byte-for-byte from frozen sources"
        )
    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch the two commit-pinned official artifacts before verification",
    )
    args = parser.parse_args()
    if args.refresh:
        refresh()
    record = verify_offline()
    print(
        json.dumps(
            {
                "summary": str(SUMMARY.relative_to(ROOT)),
                "pinned_commit": PINNED_COMMIT,
                "total_parameters_b": record["kimi_k3"]["total_parameters_b_exact"],
                "activated_parameters_b": record["kimi_k3"]["activated_parameters_b_exact"],
                "total_to_active_ratio": record["derived_quantities"]["total_to_activated_parameter_ratio"],
                "training_tokens_disclosed": record["training_disclosures"]["exact_pretraining_tokens_disclosed"],
                "training_flops_disclosed": record["training_disclosures"]["exact_pretraining_flops_disclosed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
