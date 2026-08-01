import json
import hashlib
import math
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from aa_score_availability import (
    LEDGER_PATH as AA_SCORE_AVAILABILITY_PATH,
    RAW_PATH as AA_SCORE_CHANGELOG_PATH,
    resolve_aa_score_availability,
)
from open_model_parameter_truth import (
    LEDGER_PATH as OPEN_MODEL_PARAMETER_TRUTH_PATH,
    apply_parameter_truth,
)


AS_OF = date(2026, 7, 31)
DATE_ORIGIN = date(2026, 1, 1)
ROOT = Path(__file__).resolve().parent
ECI_REPRODUCTION_PATH = ROOT / "sources" / "epoch_eci_reproduced_scores_2026-07-31.csv"
ECI_REPRODUCTION_METADATA_PATH = ROOT / "sources" / "epoch_eci_reproduction_metadata_2026-07-31.json"
ECI_REPRODUCTION_CROSSCHECK_PATH = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7" / "epoch_eci_reproduction_crosscheck_2026-07-31.csv"
ECI_REPRODUCTION_AUDIT_PATH = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7" / "epoch_eci_reproduction_audit_2026-07-31.json"
ECI_SNAPSHOT_MANIFEST_PATH = ROOT / "sources" / "epoch_snapshot_manifest_2026-07-31.json"
OPUS5_EVIDENCE_PATH = ROOT / "sources" / "claude_opus_5_evidence_2026-07-31.json"
K3_EVIDENCE_PATH = ROOT / "sources" / "kimi_k3_release_evidence_2026-07-31.json"
AA_DETAILED_PATH = ROOT / "sources" / "aa_detailed_model_signals_2026-07-31.csv"
OPUS5_EVIDENCE = json.loads(OPUS5_EVIDENCE_PATH.read_text(encoding="utf-8"))
OPUS5_IDENTITY = OPUS5_EVIDENCE["identity"]
OPUS5_AA = OPUS5_EVIDENCE["artificial_analysis"]["selected"]
OPUS5_EPOCH = OPUS5_EVIDENCE["epoch"]
K3_EVIDENCE = json.loads(K3_EVIDENCE_PATH.read_text(encoding="utf-8"))
K3_ARCHITECTURE = K3_EVIDENCE["kimi_k3"]
if (
    OPUS5_IDENTITY["canonical_name"] != "Claude Opus 5"
    or OPUS5_IDENTITY["release_date"] != "2026-07-24"
    or OPUS5_IDENTITY["parameter_disclosed"]
    or OPUS5_IDENTITY["same_weight_identity_disclosed"]
    or OPUS5_IDENTITY["base_identity_policy"] != "unique_base"
):
    raise RuntimeError("Claude Opus 5 evidence violates the distinct undisclosed-base policy")
if (
    float(K3_ARCHITECTURE["total_parameters_b_exact"]) != 2780.0
    or float(K3_ARCHITECTURE["activated_parameters_b_exact"]) != 104.2
):
    raise RuntimeError("Kimi K3 primary architecture evidence is missing or imprecise")


# One canonical Artificial Analysis Intelligence Index v4.1 endpoint per checkpoint.
# Parameter counts are billions. `estimated` is Artificial Analysis' own score-quality flag.
OPEN_MODELS_ROUNDED = [
    ("2025-07-21", "Qwen3-235B-A22B-Instruct-2507", 235, 22, 0, 18, 1, None, "MoE", "qwen3_235", 0, 0),
    ("2025-07-22", "Qwen3-Coder-480B-A35B-Instruct", 480, 35, 0, 18, 1, None, "MoE/coder", "qwen3_coder", 1, 0),
    ("2025-07-28", "GLM-4.5-Air", 106, 12, 1, 17, 1, None, "MoE/hybrid-thinking", "glm", 0, 0),
    ("2025-07-28", "GLM-4.5", 355, 32, 1, 19, 1, None, "MoE/hybrid-thinking", "glm", 0, 0),
    ("2025-07-30", "Qwen3-30B-A3B-2507 Reasoning", 30.5, 3.3, 1, 14, 0, None, "MoE", "qwen3_30", 0, 0),
    ("2025-08-05", "gpt-oss-20b (high)", 20.91, 3.61, 1, 15, 0, None, "MoE", "gpt_oss", 0, 0),
    ("2025-08-05", "gpt-oss-120b (high)", 116.83, 5.13, 1, 24, 0, None, "MoE", "gpt_oss", 0, 0),
    ("2025-08-20", "Seed-OSS-36B-Instruct", 36.2, 36.2, 1, 18, 1, None, "dense", "seed_oss", 0, 0),
    ("2025-08-21", "DeepSeek-V3.1 Reasoning", 671, 37, 1, 21, 1, None, "MoE", "deepseek_v3", 0, 0),
    ("2025-09-30", "GLM-4.6 Reasoning", 357, 32, 1, 29, 0, None, "MoE", "glm", 0, 0),
    ("2025-10-26", "MiniMax-M2", 230, 10, 1, 28, 1, None, "MoE", "minimax_m2", 0, 0),
    ("2025-11-06", "Kimi K2 Thinking", 1000, 32, 1, 33, 1, None, "MoE", "kimi_k2", 0, 0),
    ("2025-11-20", "OLMo 3 32B Think", 32.2, 32.2, 1, 6, 1, None, "dense", "olmo3", 0, 0),
    ("2025-12-01", "DeepSeek-V3.2 Reasoning", 671, 37, 1, 32, 0, None, "MoE/DSA", "deepseek_v3", 0, 0),
    ("2025-12-02", "Mistral Large 3", 675, 41, 0, 16, 0, None, "MoE/multimodal", "mistral_large3", 0, 1),
    ("2025-12-12", "OLMo 3.1 32B Think", 32.2, 32.2, 1, 8, 1, None, "dense", "olmo3", 0, 0),
    ("2025-12-15", "Nemotron 3 Nano 30B A3B Reasoning", 31.6, 3.6, 1, 14, 0, None, "MoE/Mamba-hybrid", "nemotron3", 0, 0),
    ("2025-12-16", "MiMo-V2-Flash Reasoning", 309, 15, 1, 31, 0, None, "MoE", "mimo_v2", 0, 0),
    ("2025-12-22", "GLM-4.7 Reasoning", 358, 32, 1, 34, 0, None, "MoE", "glm", 0, 0),
    ("2025-12-23", "MiniMax-M2.1", 230, 10, 1, 31, 1, None, "MoE", "minimax_m2", 0, 0),
    ("2025-12-31", "K-EXAONE", 236, 23, 1, 25, 1, None, "MoE", "k_exaone", 0, 0),
    ("2026-01-27", "Kimi K2.5 Reasoning", 1000, 32, 1, 35, 0, None, "MoE/multimodal", "kimi_k2", 0, 1),
    ("2026-01-28", "LongCat Flash Lite", 68.5, 3, 0, 17, 1, None, "MoE", "longcat", 0, 0),
    ("2026-02-11", "GLM-5 Reasoning", 744, 40, 1, 40, 1, None, "MoE/DSA", "glm", 0, 0),
    ("2026-02-12", "MiniMax-M2.5", 230, 10, 1, 34, 1, None, "MoE", "minimax_m2", 0, 0),
    ("2026-02-16", "Qwen3.5-397B-A17B Reasoning", 397, 17, 1, 34, 0, None, "MoE/Gated-DeltaNet/multimodal", "qwen35", 0, 1),
    ("2026-02-24", "Qwen3.5-35B-A3B Reasoning", 36, 3, 1, 29, 1, None, "MoE/Gated-DeltaNet", "qwen35", 0, 0),
    ("2026-02-24", "Qwen3.5-122B-A10B Reasoning", 125, 10, 1, 32, 0, 96, "MoE/Gated-DeltaNet", "qwen35", 0, 0),
    ("2026-02-24", "Qwen3.5-27B Reasoning", 27.8, 27.8, 1, 34, 1, None, "dense/Gated-DeltaNet", "qwen35", 0, 0),
    ("2026-03-02", "Qwen3.5-0.8B Reasoning", 0.873, 0.873, 1, 5, 0, None, "dense", "qwen35", 0, 0),
    ("2026-03-02", "Qwen3.5-2B Reasoning", 2.27, 2.27, 1, 8, 0, None, "dense", "qwen35", 0, 0),
    ("2026-03-02", "Qwen3.5-4B Reasoning", 4.66, 4.66, 1, 20, 1, None, "dense", "qwen35", 0, 0),
    ("2026-03-02", "Qwen3.5-9B Reasoning", 9.7, 9.7, 1, 21, 0, None, "dense", "qwen35", 0, 0),
    ("2026-03-11", "Nemotron 3 Super 120B A12B", 120.6, 12.7, 1, 25, 0, None, "MoE/LatentMoE/Mamba2", "nemotron3", 0, 0),
    ("2026-03-18", "MiniMax-M2.7", 230, 10, 1, 38, 0, None, "MoE", "minimax_m2", 0, 0),
    ("2026-04-07", "GLM-5.1 Reasoning", 744, 40, 1, 40, 0, 120, "MoE", "glm", 0, 0),
    ("2026-04-16", "Qwen3.6-35B-A3B Reasoning", 36, 3, 1, 32, 0, 150, "MoE/Gated-DeltaNet/multimodal", "qwen36", 0, 1),
    ("2026-04-20", "Kimi K2.6 Reasoning", 1000, 32, 1, 44, 0, None, "MoE/multimodal", "kimi_k2", 0, 1),
    ("2026-04-22", "MiMo-V2.5", 310, 15, 1, 37, 0, 71, "MoE/multimodal", "mimo_v2", 0, 1),
    ("2026-04-22", "Qwen3.6-27B Reasoning", 27.8, 27.8, 1, 37, 0, 140, "dense/Gated-DeltaNet/multimodal", "qwen36", 0, 1),
    ("2026-04-24", "DeepSeek-V4-Pro (max)", 1600, 49, 1, 44, 0, None, "MoE/hybrid-compressed-attention", "deepseek_v4", 0, 0),
    ("2026-06-01", "MiniMax-M3", 428, 23, 1, 44, 0, 89, "MoE/multimodal/sparse-attention", "minimax_m3", 0, 1),
    ("2026-06-02", "Nex-N2-Pro", 397, 17, 1, 41, 0, 97, "MoE/multimodal", "qwen35", 0, 1),
    ("2026-06-04", "Nemotron 3 Ultra 550B A55B", 550, 55, 1, 38, 0, None, "MoE/LatentMoE/Mamba2", "nemotron3", 0, 0),
    ("2026-06-09", "North Mini Code", 30, 3, 1, 21, 1, None, "MoE/coder", "north", 1, 0),
    ("2026-06-12", "Kimi K2.7 Code", 1000, 32, 1, 42, 0, 100, "MoE/coder/multimodal", "kimi_k2", 1, 1),
    ("2026-06-16", "GLM-5.2 (max)", 744, 40, 1, 51, 0, 140, "MoE/IndexShare-sparse-attention", "glm", 0, 0),
    ("2026-06-29", "LongCat 2.0", 1600, 48, 1, 33, 0, 130, "MoE", "longcat", 0, 0),
    ("2026-07-06", "Hy3", 295, 21, 1, 41, 0, 140, "MoE/hybrid-thinking", "hy3", 0, 0),
    ("2026-07-15", "Inkling (xhigh)", 975, 41, 1, 41, 0, 130, "MoE/multimodal", "inkling", 0, 1),
]


def aa_name_key(value: str) -> str:
    value = value.casefold()
    value = re.sub(
        r"\b(reasoning|non-reasoning|instruct|thinking|think|max|xhigh|high|medium|low|effort)\b",
        " ",
        value,
    )
    return re.sub(r"[^a-z0-9]+", "", value)


AA_MANUAL_SLUGS = {
    "Nemotron 3 Nano 30B A3B Reasoning": "nvidia-nemotron-3-nano-30b-a3b-reasoning",
    "Nemotron 3 Super 120B A12B": "nvidia-nemotron-3-super-120b-a12b",
    "DeepSeek-V4-Pro (max)": "deepseek-v4-pro",
}


def load_exact_aa_inputs():
    frame = pd.read_csv(AA_DETAILED_PATH, dtype=str, keep_default_na=False)
    if len(frame) != 587 or frame["model_id"].nunique() != 587 or frame["slug"].nunique() != 587:
        raise RuntimeError("Detailed AA snapshot must contain 587 unique model records")
    by_slug = {row["slug"]: row for _, row in frame.iterrows()}
    by_key: dict[str, list[pd.Series]] = {}
    for _, row in frame.iterrows():
        by_key.setdefault(aa_name_key(row["name"]), []).append(row)

    hydrated = []
    audit = []
    selected_slugs = set()
    for source in OPEN_MODELS_ROUNDED:
        model = source[1]
        method = "manual exact checkpoint slug" if model in AA_MANUAL_SLUGS else "normalized exact checkpoint label; highest score"
        candidates = (
            [by_slug[AA_MANUAL_SLUGS[model]]]
            if model in AA_MANUAL_SLUGS
            else by_key.get(aa_name_key(model), [])
        )
        candidates = [row for row in candidates if row["intelligence_index"]]
        if not candidates:
            raise RuntimeError(f"No exact detailed-AA checkpoint for {model}")
        selected = max(candidates, key=lambda row: float(row["intelligence_index"]))
        if selected["slug"] in selected_slugs:
            raise RuntimeError(f"Detailed-AA checkpoint reused: {selected['slug']}")
        selected_slugs.add(selected["slug"])
        release_date_delta_days = (
            date.fromisoformat(selected["release_date"])
            - date.fromisoformat(source[0])
        ).days
        exact_score = float(selected["intelligence_index"])
        score_timing = resolve_aa_score_availability(
            {
                "aa_slug": selected["slug"],
                "aa_name": selected["name"],
                "release_date": source[0],
            }
        )
        hydrated.append((*source[:5], exact_score, *source[6:]))
        audit.append(
            {
                "model": model,
                "aa_model_id": selected["model_id"],
                "aa_slug": selected["slug"],
                "aa_name": selected["name"],
                "canonical_release_date": source[0],
                "aa_release_date": selected["release_date"],
                "aa_minus_canonical_release_days": release_date_delta_days,
                "release_date_policy": (
                    "canonical first-availability date retained; AA configuration date preserved separately"
                ),
                "rounded_legacy_score": float(source[5]),
                "exact_score": exact_score,
                "candidate_count": len(candidates),
                "selection_method": method,
                "aa_score_available_date": score_timing.get(
                    "score_available_date", source[0]
                ),
                "aa_score_availability_verified": bool(score_timing),
                "aa_score_publication_event_id": score_timing.get(
                    "changelog_event_id", ""
                ),
                "aa_score_at_publication": score_timing.get(
                    "intelligence_index_at_publication"
                ),
            }
        )
    if len(hydrated) != 50 or len(selected_slugs) != 50:
        raise RuntimeError("Detailed-AA calibration crosswalk is incomplete or duplicative")
    return hydrated, audit, by_slug


OPEN_MODELS, OPEN_AA_MATCH_AUDIT, AA_DETAILED_BY_SLUG = load_exact_aa_inputs()
OPEN_AA_MATCH_BY_MODEL = {row["model"]: row for row in OPEN_AA_MATCH_AUDIT}


def frontier_aa(slug: str) -> tuple[float, float]:
    row = AA_DETAILED_BY_SLUG.get(slug)
    if row is None or not row["intelligence_index"] or not row["intelligence_output_tokens_total"]:
        raise RuntimeError(f"Missing exact detailed-AA frontier signal: {slug}")
    return (
        float(row["intelligence_index"]),
        float(row["intelligence_output_tokens_total"]) / 1_000_000,
    )


FRONTIER_INPUTS = [
    (
        OPUS5_IDENTITY["release_date"],
        OPUS5_IDENTITY["canonical_name"],
        float(OPUS5_AA["score"]),
        float(OPUS5_AA["output_tokens_total"]) / 1_000_000,
        "adaptive max + Opus 4.8 fallback",
        "fallback-enabled evaluation; distinct new pretrain; physical size and architecture undisclosed",
    ),
    ("2026-07-09", "GPT-5.6 Sol", *frontier_aa("gpt-5-6-sol"), "max", "pure"),
    ("2026-06-09", "Claude Fable 5", *frontier_aa("claude-fable-5"), "adaptive max + fallback", "pure"),
    ("2026-05-28", "Claude Opus 4.8", *frontier_aa("claude-opus-4-8"), "adaptive max", "pure"),
    ("2026-07-09", "GPT-5.6 Terra", *frontier_aa("gpt-5-6-terra"), "max", "pure"),
    ("2026-04-23", "GPT-5.5", *frontier_aa("gpt-5-5"), "xhigh", "pure"),
    ("2026-04-16", "Claude Opus 4.7", *frontier_aa("claude-opus-4-7"), "adaptive max", "pure"),
    ("2026-07-08", "Grok 4.5", *frontier_aa("grok-4-5"), "high", "pure; disclosed 1.5T total"),
    ("2026-06-30", "Claude Sonnet 5", *frontier_aa("claude-sonnet-5"), "adaptive max", "pure"),
    ("2026-07-09", "GPT-5.6 Luna", *frontier_aa("gpt-5-6-luna"), "max", "pure"),
    ("2026-05-19", "Gemini 3.5 Flash", *frontier_aa("gemini-3-5-flash"), "high", "pure"),
    ("2026-02-19", "Gemini 3.1 Pro Preview", *frontier_aa("gemini-3-1-pro-preview"), "reasoning/default", "pure"),
]

FRONTIER_AA_SLUGS = {
    "Claude Opus 5": "claude-opus-5",
    "GPT-5.6 Sol": "gpt-5-6-sol",
    "Claude Fable 5": "claude-fable-5",
    "Claude Opus 4.8": "claude-opus-4-8",
    "GPT-5.6 Terra": "gpt-5-6-terra",
    "GPT-5.5": "gpt-5-5",
    "Claude Opus 4.7": "claude-opus-4-7",
    "Grok 4.5": "grok-4-5",
    "Claude Sonnet 5": "claude-sonnet-5",
    "GPT-5.6 Luna": "gpt-5-6-luna",
    "Gemini 3.5 Flash": "gemini-3-5-flash",
    "Gemini 3.1 Pro Preview": "gemini-3-1-pro-preview",
}
FRONTIER_AA_MATCH_AUDIT = []
for release, model, exact_score, output_m, *_ in FRONTIER_INPUTS:
    slug = FRONTIER_AA_SLUGS[model]
    source = AA_DETAILED_BY_SLUG[slug]
    source_score, source_output_m = frontier_aa(slug)
    if (
        source["release_date"] != release
        or not math.isclose(source_score, exact_score, rel_tol=0, abs_tol=1e-12)
        or not math.isclose(source_output_m, output_m, rel_tol=0, abs_tol=1e-12)
    ):
        raise RuntimeError(f"Frontier detailed-AA crosscheck failed for {model}")
    FRONTIER_AA_MATCH_AUDIT.append(
        {
            "model": model,
            "aa_model_id": source["model_id"],
            "aa_slug": slug,
            "aa_name": source["name"],
            "release_date": release,
            "exact_score": exact_score,
            "output_tokens_total": int(float(source["intelligence_output_tokens_total"])),
            "selection_method": "explicit checkpoint slug; highest requested effort",
            **(
                lambda timing: {
                    "aa_score_available_date": timing.get(
                        "score_available_date", release
                    ),
                    "aa_score_availability_verified": bool(timing),
                    "aa_score_publication_event_id": timing.get(
                        "changelog_event_id", ""
                    ),
                    "aa_score_at_publication": timing.get(
                        "intelligence_index_at_publication"
                    ),
                }
            )(
                resolve_aa_score_availability(
                    {
                        "aa_slug": slug,
                        "aa_name": source["name"],
                        "release_date": release,
                    }
                )
            ),
        }
    )

ECI_FRONTIER_ALIASES = {
    "Gemini 3.1 Pro Preview": "Gemini 3.1 Pro",
}


def load_reproduced_eci_scores():
    frame = pd.read_csv(ECI_REPRODUCTION_PATH)
    required = {"Model", "eci", "eci_ci_low", "eci_ci_high", "date", "model_version", "source"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"ECI reproduction is missing columns: {sorted(missing)}")
    if len(frame) != 213 or frame["Model"].nunique() != 213:
        raise RuntimeError(
            f"Expected 213 unique reproduced ECI models; found {len(frame)}/{frame['Model'].nunique()}"
        )
    anchors = frame["Model"].isin({"Claude 3.5 Sonnet", "GPT-5"})
    if frame["eci"].isna().any() or frame.loc[~anchors, ["eci_ci_low", "eci_ci_high"]].isna().any().any():
        raise RuntimeError("Reproduced ECI ledger contains unexpected missing score or interval values")
    if frame.loc[anchors, ["eci_ci_low", "eci_ci_high"]].notna().any().any():
        raise RuntimeError("ECI anchor models should have undefined bootstrap intervals")
    bounded = frame.loc[~anchors]
    if not ((bounded["eci_ci_low"] <= bounded["eci"]) & (bounded["eci"] <= bounded["eci_ci_high"])).all():
        raise RuntimeError("Reproduced ECI intervals do not contain their central scores")
    return frame.set_index("Model", drop=False)


ECI_REPRODUCED = load_reproduced_eci_scores()


def load_eci_release_date_crosscheck():
    frame = pd.read_csv(ECI_REPRODUCTION_CROSSCHECK_PATH, keep_default_na=False)
    required = {
        "model",
        "model_version",
        "eci_input_date",
        "published_release_date",
        "regression_release_date",
        "release_date_policy",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"ECI release-date crosscheck is missing columns: {sorted(missing)}")
    if len(frame) != 213 or frame["model"].nunique() != 213 or frame["model_version"].nunique() != 213:
        raise RuntimeError("ECI release-date crosscheck is not one-to-one")
    if set(frame["model"]) != set(ECI_REPRODUCED.index):
        raise RuntimeError("ECI release-date crosscheck does not cover the reproduced score ledger")
    return frame.set_index("model", drop=False)


ECI_RELEASE_DATES = load_eci_release_date_crosscheck()


def hydrate_frontier_inputs():
    rows = []
    for release, model, aa_score, output_m, config, note in FRONTIER_INPUTS:
        eci_name = ECI_FRONTIER_ALIASES.get(model, model)
        if eci_name not in ECI_REPRODUCED.index:
            raise RuntimeError(f"Frontier model missing from reproduced ECI ledger: {model} -> {eci_name}")
        eci = ECI_REPRODUCED.loc[eci_name]
        release_audit = ECI_RELEASE_DATES.loc[eci_name]
        if str(release_audit["regression_release_date"])[:10] != release:
            raise RuntimeError(
                f"Frontier release-date mismatch for {model}: {release} vs {release_audit['regression_release_date']}"
            )
        if eci_name == "Claude Opus 5":
            exact_triplet = (
                float(eci["eci"]),
                float(eci["eci_ci_low"]),
                float(eci["eci_ci_high"]),
            )
            evidence_triplet = (
                float(OPUS5_EPOCH["eci_exact"]),
                float(OPUS5_EPOCH["eci_ci_low"]),
                float(OPUS5_EPOCH["eci_ci_high"]),
            )
            if release != OPUS5_IDENTITY["release_date"] or exact_triplet != evidence_triplet:
                raise RuntimeError("Canonical Opus 5 ECI row disagrees with its independent evidence bundle")
        rows.append((
            release,
            model,
            aa_score,
            output_m,
            config,
            float(eci["eci"]),
            float(eci["eci_ci_low"]),
            float(eci["eci_ci_high"]),
            note,
        ))
    return rows


FRONTIER = hydrate_frontier_inputs()


OPEN_URLS = {
    "Qwen3-235B-A22B-Instruct-2507": "https://artificialanalysis.ai/models/qwen3-235b-a22b-instruct-2507",
    "Qwen3-Coder-480B-A35B-Instruct": "https://artificialanalysis.ai/models/qwen3-coder-480b-a35b-instruct",
    "GLM-4.5-Air": "https://artificialanalysis.ai/models/glm-4-5-air",
    "GLM-4.5": "https://artificialanalysis.ai/models/glm-4-5",
    "Qwen3-30B-A3B-2507 Reasoning": "https://artificialanalysis.ai/models/qwen3-30b-a3b-2507-reasoning",
    "gpt-oss-20b (high)": "https://artificialanalysis.ai/models/gpt-oss-20b",
    "gpt-oss-120b (high)": "https://artificialanalysis.ai/models/gpt-oss-120b",
    "Seed-OSS-36B-Instruct": "https://artificialanalysis.ai/models/seed-oss-36b-instruct",
    "DeepSeek-V3.1 Reasoning": "https://artificialanalysis.ai/models/deepseek-v3-1-reasoning",
    "GLM-4.6 Reasoning": "https://artificialanalysis.ai/models/glm-4-6-reasoning",
    "MiniMax-M2": "https://artificialanalysis.ai/models/minimax-m2",
    "Kimi K2 Thinking": "https://artificialanalysis.ai/models/kimi-k2-thinking",
    "OLMo 3 32B Think": "https://artificialanalysis.ai/models/olmo-3-32b-think",
    "DeepSeek-V3.2 Reasoning": "https://artificialanalysis.ai/models/deepseek-v3-2-reasoning",
    "Mistral Large 3": "https://artificialanalysis.ai/models/mistral-large-3",
    "OLMo 3.1 32B Think": "https://artificialanalysis.ai/models/olmo-3-1-32b-think",
    "Nemotron 3 Nano 30B A3B Reasoning": "https://artificialanalysis.ai/models/nvidia-nemotron-3-nano-30b-a3b-reasoning",
    "MiMo-V2-Flash Reasoning": "https://artificialanalysis.ai/models/mimo-v2-flash-reasoning",
    "GLM-4.7 Reasoning": "https://artificialanalysis.ai/models/glm-4-7",
    "MiniMax-M2.1": "https://artificialanalysis.ai/models/minimax-m2-1",
    "K-EXAONE": "https://artificialanalysis.ai/models/k-exaone",
    "Kimi K2.5 Reasoning": "https://artificialanalysis.ai/models/kimi-k2-5",
    "LongCat Flash Lite": "https://artificialanalysis.ai/models/longcat-flash-lite",
    "GLM-5 Reasoning": "https://artificialanalysis.ai/models/glm-5",
    "MiniMax-M2.5": "https://artificialanalysis.ai/models/minimax-m2-5",
    "Qwen3.5-397B-A17B Reasoning": "https://artificialanalysis.ai/models/qwen3-5-397b-a17b",
    "Qwen3.5-35B-A3B Reasoning": "https://artificialanalysis.ai/models/qwen3-5-35b-a3b",
    "Qwen3.5-122B-A10B Reasoning": "https://artificialanalysis.ai/models/qwen3-5-122b-a10b",
    "Qwen3.5-27B Reasoning": "https://artificialanalysis.ai/models/qwen3-5-27b",
    "Qwen3.5-0.8B Reasoning": "https://artificialanalysis.ai/models/qwen3-5-0-8b",
    "Qwen3.5-2B Reasoning": "https://artificialanalysis.ai/models/qwen3-5-2b",
    "Qwen3.5-4B Reasoning": "https://artificialanalysis.ai/models/qwen3-5-4b",
    "Qwen3.5-9B Reasoning": "https://artificialanalysis.ai/models/qwen3-5-9b",
    "Nemotron 3 Super 120B A12B": "https://artificialanalysis.ai/models/nvidia-nemotron-3-super-120b-a12b",
    "MiniMax-M2.7": "https://artificialanalysis.ai/models/minimax-m2-7",
    "GLM-5.1 Reasoning": "https://artificialanalysis.ai/models/glm-5-1",
    "Qwen3.6-35B-A3B Reasoning": "https://artificialanalysis.ai/models/qwen3-6-35b-a3b",
    "Kimi K2.6 Reasoning": "https://artificialanalysis.ai/models/kimi-k2-6",
    "MiMo-V2.5": "https://artificialanalysis.ai/models/mimo-v2-5-0424",
    "Qwen3.6-27B Reasoning": "https://artificialanalysis.ai/models/qwen3-6-27b",
    "DeepSeek-V4-Pro (max)": "https://artificialanalysis.ai/models/deepseek-v4-pro",
    "MiniMax-M3": "https://artificialanalysis.ai/models/minimax-m3",
    "Nex-N2-Pro": "https://artificialanalysis.ai/models/nex-n2-pro",
    "Nemotron 3 Ultra 550B A55B": "https://artificialanalysis.ai/models/nvidia-nemotron-3-ultra-550b-a55b",
    "North Mini Code": "https://artificialanalysis.ai/models/north-mini-code",
    "Kimi K2.7 Code": "https://artificialanalysis.ai/models/kimi-k2-7-code",
    "GLM-5.2 (max)": "https://artificialanalysis.ai/models/glm-5-2",
    "LongCat 2.0": "https://artificialanalysis.ai/models/longcat-2-0",
    "Hy3": "https://artificialanalysis.ai/models/hy3",
    "Inkling (xhigh)": "https://artificialanalysis.ai/models/inkling",
}


PARAMETER_SOURCES = {
    "Qwen3-235B-A22B-Instruct-2507": "https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen3-Coder-480B-A35B-Instruct": "https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "GLM-4.5-Air": "https://huggingface.co/zai-org/GLM-4.5-Air-FP8",
    "GLM-4.5": "https://huggingface.co/zai-org/GLM-4.5-FP8",
    "Qwen3-30B-A3B-2507 Reasoning": "https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507",
    "gpt-oss-20b (high)": "https://openai.com/index/introducing-gpt-oss/",
    "gpt-oss-120b (high)": "https://openai.com/index/introducing-gpt-oss/",
    "Seed-OSS-36B-Instruct": "https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct",
    "DeepSeek-V3.1 Reasoning": "https://huggingface.co/deepseek-ai/DeepSeek-V3.1",
    "GLM-4.6 Reasoning": "https://huggingface.co/zai-org/GLM-4.6",
    "MiniMax-M2": "https://huggingface.co/MiniMaxAI/MiniMax-M2",
    "Kimi K2 Thinking": "https://huggingface.co/moonshotai/Kimi-K2-Thinking",
    "OLMo 3 32B Think": "https://allenai.org/blog/olmo3",
    "DeepSeek-V3.2 Reasoning": "https://huggingface.co/deepseek-ai/DeepSeek-V3.2",
    "Mistral Large 3": "https://mistral.ai/news/mistral-3/",
    "OLMo 3.1 32B Think": "https://huggingface.co/allenai/Olmo-3.1-32B-Think",
    "Nemotron 3 Nano 30B A3B Reasoning": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "MiMo-V2-Flash Reasoning": "https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash",
    "GLM-4.7 Reasoning": "https://huggingface.co/zai-org/GLM-4.7",
    "MiniMax-M2.1": "https://huggingface.co/MiniMaxAI/MiniMax-M2.1",
    "K-EXAONE": "https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B",
    "Kimi K2.5 Reasoning": "https://huggingface.co/moonshotai/Kimi-K2.5",
    "LongCat Flash Lite": "https://huggingface.co/meituan-longcat/LongCat-Flash-Lite",
    "GLM-5 Reasoning": "https://huggingface.co/zai-org/GLM-5",
    "MiniMax-M2.5": "https://huggingface.co/MiniMaxAI/MiniMax-M2.5",
    "Qwen3.5-397B-A17B Reasoning": "https://huggingface.co/Qwen/Qwen3.5-397B-A17B",
    "Qwen3.5-35B-A3B Reasoning": "https://huggingface.co/Qwen/Qwen3.5-35B-A3B",
    "Qwen3.5-122B-A10B Reasoning": "https://huggingface.co/Qwen/Qwen3.5-122B-A10B",
    "Qwen3.5-27B Reasoning": "https://huggingface.co/Qwen/Qwen3.5-27B",
    "Qwen3.5-0.8B Reasoning": "https://huggingface.co/Qwen/Qwen3.5-0.8B",
    "Qwen3.5-2B Reasoning": "https://huggingface.co/Qwen/Qwen3.5-2B",
    "Qwen3.5-4B Reasoning": "https://huggingface.co/Qwen/Qwen3.5-4B",
    "Qwen3.5-9B Reasoning": "https://huggingface.co/Qwen/Qwen3.5-9B",
    "Nemotron 3 Super 120B A12B": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
    "MiniMax-M2.7": "https://huggingface.co/MiniMaxAI/MiniMax-M2.7",
    "GLM-5.1 Reasoning": "https://huggingface.co/zai-org/GLM-5.1",
    "Qwen3.6-35B-A3B Reasoning": "https://huggingface.co/Qwen/Qwen3.6-35B-A3B",
    "Kimi K2.6 Reasoning": "https://huggingface.co/moonshotai/Kimi-K2.6",
    "MiMo-V2.5": "https://huggingface.co/XiaomiMiMo/MiMo-V2.5",
    "Qwen3.6-27B Reasoning": "https://huggingface.co/Qwen/Qwen3.6-27B",
    "DeepSeek-V4-Pro (max)": "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro",
    "MiniMax-M3": "https://huggingface.co/MiniMaxAI/MiniMax-M3",
    "Nex-N2-Pro": "https://huggingface.co/nex-agi/Nex-N2-Pro",
    "Nemotron 3 Ultra 550B A55B": "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "North Mini Code": "https://docs.cohere.com/changelog/north-mini-code-1-0",
    "Kimi K2.7 Code": "https://www.kimi.com/zh-cn/resources/kimi-k2-7-code",
    "GLM-5.2 (max)": "https://huggingface.co/zai-org/GLM-5.2",
    "LongCat 2.0": "https://huggingface.co/meituan-longcat/LongCat-2.0",
    "Hy3": "https://huggingface.co/tencent/Hy3",
    "Inkling (xhigh)": "https://thinkingmachines.ai/news/introducing-inkling/",
}


FRONTIER_URLS = {
    "Claude Opus 5": ("https://artificialanalysis.ai/models/claude-opus-5", "https://epoch.ai/data/benchmarked_models.csv"),
    "GPT-5.6 Sol": ("https://artificialanalysis.ai/models/gpt-5-6-sol", "https://epoch.ai/models/gpt-5-6-sol"),
    "Claude Fable 5": ("https://artificialanalysis.ai/models/claude-fable-5", "https://epoch.ai/models/claude-fable-5"),
    "Claude Opus 4.8": ("https://artificialanalysis.ai/models/claude-opus-4-8", "https://epoch.ai/models/claude-opus-4-8"),
    "GPT-5.6 Terra": ("https://artificialanalysis.ai/models/gpt-5-6-terra", "https://epoch.ai/models/gpt-5-6-terra"),
    "GPT-5.5": ("https://artificialanalysis.ai/models/gpt-5-5", "https://epoch.ai/models/gpt-5-5"),
    "Claude Opus 4.7": ("https://artificialanalysis.ai/models/claude-opus-4-7", "https://epoch.ai/models/claude-opus-4-7"),
    "Grok 4.5": ("https://artificialanalysis.ai/models/grok-4-5", "https://epoch.ai/models/grok-4-5"),
    "Claude Sonnet 5": ("https://artificialanalysis.ai/models/claude-sonnet-5", "https://epoch.ai/models/claude-sonnet-5"),
    "GPT-5.6 Luna": ("https://artificialanalysis.ai/models/gpt-5-6-luna", "https://epoch.ai/models/gpt-5-6-luna"),
    "Gemini 3.5 Flash": ("https://artificialanalysis.ai/models/gemini-3-5-flash", "https://epoch.ai/models/gemini-3-5-flash"),
    "Gemini 3.1 Pro Preview": ("https://artificialanalysis.ai/models/gemini-3-1-pro-preview", "https://epoch.ai/models/gemini-3-1-pro"),
}


# Extended open-weight ECI calibration panel. This deliberately goes beyond the
# requested 12-month display window so the linear vintage effect is identified
# from multiple model generations rather than size/date collinearity in one year.
ECI_PARAMETER_MAP = {
    "LLaMA-7B": (7, 7, "llama1", 0),
    "LLaMA-13B": (13, 13, "llama1", 0),
    "LLaMA-33B": (33, 33, "llama1", 0),
    "LLaMA-65B": (65, 65, "llama1", 0),
    "Falcon-40B": (40, 40, "falcon1", 0),
    "Cerebras-GPT-13B": (13, 13, "cerebras_gpt", 0),
    "Dolly 2.0-12b": (12, 12, "dolly", 0),
    "Falcon-7B": (7, 7, "falcon1", 0),
    "MPT-7B": (7, 7, "mpt", 0),
    "Baichuan1-7B": (7, 7, "baichuan", 0),
    "MPT-30B": (30, 30, "mpt", 0),
    "XGen-7B": (7, 7, "xgen", 0),
    "Llama 2-7B": (7, 7, "llama2", 0),
    "Llama 2-13B": (13, 13, "llama2", 0),
    "Llama 2-70B": (70, 70, "llama2", 0),
    "Stable Beluga 2": (70, 70, "llama2", 0),
    "Baichuan2-13B": (13, 13, "baichuan", 0),
    "Falcon-180B": (180, 180, "falcon1", 0),
    "Phi-1.5": (1.3, 1.3, "phi", 0),
    "Baichuan 2-7B": (7, 7, "baichuan", 0),
    "Mistral 7B v0.1": (7, 7, "mistral7", 0),
    "Qwen-7B": (7, 7, "qwen1", 0),
    "Qwen-14B": (14, 14, "qwen1", 0),
    "DeepSeek Coder 1.3B": (1.3, 1.3, "deepseek_coder", 0),
    "DeepSeek Coder 6.7B": (6.7, 6.7, "deepseek_coder", 0),
    "DeepSeek Coder 33B": (33, 33, "deepseek_coder", 0),
    "Yi 6B": (6, 6, "yi", 0),
    "Yi-34B": (34, 34, "yi", 0),
    "Mixtral 8x7B": (46.7, 12.9, "mixtral", 0),
    "Phi-2": (2.7, 2.7, "phi", 0),
    "StarCoder 2 7B": (7, 7, "starcoder2", 0),
    "StarCoder 2 15B": (15, 15, "starcoder2", 0),
    "Gemma 2B": (2, 2, "gemma", 0),
    "Gemma 7B": (7, 7, "gemma", 0),
    "StarCoder 2 3B": (3, 3, "starcoder2", 0),
    "Mixtral 8x22B": (141, 39, "mixtral", 0),
    "Llama 3-8B": (8, 8, "llama3", 0),
    "Llama 3-70B": (70, 70, "llama3", 0),
    "phi-3-mini 3.8B": (3.8, 3.8, "phi", 0),
    "phi-3-medium 14B": (14, 14, "phi", 0),
    "phi-3-small 7.4B": (7.4, 7.4, "phi", 0),
    "DeepSeek-V2 (MoE-236B, May 2024)": (236, 21, "deepseek_v2", 0),
    "Falcon 2 11B": (11, 11, "falcon2", 0),
    "Qwen2-72B": (72, 72, "qwen2", 0),
    "Gemma 2 9B": (9, 9, "gemma2", 0),
    "Gemma 2 27B": (27, 27, "gemma2", 0),
    "Mistral NeMo": (12, 12, "mistral_nemo", 0),
    "Llama 3.1-8B": (8, 8, "llama31", 0),
    "Llama 3.1-70B": (70, 70, "llama31", 0),
    "Llama 3.1-405B": (405, 405, "llama31", 0),
    "Mistral Large 2 (Jul 2024)": (123, 123, "mistral_large2", 0),
    "Qwen2.5-Coder (1.5B)": (1.5, 1.5, "qwen25_coder", 0),
    "Qwen2.5-Coder (7B)": (7, 7, "qwen25_coder", 0),
    "Qwen2.5-Coder-32B": (32, 32, "qwen25_coder", 0),
    "Qwen2.5-72B": (72, 72, "qwen25", 0),
    "Llama 3.2 90B": (90, 90, "llama32", 0),
    "Mistral Large 2 (Nov 2024)": (123, 123, "mistral_large2", 0),
    "Llama 3.3 70B": (70, 70, "llama33", 0),
    "Phi-4": (14, 14, "phi", 0),
    "DeepSeek-V3": (671, 37, "deepseek_v3", 0),
    "DeepSeek-R1": (671, 37, "deepseek_v3", 1),
    "Gemma 3 27B": (27, 27, "gemma3", 0),
    "Mistral Small 3.1": (24, 24, "mistral_small", 0),
    "DeepSeek-V3 (Mar 2025)": (671, 37, "deepseek_v3", 0),
    "Llama 4 Scout": (109, 17, "llama4", 0),
    "Llama 4 Maverick": (400, 17, "llama4", 0),
    "Qwen3-235B-A22B": (235, 22, "qwen3_235", 1),
    "DeepSeek-R1 (May 2025)": (671, 37, "deepseek_v3", 1),
    "Magistral Small 1.1": (24, 24, "mistral_small", 1),
    "Kimi K2 (Jul 2025)": (1000, 32, "kimi_k2", 0),
    "Qwen3-235B-A22B-Thinking (Jul 2025)": (235, 22, "qwen3_235", 1),
    "gpt-oss-120b": (116.83, 5.13, "gpt_oss", 1),
    "Qwen3-235B-A22B-Instruct (Jul 2025)": (235, 22, "qwen3_235", 0),
    "DeepSeek-V3.2-Exp": (671, 37, "deepseek_v3", 0),
    "GLM-4.6": (357, 32, "glm", 1),
    "Kimi K2 Thinking": (1000, 32, "kimi_k2", 1),
    "DeepSeek-V3.2": (671, 37, "deepseek_v3", 1),
    "GLM-4.7": (358, 32, "glm", 1),
    "Kimi K2.5": (1040, 32, "kimi_k2", 1),
    "GLM-5": (744, 40, "glm", 1),
    "MiniMax-M2.5": (229, 10, "minimax_m2", 1),
    "MiniMax-M2.7": (229, 10, "minimax_m2", 1),
    "Kimi K2.6": (1040, 32, "kimi_k2", 1),
    "DeepSeek-V4-Pro": (1600, 49, "deepseek_v4", 1),
    "Kimi K2.7 Code": (1040, 32, "kimi_k2", 1),
    "GLM-5.2": (744, 40, "glm", 1),
    # July 31 additions. K3 uses the exact primary-source 2.78T/104.2B
    # architecture facts rather than Epoch/AA's rounded 2.8T/104B display.
    # It shares a validation-family key with K2 so a held-out K3 prediction
    # cannot train on any other Moonshot/Kimi checkpoint.
    "Kimi K3": (2780, 104.2, "kimi_k2", 1),
    "Gemma 4 31B IT": (31, 31, "gemma4", 0),
    "Qwen 3.6 35B-A3B": (36, 3, "qwen36", 1),
}


def parse_date(s):
    y, m, d = map(int, s.split("-"))
    return date(y, m, d)


def date_years(s):
    return (parse_date(s) - DATE_ORIGIN).days / 365.25


def rows_to_records(rows):
    records = []
    for row in rows:
        (release, model, total_b, active_b, reasoning, score, estimated,
         output_m, architecture, family, coder, multimodal) = row
        aa_match = OPEN_AA_MATCH_BY_MODEL[model]
        records.append(apply_parameter_truth({
            "release_date": release,
            "model": model,
            "total_b": float(total_b),
            "active_b": float(active_b),
            "reasoning": int(reasoning),
            "score": float(score),
            "estimated": int(estimated),
            "output_m": output_m,
            "architecture": architecture,
            "family": family,
            "coder": int(coder),
            "multimodal": int(multimodal),
            "moe": int(total_b / active_b > 1.05),
            "date_years": date_years(release),
            "source": OPEN_URLS[model],
            "parameter_source": PARAMETER_SOURCES[model],
            "aa_slug": aa_match["aa_slug"],
            "aa_score_available_date": aa_match["aa_score_available_date"],
            "aa_score_availability_verified": aa_match[
                "aa_score_availability_verified"
            ],
            "aa_score_publication_event_id": aa_match[
                "aa_score_publication_event_id"
            ],
            "aa_score_at_publication": aa_match["aa_score_at_publication"],
        }))
    return records


def wls(X, y, weights):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    pred = X @ beta
    resid = y - pred
    sse = float(np.sum(w * resid ** 2))
    dof = max(1, len(y) - X.shape[1])
    sigma = math.sqrt(sse / dof)
    return beta, pred, resid, sse, sigma


def design(records, kappa, extras):
    X = []
    for r in records:
        ratio = r["total_b"] / r["active_b"]
        log2_eff = math.log2(r["active_b"]) + kappa * math.log2(ratio)
        row = [1.0, log2_eff, r["date_years"], r["reasoning"]]
        if "coder" in extras:
            row.append(r["coder"])
        if "multimodal" in extras:
            row.append(r["multimodal"])
        if "moe" in extras:
            row.append(r["moe"])
        X.append(row)
    return np.asarray(X, dtype=float)


def fit_grid(records, extras=(), estimated_weight=0.5, fixed_reasoning=None):
    y = np.array([r["score"] for r in records], dtype=float)
    weights = np.array([estimated_weight if r["estimated"] else 1.0 for r in records])
    best = None
    for kappa in np.linspace(0, 1, 101):
        X = design(records, float(kappa), extras)
        if fixed_reasoning is None:
            beta, pred, resid, sse, sigma = wls(X, y, weights)
        else:
            X_reduced = np.delete(X, 3, axis=1)
            y_reduced = y - fixed_reasoning * X[:, 3]
            beta_reduced, _, _, _, _ = wls(X_reduced, y_reduced, weights)
            beta = np.insert(beta_reduced, 3, fixed_reasoning)
            pred = X @ beta
            resid = y - pred
            sse = float(np.sum(weights * resid ** 2))
            sigma = math.sqrt(sse / max(1, len(y) - X_reduced.shape[1]))
        if beta[1] <= 0:
            continue
        candidate = (sse, float(kappa), beta, pred, resid, sigma)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("No positive scale coefficient")
    sse, kappa, beta, pred, resid, sigma = best
    return {
        "kappa": kappa,
        "beta": beta,
        "pred": pred,
        "resid": resid,
        "sse": sse,
        "sigma": sigma,
        "extras": tuple(extras),
        "weights": weights,
        "fixed_reasoning": fixed_reasoning,
    }


def predict_score(r, fit):
    return float(design([r], fit["kappa"], fit["extras"])[0] @ fit["beta"])


def implied_eff(score, release, fit, reasoning=1, coder=0, multimodal=0, moe=1, frontier_shift=0.0):
    b = fit["beta"]
    values = [1.0, None, date_years(release), reasoning]
    if "coder" in fit["extras"]:
        values.append(coder)
    if "multimodal" in fit["extras"]:
        values.append(multimodal)
    if "moe" in fit["extras"]:
        values.append(moe)
    offset = sum(b[i] * values[i] for i in range(len(values)) if i != 1)
    return 2 ** ((score - frontier_shift - offset) / b[1])


def weighted_quantile(values, quantile, weights=None):
    values = np.asarray(values, dtype=float)
    if weights is None:
        return float(np.quantile(values, quantile))
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return float(np.interp(quantile, cumulative, values))


def lofo_metrics(records, extras, fixed_reasoning=None):
    errors = []
    scale_errors = []
    by_family = {}
    for family in sorted(set(r["family"] for r in records)):
        train = [r for r in records if r["family"] != family]
        test = [r for r in records if r["family"] == family]
        if len(train) < 10:
            continue
        fit = fit_grid(train, extras, fixed_reasoning=fixed_reasoning)
        for r in test:
            pred = predict_score(r, fit)
            err = r["score"] - pred
            errors.append(err)
            implied = implied_eff(
                r["score"], r["release_date"], fit,
                r["reasoning"], r["coder"], r["multimodal"], r["moe"],
            )
            true_ratio = r["total_b"] / r["active_b"]
            true_eff = r["active_b"] * true_ratio ** fit["kappa"]
            le = math.log2(implied / true_eff)
            scale_errors.append(le)
            by_family.setdefault(family, []).append((err, le))
    return {
        "rmse_score": float(math.sqrt(np.mean(np.square(errors)))),
        "mae_score": float(np.mean(np.abs(errors))),
        "mae_log2_scale": float(np.mean(np.abs(scale_errors))),
        "median_abs_log2_scale": float(np.median(np.abs(scale_errors))),
        "p80_abs_log2_scale": float(np.quantile(np.abs(scale_errors), 0.8)),
        "p90_abs_log2_scale": float(np.quantile(np.abs(scale_errors), 0.9)),
        "bias_log2_scale": float(np.mean(scale_errors)),
        "n": len(errors),
    }


def bootstrap_predictions(records, extras, frontier, draws=2500, seed=20260716,
                          score_sd_by_model=None, frontier_quantile=0.90,
                          add_structural_error=False, fixed_reasoning=None):
    rng = np.random.default_rng(seed)
    families = sorted(set(r["family"] for r in records))
    grouped = {f: [r for r in records if r["family"] == f] for f in families}
    results = {t[1]: [] for t in frontier}
    coefficient_draws = []
    attempts = 0
    while len(coefficient_draws) < draws and attempts < draws * 5:
        attempts += 1
        sampled_families = rng.choice(families, size=len(families), replace=True)
        sampled = []
        for j, f in enumerate(sampled_families):
            for r in grouped[f]:
                rr = dict(r)
                rr["family"] = f"{f}__{j}"
                sampled.append(rr)
        try:
            fit = fit_grid(sampled, extras, fixed_reasoning=fixed_reasoning)
        except Exception:
            continue
        if fit["beta"][1] < 0.6:
            continue
        coefficient_draws.append({
            "kappa": fit["kappa"],
            "beta": fit["beta"].tolist(),
            "sigma": fit["sigma"],
        })
        frontier_shift = weighted_quantile(
            fit["resid"], frontier_quantile, fit["weights"]
        )
        for release, model, score, *_ in frontier:
            sd = 0.6 if score_sd_by_model is None else score_sd_by_model.get(model, 0.6)
            score_draw = score + rng.normal(0, sd)
            eff = implied_eff(
                score_draw, release, fit, reasoning=1,
                frontier_shift=frontier_shift,
            )
            if add_structural_error:
                eff *= 2 ** rng.normal(0, fit["sigma"] / max(fit["beta"][1], 0.6))
            results[model].append(float(eff))
    return results, coefficient_draws


def quantiles(xs):
    a = np.asarray(xs, dtype=float)
    return {
        "p025": float(np.quantile(a, 0.025)),
        "p10": float(np.quantile(a, 0.10)),
        "median": float(np.quantile(a, 0.50)),
        "p90": float(np.quantile(a, 0.90)),
        "p975": float(np.quantile(a, 0.975)),
    }


def load_eci_records():
    records = []
    for model, (total_b, active_b, family, reasoning) in ECI_PARAMETER_MAP.items():
        if model not in ECI_REPRODUCED.index:
            continue
        row = ECI_REPRODUCED.loc[model]
        release_audit = ECI_RELEASE_DATES.loc[model]
        release_date = str(release_audit["regression_release_date"])[:10]
        low = float(row["eci_ci_low"])
        high = float(row["eci_ci_high"])
        ci_width = high - low
        broad_eci_ci = int(ci_width > 10)
        records.append(apply_parameter_truth({
            "release_date": release_date,
            "model": model,
            "total_b": float(total_b),
            "active_b": float(active_b),
            "reasoning": int(reasoning),
            "score": float(row["eci"]),
            # This is score uncertainty, not parameter-disclosure metadata.  Keep
            # the legacy key for reproducibility while exposing an unambiguous one.
            "estimated": broad_eci_ci,
            "broad_eci_ci": broad_eci_ci,
            "parameter_disclosure_status": "not classified",
            "output_m": None,
            "architecture": "MoE" if total_b / active_b > 1.05 else "dense",
            "family": family,
            "coder": int("Coder" in model or "Code" in model),
            "multimodal": int("Vision" in model or model == "Llama 3.2 90B"),
            "moe": int(total_b / active_b > 1.05),
            "date_years": date_years(release_date),
            "source": "https://epoch.ai/data/eci_benchmarks.csv",
            "eci_input_date": str(release_audit["eci_input_date"])[:10],
            "release_date_source": str(release_audit["release_date_policy"]),
            "published_release_date": str(release_audit["published_release_date"]),
            "ci_low": low,
            "ci_high": high,
            "ci_width": ci_width,
        }))
    expected = set(ECI_PARAMETER_MAP)
    found = set(r["model"] for r in records)
    if found != expected:
        raise RuntimeError(f"ECI reproduction coverage mismatch; missing {sorted(expected - found)}")
    return records


def main():
    records = rows_to_records(OPEN_MODELS)
    fixed_reasoning_effect = 6.0
    candidate_specs = [(), ("coder",), ("multimodal",), ("coder", "multimodal"), ("moe",)]
    comparisons = []
    for spec in candidate_specs:
        fit = fit_grid(records, spec, fixed_reasoning=fixed_reasoning_effect)
        cv = lofo_metrics(records, spec, fixed_reasoning=fixed_reasoning_effect)
        comparisons.append({
            "spec": list(spec),
            "kappa": fit["kappa"],
            "beta": fit["beta"].tolist(),
            "sigma_score": fit["sigma"],
            **cv,
        })

    # Choose the best leave-family-out median parameter recovery metric; ties favor simplicity.
    comparisons.sort(key=lambda x: (round(x["median_abs_log2_scale"], 3), len(x["spec"])))
    selected_spec = tuple(comparisons[0]["spec"])
    fit = fit_grid(records, selected_spec, fixed_reasoning=fixed_reasoning_effect)
    fit["frontier_shift"] = weighted_quantile(fit["resid"], 0.90, fit["weights"])
    bootstrap, coefficient_draws = bootstrap_predictions(
        records, selected_spec, FRONTIER, frontier_quantile=0.90,
        fixed_reasoning=fixed_reasoning_effect,
    )

    eci_records = load_eci_records()
    eci_candidate_specs = [(), ("coder",), ("moe",)]
    eci_comparisons = []
    for spec in eci_candidate_specs:
        eci_fit_candidate = fit_grid(
            eci_records, spec, fixed_reasoning=fixed_reasoning_effect
        )
        cv = lofo_metrics(
            eci_records, spec, fixed_reasoning=fixed_reasoning_effect
        )
        eci_comparisons.append({
            "spec": list(spec),
            "kappa": eci_fit_candidate["kappa"],
            "beta": eci_fit_candidate["beta"].tolist(),
            "sigma_score": eci_fit_candidate["sigma"],
            **cv,
        })
    eci_comparisons.sort(key=lambda x: (round(x["median_abs_log2_scale"], 3), len(x["spec"])))
    eci_selected_spec = tuple(eci_comparisons[0]["spec"])
    eci_fit = fit_grid(
        eci_records, eci_selected_spec, fixed_reasoning=fixed_reasoning_effect
    )
    eci_fit["frontier_shift"] = weighted_quantile(
        eci_fit["resid"], 0.90, eci_fit["weights"]
    )
    eci_frontier = [(r[0], r[1], r[5]) for r in FRONTIER]
    eci_sd = {
        r[1]: (r[7] - r[6]) / (2 * 1.645)
        for r in FRONTIER
    }
    eci_bootstrap, eci_coefficient_draws = bootstrap_predictions(
        eci_records,
        eci_selected_spec,
        eci_frontier,
        seed=20260717,
        score_sd_by_model=eci_sd,
        frontier_quantile=0.90,
        fixed_reasoning=fixed_reasoning_effect,
    )

    structural_log2_halfwidth = max(
        comparisons[0]["p80_abs_log2_scale"],
        eci_comparisons[0]["p80_abs_log2_scale"],
    )

    ratios = np.array([r["total_b"] / r["active_b"] for r in records if r["moe"]], dtype=float)
    observed_ratio_median = float(np.median(ratios))

    predictions = []
    for row in FRONTIER:
        release, model, score, output_m, config, eci, eci_low, eci_high, note = row
        aa_point = implied_eff(
            score, release, fit, reasoning=1,
            frontier_shift=fit["frontier_shift"],
        )
        eci_point = implied_eff(
            eci, release, eci_fit, reasoning=1,
            frontier_shift=eci_fit["frontier_shift"],
        )
        ensemble_point = math.sqrt(aa_point * eci_point)
        r = observed_ratio_median
        aa_active = aa_point / (r ** fit["kappa"])
        aa_total = aa_point * (r ** (1 - fit["kappa"]))
        eci_active = eci_point / (r ** eci_fit["kappa"])
        eci_total = eci_point * (r ** (1 - eci_fit["kappa"]))
        active_scenario = math.sqrt(aa_active * eci_active)
        total_scenario = math.sqrt(aa_total * eci_total)
        ensemble_dense_draws = [
            math.sqrt(a * e)
            for a, e in zip(bootstrap[model], eci_bootstrap[model])
        ]
        ensemble_active_draws = []
        ensemble_total_draws = []
        for i, (a, e) in enumerate(zip(bootstrap[model], eci_bootstrap[model])):
            ka = coefficient_draws[i]["kappa"]
            ke = eci_coefficient_draws[i]["kappa"]
            ensemble_active_draws.append(math.sqrt(
                (a / (r ** ka)) * (e / (r ** ke))
            ))
            ensemble_total_draws.append(math.sqrt(
                (a * (r ** (1 - ka))) * (e * (r ** (1 - ke)))
            ))
        aa_url, epoch_url = FRONTIER_URLS[model]
        predictions.append({
            "release_date": release,
            "model": model,
            "aa_score": score,
            "aa_output_m": output_m,
            "config": config,
            "eci": eci,
            "eci_low": eci_low,
            "eci_high": eci_high,
            "classification": note,
            "aa_dense_equiv_b": aa_point,
            "eci_dense_equiv_b": eci_point,
            "dense_equiv_b": ensemble_point,
            "dense_equiv_interval": quantiles(ensemble_dense_draws),
            "moe_ratio_scenario": r,
            "moe_active_b": active_scenario,
            "moe_total_b": total_scenario,
            "moe_active_interval": quantiles(ensemble_active_draws),
            "moe_total_interval": quantiles(ensemble_total_draws),
            "structural_log2_halfwidth": structural_log2_halfwidth,
            "structural_active_low": active_scenario / (2 ** structural_log2_halfwidth),
            "structural_active_high": active_scenario * (2 ** structural_log2_halfwidth),
            "structural_total_low": total_scenario / (2 ** structural_log2_halfwidth),
            "structural_total_high": total_scenario * (2 ** structural_log2_halfwidth),
            "aa_source": aa_url,
            "epoch_source": epoch_url,
        })

    for i, r in enumerate(records):
        ratio = r["total_b"] / r["active_b"]
        r["effective_b"] = r["active_b"] * ratio ** fit["kappa"]
        r["fitted_score"] = float(fit["pred"][i])
        r["residual"] = float(fit["resid"][i])
        r["asof_adjusted_score"] = r["score"] + fit["beta"][2] * (
            date_years(AS_OF.isoformat()) - r["date_years"]
        )

    for i, r in enumerate(eci_records):
        ratio = r["total_b"] / r["active_b"]
        r["effective_b"] = r["active_b"] * ratio ** eci_fit["kappa"]
        r["fitted_score"] = float(eci_fit["pred"][i])
        r["residual"] = float(eci_fit["resid"][i])

    result = {
        "as_of": AS_OF.isoformat(),
        "date_origin": DATE_ORIGIN.isoformat(),
        "n_open_models": len(records),
        "n_families": len(set(r["family"] for r in records)),
        "selected_spec": list(selected_spec),
        "fit": {
            "kappa": fit["kappa"],
            "beta": fit["beta"].tolist(),
            "sigma_score": fit["sigma"],
            "r2_weighted": 1 - fit["sse"] / float(np.sum(fit["weights"] * (
                np.array([r["score"] for r in records]) - np.average(
                    np.array([r["score"] for r in records]), weights=fit["weights"]
                )
            ) ** 2)),
            "parameter_doublings_per_year": float(fit["beta"][2] / fit["beta"][1]),
            "algorithmic_doubling_months": float(12 * fit["beta"][1] / fit["beta"][2]),
            "observed_moe_ratio_median": observed_ratio_median,
            "frontier_residual_shift_p90": fit["frontier_shift"],
        },
        "model_comparisons": comparisons,
        "open_models": records,
        "eci": {
            "n_open_models": len(eci_records),
            "n_families": len(set(r["family"] for r in eci_records)),
            "selected_spec": list(eci_selected_spec),
            "fit": {
                "kappa": eci_fit["kappa"],
                "beta": eci_fit["beta"].tolist(),
                "sigma_score": eci_fit["sigma"],
                "frontier_residual_shift_p90": eci_fit["frontier_shift"],
                "parameter_doublings_per_year": float(eci_fit["beta"][2] / eci_fit["beta"][1]),
                "algorithmic_doubling_months": float(12 * eci_fit["beta"][1] / eci_fit["beta"][2]),
            },
            "model_comparisons": eci_comparisons,
            "open_models": eci_records,
        },
        "frontier_predictions": predictions,
        "bootstrap_draws": len(coefficient_draws),
        "structural_log2_halfwidth": structural_log2_halfwidth,
        "fixed_reasoning_effect": fixed_reasoning_effect,
        "aa_exact_input_audit": {
            "path": str(AA_DETAILED_PATH.relative_to(ROOT)),
            "sha256": hashlib.sha256(AA_DETAILED_PATH.read_bytes()).hexdigest(),
            "snapshot_models": len(AA_DETAILED_BY_SLUG),
            "calibration_matches": len(OPEN_AA_MATCH_AUDIT),
            "frontier_matches": len(FRONTIER_AA_MATCH_AUDIT),
            "duplicate_policy": "normalized exact checkpoint label; select highest Intelligence Index configuration; three explicit slug adjudications",
            "calibration_match_audit": OPEN_AA_MATCH_AUDIT,
            "frontier_match_audit": FRONTIER_AA_MATCH_AUDIT,
            "score_availability_timing": {
                "ledger_path": str(AA_SCORE_AVAILABILITY_PATH.relative_to(ROOT)),
                "ledger_sha256": hashlib.sha256(
                    AA_SCORE_AVAILABILITY_PATH.read_bytes()
                ).hexdigest(),
                "raw_changelog_path": str(AA_SCORE_CHANGELOG_PATH.relative_to(ROOT)),
                "raw_changelog_sha256": hashlib.sha256(
                    AA_SCORE_CHANGELOG_PATH.read_bytes()
                ).hexdigest(),
                "policy": "Use max(model release, verified non-null modelAdded score date) as the AA prediction-information date; unmatched rows fall back to release and remain explicitly unverified.",
            },
        },
        "open_model_parameter_truth_reconciliation": {
            "path": str(OPEN_MODEL_PARAMETER_TRUTH_PATH.relative_to(ROOT)),
            "sha256": hashlib.sha256(
                OPEN_MODEL_PARAMETER_TRUTH_PATH.read_bytes()
            ).hexdigest(),
            "corrected_aa_rows": sum(
                bool(row.get("parameter_truth_id")) for row in records
            ),
            "corrected_eci_rows": sum(
                bool(row.get("parameter_truth_id")) for row in eci_records
            ),
            "raw_values_preserved": True,
            "checkpoint_identity_changed": False,
            "global_parameter_match_tolerance_changed": False,
        },
        "eci_reproduction": {
            "snapshot_manifest_path": str(ECI_SNAPSHOT_MANIFEST_PATH.relative_to(ROOT)),
            "snapshot_manifest_sha256": hashlib.sha256(ECI_SNAPSHOT_MANIFEST_PATH.read_bytes()).hexdigest(),
            "scores_path": str(ECI_REPRODUCTION_PATH.relative_to(ROOT)),
            "scores_sha256": hashlib.sha256(ECI_REPRODUCTION_PATH.read_bytes()).hexdigest(),
            "metadata_path": str(ECI_REPRODUCTION_METADATA_PATH.relative_to(ROOT)),
            "metadata_sha256": hashlib.sha256(ECI_REPRODUCTION_METADATA_PATH.read_bytes()).hexdigest(),
            "release_date_crosscheck_path": str(ECI_REPRODUCTION_CROSSCHECK_PATH.relative_to(ROOT)),
            "release_date_crosscheck_sha256": hashlib.sha256(ECI_REPRODUCTION_CROSSCHECK_PATH.read_bytes()).hexdigest(),
            "audit_path": str(ECI_REPRODUCTION_AUDIT_PATH.relative_to(ROOT)),
            "audit_sha256": hashlib.sha256(ECI_REPRODUCTION_AUDIT_PATH.read_bytes()).hexdigest(),
            "source_commit": json.loads(ECI_REPRODUCTION_METADATA_PATH.read_text())["source_commit"],
            "models": len(ECI_REPRODUCED),
            "frontier_inputs_hydrated": len(FRONTIER),
            "opus5_identity_and_canonical_eci_crosscheck": {
                "model": "Claude Opus 5",
                "path": str(OPUS5_EVIDENCE_PATH.relative_to(ROOT)),
                "sha256": hashlib.sha256(OPUS5_EVIDENCE_PATH.read_bytes()).hexdigest(),
                "eci_exact": float(OPUS5_EPOCH["eci_exact"]),
                "eci_ci_low": float(OPUS5_EPOCH["eci_ci_low"]),
                "eci_ci_high": float(OPUS5_EPOCH["eci_ci_high"]),
                "parameter_disclosed": False,
                "base_identity_policy": "unique_base",
            },
            "k3_primary_architecture_crosscheck": {
                "path": str(K3_EVIDENCE_PATH.relative_to(ROOT)),
                "sha256": hashlib.sha256(K3_EVIDENCE_PATH.read_bytes()).hexdigest(),
                "total_parameters_b": float(K3_ARCHITECTURE["total_parameters_b_exact"]),
                "active_parameters_b": float(K3_ARCHITECTURE["activated_parameters_b_exact"]),
                "paired_architecture_fact_not_independent_vote": True,
            },
            "published_release_dates_used": int((ECI_RELEASE_DATES["release_date_policy"] == "published capabilities release date").sum()),
            "canonical_input_date_fallbacks": int((ECI_RELEASE_DATES["release_date_policy"] == "canonical ECI-input date fallback").sum()),
            "network_reads": 0,
        },
        "bootstrap_coefficient_summary": {
            "kappa": quantiles([d["kappa"] for d in coefficient_draws]),
            "scale_beta": quantiles([d["beta"][1] for d in coefficient_draws]),
            "date_beta": quantiles([d["beta"][2] for d in coefficient_draws]),
        },
        "eci_bootstrap_coefficient_summary": {
            "kappa": quantiles([d["kappa"] for d in eci_coefficient_draws]),
            "scale_beta": quantiles([d["beta"][1] for d in eci_coefficient_draws]),
            "date_beta": quantiles([d["beta"][2] for d in eci_coefficient_draws]),
        },
    }
    with open("regression_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({
        "selected_spec": selected_spec,
        "fit": result["fit"],
        "cv": comparisons[0],
        "predictions": [{
            "model": p["model"],
            "score": p["aa_score"],
            "dense_equiv_b": round(p["dense_equiv_b"], 1),
            "p10": round(p["dense_equiv_interval"]["p10"], 1),
            "p90": round(p["dense_equiv_interval"]["p90"], 1),
            "moe_total_b": round(p["moe_total_b"], 1),
            "moe_total_p10": round(p["moe_total_interval"]["p10"], 1),
            "moe_total_p90": round(p["moe_total_interval"]["p90"], 1),
            "moe_active_b": round(p["moe_active_b"], 1),
        } for p in predictions],
    }, indent=2))


if __name__ == "__main__":
    main()
