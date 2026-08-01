from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
AUDIT = OUT / "epoch_eci_reproduction_audit_2026-07-31.json"
CROSSCHECK = OUT / "epoch_eci_reproduction_crosscheck_2026-07-31.csv"
MANIFEST = ROOT / "sources/epoch_snapshot_manifest_2026-07-31.json"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


data = json.loads(AUDIT.read_text(encoding="utf-8"))
crosscheck = rows(CROSSCHECK)

reproduction = data["reproduction"]
assert data["snapshot_as_of"] == "2026-07-31"
assert reproduction["input_rows"] == 2059
assert reproduction["input_models"] == reproduction["reproduced_models"] == len(crosscheck) == 213
assert reproduction["input_benchmarks"] == 54
assert reproduction["all_models_exactly_covered"] is True
assert reproduction["anchors"] == {"Claude 3.5 Sonnet": 130.0, "GPT-5": 150.0}
assert len({row["model"] for row in crosscheck}) == 213
assert len({row["model_version"] for row in crosscheck}) == 213

scores = data["published_score_crosscheck"]
assert scores["published_nonblank_scores"] == 196
assert scores["published_blank_scores"] == 17
assert scores["maximum_absolute_score_difference"] <= scores["tolerance"]
assert scores["all_nonblank_scores_match_within_display_rounding"] is True

dates = data["release_date_crosscheck"]
assert dates["published_release_dates_used"] == 211
assert dates["canonical_input_date_fallbacks"] == 2
assert dates["published_vs_input_date_disagreements"] == 43
assert set(dates["known_fallback_models"]) == {"Claude Instant", "Gemma 2 9B"}

for row in crosscheck:
    if row["published_release_date"]:
        assert row["regression_release_date"] == row["published_release_date"]
        assert row["release_date_policy"] == "published capabilities release date"
    else:
        assert row["regression_release_date"] == row["eci_input_date"]
        assert row["release_date_policy"] == "canonical ECI-input date fallback"

by_model = {row["model"]: row for row in crosscheck}
assert "Kimi K2 (Sep 2025)" not in by_model
assert by_model["Kimi K3"]["eci_input_date"] == "2026-07-16"
assert by_model["Kimi K3"]["regression_release_date"] == "2026-07-16"
assert abs(float(by_model["Kimi K3"]["eci_score_reproduced"]) - 155.5939255295791) < 1e-12
assert abs(float(by_model["Claude Opus 5"]["eci_score_reproduced"]) - 159.3778667882398) < 1e-12
assert by_model["Mistral Large 2 (Nov 2024)"]["eci_input_date"] == "2024-07-24"
assert by_model["Mistral Large 2 (Nov 2024)"]["regression_release_date"] == "2024-11-18"

for relative, digest in data["source_hashes"].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
assert manifest["inventory"]["duplicate_model_benchmark_name_pairs"] == 0
assert manifest["inventory"]["identity_join_policy"].startswith("within_snapshot_only")
assert manifest["fit_inventory"]["eci_rows"] == 213
assert manifest["fit_inventory"]["edi_rows"] == 54

print(
    {
        "reproduced_models": len(crosscheck),
        "published_release_dates": dates["published_release_dates_used"],
        "preserved_date_disagreements": dates["published_vs_input_date_disagreements"],
        "status": "PASS",
    }
)
