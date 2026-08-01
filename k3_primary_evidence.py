#!/usr/bin/env python3
"""Single audited access point for Kimi K3's disclosed architecture counts.

The launch page rounded K3 to ``2.8T`` and ``104B``.  The subsequently released
technical report gives the exact Table 1 values used by analytical code:
2.78 trillion total parameters and 104.2 billion activated parameters.  Keeping
the display-rounded value separate prevents it from leaking into calibration or
backtest arithmetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
K3_EVIDENCE_PATH = ROOT / "sources/kimi_k3_release_evidence_2026-07-31.json"
K3_PARAMETER_SOURCE = "Kimi K3 official technical report Table 1"


def load_k3_primary_evidence() -> dict[str, Any]:
    record = json.loads(K3_EVIDENCE_PATH.read_text(encoding="utf-8"))
    k3 = record.get("kimi_k3", {})
    if (
        float(k3.get("total_parameters_b_exact", 0)) != 2780.0
        or float(k3.get("activated_parameters_b_exact", 0)) != 104.2
        or k3.get("parameter_count_disclosed") is not True
        or k3.get("activated_parameter_count_disclosed") is not True
    ):
        raise RuntimeError(
            "Kimi K3 primary evidence is missing the exact disclosed "
            "2.78T total / 104.2B activated counts"
        )
    return record


K3_PRIMARY_EVIDENCE = load_k3_primary_evidence()
K3_ARCHITECTURE = K3_PRIMARY_EVIDENCE["kimi_k3"]
K3_TOTAL_B = float(K3_ARCHITECTURE["total_parameters_b_exact"])
K3_TOTAL_T = K3_TOTAL_B / 1000.0
K3_ACTIVE_B = float(K3_ARCHITECTURE["activated_parameters_b_exact"])
K3_TOTAL_T_DISPLAY = float(K3_ARCHITECTURE["total_parameters_t_display"])

