import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(fileURLToPath(import.meta.url));
export const k3EvidencePath = path.join(
  root,
  "sources/kimi_k3_release_evidence_2026-07-31.json",
);
export const k3ParameterSource = "Kimi K3 official technical report Table 1";
export const k3Evidence = JSON.parse(await fs.readFile(k3EvidencePath, "utf8"));
export const k3Architecture = k3Evidence.kimi_k3;
export const k3TotalB = Number(k3Architecture.total_parameters_b_exact);
export const k3TotalT = k3TotalB / 1000;
export const k3ActiveB = Number(k3Architecture.activated_parameters_b_exact);
export const k3TotalTDisplay = Number(k3Architecture.total_parameters_t_display);

if (
  k3TotalB !== 2780
  || k3ActiveB !== 104.2
  || k3Architecture.parameter_count_disclosed !== true
  || k3Architecture.activated_parameter_count_disclosed !== true
) {
  throw new Error(
    "Kimi K3 primary evidence is missing the exact disclosed 2.78T total / 104.2B activated counts",
  );
}
