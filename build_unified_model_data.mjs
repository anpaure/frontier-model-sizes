import fs from "node:fs/promises";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workDir = path.dirname(fileURLToPath(import.meta.url));
const threadId = "019f6c42-2d53-7743-ab07-6293e2618dd7";
const outputDir = `${workDir}/outputs/${threadId}`;
const sourceDir = `${workDir}/sources`;
const epochPath = `${sourceDir}/epoch_all_ai_models_2026-07-31.csv`;
const eciPath = `${sourceDir}/input_eci_parameter_regression_workbook_2026-07-17.xlsx`;
const eciComponentPath = `${sourceDir}/epoch_eci_benchmarks_2026-07-31.csv`;
const eciReproducedPath = `${sourceDir}/epoch_eci_reproduced_scores_2026-07-31.csv`;
const eciReleaseCrosscheckPath = `${outputDir}/epoch_eci_reproduction_crosscheck_2026-07-31.csv`;
const aaPath = `${sourceDir}/input_artificial_analysis_leaderboard_2026-07-17.txt`;
const aaDetailedPath = `${sourceDir}/aa_detailed_model_signals_2026-07-31.csv`;
const latexPath = `${sourceDir}/input_no_cot_arxiv_2606.07157v3_source.tar.gz`;
const metrPath = `${sourceDir}/metr_horizon_official_signals_2026-07-18.csv`;
const metrRawPath = `${sourceDir}/metr_benchmark_results_1_1_2026-07-18.yaml`;
const metrMetadataPath = `${sourceDir}/metr_horizon_official_metadata_2026-07-18.json`;
const metrAuditPath = `${outputDir}/metr_primary_source_audit_2026-07-18.json`;
const metrLegacyPath = `${sourceDir}/metr_horizon_user_snapshot_2026-07-17.csv`;
const metrOfficialUrl = "https://metr.org/assets/benchmark_results_1_1.yaml";
const epochArchivePath = `${sourceDir}/input_epoch_ai_models_archive_2026-07-17.zip`;
const noCotDateOverridePath = `${sourceDir}/no_cot_exact_date_overrides_2026-07-18.csv`;
const noCotDateMetadataPath = `${sourceDir}/no_cot_exact_date_collection_metadata_2026-07-18.json`;
const qwenDateRawPath = `${sourceDir}/qwen3_30b_a3b_instruct_2507_hf_commits_2026-07-18.json.gz`;
const frontierPrimaryEvidencePath = `${sourceDir}/frontier_primary_evidence_2026-07-18.csv`;
const frontierPrimaryMetadataPath = `${sourceDir}/frontier_primary_evidence_collection_metadata_2026-07-18.json`;
const openAiGpt56SystemCardPath = `${sourceDir}/openai_gpt_5_6_system_card_2026-07-18.html.gz`;
const anthropicFableMythosClaimsPath = `${sourceDir}/anthropic_fable_mythos_primary_claims_2026-07-18.json`;
const k3ReleaseEvidencePath = `${sourceDir}/kimi_k3_release_evidence_2026-07-31.json`;
const openModelParameterTruthPath = `${sourceDir}/open_model_parameter_truth_reconciliation_2026-07-31.json`;

const observationsPath = `${outputDir}/unified_model_observations_compute_enriched_2026-07-17.csv`;
const measurementsPath = `${outputDir}/unified_model_measurements_long_compute_enriched_2026-07-17.csv`;
const aaAuditPath = `${outputDir}/aa_epoch_match_audit_compute_enriched_2026-07-17.csv`;
const epochViewAuditPath = `${outputDir}/epoch_archive_view_match_audit_2026-07-17.csv`;
const manifestPath = `${outputDir}/unified_model_source_manifest_compute_enriched_2026-07-17.csv`;

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(sourceDir, { recursive: true });

const csvEscape = (value) => {
  if (value == null) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};

const toCsv = (headers, rows) => [
  headers.map(csvEscape).join(","),
  ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(",")),
].join("\n") + "\n";

const writeCsv = async (path, headers, rows) => fs.writeFile(path, toCsv(headers, rows), "utf8");

const portablePath = (value) => {
  const absolute = path.resolve(value);
  const relative = path.relative(workDir, absolute);
  if (relative && relative !== ".." && !relative.startsWith(`..${path.sep}`)) {
    return relative.split(path.sep).join("/");
  }
  return absolute;
};

const parseCsv = (text) => {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          cell += '"';
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else if (ch !== "\r") {
      cell += ch;
    }
  }
  if (quoted) throw new Error("Unclosed quoted field in CSV");
  if (cell.length || row.length) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
};

const sha256 = async (path) => createHash("sha256").update(await fs.readFile(path)).digest("hex");
const sha256Text = (text) => createHash("sha256").update(text).digest("hex");
const finite = (value) => value !== "" && value != null && Number.isFinite(Number(value)) ? Number(value) : null;
const isoFromExcel = (value) => {
  if (value instanceof Date) return value.toISOString().slice(0, 10);
  if (typeof value === "number") return new Date((value - 25569) * 86400000).toISOString().slice(0, 10);
  if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}/.test(value)) return value.slice(0, 10);
  return "";
};
const dayDelta = (a, b) => a && b ? Math.round((Date.parse(a) - Date.parse(b)) / 86400000) : "";
const slug = (value) => String(value || "unknown").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
const alnumKey = (value) => String(value || "").toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "");
const openModelParameterTruth = JSON.parse(await fs.readFile(openModelParameterTruthPath, "utf8"));
if (openModelParameterTruth.schema_version !== "1.0" || openModelParameterTruth.snapshot_date !== "2026-07-31") {
  throw new Error("Open-model parameter-truth ledger has an unsupported schema/vintage");
}
for (const source of openModelParameterTruth.source_files) {
  const sourcePath = `${workDir}/${source.path}`;
  if (await sha256(sourcePath) !== source.sha256) throw new Error(`Parameter-truth source hash mismatch: ${source.path}`);
}
const parameterTruthByAlias = new Map();
for (const record of openModelParameterTruth.records) {
  for (const alias of record.aliases) {
    const key = alnumKey(alias);
    const existing = parameterTruthByAlias.get(key);
    if (existing && existing.truth_id !== record.truth_id) throw new Error(`Conflicting parameter-truth alias: ${alias}`);
    parameterTruthByAlias.set(key, record);
  }
}
const applyOpenModelParameterTruth = (row) => {
  const record = [row.canonical_display_name, row.source_model_name, row.matched_epoch_model]
    .map((name) => parameterTruthByAlias.get(alnumKey(name)))
    .find(Boolean);
  const rawTotal = finite(row.total_parameters_b);
  if (!record || rawTotal == null) return row;
  if (!record.accepted_raw_total_parameters_b.some((value) => Math.abs(Number(value) - rawTotal) <= 1e-9)) {
    throw new Error(`Unexpected raw total for ${record.truth_id}: ${rawTotal}`);
  }
  const rawActive = finite(row.active_parameters_b);
  if (rawActive != null && !record.accepted_raw_active_parameters_b.some((value) => Math.abs(Number(value) - rawActive) <= 1e-9)) {
    throw new Error(`Unexpected raw active count for ${record.truth_id}: ${rawActive}`);
  }
  return {
    ...row,
    raw_total_parameters_b: rawTotal,
    raw_active_parameters_b: rawActive ?? "",
    total_parameters_b: Number(record.canonical_total_parameters_b),
    active_parameters_b: rawActive == null ? row.active_parameters_b : Number(record.canonical_active_parameters_b),
    parameter_truth_id: record.truth_id,
    parameter_truth_basis: record.parameter_value_basis,
    parameter_value_source: `${row.parameter_value_source || "source parameter label"}; canonicalized by ${record.truth_id}`,
  };
};
const providerTag = (organization) => {
  const org = String(organization || "").toLowerCase();
  if (org.includes("openai")) return "openai";
  if (org.includes("anthropic")) return "anthropic";
  if (org.includes("meta")) return "meta";
  if (org.includes("mistral")) return "mistral";
  if (org.includes("alibaba") || org.includes("qwen")) return "alibaba";
  if (org.includes("google")) return "google";
  if (org.includes("deepseek")) return "deepseek";
  if (org.includes("moonshot") || org.includes("kimi")) return "moonshot";
  if (org.includes("xai") || org.includes("spacex")) return "xai";
  return slug(organization || "unknown");
};

const objectFromRow = (headers, row) => Object.fromEntries(headers.map((header, index) => [header, row[index] ?? ""]));
const unique = (values) => [...new Set(values.filter(Boolean))];

const noCotDateMatrix = parseCsv(await fs.readFile(noCotDateOverridePath, "utf8"));
const noCotDateHeaders = noCotDateMatrix[0];
const noCotDateRows = noCotDateMatrix.slice(1).filter((row) => row.some((cell) => cell !== "")).map((row, index) => {
  if (row.length !== noCotDateHeaders.length) throw new Error(`No-CoT date override row ${index + 2} has ${row.length}/${noCotDateHeaders.length} fields`);
  return objectFromRow(noCotDateHeaders, row);
});
if (noCotDateRows.length !== 4 || new Set(noCotDateRows.map((row) => row.paper_model)).size !== 4) {
  throw new Error("Expected four unique no-CoT date overrides");
}
if (noCotDateRows.some((row) => row.parameter_join_policy !== "date_only_no_epoch_parameter_join")) {
  throw new Error("No-CoT date override unexpectedly authorizes a parameter join");
}
const noCotDateByModel = new Map(noCotDateRows.map((row) => [row.paper_model, row]));

const frontierPrimaryMatrix = parseCsv(await fs.readFile(frontierPrimaryEvidencePath, "utf8"));
const frontierPrimaryHeaders = frontierPrimaryMatrix[0];
const frontierPrimaryRows = frontierPrimaryMatrix.slice(1).filter((row) => row.some((cell) => cell !== "")).map((row, index) => {
  if (row.length !== frontierPrimaryHeaders.length) throw new Error(`Frontier primary evidence row ${index + 2} has ${row.length}/${frontierPrimaryHeaders.length} fields`);
  return objectFromRow(frontierPrimaryHeaders, row);
});
const frontierPrimaryById = new Map(frontierPrimaryRows.map((row) => [row.evidence_id, row]));
if (frontierPrimaryRows.length !== 5 || frontierPrimaryById.size !== 5) throw new Error("Expected five unique frontier primary-evidence rows");
const fableMythosIdentity = frontierPrimaryById.get("anthropic_fable_mythos_shared_weights");
if (fableMythosIdentity?.model !== "Claude Fable 5" || fableMythosIdentity?.comparator_model !== "Claude Mythos 5" || fableMythosIdentity?.parameter_identity_policy !== "same_underlying_weights_single_parameter_target") {
  throw new Error("Official Fable/Mythos shared-weight identity is missing or malformed");
}
const fableFallbackCaveat = frontierPrimaryById.get("anthropic_fable_fallback_scope");
if (fableFallbackCaveat?.parameter_identity_policy !== "fallback_is_serving_behavior_not_shared_base") {
  throw new Error("Fable/Opus fallback caveat unexpectedly authorizes a shared base");
}

const k3ReleaseEvidence = JSON.parse(await fs.readFile(k3ReleaseEvidencePath, "utf8"));
const k3Architecture = k3ReleaseEvidence.kimi_k3;
if (Number(k3Architecture.total_parameters_b_exact) !== 2780 || Number(k3Architecture.activated_parameters_b_exact) !== 104.2) {
  throw new Error("Kimi K3 exact primary-source parameter evidence is missing or rounded");
}
const exactParameterOverrides = new Map([
  ["Kimi K3", {
    totalB: Number(k3Architecture.total_parameters_b_exact),
    activeB: Number(k3Architecture.activated_parameters_b_exact),
    source: "Kimi K3 official technical report Table 1",
    sourcePath: k3ReleaseEvidencePath,
    sourceUrl: "https://github.com/MoonshotAI/Kimi-K3/blob/7c5be9599120d7993748de66a76128614f15f210/k3_tech_report.pdf",
  }],
]);

// ------------------------- Epoch source -------------------------

const epochBytes = await fs.readFile(epochPath);
const epochText = epochBytes.toString("utf8");
const epochMatrix = parseCsv(epochText);
const epochHeaders = [...epochMatrix[0]];
epochHeaders[0] = epochHeaders[0].replace(/^\uFEFF/, "");
if (epochHeaders.length !== 57) throw new Error(`Expected 57 Epoch columns; found ${epochHeaders.length}`);
if (new Set(epochHeaders).size !== epochHeaders.length) throw new Error("Epoch headers are not unique");

const epochRecords = epochMatrix.slice(1).filter((row) => row.some((cell) => cell !== "")).map((row, index) => {
  if (row.length !== epochHeaders.length) throw new Error(`Epoch CSV row ${index + 2} has ${row.length} fields`);
  const raw = objectFromRow(epochHeaders, row);
  return {
    sourceRow: index + 2,
    raw,
    model: raw.Model,
    organization: raw.Organization,
    releaseDate: raw["Publication date"],
    parametersB: finite(raw.Parameters) == null ? null : finite(raw.Parameters) / 1e9,
    baseModel: raw["Base model"],
    link: raw.Link,
  };
});
if (epochRecords.length !== 3574) throw new Error(`Expected 3574 Epoch records; found ${epochRecords.length}`);

const epochByLabel = new Map();
const epochLabelsByKey = new Map();
for (const record of epochRecords) {
  if (!epochByLabel.has(record.model)) epochByLabel.set(record.model, []);
  epochByLabel.get(record.model).push(record);
  const key = alnumKey(record.model);
  if (!epochLabelsByKey.has(key)) epochLabelsByKey.set(key, new Set());
  epochLabelsByKey.get(key).add(record.model);
}
const duplicateEpochLabels = [...epochByLabel.entries()].filter(([, rows]) => rows.length > 1).map(([label]) => label);
if (duplicateEpochLabels.length !== 5) throw new Error(`Expected five duplicated Epoch labels; found ${duplicateEpochLabels.length}`);

// ------------------------- Epoch archive source views -------------------------

const epochArchiveEntrySpecs = [
  { entry: "frontier_ai_models.csv", source: "Epoch Frontier View", expected: 137, prefix: "epoch_frontier_view" },
  { entry: "notable_ai_models.csv", source: "Epoch Notable View", expected: 1035, prefix: "epoch_notable_view" },
  { entry: "large_scale_ai_models.csv", source: "Epoch Large-Scale View", expected: 518, prefix: "epoch_large_scale_view" },
];
const epochArchiveEntries = epochArchiveEntrySpecs.map((spec) => {
  const text = execFileSync("unzip", ["-p", epochArchivePath, spec.entry], {
    encoding: "utf8",
    maxBuffer: 100 * 1024 * 1024,
  });
  const matrix = parseCsv(text);
  const headers = [...matrix[0]];
  headers[0] = headers[0].replace(/^\uFEFF/, "");
  const records = matrix.slice(1).filter((row) => row.some((cell) => cell !== "")).map((row, index) => {
    if (row.length !== headers.length) throw new Error(`${spec.entry} row ${index + 2} has ${row.length}/${headers.length} fields`);
    return { sourceRow: index + 2, raw: objectFromRow(headers, row) };
  });
  if (records.length !== spec.expected) throw new Error(`${spec.entry}: expected ${spec.expected} records; found ${records.length}`);
  return { ...spec, text, headers, records, sha256: sha256Text(text) };
});
const epochArchiveAllAiText = execFileSync("unzip", ["-p", epochArchivePath, "all_ai_models.csv"], {
  encoding: "utf8",
  maxBuffer: 100 * 1024 * 1024,
});
const epochArchiveAllAiHash = sha256Text(epochArchiveAllAiText);
if (epochArchiveAllAiHash !== "a72dc6d9711835d8b1078aa6e6ef89d51a53de7b26092bf5a5aec18d51909437") {
  throw new Error("ai_models.zip no longer matches the frozen July 17 historical source view");
}
const epochArchiveMatchesCurrentSnapshot = epochArchiveAllAiHash === sha256Text(epochText);

const epochArchiveAllAiMatrix = parseCsv(epochArchiveAllAiText);
const epochArchiveAllAiHeaders = [...epochArchiveAllAiMatrix[0]];
epochArchiveAllAiHeaders[0] = epochArchiveAllAiHeaders[0].replace(/^\uFEFF/, "");
const epochHistoricalRecords = epochArchiveAllAiMatrix.slice(1)
  .filter((row) => row.some((cell) => cell !== ""))
  .map((row, index) => {
    if (row.length !== epochArchiveAllAiHeaders.length) throw new Error(`Historical Epoch all_ai row ${index + 2} has ${row.length}/${epochArchiveAllAiHeaders.length} fields`);
    const raw = objectFromRow(epochArchiveAllAiHeaders, row);
    return { sourceRow: index + 2, raw, model: raw.Model, releaseDate: raw["Publication date"] };
  });
if (epochHistoricalRecords.length !== 3555) throw new Error(`Expected 3555 rows in the frozen July 17 archive; found ${epochHistoricalRecords.length}`);

const epochRecordKey = (raw) => [raw.Model, raw["Publication date"], raw.Organization]
  .map((value) => String(value || "").trim()).join("\u001f");
const uniqueMinimumDisagreementMatch = (referenceRaw, candidates) => {
  const ranked = candidates
    .map((candidate) => ({ candidate, disagreements: disagreementCount(referenceRaw, candidate.raw) }))
    .sort((a, b) => a.disagreements - b.disagreements || a.candidate.sourceRow - b.candidate.sourceRow);
  return ranked.length && (ranked.length === 1 || ranked[0].disagreements < ranked[1].disagreements)
    ? ranked[0] : null;
};
const epochRecordsByExactKey = new Map();
for (const record of epochHistoricalRecords) {
  const key = epochRecordKey(record.raw);
  if (!epochRecordsByExactKey.has(key)) epochRecordsByExactKey.set(key, []);
  epochRecordsByExactKey.get(key).push(record);
}
const currentEpochRecordsByExactKey = new Map();
for (const record of epochRecords) {
  const key = epochRecordKey(record.raw);
  if (!currentEpochRecordsByExactKey.has(key)) currentEpochRecordsByExactKey.set(key, []);
  currentEpochRecordsByExactKey.get(key).push(record);
}
const disagreementCount = (viewRaw, epochRaw) => Object.keys(viewRaw)
  .filter((field) => field in epochRaw)
  .filter((field) => String(viewRaw[field] || "").trim() !== String(epochRaw[field] || "").trim())
  .length;
const epochViewMatches = [];
for (const view of epochArchiveEntries) {
  for (const record of view.records) {
    const candidates = epochRecordsByExactKey.get(epochRecordKey(record.raw)) || [];
    const selectedMatch = uniqueMinimumDisagreementMatch(record.raw, candidates);
    const selected = selectedMatch?.candidate || null;
    const currentCandidates = selected
      ? (currentEpochRecordsByExactKey.get(epochRecordKey(selected.raw)) || [])
      : [];
    const currentMatch = selected
      ? uniqueMinimumDisagreementMatch(selected.raw, currentCandidates)
      : null;
    const currentSelected = currentMatch?.candidate || null;
    epochViewMatches.push({
      view,
      record,
      selected,
      candidates,
      currentSelected,
      currentCandidates,
      status: selected ? (candidates.length === 1 ? "matched_exact_key" : "matched_minimum_field_disagreement")
        : candidates.length ? "ambiguous_exact_key" : "not_in_all_ai_view",
      selectedDisagreements: selected ? disagreementCount(record.raw, selected.raw) : "",
      currentSelectedDisagreements: currentSelected ? disagreementCount(selected.raw, currentSelected.raw) : "",
    });
  }
}
const frontierMatches = epochViewMatches.filter((row) => row.view.entry === "frontier_ai_models.csv");
if (frontierMatches.some((row) => !row.selected) || frontierMatches.some((row) => row.selectedDisagreements !== 0)) {
  throw new Error("Every Epoch frontier view row must match exactly one all_ai source record with zero shared-field disagreements");
}

const makeEpochLink = ({ label = "", status, method, confidence, linkLevel = "none", candidates = [], notes = "" }) => {
  const rows = label ? (epochByLabel.get(label) || []) : [];
  if (label && !rows.length) throw new Error(`Epoch link points to missing label: ${label}`);
  let resolvedStatus = status;
  if (!resolvedStatus && label) {
    resolvedStatus = rows.length === 1 ? `matched_${linkLevel}` : `matched_${linkLevel}_duplicate_label`;
  }
  return {
    label,
    rows,
    status: resolvedStatus || "not_in_epoch",
    method: method || "manual no-match review",
    confidence: confidence || "none",
    linkLevel,
    candidates,
    notes,
  };
};

const exactEpochLink = (name, { methodPrefix = "alphanumeric exact", linkLevel = "checkpoint" } = {}) => {
  const labels = [...(epochLabelsByKey.get(alnumKey(name)) || [])];
  if (labels.length !== 1) return null;
  return makeEpochLink({
    label: labels[0],
    method: methodPrefix,
    confidence: "high",
    linkLevel,
  });
};

// ------------------------- ECI source -------------------------

const eciWorkbook = await SpreadsheetFile.importXlsx(await FileBlob.load(eciPath));
const eciLegacyValues = eciWorkbook.worksheets.getItem("ECI Graph Data").getUsedRange().values;
const eciLegacyHeaders = eciLegacyValues[0].map((value) => String(value ?? ""));
const eciLegacyIndex = Object.fromEntries(eciLegacyHeaders.map((header, index) => [header, index]));
const eciLegacyByName = new Map(eciLegacyValues.slice(1).filter((row) => row[eciLegacyIndex.Model]).map((row, index) => [String(row[eciLegacyIndex.Model]), {
  sourceRow: index + 2,
  raw: objectFromRow(eciLegacyHeaders, row),
  display: String(row[eciLegacyIndex["Display name"]] || row[eciLegacyIndex.Model]),
  organization: String(row[eciLegacyIndex.Organization] || ""),
  accessibility: String(row[eciLegacyIndex["Model accessibility"]] || ""),
  accessibilityGroup: String(row[eciLegacyIndex["Accessibility group"]] || ""),
}]));

const eciReproducedMatrix = parseCsv(await fs.readFile(eciReproducedPath, "utf8"));
const eciReproducedHeaders = [...eciReproducedMatrix[0]];
const expectedEciReproducedHeaders = ["Model", "eci", "eci_ci_low", "eci_ci_high", "date", "model_version", "source"];
if (eciReproducedHeaders.join("\u001f") !== expectedEciReproducedHeaders.join("\u001f")) throw new Error("Unexpected reproduced ECI schema");
const eciReleaseMatrix = parseCsv(await fs.readFile(eciReleaseCrosscheckPath, "utf8"));
const eciReleaseHeaders = [...eciReleaseMatrix[0]];
const eciReleaseByModel = new Map(eciReleaseMatrix.slice(1).filter((row) => row.some((cell) => cell !== "")).map((row) => {
  const raw = objectFromRow(eciReleaseHeaders, row);
  return [raw.model, raw];
}));
const eciRecords = eciReproducedMatrix.slice(1).filter((row) => row.some((cell) => cell !== "")).map((row, index) => {
  const raw = objectFromRow(eciReproducedHeaders, row);
  const legacy = eciLegacyByName.get(raw.Model);
  const release = eciReleaseByModel.get(raw.Model);
  const epochMatches = epochByLabel.get(raw.Model) || [];
  const epochOrganizations = unique(epochMatches.map((match) => match.organization));
  const epochAccess = unique(epochMatches.map((match) => match.raw["Model accessibility"]));
  if (!release) throw new Error(`Missing release-date crosscheck for ${raw.Model}`);
  return {
    sourceRow: index + 2,
    raw: { reproduced_eci: raw, release_date_crosscheck: release, legacy_graph_data: legacy?.raw || null },
    model: raw.Model,
    display: legacy?.display || raw.Model,
    releaseDate: release.regression_release_date,
    eci: finite(raw.eci),
    eciLow: finite(raw.eci_ci_low),
    eciHigh: finite(raw.eci_ci_high),
    organization: epochOrganizations.length === 1 ? epochOrganizations[0] : (legacy?.organization || ""),
    accessibility: epochAccess.length === 1 ? epochAccess[0] : (legacy?.accessibility || ""),
    accessibilityGroup: legacy?.accessibilityGroup || "",
  };
});
if (eciRecords.length !== 213) throw new Error(`Expected 213 reproduced ECI rows; found ${eciRecords.length}`);
if (new Set(eciRecords.map((row) => row.model)).size !== eciRecords.length) throw new Error("ECI model names are not unique");
if (eciReleaseByModel.size !== eciRecords.length) throw new Error("ECI release crosscheck must exactly cover the reproduced score ledger");

const eciRegressionValues = eciWorkbook.worksheets.getItem("Regression Data").getUsedRange().values;
const eciRegressionHeaders = eciRegressionValues[0].map((value) => String(value ?? ""));
const eciRegressionIndex = Object.fromEntries(eciRegressionHeaders.map((header, index) => [header, index]));
const eciRegressionByName = new Map(eciRegressionValues.slice(1).filter((row) => row[eciRegressionIndex.Model]).map((row, index) => [String(row[eciRegressionIndex.Model]), {
  sourceRow: index + 2,
  totalParametersB: finite(row[eciRegressionIndex["Total params (B)"]]),
  sourceUrl: String(row[eciRegressionIndex["Source URL"]] || ""),
  matchAssumption: String(row[eciRegressionIndex["Match / assumption"]] || ""),
  raw: objectFromRow(eciRegressionHeaders, row),
}]));

const eciByName = new Map(eciRecords.map((row) => [row.model, row]));
const eciLegacyOnlyRecords = [...eciLegacyByName.entries()]
  .filter(([model]) => !eciByName.has(model))
  .map(([model, record]) => ({ model, ...record }));
if (
  eciLegacyOnlyRecords.length !== 2
  || new Set(eciLegacyOnlyRecords.map((row) => row.model)).size !== 2
  || !eciLegacyOnlyRecords.some((row) => row.model === "DeepSeek-V3.1")
  || !eciLegacyOnlyRecords.some((row) => row.model === "Kimi K2 (Sep 2025)")
) throw new Error(`Unexpected retired legacy ECI inventory: ${eciLegacyOnlyRecords.map((row) => row.model).join("|")}`);
const eciNamesByKey = new Map();
for (const record of eciRecords) {
  const key = alnumKey(record.model);
  if (!eciNamesByKey.has(key)) eciNamesByKey.set(key, []);
  eciNamesByKey.get(key).push(record.model);
}

const eciComponentText = await fs.readFile(eciComponentPath, "utf8");
const eciComponentMatrix = parseCsv(eciComponentText);
const eciComponentHeaders = [...eciComponentMatrix[0]];
eciComponentHeaders[0] = eciComponentHeaders[0].replace(/^\uFEFF/, "");
const expectedEciComponentHeaders = [
  "model_id", "benchmark_id", "performance", "benchmark", "benchmark_release_date", "optimized",
  "model", "model_version", "Model", "model_group", "Model aggregation", "Model Aggregation Date", "date", "source",
];
if (eciComponentHeaders.join("\u001f") !== expectedEciComponentHeaders.join("\u001f")) {
  throw new Error(`Unexpected ECI component schema: ${eciComponentHeaders.join(" | ")}`);
}
const eciComponentRecords = eciComponentMatrix.slice(1)
  .filter((row) => row.some((cell) => cell !== ""))
  .map((row, index) => {
    if (row.length !== eciComponentHeaders.length) throw new Error(`ECI component row ${index + 2} has ${row.length}/${eciComponentHeaders.length} fields`);
    const raw = objectFromRow(eciComponentHeaders, row);
    return {
      sourceRow: index + 2,
      raw,
      model: raw.model,
      modelVersion: raw.model_version,
      benchmark: raw.benchmark,
      performance: finite(raw.performance),
      optimized: raw.optimized,
      releaseDate: raw.date,
      benchmarkReleaseDate: raw.benchmark_release_date,
    };
  });
if (eciComponentRecords.length !== 2059) throw new Error(`Expected 2059 ECI component rows; found ${eciComponentRecords.length}`);
if (new Set(eciComponentRecords.map((row) => row.model)).size !== 213) throw new Error("ECI component snapshot must contain 213 models");
if (new Set(eciComponentRecords.map((row) => row.benchmark)).size !== 54) throw new Error("ECI component snapshot must contain 54 benchmarks");
if (eciComponentRecords.some((row) => !eciByName.has(row.model))) throw new Error("Every ECI component model must match an aggregate ECI model exactly");
if (new Set(eciComponentRecords.map((row) => `${row.model}\u001f${row.benchmark}`)).size !== eciComponentRecords.length) {
  throw new Error("ECI component model/benchmark pairs are not unique");
}
const canonicalForEci = (record) => `checkpoint:${providerTag(record.organization)}:${slug(record.model)}`;

const eciEpochManual = new Map([
  ["Qwen3.7-Max", "Qwen 3.7 Max"],
]);
const eciEpochLinks = new Map();
for (const record of eciRecords) {
  const manual = eciEpochManual.get(record.model);
  const link = manual
    ? makeEpochLink({ label: manual, method: "manual spelling/date-label alias", confidence: "high", linkLevel: "checkpoint" })
    : exactEpochLink(record.model);
  eciEpochLinks.set(record.model, link || makeEpochLink({ status: "not_in_epoch", notes: "No unique exact Epoch checkpoint label" }));
}

const epochCanonical = new Map([...epochByLabel.keys()].map((label) => [label, `checkpoint:epoch:${slug(label)}`]));
for (const eci of eciRecords) {
  const link = eciEpochLinks.get(eci.model);
  if (link.label && link.linkLevel === "checkpoint") {
    const prior = epochCanonical.get(link.label);
    const next = canonicalForEci(eci);
    if (prior && !prior.startsWith("checkpoint:epoch:") && prior !== next) {
      throw new Error(`Two ECI checkpoints map to one Epoch label: ${link.label}`);
    }
    epochCanonical.set(link.label, next);
  }
}

// ------------------------- Artificial Analysis source -------------------------

const aaLines = (await fs.readFile(aaPath, "utf8")).split(/\r?\n/).map((line) => line.trim());
const aaStart = aaLines.indexOf("Claude Fable 5 (with fallback)");
const aaEnd = aaLines.indexOf("Key definitions");
if (aaStart < 0 || aaEnd < 0 || (aaEnd - aaStart) % 11 !== 0) throw new Error("AA attachment does not contain a complete 11-line leaderboard table");

const aaConfigPattern = /\s*\((max|xhigh|high|medium|low|minimal|non-reasoning(?:,\s*(?:high|low effort))?|with fallback)\)\s*$/i;
const stripAaConfiguration = (name) => String(name).replace(aaConfigPattern, "").trim();
const aaConfiguration = (name) => String(name).match(aaConfigPattern)?.[1] || "";
const parseMagnitude = (value) => {
  const match = String(value || "").trim().match(/^([0-9.]+)\s*([kKmMtT])?$/);
  if (!match) return null;
  const multiplier = { k: 1e3, m: 1e6, t: 1e12 }[(match[2] || "").toLowerCase()] || 1;
  return Number(match[1]) * multiplier;
};
const parseAaNumber = (value) => {
  const match = String(value || "").replace(/^\$/, "").match(/-?[0-9]+(?:\.[0-9]+)?/);
  return match ? Number(match[0]) : null;
};

const aaRecords = [];
for (let i = aaStart; i < aaEnd; i += 11) {
  const cells = aaLines.slice(i, i + 11);
  if (cells[9] !== "Model" || cells[10] !== "Providers") throw new Error(`Malformed AA record at attachment line ${i + 1}`);
  aaRecords.push({
    sourceRow: i + 1,
    displayName: cells[0],
    baseName: stripAaConfiguration(cells[0]),
    configuration: aaConfiguration(cells[0]),
    contextRaw: cells[1],
    creator: cells[2],
    provider: cells[3],
    scoreRaw: cells[4],
    priceRaw: cells[5],
    speedRaw: cells[6],
    latencyRaw: cells[7],
    totalResponseRaw: cells[8],
    raw: {
      display_name: cells[0], context_window: cells[1], creator: cells[2], provider: cells[3],
      intelligence_index: cells[4], blended_price_usd_per_million_tokens: cells[5],
      output_speed_tokens_per_second: cells[6], latency_first_chunk_seconds: cells[7],
      total_response_seconds: cells[8], footer_token_1: cells[9], footer_token_2: cells[10],
    },
  });
}
if (aaRecords.length !== 274) throw new Error(`Expected 274 AA records; found ${aaRecords.length}`);

const aaDetailedMatrix = parseCsv(await fs.readFile(aaDetailedPath, "utf8"));
const aaDetailedHeaders = [...aaDetailedMatrix[0]];
const expectedAaDetailedHeaders = [
  "model_id", "slug", "name", "short_name", "creator_id", "creator_slug", "creator_name", "creator_url",
  "release_date", "knowledge_cutoff_date", "is_reasoning", "reasoning_tokens_setting", "is_open_weights",
  "open_source_categorization", "parameters_b", "active_parameters_b", "size_class", "context_window_tokens",
  "intelligence_index", "intelligence_index_estimated", "coding_index", "agentic_index", "gdpval",
  "gdpval_normalized", "tau_banking", "terminal_bench_v2_1", "scicode", "hle", "gpqa", "critpt",
  "omniscience", "omniscience_accuracy", "omniscience_hallucination_rate", "lcr", "ifbench", "apex_agents",
  "automation_bench_partial_score", "enterprise_ops_gym", "mmmu_pro", "price_input_usd_per_mtoken",
  "price_output_usd_per_mtoken", "price_cache_hit_usd_per_mtoken", "price_blended_7_2_1_usd_per_mtoken",
  "intelligence_time_per_task_seconds", "intelligence_cost_total_usd", "intelligence_cost_per_task_usd",
  "intelligence_input_tokens_total", "intelligence_output_tokens_total", "intelligence_answer_tokens_total",
  "intelligence_reasoning_tokens_total", "intelligence_output_tokens_per_task", "intelligence_answer_tokens_per_task",
  "intelligence_reasoning_tokens_per_task", "median_output_speed_tps", "median_time_to_first_chunk_seconds",
  "performance_data_source_type", "performance_provider_name", "model_weights_source_url", "license_name",
  "license_url", "commercial_allowed", "deprecated", "source_page_url", "snapshot_html_sha256", "source_record_json",
];
if (aaDetailedHeaders.join("\u001f") !== expectedAaDetailedHeaders.join("\u001f")) {
  throw new Error(`Unexpected AA detailed schema: ${aaDetailedHeaders.join(" | ")}`);
}
const aaDetailedRecords = aaDetailedMatrix.slice(1)
  .filter((row) => row.some((cell) => cell !== ""))
  .map((row, index) => {
    if (row.length !== aaDetailedHeaders.length) {
      throw new Error(`AA detailed row ${index + 2} has ${row.length}/${aaDetailedHeaders.length} fields`);
    }
    const raw = objectFromRow(aaDetailedHeaders, row);
    let sourceRecord;
    try {
      sourceRecord = JSON.parse(raw.source_record_json);
    } catch (error) {
      throw new Error(`AA detailed row ${index + 2} has invalid source_record_json: ${error.message}`);
    }
    if (
      sourceRecord.id !== raw.model_id
      || sourceRecord.slug !== raw.slug
      || sourceRecord.name !== raw.name
      || sourceRecord.releaseDate !== raw.release_date
    ) {
      throw new Error(`AA detailed row ${index + 2} normalized identity disagrees with source_record_json`);
    }
    return { sourceRow: index + 2, raw, sourceRecord };
  });
if (aaDetailedRecords.length !== 587) throw new Error(`Expected 587 AA detailed records; found ${aaDetailedRecords.length}`);
if (new Set(aaDetailedRecords.map((row) => row.raw.model_id)).size !== aaDetailedRecords.length) {
  throw new Error("AA detailed model IDs are not unique");
}
if (new Set(aaDetailedRecords.map((row) => row.raw.slug)).size !== aaDetailedRecords.length) {
  throw new Error("AA detailed slugs are not unique");
}

const aaManualEpoch = new Map([
  ["Qwen3.7 Max", { label: "Qwen 3.7 Max", method: "manual duplicate-alias collapse; same Alibaba release/reference/date", confidence: "high", linkLevel: "checkpoint" }],
  ["Gemini 3.1 Pro Preview", { label: "Gemini 3.1 Pro", method: "manual Preview-label alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Grok 4.3", { label: "Grok 4.3 Beta", method: "manual Beta-label alias", confidence: "medium", linkLevel: "checkpoint" }],
  ["Hy3-preview", { label: "Tencent Hy3 preview", method: "manual developer-prefix alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Claude 4.5 Haiku", { label: "Claude Haiku 4.5", method: "manual token-order alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Gemma 4 31B", { label: "Gemma 4 31B IT", method: "manual instruction-tuned suffix alias", confidence: "high", linkLevel: "checkpoint" }],
  ["NVIDIA Nemotron 3 Super", { label: "Nemotron 3 Super", method: "manual developer-prefix alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Trinity Large Thinking", { label: "Arcee Trinity Large", method: "manual product/technical-report alias", confidence: "medium", linkLevel: "checkpoint" }],
  ["EXAONE 4.5 33B", { label: "EXAONE 4.5", method: "manual size-suffix alias; Epoch reports 33.2B", confidence: "high", linkLevel: "checkpoint" }],
  ["Nova 2.0 Pro Preview", { label: "Nova 2 Pro (Preview)", method: "manual version-format alias", confidence: "high", linkLevel: "checkpoint" }],
  ["HyperCLOVA X SEED Think (32B)", { label: "HyperCLOVA X SEED 32B Think", method: "manual size-token-order alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Mistral Large 3", { label: "Mistral 3 Large", method: "manual token-order alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Nova Premier", { label: "Amazon Nova Premier", method: "manual developer-prefix alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Gemma 4 E4B", { label: "Gemma 4 26B A4B", method: "manual active-expert naming alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Gemma 4 E2B", { label: "Gemma 4 12B", method: "manual active-expert naming alias", confidence: "medium", linkLevel: "checkpoint" }],
  ["Llama Nemotron Super 49B v1.5", { label: "Llama Nemotron Super v1.5", method: "manual size-suffix alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Llama Nemotron Ultra", { label: "Llama Nemotron Ultra 253B", method: "manual size-suffix alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Kimi Linear 48B A3B Instruct", { label: "Kimi Linear", method: "manual architecture/configuration suffix alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Ring-flash-2.0", { label: "Ring-flash-linear-2.0", method: "manual architecture-token alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Command A", { label: "Cohere Command A", method: "manual developer-prefix alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Llama 3.1 Nemotron 70B", { label: "Llama-3.1-Nemotron-70B-Instruct", method: "manual Instruct-suffix alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Llama 3.2 90B (Vision)", { label: "Llama 3.2 90B", method: "manual modality-label alias; Epoch parameter notes identify Vision", confidence: "high", linkLevel: "checkpoint" }],
  ["Llama 3.2 11B (Vision)", { label: "Llama 3.2 11B", method: "manual modality-label alias; Epoch parameter notes identify Vision", confidence: "high", linkLevel: "checkpoint" }],
  ["Nova Micro", { label: "Amazon Nova Micro", method: "manual developer-prefix alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Granite 4.0 H 1B", { label: "Granite-4.0-H-Tiny", method: "manual active-parameter alias; Epoch reports 7B total/1B active", confidence: "high", linkLevel: "checkpoint" }],
  ["Granite 4.0 Micro", { label: "Granite-4.0-H-Micro", method: "manual hybrid-family alias", confidence: "high", linkLevel: "checkpoint" }],
  ["Apertus 70B Instruct", { label: "Apertus 70B", method: "manual Epoch family record for Base/Instruct release", confidence: "medium", linkLevel: "family_record" }],
  ["Apertus 8B Instruct", { label: "Apertus 8B", method: "manual Epoch family record for Base/Instruct release", confidence: "medium", linkLevel: "family_record" }],
  ["GLM-5-Turbo", { label: "GLM-5", method: "manual post-training/base relationship", confidence: "medium", linkLevel: "base_only" }],
  ["Nemotron 3 Nano Omni 30B A3B Reasoning", { label: "Nemotron 3-Nano-30B-A3B", method: "manual Omni/Reasoning variant to base architecture", confidence: "medium", linkLevel: "base_only" }],
  ["DiffusionGemma 26B A4B", { label: "Gemma 4 26B A4B", method: "manual derivative to declared base architecture", confidence: "medium", linkLevel: "base_only" }],
  ["Tri-21B-think Preview", { label: "Tri-21B", method: "manual reasoning variant to pretrain/base record", confidence: "medium", linkLevel: "base_only" }],
  ["Tri-21B-Think", { label: "Tri-21B", method: "manual reasoning variant to pretrain/base record", confidence: "medium", linkLevel: "base_only" }],
  ["NVIDIA Nemotron Nano 12B v2 VL", { label: "NVIDIA-Nemotron-Nano-12B-v2", method: "manual VL variant to 12B v2 base", confidence: "medium", linkLevel: "base_only" }],
  ["Olmo 3.1 32B Instruct", { label: "Olmo 3 32B Instruct", method: "manual 3.1 post-training variant to 32B base family", confidence: "medium", linkLevel: "base_only" }],
  ["Command A+", { label: "Cohere Command A", method: "manual A+ post-training variant to Command A base", confidence: "low", linkLevel: "base_only" }],
]);

const aaAmbiguousEpoch = new Map([
  ["Gemini 2.5 Pro", ["Gemini 2.5 Pro (Mar 2025)", "Gemini 2.5 Pro (May 2025)", "Gemini 2.5 Pro (Jun 2025)"]],
  ["Devstral 2", ["Devstral 2 (24B)", "Devstral 2 (123B)"]],
  ["NVIDIA Nemotron 3 Nano", ["NVIDIA-Nemotron-Nano-9B-v2", "NVIDIA-Nemotron-Nano-12B-v2", "Nemotron 3-Nano-30B-A3B"]],
  ["MiMo-V2-Flash (Feb 2026)", ["MiMo-V2-Flash"]],
  ["Olmo 3 7B", ["Olmo 3", "Olmo 3 7B Think"]],
]);

const aaReviewedNoEpoch = new Set([
  // Explicit no-match decisions after checking exact names, organizations, versions, sizes, and plausible aliases.
  "GPT-5.6 Sol", "Kimi K3", "Claude Opus 4.8", "GPT-5.6 Terra", "Grok 4.5", "GPT-5.6 Luna",
  "Muse Spark 1.1", "Hy3", "Nex-N2-Pro", "Inkling", "Grok Build 0.1 0616", "Qwen3.7 Plus",
  "JT-4.1 Flash 236B A21B", "MiniMax-M2.7", "MiMo-V2.5", "MiMo-V2-Omni-0327", "MiMo-V2-Omni",
  "GLM 5V Turbo", "KAT-Coder-Pro V2", "LongCat 2.0", "Qwen3.6 35B A3B", "Ring-2.6-1T",
  "Step 3.7 Flash", "GPT-5.5 Instant (June 2026)", "JT-35B-Flash", "KAT-Coder-Pro V1",
  "Ling-2.6-1T", "Step 3.5 Flash 2603", "Doubao Seed Code", "Mercury 2", "Qwen3.5 35B A3B",
  "ERNIE 5.0 Thinking Preview", "Nemotron Cascade 2 30B A3B", "Nova 2.0 Omni",
  "Apriel-v1.6-15B-Thinker", "Mistral Small 4", "Nova 2.0 Lite", "JT-MINI", "HyperNova 60B 2605",
  "Devstral Small 2", "K2 Think V2", "LongCat Flash Lite", "Mi:dm K 2.5 Pro", "K2-V2", "Solar Pro 3",
  "Motif-2-12.7B", "MiniCPM5-1B", "Sarvam 105B", "Nanbeige4.1-3B", "Hermes 4 70B",
  "Falcon-H1R-7B", "Step3 VL 10B", "Hermes 4 405B", "Solar Pro 2", "Granite 4.1 30B",
  "NVIDIA Nemotron 3 Nano 4B", "LFM2.5-8B-A1B", "Granite 4.1 8B", "Sarvam 30B",
  "Jamba 1.7 Large", "LFM2 24B A2B", "Granite 4.1 3B", "MiniCPM-V 4.6 1.3B",
  "Jamba Reasoning 3B", "Molmo 7B-D", "Ling-mini-2.0", "LFM2.5-1.2B-Thinking", "Jamba 1.7 Mini",
  "LFM2 2.6B", "LFM2.5-1.2B-Instruct", "DeepHermes 3 - Llama-3.1 8B", "Granite 4.0 1B",
  "Molmo2-8B", "LFM2 8B A1B", "LFM2.5-VL-1.6B", "Granite 4.0 350M", "Tiny Aya Global",
  "Granite 4.0 H 350M", "Gemini 3 Deep Think", "Mi:dm K 2.5 Pro Preview", "Cogito v2.1",
]);

const matchAaToEpoch = (record) => {
  const manual = aaManualEpoch.get(record.baseName);
  if (manual) return makeEpochLink(manual);
  const exact = exactEpochLink(record.baseName, {
    methodPrefix: record.configuration ? "configuration stripped + alphanumeric exact" : "alphanumeric exact",
    linkLevel: record.configuration.toLowerCase() === "with fallback" ? "checkpoint_system_configuration" : "checkpoint",
  });
  if (exact) {
    if (record.configuration.toLowerCase() === "with fallback") {
      exact.notes = "AA score is a fallback/system configuration; identity is linked to the regular checkpoint without treating the score as a pure base-model measurement.";
    }
    return exact;
  }
  const candidates = aaAmbiguousEpoch.get(record.baseName);
  if (candidates) {
    for (const candidate of candidates) if (!epochByLabel.has(candidate)) throw new Error(`AA candidate is absent from Epoch: ${record.baseName} -> ${candidate}`);
    return makeEpochLink({
      status: "ambiguous_epoch_candidate",
      method: "manual ambiguity review",
      confidence: "none",
      linkLevel: "candidate_only",
      candidates,
      notes: "No checkpoint was selected because the AA label does not uniquely identify an Epoch record.",
    });
  }
  if (aaReviewedNoEpoch.has(record.baseName)) {
    return makeEpochLink({
      status: "not_in_epoch_after_manual_review",
      method: "manual no-match review",
      confidence: "none",
      linkLevel: "none",
      notes: "No exact checkpoint, family, or defensible base record was found in the 2026-07-31 Epoch snapshot.",
    });
  }
  return makeEpochLink({
    status: "UNREVIEWED",
    method: "none",
    confidence: "none",
    linkLevel: "none",
    notes: "Build-blocking unreviewed AA identity",
  });
};

const aaUniqueBaseCreator = unique(aaRecords.map((row) => `${row.baseName}\u001f${row.creator}`));
const aaUnreviewed = unique(aaRecords.filter((row) => matchAaToEpoch(row).status === "UNREVIEWED").map((row) => row.baseName));
if (aaUnreviewed.length) {
  throw new Error(`AA identities still require review (${aaUnreviewed.length}):\n${aaUnreviewed.map((name) => `  ${JSON.stringify(name)},`).join("\n")}`);
}

const aaRowsByDisplayName = new Map();
for (const record of aaRecords) {
  if (!aaRowsByDisplayName.has(record.displayName)) aaRowsByDisplayName.set(record.displayName, []);
  aaRowsByDisplayName.get(record.displayName).push(record);
}
for (const rows of aaRowsByDisplayName.values()) {
  const ranked = [...rows].sort((a, b) => {
    const aScore = parseAaNumber(a.scoreRaw);
    const bScore = parseAaNumber(b.scoreRaw);
    return (bScore ?? -Infinity) - (aScore ?? -Infinity) || a.sourceRow - b.sourceRow;
  });
  const selectedSourceRow = ranked[0].sourceRow;
  for (const record of rows) {
    record.selectedForModelLevel = record.sourceRow === selectedSourceRow;
    record.selectionReason = record.selectedForModelLevel
      ? (rows.length === 1 ? "unique AA display name" : "highest score among identical AA display names")
      : `excluded duplicate AA display name; selected attachment line ${selectedSourceRow}`;
  }
}

const aaEciManual = new Map([
  ["Gemini 3.1 Pro Preview", "Gemini 3.1 Pro"],
  ["Grok 4.3", "Grok 4.3 Beta"],
  ["Claude 4.5 Haiku", "Claude Haiku 4.5"],
]);
const matchAaToEci = (record) => {
  const manual = aaEciManual.get(record.baseName);
  if (manual) return { name: manual, status: "manual checkpoint alias", confidence: "high" };
  const names = eciNamesByKey.get(alnumKey(record.baseName)) || [];
  if (names.length === 1) return { name: names[0], status: "alphanumeric exact", confidence: "high" };
  return { name: "", status: names.length > 1 ? "ambiguous ECI label" : "not in ECI", confidence: "none" };
};

// ------------------------- No-CoT paper source -------------------------

const tex = execFileSync("tar", ["-xOzf", latexPath, "arxiv_version.tex"], {
  encoding: "utf8",
  maxBuffer: 80 * 1024 * 1024,
});

const normalizeTex = (value) => value
  .replace(/\\textbf\{([^{}]*)\}/g, "$1")
  .replace(/\\textit\{([^{}]*)\}/g, "$1")
  .replace(/\\text\{k\}/g, "k")
  .replace(/\\,/g, "")
  .replace(/~/g, " ")
  .replace(/[${}]/g, "")
  .replace(/\\%/g, "%")
  .replace(/\s+/g, " ")
  .trim();

const tableByLabel = (label) => {
  const labelIndex = tex.indexOf(`\\label{${label}}`);
  if (labelIndex < 0) throw new Error(`Missing LaTeX table: ${label}`);
  const start = tex.lastIndexOf("\\begin{table", labelIndex);
  const end = tex.indexOf("\\end{table}", labelIndex);
  if (start < 0 || end < 0) throw new Error(`Malformed LaTeX table: ${label}`);
  return tex.slice(start, end + "\\end{table}".length);
};

const dataLines = (block) => block
  .split(/\r?\n/)
  .map((line) => line.trim())
  .filter((line) => line.includes("&") && line.endsWith("\\\\") && !line.includes("multicolumn"));
const splitCells = (line) => line.replace(/\\\\\s*$/, "").split("&").map((cell) => normalizeTex(cell));

const monthNumbers = new Map([
  ["January", 1], ["February", 2], ["March", 3], ["April", 4], ["May", 5], ["June", 6],
  ["July", 7], ["August", 8], ["September", 9], ["October", 10], ["November", 11], ["December", 12],
]);
const monthIso = (text) => {
  const match = normalizeTex(text).match(/^([A-Za-z]+)\s+(\d{4})$/);
  if (!match || !monthNumbers.has(match[1])) throw new Error(`Bad paper month: ${text}`);
  return `${match[2]}-${String(monthNumbers.get(match[1])).padStart(2, "0")}-01`;
};
const magnitude = (text) => {
  const cleaned = normalizeTex(text).replace(/,/g, "").replace(/^</, "").trim();
  const match = cleaned.match(/^([0-9.]+)\s*([kKmMtT])?$/);
  if (!match) throw new Error(`Bad magnitude: ${text}`);
  const suffix = (match[2] || "").toLowerCase();
  return Number(match[1]) * (suffix === "k" ? 1e3 : suffix === "m" ? 1e6 : suffix === "t" ? 1e12 : 1);
};
const paramsB = (text) => {
  const match = normalizeTex(text).replace(/\s+/g, "").match(/^([0-9.]+)([BT])$/i);
  if (!match) throw new Error(`Bad paper parameter cell: ${text}`);
  return Number(match[1]) * (match[2].toUpperCase() === "T" ? 1000 : 1);
};
const estimateWithCi = (cell) => {
  const cleaned = normalizeTex(cell.replace(/\\tiny\{([^{}]*)\}/g, "$1"));
  const ci = cleaned.match(/\[([^,]+),\s*([^\]]+)\]/);
  const estimateText = cleaned.replace(/\[[^\]]+\]/, "").trim();
  return {
    estimate: magnitude(estimateText),
    low: ci ? magnitude(ci[1]) : null,
    high: ci ? magnitude(ci[2]) : null,
    floor: estimateText.startsWith("<"),
    raw: cleaned,
  };
};
const macroNumber = (name) => {
  const match = tex.match(new RegExp(`\\\\newcommand\\{\\\\${name}\\}\\{([0-9.]+)\\}`));
  if (!match) throw new Error(`Missing LaTeX macro ${name}`);
  return Number(match[1]);
};

const frontierRelease = new Map();
for (const line of dataLines(tableByLabel("tab:frontier-models"))) {
  const cells = splitCells(line);
  if (cells.length === 2 && cells[0] !== "Model") frontierRelease.set(cells[0], monthIso(cells[1]));
}
frontierRelease.set("Sonnet 3.7", frontierRelease.get("Claude 3.7 Sonnet"));
for (const version of ["4", "4.1", "4.5", "4.6", "4.7"]) {
  frontierRelease.set(`Opus ${version}`, frontierRelease.get(`Claude Opus ${version}`));
}

const noCotFrontier = [];
for (const line of dataLines(tableByLabel("tab:horizons-per-model"))) {
  const raw = line.replace(/\\\\\s*$/, "").split("&").map((cell) => cell.trim());
  if (raw.length !== 3 || normalizeTex(raw[0]) === "Model") continue;
  const model = normalizeTex(raw[0]);
  noCotFrontier.push({
    suite: "frontier",
    model,
    organization: model.startsWith("Opus") || model.startsWith("Sonnet") ? "Anthropic" : "OpenAI",
    releaseDate: frontierRelease.get(model),
    time: estimateWithCi(raw[1]),
    tokens: estimateWithCi(raw[2]),
    raw: { model, release_month: frontierRelease.get(model), time_horizon: normalizeTex(raw[1]), token_horizon: normalizeTex(raw[2]) },
  });
}

const openMetadata = new Map();
let noCotDeveloper = "";
for (const line of tableByLabel("tab:open-source-models").split(/\r?\n/)) {
  const group = line.match(/\\textit\{([^}]+)\}/);
  if (line.includes("multicolumn") && group) {
    noCotDeveloper = group[1].replace(/\s*\([^)]*\)\s*/, "").trim();
    continue;
  }
  if (!line.includes("&") || !line.trim().endsWith("\\\\")) continue;
  const cells = splitCells(line.trim());
  if (cells.length !== 7 || cells[0] === "Model") continue;
  const parameterParts = cells[2].split("/").map((part) => part.trim());
  const totalB = paramsB(parameterParts[0]);
  const activeB = parameterParts.length === 2 ? paramsB(parameterParts[1]) : totalB;
  openMetadata.set(cells[0], {
    developer: noCotDeveloper,
    model: cells[0],
    releaseDate: monthIso(cells[1]),
    totalB,
    activeB,
    layers: Number(cells[3]),
    architecture: cells[4],
    reasoning: cells[5],
    modality: cells[6],
  });
}

const openHorizons = new Map();
for (const line of dataLines(tableByLabel("tab:open-weight-precise-time-horizons"))) {
  const cells = splitCells(line);
  if (cells.length === 5 && cells[0] !== "Model") {
    openHorizons.set(cells[0], { point: Number(cells[1]), median: Number(cells[2]), low: Number(cells[3]), high: Number(cells[4]) });
  }
}

const noCotOpen = [...openMetadata.values()].map((meta) => {
  const horizon = openHorizons.get(meta.model);
  if (!horizon) throw new Error(`Missing open-weight horizon for ${meta.model}`);
  return {
    suite: "open-weight",
    organization: ({ Llama: "Meta AI", Mistral: "Mistral AI", Qwen: "Alibaba", Gemma: "Google DeepMind", DeepSeek: "DeepSeek", Kimi: "Moonshot" })[meta.developer] || meta.developer,
    ...meta,
    horizon,
    raw: { ...meta, ...horizon },
  };
});
if (noCotFrontier.length !== 14 || noCotOpen.length !== 35) throw new Error(`Unexpected No-CoT counts: ${noCotFrontier.length}/${noCotOpen.length}`);

const scalingMatch = tex.match(/Doubling the 50\\% TH requires a \$([0-9.]+)\\times\$ increase in total parameters, a \$([0-9.]+)\\times\$ increase in active parameters, a \$([0-9.]+)\\times\$ increase in the layer count, or a \$([0-9.]+)\\times\$ increase in pretraining FLOPs/);
if (!scalingMatch) throw new Error("Could not parse No-CoT scaling law");
const noCotLaw = {
  totalParametersMultiplier: Number(scalingMatch[1]),
  activeParametersMultiplier: Number(scalingMatch[2]),
  layerMultiplier: Number(scalingMatch[3]),
  pretrainingFlopsMultiplier: Number(scalingMatch[4]),
  timeHorizonDoublingDays: macroNumber("FINALDOUBLINGTIMETIME"),
  tokenHorizonDoublingDays: macroNumber("FINALDOUBLINGTIMETOKENS"),
};

const paperToEci = new Map(Object.entries({
  "GPT-2": null, "GPT-3": null, "GPT-3.5": null,
  "GPT-4": "GPT-4 (Mar 2023)", "GPT-4 Turbo": "GPT-4 Turbo (Apr 2024)", "GPT-4o": "GPT-4o (May 2024)",
  "Sonnet 3.7": "Claude 3.7 Sonnet", "Opus 4": "Claude Opus 4", "Opus 4.1": "Claude Opus 4.1",
  "Opus 4.5": "Claude Opus 4.5", "Opus 4.6": "Claude Opus 4.6", "GPT-5.4": "GPT-5.4",
  "Opus 4.7": "Claude Opus 4.7", "GPT-5.5": "GPT-5.5",
  "Llama 3 8B Instruct": "Llama 3-8B", "Llama 3.1 8B Instruct": "Llama 3.1-8B",
  "Llama 3.1 70B Instruct": "Llama 3.1-70B", "Llama 3.2 1B Instruct": null,
  "Llama 3.2 3B Instruct": null, "Llama 3.3 70B Instruct": "Llama 3.3 70B",
  "Llama 4 Scout": "Llama 4 Scout", "Llama 4 Maverick": "Llama 4 Maverick", "Mistral Nemo 12B": "Mistral NeMo",
  "Ministral 3 3B": null, "Ministral 3 8B": null, "Ministral 3 14B": null,
  "Qwen 3 8B": null, "Qwen 3 14B": null, "Qwen 3 32B": null, "Qwen 3 30B-A3B (2507)": null,
  "Qwen 3 235B-A22B (2507)": "Qwen3-235B-A22B-Instruct (Jul 2025)", "Qwen 3-Next 80B-A3B": null,
  "Qwen 3.5 9B": null, "Qwen 3.5 27B": null, "Qwen 3.5 35B-A3B": "Qwen 3.5 Flash (hosted 35B-A3B)",
  "Qwen 3.5 122B-A10B": null, "Qwen 3.5 397B-A17B": "Qwen 3.5 Plus (hosted 397B-A17B)",
  "Gemma 3 4B IT": null, "Gemma 3 12B IT": null, "Gemma 3 27B IT": "Gemma 3 27B",
  "Gemma 4 26B-A4B": null, "Gemma 4 31B": "Gemma 4 31B IT",
  "DeepSeek V3 (0324)": "DeepSeek-V3 (Mar 2025)", "DeepSeek V3.1-terminus": null,
  "DeepSeek V3.2": "DeepSeek-V3.2", "DeepSeek V4-flash": null,
  "Kimi K2-0905": null, "Kimi K2.5": "Kimi K2.5", "Kimi K2.6": "Kimi K2.6",
}));
const paperToEpoch = new Map(Object.entries({
  "Llama 3.2 1B Instruct": "Llama 3.2 1B", "Llama 3.2 3B Instruct": "Llama 3.2 3B",
  "Ministral 3 3B": "Ministral 3 3B", "Ministral 3 8B": "Ministral 3 8B", "Ministral 3 14B": "Ministral 3 14B",
  "Qwen 3 8B": "Qwen3-8B", "Qwen 3 14B": "Qwen3-14B", "Qwen 3 32B": "Qwen3-32B",
  "Qwen 3 30B-A3B (2507)": null, "Qwen 3-Next 80B-A3B": "Qwen3-Next-80B-A3B",
  "Qwen 3.5 9B": "Qwen3.5-9B", "Qwen 3.5 27B": "Qwen3.5-27B", "Qwen 3.5 122B-A10B": "Qwen3.5-122B-A10B",
  "Gemma 3 4B IT": "Gemma 3 4B", "Gemma 3 12B IT": "Gemma 3 12B", "Gemma 4 26B-A4B": "Gemma 4 26B A4B",
  "Gemma 4 31B": "Gemma 4 31B IT", "DeepSeek V3.1-terminus": "DeepSeek-V3.1-Terminus", "DeepSeek V4-flash": "DeepSeek-V4-Flash",
}));

const noCotEpochExcluded = new Set(["GPT-2", "GPT-3", "GPT-3.5"]);
const matchNoCotToEpoch = (model) => {
  if (noCotEpochExcluded.has(model)) return makeEpochLink({
    status: "excluded_by_prior_user_instruction",
    method: "explicit prior user instruction",
    confidence: "none",
    linkLevel: "none",
    notes: "Record retained, but GPT-2/3/3.5 were intentionally excluded from No-CoT→Epoch joining.",
  });
  if (paperToEpoch.has(model)) {
    const label = paperToEpoch.get(model);
    return label
      ? makeEpochLink({ label, method: "manual paper/Epoch checkpoint alias", confidence: "high", linkLevel: "checkpoint" })
      : makeEpochLink({ status: "not_in_epoch_after_manual_review", method: "manual paper/Epoch review", confidence: "none", linkLevel: "none" });
  }
  const eciName = paperToEci.get(model);
  if (eciName) {
    const eciLink = eciEpochLinks.get(eciName);
    if (eciLink?.label) return { ...eciLink, method: `via ECI checkpoint ${eciName}; ${eciLink.method}` };
  }
  return exactEpochLink(model) || makeEpochLink({ status: "not_in_epoch_after_manual_review", method: "manual paper/Epoch review", confidence: "none", linkLevel: "none" });
};

// ------------------------- METR source -------------------------

const metrMetadata = JSON.parse(await fs.readFile(metrMetadataPath, "utf8"));
const metrAudit = JSON.parse(await fs.readFile(metrAuditPath, "utf8"));
if (metrAudit.status !== "PASS" || metrAudit.legacy_exact_crosscheck?.exact_rows !== 26 || metrAudit.legacy_exact_crosscheck?.mismatch_count !== 0) {
  throw new Error("METR primary-source audit is not an exact 26/26 pass");
}
const metrMatrix = parseCsv(await fs.readFile(metrPath, "utf8"));
const metrHeaders = metrMatrix[0];
const metrSourceRows = metrMatrix.slice(1).filter((row) => row.some((cell) => cell !== "")).map((row, index) => {
  if (row.length !== metrHeaders.length) throw new Error(`METR official row ${index + 2} has ${row.length}/${metrHeaders.length} fields`);
  return objectFromRow(metrHeaders, row);
});
const metrRecords = metrSourceRows.map((row) => {
  const scaffolds = JSON.parse(row.scaffolds_json);
  if (!Array.isArray(scaffolds) || scaffolds.length === 0) throw new Error(`METR row ${row.source_id} has no preserved scaffold array`);
  const numeric = (field) => {
    const value = finite(row[field]);
    if (value == null) throw new Error(`METR row ${row.source_id} has invalid ${field}`);
    return value;
  };
  if (!["true", "false"].includes(row.is_sota)) throw new Error(`METR row ${row.source_id} has invalid is_sota`);
  return {
    id: row.source_id,
    benchmark: row.benchmark_name,
    date: row.release_date,
    average: numeric("average_score"),
    sota: row.is_sota === "true",
    p50: numeric("p50_estimate_minutes"),
    p50Low: numeric("p50_ci_low_minutes"),
    p50High: numeric("p50_ci_high_minutes"),
    p80: numeric("p80_estimate_minutes"),
    p80Low: numeric("p80_ci_low_minutes"),
    p80High: numeric("p80_ci_high_minutes"),
    scaffold: row.scaffold_family,
    scaffolds,
    raw: row,
  };
});
if (metrRecords.length !== 26 || new Set(metrRecords.map((row) => row.id)).size !== 26) {
  throw new Error(`Expected 26 unique official METR model rows; found ${metrRecords.length}`);
}
const METR_LONG_TASKS_VERSION = metrMetadata.trend.long_tasks_version;
const METR_SWAA_VERSION = metrMetadata.trend.swaa_version;
const metrEpochRules = new Map([
  ["claude_3_5_sonnet_20240620_inspect", { label: "Claude 3.5 Sonnet", confidence: "high", linkLevel: "checkpoint" }],
  ["claude_3_5_sonnet_20241022_inspect", { candidates: ["Claude 3.5 Sonnet"], status: "candidate_only_date_mismatch" }],
  ["claude_3_7_sonnet_inspect", { label: "Claude 3.7 Sonnet", confidence: "high", linkLevel: "checkpoint" }],
  ["claude_3_opus_inspect", { label: "Claude 3 Opus", confidence: "high", linkLevel: "checkpoint" }],
  ["claude_4_1_opus_inspect", { label: "Claude Opus 4.1", confidence: "high", linkLevel: "checkpoint" }],
  ["claude_4_opus_inspect", { label: "Claude Opus 4", confidence: "high", linkLevel: "checkpoint" }],
  ["claude_opus_4_5_inspect", { label: "Claude Opus 4.5", confidence: "high", linkLevel: "checkpoint" }],
  ["claude_opus_4_6_inspect", { label: "Claude Opus 4.6", confidence: "high", linkLevel: "checkpoint" }],
  ["davinci_002", { candidates: ["GPT-3.5 (davinci-002)\n", "GPT-3 175B (davinci)"], status: "ambiguous_identifier_vs_release_date" }],
  ["gpt_5_3_codex", { label: "GPT-5.3 Codex", confidence: "high", linkLevel: "checkpoint" }],
  ["gemini_3_pro", { label: "Gemini 3 Pro", confidence: "high", linkLevel: "checkpoint" }],
  ["gemini_3_1_pro", { label: "Gemini 3.1 Pro", confidence: "high", linkLevel: "checkpoint" }],
  ["gpt2", { candidates: ["GPT-2 (1.5B)", "GPT-2 (124M)", "GPT-2 (355M)", "GPT-2 (774M)"], status: "ambiguous_epoch_candidate" }],
  ["gpt_3_5_turbo_instruct", { label: "GPT-3.5 Turbo Instruct", confidence: "medium", linkLevel: "checkpoint", status: "matched_checkpoint_date_conflict" }],
  ["gpt_4", { label: "GPT-4 (Mar 2023)", confidence: "high", linkLevel: "checkpoint" }],
  ["gpt_4_1106_inspect", { label: "GPT-4 Turbo (Nov 2023)", confidence: "high", linkLevel: "checkpoint" }],
  ["gpt_4_turbo_inspect", { label: "GPT-4 Turbo (Apr 2024)", confidence: "high", linkLevel: "checkpoint" }],
  ["gpt_4o_inspect", { label: "GPT-4o", confidence: "high", linkLevel: "checkpoint" }],
  ["gpt_5_1_codex_max_inspect", { label: "GPT-5.1-Codex-Max", confidence: "high", linkLevel: "checkpoint" }],
  ["gpt_5_2", { label: "GPT-5.2", confidence: "high", linkLevel: "checkpoint" }],
  ["gpt_5_2025_08_07_inspect", { label: "GPT-5", confidence: "high", linkLevel: "checkpoint" }],
  ["gpt_5_4", { label: "GPT-5.4", confidence: "high", linkLevel: "checkpoint" }],
  ["o1_inspect", { label: "o1", confidence: "high", linkLevel: "checkpoint" }],
  ["o1_preview", { label: "o1-preview", confidence: "high", linkLevel: "checkpoint" }],
  ["o3_inspect", { label: "o3", confidence: "medium", linkLevel: "checkpoint", status: "matched_checkpoint_date_conflict" }],
  ["claude_mythos_preview_early_inspect", { candidates: ["Claude Mythos 5"], status: "candidate_only_later_checkpoint" }],
]);
if (metrEpochRules.size !== metrRecords.length) throw new Error("METR→Epoch manual rule table is incomplete");

const matchMetrToEpoch = (record) => {
  const rule = metrEpochRules.get(record.id);
  if (rule.label) return makeEpochLink({ label: rule.label, status: rule.status, method: "manual METR identifier/Epoch checkpoint alias", confidence: rule.confidence, linkLevel: rule.linkLevel });
  for (const candidate of rule.candidates) if (!epochByLabel.has(candidate)) throw new Error(`METR candidate missing in Epoch: ${candidate}`);
  return makeEpochLink({ status: rule.status, method: "manual METR ambiguity/date review", confidence: "none", linkLevel: "candidate_only", candidates: rule.candidates });
};

const metrEciRules = new Map([
  ["claude_3_5_sonnet_20240620_inspect", "Claude 3.5 Sonnet"],
  ["claude_3_5_sonnet_20241022_inspect", "Claude 3.5 Sonnet (October 2024)"],
  ["claude_3_7_sonnet_inspect", "Claude 3.7 Sonnet"], ["claude_3_opus_inspect", "Claude 3 Opus"],
  ["claude_4_1_opus_inspect", "Claude Opus 4.1"], ["claude_4_opus_inspect", "Claude Opus 4"],
  ["claude_opus_4_5_inspect", "Claude Opus 4.5"], ["claude_opus_4_6_inspect", "Claude Opus 4.6"],
  ["gpt_5_3_codex", "GPT-5.3 Codex"], ["gemini_3_pro", "Gemini 3 Pro"], ["gemini_3_1_pro", "Gemini 3.1 Pro"],
  ["gpt_4", "GPT-4 (Mar 2023)"], ["gpt_4_turbo_inspect", "GPT-4 Turbo (Apr 2024)"], ["gpt_4o_inspect", "GPT-4o (May 2024)"],
  ["gpt_5_2", "GPT-5.2"], ["gpt_5_2025_08_07_inspect", "GPT-5"], ["gpt_5_4", "GPT-5.4"],
  ["o1_inspect", "o1"], ["o1_preview", "o1-preview"], ["o3_inspect", "o3"],
]);

const metrLaw = {
  allTimeDoublingDays: Number(metrMetadata.trend.all_time_stitched_point_estimate_days),
  from2023DoublingDays: Number(metrMetadata.trend.from_2023_on_point_estimate_days),
  from2023CiLowDays: Number(metrMetadata.trend.from_2023_on_ci_low_days),
  from2023CiHighDays: Number(metrMetadata.trend.from_2023_on_ci_high_days),
  exclusionRule: metrMetadata.trend.exclusion_rule,
};

// ------------------------- Canonical identity and observation table -------------------------

const eciNamesByEpochLabel = new Map();
for (const [eciName, link] of eciEpochLinks) {
  if (!link.label) continue;
  if (!eciNamesByEpochLabel.has(link.label)) eciNamesByEpochLabel.set(link.label, []);
  eciNamesByEpochLabel.get(link.label).push(eciName);
}

const aggregateEpochLink = (link) => {
  const rows = link?.rows || [];
  const numericUnique = (field) => unique(rows.map((row) => row.raw[field]).filter((value) => finite(value) != null).map((value) => Number(value)));
  const textUnique = (field) => unique(rows.map((row) => row.raw[field]).filter(Boolean));
  const parameterValues = numericUnique("Parameters");
  return {
    sourceRows: rows.map((row) => row.sourceRow),
    releaseDates: textUnique("Publication date"),
    parametersB: parameterValues.length === 1 ? parameterValues[0] / 1e9 : null,
    parameterConflict: parameterValues.length > 1 ? parameterValues.join("|") : "",
    baseModels: textUnique("Base model"),
    links: textUnique("Link"),
  };
};

const manualExactDates = new Map([
  ...noCotDateRows.map((row) => [row.paper_model, { date: row.exact_release_date, source: `${row.date_source} (${row.source_checkpoint})` }]),
  ["Kimi K2-0905", { date: "2025-09-05", source: "checkpoint identifier exact date (0905)" }],
  ["Kimi K3", { date: "2026-07-16", source: "user-supplied exact release date" }],
]);

const inferBaseId = ({ display, canonicalId, epochLink }) => {
  const name = String(display || "");
  if (/^Claude (?:Fable|Mythos) 5(?:\b|$)/.test(name)) return "base:anthropic-claude-fable-mythos-5";
  if (/^(?:Claude )?Opus 4\.[5-8](?:\b|$)/.test(name)) return "base:anthropic-opus-4-5plus";
  if (/^GPT-5(?:$|\.[1-5](?:\b|$))/.test(name)) return "base:openai-gpt-5-shared";
  if (epochLink?.label && ["family_record", "base_only"].includes(epochLink.linkLevel)) {
    return `base-ref:${epochCanonical.get(epochLink.label)}`;
  }
  const aggregate = aggregateEpochLink(epochLink);
  if (aggregate.baseModels.length === 1) {
    const base = aggregate.baseModels[0];
    return epochCanonical.has(base) ? `base-ref:${epochCanonical.get(base)}` : `base-label:${slug(base)}`;
  }
  return `base:${canonicalId}`;
};

const identityFields = ({ source, model, organization = "", sourceReleaseDate = "", epochLink = null, eciName = "", sourceIdentity = "" }) => {
  const eci = eciName ? eciByName.get(eciName) : null;
  if (eciName && !eci) throw new Error(`Identity points to missing ECI model: ${eciName}`);
  const checkpointEpochLevels = new Set(["checkpoint", "checkpoint_system_configuration"]);
  const canonicalId = eci
    ? canonicalForEci(eci)
    : epochLink?.label && checkpointEpochLevels.has(epochLink.linkLevel)
      ? epochCanonical.get(epochLink.label)
      : `checkpoint:${source.toLowerCase().replace(/[^a-z0-9]+/g, "-")}:${providerTag(organization)}:${slug(sourceIdentity || model)}`;
  const canonicalDisplay = eci?.display || (epochLink?.label && checkpointEpochLevels.has(epochLink.linkLevel) ? epochLink.label : model);

  const manualDate = manualExactDates.get(model);
  let canonicalReleaseDate = "";
  let canonicalReleaseDateSource = "";
  if (eci?.releaseDate) {
    canonicalReleaseDate = eci.releaseDate;
    canonicalReleaseDateSource = `ECI exact date (${eci.model})`;
  } else if (epochLink?.label && checkpointEpochLevels.has(epochLink.linkLevel)) {
    const dates = aggregateEpochLink(epochLink).releaseDates;
    if (dates.length === 1) {
      canonicalReleaseDate = dates[0];
      canonicalReleaseDateSource = `Epoch exact date (${epochLink.label})`;
    }
  } else if (manualDate) {
    canonicalReleaseDate = manualDate.date;
    canonicalReleaseDateSource = manualDate.source;
  } else if (sourceReleaseDate) {
    canonicalReleaseDate = sourceReleaseDate;
    canonicalReleaseDateSource = "source release date";
  }

  return {
    canonical_checkpoint_id: canonicalId,
    canonical_display_name: canonicalDisplay,
    canonical_base_id: inferBaseId({ display: canonicalDisplay, canonicalId, epochLink }),
    canonical_release_date: canonicalReleaseDate,
    canonical_release_date_source: canonicalReleaseDateSource,
    release_date_delta_days: dayDelta(canonicalReleaseDate, sourceReleaseDate),
  };
};

const observationHeaders = [
  "observation_id", "source", "record_type", "dataset", "snapshot_date", "source_locator", "source_model_name",
  "source_configuration", "source_organization", "source_provider", "model_level_include", "model_level_selection_reason",
  "canonical_checkpoint_id", "canonical_display_name", "canonical_base_id", "canonical_release_date",
  "canonical_release_date_source", "source_release_date", "source_release_date_precision", "release_date_delta_days",
  "date_conflict_flag", "date_conflict_details",
  "matched_epoch_model", "epoch_match_status", "epoch_match_method", "epoch_match_confidence", "epoch_link_level",
  "epoch_candidate_count", "epoch_alternative_candidates", "epoch_source_rows", "epoch_release_dates", "epoch_base_model",
  "matched_eci_model", "eci_match_status", "eci_match_confidence", "benchmark_name",
  "aa_intelligence_index", "aa_score_raw", "aa_score_qualifier", "aa_context_window_tokens", "aa_context_window_raw",
  "aa_blended_price_usd_per_million_tokens", "aa_price_raw", "aa_output_speed_tokens_per_second", "aa_speed_raw",
  "aa_latency_first_chunk_seconds", "aa_latency_raw", "aa_total_response_seconds", "aa_total_response_raw",
  "eci_score", "eci_ci_low", "eci_ci_high", "eci_component_performance", "eci_component_optimized",
  "nocot_time_horizon_minutes", "nocot_time_horizon_median_minutes",
  "nocot_time_horizon_ci_low_minutes", "nocot_time_horizon_ci_high_minutes", "nocot_token_horizon_tokens",
  "nocot_token_horizon_ci_low_tokens", "nocot_token_horizon_ci_high_tokens", "metr_average_score", "metr_is_sota",
  "metr_p50_horizon_minutes", "metr_p50_ci_low_minutes", "metr_p50_ci_high_minutes", "metr_p80_horizon_minutes",
  "metr_p80_ci_low_minutes", "metr_p80_ci_high_minutes", "total_parameters_b", "active_parameters_b",
  "raw_total_parameters_b", "raw_active_parameters_b", "parameter_truth_id", "parameter_truth_basis",
  "parameter_value_source", "epoch_parameters_b", "eci_regression_total_parameters_b", "layers", "architecture",
  "reasoning", "modality", "epoch_training_compute_flop", "epoch_training_dataset_size_total",
  "epoch_training_time_hours", "epoch_hardware_quantity", "epoch_post_training_compute_flop", "epoch_model_accessibility",
  "epoch_open_model_weights", "source_url", "notes", "source_record_json",
];

const observations = [];
const addObservation = (row) => {
  const truthAdjusted = applyOpenModelParameterTruth(row);
  const normalized = Object.fromEntries(observationHeaders.map((header) => [header, truthAdjusted[header] ?? ""]));
  if (!normalized.observation_id) throw new Error("Observation is missing observation_id");
  const delta = finite(normalized.release_date_delta_days);
  if (delta != null && Math.abs(delta) > 31) {
    normalized.date_conflict_flag = "true";
    normalized.date_conflict_details = `Canonical date ${normalized.canonical_release_date} differs from source date ${normalized.source_release_date} by ${delta} days; both are preserved.`;
  } else {
    normalized.date_conflict_flag = "false";
  }
  normalized.source_record_json = typeof truthAdjusted.source_record_json === "string" ? truthAdjusted.source_record_json : JSON.stringify(truthAdjusted.source_record_json ?? {});
  observations.push(normalized);
  return normalized;
};

const epochNumeric = (raw, field) => finite(raw[field]);
const observationFromEpoch = (record) => {
  const reverseEci = eciNamesByEpochLabel.get(record.model) || [];
  const eciName = reverseEci.length === 1 ? reverseEci[0] : "";
  const epochLink = makeEpochLink({ label: record.model, method: "Epoch self identity", confidence: "high", linkLevel: "checkpoint", status: epochByLabel.get(record.model).length > 1 ? "self_duplicate_label" : "self" });
  const identity = identityFields({ source: "Epoch", model: record.model, organization: record.organization, sourceReleaseDate: record.releaseDate, epochLink, eciName, sourceIdentity: record.model });
  return addObservation({
    observation_id: `epoch:${String(record.sourceRow).padStart(5, "0")}`,
    source: "Epoch", record_type: "model", dataset: "Epoch AI All AI Models", snapshot_date: "2026-07-31",
    source_locator: `CSV row ${record.sourceRow}`, source_model_name: record.model, source_organization: record.organization,
    source_provider: record.organization, model_level_include: "true", model_level_selection_reason: "source row",
    ...identity, source_release_date: record.releaseDate, source_release_date_precision: "day",
    matched_epoch_model: record.model, epoch_match_status: epochLink.status, epoch_match_method: epochLink.method,
    epoch_match_confidence: epochLink.confidence, epoch_link_level: epochLink.linkLevel,
    epoch_candidate_count: epochByLabel.get(record.model).length, epoch_source_rows: epochByLabel.get(record.model).map((row) => row.sourceRow).join("|"),
    epoch_release_dates: unique(epochByLabel.get(record.model).map((row) => row.releaseDate)).join("|"), epoch_base_model: record.baseModel,
    matched_eci_model: eciName, eci_match_status: eciName ? "reverse exact checkpoint bridge" : "not in ECI",
    eci_match_confidence: eciName ? "high" : "none", benchmark_name: "Epoch model metadata",
    total_parameters_b: record.parametersB, parameter_value_source: record.parametersB == null ? "" : "Epoch Parameters",
    epoch_parameters_b: record.parametersB, epoch_training_compute_flop: epochNumeric(record.raw, "Training compute (FLOP)"),
    epoch_training_dataset_size_total: epochNumeric(record.raw, "Training dataset size (total)"),
    epoch_training_time_hours: epochNumeric(record.raw, "Training time (hours)"),
    epoch_hardware_quantity: epochNumeric(record.raw, "Hardware quantity"),
    epoch_post_training_compute_flop: epochNumeric(record.raw, "Post-training compute (FLOP)"),
    epoch_model_accessibility: record.raw["Model accessibility"], epoch_open_model_weights: record.raw["Open model weights?"],
    source_url: record.link, source_record_json: record.raw,
  });
};
const epochObservationBySourceRow = new Map();
for (const record of epochRecords) {
  epochObservationBySourceRow.set(record.sourceRow, observationFromEpoch(record));
}

const epochViewAuditRows = [];
for (const match of epochViewMatches) {
  const {
    view, record, selected, candidates, currentSelected, currentCandidates,
    status, selectedDisagreements, currentSelectedDisagreements,
  } = match;
  // Archive and live all_ai row numbers are not stable across snapshots.  Only
  // bridge through an independently re-matched current exact key; never reuse
  // a historical row number as a live row identifier.
  const matchedObservation = currentSelected
    ? epochObservationBySourceRow.get(currentSelected.sourceRow)
    : null;
  const raw = record.raw;
  const sourceSlug = view.prefix.replace(/^epoch_/, "").replaceAll("_", "-");
  const fallbackCheckpoint = selected
    ? `checkpoint:epoch-archive:${providerTag(selected.raw.Organization)}:${slug(selected.model)}:${selected.releaseDate}`
    : `checkpoint:${sourceSlug}:${providerTag(raw.Organization)}:${slug(raw.Model)}:${String(record.sourceRow).padStart(5, "0")}`;
  const canonicalCheckpointId = matchedObservation?.canonical_checkpoint_id || fallbackCheckpoint;
  const canonicalBaseId = matchedObservation?.canonical_base_id || `base:${fallbackCheckpoint}`;
  const parameterB = finite(raw.Parameters) == null ? null : finite(raw.Parameters) / 1e9;
  const observationId = `${view.prefix}:${String(record.sourceRow).padStart(5, "0")}`;
  addObservation({
    observation_id: observationId,
    source: view.source,
    record_type: "source_view",
    dataset: `Epoch AI ${view.entry}`,
    snapshot_date: "2026-07-17",
    source_locator: `${view.entry} row ${record.sourceRow} inside ai_models.zip`,
    source_model_name: raw.Model,
    source_organization: raw.Organization,
    source_provider: raw.Organization,
    model_level_include: "false",
    model_level_selection_reason: "Correlated Epoch source view retained losslessly; excluded from independent model-level evidence.",
    canonical_checkpoint_id: canonicalCheckpointId,
    canonical_display_name: matchedObservation?.canonical_display_name || raw.Model,
    canonical_base_id: canonicalBaseId,
    canonical_release_date: matchedObservation?.canonical_release_date || raw["Publication date"],
    canonical_release_date_source: matchedObservation?.canonical_release_date_source || `${view.entry} source date`,
    source_release_date: raw["Publication date"],
    source_release_date_precision: "day",
    release_date_delta_days: dayDelta(matchedObservation?.canonical_release_date || raw["Publication date"], raw["Publication date"]),
    matched_epoch_model: selected?.model || "",
    epoch_match_status: status,
    epoch_match_method: selected ? "exact Model + Publication date + Organization; minimum shared-field disagreement breaks duplicate keys" : "exact-key review",
    epoch_match_confidence: selected ? "high" : "none",
    epoch_link_level: selected ? "checkpoint_source_view" : candidates.length ? "candidate_only" : "none",
    epoch_candidate_count: candidates.length,
    epoch_alternative_candidates: candidates.map((candidate) => `${candidate.model} [all_ai row ${candidate.sourceRow}]`).join("|"),
    epoch_source_rows: selected ? String(selected.sourceRow) : candidates.map((candidate) => candidate.sourceRow).join("|"),
    epoch_release_dates: unique(candidates.map((candidate) => candidate.releaseDate)).join("|"),
    epoch_base_model: selected?.baseModel || raw["Base model"] || "",
    matched_eci_model: matchedObservation?.matched_eci_model || "",
    eci_match_status: matchedObservation?.matched_eci_model ? "inherited from matched all_ai record" : "not in ECI",
    eci_match_confidence: matchedObservation?.matched_eci_model ? "high" : "none",
    benchmark_name: "Epoch archive source view",
    total_parameters_b: parameterB,
    parameter_value_source: parameterB == null ? "" : `${view.entry} Parameters`,
    epoch_parameters_b: parameterB,
    epoch_training_compute_flop: epochNumeric(raw, "Training compute (FLOP)"),
    epoch_training_dataset_size_total: epochNumeric(raw, "Training dataset size (total)") ?? epochNumeric(raw, "Training dataset size"),
    epoch_training_time_hours: epochNumeric(raw, "Training time (hours)"),
    epoch_hardware_quantity: epochNumeric(raw, "Hardware quantity"),
    epoch_post_training_compute_flop: epochNumeric(raw, "Post-training compute (FLOP)"),
    epoch_model_accessibility: raw["Model accessibility"],
    epoch_open_model_weights: raw["Open model weights?"],
    source_url: raw.Link,
    notes: `Correlated Epoch archive view. Match status: ${status}. Shared-field disagreements to selected all_ai row: ${selectedDisagreements === "" ? "n/a" : selectedDisagreements}.`,
    source_record_json: raw,
  });
  epochViewAuditRows.push({
    view: view.entry,
    view_source: view.source,
    view_row: record.sourceRow,
    observation_id: observationId,
    model: raw.Model,
    organization: raw.Organization,
    publication_date: raw["Publication date"],
    match_status: status,
    candidate_count: candidates.length,
    matched_all_ai_row: selected?.sourceRow || "",
    matched_all_ai_model: selected?.model || "",
    selected_shared_field_disagreements: selectedDisagreements,
    all_candidate_rows: candidates.map((candidate) => candidate.sourceRow).join("|"),
    matched_current_all_ai_row: currentSelected?.sourceRow || "",
    current_candidate_count: currentCandidates.length,
    current_selected_shared_field_disagreements: currentSelectedDisagreements,
    notes: selected ? "Correlated source view; do not count as independent evidence." : "Preserved without forced checkpoint match.",
  });
}

for (const record of eciRecords) {
  const epochLink = eciEpochLinks.get(record.model);
  const aggregate = aggregateEpochLink(epochLink);
  const regression = eciRegressionByName.get(record.model);
  const identity = identityFields({ source: "ECI", model: record.display, organization: record.organization, sourceReleaseDate: record.releaseDate, epochLink, eciName: record.model, sourceIdentity: record.model });
  const sourceParameter = regression?.totalParametersB ?? null;
  const exactParameter = exactParameterOverrides.get(record.model);
  addObservation({
    observation_id: `eci:${String(record.sourceRow).padStart(4, "0")}`,
    source: "ECI", record_type: "model", dataset: "Official reproduced ECI + legacy Regression Data parameter enrichment", snapshot_date: "2026-07-31",
    source_locator: `epoch_eci_reproduced_scores_2026-07-31.csv row ${record.sourceRow}${regression ? `; legacy Regression Data row ${regression.sourceRow}` : ""}`,
    source_model_name: record.model, source_organization: record.organization, source_provider: record.organization,
    model_level_include: "true", model_level_selection_reason: "source row", ...identity,
    source_release_date: record.releaseDate, source_release_date_precision: "day",
    matched_epoch_model: epochLink.label, epoch_match_status: epochLink.status, epoch_match_method: epochLink.method,
    epoch_match_confidence: epochLink.confidence, epoch_link_level: epochLink.linkLevel,
    epoch_candidate_count: epochLink.label ? epochLink.rows.length : epochLink.candidates.length,
    epoch_alternative_candidates: epochLink.candidates.join("|"), epoch_source_rows: aggregate.sourceRows.join("|"),
    epoch_release_dates: aggregate.releaseDates.join("|"), epoch_base_model: aggregate.baseModels.join("|"),
    matched_eci_model: record.model, eci_match_status: "self", eci_match_confidence: "high",
    benchmark_name: "ECI", eci_score: record.eci, eci_ci_low: record.eciLow, eci_ci_high: record.eciHigh,
    total_parameters_b: exactParameter?.totalB ?? sourceParameter ?? aggregate.parametersB,
    active_parameters_b: exactParameter?.activeB ?? "",
    parameter_value_source: exactParameter?.source ?? (sourceParameter != null ? "ECI Regression Data" : aggregate.parametersB != null ? "matched Epoch Parameters" : ""),
    epoch_parameters_b: aggregate.parametersB, eci_regression_total_parameters_b: sourceParameter,
    epoch_model_accessibility: aggregate.rows?.[0]?.raw?.["Model accessibility"] || "",
    source_url: exactParameter?.sourceUrl ?? regression?.sourceUrl ?? "",
    notes: [regression?.matchAssumption, exactParameter && aggregate.parametersB != null && exactParameter.totalB !== aggregate.parametersB ? `Primary exact total ${exactParameter.totalB}B supersedes Epoch's rounded ${aggregate.parametersB}B` : "", aggregate.parameterConflict ? `Epoch parameter conflict: ${aggregate.parameterConflict}` : ""].filter(Boolean).join("; "),
    source_record_json: { graph_data: record.raw, regression_data: regression?.raw || null },
  });
}

for (const record of eciLegacyOnlyRecords) {
  const epochLink = exactEpochLink(record.model) || makeEpochLink({
    status: "not_in_current_epoch",
    method: "retired legacy ECI row; no forced current Epoch match",
    confidence: "none",
    linkLevel: "historical_source_view",
  });
  const aggregate = aggregateEpochLink(epochLink);
  const regression = eciRegressionByName.get(record.model);
  const sourceReleaseDate = isoFromExcel(record.raw.Date);
  const checkpointId = epochLink.label
    ? `checkpoint:epoch:${slug(epochLink.label)}`
    : `checkpoint:eci-legacy:${providerTag(record.organization)}:${slug(record.model)}`;
  addObservation({
    observation_id: `eci_legacy:${String(record.sourceRow).padStart(4, "0")}`,
    source: "ECI Legacy View",
    record_type: "source_view",
    dataset: "Supplied ECI workbook — retired July 18 aggregate",
    snapshot_date: "2026-07-18",
    source_locator: `ECI Graph Data row ${record.sourceRow}${regression ? `; Regression Data row ${regression.sourceRow}` : ""}`,
    source_model_name: record.model,
    source_organization: record.organization,
    source_provider: record.organization,
    model_level_include: "false",
    model_level_selection_reason: "Retired historical aggregate preserved losslessly; excluded from the current ECI likelihood.",
    canonical_checkpoint_id: checkpointId,
    canonical_display_name: record.display,
    canonical_base_id: `base:${checkpointId}`,
    canonical_release_date: sourceReleaseDate,
    canonical_release_date_source: "legacy ECI workbook date",
    source_release_date: sourceReleaseDate,
    source_release_date_precision: "day",
    release_date_delta_days: 0,
    matched_epoch_model: epochLink.label,
    epoch_match_status: epochLink.status,
    epoch_match_method: epochLink.method,
    epoch_match_confidence: epochLink.confidence,
    epoch_link_level: epochLink.linkLevel,
    epoch_candidate_count: epochLink.rows.length,
    epoch_source_rows: aggregate.sourceRows.join("|"),
    epoch_release_dates: aggregate.releaseDates.join("|"),
    epoch_base_model: aggregate.baseModels.join("|"),
    matched_eci_model: "",
    eci_match_status: "retired from current ECI snapshot",
    eci_match_confidence: "historical exact",
    benchmark_name: "ECI legacy aggregate",
    eci_score: finite(record.raw.ECI),
    eci_ci_low: finite(record.raw["ECI CI low"]),
    eci_ci_high: finite(record.raw["ECI CI high"]),
    total_parameters_b: regression?.totalParametersB ?? aggregate.parametersB,
    parameter_value_source: regression?.totalParametersB != null ? "Legacy ECI Regression Data" : aggregate.parametersB != null ? "matched Epoch Parameters" : "",
    epoch_parameters_b: aggregate.parametersB,
    eci_regression_total_parameters_b: regression?.totalParametersB,
    epoch_model_accessibility: aggregate.rows?.[0]?.raw?.["Model accessibility"] || record.accessibility,
    source_url: regression?.sourceUrl || "",
    notes: "Historical source view only; retained to satisfy lossless snapshot preservation after Epoch retired this aggregate.",
    source_record_json: { graph_data: record.raw, regression_data: regression?.raw || null },
  });
}

for (const record of eciComponentRecords) {
  const aggregateEci = eciByName.get(record.model);
  const epochLink = eciEpochLinks.get(record.model);
  const aggregateEpoch = aggregateEpochLink(epochLink);
  const regression = eciRegressionByName.get(record.model);
  const identity = identityFields({
    source: "ECI Component", model: aggregateEci.display, organization: aggregateEci.organization,
    sourceReleaseDate: record.releaseDate, epochLink, eciName: record.model, sourceIdentity: record.model,
  });
  const sourceParameter = regression?.totalParametersB ?? null;
  const exactParameter = exactParameterOverrides.get(record.model);
  addObservation({
    observation_id: `eci_component:${String(record.sourceRow).padStart(5, "0")}`,
    source: "ECI Component", record_type: "benchmark_measurement", dataset: "Epoch ECI component benchmarks",
    snapshot_date: "2026-07-31", source_locator: `epoch_eci_benchmarks_2026-07-31.csv row ${record.sourceRow}`,
    source_model_name: record.model, source_configuration: record.modelVersion,
    source_organization: aggregateEci.organization, source_provider: aggregateEci.organization,
    model_level_include: "false",
    model_level_selection_reason: "Component benchmark record retained losslessly; aggregate ECI row is the model-level capability observation.",
    ...identity, source_release_date: record.releaseDate, source_release_date_precision: "day",
    matched_epoch_model: epochLink.label, epoch_match_status: epochLink.status, epoch_match_method: epochLink.method,
    epoch_match_confidence: epochLink.confidence, epoch_link_level: epochLink.linkLevel,
    epoch_candidate_count: epochLink.label ? epochLink.rows.length : epochLink.candidates.length,
    epoch_alternative_candidates: epochLink.candidates.join("|"), epoch_source_rows: aggregateEpoch.sourceRows.join("|"),
    epoch_release_dates: aggregateEpoch.releaseDates.join("|"), epoch_base_model: aggregateEpoch.baseModels.join("|"),
    matched_eci_model: record.model, eci_match_status: "exact aggregate-model key", eci_match_confidence: "high",
    benchmark_name: record.benchmark, eci_score: aggregateEci.eci,
    eci_component_performance: record.performance, eci_component_optimized: record.optimized,
    total_parameters_b: exactParameter?.totalB ?? sourceParameter ?? aggregateEpoch.parametersB,
    active_parameters_b: exactParameter?.activeB ?? "",
    parameter_value_source: exactParameter?.source ?? (sourceParameter != null ? "matched ECI Regression Data" : aggregateEpoch.parametersB != null ? "matched Epoch Parameters" : ""),
    epoch_parameters_b: aggregateEpoch.parametersB, eci_regression_total_parameters_b: sourceParameter,
    epoch_model_accessibility: aggregateEpoch.rows?.[0]?.raw?.["Model accessibility"] || "",
    source_url: "https://epoch.ai/data/eci_benchmarks.csv",
    notes: [
      `Benchmark release date: ${record.benchmarkReleaseDate || "not supplied"}`,
      `Optimized evaluation: ${record.optimized}`,
      record.raw.source ? `Source: ${record.raw.source}` : "",
    ].filter(Boolean).join("; "),
    source_record_json: record.raw,
  });
}

const aaAuditRows = [];
for (const record of aaRecords) {
  const epochLink = matchAaToEpoch(record);
  const aggregate = aggregateEpochLink(epochLink);
  const eciMatch = matchAaToEci(record);
  const exactParameter = exactParameterOverrides.get(record.baseName);
  const identity = identityFields({
    source: "AA", model: record.baseName, organization: record.creator, epochLink,
    eciName: eciMatch.name, sourceIdentity: `${record.creator}:${record.baseName}`,
  });
  const observation = addObservation({
    observation_id: `aa:${String(record.sourceRow).padStart(4, "0")}`,
    source: "AA", record_type: "model_configuration", dataset: "Artificial Analysis Intelligence Leaderboard",
    snapshot_date: "2026-07-17", source_locator: `attachment line ${record.sourceRow}`, source_model_name: record.displayName,
    source_configuration: record.configuration, source_organization: record.creator, source_provider: record.provider,
    model_level_include: String(record.selectedForModelLevel), model_level_selection_reason: record.selectionReason,
    ...identity, source_release_date_precision: "not supplied by AA",
    matched_epoch_model: epochLink.label, epoch_match_status: epochLink.status, epoch_match_method: epochLink.method,
    epoch_match_confidence: epochLink.confidence, epoch_link_level: epochLink.linkLevel,
    epoch_candidate_count: epochLink.label ? epochLink.rows.length : epochLink.candidates.length,
    epoch_alternative_candidates: epochLink.candidates.join("|"), epoch_source_rows: aggregate.sourceRows.join("|"),
    epoch_release_dates: aggregate.releaseDates.join("|"), epoch_base_model: aggregate.baseModels.join("|"),
    matched_eci_model: eciMatch.name, eci_match_status: eciMatch.status, eci_match_confidence: eciMatch.confidence,
    benchmark_name: "Artificial Analysis Intelligence Index", aa_intelligence_index: parseAaNumber(record.scoreRaw),
    aa_score_raw: record.scoreRaw, aa_score_qualifier: record.scoreRaw.includes("*") ? "asterisk" : record.scoreRaw === "--" ? "not reported" : "",
    aa_context_window_tokens: parseMagnitude(record.contextRaw), aa_context_window_raw: record.contextRaw,
    aa_blended_price_usd_per_million_tokens: parseAaNumber(record.priceRaw), aa_price_raw: record.priceRaw,
    aa_output_speed_tokens_per_second: parseAaNumber(record.speedRaw), aa_speed_raw: record.speedRaw,
    aa_latency_first_chunk_seconds: parseAaNumber(record.latencyRaw), aa_latency_raw: record.latencyRaw,
    aa_total_response_seconds: parseAaNumber(record.totalResponseRaw), aa_total_response_raw: record.totalResponseRaw,
    total_parameters_b: exactParameter?.totalB ?? aggregate.parametersB,
    active_parameters_b: exactParameter?.activeB ?? "",
    parameter_value_source: exactParameter?.source ?? (aggregate.parametersB == null ? "" : `matched Epoch ${epochLink.linkLevel} Parameters`),
    epoch_parameters_b: aggregate.parametersB,
    source_url: portablePath(aaPath),
    notes: [epochLink.notes, aggregate.parameterConflict ? `Epoch parameter conflict: ${aggregate.parameterConflict}` : ""].filter(Boolean).join("; "),
    source_record_json: record.raw,
  });
  aaAuditRows.push({
    aa_observation_id: observation.observation_id, aa_source_line: record.sourceRow, aa_display_name: record.displayName,
    aa_base_name: record.baseName, aa_configuration: record.configuration, aa_creator: record.creator, aa_provider: record.provider,
    aa_score_raw: record.scoreRaw, aa_score_numeric: parseAaNumber(record.scoreRaw), selected_for_model_level: String(record.selectedForModelLevel),
    selection_reason: record.selectionReason, matched_epoch_model: epochLink.label, epoch_match_status: epochLink.status,
    epoch_match_method: epochLink.method, epoch_match_confidence: epochLink.confidence, epoch_link_level: epochLink.linkLevel,
    epoch_label_row_count: epochLink.rows.length, epoch_source_rows: aggregate.sourceRows.join("|"),
    epoch_parameter_b: aggregate.parametersB, alternative_candidates: epochLink.candidates.join("|"),
    matched_eci_model: eciMatch.name, eci_match_status: eciMatch.status,
    canonical_checkpoint_id: observation.canonical_checkpoint_id, canonical_base_id: observation.canonical_base_id,
    notes: observation.notes,
  });
}

const aaDetailedObservationId = (record) => `aa_detailed:${String(record.sourceRow).padStart(4, "0")}`;
const aaDetailedModality = (sourceRecord) => {
  const inputs = ["Text", "Image", "Speech", "Video"]
    .filter((name) => sourceRecord[`inputModality${name}`])
    .map((name) => name.toLowerCase());
  const outputs = ["Text", "Image", "Speech", "Video"]
    .filter((name) => sourceRecord[`outputModality${name}`])
    .map((name) => name.toLowerCase());
  return inputs.length || outputs.length ? `${inputs.join("+") || "unknown"}->${outputs.join("+") || "unknown"}` : "";
};
for (const record of aaDetailedRecords) {
  const raw = record.raw;
  const sourceIdentity = `aa-detailed:${raw.model_id}`;
  addObservation({
    observation_id: aaDetailedObservationId(record),
    source: "AA Detailed View",
    record_type: "source_view",
    dataset: "Artificial Analysis Detailed Model Snapshot",
    snapshot_date: "2026-07-31",
    source_locator: `aa_detailed_model_signals_2026-07-31.csv row ${record.sourceRow}`,
    source_model_name: raw.name,
    source_configuration: raw.slug,
    source_organization: raw.creator_name,
    source_provider: raw.performance_provider_name || raw.creator_name,
    model_level_include: "false",
    model_level_selection_reason: "Correlated detailed AA source view retained losslessly; excluded from independent model-level evidence.",
    canonical_checkpoint_id: `checkpoint:${sourceIdentity}`,
    canonical_display_name: raw.name,
    canonical_base_id: `base:${sourceIdentity}`,
    canonical_release_date: raw.release_date,
    canonical_release_date_source: "AA detailed snapshot release_date",
    source_release_date: raw.release_date,
    source_release_date_precision: "day",
    release_date_delta_days: raw.release_date ? 0 : "",
    epoch_match_status: "not_joined_correlated_source_view",
    epoch_match_method: "source UUID identity only; no cross-source join asserted",
    epoch_match_confidence: "none",
    epoch_link_level: "none",
    eci_match_status: "not_joined_correlated_source_view",
    eci_match_confidence: "none",
    benchmark_name: "Artificial Analysis detailed model signals",
    aa_intelligence_index: finite(raw.intelligence_index),
    aa_score_raw: raw.intelligence_index,
    aa_score_qualifier: raw.intelligence_index_estimated === "True" ? "estimated" : "",
    aa_context_window_tokens: finite(raw.context_window_tokens),
    aa_context_window_raw: raw.context_window_tokens,
    aa_blended_price_usd_per_million_tokens: finite(raw.price_blended_7_2_1_usd_per_mtoken),
    aa_price_raw: raw.price_blended_7_2_1_usd_per_mtoken,
    aa_output_speed_tokens_per_second: finite(raw.median_output_speed_tps),
    aa_speed_raw: raw.median_output_speed_tps,
    aa_latency_first_chunk_seconds: finite(raw.median_time_to_first_chunk_seconds),
    aa_latency_raw: raw.median_time_to_first_chunk_seconds,
    total_parameters_b: finite(raw.parameters_b),
    active_parameters_b: finite(raw.active_parameters_b),
    parameter_value_source: raw.parameters_b === "" ? "" : "Artificial Analysis public model metadata (correlated source view)",
    reasoning: raw.is_reasoning,
    modality: aaDetailedModality(record.sourceRecord),
    source_url: raw.source_page_url,
    notes: `Source UUID ${raw.model_id}; source slug ${raw.slug}; source-only checkpoint/base identity prevents an unreviewed cross-source merge; normalized CSV row remains available at the source locator.`,
    source_record_json: raw.source_record_json,
  });
}

const noCotAll = [...noCotFrontier, ...noCotOpen];
for (let index = 0; index < noCotAll.length; index += 1) {
  const record = noCotAll[index];
  const dateOverride = noCotDateByModel.get(record.model);
  const epochLink = matchNoCotToEpoch(record.model);
  const aggregate = aggregateEpochLink(epochLink);
  const eciName = paperToEci.get(record.model) || "";
  const identity = identityFields({
    source: "No-CoT", model: record.model, organization: record.organization, sourceReleaseDate: record.releaseDate,
    epochLink, eciName, sourceIdentity: record.model,
  });
  const isFrontier = record.suite === "frontier";
  const paperParameters = isFrontier ? null : record.totalB;
  addObservation({
    observation_id: `nocot:${String(index + 1).padStart(3, "0")}`,
    source: "No-CoT", record_type: "model", dataset: "No-CoT Task-Completion Time Horizons (arXiv:2606.07157v3)",
    snapshot_date: "2026-07-17", source_locator: isFrontier ? "tab:horizons-per-model" : "tab:open-source-models + tab:open-weight-precise-time-horizons",
    source_model_name: record.model, source_organization: record.organization, source_provider: record.organization,
    model_level_include: "true", model_level_selection_reason: "source model row", ...identity,
    source_release_date: record.releaseDate, source_release_date_precision: "month (paper source)",
    matched_epoch_model: epochLink.label, epoch_match_status: epochLink.status, epoch_match_method: epochLink.method,
    epoch_match_confidence: epochLink.confidence, epoch_link_level: epochLink.linkLevel,
    epoch_candidate_count: epochLink.label ? epochLink.rows.length : epochLink.candidates.length,
    epoch_alternative_candidates: epochLink.candidates.join("|"), epoch_source_rows: aggregate.sourceRows.join("|"),
    epoch_release_dates: aggregate.releaseDates.join("|"), epoch_base_model: aggregate.baseModels.join("|"),
    matched_eci_model: eciName, eci_match_status: eciName ? "manual paper/ECI checkpoint alias" : "not in ECI",
    eci_match_confidence: eciName ? "high" : "none", benchmark_name: "No-CoT 50% task-completion horizon",
    nocot_time_horizon_minutes: isFrontier ? record.time.estimate : record.horizon.point,
    nocot_time_horizon_median_minutes: isFrontier ? "" : record.horizon.median,
    nocot_time_horizon_ci_low_minutes: isFrontier ? record.time.low : record.horizon.low,
    nocot_time_horizon_ci_high_minutes: isFrontier ? record.time.high : record.horizon.high,
    nocot_token_horizon_tokens: isFrontier ? record.tokens.estimate : "",
    nocot_token_horizon_ci_low_tokens: isFrontier ? record.tokens.low : "",
    nocot_token_horizon_ci_high_tokens: isFrontier ? record.tokens.high : "",
    total_parameters_b: paperParameters ?? aggregate.parametersB,
    active_parameters_b: isFrontier ? "" : record.activeB,
    parameter_value_source: paperParameters != null ? "No-CoT paper table" : aggregate.parametersB != null ? "matched Epoch Parameters" : "",
    epoch_parameters_b: aggregate.parametersB, layers: isFrontier ? "" : record.layers,
    architecture: isFrontier ? "" : record.architecture, reasoning: isFrontier ? "" : record.reasoning,
    modality: isFrontier ? "" : record.modality, source_url: "https://arxiv.org/abs/2606.07157",
    notes: [
      aggregate.parameterConflict ? `Epoch parameter conflict: ${aggregate.parameterConflict}` : "",
      dateOverride ? `Day-level date only: ${dateOverride.date_source}; parameter join remains prohibited.` : "",
    ].filter(Boolean).join("; "),
    source_record_json: record.raw,
  });
}

addObservation({
  observation_id: "nocot:law", source: "No-CoT", record_type: "scaling_law",
  dataset: "No-CoT Task-Completion Time Horizons (arXiv:2606.07157v3)", snapshot_date: "2026-07-17",
  source_locator: "paper text + LaTeX macros", source_model_name: "__scaling_law__", model_level_include: "false",
  model_level_selection_reason: "non-model law", benchmark_name: "No-CoT open-weight scaling and frontier trend laws",
  source_url: "https://arxiv.org/abs/2606.07157", source_record_json: noCotLaw,
});

const metrOrganizations = (id) => id.startsWith("claude") ? "Anthropic" : id.startsWith("gemini") ? "Google DeepMind" : "OpenAI";
for (let index = 0; index < metrRecords.length; index += 1) {
  const record = metrRecords[index];
  const epochLink = matchMetrToEpoch(record);
  const aggregate = aggregateEpochLink(epochLink);
  const eciName = metrEciRules.get(record.id) || "";
  const identity = identityFields({
    source: "METR", model: record.id, organization: metrOrganizations(record.id), sourceReleaseDate: record.date,
    epochLink, eciName, sourceIdentity: record.id,
  });
  addObservation({
    observation_id: `metr:${String(index + 1).padStart(3, "0")}`,
    source: "METR", record_type: "model", dataset: record.benchmark, snapshot_date: "2026-07-18",
    source_locator: `official benchmark_results_1_1.yaml result ${record.id}`, source_model_name: record.id,
    source_organization: metrOrganizations(record.id), source_provider: metrOrganizations(record.id),
    model_level_include: "true", model_level_selection_reason: "source model row", ...identity,
    source_release_date: record.date, source_release_date_precision: "day",
    matched_epoch_model: epochLink.label, epoch_match_status: epochLink.status, epoch_match_method: epochLink.method,
    epoch_match_confidence: epochLink.confidence, epoch_link_level: epochLink.linkLevel,
    epoch_candidate_count: epochLink.label ? epochLink.rows.length : epochLink.candidates.length,
    epoch_alternative_candidates: epochLink.candidates.join("|"), epoch_source_rows: aggregate.sourceRows.join("|"),
    epoch_release_dates: aggregate.releaseDates.join("|"), epoch_base_model: aggregate.baseModels.join("|"),
    matched_eci_model: eciName, eci_match_status: eciName ? "manual METR/ECI checkpoint alias" : "not in ECI",
    eci_match_confidence: eciName ? "high" : "none", benchmark_name: record.benchmark,
    metr_average_score: record.average, metr_is_sota: String(record.sota),
    metr_p50_horizon_minutes: record.p50, metr_p50_ci_low_minutes: record.p50Low, metr_p50_ci_high_minutes: record.p50High,
    metr_p80_horizon_minutes: record.p80, metr_p80_ci_low_minutes: record.p80Low, metr_p80_ci_high_minutes: record.p80High,
    total_parameters_b: aggregate.parametersB, parameter_value_source: aggregate.parametersB == null ? "" : `matched Epoch ${epochLink.linkLevel} Parameters`,
    epoch_parameters_b: aggregate.parametersB, source_url: metrOfficialUrl,
    notes: [`Primary scaffold family: ${record.scaffold}; all ${record.scaffolds.length} official scaffold entries retained`, aggregate.parameterConflict ? `Epoch parameter conflict: ${aggregate.parameterConflict}` : ""].filter(Boolean).join("; "),
    source_record_json: { ...record.raw, parsed_scaffolds: record.scaffolds },
  });
}

addObservation({
  observation_id: "metr:law", source: "METR", record_type: "trend_law", dataset: "METR-Horizon-v1.1",
  snapshot_date: "2026-07-18", source_locator: "official benchmark_results_1_1.yaml top-level trend metadata", source_model_name: "__trend_law__",
  model_level_include: "false", model_level_selection_reason: "non-model law", benchmark_name: "METR Horizon doubling time",
  source_url: metrOfficialUrl, notes: `${metrLaw.exclusionRule}; official asset exactly matches the earlier 26-row user snapshot`,
  source_record_json: { ...metrLaw, long_tasks_version: METR_LONG_TASKS_VERSION, swaa_version: METR_SWAA_VERSION },
});

if (new Set(observations.map((row) => row.observation_id)).size !== observations.length) throw new Error("Observation IDs are not unique");
if (observations.length !== 8476) throw new Error(`Expected 8476 observations; found ${observations.length}`);

// ------------------------- Long measurement table -------------------------

const measurementHeaders = [
  "measurement_id", "observation_id", "canonical_checkpoint_id", "canonical_base_id", "source", "source_model_name",
  "source_configuration", "matched_epoch_model", "epoch_link_level", "benchmark_name", "metric_name", "value",
  "value_raw", "unit", "ci_low", "ci_high", "measurement_notes",
];
const measurements = [];
let measurementCounter = 0;
const observationById = new Map(observations.map((row) => [row.observation_id, row]));
const addMeasurement = ({ observationId, metric, value, raw = "", unit = "", low = "", high = "", notes = "" }) => {
  if (value == null || value === "" || !Number.isFinite(Number(value))) return;
  const observation = observationById.get(observationId);
  if (!observation) throw new Error(`Measurement points to missing observation: ${observationId}`);
  measurementCounter += 1;
  measurements.push({
    measurement_id: `m:${String(measurementCounter).padStart(6, "0")}`,
    observation_id: observationId,
    canonical_checkpoint_id: observation.canonical_checkpoint_id,
    canonical_base_id: observation.canonical_base_id,
    source: observation.source,
    source_model_name: observation.source_model_name,
    source_configuration: observation.source_configuration,
    matched_epoch_model: observation.matched_epoch_model,
    epoch_link_level: observation.epoch_link_level,
    benchmark_name: observation.benchmark_name,
    metric_name: metric,
    value: Number(value),
    value_raw: raw === "" ? value : raw,
    unit,
    ci_low: low,
    ci_high: high,
    measurement_notes: notes,
  });
};

const epochNumericMetrics = new Map([
  ["Citations", "count"], ["Parameters", "parameters"], ["Training compute (FLOP)", "FLOP"],
  ["Training dataset size (total)", "source-native total"], ["Training time (hours)", "hours"], ["Epochs", "epochs"],
  ["Finetune compute (FLOP)", "FLOP"], ["Hardware quantity", "accelerators"], ["Hardware utilization (MFU)", "fraction"],
  ["Batch size", "source-native"], ["Training compute lower bound", "FLOP"], ["Training compute upper bound", "FLOP"],
  ["Training chip-hours", "chip-hours"], ["Training compute cost (2023 USD)", "2023 USD"],
  ["Training power draw (W)", "watts"], ["Post-training compute (FLOP)", "FLOP"], ["Hardware utilization (HFU)", "fraction"],
]);
for (const record of epochRecords) {
  const observationId = `epoch:${String(record.sourceRow).padStart(5, "0")}`;
  for (const [field, unit] of epochNumericMetrics) {
    addMeasurement({ observationId, metric: `epoch.${field}`, value: finite(record.raw[field]), raw: record.raw[field], unit });
  }
}

const epochViewNumericMetrics = new Map([
  ["Parameters", "parameters"], ["Training compute (FLOP)", "FLOP"],
  ["Training dataset size (total)", "source-native total"], ["Training dataset size", "source-native total"],
  ["Training time (hours)", "hours"], ["Epochs", "epochs"], ["Finetune compute (FLOP)", "FLOP"],
  ["Hardware quantity", "accelerators"], ["Hardware utilization (MFU)", "fraction"],
  ["Batch size", "source-native"], ["Training chip-hours", "chip-hours"],
  ["Training compute cost (2023 USD)", "2023 USD"], ["Training compute cost (cloud)", "USD"],
  ["Training compute cost (upfront)", "USD"], ["Training power draw (W)", "watts"],
  ["Post-training compute (FLOP)", "FLOP"], ["Hardware utilization (HFU)", "fraction"],
  ["Base model compute", "FLOP"], ["FLOP/$", "FLOP per USD"], ["Power per GPU", "watts"],
]);
for (const match of epochViewMatches) {
  const { view, record, status } = match;
  const observationId = `${view.prefix}:${String(record.sourceRow).padStart(5, "0")}`;
  for (const [field, unit] of epochViewNumericMetrics) {
    addMeasurement({
      observationId,
      metric: `${view.prefix}.${field}`,
      value: finite(record.raw[field]),
      raw: record.raw[field],
      unit,
      notes: `Correlated Epoch archive source view (${status}); exclude from independent likelihood counts.`,
    });
  }
}

for (const record of eciRecords) {
  const observationId = `eci:${String(record.sourceRow).padStart(4, "0")}`;
  addMeasurement({ observationId, metric: "eci.score", value: record.eci, raw: record.eci, unit: "ECI points", low: record.eciLow, high: record.eciHigh, notes: "95% confidence interval from ECI Graph Data" });
  const regression = eciRegressionByName.get(record.model);
  addMeasurement({ observationId, metric: "eci_regression.total_parameters", value: regression?.totalParametersB, raw: regression?.totalParametersB, unit: "billions of parameters", notes: regression?.matchAssumption || "" });
}

for (const record of eciLegacyOnlyRecords) {
  const observationId = `eci_legacy:${String(record.sourceRow).padStart(4, "0")}`;
  const regression = eciRegressionByName.get(record.model);
  addMeasurement({ observationId, metric: "eci_legacy.score", value: finite(record.raw.ECI), raw: record.raw.ECI, unit: "ECI points", low: finite(record.raw["ECI CI low"]), high: finite(record.raw["ECI CI high"]), notes: "Retired July 18 ECI aggregate; historical source view only." });
  addMeasurement({ observationId, metric: "eci_legacy_regression.total_parameters", value: regression?.totalParametersB, raw: regression?.totalParametersB, unit: "billions of parameters", notes: regression?.matchAssumption || "" });
}

for (const record of eciComponentRecords) {
  const observationId = `eci_component:${String(record.sourceRow).padStart(5, "0")}`;
  addMeasurement({
    observationId,
    metric: "eci_component.performance",
    value: record.performance,
    raw: record.raw.performance,
    unit: "random-baseline-corrected proportion",
    notes: `Optimized=${record.optimized}; source=${record.raw.source || "not supplied"}`,
  });
}

for (const record of aaRecords) {
  if (!record.selectedForModelLevel) continue;
  const observationId = `aa:${String(record.sourceRow).padStart(4, "0")}`;
  addMeasurement({ observationId, metric: "aa.intelligence_index", value: parseAaNumber(record.scoreRaw), raw: record.scoreRaw, unit: "index points", notes: record.scoreRaw.includes("*") ? "AA asterisk qualifier preserved" : "" });
  addMeasurement({ observationId, metric: "aa.context_window", value: parseMagnitude(record.contextRaw), raw: record.contextRaw, unit: "tokens" });
  addMeasurement({ observationId, metric: "aa.blended_price", value: parseAaNumber(record.priceRaw), raw: record.priceRaw, unit: "USD per million tokens" });
  addMeasurement({ observationId, metric: "aa.output_speed", value: parseAaNumber(record.speedRaw), raw: record.speedRaw, unit: "tokens per second" });
  addMeasurement({ observationId, metric: "aa.latency_first_chunk", value: parseAaNumber(record.latencyRaw), raw: record.latencyRaw, unit: "seconds" });
  addMeasurement({ observationId, metric: "aa.total_response_time", value: parseAaNumber(record.totalResponseRaw), raw: record.totalResponseRaw, unit: "seconds" });
}

const aaDetailedNumericMetrics = new Map([
  ["reasoning_tokens_setting", "tokens"],
  ["parameters_b", "billions of parameters"],
  ["active_parameters_b", "billions of parameters"],
  ["context_window_tokens", "tokens"],
  ["intelligence_index", "index points"],
  ["coding_index", "index points"],
  ["agentic_index", "index points"],
  ["gdpval", "source-native score"],
  ["gdpval_normalized", "proportion"],
  ["tau_banking", "proportion"],
  ["terminal_bench_v2_1", "proportion"],
  ["scicode", "proportion"],
  ["hle", "proportion"],
  ["gpqa", "proportion"],
  ["critpt", "proportion"],
  ["omniscience", "source-native score"],
  ["omniscience_accuracy", "proportion"],
  ["omniscience_hallucination_rate", "proportion"],
  ["lcr", "proportion"],
  ["ifbench", "proportion"],
  ["apex_agents", "proportion"],
  ["automation_bench_partial_score", "proportion"],
  ["enterprise_ops_gym", "proportion"],
  ["mmmu_pro", "proportion"],
  ["price_input_usd_per_mtoken", "USD per million tokens"],
  ["price_output_usd_per_mtoken", "USD per million tokens"],
  ["price_cache_hit_usd_per_mtoken", "USD per million tokens"],
  ["price_blended_7_2_1_usd_per_mtoken", "USD per million tokens"],
  ["intelligence_time_per_task_seconds", "seconds per task"],
  ["intelligence_cost_total_usd", "USD"],
  ["intelligence_cost_per_task_usd", "USD per task"],
  ["intelligence_input_tokens_total", "tokens"],
  ["intelligence_output_tokens_total", "tokens"],
  ["intelligence_answer_tokens_total", "tokens"],
  ["intelligence_reasoning_tokens_total", "tokens"],
  ["intelligence_output_tokens_per_task", "tokens per task"],
  ["intelligence_answer_tokens_per_task", "tokens per task"],
  ["intelligence_reasoning_tokens_per_task", "tokens per task"],
  ["median_output_speed_tps", "tokens per second"],
  ["median_time_to_first_chunk_seconds", "seconds"],
]);
for (const record of aaDetailedRecords) {
  const observationId = aaDetailedObservationId(record);
  for (const [field, unit] of aaDetailedNumericMetrics) {
    const qualifier = field === "intelligence_index"
      ? `; intelligence_index_estimated=${record.raw.intelligence_index_estimated}`
      : "";
    addMeasurement({
      observationId,
      metric: `aa_detailed.${field}`,
      value: finite(record.raw[field]),
      raw: record.raw[field],
      unit,
      notes: `Correlated AA detailed snapshot source view; exclude from independent likelihood counts${qualifier}.`,
    });
  }
}

for (let index = 0; index < noCotAll.length; index += 1) {
  const record = noCotAll[index];
  const observationId = `nocot:${String(index + 1).padStart(3, "0")}`;
  if (record.suite === "frontier") {
    addMeasurement({ observationId, metric: "nocot.time_horizon_50pct", value: record.time.estimate, raw: record.time.raw, unit: "minutes", low: record.time.low, high: record.time.high, notes: record.time.floor ? "measurement-floor estimate" : "" });
    addMeasurement({ observationId, metric: "nocot.reasoning_token_horizon_50pct", value: record.tokens.estimate, raw: record.tokens.raw, unit: "o3-mini reasoning tokens", low: record.tokens.low, high: record.tokens.high, notes: record.tokens.floor ? "measurement-floor estimate" : "" });
  } else {
    addMeasurement({ observationId, metric: "nocot.time_horizon_50pct_point", value: record.horizon.point, unit: "minutes", low: record.horizon.low, high: record.horizon.high });
    addMeasurement({ observationId, metric: "nocot.time_horizon_50pct_bootstrap_median", value: record.horizon.median, unit: "minutes" });
    addMeasurement({ observationId, metric: "nocot.total_parameters", value: record.totalB, unit: "billions of parameters" });
    addMeasurement({ observationId, metric: "nocot.active_parameters", value: record.activeB, unit: "billions of parameters" });
    addMeasurement({ observationId, metric: "nocot.layers", value: record.layers, unit: "layers" });
  }
}
addMeasurement({ observationId: "nocot:law", metric: "nocot.total_parameters_per_horizon_doubling", value: noCotLaw.totalParametersMultiplier, unit: "multiplicative factor" });
addMeasurement({ observationId: "nocot:law", metric: "nocot.active_parameters_per_horizon_doubling", value: noCotLaw.activeParametersMultiplier, unit: "multiplicative factor" });
addMeasurement({ observationId: "nocot:law", metric: "nocot.layers_per_horizon_doubling", value: noCotLaw.layerMultiplier, unit: "multiplicative factor" });
addMeasurement({ observationId: "nocot:law", metric: "nocot.pretraining_flops_per_horizon_doubling", value: noCotLaw.pretrainingFlopsMultiplier, unit: "multiplicative factor" });
addMeasurement({ observationId: "nocot:law", metric: "nocot.frontier_time_horizon_doubling_time", value: noCotLaw.timeHorizonDoublingDays, unit: "days" });
addMeasurement({ observationId: "nocot:law", metric: "nocot.frontier_token_horizon_doubling_time", value: noCotLaw.tokenHorizonDoublingDays, unit: "days" });

for (let index = 0; index < metrRecords.length; index += 1) {
  const record = metrRecords[index];
  const observationId = `metr:${String(index + 1).padStart(3, "0")}`;
  addMeasurement({ observationId, metric: "metr.average_score", value: record.average, unit: "proportion" });
  addMeasurement({ observationId, metric: "metr.p50_horizon", value: record.p50, unit: "minutes", low: record.p50Low, high: record.p50High });
  addMeasurement({ observationId, metric: "metr.p80_horizon", value: record.p80, unit: "minutes", low: record.p80Low, high: record.p80High });
}
addMeasurement({ observationId: "metr:law", metric: "metr.horizon_doubling_time_all_time_stitched", value: metrLaw.allTimeDoublingDays, unit: "days", notes: metrLaw.exclusionRule });
addMeasurement({ observationId: "metr:law", metric: "metr.horizon_doubling_time_from_2023", value: metrLaw.from2023DoublingDays, unit: "days", low: metrLaw.from2023CiLowDays, high: metrLaw.from2023CiHighDays, notes: metrLaw.exclusionRule });

if (new Set(measurements.map((row) => row.measurement_id)).size !== measurements.length) throw new Error("Measurement IDs are not unique");
if (measurements.length !== 31155) throw new Error(`Expected 31155 measurements; found ${measurements.length}`);

// ------------------------- Output and manifest -------------------------

await writeCsv(observationsPath, observationHeaders, observations);
await writeCsv(measurementsPath, measurementHeaders, measurements);
const aaAuditHeaders = [
  "aa_observation_id", "aa_source_line", "aa_display_name", "aa_base_name", "aa_configuration", "aa_creator", "aa_provider",
  "aa_score_raw", "aa_score_numeric", "selected_for_model_level", "selection_reason", "matched_epoch_model",
  "epoch_match_status", "epoch_match_method", "epoch_match_confidence", "epoch_link_level", "epoch_label_row_count",
  "epoch_source_rows", "epoch_parameter_b", "alternative_candidates", "matched_eci_model", "eci_match_status",
  "canonical_checkpoint_id", "canonical_base_id", "notes",
];
await writeCsv(aaAuditPath, aaAuditHeaders, aaAuditRows);
const epochViewAuditHeaders = [
  "view", "view_source", "view_row", "observation_id", "model", "organization", "publication_date",
  "match_status", "candidate_count", "matched_all_ai_row", "matched_all_ai_model",
  "selected_shared_field_disagreements", "all_candidate_rows", "matched_current_all_ai_row",
  "current_candidate_count", "current_selected_shared_field_disagreements", "notes",
];
await writeCsv(epochViewAuditPath, epochViewAuditHeaders, epochViewAuditRows);

const sourceSpecs = [
  { source: "Epoch", path: epochPath, parsed: 3574, structure: "57 source columns", notes: `Five duplicated labels retained: ${duplicateEpochLabels.join(" | ")}` },
  {
    source: "Epoch AI models archive",
    path: epochArchivePath,
    parsed: epochViewMatches.length,
    structure: "137 frontier + 1,035 notable + 518 large-scale historical source-view records; all_ai entry is the frozen July 17 snapshot",
    notes: [`archive all_ai SHA-256 ${epochArchiveAllAiHash}`, `matches current snapshot=${epochArchiveMatchesCurrentSnapshot}`, ...epochArchiveEntries.map((entry) => `${entry.entry} SHA-256 ${entry.sha256}`)].join(" | "),
  },
  { source: "ECI", path: eciReproducedPath, parsed: 213, structure: "Official reproduced ECI + exact-name legacy Regression Data parameter enrichment", notes: "All 213 current ECI aggregates retained; the stale graph sheet supplies parameters only" },
  { source: "ECI Component", path: eciComponentPath, parsed: 2059, structure: "14 source columns; 213 models × sparse coverage across 54 benchmarks", notes: "All model/benchmark rows retained; every model matches one reproduced aggregate exactly" },
  { source: "Artificial Analysis", path: aaPath, parsed: 274, structure: "11-line leaderboard records", notes: "All raw rows retained; model-level measurement table selects highest score for identical display names" },
  { source: "AA Detailed View", path: aaDetailedPath, parsed: 587, structure: "65 normalized columns plus one verbatim source_record_json object per unique source UUID", notes: "Current July 31 detailed snapshot retained as a correlated source view; all rows excluded from independent model-level evidence" },
  { source: "No-CoT", path: latexPath, parsed: 50, structure: "14 frontier + 35 open-weight model rows + 1 scaling-law row", notes: "Parsed directly from arxiv_version.tex" },
  { source: "No-CoT exact-date overrides", path: noCotDateOverridePath, parsed: 4, structure: "four previously month-only paper checkpoints", notes: "Date-only provenance; no parameter identities added" },
  { source: "No-CoT exact-date metadata", path: noCotDateMetadataPath, parsed: 1, structure: "JSON collection and integrity audit", notes: "Epoch selectors, first-party Qwen commit, hashes, and join prohibitions" },
  { source: "Qwen first-party commit history", path: qwenDateRawPath, parsed: 21, structure: "verbatim gzip JSON from Hugging Face commits API", notes: "Initial commit c9051e5f23e735fd6549f86b616377617848a621 at 2025-07-28T07:31:28Z" },
  { source: "Frontier primary evidence", path: frontierPrimaryEvidencePath, parsed: 5, structure: "normalized first-party measurements, identity constraints, and caveats", notes: "Direct Sol no-CoT values; official Fable/Mythos shared weights; Opus fallback kept separate" },
  { source: "Frontier primary evidence metadata", path: frontierPrimaryMetadataPath, parsed: 1, structure: "JSON provenance and integrity audit", notes: "Official-source URLs, hashes, claim locations, and no-promotion policies" },
  { source: "OpenAI GPT-5.6 system card", path: openAiGpt56SystemCardPath, parsed: 1, structure: "deterministic gzip of official HTML", notes: "Verbatim source for the 3.6-minute Sol and 2.3-minute GPT-5.5 no-CoT statement" },
  { source: "Anthropic Fable/Mythos claim manifest", path: anthropicFableMythosClaimsPath, parsed: 5, structure: "hash-pinned PDF claim manifest", notes: "Shared weights p.12; fallback p.14; underlying-model control p.131; GPQA saturation p.258" },
  { source: "Kimi K3 release evidence", path: k3ReleaseEvidencePath, parsed: 1, structure: "hash-pinned first-party architecture and release evidence", notes: "Exact 2.78T total and 104.2B active parameters; paired architecture fact, not two independent votes" },
  { source: "Open-model parameter truth reconciliation", path: openModelParameterTruthPath, parsed: openModelParameterTruth.records.length, structure: "hash-pinned narrow canonical overlays with raw values preserved", notes: "Kimi K2 1.04T/32.6B report truth and MiniMax M2.5/M2.7 exact safetensors totals; checkpoint identities remain distinct" },
  { source: "METR official signals", path: metrPath, parsed: 26, structure: "15 columns; 26 model rows with full scaffold arrays", notes: "Deterministically normalized from the official primary YAML; every source scalar retained" },
  { source: "METR official raw asset", path: metrRawPath, parsed: 27, structure: "verbatim YAML; 26 result blocks + 1 top-level trend law", notes: `First-party asset ${metrOfficialUrl}; long-tasks ${METR_LONG_TASKS_VERSION}; SWAA ${METR_SWAA_VERSION}` },
  { source: "METR official metadata", path: metrMetadataPath, parsed: 1, structure: "JSON provenance, inventory, trend law, and file hashes", notes: "Fail-closed schema parser; 114 full scaffold entries retained" },
  { source: "METR primary-source audit", path: metrAuditPath, parsed: 1, structure: "JSON official/legacy reconciliation and losslessness audit", notes: "PASS: 26/26 legacy rows match every common field; zero mismatches" },
  { source: "METR legacy crosscheck", path: metrLegacyPath, parsed: 26, structure: "14-column historical user-supplied CSV", notes: "Retained only as an exact crosscheck; no longer an authoritative pipeline input" },
];
const existingSourcePaths = new Set(sourceSpecs.map((spec) => spec.path));
for (const source of openModelParameterTruth.source_files) {
  const sourcePath = `${workDir}/${source.path}`;
  if (existingSourcePaths.has(sourcePath)) continue;
  existingSourcePaths.add(sourcePath);
  sourceSpecs.push({
    source: "Open-model parameter truth evidence",
    path: sourcePath,
    parsed: 1,
    structure: source.role,
    notes: `Pinned evidence for the narrow parameter-truth overlay; ${source.url}`,
  });
}
const manifestRows = [];
for (const spec of sourceSpecs) {
  const stat = await fs.stat(spec.path);
  manifestRows.push({
    source: spec.source, path: portablePath(spec.path), size_bytes: stat.size, sha256: await sha256(spec.path),
    records_parsed: spec.parsed, structure: spec.structure, notes: spec.notes,
  });
}
const manifestHeaders = ["source", "path", "size_bytes", "sha256", "records_parsed", "structure", "notes"];
await writeCsv(manifestPath, manifestHeaders, manifestRows);

const countBy = (rows, key) => Object.fromEntries([...rows.reduce((map, row) => map.set(row[key], (map.get(row[key]) || 0) + 1), new Map()).entries()].sort());
const summary = {
  observations: observations.length,
  measurements: measurements.length,
  observationsBySource: countBy(observations, "source"),
  aaRows: aaRecords.length,
  aaUniqueDisplayNames: aaRowsByDisplayName.size,
  aaSelectedForModelLevel: aaRecords.filter((row) => row.selectedForModelLevel).length,
  aaUniqueBaseCreatorPairs: aaUniqueBaseCreator.length,
  aaDetailedRows: aaDetailedRecords.length,
  aaDetailedMeasurements: measurements.filter((row) => row.source === "AA Detailed View").length,
  aaDetailedModelLevelIncluded: observations.filter((row) => row.source === "AA Detailed View" && row.model_level_include === "true").length,
  aaEpochStatus: countBy(aaAuditRows, "epoch_match_status"),
  aaEpochLinkLevel: countBy(aaAuditRows, "epoch_link_level"),
  duplicateEpochLabels,
  epochArchiveViews: Object.fromEntries(epochArchiveEntries.map((entry) => [entry.entry, {
    records: entry.records.length,
    sha256: entry.sha256,
    matched: epochViewMatches.filter((row) => row.view.entry === entry.entry && row.selected).length,
    ambiguous: epochViewMatches.filter((row) => row.view.entry === entry.entry && row.status === "ambiguous_exact_key").length,
    unmatched: epochViewMatches.filter((row) => row.view.entry === entry.entry && row.status === "not_in_all_ai_view").length,
  }])),
  outputs: Object.fromEntries(Object.entries({
    observationsPath,
    measurementsPath,
    aaAuditPath,
    epochViewAuditPath,
    manifestPath,
    metrPath,
    metrRawPath,
    metrMetadataPath,
    metrAuditPath,
  }).map(([key, value]) => [key, portablePath(value)])),
};
await fs.writeFile(`${outputDir}/unified_model_data_summary_compute_enriched_2026-07-17.json`, `${JSON.stringify(summary, null, 2)}\n`, "utf8");
console.log(JSON.stringify(summary, null, 2));
