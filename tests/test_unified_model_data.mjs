import fs from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";

const workDir = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const portableLocalPath = (value) => {
  const resolved = path.resolve(value);
  if (resolved === workDir || resolved.startsWith(`${workDir}${path.sep}`)) {
    return path.relative(workDir, resolved).split(path.sep).join("/");
  }
  return value;
};
const outputDir = `${workDir}/outputs/019f6c42-2d53-7743-ab07-6293e2618dd7`;
const observationsPath = `${outputDir}/unified_model_observations_compute_enriched_2026-07-17.csv`;
const measurementsPath = `${outputDir}/unified_model_measurements_long_compute_enriched_2026-07-17.csv`;
const aaAuditPath = `${outputDir}/aa_epoch_match_audit_compute_enriched_2026-07-17.csv`;
const epochViewAuditPath = `${outputDir}/epoch_archive_view_match_audit_2026-07-17.csv`;
const manifestPath = `${outputDir}/unified_model_source_manifest_compute_enriched_2026-07-17.csv`;
const metrPath = `${workDir}/sources/metr_horizon_official_signals_2026-07-18.csv`;
const epochPath = `${workDir}/sources/epoch_all_ai_models_2026-07-31.csv`;
const eciComponentPath = `${workDir}/sources/epoch_eci_benchmarks_2026-07-31.csv`;
const aaDetailedPath = `${workDir}/sources/aa_detailed_model_signals_2026-07-31.csv`;
const resultsPath = `${outputDir}/unified_model_data_test_results_compute_enriched_2026-07-17.csv`;

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
        } else quoted = false;
      } else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { row.push(cell); cell = ""; }
    else if (ch === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; }
    else if (ch !== "\r") cell += ch;
  }
  if (quoted) throw new Error("Unclosed CSV quote");
  if (cell.length || row.length) { row.push(cell); rows.push(row); }
  return rows;
};

const readObjects = async (path) => {
  const matrix = parseCsv(await fs.readFile(path, "utf8"));
  const headers = matrix[0];
  const rows = matrix.slice(1).filter((row) => row.some((cell) => cell !== ""));
  for (let index = 0; index < rows.length; index += 1) {
    if (rows[index].length !== headers.length) throw new Error(`${path} row ${index + 2} has ${rows[index].length}/${headers.length} fields`);
  }
  return { headers, rows: rows.map((row) => Object.fromEntries(headers.map((header, index) => [header, row[index]]))) };
};

const { headers: observationHeaders, rows: observations } = await readObjects(observationsPath);
const { headers: measurementHeaders, rows: measurements } = await readObjects(measurementsPath);
const { rows: aaAudit } = await readObjects(aaAuditPath);
const { rows: epochViewAudit } = await readObjects(epochViewAuditPath);
const { rows: manifest } = await readObjects(manifestPath);
const { rows: metrSnapshot } = await readObjects(metrPath);
const { headers: eciComponentSourceHeaders, rows: eciComponentSource } = await readObjects(eciComponentPath);
const { headers: aaDetailedSourceHeaders, rows: aaDetailedSource } = await readObjects(aaDetailedPath);

const bySource = Object.groupBy(observations, (row) => row.source);
const epoch = bySource.Epoch;
const eci = bySource.ECI;
const eciLegacyView = bySource["ECI Legacy View"];
const eciComponent = bySource["ECI Component"];
const aa = bySource.AA;
const aaDetailed = bySource["AA Detailed View"];
const noCot = bySource["No-CoT"];
const metr = bySource.METR;
const epochFrontierView = bySource["Epoch Frontier View"];
const epochNotableView = bySource["Epoch Notable View"];
const epochLargeScaleView = bySource["Epoch Large-Scale View"];
const observationById = new Map(observations.map((row) => [row.observation_id, row]));
const measurementsByObservation = Object.groupBy(measurements, (row) => row.observation_id);

const results = [];
const check = (test, condition, expected, actual, details = "") => {
  results.push({ test, status: condition ? "PASS" : "FAIL", expected: String(expected), actual: String(actual), details });
};
const equal = (test, actual, expected, details = "") => check(test, actual === expected, expected, actual, details);
const approx = (test, actual, expected, tolerance = 1e-9, details = "") => check(test, Math.abs(Number(actual) - expected) <= tolerance, `${expected} ± ${tolerance}`, actual, details);
const unique = (values) => [...new Set(values)];
const numeric = (value) => value === "" ? null : Number(value);

equal("Observation schema column count", observationHeaders.length, 92);
equal("Measurement schema column count", measurementHeaders.length, 17);
equal("Unified observation count", observations.length, 8476);
equal("Epoch observation count", epoch.length, 3574);
check("Current Epoch observations have current snapshot provenance", epoch.every((row) => row.dataset === "Epoch AI All AI Models" && row.snapshot_date === "2026-07-31"), true, unique(epoch.map((row) => `${row.dataset}|${row.snapshot_date}`)).join(" | "));
equal("ECI observation count", eci.length, 213);
equal("Retired ECI legacy source-view count", eciLegacyView.length, 2);
equal("Retired ECI legacy source-view exact set", eciLegacyView.map((row) => row.source_model_name).sort().join("|"), ["DeepSeek-V3.1", "Kimi K2 (Sep 2025)"].sort().join("|"));
check("Retired ECI legacy rows are excluded from current model-level evidence", eciLegacyView.every((row) => row.record_type === "source_view" && row.model_level_include === "false"), true, "all historical rows excluded");
equal("ECI component observation count", eciComponent.length, 2059);
equal("AA raw observation count", aa.length, 274);
equal("AA detailed source-view observation count", aaDetailed.length, 587);
equal("No-CoT observation count including law", noCot.length, 50);
equal("METR observation count including law", metr.length, 27);
equal("Epoch frontier source-view count", epochFrontierView.length, 137);
equal("Epoch notable source-view count", epochNotableView.length, 1035);
equal("Epoch large-scale source-view count", epochLargeScaleView.length, 518);
equal("Unique observation IDs", new Set(observations.map((row) => row.observation_id)).size, observations.length);
equal("Unique measurement IDs", new Set(measurements.map((row) => row.measurement_id)).size, measurements.length);
check("Every measurement references an observation", measurements.every((row) => observationById.has(row.observation_id)), true, measurements.filter((row) => !observationById.has(row.observation_id)).length);
check("All source_record_json values parse", observations.every((row) => { try { JSON.parse(row.source_record_json); return true; } catch { return false; } }), true, "all parsed");

const epochRawKeyCounts = epoch.map((row) => Object.keys(JSON.parse(row.source_record_json)).length);
check("All Epoch raw records preserve 57 fields", epochRawKeyCounts.every((count) => count === 57), true, `${Math.min(...epochRawKeyCounts)}..${Math.max(...epochRawKeyCounts)}`);
const epochHash = createHash("sha256").update(await fs.readFile(epochPath)).digest("hex");
equal("Epoch raw snapshot SHA-256", epochHash, "0d1fcfc497cccc1079068a1ec3031e30e97ad5c8e6f5b5d43baceac3778ba579");
equal("Source manifest row count", manifest.length, 25);
equal("Epoch archive source-view audit row count", epochViewAudit.length, 1690);
check("Every Epoch archive source-view row is excluded from independent model-level evidence", [...epochFrontierView, ...epochNotableView, ...epochLargeScaleView].every((row) => row.model_level_include === "false"), true, "all source-view rows excluded");
equal("All Epoch frontier view rows match all_ai exactly", epochViewAudit.filter((row) => row.view === "frontier_ai_models.csv" && row.match_status === "matched_exact_key" && row.selected_shared_field_disagreements === "0").length, 137);
equal("Epoch notable view rows matched", epochViewAudit.filter((row) => row.view === "notable_ai_models.csv" && row.matched_all_ai_row).length, 1035);
equal("Epoch large-scale view unmatched records", epochViewAudit.filter((row) => row.view === "large_scale_ai_models.csv" && row.match_status === "not_in_all_ai_view").length, 3);
const archiveRowsWithCurrentBridge = epochViewAudit.filter((row) => row.matched_current_all_ai_row);
check(
  "Epoch archive views bridge by exact identity rather than unstable row number",
  archiveRowsWithCurrentBridge.every((audit) => {
    const viewObservation = observationById.get(audit.observation_id);
    const currentObservation = observationById.get(`epoch:${String(audit.matched_current_all_ai_row).padStart(5, "0")}`);
    return viewObservation && currentObservation
      && viewObservation.source_model_name === currentObservation.source_model_name
      && viewObservation.source_organization === currentObservation.source_organization
      && viewObservation.source_release_date === currentObservation.source_release_date
      && viewObservation.canonical_checkpoint_id === currentObservation.canonical_checkpoint_id;
  }),
  true,
  `${archiveRowsWithCurrentBridge.length} current exact-key bridges`,
);
const nemotronEpoch = epoch.find((row) => row.source_model_name === "Nemotron 3 Ultra" && row.source_release_date === "2026-06-04");
const nemotronArchiveViews = [...epochNotableView, ...epochLargeScaleView].filter((row) => row.source_model_name === "Nemotron 3 Ultra" && row.source_release_date === "2026-06-04");
equal("Nemotron archive views retained as Nemotron", nemotronArchiveViews.length, 2);
check("Nemotron archive views never inherit Terra identity", nemotronArchiveViews.every((row) => row.canonical_checkpoint_id === nemotronEpoch?.canonical_checkpoint_id && row.canonical_display_name === "Nemotron 3 Ultra"), true, nemotronArchiveViews.map((row) => `${row.canonical_display_name}|${row.canonical_checkpoint_id}`).join(" ; "));

equal("ECI component source schema column count", eciComponentSourceHeaders.length, 14);
equal("ECI component source row count", eciComponentSource.length, 2059);
equal("ECI component unique model count", new Set(eciComponent.map((row) => row.source_model_name)).size, 213);
equal("ECI component unique benchmark count", new Set(eciComponent.map((row) => row.benchmark_name)).size, 54);
equal("ECI component unique model/benchmark pairs", new Set(eciComponent.map((row) => `${row.source_model_name}\u001f${row.benchmark_name}`)).size, 2059);
check("Every ECI component model matches one aggregate ECI row", eciComponent.every((row) => eci.some((aggregate) => aggregate.source_model_name === row.matched_eci_model && aggregate.source_model_name === row.source_model_name)), true, "all exact");
check("ECI component rows are excluded from independent model-level counts", eciComponent.every((row) => row.model_level_include === "false"), true, "all excluded");
check("ECI component raw records preserve 14 fields", eciComponent.every((row) => Object.keys(JSON.parse(row.source_record_json)).length === 14), true, "all raw rows preserved");
const eciComponentMeasurements = measurements.filter((row) => row.source === "ECI Component");
equal("ECI component measurement count", eciComponentMeasurements.length, 2059);
check("Every ECI component observation emits exactly one performance measurement", eciComponent.every((row) => (measurementsByObservation[row.observation_id] || []).length === 1 && measurementsByObservation[row.observation_id][0].metric_name === "eci_component.performance"), true, "one per source row");
const eciComponentHash = createHash("sha256").update(await fs.readFile(eciComponentPath)).digest("hex");
equal("ECI component snapshot SHA-256", eciComponentHash, "d7cad7a8595347a62a2f832205aae579e371f05afe6ab08bc9506631e38c70d1");

const epochLabels = new Set(epoch.map((row) => row.source_model_name));
const epochLabelCounts = Object.groupBy(epoch, (row) => row.source_model_name);
const duplicatedEpochLabels = Object.entries(epochLabelCounts).filter(([, rows]) => rows.length > 1).map(([label]) => label).sort();
equal("Epoch duplicated label count", duplicatedEpochLabels.length, 5);
equal("Epoch duplicated labels exact set", duplicatedEpochLabels.join("|"), ["Eurus-2-7B-PRIME", "GLM-5", "Gemini 3.1 Pro", "SAM 3", "Tulu 3 (Tülu 3) 70B"].sort().join("|"));

equal("AA audit row count", aaAudit.length, 274);
const aaByDisplay = Object.groupBy(aa, (row) => row.source_model_name);
const duplicateAaEntries = Object.entries(aaByDisplay).filter(([, rows]) => rows.length > 1);
equal("AA unique display names", Object.keys(aaByDisplay).length, 241);
equal("AA duplicated display-name count", duplicateAaEntries.length, 33);
equal("AA rows covered by duplicated display names", duplicateAaEntries.reduce((sum, [, rows]) => sum + rows.length, 0), 66);
equal("AA selected model-level row count", aa.filter((row) => row.model_level_include === "true").length, 241);
check("Exactly one selected row per AA display name", Object.values(aaByDisplay).every((rows) => rows.filter((row) => row.model_level_include === "true").length === 1), true, "one per display name");
check("Selected duplicate AA row has highest score", duplicateAaEntries.every(([, rows]) => {
  const selected = rows.find((row) => row.model_level_include === "true");
  const score = (row) => numeric(row.aa_intelligence_index) ?? -Infinity;
  return score(selected) === Math.max(...rows.map(score));
}), true, "highest score, first source row breaks ties");
equal("AA unique base/creator decision pairs", new Set(aa.map((row) => `${row.source_model_name.replace(/\s*\((?:max|xhigh|high|medium|low|minimal|non-reasoning(?:,\s*(?:high|low effort))?|with fallback)\)\s*$/i, "")}\u001f${row.source_organization}`)).size, 191);
check("No unreviewed AA identities", aa.every((row) => row.epoch_match_status !== "UNREVIEWED"), true, aa.filter((row) => row.epoch_match_status === "UNREVIEWED").length);
check("Every selected AA Epoch label exists", aa.filter((row) => row.matched_epoch_model).every((row) => epochLabels.has(row.matched_epoch_model)), true, "all labels found");
check("AA resolved Epoch-link coverage is at least 159 raw rows", aa.filter((row) => row.matched_epoch_model).length >= 159, ">=159", aa.filter((row) => row.matched_epoch_model).length);
equal("AA explicitly ambiguous Epoch-candidate rows", aa.filter((row) => row.epoch_match_status === "ambiguous_epoch_candidate").length, 6);
check("AA duplicate-label statuses are explicit", aa.filter((row) => row.matched_epoch_model && epochLabelCounts[row.matched_epoch_model].length > 1).every((row) => row.epoch_match_status.includes("duplicate_label")), true, "explicit duplicate status");

equal("AA detailed source schema column count", aaDetailedSourceHeaders.length, 65);
equal("AA detailed source row count", aaDetailedSource.length, 587);
equal("AA detailed source unique model IDs", new Set(aaDetailedSource.map((row) => row.model_id)).size, 587);
equal("AA detailed source unique slugs", new Set(aaDetailedSource.map((row) => row.slug)).size, 587);
check("AA detailed rows are correlated source views excluded from model-level evidence", aaDetailed.every((row) => (
  row.record_type === "source_view"
  && row.model_level_include === "false"
  && row.model_level_selection_reason.includes("Correlated detailed AA source view")
)), true, "all 587 excluded");
check("AA detailed observations carry current snapshot provenance", aaDetailed.every((row) => (
  row.dataset === "Artificial Analysis Detailed Model Snapshot"
  && row.snapshot_date === "2026-07-31"
)), true, unique(aaDetailed.map((row) => `${row.dataset}|${row.snapshot_date}`)).join(" | "));
check("AA detailed source-only identities are unique and make no cross-source join assertion", aaDetailed.every((row) => (
  row.canonical_checkpoint_id.startsWith("checkpoint:aa-detailed:")
  && row.canonical_base_id.startsWith("base:aa-detailed:")
  && row.matched_epoch_model === ""
  && row.matched_eci_model === ""
  && row.epoch_match_status === "not_joined_correlated_source_view"
  && row.eci_match_status === "not_joined_correlated_source_view"
)), true, "source UUID identity only");
const aaDetailedSourceById = new Map(aaDetailedSource.map((row) => [row.model_id, row]));
check("AA detailed source_record_json is preserved byte-for-byte for every row", aaDetailed.every((row) => {
  const raw = JSON.parse(row.source_record_json);
  const source = aaDetailedSourceById.get(raw.id);
  return source
    && row.source_record_json === source.source_record_json
    && row.source_model_name === source.name
    && row.source_configuration === source.slug
    && row.source_url === source.source_page_url;
}), true, "587 exact raw JSON strings and normalized identity fields");
check("AA detailed parameter fields preserve source values or expose raw values beside a pinned truth overlay", aaDetailed.every((row) => {
  const raw = JSON.parse(row.source_record_json);
  const source = aaDetailedSourceById.get(raw.id);
  if (!row.parameter_truth_id) {
    return numeric(row.total_parameters_b) === numeric(source.parameters_b)
      && numeric(row.active_parameters_b) === numeric(source.active_parameters_b);
  }
  return numeric(row.raw_total_parameters_b) === numeric(source.parameters_b)
    && numeric(row.raw_active_parameters_b) === numeric(source.active_parameters_b)
    && row.parameter_truth_basis !== "";
}), true, "raw source values retained beside narrow canonical overlays");
const aaDetailedHash = createHash("sha256").update(await fs.readFile(aaDetailedPath)).digest("hex");
equal("AA detailed snapshot SHA-256", aaDetailedHash, "c00553aa052e8e5fd40bf84637bc17663e407b5f6d348ddb1140c2df48b39404");
const aaDetailedManifest = manifest.find((row) => row.source === "AA Detailed View");
check("Source manifest contains exact AA detailed view", aaDetailedManifest
  && aaDetailedManifest.path === portableLocalPath(aaDetailedPath)
  && aaDetailedManifest.records_parsed === "587"
  && aaDetailedManifest.sha256 === aaDetailedHash
  && aaDetailedManifest.notes.includes("excluded from independent model-level evidence"), true, aaDetailedManifest ? JSON.stringify(aaDetailedManifest) : "missing");

const aaMeasurements = measurements.filter((row) => row.source === "AA");
const excludedAaIds = new Set(aa.filter((row) => row.model_level_include === "false").map((row) => row.observation_id));
check("Excluded duplicate AA rows emit no model-level measurements", aaMeasurements.every((row) => !excludedAaIds.has(row.observation_id)), true, aaMeasurements.filter((row) => excludedAaIds.has(row.observation_id)).length);
check("Every selected AA row with numeric data emits a measurement", aa.filter((row) => row.model_level_include === "true").every((row) => (measurementsByObservation[row.observation_id] || []).length > 0), true, "all selected rows measured");

const aaDetailedMeasurements = measurements.filter((row) => row.source === "AA Detailed View");
equal("AA detailed normalized numeric measurement count", aaDetailedMeasurements.length, 11637);
equal("AA detailed normalized metric count", new Set(aaDetailedMeasurements.map((row) => row.metric_name)).size, 40);
check("Every AA detailed measurement is explicitly labeled correlated", aaDetailedMeasurements.every((row) => row.measurement_notes.includes("Correlated AA detailed snapshot source view")), true, "all measurement rows carry dependence warning");
check("Every AA detailed source row emits normalized numeric measurements", aaDetailed.every((row) => (measurementsByObservation[row.observation_id] || []).length >= 2), true, "context window and reasoning setting guarantee at least two source-view metrics");
equal("AA detailed total-parameter measurement coverage", aaDetailedMeasurements.filter((row) => row.metric_name === "aa_detailed.parameters_b").length, 347);
equal("AA detailed active-parameter measurement coverage", aaDetailedMeasurements.filter((row) => row.metric_name === "aa_detailed.active_parameters_b").length, 167);
equal("AA detailed intelligence-index measurement coverage", aaDetailedMeasurements.filter((row) => row.metric_name === "aa_detailed.intelligence_index").length, 574);
const aaDetailedK3 = aaDetailed.find((row) => row.source_configuration === "kimi-k3");
approx("AA detailed K3 source-view total remains source-specific 2.8T", aaDetailedK3.total_parameters_b, 2800, 1e-12);
approx("AA detailed K3 source-view active count remains source-specific 104B", aaDetailedK3.active_parameters_b, 104, 1e-12);
const aaDetailedOpus5 = aaDetailed.find((row) => row.source_configuration === "claude-opus-5");
approx("AA detailed Opus 5 intelligence index", aaDetailedOpus5.aa_intelligence_index, 60.6918740157091, 1e-12);

const aaScore = (name) => numeric(aa.find((row) => row.source_model_name === name)?.aa_intelligence_index);
equal("AA Fable score", aaScore("Claude Fable 5 (with fallback)"), 60);
equal("AA Sol max score", aaScore("GPT-5.6 Sol (max)"), 59);
equal("AA Kimi K3 score", aaScore("Kimi K3"), 57);
equal("AA Opus 4.8 max score", aaScore("Claude Opus 4.8 (max)"), 56);
equal("AA GPT-5.5 xhigh score", aaScore("GPT-5.5 (xhigh)"), 55);
equal("AA Opus 4.7 max score", aaScore("Claude Opus 4.7 (max)"), 54);
equal("AA Terra max score", aaScore("GPT-5.6 Terra (max)"), 55);
equal("AA Sonnet 5 max score", aaScore("Claude Sonnet 5 (max)"), 53);
equal("AA Luna max score", aaScore("GPT-5.6 Luna (max)"), 51);
equal("AA Grok 4.5 high score", aaScore("Grok 4.5 (high)"), 54);

const fable = aa.find((row) => row.source_model_name === "Claude Fable 5 (with fallback)");
equal("Fable linked to regular Epoch checkpoint", fable.matched_epoch_model, "Claude Fable 5");
equal("Fable fallback link level explicit", fable.epoch_link_level, "checkpoint_system_configuration");
check("Fable fallback caveat retained", fable.notes.includes("fallback/system configuration"), true, fable.notes);
equal("Fable canonical exact date", fable.canonical_release_date, "2026-06-09");
equal("Kimi K3 manual exact date", aa.find((row) => row.source_model_name === "Kimi K3").canonical_release_date, "2026-07-16");
approx("Current reproduced K3 ECI", eci.find((row) => row.source_model_name === "Kimi K3").eci_score, 155.5939255295791, 1e-12);
approx("K3 exact primary total supersedes rounded Epoch value", eci.find((row) => row.source_model_name === "Kimi K3").total_parameters_b, 2780, 1e-12);
approx("K3 exact primary active parameters retained", eci.find((row) => row.source_model_name === "Kimi K3").active_parameters_b, 104.2, 1e-12);
approx("K3 rounded Epoch parameter field remains auditable", eci.find((row) => row.source_model_name === "Kimi K3").epoch_parameters_b, 2800, 1e-12);
approx("Current reproduced Opus 5 ECI", eci.find((row) => row.source_model_name === "Claude Opus 5").eci_score, 159.3778667882398, 1e-12);
check("Retired ECI aggregates are absent from current ECI", !eci.some((row) => ["DeepSeek-V3.1", "Kimi K2 (Sep 2025)"].includes(row.source_model_name)), true, "removed set absent from current likelihood");
check("Retired ECI aggregates remain losslessly preserved", eciLegacyView.every((row) => {
  const raw = JSON.parse(row.source_record_json);
  return raw.graph_data?.Model === row.source_model_name
    && (row.source_model_name === "DeepSeek-V3.1"
      ? raw.regression_data?.Model === row.source_model_name
      : raw.regression_data === null);
}), true, "both legacy graph rows and the one available legacy regression row embedded; missing Kimi regression row remains null");

const opus47 = aa.find((row) => row.source_model_name === "Claude Opus 4.7 (max)");
const opus48 = aa.find((row) => row.source_model_name === "Claude Opus 4.8 (max)");
equal("Opus 4.7/4.8 shared base ID", opus47.canonical_base_id, opus48.canonical_base_id);
const mythosEpoch = epoch.find((row) => row.source_model_name === "Claude Mythos 5");
equal("Official Fable/Mythos shared underlying-weight base ID", fable.canonical_base_id, mythosEpoch.canonical_base_id);
check("Fable/Mythos base stays distinct from Opus fallback", fable.canonical_base_id !== opus48.canonical_base_id, true, `${fable.canonical_base_id} != ${opus48.canonical_base_id}`);
const gpt5Eci = eci.find((row) => row.source_model_name === "GPT-5");
const gpt55Eci = eci.find((row) => row.source_model_name === "GPT-5.5");
equal("GPT-5 through GPT-5.5 shared base ID", gpt5Eci.canonical_base_id, gpt55Eci.canonical_base_id);

const checkpointAa = aa.filter((row) => ["checkpoint", "checkpoint_system_configuration"].includes(row.epoch_link_level));
check("Checkpoint-level AA links share Epoch canonical IDs", checkpointAa.every((row) => epoch.filter((candidate) => candidate.source_model_name === row.matched_epoch_model).some((candidate) => candidate.canonical_checkpoint_id === row.canonical_checkpoint_id)), true, "canonical IDs aligned");
const nonCheckpointAa = aa.filter((row) => ["base_only", "family_record"].includes(row.epoch_link_level));
check("Base/family AA links do not masquerade as Epoch checkpoints", nonCheckpointAa.every((row) => epoch.filter((candidate) => candidate.source_model_name === row.matched_epoch_model).every((candidate) => candidate.canonical_checkpoint_id !== row.canonical_checkpoint_id)), true, "checkpoint identity remains distinct");

equal("No-CoT model count", noCot.filter((row) => row.record_type === "model").length, 49);
equal("No-CoT frontier model count", noCot.filter((row) => row.record_type === "model" && row.source_locator === "tab:horizons-per-model").length, 14);
equal("No-CoT open-weight model count", noCot.filter((row) => row.record_type === "model" && row.source_locator.includes("tab:open-source-models")).length, 35);
equal("No-CoT scaling-law row count", noCot.filter((row) => row.record_type === "scaling_law").length, 1);
const noCotLawMeasurements = measurements.filter((row) => row.observation_id === "nocot:law");
equal("No-CoT law measurement count", noCotLawMeasurements.length, 6);
approx("No-CoT total-parameter horizon multiplier", noCotLawMeasurements.find((row) => row.metric_name === "nocot.total_parameters_per_horizon_doubling").value, 4.2);
approx("No-CoT active-parameter horizon multiplier", noCotLawMeasurements.find((row) => row.metric_name === "nocot.active_parameters_per_horizon_doubling").value, 2.1);
approx("No-CoT layer horizon multiplier", noCotLawMeasurements.find((row) => row.metric_name === "nocot.layers_per_horizon_doubling").value, 1.3);
approx("No-CoT pretraining-FLOP horizon multiplier", noCotLawMeasurements.find((row) => row.metric_name === "nocot.pretraining_flops_per_horizon_doubling").value, 3.1);
const noCotGpt55 = noCot.find((row) => row.source_model_name === "GPT-5.5");
equal("No-CoT GPT-5.5 source month preserved", noCotGpt55.source_release_date, "2026-04-01");
equal("No-CoT GPT-5.5 exact canonical date", noCotGpt55.canonical_release_date, "2026-04-23");
const noCotExactOverrides = new Map(noCot.filter((row) => ["GPT-2", "GPT-3", "GPT-3.5", "Qwen 3 30B-A3B (2507)"].includes(row.source_model_name)).map((row) => [row.source_model_name, row]));
equal("No-CoT GPT-2 exact date override", noCotExactOverrides.get("GPT-2").canonical_release_date, "2019-02-14");
equal("No-CoT GPT-3 exact date override", noCotExactOverrides.get("GPT-3").canonical_release_date, "2020-05-28");
equal("No-CoT GPT-3.5 exact date override", noCotExactOverrides.get("GPT-3.5").canonical_release_date, "2022-03-15");
equal("No-CoT Qwen 2507 exact date override", noCotExactOverrides.get("Qwen 3 30B-A3B (2507)").canonical_release_date, "2025-07-28");
check("No-CoT exact-date overrides preserve parameter nonjoin policy", [...noCotExactOverrides.values()].every((row) => row.notes.includes("parameter join remains prohibited")), true, [...noCotExactOverrides.values()].map((row) => row.notes).join(" | "));
equal("No-CoT Kimi K2-0905 checkpoint date", noCot.find((row) => row.source_model_name === "Kimi K2-0905").canonical_release_date, "2025-09-05");

const truthAdjusted = observations.filter((row) => row.parameter_truth_id);
check("Parameter-truth overlays preserve every raw total", truthAdjusted.every((row) => row.raw_total_parameters_b !== ""), true, truthAdjusted.length);
const kimiTruth = truthAdjusted.filter((row) => row.parameter_truth_id === "moonshot-kimi-k2-family-report-table-1");
check("All reconciled Kimi K2 rows use 1.04T", kimiTruth.length > 0 && kimiTruth.every((row) => Number(row.total_parameters_b) === 1040), true, unique(kimiTruth.map((row) => row.total_parameters_b)).join("|"));
check("Kimi K2 raw 1T shorthand remains visible", kimiTruth.some((row) => Number(row.raw_total_parameters_b) === 1000), true, unique(kimiTruth.map((row) => row.raw_total_parameters_b)).join("|"));
const minimaxTruth = truthAdjusted.filter((row) => row.parameter_truth_id.startsWith("minimax-m2-"));
check("MiniMax M2.5/M2.7 exact tensor totals reconcile", minimaxTruth.length > 0 && minimaxTruth.every((row) => Math.abs(Number(row.total_parameters_b) - 228.703644928) < 1e-12), true, unique(minimaxTruth.map((row) => row.total_parameters_b)).join("|"));
check("Parameter-truth manifest chain is complete", ["Open-model parameter truth reconciliation", "Open-model parameter truth evidence"].every((source) => manifest.some((row) => row.source === source)), true, manifest.filter((row) => row.source.startsWith("Open-model parameter truth")).length);

equal("METR model count", metr.filter((row) => row.record_type === "model").length, 26);
equal("METR trend-law row count", metr.filter((row) => row.record_type === "trend_law").length, 1);
equal("METR model snapshot row count", metrSnapshot.length, 26);
check("METR observations use first-party URL", metr.filter((row) => row.record_type === "model").every((row) => row.source_url === "https://metr.org/assets/benchmark_results_1_1.yaml"), true, "all official");
check("METR full scaffold arrays survive into raw records", metr.filter((row) => row.record_type === "model").every((row) => {
  const raw = JSON.parse(row.source_record_json);
  return Array.isArray(raw.parsed_scaffolds) && raw.parsed_scaffolds.length > 0 && JSON.parse(raw.scaffolds_json).length === raw.parsed_scaffolds.length;
}), true, "all arrays preserved twice: normalized JSON scalar and parsed array");
check("METR manifest contains complete primary provenance chain", ["METR official signals", "METR official raw asset", "METR official metadata", "METR primary-source audit", "METR legacy crosscheck"].every((source) => manifest.some((row) => row.source === source)), true, manifest.filter((row) => row.source.startsWith("METR")).map((row) => row.source).join("|"));
const metrLawMeasurements = measurements.filter((row) => row.observation_id === "metr:law");
equal("METR law measurement count", metrLawMeasurements.length, 2);
approx("METR all-time stitched doubling days", metrLawMeasurements.find((row) => row.metric_name === "metr.horizon_doubling_time_all_time_stitched").value, 187.778);
const from2023 = metrLawMeasurements.find((row) => row.metric_name === "metr.horizon_doubling_time_from_2023");
approx("METR from-2023 doubling days", from2023.value, 128.744);
approx("METR from-2023 CI low", from2023.ci_low, 104.428);
approx("METR from-2023 CI high", from2023.ci_high, 158.012);
approx("METR Mythos p50", metr.find((row) => row.source_model_name === "claude_mythos_preview_early_inspect").metr_p50_horizon_minutes, 1044.780145);
const davinci = metr.find((row) => row.source_model_name === "davinci_002");
equal("METR davinci identifier/date conflict remains ambiguous", davinci.epoch_match_status, "ambiguous_identifier_vs_release_date");
equal("METR davinci has no forced Epoch checkpoint", davinci.matched_epoch_model, "");
const largeDateConflicts = observations.filter((row) => row.release_date_delta_days && Math.abs(Number(row.release_date_delta_days)) > 31);
check("Large date conflicts are explicitly labeled", largeDateConflicts.every((row) => row.date_conflict_flag === "true" && row.date_conflict_details.includes("both are preserved")), true, largeDateConflicts.map((row) => `${row.observation_id}:${row.release_date_delta_days}:${row.date_conflict_flag}`).join(" | "));

check("Measurement table contains all expected source families", ["Epoch", "ECI", "ECI Legacy View", "ECI Component", "AA", "AA Detailed View", "No-CoT", "METR"].every((source) => measurements.some((row) => row.source === source)), true, unique(measurements.map((row) => row.source)).join("|"));
equal("Retired ECI legacy measurement count", measurements.filter((row) => row.source === "ECI Legacy View").length, 3);
equal("Compute-enriched measurement row count", measurements.length, 31155);
check("Epoch source-view measurements are explicitly labeled correlated", measurements.filter((row) => row.source.startsWith("Epoch ") && row.source.endsWith(" View")).every((row) => row.measurement_notes.includes("Correlated Epoch archive source view")), true, "all view metrics carry dependence warning");

const csvEscape = (value) => {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};
const resultHeaders = ["test", "status", "expected", "actual", "details"];
const resultCsv = [resultHeaders.join(","), ...results.map((row) => resultHeaders.map((header) => csvEscape(row[header])).join(","))].join("\n") + "\n";
await fs.writeFile(resultsPath, resultCsv, "utf8");

const failures = results.filter((row) => row.status === "FAIL");
console.log(JSON.stringify({ tests: results.length, passed: results.length - failures.length, failed: failures.length, resultsPath, failures }, null, 2));
if (failures.length) process.exitCode = 1;
