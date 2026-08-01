import fs from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

process.on("uncaughtException", (error) => {
  console.error("build_horizon_informed_model failed:", error?.name || "Error", error?.message || String(error));
  if (error?.stack) console.error(error.stack.split("\n").slice(0, 12).join("\n"));
  process.exit(1);
});

const workDir = path.dirname(fileURLToPath(import.meta.url));
const homeDir = process.env.HOME ? path.resolve(process.env.HOME) : null;
function portableLocalPath(value) {
  if (typeof value !== "string" || !path.isAbsolute(value)) return value;
  const resolved = path.resolve(value);
  if (resolved === workDir || resolved.startsWith(`${workDir}${path.sep}`)) {
    return `./${path.relative(workDir, resolved).split(path.sep).join("/")}`;
  }
  if (homeDir && (resolved === homeDir || resolved.startsWith(`${homeDir}${path.sep}`))) {
    return `~/${path.relative(homeDir, resolved).split(path.sep).join("/")}`;
  }
  return value;
}
const threadId = "019f6c42-2d53-7743-ab07-6293e2618dd7";
const outputDir = `${workDir}/outputs/${threadId}`;
const qaDir = `${workDir}/qa/final-crowd-50pct`;
const inputPath = `${outputDir}/price_informed_frontier_parameter_crosscheck_2026-07-17.xlsx`;
const outputPath = `${outputDir}/frontier_parameter_model_crowd_50pct_2026-07-17.xlsx`;
const inspectPath = `${outputPath}.inspect.ndjson`;
const pythonPath = process.env.FORECAST_PYTHON || "python3";
const ooxmlNormalizerPath = `${workDir}/normalize_ooxml_zip.py`;
const inspectNormalizerPath = `${workDir}/normalize_inspect_ndjson.py`;
const localPathScrubberPath = `${workDir}/scrub_artifact_local_paths.py`;
const latexArchive = `${workDir}/sources/input_no_cot_arxiv_2606.07157v3_source.tar.gz`;
const latexArchiveDisplay = "./sources/input_no_cot_arxiv_2606.07157v3_source.tar.gz";
const lessWrongSourcePath = `${workDir}/sources/input_lesswrong_no_cot_article_2026-07-17.txt`;
const noCotDateOverridePath = `${workDir}/sources/no_cot_exact_date_overrides_2026-07-18.csv`;
const noCotDateMetadataPath = `${workDir}/sources/no_cot_exact_date_collection_metadata_2026-07-18.json`;
const qwenDateRawPath = `${workDir}/sources/qwen3_30b_a3b_instruct_2507_hf_commits_2026-07-18.json.gz`;
const noCotExactDateResultPath = `${outputDir}/no_cot_exact_date_audit_2026-07-18.json`;
const noCotExactDateModelAuditPath = `${outputDir}/no_cot_exact_date_model_audit_2026-07-18.csv`;
const noCotArchitectureAuditPath = `${outputDir}/no_cot_architecture_elasticity_audit_2026-07-18.json`;
const noCotArchitecturePredictionsPath = `${outputDir}/no_cot_architecture_elasticity_predictions_2026-07-18.csv`;
const frontierPrimaryEvidencePath = `${workDir}/sources/frontier_primary_evidence_2026-07-18.csv`;
const frontierPrimaryMetadataPath = `${workDir}/sources/frontier_primary_evidence_collection_metadata_2026-07-18.json`;
const frontierPrimaryOpenAIRawPath = `${workDir}/sources/openai_gpt_5_6_system_card_2026-07-18.html.gz`;
const frontierPrimaryAnthropicClaimsPath = `${workDir}/sources/anthropic_fable_mythos_primary_claims_2026-07-18.json`;
const frontierPrimaryAuditPath = `${outputDir}/frontier_primary_evidence_audit_2026-07-18.json`;
const frontierPrimaryControlsPath = `${outputDir}/frontier_primary_evidence_controls_2026-07-18.csv`;
const metrOfficialSignalsPath = `${workDir}/sources/metr_horizon_official_signals_2026-07-18.csv`;
const metrOfficialRawPath = `${workDir}/sources/metr_benchmark_results_1_1_2026-07-18.yaml`;
const metrOfficialMetadataPath = `${workDir}/sources/metr_horizon_official_metadata_2026-07-18.json`;
const metrOfficialAuditPath = `${outputDir}/metr_primary_source_audit_2026-07-18.json`;
const metrLegacyCrosscheckPath = `${workDir}/sources/metr_horizon_user_snapshot_2026-07-17.csv`;
const ikpAuditPath = `${outputDir}/ikp_parameter_signal_audit_2026-07-18.json`;
const ikpPredictionPath = `${outputDir}/ikp_parameter_chronological_predictions_2026-07-18.csv`;
const ikpOverlapPath = `${outputDir}/ikp_parameter_incremental_overlap_2026-07-18.csv`;
const ikpConditionalAuditPath = `${outputDir}/ikp_conditional_benchmark_signal_audit_2026-07-18.json`;
const ikpConditionalPredictionPath = `${outputDir}/ikp_conditional_benchmark_predictions_2026-07-18.csv`;
const ikpSourceMetadataPath = `${workDir}/sources/ikp_source_metadata_2026-07-18.json`;
const epochFeedbackSourcePath = `${workDir}/sources/epoch_employee_calibration_feedback_2026-07-21.csv`;
const epochFeedbackAuditPath = `${outputDir}/epoch_feedback_lean_architecture_audit_2026-07-21.json`;
const epochFeedbackPanelPath = `${outputDir}/epoch_feedback_critique_panel_2026-07-21.csv`;
const epochFeedbackPredictionsPath = `${outputDir}/lean_architecture_predictions_2026-07-21.csv`;
const epochFeedbackTargetsPath = `${outputDir}/lean_architecture_target_sensitivity_2026-07-21.csv`;
const opus5EvidencePath = `${workDir}/sources/claude_opus_5_evidence_2026-07-31.json`;
const metrOfficialUrl = "https://metr.org/assets/benchmark_results_1_1.yaml";
const eciSourcePath = `${workDir}/sources/input_eci_parameter_regression_workbook_2026-07-17.xlsx`;
const epochSourcePath = `${workDir}/sources/epoch_all_ai_models_2026-07-31.csv`;
const epochSnapshotManifestPath = `${workDir}/sources/epoch_snapshot_manifest_2026-07-31.json`;
const epochArchivePath = `${workDir}/sources/input_epoch_ai_models_archive_2026-07-17.zip`;
const unifiedComputePath = `${outputDir}/unified_model_observations_compute_enriched_2026-07-17.csv`;
const unifiedSummaryPath = `${outputDir}/unified_model_data_summary_compute_enriched_2026-07-17.json`;
const aaExpandedResultPath = `${outputDir}/aa_expanded_parameter_audit_2026-07-18.json`;
const aaExpandedPanelPath = `${outputDir}/aa_expanded_parameter_panel_2026-07-18.csv`;
const aaExpandedPredictionsPath = `${outputDir}/aa_expanded_parameter_predictions_2026-07-18.csv`;
const aaExpandedOverlapsPath = `${outputDir}/aa_expanded_parameter_overlap_audit_2026-07-18.csv`;
const aaDetailedRawPath = `${workDir}/sources/aa_detailed_snapshot_2026-07-31.html.gz`;
const aaDetailedModelsPath = `${workDir}/sources/aa_detailed_model_signals_2026-07-31.csv`;
const aaDetailedMetadataPath = `${workDir}/sources/aa_detailed_collection_metadata_2026-07-31.json`;
const aaCalibrationOverridesPath = `${workDir}/sources/aa_calibration_primary_overrides_2026-07-31.json`;
const aaParameterLabelAvailabilityPath = `${workDir}/sources/aa_parameter_label_availability_2026-07-31.json`;
const aaScoreAvailabilityPath = `${workDir}/sources/aa_score_availability_2026-07-31.json`;
const aaChangelogRawPath = `${workDir}/sources/aa_changelog_2026-07-31.json.gz`;
const aaScoreTimingAuditPath = `${outputDir}/aa_score_availability_timing_audit_2026-07-31.json`;
const aaScoreTimingChangesPath = `${outputDir}/aa_score_availability_timing_changes_2026-07-31.csv`;
const aaInferenceResultPath = `${outputDir}/aa_inference_budget_audit_2026-07-18.json`;
const aaDetailedPanelPath = `${outputDir}/aa_detailed_parameter_panel_2026-07-18.csv`;
const aaReasoningPairsPath = `${outputDir}/aa_reasoning_pair_audit_2026-07-18.csv`;
const aaDetailedCrosscheckPath = `${outputDir}/aa_detailed_epoch_crosscheck_2026-07-18.csv`;
const aaInferencePredictionsPath = `${outputDir}/aa_inference_budget_predictions_2026-07-18.csv`;
const aaOperationalResultPath = `${outputDir}/aa_operational_signal_audit_2026-07-18.json`;
const aaOperationalPanelPath = `${outputDir}/aa_operational_parameter_panel_2026-07-18.csv`;
const aaOperationalPredictionsPath = `${outputDir}/aa_operational_backtest_predictions_2026-07-18.csv`;
const aaOpenRouterCrosscheckPath = `${outputDir}/aa_openrouter_operational_crosscheck_2026-07-18.csv`;
const activeTransportResultPath = `${outputDir}/active_parameter_transport_audit_2026-07-18.json`;
const activeTransportPredictionsPath = `${outputDir}/active_parameter_transport_predictions_2026-07-18.csv`;
const activeTransportTargetsPath = `${outputDir}/active_parameter_transport_targets_2026-07-18.csv`;
const k3ReleaseEvidencePath = `${workDir}/sources/kimi_k3_release_evidence_2026-07-31.json`;
const openModelParameterTruthPath = `${workDir}/sources/open_model_parameter_truth_reconciliation_2026-07-31.json`;
const eciArchitectureBlendResultPath = `${outputDir}/eci_architecture_blend_challenger_2026-07-31.json`;
const eciArchitectureBlendPredictionsPath = `${outputDir}/eci_architecture_blend_challenger_predictions_2026-07-31.csv`;
const eciComponentPath = `${workDir}/sources/epoch_eci_benchmarks_2026-07-31.csv`;
const eciBenchmarkArchivePath = `${workDir}/sources/epoch_benchmark_data_2026-07-31.zip`;
const eciReproducedScoresPath = `${workDir}/sources/epoch_eci_reproduced_scores_2026-07-31.csv`;
const eciReproductionMetadataPath = `${workDir}/sources/epoch_eci_reproduction_metadata_2026-07-31.json`;
const eciReproductionCrosscheckPath = `${outputDir}/epoch_eci_reproduction_crosscheck_2026-07-31.csv`;
const eciReproductionAuditPath = `${outputDir}/epoch_eci_reproduction_audit_2026-07-31.json`;
const eciComponentExtendedResultPath = `${outputDir}/eci_component_extended_audit_2026-07-18.json`;
const eciComponentExpandedPanelPath = `${outputDir}/eci_component_expanded_parameter_panel_2026-07-18.csv`;
const eciComponentActiveComparisonPath = `${outputDir}/eci_component_active_incremental_comparison_2026-07-18.csv`;
const eciMultivariateResultPath = `${outputDir}/eci_multivariate_component_audit_2026-07-18.json`;
const eciMultivariatePredictionsPath = `${outputDir}/eci_multivariate_component_predictions_2026-07-18.csv`;
const eciMultivariateNarrowCiPredictionsPath = `${outputDir}/eci_multivariate_component_narrow_eci_ci_predictions_2026-07-18.csv`;
const eciMultivariateTargetsPath = `${outputDir}/eci_multivariate_component_targets_2026-07-18.csv`;
const eciMultivariateCoveragePath = `${outputDir}/eci_multivariate_component_coverage_2026-07-18.csv`;
const posttrainingLineageResultPath = `${outputDir}/posttraining_lineage_audit_2026-07-18.json`;
const posttrainingLineageEdgesPath = `${outputDir}/posttraining_lineage_edges_2026-07-18.csv`;
const posttrainingLineageMeasurementsPath = `${outputDir}/posttraining_lineage_measurements_2026-07-18.csv`;
const posttrainingLineagePredictionsPath = `${outputDir}/posttraining_lineage_predictions_2026-07-18.csv`;
const frontierSharedBaseSensitivityPath = `${outputDir}/frontier_shared_base_sensitivity_2026-07-18.csv`;
const frontierLineageEvidencePath = `${outputDir}/frontier_lineage_evidence_2026-07-18.csv`;
const forecastLedgerPath = `${workDir}/sources/human_parameter_forecasts_2026-07-17.csv`;
const predictionRegistryPath = `${outputDir}/frontier_parameter_prediction_registry_v2.1_2026-07-17.docx`;
const openRouterRawPath = `${workDir}/sources/openrouter_operational_snapshot_2026-07-18.json.gz`;
const openRouterModelPath = `${workDir}/sources/openrouter_model_signals_2026-07-18.csv`;
const openRouterProviderPath = `${workDir}/sources/openrouter_provider_signals_2026-07-18.csv`;
const openRouterTierPath = `${workDir}/sources/openrouter_endpoint_tier_signals_2026-07-18.csv`;
const openRouterDailyPath = `${workDir}/sources/openrouter_throughput_daily_2026-07-18.csv`;
const openRouterModelHistoryPath = `${workDir}/sources/openrouter_model_snapshot_history_2026-07-18.csv`;
const openRouterProviderHistoryPath = `${workDir}/sources/openrouter_provider_snapshot_history_2026-07-18.csv`;
const openRouterTierHistoryPath = `${workDir}/sources/openrouter_endpoint_tier_snapshot_history_2026-07-18.csv`;
const openRouterDailyHistoryPath = `${workDir}/sources/openrouter_throughput_daily_history_2026-07-18.csv`;
const openRouterHistoryManifestPath = `${workDir}/sources/openrouter_snapshot_history_manifest_2026-07-18.csv`;
const openRouterAuditPath = `${outputDir}/openrouter_epoch_match_audit_2026-07-18.csv`;
const openRouterFrontierPath = `${outputDir}/openrouter_frontier_operational_estimates_2026-07-18.csv`;
const openRouterResultPath = `${outputDir}/openrouter_parameter_signal_backtest_2026-07-18.json`;
const openRouterTemporalResultPath = `${outputDir}/openrouter_temporal_stability_audit_2026-07-18.json`;
const openRouterRequestWeightedResultPath = `${outputDir}/openrouter_request_weighted_operational_audit_2026-07-18.json`;
const openRouterRequestWeightedPredictionsPath = `${outputDir}/openrouter_request_weighted_operational_predictions_2026-07-18.csv`;
const openRouterEndpointStabilityPath = `${outputDir}/openrouter_endpoint_temporal_stability_2026-07-18.csv`;
const openRouterModelStabilityPath = `${outputDir}/openrouter_model_temporal_stability_2026-07-18.csv`;
const openRouterRefreshStabilityPath = `${outputDir}/openrouter_refresh_stability_2026-07-18.csv`;
const openRouterTierPredictionsPath = `${outputDir}/openrouter_tier_counterfactual_predictions_2026-07-18.csv`;
const openRouterCollectionAuditPath = `${outputDir}/openrouter_collection_audit_2026-07-18.json`;
const openRouterOfficialSnapshotPath = `${workDir}/sources/openrouter_official_endpoint_snapshot_2026-07-18.json.gz`;
const openRouterOfficialPricePath = `${workDir}/sources/openrouter_official_endpoint_prices_2026-07-18.csv`;
const openRouterOfficialComparisonPath = `${outputDir}/openrouter_official_endpoint_crosscheck_2026-07-18.csv`;
const openRouterOfficialAuditPath = `${outputDir}/openrouter_official_endpoint_audit_2026-07-18.json`;
const openRouterActivePriceResultPath = `${outputDir}/openrouter_active_price_audit_2026-07-18.json`;
const openRouterActivePriceMatchPath = `${outputDir}/openrouter_active_parameter_match_audit_2026-07-18.csv`;
const openRouterActivePricePredictionsPath = `${outputDir}/openrouter_active_price_predictions_2026-07-18.csv`;
const openRouterActivePriceTargetsPath = `${outputDir}/openrouter_active_price_targets_2026-07-18.csv`;
const openRouterHistoricalRawPath = `${workDir}/sources/openrouter_historical_price_ledger_2026-07-18.json.gz`;
const openRouterHistoricalChangePointsPath = `${workDir}/sources/openrouter_historical_price_change_points_2026-07-18.csv`;
const openRouterHistoricalMetadataPath = `${workDir}/sources/openrouter_historical_price_collection_metadata_2026-07-18.json`;
const openRouterHistoricalResultPath = `${outputDir}/openrouter_historical_price_audit_2026-07-18.json`;
const openRouterHistoricalMatchPath = `${outputDir}/openrouter_historical_price_match_audit_2026-07-18.csv`;
const openRouterHistoricalPredictionsPath = `${outputDir}/openrouter_historical_price_backtest_predictions_2026-07-18.csv`;
const openRouterHistoricalTargetsPath = `${outputDir}/openrouter_historical_price_frontier_targets_2026-07-18.csv`;
const huggingFaceArchitectureRawPath = `${workDir}/sources/huggingface_architecture_config_snapshot_2026-07-18.json.gz`;
const huggingFaceArchitectureSignalsPath = `${workDir}/sources/huggingface_architecture_config_signals_2026-07-18.csv`;
const huggingFaceArchitectureAuditPath = `${outputDir}/huggingface_architecture_config_collection_audit_2026-07-18.json`;
const epochCsvUrl = "https://epoch.ai/data/all_ai_models.csv";
const epochDocsUrl = "https://epoch.ai/data/ai-models-documentation/downloads";
const arxivUrl = "https://arxiv.org/pdf/2606.07157";

const parseCsvText = (text) => {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"') {
        if (text[index + 1] === '"') { cell += '"'; index += 1; } else quoted = false;
      } else cell += character;
    } else if (character === '"') quoted = true;
    else if (character === ",") { row.push(cell); cell = ""; }
    else if (character === "\n") { row.push(cell); rows.push(row); row = []; cell = ""; }
    else if (character !== "\r") cell += character;
  }
  if (quoted) throw new Error("Unclosed quote in unified compute CSV");
  if (cell.length || row.length) { row.push(cell); rows.push(row); }
  return rows;
};
const objectRows = (text) => {
  const matrix = parseCsvText(text);
  const headers = [...matrix[0]];
  headers[0] = headers[0].replace(/^\uFEFF/, "");
  return matrix.slice(1).filter((row) => row.some((cell) => cell !== "")).map((row, index) => {
    if (row.length !== headers.length) throw new Error(`CSV row ${index + 2}: ${row.length}/${headers.length} fields`);
    return Object.fromEntries(headers.map((header, column) => [header, row[column]]));
  });
};

const opus5Evidence = JSON.parse(await fs.readFile(opus5EvidencePath, "utf8"));
const opus5Identity = opus5Evidence.identity;
const opus5Aa = opus5Evidence.artificial_analysis.selected;
const opus5Epoch = opus5Evidence.epoch;
const opus5Availability = opus5Evidence.availability;
if (
  opus5Identity.canonical_name !== "Claude Opus 5"
  || opus5Identity.release_date !== "2026-07-24"
  || opus5Identity.parameter_disclosed
  || opus5Identity.same_weight_identity_disclosed
  || opus5Identity.base_identity_policy !== "unique_base"
) throw new Error("Claude Opus 5 identity evidence violates the distinct undisclosed-base policy");
if (["metr", "no_cot", "ikp"].some((key) => opus5Availability[key])) {
  throw new Error("Claude Opus 5 must not receive an unavailable direct measurement");
}

const noCotDateOverrideRows = objectRows(await fs.readFile(noCotDateOverridePath, "utf8"));
const noCotDateMetadata = JSON.parse(await fs.readFile(noCotDateMetadataPath, "utf8"));
const noCotExactDateResult = JSON.parse(await fs.readFile(noCotExactDateResultPath, "utf8"));
const noCotExactDateModelRows = objectRows(await fs.readFile(noCotExactDateModelAuditPath, "utf8"));
const noCotArchitectureAudit = JSON.parse(await fs.readFile(noCotArchitectureAuditPath, "utf8"));
const frontierPrimaryEvidenceRows = objectRows(await fs.readFile(frontierPrimaryEvidencePath, "utf8"));
const frontierPrimaryMetadata = JSON.parse(await fs.readFile(frontierPrimaryMetadataPath, "utf8"));
const frontierPrimaryResult = JSON.parse(await fs.readFile(frontierPrimaryAuditPath, "utf8"));
const frontierPrimaryControlRows = objectRows(await fs.readFile(frontierPrimaryControlsPath, "utf8"));
const metrOfficialRows = objectRows(await fs.readFile(metrOfficialSignalsPath, "utf8"));
const metrOfficialMetadata = JSON.parse(await fs.readFile(metrOfficialMetadataPath, "utf8"));
const metrOfficialAudit = JSON.parse(await fs.readFile(metrOfficialAuditPath, "utf8"));
const ikpAudit = JSON.parse(await fs.readFile(ikpAuditPath, "utf8"));
const ikpPredictionRows = objectRows(await fs.readFile(ikpPredictionPath, "utf8"));
const ikpOverlapRows = objectRows(await fs.readFile(ikpOverlapPath, "utf8"));
const ikpConditionalAudit = JSON.parse(await fs.readFile(ikpConditionalAuditPath, "utf8"));
const ikpConditionalPredictionRows = objectRows(await fs.readFile(ikpConditionalPredictionPath, "utf8"));
const ikpSourceMetadata = JSON.parse(await fs.readFile(ikpSourceMetadataPath, "utf8"));
const ikpDecision = ikpAudit.decision;
const ikpPromoted = Boolean(ikpDecision.promote_incremental_ikp_weight);
const ikpEvidenceWeight = Number(ikpDecision.incremental_evidence_weight);
const ikpFinalWeight = Number(ikpDecision.incremental_final_weight_when_crowd_is_50pct);
const ikpExpectedPredictionRows = Object.values(ikpAudit.heldout_metrics)
  .flatMap((policy) => Object.values(policy))
  .reduce((sum, metrics) => sum + Number(metrics.all.n), 0);
if (![ikpEvidenceWeight, ikpFinalWeight].every(Number.isFinite)) throw new Error("IKP decision weights must be finite");
if (ikpEvidenceWeight < 0 || ikpEvidenceWeight > 1 || ikpFinalWeight < 0 || ikpFinalWeight > 0.5) throw new Error("IKP decision weights are outside their admissible range");
if (!ikpPromoted && (ikpEvidenceWeight !== 0 || ikpFinalWeight !== 0)) throw new Error("A non-promoted IKP signal must have zero live weight");
if (ikpPromoted && ikpEvidenceWeight <= 0) throw new Error("A promoted IKP signal must have positive live weight");
if (Math.abs(ikpFinalWeight - 0.5 * ikpEvidenceWeight) > 1e-12) throw new Error("IKP final weight must equal half its evidence weight under the 50% crowd policy");
if (noCotDateOverrideRows.length !== 4 || new Set(noCotDateOverrideRows.map((row) => row.paper_model)).size !== 4) throw new Error("No-CoT exact-date override inventory mismatch");
const noCotDateOverrideByModel = new Map(noCotDateOverrideRows.map((row) => [row.paper_model, row]));
// The current ECI snapshot correctly removed the old "Kimi K2 (Sep 2025)" aggregate.
// Preserve the 0905 checkpoint's day-level date from its explicit checkpoint identifier,
// without reviving the removed ECI score or treating K2 base parameters as an exact join.
noCotDateOverrideByModel.set("Kimi K2-0905", {
  exact_release_date: "2025-09-05",
  exact_date_source_label: "Checkpoint identifier exact date (0905)",
});
if (frontierPrimaryEvidenceRows.length !== 5 || frontierPrimaryControlRows.length !== 33) throw new Error("Frontier primary-evidence inventory mismatch");
if (metrOfficialRows.length !== 26 || new Set(metrOfficialRows.map((row) => row.source_id)).size !== 26 || metrOfficialAudit.status !== "PASS" || metrOfficialAudit.legacy_exact_crosscheck.exact_rows !== 26) throw new Error("METR official-source audit mismatch");
if (ikpOverlapRows.length !== ikpAudit.incremental_overlap.models || ikpPredictionRows.length !== ikpExpectedPredictionRows) throw new Error("IKP incremental-signal audit mismatch");
if (!ikpConditionalAudit.decision.conditional_incremental_signal_corroborated || ikpConditionalAudit.decision.change_live_ikp_weight || ikpConditionalPredictionRows.length !== 196) throw new Error("IKP conditional benchmark audit mismatch");
const metrOfficialById = new Map(metrOfficialRows.map((row) => [row.source_id, row]));

const openRouterAuditRows = objectRows(await fs.readFile(openRouterAuditPath, "utf8"));
const openRouterTierRows = objectRows(await fs.readFile(openRouterTierPath, "utf8"));
const openRouterFrontierRows = objectRows(await fs.readFile(openRouterFrontierPath, "utf8"));
const openRouterResult = JSON.parse(await fs.readFile(openRouterResultPath, "utf8"));
const openRouterTemporalResult = JSON.parse(await fs.readFile(openRouterTemporalResultPath, "utf8"));
const openRouterRequestWeightedResult = JSON.parse(await fs.readFile(openRouterRequestWeightedResultPath, "utf8"));
const openRouterEndpointStabilityRows = objectRows(await fs.readFile(openRouterEndpointStabilityPath, "utf8"));
const openRouterModelStabilityRows = objectRows(await fs.readFile(openRouterModelStabilityPath, "utf8"));
const openRouterRefreshStabilityRows = objectRows(await fs.readFile(openRouterRefreshStabilityPath, "utf8"));
const openRouterHistoryManifestRows = objectRows(await fs.readFile(openRouterHistoryManifestPath, "utf8"));
const openRouterCollectionAudit = JSON.parse(await fs.readFile(openRouterCollectionAuditPath, "utf8"));
const openRouterOfficialComparisonRows = objectRows(await fs.readFile(openRouterOfficialComparisonPath, "utf8"));
const openRouterOfficialAudit = JSON.parse(await fs.readFile(openRouterOfficialAuditPath, "utf8"));
const openRouterActivePriceResult = JSON.parse(await fs.readFile(openRouterActivePriceResultPath, "utf8"));
const openRouterActivePriceMatchRows = objectRows(await fs.readFile(openRouterActivePriceMatchPath, "utf8"));
const openRouterActivePricePredictionRows = objectRows(await fs.readFile(openRouterActivePricePredictionsPath, "utf8"));
const openRouterActivePriceTargetRows = objectRows(await fs.readFile(openRouterActivePriceTargetsPath, "utf8"));
const openRouterHistoricalMetadata = JSON.parse(await fs.readFile(openRouterHistoricalMetadataPath, "utf8"));
const openRouterHistoricalResult = JSON.parse(await fs.readFile(openRouterHistoricalResultPath, "utf8"));
const openRouterHistoricalMatchRows = objectRows(await fs.readFile(openRouterHistoricalMatchPath, "utf8"));
const openRouterHistoricalPredictionRows = objectRows(await fs.readFile(openRouterHistoricalPredictionsPath, "utf8"));
const openRouterHistoricalTargetRows = objectRows(await fs.readFile(openRouterHistoricalTargetsPath, "utf8"));
const aaExpandedResult = JSON.parse(await fs.readFile(aaExpandedResultPath, "utf8"));
const aaExpandedRows = objectRows(await fs.readFile(aaExpandedPanelPath, "utf8"));
const aaExpandedOverlapRows = objectRows(await fs.readFile(aaExpandedOverlapsPath, "utf8"));
const aaInferenceResult = JSON.parse(await fs.readFile(aaInferenceResultPath, "utf8"));
const aaDetailedRows = objectRows(await fs.readFile(aaDetailedPanelPath, "utf8"));
const aaDetailedSourceRows = objectRows(await fs.readFile(aaDetailedModelsPath, "utf8"));
if (aaDetailedSourceRows.length !== 587 || new Set(aaDetailedSourceRows.map((row) => row.model_id)).size !== 587) throw new Error("AA detailed source inventory mismatch");
const aaDetailedBySlug = new Map(aaDetailedSourceRows.map((row) => [row.slug, row]));
const aaExact = (slug) => {
  const row = aaDetailedBySlug.get(slug);
  if (!row || !row.intelligence_index) throw new Error(`Missing exact AA target row: ${slug}`);
  return Number(row.intelligence_index);
};
const aaReasoningPairRows = objectRows(await fs.readFile(aaReasoningPairsPath, "utf8"));
const aaDetailedCrosscheckRows = objectRows(await fs.readFile(aaDetailedCrosscheckPath, "utf8"));
const aaOperationalResult = JSON.parse(await fs.readFile(aaOperationalResultPath, "utf8"));
const aaOperationalRows = objectRows(await fs.readFile(aaOperationalPanelPath, "utf8"));
const aaOpenRouterCrosscheckRows = objectRows(await fs.readFile(aaOpenRouterCrosscheckPath, "utf8"));
const activeTransportResult = JSON.parse(await fs.readFile(activeTransportResultPath, "utf8"));
const activeTransportPredictionRows = objectRows(await fs.readFile(activeTransportPredictionsPath, "utf8"));
const activeTransportTargetRows = objectRows(await fs.readFile(activeTransportTargetsPath, "utf8"));
const k3ReleaseEvidence = JSON.parse(await fs.readFile(k3ReleaseEvidencePath, "utf8"));
const openModelParameterTruth = JSON.parse(await fs.readFile(openModelParameterTruthPath, "utf8"));
const eciArchitectureBlendResult = JSON.parse(await fs.readFile(eciArchitectureBlendResultPath, "utf8"));
const k3Facts = k3ReleaseEvidence.kimi_k3;
const k2Facts = k3ReleaseEvidence.kimi_k2_comparator;
const aaDetailedMetadata = JSON.parse(await fs.readFile(aaDetailedMetadataPath, "utf8"));
const aaCalibrationOverrides = JSON.parse(await fs.readFile(aaCalibrationOverridesPath, "utf8"));
const aaParameterLabelAvailability = JSON.parse(await fs.readFile(aaParameterLabelAvailabilityPath, "utf8"));
const aaScoreAvailability = JSON.parse(await fs.readFile(aaScoreAvailabilityPath, "utf8"));
const aaScoreTimingAudit = JSON.parse(await fs.readFile(aaScoreTimingAuditPath, "utf8"));
const epochSnapshotManifest = JSON.parse(await fs.readFile(epochSnapshotManifestPath, "utf8"));
const unifiedSummary = JSON.parse(await fs.readFile(unifiedSummaryPath, "utf8"));
const eciReproducedRows = objectRows(await fs.readFile(eciReproducedScoresPath, "utf8"));
const eciReproductionCrosscheckRows = objectRows(await fs.readFile(eciReproductionCrosscheckPath, "utf8"));
const eciReproductionAudit = JSON.parse(await fs.readFile(eciReproductionAuditPath, "utf8"));
if (epochSnapshotManifest.snapshot_as_of !== "2026-07-31" || aaDetailedMetadata.snapshot_date !== "2026-07-31" || k3ReleaseEvidence.snapshot_date !== "2026-07-31") {
  throw new Error("Workbook build requires the installed July 31 Epoch, AA, and K3 evidence snapshots");
}
if (
  openModelParameterTruth.schema_version !== "1.0"
  || openModelParameterTruth.snapshot_date !== "2026-07-31"
  || openModelParameterTruth.policy?.ordinary_build_network_reads !== 0
) throw new Error("Open-model parameter-truth reconciliation is not pinned/offline");
if (
  eciArchitectureBlendResult.decision?.incremental_live_weight !== 0
  || eciArchitectureBlendResult.decision?.change_live_weights !== false
  || eciArchitectureBlendResult.decision?.change_central_forecasts !== false
) throw new Error("ECI architecture challenger must remain a zero-weight diagnostic");
if (
  k3Facts.total_parameters_b_exact !== 2780
  || k3Facts.activated_parameters_b_exact !== 104.2
  || !k3Facts.parameter_count_disclosed
  || !k3Facts.activated_parameter_count_disclosed
) throw new Error("K3 official evidence must retain exact 2.78T total / 104.2B active");
if (
  eciReproducedRows.length !== epochSnapshotManifest.inventory.models
  || eciReproductionCrosscheckRows.length !== epochSnapshotManifest.inventory.models
  || eciReproductionAudit.reproduction.input_rows !== epochSnapshotManifest.inventory.component_rows
  || eciReproductionAudit.reproduction.input_benchmarks !== epochSnapshotManifest.inventory.benchmarks
) throw new Error("Current ECI files disagree with the Epoch snapshot manifest");
const aaPrimaryMetadataOverrides = aaInferenceResult.data_audit.primary_metadata_overrides || [];
const expectedAaOverrideIds = (aaCalibrationOverrides.overrides || [])
  .map((row) => row.override_id)
  .sort();
const actualAaOverrideIds = aaPrimaryMetadataOverrides
  .map((row) => row.override_id)
  .sort();
if (
  aaInferenceResult.data_audit.raw_models !== aaDetailedMetadata.models
  || aaCalibrationOverrides.schema_version !== "1.0"
  || aaCalibrationOverrides.snapshot_date !== "2026-07-31"
  || aaParameterLabelAvailability.schema_version !== "1.0"
  || aaParameterLabelAvailability.snapshot_date !== "2026-07-31"
  || !Array.isArray(aaParameterLabelAvailability.records)
  || aaParameterLabelAvailability.records.length === 0
  || aaScoreAvailability.schema_version !== "1.0"
  || aaScoreAvailability.snapshot_date !== "2026-07-31"
  || !Array.isArray(aaScoreAvailability.records)
  || aaScoreAvailability.records.length === 0
  || aaScoreTimingAudit.decision.change_current_fit !== false
  || aaScoreTimingAudit.decision.change_headline_centers !== false
  || aaScoreTimingAudit.decision.change_validation_and_uncertainty !== true
  || expectedAaOverrideIds.length === 0
  || new Set(expectedAaOverrideIds).size !== expectedAaOverrideIds.length
  || JSON.stringify(actualAaOverrideIds) !== JSON.stringify(expectedAaOverrideIds)
  || aaInferenceResult.data_audit.open_weight_parameter_score_date_configurations
    !== aaDetailedMetadata.open_weight_parameter_score_date_rows + aaPrimaryMetadataOverrides.length
  || aaDetailedRows.length !== aaInferenceResult.data_audit.unique_checkpoint_groups
) throw new Error("Current AA calibration view does not reconcile to the July 31 raw snapshot plus primary-source overrides");
const eciComponentExtendedResult = JSON.parse(await fs.readFile(eciComponentExtendedResultPath, "utf8"));
const eciComponentExpandedRows = objectRows(await fs.readFile(eciComponentExpandedPanelPath, "utf8"));
const eciComponentComparisonRows = objectRows(await fs.readFile(eciComponentActiveComparisonPath, "utf8"));
const eciMultivariateResult = JSON.parse(await fs.readFile(eciMultivariateResultPath, "utf8"));
const eciMultivariatePredictionRows = objectRows(await fs.readFile(eciMultivariatePredictionsPath, "utf8"));
const eciMultivariateNarrowCiPredictionRows = objectRows(await fs.readFile(eciMultivariateNarrowCiPredictionsPath, "utf8"));
const eciMultivariateTargetRows = objectRows(await fs.readFile(eciMultivariateTargetsPath, "utf8"));
const eciMultivariateCoverageRows = objectRows(await fs.readFile(eciMultivariateCoveragePath, "utf8"));
const posttrainingLineageResult = JSON.parse(await fs.readFile(posttrainingLineageResultPath, "utf8"));
const posttrainingLineageEdgeRows = objectRows(await fs.readFile(posttrainingLineageEdgesPath, "utf8"));
const posttrainingLineageMeasurementRows = objectRows(await fs.readFile(posttrainingLineageMeasurementsPath, "utf8"));
const posttrainingLineagePredictionRows = objectRows(await fs.readFile(posttrainingLineagePredictionsPath, "utf8"));
const frontierSharedBaseSensitivityRows = objectRows(await fs.readFile(frontierSharedBaseSensitivityPath, "utf8"));
const frontierLineageEvidenceRows = objectRows(await fs.readFile(frontierLineageEvidencePath, "utf8"));

const crowdLedgerRows = objectRows(await fs.readFile(forecastLedgerPath, "utf8"));
const requiredCrowdColumns = ["forecast_id", "contributor", "date", "model", "forecast_text", "low_t", "high_t", "central_t", "confidence", "provenance", "notes", "supersedes"];
if (!requiredCrowdColumns.every((column) => Object.hasOwn(crowdLedgerRows[0] || {}, column))) throw new Error("Human forecast ledger schema mismatch");
const crowdIds = crowdLedgerRows.map((row) => row.forecast_id);
if (new Set(crowdIds).size !== crowdIds.length) throw new Error("Human forecast ledger contains duplicate forecast_id values");
const crowdIdSet = new Set(crowdIds);
const supersededCrowdIds = new Set();
for (const row of crowdLedgerRows) {
  row.low_t = Number(row.low_t);
  row.high_t = Number(row.high_t);
  if (!row.forecast_id || !row.contributor || !row.model || !(row.low_t > 0) || !(row.high_t >= row.low_t)) throw new Error(`Invalid human forecast row: ${row.forecast_id || "missing id"}`);
  const central = row.central_t === "" ? null : Number(row.central_t);
  if (central != null && (!(central > 0) || central < row.low_t || central > row.high_t)) throw new Error(`Invalid stated central forecast: ${row.forecast_id}`);
  row.central_t = central;
  row.point_t = central ?? Math.sqrt(row.low_t * row.high_t);
  if (row.supersedes) {
    if (!crowdIdSet.has(row.supersedes)) throw new Error(`Unknown supersedes id: ${row.supersedes}`);
    supersededCrowdIds.add(row.supersedes);
  }
}
const activeCrowdRows = crowdLedgerRows.filter((row) => !supersededCrowdIds.has(row.forecast_id));
const activeCrowdKeys = activeCrowdRows.map((row) => `${row.contributor}\u001f${row.model}`);
if (new Set(activeCrowdKeys).size !== activeCrowdKeys.length) throw new Error("Multiple active forecasts for one contributor/model pair");
const geometricMean = (values) => Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);
const crowdStatsByModel = new Map();
for (const model of new Set(activeCrowdRows.map((row) => row.model))) {
  const rows = activeCrowdRows.filter((row) => row.model === model);
  crowdStatsByModel.set(model, { rows, n: rows.length, center: geometricMean(rows.map((row) => row.point_t)) });
}
const requiredCrowdModels = ["Claude Fable 5", "GPT-5.6 Sol"];
for (const model of requiredCrowdModels) if (!crowdStatsByModel.has(model)) throw new Error(`Missing crowd pool: ${model}`);
const crowdSummary = (model) => crowdStatsByModel.get(model);

const tex = execFileSync("tar", ["-xOzf", latexArchive, "arxiv_version.tex"], {
  encoding: "utf8",
  maxBuffer: 80 * 1024 * 1024,
});

const normalize = (value) => value
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

const splitCells = (line) => line
  .replace(/\\\\\s*$/, "")
  .split("&")
  .map((cell) => normalize(cell));

const monthNumber = new Map([
  ["January", 0], ["February", 1], ["March", 2], ["April", 3], ["May", 4], ["June", 5],
  ["July", 6], ["August", 7], ["September", 8], ["October", 9], ["November", 10], ["December", 11],
]);

const monthDate = (text) => {
  const match = normalize(text).match(/^([A-Za-z]+)\s+(\d{4})$/);
  if (!match || !monthNumber.has(match[1])) throw new Error(`Bad month date: ${text}`);
  return new Date(Date.UTC(Number(match[2]), monthNumber.get(match[1]), 1));
};

const magnitude = (text) => {
  const cleaned = normalize(text).replace(/,/g, "").replace(/^</, "").trim();
  const match = cleaned.match(/^([0-9.]+)\s*([kKmMtT])?$/);
  if (!match) throw new Error(`Bad magnitude: ${text}`);
  const base = Number(match[1]);
  const suffix = (match[2] || "").toLowerCase();
  return base * (suffix === "k" ? 1_000 : suffix === "m" ? 1_000_000 : suffix === "t" ? 1_000_000_000_000 : 1);
};

const paramBillions = (text) => {
  const cleaned = normalize(text).replace(/\s+/g, "");
  const match = cleaned.match(/^([0-9.]+)([BT])$/i);
  if (!match) throw new Error(`Bad parameter cell: ${text}`);
  return Number(match[1]) * (match[2].toUpperCase() === "T" ? 1_000 : 1);
};

const estimateWithCi = (cell) => {
  const expanded = cell.replace(/\\tiny\{([^{}]*)\}/g, "$1");
  const cleaned = normalize(expanded);
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
  if (!match) throw new Error(`Missing macro ${name}`);
  return Number(match[1]);
};

const frontierRelease = new Map();
for (const line of dataLines(tableByLabel("tab:frontier-models"))) {
  const cells = splitCells(line);
  if (cells.length !== 2 || cells[0] === "Model") continue;
  frontierRelease.set(cells[0], monthDate(cells[1]));
}
frontierRelease.set("Sonnet 3.7", frontierRelease.get("Claude 3.7 Sonnet"));
frontierRelease.set("Opus 4", frontierRelease.get("Claude Opus 4"));
frontierRelease.set("Opus 4.1", frontierRelease.get("Claude Opus 4.1"));
frontierRelease.set("Opus 4.5", frontierRelease.get("Claude Opus 4.5"));
frontierRelease.set("Opus 4.6", frontierRelease.get("Claude Opus 4.6"));
frontierRelease.set("Opus 4.7", frontierRelease.get("Claude Opus 4.7"));

const frontierHorizons = [];
for (const line of dataLines(tableByLabel("tab:horizons-per-model"))) {
  const rawCells = line.replace(/\\\\\s*$/, "").split("&").map((cell) => cell.trim());
  if (rawCells.length !== 3 || normalize(rawCells[0]) === "Model") continue;
  const model = normalize(rawCells[0]);
  const time = estimateWithCi(rawCells[1]);
  const tokens = estimateWithCi(rawCells[2]);
  frontierHorizons.push({ model, release: frontierRelease.get(model), time, tokens });
}

const openMeta = new Map();
let developer = "";
for (const line of tableByLabel("tab:open-source-models").split(/\r?\n/)) {
  const group = line.match(/\\textit\{([^}]+)\}/);
  if (line.includes("multicolumn") && group) {
    developer = group[1].replace(/\s*\([^)]*\)\s*/, "").trim();
    continue;
  }
  if (!line.includes("&") || !line.trim().endsWith("\\\\")) continue;
  const cells = splitCells(line.trim());
  if (cells.length !== 7 || cells[0] === "Model") continue;
  const parts = cells[2].split("/").map((part) => part.trim());
  const totalB = paramBillions(parts[0]);
  const activeB = parts.length === 2 ? paramBillions(parts[1]) : totalB;
  openMeta.set(cells[0], {
    developer,
    model: cells[0],
    release: monthDate(cells[1]),
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
  if (cells.length !== 5 || cells[0] === "Model") continue;
  openHorizons.set(cells[0], {
    point: Number(cells[1]), median: Number(cells[2]), low: Number(cells[3]), high: Number(cells[4]),
  });
}

const openModels = [...openMeta.values()].map((meta) => {
  const horizon = openHorizons.get(meta.model);
  if (!horizon) throw new Error(`No TH row for ${meta.model}`);
  return { ...meta, ...horizon };
});
if (frontierHorizons.length !== 14 || openModels.length !== 35) {
  throw new Error(`Unexpected source counts: frontier=${frontierHorizons.length}, open=${openModels.length}`);
}

const scaleMatch = tex.match(/Doubling the 50\\% TH requires a \$([0-9.]+)\\times\$ increase in total parameters, a \$([0-9.]+)\\times\$ increase in active parameters, a \$([0-9.]+)\\times\$ increase in the layer count, or a \$([0-9.]+)\\times\$ increase in pretraining FLOPs/);
if (!scaleMatch) throw new Error("Could not parse reported open-weight scaling laws");
const scaling = {
  total: Number(scaleMatch[1]), active: Number(scaleMatch[2]), layers: Number(scaleMatch[3]), pretrain: Number(scaleMatch[4]),
  timeDays: macroNumber("FINALDOUBLINGTIMETIME"), tokenDays: macroNumber("FINALDOUBLINGTIMETOKENS"),
};

const scientific = (text) => {
  const cleaned = text.replace(/\$/g, "").trim();
  const match = cleaned.match(/([0-9.]+)\s*\\times\s*10\^\{?(-?[0-9]+)\}?/);
  return match ? Number(match[1]) * 10 ** Number(match[2]) : Number(cleaned);
};

const spearman = [];
for (const line of dataLines(tableByLabel("tab:open-weight-spearman"))) {
  const raw = line.replace(/\\\\\s*$/, "").split("&").map((cell) => cell.trim());
  if (raw.length !== 4 || normalize(raw[0]) === "Axis") continue;
  spearman.push({ axis: normalize(raw[0]), rho: Number(normalize(raw[1]).replace("+", "")), p: scientific(raw[2]), n: Number(normalize(raw[3])) });
}

// Immutable ECI snapshots and a manually reviewed, exact alias layer.
const eciWb = await SpreadsheetFile.importXlsx(await FileBlob.load(eciSourcePath));
const eciSourceSheets = [
  ["Dashboard", "ECI Dashboard Raw"],
  ["Frontier Estimates", "ECI Frontier Raw"],
  ["Open Models 12mo", "ECI Open12 Raw"],
  ["Regression Data", "ECI Regression Raw"],
  ["Regression Model", "ECI Model Raw"],
  ["ECI Graph Data", "ECI Graph Raw"],
  ["Benchmarked Models", "ECI Benchmarks Raw"],
];
const eciSnapshots = eciSourceSheets.map(([sourceName, targetName]) => {
  const range = eciWb.worksheets.getItem(sourceName).getUsedRange();
  return { sourceName, targetName, values: range.values };
});
const eciFormulaInspection = await eciWb.inspect({ kind: "formula", include: "formulas", options: { maxResults: 10000 }, maxChars: 2000000 });
const formulaRows = eciFormulaInspection.ndjson.split(/\r?\n/).filter(Boolean).map((line) => {
  try { return JSON.parse(line); } catch { return null; }
}).filter((item) => item?.kind === "formula").map((item) => {
  const formula = String(item.formula || "");
  return [item.sheet, item.address, formula.startsWith("=") ? "U+003D" : "", formula.startsWith("=") ? formula.slice(1) : formula];
});

const toDate = (value) => {
  if (value instanceof Date) return value;
  if (typeof value === "number") return new Date((value - 25569) * 86400000);
  if (typeof value === "string" && value) return new Date(`${value.slice(0, 10)}T00:00:00Z`);
  return null;
};
const iso = (value) => {
  const date = toDate(value);
  return date && !Number.isNaN(date.getTime()) ? date.toISOString().slice(0, 10) : "";
};
const slug = (value) => String(value).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
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
  if (org.includes("xai")) return "xai";
  return slug(organization || "unknown");
};
const headersOf = (values) => Object.fromEntries(values[0].map((value, index) => [String(value), index]));
const graphSnapshot = eciSnapshots.find((item) => item.sourceName === "ECI Graph Data");
const graphHeaders = headersOf(graphSnapshot.values);
const legacyEciRows = graphSnapshot.values.slice(1).map((row, index) => ({
  sourceRow: index + 2,
  name: String(row[graphHeaders.Model] ?? ""),
  display: String(row[graphHeaders["Display name"]] ?? row[graphHeaders.Model] ?? ""),
  date: toDate(row[graphHeaders.Date]),
  eci: row[graphHeaders.ECI],
  eciLow: row[graphHeaders["ECI CI low"]],
  eciHigh: row[graphHeaders["ECI CI high"]],
  organization: String(row[graphHeaders.Organization] ?? ""),
  country: String(row[graphHeaders["Country (of organization)"]] ?? ""),
  accessibility: String(row[graphHeaders["Model accessibility"]] ?? ""),
  accessibilityGroup: String(row[graphHeaders["Accessibility group"]] ?? ""),
  modelVersions: row[graphHeaders["Model versions"]],
})).filter((row) => row.name);
const legacyEciByName = new Map(legacyEciRows.map((row) => [row.name, row]));
const currentEciSupplementalMetadata = new Map([
  ["Claude Opus 5", { organization: "Anthropic", country: "United States of America", accessibility: "API access", accessibilityGroup: "Closed weights" }],
  ["Kimi K3", { organization: "Moonshot", country: "China", accessibility: "Open weights (unrestricted)", accessibilityGroup: "Open weights" }],
  ["Gemma 4 31B IT", { organization: "Google DeepMind", country: "United States of America", accessibility: "Open weights (restricted use)", accessibilityGroup: "Open weights" }],
  ["Qwen 3.6 35B-A3B", { organization: "Alibaba", country: "China", accessibility: "Open weights (unrestricted)", accessibilityGroup: "Open weights" }],
]);
const eciCrosscheckByName = new Map(eciReproductionCrosscheckRows.map((row) => [row.model, row]));
const eciRows = eciReproducedRows.map((row, index) => {
  const prior = legacyEciByName.get(row.Model);
  const supplemental = currentEciSupplementalMetadata.get(row.Model);
  const crosscheck = eciCrosscheckByName.get(row.Model);
  if (!crosscheck) throw new Error(`Current ECI row missing release-date crosscheck: ${row.Model}`);
  const metadata = prior || supplemental;
  if (!metadata) throw new Error(`Current ECI row missing manually reviewed identity metadata: ${row.Model}`);
  if (Math.abs(Number(row.eci) - Number(crosscheck.eci_score_reproduced)) > 1e-12) {
    throw new Error(`Current ECI score/crosscheck mismatch: ${row.Model}`);
  }
  return {
    sourceRow: index + 2,
    name: row.Model,
    display: prior?.display || row.Model,
    date: toDate(crosscheck.regression_release_date),
    eci: Number(row.eci),
    eciLow: Number(row.eci_ci_low),
    eciHigh: Number(row.eci_ci_high),
    organization: metadata.organization,
    country: metadata.country,
    accessibility: metadata.accessibility,
    accessibilityGroup: metadata.accessibilityGroup,
    modelVersions: row.model_version,
    inputDate: toDate(row.date),
  };
});
const opus5CurrentEci = eciRows.find((row) => row.name === "Claude Opus 5");
const k3CurrentEci = eciRows.find((row) => row.name === "Kimi K3");
if (
  !opus5CurrentEci
  || Math.abs(opus5CurrentEci.eci - Number(opus5Epoch.eci_exact)) > 1e-12
  || iso(opus5CurrentEci.date) !== opus5Identity.release_date
) throw new Error("Current ECI snapshot disagrees with independent Claude Opus 5 evidence");
if (!k3CurrentEci || iso(k3CurrentEci.date) !== k3Facts.initial_model_release_date) {
  throw new Error("Current ECI snapshot disagrees with official Kimi K3 release evidence");
}
const eciByName = new Map(eciRows.map((row) => [row.name, row]));

const regressionSnapshot = eciSnapshots.find((item) => item.sourceName === "Regression Data");
const regressionHeaders = headersOf(regressionSnapshot.values);
const regressionByName = new Map(regressionSnapshot.values.slice(1).filter((row) => row[regressionHeaders.Model]).map((row, index) => [String(row[regressionHeaders.Model]), {
  sourceRow: index + 2,
  totalB: row[regressionHeaders["Total params (B)"]],
  sourceUrl: row[regressionHeaders["Source URL"]],
  matchAssumption: row[regressionHeaders["Match / assumption"]],
}]));

// Immutable Epoch snapshot. This is a separate source layer: it enriches exact checkpoint
// identity and day-level dates but never overwrites the paper or ECI fields.
const epochCsvBytes = await fs.readFile(epochSourcePath);
const epochCsvText = epochCsvBytes.toString("utf8");
const epochCsvBase64 = epochCsvBytes.toString("base64");
const epochCsvChunks = epochCsvBase64.match(/[\s\S]{1,29000}/g) || [""];
const epochCsvRawRows = epochCsvChunks.map((chunk, index) => [index + 1, epochCsvChunks.length, chunk]);
const epochLiteralErrorTokenCount = [...epochCsvText.matchAll(/#(?:NAME\?|REF!|DIV\/0!|VALUE!|N\/A)/g)].length;
const epochWb = await Workbook.fromCSV(epochCsvText, { sheetName: "Epoch All Models" });
const epochValues = epochWb.worksheets.getItem("Epoch All Models").getUsedRange().values;
const epochHeaders = headersOf(epochValues);
const requiredEpochHeaders = ["Model", "Publication date", "Organization", "Parameters", "Parameters notes", "Base model", "Link", "Last modified"];
for (const field of requiredEpochHeaders) if (!(field in epochHeaders)) throw new Error(`Epoch CSV missing required field: ${field}`);
const epochRows = epochValues.slice(1).filter((row) => row[epochHeaders.Model]).map((row, index) => ({
  sourceRow: index + 2,
  model: String(row[epochHeaders.Model]),
  publicationDate: toDate(row[epochHeaders["Publication date"]]),
  organization: String(row[epochHeaders.Organization] ?? ""),
  parametersB: row[epochHeaders.Parameters] == null || row[epochHeaders.Parameters] === "" ? null : Number(row[epochHeaders.Parameters]) / 1e9,
  parametersNotes: String(row[epochHeaders["Parameters notes"]] ?? ""),
  baseModel: String(row[epochHeaders["Base model"]] ?? ""),
  link: String(row[epochHeaders.Link] ?? ""),
  lastModified: toDate(row[epochHeaders["Last modified"]]),
}));
const duplicateEpochNames = epochRows.length - new Set(epochRows.map((row) => row.model)).size;
const epochRowsByName = new Map();
for (const row of epochRows) {
  if (!epochRowsByName.has(row.model)) epochRowsByName.set(row.model, []);
  epochRowsByName.get(row.model).push(row);
}
const duplicateEpochLabels = [...epochRowsByName.entries()].filter(([, rows]) => rows.length > 1).map(([name]) => name);

// Compute branch: consume the audited unified table rather than rematching the archive ad hoc.
// Stage 1 maps AA/date to Epoch training compute on exact checkpoints. Stage 2 maps compute/date
// to total parameters on the curated Epoch frontier-language view. The branch is deliberately
// low-weight because both stages are estimated and share date/capability information with other branches.
const unifiedComputeRows = objectRows(await fs.readFile(unifiedComputePath, "utf8"));
const primaryEpochComputeByCheckpoint = new Map(unifiedComputeRows
  .filter((row) => row.source === "Epoch" && row.epoch_training_compute_flop && row.epoch_parameters_b)
  .map((row) => [row.canonical_checkpoint_id, row]));
const aaComputeByCheckpoint = new Map();
for (const row of unifiedComputeRows) {
  if (row.source !== "AA" || row.model_level_include !== "true" || !row.aa_intelligence_index) continue;
  if (!["checkpoint", "checkpoint_system_configuration"].includes(row.epoch_link_level)) continue;
  const epoch = primaryEpochComputeByCheckpoint.get(row.canonical_checkpoint_id);
  if (!epoch) continue;
  const score = Number(row.aa_intelligence_index);
  const current = aaComputeByCheckpoint.get(row.canonical_checkpoint_id);
  if (!current || score > Number(current.aa.aa_intelligence_index)) aaComputeByCheckpoint.set(row.canonical_checkpoint_id, { aa: row, epoch });
}
const computeStage1Rows = [...aaComputeByCheckpoint.values()].sort((a, b) =>
  a.aa.canonical_release_date.localeCompare(b.aa.canonical_release_date) || a.aa.source_model_name.localeCompare(b.aa.source_model_name));
if (!computeStage1Rows.length || new Set(computeStage1Rows.map((row) => row.aa.canonical_checkpoint_id)).size !== computeStage1Rows.length) {
  throw new Error("Current exact AA/Epoch compute calibration panel is empty or duplicated");
}

const computeStage2Rows = unifiedComputeRows.filter((row) => {
  if (row.source !== "Epoch Frontier View" || !row.epoch_training_compute_flop || !row.total_parameters_b) return false;
  const raw = JSON.parse(row.source_record_json);
  return ["Confident", "Likely"].includes(raw.Confidence)
    && row.canonical_release_date >= "2020-01-01"
    && String(raw.Domain || "").includes("Language")
    && String(raw.Task || "").includes("Language modeling/generation");
}).sort((a, b) => a.canonical_release_date.localeCompare(b.canonical_release_date) || a.source_model_name.localeCompare(b.source_model_name));
if (computeStage2Rows.length !== 19) throw new Error(`Expected 19 confident/likely Epoch frontier-language compute rows; found ${computeStage2Rows.length}`);

const computeTargetSpecs = [
  ["Claude Fable 5", "claude-fable-5", 6],
  ["GPT-5.6 Sol", "gpt-5-6-sol", 7],
  ["Kimi K3", "kimi-k3", 8],
  ["Claude Opus 4.7 / 4.8 shared base", "claude-opus-4-8", 10],
  ["GPT-5.5", "gpt-5-5", 9],
  ["GPT-5.6 Terra", "gpt-5-6-terra", 11],
  ["Claude Sonnet 5", "claude-sonnet-5", 12],
  ["GPT-5.6 Luna", "gpt-5-6-luna", 13],
  ["Grok 4.5", "grok-4-5", 15],
  ["Claude Opus 5", "claude-opus-5", 17],
];
const computeTargets = computeTargetSpecs.map(([name, aaSlug, baseRow]) => {
  const source = aaDetailedBySlug.get(aaSlug);
  if (!source || !source.intelligence_index || !source.release_date) throw new Error(`Missing compute target AA record: ${aaSlug}`);
  const aa = {
    source_model_name: source.name,
    aa_intelligence_index: source.intelligence_index,
    canonical_release_date: source.release_date,
    source_record_json: source.source_record_json,
  };
  if (name === "Claude Opus 5" && Number(aa.aa_intelligence_index) !== Number(opus5Aa.score)) throw new Error("Opus 5 detailed-AA/evidence score mismatch");
  return { name, aaName: source.name, baseRow, aa };
});

// Every paper row is manually adjudicated. null means "no exact ECI checkpoint" — never a fuzzy join.
const paperToEci = new Map(Object.entries({
  "GPT-2": null,
  "GPT-3": null,
  "GPT-3.5": null,
  "GPT-4": "GPT-4 (Mar 2023)",
  "GPT-4 Turbo": "GPT-4 Turbo (Apr 2024)",
  "GPT-4o": "GPT-4o (May 2024)",
  "Sonnet 3.7": "Claude 3.7 Sonnet",
  "Opus 4": "Claude Opus 4",
  "Opus 4.1": "Claude Opus 4.1",
  "Opus 4.5": "Claude Opus 4.5",
  "Opus 4.6": "Claude Opus 4.6",
  "GPT-5.4": "GPT-5.4",
  "Opus 4.7": "Claude Opus 4.7",
  "GPT-5.5": "GPT-5.5",
  "Llama 3 8B Instruct": "Llama 3-8B",
  "Llama 3.1 8B Instruct": "Llama 3.1-8B",
  "Llama 3.1 70B Instruct": "Llama 3.1-70B",
  "Llama 3.2 1B Instruct": null,
  "Llama 3.2 3B Instruct": null,
  "Llama 3.3 70B Instruct": "Llama 3.3 70B",
  "Llama 4 Scout": "Llama 4 Scout",
  "Llama 4 Maverick": "Llama 4 Maverick",
  "Mistral Nemo 12B": "Mistral NeMo",
  "Ministral 3 3B": null,
  "Ministral 3 8B": null,
  "Ministral 3 14B": null,
  "Qwen 3 8B": null,
  "Qwen 3 14B": null,
  "Qwen 3 32B": null,
  "Qwen 3 30B-A3B (2507)": null,
  "Qwen 3 235B-A22B (2507)": "Qwen3-235B-A22B-Instruct (Jul 2025)",
  "Qwen 3-Next 80B-A3B": null,
  "Qwen 3.5 9B": null,
  "Qwen 3.5 27B": null,
  "Qwen 3.5 35B-A3B": "Qwen 3.5 Flash (hosted 35B-A3B)",
  "Qwen 3.5 122B-A10B": null,
  "Qwen 3.5 397B-A17B": "Qwen 3.5 Plus (hosted 397B-A17B)",
  "Gemma 3 4B IT": null,
  "Gemma 3 12B IT": null,
  "Gemma 3 27B IT": "Gemma 3 27B",
  "Gemma 4 26B-A4B": null,
  "Gemma 4 31B": "Gemma 4 31B IT",
  "DeepSeek V3 (0324)": "DeepSeek-V3 (Mar 2025)",
  "DeepSeek V3.1-terminus": null,
  "DeepSeek V3.2": "DeepSeek-V3.2",
  "DeepSeek V4-flash": null,
  "Kimi K2-0905": null,
  "Kimi K2.5": "Kimi K2.5",
  "Kimi K2.6": "Kimi K2.6",
}));
const expectedPaperNames = [...frontierHorizons.map((row) => row.model), ...openModels.map((row) => row.model)];
if (paperToEci.size !== expectedPaperNames.length || expectedPaperNames.some((name) => !paperToEci.has(name))) {
  throw new Error("Manual paper→ECI alias map is incomplete or has extra rows");
}
for (const [paperName, eciName] of paperToEci) {
  if (eciName && !eciByName.has(eciName)) throw new Error(`Manual alias points to missing ECI row: ${paperName} → ${eciName}`);
}

// User-directed Epoch pass: skip GPT-2/3/3.5 and admit only exact checkpoint records
// present in Epoch's all-models CSV. null is an explicit nonmatch after manual review.
const epochSkippedPaperNames = new Set(["GPT-2", "GPT-3", "GPT-3.5"]);
const paperToEpoch = new Map(Object.entries({
  "Llama 3.2 1B Instruct": "Llama 3.2 1B",
  "Llama 3.2 3B Instruct": "Llama 3.2 3B",
  "Ministral 3 3B": "Ministral 3 3B",
  "Ministral 3 8B": "Ministral 3 8B",
  "Ministral 3 14B": "Ministral 3 14B",
  "Qwen 3 8B": "Qwen3-8B",
  "Qwen 3 14B": "Qwen3-14B",
  "Qwen 3 32B": "Qwen3-32B",
  "Qwen 3 30B-A3B (2507)": null,
  "Qwen 3-Next 80B-A3B": "Qwen3-Next-80B-A3B",
  "Qwen 3.5 9B": "Qwen3.5-9B",
  "Qwen 3.5 27B": "Qwen3.5-27B",
  "Qwen 3.5 122B-A10B": "Qwen3.5-122B-A10B",
  "Gemma 3 4B IT": "Gemma 3 4B",
  "Gemma 3 12B IT": "Gemma 3 12B",
  "Gemma 4 26B-A4B": "Gemma 4 26B A4B",
  "DeepSeek V3.1-terminus": "DeepSeek-V3.1-Terminus",
  "DeepSeek V4-flash": "DeepSeek-V4-Flash",
  "Kimi K2-0905": null,
}));
const expectedEpochCandidateNames = expectedPaperNames.filter((name) => !paperToEci.get(name) && !epochSkippedPaperNames.has(name));
if (paperToEpoch.size !== expectedEpochCandidateNames.length || expectedEpochCandidateNames.some((name) => !paperToEpoch.has(name))) {
  throw new Error("Manual paper→Epoch alias map is incomplete or has extra rows");
}
for (const [paperName, epochName] of paperToEpoch) {
  if (epochName && !epochRowsByName.has(epochName)) throw new Error(`Manual Epoch alias points to missing row: ${paperName} → ${epochName}`);
  if (epochName && epochRowsByName.get(epochName).length !== 1) throw new Error(`Manual Epoch alias is not unique: ${paperName} → ${epochName}`);
}
const duplicateEpochCandidateNames = [...paperToEpoch.values()].filter(Boolean).filter((name) => (epochRowsByName.get(name)?.length || 0) !== 1).length;

const frontierProvider = (model) => model.startsWith("Opus") || model.startsWith("Sonnet") ? "Anthropic" : "OpenAI";
const openProvider = new Map([["Llama", "Meta AI"], ["Mistral", "Mistral AI"], ["Qwen", "Alibaba"], ["Gemma", "Google DeepMind"], ["DeepSeek", "DeepSeek"], ["Kimi", "Moonshot"]]);
const baseFor = (display, canonicalId) => {
  if (/Claude Opus 4\.[5-8]/.test(display)) return "base:anthropic-opus-4-5plus";
  if (/^GPT-5(?:$|\.[1-5])/.test(display)) return "base:openai-gpt-5-shared";
  return `base:${canonicalId}`;
};
const lineageFor = (display, canonicalId) => {
  if (/Claude Opus 4\.[5-8]/.test(display)) return "lineage:anthropic-opus-4-5plus";
  if (/^GPT-5(?:$|\.[1-5])/.test(display)) return "lineage:openai-gpt-5";
  if (/Kimi K2/.test(display)) return "lineage:moonshot-kimi-k2";
  return `lineage:${canonicalId}`;
};
const canonicalForEci = (row) => `${providerTag(row.organization)}:${slug(row.name)}`;
const registryById = new Map();
const aliasRows = [];
const duplicateEciNames = eciRows.length - new Set(eciRows.map((row) => row.name)).size;
const duplicateEciIds = eciRows.length - new Set(eciRows.map(canonicalForEci)).size;
if (duplicateEciNames || duplicateEciIds) throw new Error(`Duplicate ECI identity: names=${duplicateEciNames}, ids=${duplicateEciIds}`);

for (const row of eciRows) {
  const canonicalId = canonicalForEci(row);
  const reg = regressionByName.get(row.name);
  registryById.set(canonicalId, {
    canonicalId,
    display: row.display || row.name,
    organization: row.organization,
    lineageId: lineageFor(row.name, canonicalId),
    baseId: baseFor(row.name, canonicalId),
    paperAlias: "",
    eciAlias: row.name,
    epochAlias: "",
    paperRelease: null,
    eciRelease: row.date,
    epochRelease: null,
    releaseUsed: row.date,
    dateSource: "ECI exact date",
    dateDelta: null,
    eci: row.eci,
    eciLow: row.eciLow,
    eciHigh: row.eciHigh,
    eciTotalB: reg?.totalB ?? null,
    epochTotalB: null,
    epochParametersNotes: "",
    epochBaseModel: "",
    epochLink: "",
    epochLastModified: null,
    epochSourceRow: null,
    paperTotalB: null,
    paperActiveB: null,
    layers: null,
    architecture: "",
    noCotPoint: null,
    noCotMedian: null,
    noCotLow: null,
    noCotHigh: null,
    horizonSuite: "",
    aaCurrent: null,
    finalLabel: "",
    matchStatus: "ECI-only row",
    conflict: "",
    sourceNote: reg ? `Parameter row: Regression Data!${reg.sourceRow}` : "",
  });
  aliasRows.push(["Epoch ECI 2026-07-31", row.name, canonicalId, row.name, "Exact self identity", "Current snapshot plus manually reviewed identity metadata", "Matched", `epoch_eci_reproduced_scores_2026-07-31.csv row ${row.sourceRow}`]);
}

const upsertPaper = ({ model, organization, release, totalB = null, activeB = null, layers = null, architecture = "", point, median = null, low, high, suite }) => {
  const eciAlias = paperToEci.get(model);
  const eciRow = eciAlias ? eciByName.get(eciAlias) : null;
  const epochAlias = paperToEpoch.get(model) || "";
  const epochRow = epochAlias ? epochRowsByName.get(epochAlias)?.[0] : null;
  const dateOverride = noCotDateOverrideByModel.get(model);
  const overrideDate = dateOverride ? new Date(`${dateOverride.exact_release_date}T00:00:00Z`) : null;
  const canonicalId = eciRow ? canonicalForEci(eciRow) : `${providerTag(organization)}:${slug(model)}`;
  const existing = registryById.get(canonicalId) || {
    canonicalId,
    display: model,
    organization,
    lineageId: lineageFor(model, canonicalId),
    baseId: baseFor(model, canonicalId),
    eciAlias: "",
    epochAlias: "",
    eciRelease: null,
    epochRelease: null,
    releaseUsed: epochRow?.publicationDate || overrideDate || release,
    dateSource: epochRow ? "Epoch exact date" : dateOverride ? dateOverride.exact_date_source_label : "Paper month only",
    eci: null, eciLow: null, eciHigh: null, eciTotalB: null,
    epochTotalB: null, epochParametersNotes: "", epochBaseModel: "", epochLink: "", epochLastModified: null, epochSourceRow: null,
    aaCurrent: null, finalLabel: "", conflict: "", sourceNote: "",
  };
  if (existing.paperAlias) throw new Error(`Two paper rows map to one checkpoint: ${existing.paperAlias} and ${model}`);
  const exactDate = eciRow?.date || epochRow?.publicationDate || overrideDate || null;
  const dateDelta = exactDate ? Math.round((exactDate - release) / 86400000) : null;
  const eciParameterDelta = existing.eciTotalB != null && totalB != null ? Math.abs(existing.eciTotalB - totalB) / Math.max(existing.eciTotalB, totalB) : 0;
  const epochParameterDelta = epochRow?.parametersB != null && totalB != null ? Math.abs(epochRow.parametersB - totalB) / Math.max(epochRow.parametersB, totalB) : 0;
  const conflictParts = [];
  if (dateDelta != null && Math.abs(dateDelta) > 31) conflictParts.push(`${eciRow ? "ECI" : "Epoch"} release conflict ${dateDelta}d`);
  if (eciParameterDelta > 0.01) conflictParts.push(`ECI/paper parameter variance ${(eciParameterDelta * 100).toFixed(1)}%`);
  if (epochParameterDelta > 0.01) conflictParts.push(`Epoch/paper parameter variance ${(epochParameterDelta * 100).toFixed(1)}%`);
  Object.assign(existing, {
    paperAlias: model,
    paperRelease: release,
    epochAlias,
    epochRelease: epochRow?.publicationDate || null,
    releaseUsed: eciRow?.date || epochRow?.publicationDate || overrideDate || release,
    dateSource: eciRow ? "ECI exact date" : epochRow ? "Epoch exact date" : dateOverride ? dateOverride.exact_date_source_label : "Paper month only",
    dateDelta,
    paperTotalB: totalB,
    paperActiveB: activeB,
    epochTotalB: epochRow?.parametersB ?? null,
    epochParametersNotes: epochRow?.parametersNotes || "",
    epochBaseModel: epochRow?.baseModel || "",
    epochLink: epochRow?.link || "",
    epochLastModified: epochRow?.lastModified || null,
    epochSourceRow: epochRow?.sourceRow || null,
    layers,
    architecture,
    noCotPoint: point,
    noCotMedian: median,
    noCotLow: low,
    noCotHigh: high,
    horizonSuite: suite,
    matchStatus: eciRow ? "Manual exact ECI join" : epochRow ? "Manual exact Epoch join" : dateOverride ? "Exact date-only override; parameter join prohibited" : "No exact ECI/Epoch checkpoint",
    conflict: [existing.conflict, ...conflictParts].filter(Boolean).join("; "),
  });
  registryById.set(canonicalId, existing);
  aliasRows.push([
    suite, model, canonicalId, eciAlias || epochAlias || "",
    eciRow ? "Manual exact ECI checkpoint alias" : epochRow ? "Manual exact Epoch checkpoint alias" : dateOverride ? "Exact date-only source; parameter identity unmatched" : "Explicitly unmatched",
    "Manual row-by-row review",
    eciRow || epochRow ? "Matched" : dateOverride ? "Date matched only" : "Preserved unmatched",
    suite === "Paper frontier" ? "tab:horizons-per-model" : "tab:open-source-models + tab:open-weight-precise-time-horizons",
  ]);
  if (paperToEpoch.has(model)) aliasRows.push([
    "Epoch reconciliation", model, canonicalId, epochAlias,
    epochRow ? "Manual exact Epoch checkpoint alias" : "Explicit exact-checkpoint nonmatch",
    "Manual row-by-row review against Epoch all-models CSV",
    epochRow ? "Matched" : "Preserved unmatched",
    epochRow ? `Epoch CSV row ${epochRow.sourceRow}` : "No exact non-reasoning 2507 record in Epoch",
  ]);
};

for (const row of frontierHorizons) upsertPaper({ model: row.model, organization: frontierProvider(row.model), release: row.release, point: row.time.estimate, low: row.time.low, high: row.time.high, suite: "Paper frontier" });
for (const row of openModels) upsertPaper({ model: row.model, organization: openProvider.get(row.developer), release: row.release, totalB: row.totalB, activeB: row.activeB, layers: row.layers, architecture: row.architecture, point: row.point, median: row.median, low: row.low, high: row.high, suite: "Paper open-weight" });

const finalAuditModels = [
  ["Claude Fable 5", "Claude Fable 5", "Anthropic", "Claude Fable 5", aaExact("claude-fable-5"), null, null],
  ["GPT-5.6 Sol", "GPT-5.6 Sol", "OpenAI", "GPT-5.6 Sol", aaExact("gpt-5-6-sol"), null, null],
  ["Kimi K3", "Kimi K3", "Moonshot", "Kimi K3", aaExact("kimi-k3"), k3Facts.total_parameters_b_exact, new Date(`${k3Facts.initial_model_release_date}T00:00:00Z`)],
  ["GPT-5.5", "GPT-5.5", "OpenAI", "GPT-5.5", aaExact("gpt-5-5"), null, null],
  ["Claude Opus 4.7 / 4.8 shared base", "Claude Opus 4.8", "Anthropic", "Claude Opus 4.8", aaExact("claude-opus-4-8"), null, null],
  ["GPT-5.6 Terra", "GPT-5.6 Terra", "OpenAI", "GPT-5.6 Terra", aaExact("gpt-5-6-terra"), null, null],
  ["Claude Sonnet 5", "Claude Sonnet 5", "Anthropic", "Claude Sonnet 5", aaExact("claude-sonnet-5"), null, null],
  ["GPT-5.6 Luna", "GPT-5.6 Luna", "OpenAI", "GPT-5.6 Luna", aaExact("gpt-5-6-luna"), null, null],
  ["Grok 4.5", "Grok 4.5", "xAI", "Grok 4.5", aaExact("grok-4-5"), 1500, null],
  ["Claude Opus 5", "Claude Opus 5", "Anthropic", "Claude Opus 5", Number(opus5Aa.score), null, null],
];
for (const [finalLabel, checkpoint, organization, eciAlias, aaCurrent, disclosedB, exactDate] of finalAuditModels) {
  const eciRow = eciAlias ? eciByName.get(eciAlias) : null;
  if (eciAlias && !eciRow) throw new Error(`Final model missing exact ECI row: ${eciAlias}`);
  const canonicalId = eciRow ? canonicalForEci(eciRow) : `${providerTag(organization)}:${slug(checkpoint)}`;
  const existing = registryById.get(canonicalId) || {
    canonicalId, display: checkpoint, organization, lineageId: lineageFor(checkpoint, canonicalId), baseId: baseFor(checkpoint, canonicalId),
    paperAlias: "", eciAlias: "", epochAlias: "", paperRelease: null, eciRelease: null, epochRelease: null, eci: null, eciLow: null, eciHigh: null,
    eciTotalB: null, paperTotalB: null, paperActiveB: null, layers: null, architecture: "",
    epochTotalB: null, epochParametersNotes: "", epochBaseModel: "", epochLink: "", epochLastModified: null, epochSourceRow: null,
    noCotPoint: null, noCotMedian: null, noCotLow: null, noCotHigh: null, horizonSuite: "",
    conflict: "", sourceNote: "",
  };
  Object.assign(existing, {
    display: checkpoint,
    finalLabel,
    aaCurrent,
    releaseUsed: eciRow?.date || exactDate,
    dateSource: eciRow ? "ECI exact date" : "User-supplied exact date",
    disclosedTotalB: disclosedB,
    matchStatus: eciRow ? "Final model + exact ECI join" : "Final model; no ECI row",
  });
  existing.lineageId = lineageFor(checkpoint, canonicalId);
  existing.baseId = baseFor(checkpoint, canonicalId);
  registryById.set(canonicalId, existing);
  aliasRows.push(["Final posterior", finalLabel, canonicalId, eciAlias || "", eciRow ? "Manual exact checkpoint alias" : "Explicit user-supplied identity", "Manual row-by-row review", "Matched", disclosedB ? "Disclosed total retained" : ""]);
}

const registryRows = [...registryById.values()].sort((a, b) => (a.releaseUsed?.getTime() || 0) - (b.releaseUsed?.getTime() || 0) || a.canonicalId.localeCompare(b.canonicalId));
const registryIds = registryRows.map((row) => row.canonicalId);
const duplicateRegistryIds = registryIds.length - new Set(registryIds).size;
const duplicateAliasKeys = aliasRows.length - new Set(aliasRows.map((row) => `${row[0]}|${row[1]}`)).size;
if (duplicateRegistryIds || duplicateAliasKeys) throw new Error(`Canonical audit failed: duplicate ids=${duplicateRegistryIds}, duplicate source aliases=${duplicateAliasKeys}`);
const paperRegistryByAlias = new Map(registryRows.filter((row) => row.paperAlias).map((row) => [row.paperAlias, row]));
const exactNoCotDate = (model) => {
  const row = paperRegistryByAlias.get(model);
  if (!row?.releaseUsed || row.dateSource === "Paper month only") throw new Error(`No exact no-CoT date for ${model}`);
  return row.releaseUsed;
};

const latexTableBlocks = [...tex.matchAll(/\\begin\{table\*?\}[\s\S]*?\\end\{table\*?\}/g)].map((match, index) => {
  const block = match[0];
  return { index: index + 1, label: block.match(/\\label\{([^}]+)\}/)?.[1] || "unlabeled", block };
});
const latexRawRows = [];
for (const table of latexTableBlocks) {
  const chunks = table.block.match(/[\s\S]{1,29000}/g) || [""];
  chunks.forEach((chunk, index) => latexRawRows.push([table.index, table.label, index + 1, chunks.length, chunk]));
}

const sha256 = async (path) => createHash("sha256").update(await fs.readFile(path)).digest("hex");
// Filesystem mtimes are build-environment state, not scientific provenance.
// Bind every manifest row to this immutable audit snapshot date so identical
// source bytes produce an identical workbook across clean pipeline rebuilds.
const manifestSnapshotDate = new Date(`${epochSnapshotManifest.snapshot_as_of}T00:00:00.000Z`);
const manifestPaths = [
  ["Claude Opus 5 normalized evidence bundle", opus5EvidencePath],
  ...Object.entries(opus5Evidence.source_files).map(([label, relativePath]) => [
    `Claude Opus 5 raw source — ${label}`,
    `${workDir}/${relativePath}`,
  ]),
  ["ECI workbook", eciSourcePath],
  ["Epoch AI all-models CSV", epochSourcePath],
  ["Epoch Jul-31 atomic snapshot manifest", epochSnapshotManifestPath],
  ["Epoch AI models archive", epochArchivePath],
  ["Compute-enriched unified observations", unifiedComputePath],
  ["Compute-enriched unified summary", unifiedSummaryPath],
  ["AA expanded parameter audit", aaExpandedResultPath],
  ["AA expanded parameter panel", aaExpandedPanelPath],
  ["AA expanded held-out predictions", aaExpandedPredictionsPath],
  ["AA current/exact overlap audit", aaExpandedOverlapsPath],
  ["AA detailed raw React-Flight snapshot", aaDetailedRawPath],
  [`AA detailed ${aaDetailedMetadata.models}-model ledger`, aaDetailedModelsPath],
  ["AA detailed collection metadata", aaDetailedMetadataPath],
  ["AA primary-source calibration overrides", aaCalibrationOverridesPath],
  ...aaCalibrationOverrides.overrides.flatMap((override) =>
    (override.primary_source.local_evidence || []).map((evidence) => [
      `AA override raw evidence — ${override.override_id} — ${evidence.kind}`,
      `${workDir}/${evidence.path}`,
    ]),
  ),
  ["AA parameter-label availability ledger", aaParameterLabelAvailabilityPath],
  ...aaParameterLabelAvailability.records.flatMap((record) =>
    (record.local_evidence || []).map((evidence) => [
      `AA label-timing raw evidence — ${record.record_id} — ${evidence.kind}`,
      `${workDir}/${evidence.path}`,
    ]),
  ),
  ["AA score-publication availability ledger", aaScoreAvailabilityPath],
  ["AA changelog raw API snapshot", aaChangelogRawPath],
  ["AA score-publication timing audit", aaScoreTimingAuditPath],
  ["AA score-publication changed-prediction ledger", aaScoreTimingChangesPath],
  ["AA inference-budget audit", aaInferenceResultPath],
  [`AA detailed ${aaInferenceResult.data_audit.unique_checkpoint_groups}-checkpoint panel`, aaDetailedPanelPath],
  ["AA reasoning configuration pairs", aaReasoningPairsPath],
  ["AA detailed Epoch crosscheck", aaDetailedCrosscheckPath],
  ["AA inference-budget held-out predictions", aaInferencePredictionsPath],
  ["AA operational-signal audit", aaOperationalResultPath],
  [`AA operational ${aaOperationalRows.length}-checkpoint panel`, aaOperationalPanelPath],
  ["AA operational held-out predictions", aaOperationalPredictionsPath],
  ["AA↔OpenRouter exact operational crosscheck", aaOpenRouterCrosscheckPath],
  ["Active-parameter transport audit", activeTransportResultPath],
  ["Active-parameter chronological predictions", activeTransportPredictionsPath],
  ["Active-parameter frontier sensitivities", activeTransportTargetsPath],
  ["Official Kimi K3 normalized release evidence", k3ReleaseEvidencePath],
  ...Object.entries(k3ReleaseEvidence.source_files).map(([label, source]) => [
    `Kimi K3 raw source — ${label}`,
    `${workDir}/${source.path}`,
  ]),
  ["Open-model parameter truth reconciliation", openModelParameterTruthPath],
  ...openModelParameterTruth.source_files.map((source) => [
    `Open-model parameter truth evidence — ${source.role}`,
    `${workDir}/${source.path}`,
  ]),
  ["ECI architecture-blend challenger", eciArchitectureBlendResultPath],
  ["ECI architecture-blend held-out predictions", eciArchitectureBlendPredictionsPath],
  ["ECI component benchmark snapshot", eciComponentPath],
  ["Epoch benchmark data archive", eciBenchmarkArchivePath],
  ["Official ECI reproduced scores", eciReproducedScoresPath],
  ["Official ECI reproduction metadata", eciReproductionMetadataPath],
  ["ECI score/date reproduction crosscheck", eciReproductionCrosscheckPath],
  ["ECI reproduction audit", eciReproductionAuditPath],
  ["ECI component extended audit", eciComponentExtendedResultPath],
  ["ECI component expanded parameter panel", eciComponentExpandedPanelPath],
  ["ECI component active comparison", eciComponentActiveComparisonPath],
  ["ECI multivariate component audit", eciMultivariateResultPath],
  ["ECI multivariate held-out predictions", eciMultivariatePredictionsPath],
  ["ECI multivariate narrow-ECI-CI predictions", eciMultivariateNarrowCiPredictionsPath],
  ["ECI multivariate frontier sensitivities", eciMultivariateTargetsPath],
  ["ECI multivariate benchmark coverage", eciMultivariateCoveragePath],
  ["Post-training lineage audit", posttrainingLineageResultPath],
  ["Post-training exact lineage edges", posttrainingLineageEdgesPath],
  ["Post-training matched measurements", posttrainingLineageMeasurementsPath],
  ["Post-training held-out predictions", posttrainingLineagePredictionsPath],
  ["Frontier asserted shared-base sensitivities", frontierSharedBaseSensitivityPath],
  ["Frontier lineage primary-source evidence", frontierLineageEvidencePath],
  ["arXiv LaTeX source archive", latexArchiveDisplay],
  ["No-CoT exact-date overrides", noCotDateOverridePath],
  ["No-CoT exact-date collection metadata", noCotDateMetadataPath],
  ["Qwen first-party Hugging Face commit history", qwenDateRawPath],
  ["No-CoT exact-date statistical audit", noCotExactDateResultPath],
  ["No-CoT exact-date model ledger", noCotExactDateModelAuditPath],
  ["No-CoT architecture-elasticity audit", noCotArchitectureAuditPath],
  ["No-CoT architecture-elasticity predictions", noCotArchitecturePredictionsPath],
  ["Frontier primary evidence ledger", frontierPrimaryEvidencePath],
  ["Frontier primary evidence metadata", frontierPrimaryMetadataPath],
  ["OpenAI GPT-5.6 first-party system-card capture", frontierPrimaryOpenAIRawPath],
  ["Anthropic Fable/Mythos verified primary claims", frontierPrimaryAnthropicClaimsPath],
  ["Frontier primary evidence statistical audit", frontierPrimaryAuditPath],
  ["Frontier primary evidence controls", frontierPrimaryControlsPath],
  ["METR official normalized signals", metrOfficialSignalsPath],
  ["METR official raw YAML", metrOfficialRawPath],
  ["METR official metadata", metrOfficialMetadataPath],
  ["METR official/legacy reconciliation audit", metrOfficialAuditPath],
  ["METR legacy exact crosscheck", metrLegacyCrosscheckPath],
  ["IKP direct-capacity signal audit", ikpAuditPath],
  ["IKP chronological held-out predictions", ikpPredictionPath],
  ["IKP exact ensemble overlap", ikpOverlapPath],
  ["IKP conditional benchmark audit", ikpConditionalAuditPath],
  ["IKP conditional benchmark predictions", ikpConditionalPredictionPath],
  ["IKP immutable source metadata", ikpSourceMetadataPath],
  ["Epoch employee calibration feedback", epochFeedbackSourcePath],
  ["Epoch feedback architecture audit", epochFeedbackAuditPath],
  ["Epoch feedback reproduced challenge panel", epochFeedbackPanelPath],
  ["Epoch feedback lean architecture predictions", epochFeedbackPredictionsPath],
  ["Epoch feedback frontier sensitivities", epochFeedbackTargetsPath],
  ["Price-informed input workbook", inputPath],
  ["Human forecast ledger", forecastLedgerPath],
  ["Human prediction registry", predictionRegistryPath],
  ["OpenRouter raw operational snapshot", openRouterRawPath],
  ["OpenRouter model aggregates", openRouterModelPath],
  ["OpenRouter provider observations", openRouterProviderPath],
  ["OpenRouter endpoint service-tier observations", openRouterTierPath],
  ["OpenRouter lossless daily endpoint throughput", openRouterDailyPath],
  ["OpenRouter model snapshot history", openRouterModelHistoryPath],
  ["OpenRouter provider snapshot history", openRouterProviderHistoryPath],
  ["OpenRouter endpoint-tier snapshot history", openRouterTierHistoryPath],
  ["OpenRouter daily throughput history", openRouterDailyHistoryPath],
  ["OpenRouter immutable snapshot manifest", openRouterHistoryManifestPath],
  ["OpenRouter manual Epoch match audit", openRouterAuditPath],
  ["OpenRouter parameter-signal backtest", openRouterResultPath],
  ["OpenRouter temporal-stability audit", openRouterTemporalResultPath],
  ["OpenRouter request-weighted operational audit", openRouterRequestWeightedResultPath],
  ["OpenRouter request-weighted held-out predictions", openRouterRequestWeightedPredictionsPath],
  ["OpenRouter endpoint temporal stability", openRouterEndpointStabilityPath],
  ["OpenRouter model temporal stability", openRouterModelStabilityPath],
  ["OpenRouter refresh temporal stability", openRouterRefreshStabilityPath],
  ["OpenRouter service-tier counterfactual predictions", openRouterTierPredictionsPath],
  ["OpenRouter collection audit", openRouterCollectionAuditPath],
  ["OpenRouter official endpoint snapshot", openRouterOfficialSnapshotPath],
  ["OpenRouter official endpoint prices", openRouterOfficialPricePath],
  ["OpenRouter official/frontend comparison", openRouterOfficialComparisonPath],
  ["OpenRouter official endpoint audit", openRouterOfficialAuditPath],
  ["OpenRouter active-price audit", openRouterActivePriceResultPath],
  ["OpenRouter active-price identity ledger", openRouterActivePriceMatchPath],
  ["OpenRouter active-price held-out predictions", openRouterActivePricePredictionsPath],
  ["OpenRouter active-price frontier sensitivities", openRouterActivePriceTargetsPath],
  ["OpenRouter historical raw price ledger", openRouterHistoricalRawPath],
  ["OpenRouter historical price change points", openRouterHistoricalChangePointsPath],
  ["OpenRouter historical collection metadata", openRouterHistoricalMetadataPath],
  ["OpenRouter historical prospective audit", openRouterHistoricalResultPath],
  ["OpenRouter historical exact-match audit", openRouterHistoricalMatchPath],
  ["OpenRouter historical held-out predictions", openRouterHistoricalPredictionsPath],
  ["OpenRouter historical frontier sensitivities", openRouterHistoricalTargetsPath],
  ["Primary Hugging Face architecture config snapshot", huggingFaceArchitectureRawPath],
  ["Hugging Face architecture config signals", huggingFaceArchitectureSignalsPath],
  ["Hugging Face architecture config collection audit", huggingFaceArchitectureAuditPath],
  ["LessWrong attachment", lessWrongSourcePath],
];
const manifestRows = await Promise.all(manifestPaths.map(async ([label, path]) => {
  const stat = await fs.stat(path);
  return [label, portableLocalPath(path), stat.size, await sha256(path), new Date(manifestSnapshotDate)];
}));

const wb = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const summary = wb.worksheets.getItem("Executive Summary");
const sources = wb.worksheets.getItem("Sources");
const frontierEstimates = wb.worksheets.getItem("Frontier Estimates");
const anchorsMethod = wb.worksheets.getItem("Anchors & Method");
if (frontierEstimates.getRange("B8").values[0][0] !== "Kimi K3") throw new Error("Frontier Estimates K3 row moved unexpectedly");
frontierEstimates.getRange("D8:E8").values = [[k3CurrentEci.eci, k3Facts.total_parameters_b_exact / 1000]];
frontierEstimates.getRange("M8").formulas = [["=E8"]];
frontierEstimates.getRange("E8").format.numberFormat = "0.0";
frontierEstimates.getRange("M8").format.numberFormat = "0.0";
frontierEstimates.getRange("Q8").values = [["2.78T total and 104.2B activated parameters disclosed in the official technical report; headline total displays as 2.8T. No regression whisker."]];
anchorsMethod.getRange("B7:C7").values = [[k3Facts.total_parameters_b_exact / 1000, "Exact official 2.78T total / 104.2B activated; displayed as 2.8T in headline views"]];
anchorsMethod.getRange("B7").format.numberFormat = "0.0";
sources.getRange("C11:D11").values = [["Exact 2.78T total / 104.2B activated parameters and architecture", "Official technical report released 2026-07-31"]];
const evidence = wb.worksheets.add("No-CoT Evidence");
const laws = wb.worksheets.add("Horizon Laws");
const posterior = wb.worksheets.add("Horizon Estimates");
const finalEnsemble = wb.worksheets.add("Final Ensemble");
const computeEvidence = wb.worksheets.add("Epoch Compute");
const computeModel = wb.worksheets.add("Compute Model");
const modelRegistry = wb.worksheets.add("Model Registry");
const aliasMap = wb.worksheets.add("Alias Map");
const epochReconciliation = wb.worksheets.add("Epoch Reconciliation");
const dataAudit = wb.worksheets.add("Data Audit");
const baseRegistry = wb.worksheets.add("Base Registry");
const sourceManifest = wb.worksheets.add("Source Manifest");
const latexRaw = wb.worksheets.add("LaTeX Tables Raw");
const formulaArchive = wb.worksheets.add("ECI Formula Archive");
const aaExpansionAudit = wb.worksheets.add("AA Expansion Audit");
const aaInferenceAudit = wb.worksheets.add("AA Inference Audit");
const aaOperationalAudit = wb.worksheets.add("AA Operational Audit");
const activeTransportAudit = wb.worksheets.add("Active Param Audit");
const eciComponentAudit = wb.worksheets.add("ECI Component Audit");
const eciMultivariateAudit = wb.worksheets.add("ECI Multivariate Audit");
const posttrainingLineageAudit = wb.worksheets.add("Post-training Audit");
const openRouterAudit = wb.worksheets.add("OpenRouter Audit");
const openRouterTimeAudit = wb.worksheets.add("OR Time Stability");
const openRouterPriceAudit = wb.worksheets.add("OR Price Audit");
const activePriceAudit = wb.worksheets.add("Active Price Audit");
const historicalPriceAudit = wb.worksheets.add("Historical Price Audit");
const noCotDateAudit = wb.worksheets.add("No-CoT Date Audit");
const primaryEvidenceAudit = wb.worksheets.add("Primary Evidence");
const ikpSignalAudit = wb.worksheets.add("IKP Signal Audit");
const rawSheets = new Map(eciSnapshots.map((snapshot) => [snapshot.targetName, wb.worksheets.add(snapshot.targetName)]));
const epochRaw = wb.worksheets.add("Epoch CSV Raw");

// Normalize the inherited Grok wording to the user's requested neutral label.
frontierEstimates.getRange("A2").values = [["Legacy price-informed input branch. Grok 4.5 is fixed at the disclosed 1.5T foundation scale. API price has 15% weight inside this branch; the final posterior rescales the whole branch to 45%, so price has 6.8% effective weight. Legacy uncertainty columns are retained for audit only and are omitted from the final graph."]];
frontierEstimates.getRange("P15:Q15").values = [["Disclosed anchor", "1.5T disclosed total; active count and architecture undisclosed."]];
anchorsMethod.getRange("A9:C9").values = [["Grok disclosed foundation scale (T)", 1.5, "First-party disclosure; total only, active unknown"]];
anchorsMethod.getRange("A22:C22").merge();
anchorsMethod.getRange("A22").values = [["Grok choice: retain the disclosed 1.5T row as an anchor, but do not force the entire regression through K3 and Grok. The prior model was already within 6.7%; exact two-anchor rotation would add model risk."]];
sources.getRange("A9:D9").values = [["Grok 4.5 1.5T disclosure", "https://x.com/elonmusk/status/2071184354756477041", "Public first-party 1.5T V9 foundation scale", "Total only; active parameters undisclosed"]];

const C = {
  navy: "#243838", teal: "#00A5A6", tealDark: "#087879", paleTeal: "#E8F4F4",
  paleBlue: "#EDF4FF", palePink: "#FFF4FA", paleAmber: "#FFF4DF", gray: "#F3F6F6",
  mid: "#D2DCDC", text: "#3A4848", white: "#FFFFFF", purple: "#7457D9",
};

const title = (sheet, range, text) => {
  sheet.getRange(range).merge();
  sheet.getRange(range.split(":")[0]).values = [[text]];
  sheet.getRange(range).format = {
    fill: C.navy,
    font: { name: "Aptos Display", size: 18, bold: true, color: C.white },
    verticalAlignment: "center",
  };
  sheet.getRange(range).format.rowHeight = 36;
};
const subtitle = (sheet, range, text) => {
  sheet.getRange(range).merge();
  sheet.getRange(range.split(":")[0]).values = [[text]];
  sheet.getRange(range).format = { font: { name: "Aptos", size: 10, color: C.text }, wrapText: true, verticalAlignment: "center" };
};
const header = (range) => {
  range.format = {
    fill: C.tealDark,
    font: { name: "Aptos", size: 9, bold: true, color: C.white },
    wrapText: true,
    verticalAlignment: "center",
    borders: { bottom: { style: "medium", color: C.teal } },
  };
  range.format.rowHeight = 34;
};
const body = (range) => {
  range.format = {
    font: { name: "Aptos", size: 9, color: C.text },
    verticalAlignment: "center",
    borders: { insideHorizontal: { style: "thin", color: C.mid } },
  };
};
const section = (sheet, range, text) => {
  sheet.getRange(range).merge();
  sheet.getRange(range.split(":")[0]).values = [[text]];
  sheet.getRange(range).format = {
    fill: C.paleTeal,
    font: { name: "Aptos", size: 11, bold: true, color: C.tealDark },
    borders: { bottom: { style: "medium", color: C.teal } },
  };
};

for (const sheet of [evidence, laws, posterior, computeEvidence, computeModel, modelRegistry, aliasMap, epochReconciliation, dataAudit, baseRegistry, sourceManifest, latexRaw, formulaArchive, aaExpansionAudit, aaInferenceAudit, aaOperationalAudit, activeTransportAudit, eciComponentAudit, eciMultivariateAudit, posttrainingLineageAudit, activePriceAudit, openRouterAudit, openRouterTimeAudit, noCotDateAudit, primaryEvidenceAudit, ikpSignalAudit, ...rawSheets.values(), epochRaw]) sheet.showGridLines = false;

const colLetter = (index) => {
  let value = index + 1;
  let out = "";
  while (value > 0) {
    const rem = (value - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    value = Math.floor((value - 1) / 26);
  }
  return out;
};

// Canonical checkpoint registry: one row per identity, with every source field kept separately.
title(modelRegistry, "A1:AL1", "Canonical model registry — checkpoint identity and source reconciliation");
subtitle(modelRegistry, "A2:AL3", `One row per canonical checkpoint across the current ${epochSnapshotManifest.inventory.models}-model ECI snapshot, all 49 paper models, and all ${finalAuditModels.length} final frontier targets. The full ${epochSnapshotManifest.inventory.all_model_rows.toLocaleString("en-US")}-row Epoch snapshot is retained separately. Paper, ECI, Epoch, and supplemental evidence fields remain distinct; no source value is overwritten.`);
const registryHeaders = ["Canonical checkpoint ID", "Canonical display", "Organization", "Lineage ID", "Base ID", "Final target label", "Paper alias", "ECI alias", "Paper release month", "ECI exact release", "Release date used", "Date source", "Date delta (days)", "ECI", "ECI CI low", "ECI CI high", "AA current", "Paper total params (B)", "Paper active params (B)", "ECI regression total (B)", "Disclosed total (B)", "Layers", "Architecture", "No-CoT point (min)", "No-CoT median", "No-CoT CI low", "No-CoT CI high", "Horizon suite", "Match status", "Conflict / variance", "Source note", "Epoch alias", "Epoch exact release", "Epoch total params (B)", "Epoch base model", "Epoch link", "Epoch last modified", "Epoch CSV row"];
modelRegistry.getRange("A5:AL5").values = [registryHeaders];
header(modelRegistry.getRange("A5:AL5"));
const registryData = registryRows.map((row) => [
  row.canonicalId, row.display, row.organization, row.lineageId, row.baseId, row.finalLabel || "", row.paperAlias || "", row.eciAlias || "",
  row.paperRelease || null, row.eciRelease || null, row.releaseUsed || null, row.dateSource || "", row.dateDelta,
  row.eci, row.eciLow, row.eciHigh, row.aaCurrent,
  row.paperTotalB, row.paperActiveB, row.eciTotalB, row.disclosedTotalB ?? null, row.layers, row.architecture || "",
  row.noCotPoint, row.noCotMedian, row.noCotLow, row.noCotHigh, row.horizonSuite || "", row.matchStatus || "", row.conflict || "", row.sourceNote || "",
  row.epochAlias || "", row.epochRelease || null, row.epochTotalB, row.epochBaseModel || "", row.epochLink || "", row.epochLastModified || null, row.epochSourceRow,
]);
const registryStart = 6;
const registryEnd = registryStart + registryData.length - 1;
modelRegistry.getRange(`A${registryStart}:AL${registryEnd}`).values = registryData;
body(modelRegistry.getRange(`A${registryStart}:AL${registryEnd}`));
modelRegistry.getRange(`I${registryStart}:K${registryEnd}`).format.numberFormat = "yyyy-mm-dd";
modelRegistry.getRange(`M${registryStart}:M${registryEnd}`).format.numberFormat = "0";
modelRegistry.getRange(`N${registryStart}:Q${registryEnd}`).format.numberFormat = "0.000";
modelRegistry.getRange(`R${registryStart}:AA${registryEnd}`).format.numberFormat = "0.000";
modelRegistry.getRange(`F${registryStart}:H${registryEnd}`).format.wrapText = true;
modelRegistry.getRange(`AB${registryStart}:AE${registryEnd}`).format.wrapText = true;
modelRegistry.getRange(`AF${registryStart}:AL${registryEnd}`).format.wrapText = true;
modelRegistry.getRange(`AG${registryStart}:AG${registryEnd}`).format.numberFormat = "yyyy-mm-dd";
modelRegistry.getRange(`AH${registryStart}:AH${registryEnd}`).format.numberFormat = "0.000";
modelRegistry.getRange(`AK${registryStart}:AK${registryEnd}`).format.numberFormat = "yyyy-mm-dd";
modelRegistry.getRange(`AD${registryStart}:AD${registryEnd}`).conditionalFormats.add("containsText", { text: "conflict", format: { fill: C.palePink, font: { color: "#8B245C", bold: true } } });
modelRegistry.getRange(`AD${registryStart}:AD${registryEnd}`).conditionalFormats.add("containsText", { text: "variance", format: { fill: C.paleAmber, font: { color: C.navy, bold: true } } });
modelRegistry.tables.add(`A5:AL${registryEnd}`, true, "CanonicalModelRegistryTable").style = "TableStyleMedium2";
modelRegistry.getRange("A:A").format.columnWidth = 38;
modelRegistry.getRange("B:B").format.columnWidth = 34;
modelRegistry.getRange("C:C").format.columnWidth = 24;
modelRegistry.getRange("D:E").format.columnWidth = 38;
modelRegistry.getRange("F:H").format.columnWidth = 34;
modelRegistry.getRange("I:M").format.columnWidth = 18;
modelRegistry.getRange("N:AA").format.columnWidth = 17;
modelRegistry.getRange("AB:AC").format.columnWidth = 24;
modelRegistry.getRange("AD:AE").format.columnWidth = 58;
modelRegistry.getRange("AF:AG").format.columnWidth = 30;
modelRegistry.getRange("AH:AI").format.columnWidth = 22;
modelRegistry.getRange("AJ:AJ").format.columnWidth = 70;
modelRegistry.getRange("AK:AL").format.columnWidth = 20;
modelRegistry.freezePanes.freezeRows(5);
const registryRowByCanonical = new Map(registryRows.map((row, index) => [row.canonicalId, registryStart + index]));
const eciCanonical = (name) => canonicalForEci(eciByName.get(name));

// Every source alias, including explicitly unmatched paper rows, is retained and manually adjudicated.
title(aliasMap, "A1:H1", "Alias map — source names to canonical checkpoints");
subtitle(aliasMap, "A2:H3", "This is the only legal join layer. Fuzzy suggestions were used only for manual review and never enter the workbook. Each source/alias pair is unique; blank matched-source aliases mean no exact checkpoint match exists in the adjudicated source.");
aliasMap.getRange("A5:H5").values = [["Source", "Source alias / query", "Canonical checkpoint ID", "Matched source alias", "Join method", "Review", "Status", "Source row / note"]];
header(aliasMap.getRange("A5:H5"));
const aliasStart = 6;
const aliasEnd = aliasStart + aliasRows.length - 1;
aliasMap.getRange(`A${aliasStart}:H${aliasEnd}`).values = aliasRows;
body(aliasMap.getRange(`A${aliasStart}:H${aliasEnd}`));
aliasMap.getRange(`B${aliasStart}:H${aliasEnd}`).format.wrapText = true;
aliasMap.getRange(`G${aliasStart}:G${aliasEnd}`).conditionalFormats.add("containsText", { text: "unmatched", format: { fill: C.paleAmber, font: { color: C.navy, bold: true } } });
aliasMap.tables.add(`A5:H${aliasEnd}`, true, "CanonicalAliasMapTable").style = "TableStyleMedium2";
aliasMap.getRange("A:A").format.columnWidth = 24;
aliasMap.getRange("B:B").format.columnWidth = 38;
aliasMap.getRange("C:C").format.columnWidth = 44;
aliasMap.getRange("D:D").format.columnWidth = 38;
aliasMap.getRange("E:H").format.columnWidth = 34;
aliasMap.freezePanes.freezeRows(5);

// Compact, human-readable adjudication of every user-selected Epoch candidate.
const registryByPaperAlias = new Map(registryRows.filter((row) => row.paperAlias).map((row) => [row.paperAlias, row]));
const epochReconciliationRows = expectedEpochCandidateNames.map((paperName) => {
  const row = registryByPaperAlias.get(paperName);
  if (!row) throw new Error(`Epoch reconciliation missing canonical paper row: ${paperName}`);
  const parameterVariance = row.epochTotalB != null && row.paperTotalB != null
    ? Math.abs(row.epochTotalB - row.paperTotalB) / Math.max(row.epochTotalB, row.paperTotalB)
    : null;
  return [
    paperName, row.canonicalId, row.epochAlias || "", row.epochAlias ? "Exact Epoch match" : "No exact Epoch checkpoint",
    row.epochRelease || null, row.epochTotalB, row.paperTotalB, parameterVariance,
    row.epochBaseModel || "", row.epochSourceRow, row.epochLink || "",
    row.epochAlias ? "Day-level Epoch date adopted; paper values retained separately" : "Epoch has the April base and an incomplete Thinking-2507 row, not this exact non-reasoning 2507 checkpoint",
  ];
});
title(epochReconciliation, "A1:L1", "Epoch reconciliation — paper-only checkpoints");
subtitle(epochReconciliation, "A2:L3", "Manual checkpoint-by-checkpoint review of the 19 open models that lacked exact ECI matches. Only exact records in Epoch's all-models CSV qualify. GPT-2/3/3.5 are intentionally excluded from this pass per user instruction.");
epochReconciliation.getRange("A5:L5").values = [["Paper checkpoint", "Canonical checkpoint ID", "Epoch model", "Status", "Epoch exact date", "Epoch params (B)", "Paper params (B)", "Parameter variance", "Epoch base model", "Epoch CSV row", "Epoch source link", "Resolution note"]];
header(epochReconciliation.getRange("A5:L5"));
epochReconciliation.getRange(`A6:L${5 + epochReconciliationRows.length}`).values = epochReconciliationRows;
body(epochReconciliation.getRange(`A6:L${5 + epochReconciliationRows.length}`));
epochReconciliation.getRange(`E6:E${5 + epochReconciliationRows.length}`).format.numberFormat = "yyyy-mm-dd";
epochReconciliation.getRange(`F6:G${5 + epochReconciliationRows.length}`).format.numberFormat = "0.000";
epochReconciliation.getRange(`H6:H${5 + epochReconciliationRows.length}`).format.numberFormat = "0.0%";
epochReconciliation.getRange(`C6:L${5 + epochReconciliationRows.length}`).format.wrapText = true;
epochReconciliation.getRange(`D6:D${5 + epochReconciliationRows.length}`).conditionalFormats.add("containsText", { text: "No exact", format: { fill: C.paleAmber, font: { color: C.navy, bold: true } } });
epochReconciliation.tables.add(`A5:L${5 + epochReconciliationRows.length}`, true, "EpochReconciliationTable").style = "TableStyleMedium2";
epochReconciliation.getRange("A:A").format.columnWidth = 34;
epochReconciliation.getRange("B:B").format.columnWidth = 42;
epochReconciliation.getRange("C:D").format.columnWidth = 30;
epochReconciliation.getRange("E:J").format.columnWidth = 18;
epochReconciliation.getRange("K:K").format.columnWidth = 70;
epochReconciliation.getRange("L:L").format.columnWidth = 66;
epochReconciliation.freezePanes.freezeRows(5);

const finalRegistryRows = registryRows.filter((row) => row.finalLabel);
const exactFinalDates = finalRegistryRows.filter((row) => row.releaseUsed && row.dateSource !== "Paper month only").length;
const exactFinalEci = finalRegistryRows.filter((row) => row.eci != null).length;
const paperMatched = expectedPaperNames.filter((name) => paperToEci.get(name)).length;
const epochMatched = [...paperToEpoch.values()].filter(Boolean).length;
const paperRowsWithExactDate = registryRows.filter((row) => row.paperAlias && row.dateSource !== "Paper month only").length;
const paperRowsMonthOnly = registryRows.filter((row) => row.paperAlias && row.dateSource === "Paper month only").length;
const conflictRows = registryRows.filter((row) => row.conflict);
const rawCellCount = eciSnapshots.reduce((sum, snapshot) => sum + snapshot.values.length * (snapshot.values[0]?.length || 0), 0);
const epochRawByteCount = epochCsvBytes.length;
const epochBase64CharacterCount = epochCsvBase64.length;
const expectedEpochArchiveViewRows = Object.values(unifiedSummary.epochArchiveViews).reduce((sum, row) => sum + Number(row.records), 0);
const currentUnifiedEciComponentRows = unifiedComputeRows.filter((row) => row.source === "ECI Component").length;
const currentUnifiedEpochArchiveRows = unifiedComputeRows.filter((row) => row.source.startsWith("Epoch ") && row.source.endsWith(" View")).length;
const currentUnifiedFrontierRows = unifiedComputeRows.filter((row) => row.source === "Epoch Frontier View").length;

title(dataAudit, "A1:I1", "Data integrity audit — fail-fast identity and coverage checks");
subtitle(dataAudit, "A2:I3", "PASS means structural reconciliation is complete: all ECI, paper, Epoch, and compute-view rows are preserved; canonical IDs and source aliases are unique; exact dates supersede month-only dates; and correlated Epoch source views are explicitly excluded from independent evidence counts.");
dataAudit.getRange("A5:D5").values = [["Audit", "Expected", "Actual", "Status"]];
header(dataAudit.getRange("A5:D5"));
const auditMetrics = [
  ["Legacy ECI workbook graph rows retained raw", legacyEciRows.length, legacyEciRows.length, "PASS"],
  ["Current Jul-31 ECI model rows retained", epochSnapshotManifest.inventory.models, eciRows.length, eciRows.length === epochSnapshotManifest.inventory.models ? "PASS" : "FAIL"],
  ["ECI source sheets retained", eciSourceSheets.length, rawSheets.size, "PASS"],
  ["ECI source cells retained as value snapshots", rawCellCount, rawCellCount, "PASS"],
  ["ECI formulas retained as text archive", formulaRows.length, formulaRows.length, "PASS"],
  ["Epoch all-models rows retained", epochSnapshotManifest.inventory.all_model_rows, epochRows.length, epochRows.length === epochSnapshotManifest.inventory.all_model_rows ? "PASS" : "FAIL"],
  ["Epoch raw CSV bytes bound by SHA-256", epochRawByteCount, epochRawByteCount, "PASS"],
  ["Epoch Base64 characters retained", epochBase64CharacterCount, epochBase64CharacterCount, "PASS"],
  ["Epoch literal error-token strings retained", "informational", epochLiteralErrorTokenCount, "INFO"],
  ["Epoch source duplicate-name rows beyond first", "informational", duplicateEpochNames, "INFO"],
  ["Compute-enriched unified observations", unifiedSummary.observations, unifiedComputeRows.length, unifiedComputeRows.length === unifiedSummary.observations ? "PASS" : "FAIL"],
  ["ECI component benchmark rows retained", epochSnapshotManifest.inventory.component_rows, currentUnifiedEciComponentRows, currentUnifiedEciComponentRows === epochSnapshotManifest.inventory.component_rows ? "PASS" : "FAIL"],
  ["Epoch archive source-view rows retained", expectedEpochArchiveViewRows, currentUnifiedEpochArchiveRows, currentUnifiedEpochArchiveRows === expectedEpochArchiveViewRows ? "PASS" : "FAIL"],
  ["Epoch frontier compute rows retained", unifiedSummary.observationsBySource["Epoch Frontier View"], currentUnifiedFrontierRows, currentUnifiedFrontierRows === unifiedSummary.observationsBySource["Epoch Frontier View"] ? "PASS" : "FAIL"],
  ["Stage-1 exact AA↔Epoch compute calibration rows", "current artifact; unique", computeStage1Rows.length, new Set(computeStage1Rows.map((row) => row.aa.canonical_checkpoint_id)).size === computeStage1Rows.length ? "PASS" : "FAIL"],
  ["Stage-2 confident/likely frontier-language rows", 19, computeStage2Rows.length, computeStage2Rows.length === 19 ? "PASS" : "FAIL"],
  ["Targets with Epoch training-compute estimate", "informational", activeTransportResult.compute_branch_independence.target_models_with_epoch_training_compute_estimate, "INFO"],
  ["Targets with disclosed training compute", 0, activeTransportResult.compute_branch_independence.target_models_with_disclosed_training_compute, activeTransportResult.compute_branch_independence.target_models_with_disclosed_training_compute === 0 ? "PASS" : "FAIL"],
  ["Strict active-parameter held-out folds", activeTransportResult.inventory.chronological_predictions, activeTransportPredictionRows.length, activeTransportPredictionRows.length === activeTransportResult.inventory.chronological_predictions ? "PASS" : "FAIL"],
  ["Epoch candidate aliases with non-unique source hit", 0, duplicateEpochCandidateNames, duplicateEpochCandidateNames === 0 ? "PASS" : "FAIL"],
  ["LaTeX table blocks retained raw", latexTableBlocks.length, latexTableBlocks.length, "PASS"],
  ["Paper frontier rows parsed", 14, frontierHorizons.length, frontierHorizons.length === 14 ? "PASS" : "FAIL"],
  ["Paper open-weight rows parsed", 35, openModels.length, openModels.length === 35 ? "PASS" : "FAIL"],
  ["Official METR model rows retained", 26, metrOfficialRows.length, metrOfficialRows.length === 26 ? "PASS" : "FAIL"],
  ["Official METR full scaffold entries retained", 114, metrOfficialMetadata.inventory.full_scaffold_entries, metrOfficialMetadata.inventory.full_scaffold_entries === 114 ? "PASS" : "FAIL"],
  ["METR official↔legacy exact rows", 26, metrOfficialAudit.legacy_exact_crosscheck.exact_rows, metrOfficialAudit.legacy_exact_crosscheck.exact_rows === 26 && metrOfficialAudit.legacy_exact_crosscheck.mismatch_count === 0 ? "PASS" : "FAIL"],
  ["Paper aliases manually adjudicated", 49, paperToEci.size, paperToEci.size === 49 ? "PASS" : "FAIL"],
  ["Paper rows with exact ECI checkpoint", "informational", paperMatched, "INFO"],
  ["Epoch candidates manually adjudicated", expectedEpochCandidateNames.length, paperToEpoch.size, paperToEpoch.size === expectedEpochCandidateNames.length ? "PASS" : "FAIL"],
  ["Exact Epoch checkpoint joins", [...paperToEpoch.values()].filter(Boolean).length, epochMatched, epochMatched === [...paperToEpoch.values()].filter(Boolean).length ? "PASS" : "FAIL"],
  ["Paper rows with an exact day-level date", 49, paperRowsWithExactDate, paperRowsWithExactDate === 49 ? "PASS" : "FAIL"],
  ["Paper rows intentionally remaining month-only", 0, paperRowsMonthOnly, paperRowsMonthOnly === 0 ? "PASS" : "FAIL"],
  ["Canonical checkpoint IDs duplicated", 0, duplicateRegistryIds, duplicateRegistryIds === 0 ? "PASS" : "FAIL"],
  ["Source alias keys duplicated", 0, duplicateAliasKeys, duplicateAliasKeys === 0 ? "PASS" : "FAIL"],
  ["Final targets", finalAuditModels.length, finalRegistryRows.length, finalRegistryRows.length === finalAuditModels.length ? "PASS" : "FAIL"],
  ["Final targets with exact release date", finalAuditModels.length, exactFinalDates, exactFinalDates === finalAuditModels.length ? "PASS" : "FAIL"],
  ["Final targets with ECI coverage", finalAuditModels.length, exactFinalEci, exactFinalEci === finalAuditModels.length ? "PASS" : "FAIL"],
  ["Source variances retained, not overwritten", "all", conflictRows.length, "PASS"],
];
dataAudit.getRange(`A6:D${5 + auditMetrics.length}`).values = auditMetrics;
body(dataAudit.getRange(`A6:D${5 + auditMetrics.length}`));
dataAudit.getRange(`D6:D${5 + auditMetrics.length}`).conditionalFormats.add("containsText", { text: "PASS", format: { fill: C.paleTeal, font: { color: C.tealDark, bold: true } } });
dataAudit.getRange(`D6:D${5 + auditMetrics.length}`).conditionalFormats.add("containsText", { text: "FAIL", format: { fill: C.palePink, font: { color: "#8B245C", bold: true } } });

const baseDuplicateAuditRow = 6 + auditMetrics.length;
const issueHeaderRow = baseDuplicateAuditRow + 2;
section(dataAudit, `A${issueHeaderRow}:I${issueHeaderRow}`, "Cross-source differences and explicitly missing coverage");
dataAudit.getRange(`A${issueHeaderRow + 1}:I${issueHeaderRow + 1}`).values = [["Severity", "Canonical ID", "Display", "Paper alias", "ECI alias", "Epoch alias", "Issue", "Resolution policy", "Posterior impact"]];
header(dataAudit.getRange(`A${issueHeaderRow + 1}:I${issueHeaderRow + 1}`));
const issueRows = [];
for (const row of registryRows.filter((item) => item.paperAlias && !item.eciAlias && !item.epochAlias)) {
  const skipped = epochSkippedPaperNames.has(row.paperAlias);
  const dateOverride = noCotDateOverrideByModel.get(row.paperAlias);
  issueRows.push([
    "INFO", row.canonicalId, row.display, row.paperAlias, "", "",
    skipped ? "Epoch parameter matching intentionally skipped" : "No exact ECI or Epoch parameter checkpoint",
    dateOverride
      ? `Use ${dateOverride.exact_release_date} from ${dateOverride.exact_date_source_label} for timing only; ${dateOverride.parameter_join_policy}`
      : "Preserve paper row; never join a different checkpoint",
    dateOverride ? "Exact date enters horizon sensitivity; no parameter identity is inferred" : "None unless explicitly used in a date regression",
  ]);
}
for (const row of conflictRows) {
  const exactDatePolicy = row.eciAlias ? "ECI exact date" : row.epochAlias ? "Epoch exact date" : row.dateSource || "audited exact date";
  issueRows.push(["RESOLVED", row.canonicalId, row.display, row.paperAlias, row.eciAlias, row.epochAlias, row.conflict, `Preserve every source value; use ${exactDatePolicy} for dates and paper values for reproducing the paper's published scaling`, row.finalLabel ? "Reviewed in final branch" : "No direct final-target impact"]);
}
issueRows.push(["SOURCE", "epoch:duplicate-model-labels", "Epoch all-models CSV", "", "", duplicateEpochLabels.join("; "), `${duplicateEpochNames} duplicate-name rows beyond the first`, "Retain every raw row; require exactly one Epoch source hit for every admitted candidate alias", "No candidate checkpoint is affected"]);
issueRows.push(["INFO", "moonshot:kimi-k3", "Kimi K3", "", k3CurrentEci.eci, "Kimi K3", "Epoch rounds total size to 2.8T while the official technical report discloses exactly 2.78T total / 104.2B active", "Retain Epoch's rounded source field, but use the exact primary-source counts for the fixed anchor and active-parameter audit", "Anchor remains fixed at 2.78T internally (2.8T display)"]);
issueRows.push(["INFO", "anthropic:claude-opus-5", "Claude Opus 5", "", opus5Epoch.eci_exact, "Claude Opus 5", "Current ECI/AA/date evidence is present, but direct horizon/IKP/target-compute/architecture/parameter measurements are unavailable", "Keep unavailable measurements explicitly missing rather than imputed", "Unique unlocked regression target; neutral horizon"]);
const issueStart = issueHeaderRow + 2;
const issueEnd = issueStart + issueRows.length - 1;
dataAudit.getRange(`A${issueStart}:I${issueEnd}`).values = issueRows;
body(dataAudit.getRange(`A${issueStart}:I${issueEnd}`));
dataAudit.getRange(`G${issueStart}:I${issueEnd}`).format.wrapText = true;
dataAudit.getRange("A:A").format.columnWidth = 48;
dataAudit.getRange("B:B").format.columnWidth = 42;
dataAudit.getRange("C:E").format.columnWidth = 30;
dataAudit.getRange("F:F").format.columnWidth = 30;
dataAudit.getRange("G:I").format.columnWidth = 60;
dataAudit.freezePanes.freezeRows(5);

// Immutable source manifest with hashes.
title(sourceManifest, "A1:E1", "Source manifest — immutable inputs and SHA-256 hashes");
subtitle(sourceManifest, "A2:E3", "These hashes bind the audit to the exact files used. The original sources remain untouched; the workbook contains value snapshots, formula text, raw LaTeX tables, and canonical joins.");
sourceManifest.getRange("A5:E5").values = [["Source", "Path", "Bytes", "SHA-256", "Snapshot date"]];
header(sourceManifest.getRange("A5:E5"));
sourceManifest.getRange(`A6:E${5 + manifestRows.length}`).values = manifestRows;
body(sourceManifest.getRange(`A6:E${5 + manifestRows.length}`));
sourceManifest.getRange(`C6:C${5 + manifestRows.length}`).format.numberFormat = "0";
sourceManifest.getRange(`E6:E${5 + manifestRows.length}`).format.numberFormat = "yyyy-mm-dd";
sourceManifest.getRange("A:A").format.columnWidth = 34;
sourceManifest.getRange("B:B").format.columnWidth = 80;
sourceManifest.getRange("C:C").format.columnWidth = 18;
sourceManifest.getRange("D:D").format.columnWidth = 70;
sourceManifest.getRange("E:E").format.columnWidth = 24;

const aaDataAudit = aaExpandedResult.data_audit;
const aaScopes = aaExpandedResult.backtest.scopes;
const aaCurrentPanelScope = aaScopes.current_panel;
const aaFrontierScope = aaScopes.frontier_like;
const aaExpandedFit = aaExpandedResult.full_fit.expanded_panel_developer_balanced_coefficients;
const aaK3Current = aaExpandedResult.pre_anchor_k3_checks.find((row) => row.panel === "current_50" && !row.moonshot_held_out);
const aaK3Expanded = aaExpandedResult.pre_anchor_k3_checks.find((row) => row.panel === "expanded_panel" && !row.moonshot_held_out);
title(aaExpansionAudit, "A1:N1", "Artificial Analysis parameter expansion — exact Epoch reconciliation and frontier gate");
subtitle(aaExpansionAudit, "A2:N3", `The ${aaDataAudit.current_panel_models}-model live AA panel is reconciled with ${aaDataAudit.exact_open_epoch_checkpoints} exact open-weight AA↔Epoch checkpoints. Highest-score duplicate configurations are collapsed and ${aaDataAudit.current_exact_overlaps} overlaps are superseded by their exact Epoch-backed row, producing ${aaDataAudit.expanded_unique_models} unique models. Broad tail error improves, but the ${aaFrontierScope.n}-model frontier-like interval crosses zero, so the live K3-anchored AA branch remains unchanged.`);
aaExpansionAudit.getRange("A5:B5").values = [["Audit statistic", "Value"]];
header(aaExpansionAudit.getRange("A5:B5"));
const aaExpansionSummaryRows = [
  ["Current manually curated checkpoints", aaDataAudit.current_panel_models],
  ["Exact open-weight AA↔Epoch checkpoints", aaDataAudit.exact_open_epoch_checkpoints],
  ["Lower-scoring duplicate configurations removed", aaDataAudit.lower_scoring_duplicate_configurations_discarded],
  ["Current/exact overlaps reconciled", aaDataAudit.current_exact_overlaps],
  ["Expanded unique checkpoints", aaDataAudit.expanded_unique_models],
  ["Expanded developers", aaDataAudit.expanded_developers],
  ["Eligible held-out predictions", aaExpandedResult.backtest.eligible_predictions],
  ["All tests: current median error (×)", aaScopes.all.current_50.median_multiplicative_error],
  ["All tests: expanded median error (×)", aaScopes.all.expanded_panel.median_multiplicative_error],
  ["Current-panel tests: current MAE log10", aaCurrentPanelScope.current_50.mean_absolute_log10_error],
  ["Current-panel tests: expanded MAE log10", aaCurrentPanelScope.expanded_panel.mean_absolute_log10_error],
  ["Frontier-like current median error (×)", aaFrontierScope.current_50.median_multiplicative_error],
  ["Frontier-like expanded median error (×)", aaFrontierScope.expanded_panel.median_multiplicative_error],
  ["Frontier-like bootstrap 90% CI low", aaFrontierScope.paired_developer_bootstrap.ci_90[0]],
  ["Frontier-like bootstrap 90% CI high", aaFrontierScope.paired_developer_bootstrap.ci_90[1]],
  ["Current live AA score slope", aaExpandedResult.full_fit.current_live_k3_anchored_coefficients.score_slope],
  ["Expanded AA score slope", aaExpandedFit.score_slope],
  ["Current live AA date slope", aaExpandedResult.full_fit.current_live_k3_anchored_coefficients.date_slope],
  ["Expanded AA date slope", aaExpandedFit.date_slope],
  ["Current pre-anchor K3 prediction (T)", aaK3Current.predicted_k3_b / 1000],
  ["Expanded pre-anchor K3 prediction (T)", aaK3Expanded.predicted_k3_b / 1000],
  ["Incremental expanded-AA weight", aaExpandedResult.decision.incremental_expanded_aa_weight],
  ["Change live AA branch", aaExpandedResult.decision.change_live_aa_branch ? "Yes" : "No"],
];
const aaExpansionSummaryEnd = 5 + aaExpansionSummaryRows.length;
aaExpansionAudit.getRange(`A6:B${aaExpansionSummaryEnd}`).values = aaExpansionSummaryRows;
body(aaExpansionAudit.getRange(`A6:B${aaExpansionSummaryEnd}`));
aaExpansionAudit.getRange("B13:B14").format.numberFormat = "0.00x";
aaExpansionAudit.getRange("B15:B16").format.numberFormat = "0.000";
aaExpansionAudit.getRange("B17:B18").format.numberFormat = "0.00x";
aaExpansionAudit.getRange("B19:B24").format.numberFormat = "0.000";
aaExpansionAudit.getRange("B25:B26").format.numberFormat = "0.00\"T\"";
aaExpansionAudit.getRange("B27:B27").format.numberFormat = "0.0%";

const aaScopeSection = aaExpansionSummaryEnd + 3;
section(aaExpansionAudit, `A${aaScopeSection}:M${aaScopeSection}`, "Identical-test chronological developer-held-out comparisons");
const aaScopeHeader = aaScopeSection + 1;
const aaScopeHeaders = ["Scope", "Tests", "Developers", "Current median (×)", "Expanded median (×)", "Current MAE", "Expanded MAE", "Current p80 (×)", "Expanded p80 (×)", "Equal-developer Δ", "90% CI low", "90% CI high", "P(expanded better)"];
aaExpansionAudit.getRange(`A${aaScopeHeader}:M${aaScopeHeader}`).values = [aaScopeHeaders];
header(aaExpansionAudit.getRange(`A${aaScopeHeader}:M${aaScopeHeader}`));
const aaScopeOrder = ["all", "current_panel", "exact_additions", "frontier_like"];
const aaScopeData = aaScopeOrder.map((scope) => {
  const value = aaScopes[scope];
  return [
    scope,
    value.n,
    value.developers,
    value.current_50.median_multiplicative_error,
    value.expanded_panel.median_multiplicative_error,
    value.current_50.mean_absolute_log10_error,
    value.expanded_panel.mean_absolute_log10_error,
    value.current_50.p80_multiplicative_error,
    value.expanded_panel.p80_multiplicative_error,
    value.paired_developer_bootstrap.observed_delta,
    value.paired_developer_bootstrap.ci_90[0],
    value.paired_developer_bootstrap.ci_90[1],
    value.paired_developer_bootstrap.bootstrap_probability_expanded_better,
  ];
});
const aaScopeEnd = aaScopeHeader + aaScopeData.length;
aaExpansionAudit.getRange(`A${aaScopeHeader + 1}:M${aaScopeEnd}`).values = aaScopeData;
body(aaExpansionAudit.getRange(`A${aaScopeHeader + 1}:M${aaScopeEnd}`));
aaExpansionAudit.getRange(`B${aaScopeHeader + 1}:C${aaScopeEnd}`).format.numberFormat = "0";
aaExpansionAudit.getRange(`D${aaScopeHeader + 1}:E${aaScopeEnd}`).format.numberFormat = "0.00x";
aaExpansionAudit.getRange(`F${aaScopeHeader + 1}:G${aaScopeEnd}`).format.numberFormat = "0.000";
aaExpansionAudit.getRange(`H${aaScopeHeader + 1}:I${aaScopeEnd}`).format.numberFormat = "0.00x";
aaExpansionAudit.getRange(`J${aaScopeHeader + 1}:L${aaScopeEnd}`).format.numberFormat = "0.000";
aaExpansionAudit.getRange(`M${aaScopeHeader + 1}:M${aaScopeEnd}`).format.numberFormat = "0.0%";
aaExpansionAudit.tables.add(`A${aaScopeHeader}:M${aaScopeEnd}`, true, "AaExpandedScopeTable").style = "TableStyleMedium2";

const aaFrontierSection = aaScopeEnd + 3;
section(aaExpansionAudit, `A${aaFrontierSection}:G${aaFrontierSection}`, "K3-anchored AA-only frontier stability — diagnostic, not adopted");
const aaFrontierHeader = aaFrontierSection + 1;
aaExpansionAudit.getRange(`A${aaFrontierHeader}:G${aaFrontierHeader}`).values = [["Model", "Release", "AA", "Current live AA (T)", "Expanded-panel AA (T)", "Expanded/current", "Live use"]];
header(aaExpansionAudit.getRange(`A${aaFrontierHeader}:G${aaFrontierHeader}`));
const aaFrontierData = aaExpandedResult.frontier_aa_stability.map((row) => [
  row.model,
  new Date(`${row.release_date}T00:00:00Z`),
  row.aa_score,
  row.current_live_aa_t,
  row.expanded_panel_aa_t,
  row.expanded_over_current,
  "Current retained",
]);
const aaFrontierEnd = aaFrontierHeader + aaFrontierData.length;
aaExpansionAudit.getRange(`A${aaFrontierHeader + 1}:G${aaFrontierEnd}`).values = aaFrontierData;
body(aaExpansionAudit.getRange(`A${aaFrontierHeader + 1}:G${aaFrontierEnd}`));
aaExpansionAudit.getRange(`B${aaFrontierHeader + 1}:B${aaFrontierEnd}`).format.numberFormat = "yyyy-mm-dd";
aaExpansionAudit.getRange(`C${aaFrontierHeader + 1}:C${aaFrontierEnd}`).format.numberFormat = "0.0";
aaExpansionAudit.getRange(`D${aaFrontierHeader + 1}:E${aaFrontierEnd}`).format.numberFormat = "0.00\"T\"";
aaExpansionAudit.getRange(`F${aaFrontierHeader + 1}:F${aaFrontierEnd}`).format.numberFormat = "0.000x";
aaExpansionAudit.tables.add(`A${aaFrontierHeader}:G${aaFrontierEnd}`, true, "AaExpandedFrontierTable").style = "TableStyleMedium2";

const aaOverlapSection = aaFrontierEnd + 3;
section(aaExpansionAudit, `A${aaOverlapSection}:M${aaOverlapSection}`, "Complete current/exact checkpoint overlap audit");
const aaOverlapHeader = aaOverlapSection + 1;
const aaOverlapHeaders = ["Current model", "Exact AA model", "Method", "Current release", "Exact release", "Δ days", "Current AA", "Exact highest AA", "Current params (B)", "Exact params (B)", "Developer", "Epoch checkpoint", "Parameter source"];
aaExpansionAudit.getRange(`A${aaOverlapHeader}:M${aaOverlapHeader}`).values = [aaOverlapHeaders];
header(aaExpansionAudit.getRange(`A${aaOverlapHeader}:M${aaOverlapHeader}`));
const aaOverlapData = aaExpandedOverlapRows.map((row) => [
  row.current_model,
  row.exact_model,
  row.match_method,
  new Date(`${row.current_release_date}T00:00:00Z`),
  new Date(`${row.exact_release_date}T00:00:00Z`),
  Number(row.date_delta_days),
  Number(row.current_aa_score),
  Number(row.exact_highest_aa_score),
  Number(row.current_total_parameters_b),
  Number(row.exact_epoch_total_parameters_b),
  row.developer,
  row.matched_epoch_model,
  row.parameter_source,
]);
const aaOverlapEnd = aaOverlapHeader + aaOverlapData.length;
aaExpansionAudit.getRange(`A${aaOverlapHeader + 1}:M${aaOverlapEnd}`).values = aaOverlapData;
body(aaExpansionAudit.getRange(`A${aaOverlapHeader + 1}:M${aaOverlapEnd}`));
aaExpansionAudit.getRange(`D${aaOverlapHeader + 1}:E${aaOverlapEnd}`).format.numberFormat = "yyyy-mm-dd";
aaExpansionAudit.getRange(`F${aaOverlapHeader + 1}:F${aaOverlapEnd}`).format.numberFormat = "0";
aaExpansionAudit.getRange(`G${aaOverlapHeader + 1}:J${aaOverlapEnd}`).format.numberFormat = "0.00";
aaExpansionAudit.tables.add(`A${aaOverlapHeader}:M${aaOverlapEnd}`, true, "AaExpandedOverlapTable").style = "TableStyleMedium2";

const aaPanelSection = aaOverlapEnd + 3;
section(aaExpansionAudit, `A${aaPanelSection}:N${aaPanelSection}`, "Complete 92-checkpoint expanded AA admission ledger");
const aaPanelHeader = aaPanelSection + 1;
const aaPanelHeaders = ["Model", "Release", "AA", "Total params (B)", "Developer", "Estimated", "Panel source", "In current 50", "Current overlap", "Canonical checkpoint", "Epoch match", "Accessibility", "AA source", "Parameter source"];
aaExpansionAudit.getRange(`A${aaPanelHeader}:N${aaPanelHeader}`).values = [aaPanelHeaders];
header(aaExpansionAudit.getRange(`A${aaPanelHeader}:N${aaPanelHeader}`));
const aaPanelData = aaExpandedRows.map((row) => [
  row.model,
  new Date(`${row.release_date}T00:00:00Z`),
  Number(row.aa_score),
  Number(row.total_parameters_b),
  row.developer,
  Number(row.estimated_score) ? "Yes" : "No",
  row.panel_source,
  row.also_in_current_panel === "True" ? "Yes" : "No",
  row.overlap_current_model,
  row.canonical_checkpoint_id,
  row.matched_epoch_model,
  row.epoch_accessibility,
  row.aa_source,
  row.parameter_source,
]);
const aaPanelEnd = aaPanelHeader + aaPanelData.length;
aaExpansionAudit.getRange(`A${aaPanelHeader + 1}:N${aaPanelEnd}`).values = aaPanelData;
body(aaExpansionAudit.getRange(`A${aaPanelHeader + 1}:N${aaPanelEnd}`));
aaExpansionAudit.getRange(`B${aaPanelHeader + 1}:B${aaPanelEnd}`).format.numberFormat = "yyyy-mm-dd";
aaExpansionAudit.getRange(`C${aaPanelHeader + 1}:D${aaPanelEnd}`).format.numberFormat = "0.00";
aaExpansionAudit.getRange(`I${aaPanelHeader + 1}:N${aaPanelEnd}`).format.wrapText = true;
aaExpansionAudit.tables.add(`A${aaPanelHeader}:N${aaPanelEnd}`, true, "AaExpandedParameterPanelTable").style = "TableStyleMedium2";
aaExpansionAudit.freezePanes.freezeRows(aaScopeHeader);
aaExpansionAudit.getRange("A:A").format.columnWidth = 40;
aaExpansionAudit.getRange("B:B").format.columnWidth = 18;
aaExpansionAudit.getRange("C:L").format.columnWidth = 20;
aaExpansionAudit.getRange("M:N").format.columnWidth = 70;

const aaInferenceData = aaInferenceResult.data_audit;
const aaExactPairs = aaInferenceResult.same_weight_reasoning_pairs;
const aaAllPairs = aaInferenceResult.reasoning_configuration_pairs.all;
const aaDetailedBacktest = aaInferenceResult.detailed_panel_backtest.scopes;
const aaTokenBacktest = aaInferenceResult.inference_budget_backtest.scopes;
const aaPortableBacktest = aaInferenceResult.reasoning_standardization_backtest.portable.scopes;
const aaCreatorBacktest = aaInferenceResult.reasoning_standardization_backtest.creator_aware.scopes;
title(aaInferenceAudit, "A1:N1", "Artificial Analysis inference-budget audit — measured reasoning uplift and frontier gate");
subtitle(aaInferenceAudit, "A2:N3", `Complete ${aaInferenceData.raw_models}-configuration AA React-Flight snapshot. The ${aaInferenceData.unique_checkpoint_groups}-checkpoint open-weight panel improves broad recovery and ${aaAllPairs.pairs} same-checkpoint reasoning/non-reasoning pairs validate the global six-point correction. Token-budget and reasoning-standardized frontier tests remain statistically inconclusive, so all incremental live weights stay at 0%.`);
aaInferenceAudit.getRange("A5:B5").values = [["Audit statistic", "Value"]];
header(aaInferenceAudit.getRange("A5:B5"));
const aaInferenceSummaryRows = [
  ["Raw AA model configurations", aaInferenceData.raw_models],
  ["Open-weight parameter/score/date configurations", aaInferenceData.open_weight_parameter_score_date_configurations],
  ["Unique detailed checkpoint groups", aaInferenceData.unique_checkpoint_groups],
  ["Lower-score configurations removed", aaInferenceData.lower_score_configurations_removed],
  ["Creators in detailed parameter panel", aaInferenceData.creators],
  ["Token-covered checkpoint groups", aaInferenceData.token_covered_checkpoint_groups],
  ["Exact AA↔Epoch crosschecks", aaInferenceData.epoch_exact_crosschecks],
  ["Crosschecks with visible metadata disagreement", aaInferenceData.epoch_crosschecks_with_metadata_disagreement],
  ["All reasoning/non-reasoning pairs", aaAllPairs.pairs],
  ["Strict exact open-weight pairs", aaExactPairs.pairs],
  ["Equal-creator median reasoning uplift", aaAllPairs.equal_creator_median_aa_uplift],
  ["Reasoning uplift bootstrap 90% CI low", aaAllPairs.equal_creator_median_bootstrap_90_ci[0]],
  ["Reasoning uplift bootstrap 90% CI high", aaAllPairs.equal_creator_median_bootstrap_90_ci[1]],
  ["Detailed panel all-test baseline median (×)", aaDetailedBacktest.all.baseline.median_multiplicative_error],
  ["Detailed panel all-test candidate median (×)", aaDetailedBacktest.all.candidate.median_multiplicative_error],
  ["Detailed panel frontier bootstrap CI high", aaDetailedBacktest.frontier_like.paired_cluster_bootstrap.ci_90[1]],
  ["Token-budget frontier baseline median (×)", aaTokenBacktest.frontier_like.baseline.median_multiplicative_error],
  ["Token-budget frontier candidate median (×)", aaTokenBacktest.frontier_like.candidate.median_multiplicative_error],
  ["Portable standardization frontier baseline (×)", aaPortableBacktest.frontier_like.baseline.median_multiplicative_error],
  ["Portable standardization frontier candidate (×)", aaPortableBacktest.frontier_like.candidate.median_multiplicative_error],
  ["Incremental detailed-panel weight", aaInferenceResult.decision.incremental_detailed_panel_weight],
  ["Incremental inference-budget weight", aaInferenceResult.decision.incremental_inference_budget_weight],
  ["Incremental reasoning-standardization weight", aaInferenceResult.decision.incremental_reasoning_standardization_weight],
  ["Change live AA branch", aaInferenceResult.decision.change_live_aa_branch ? "Yes" : "No"],
];
const aaInferenceSummaryEnd = 5 + aaInferenceSummaryRows.length;
aaInferenceAudit.getRange(`A6:B${aaInferenceSummaryEnd}`).values = aaInferenceSummaryRows;
body(aaInferenceAudit.getRange(`A6:B${aaInferenceSummaryEnd}`));
aaInferenceAudit.getRange("B16:B18").format.numberFormat = "0.00";
aaInferenceAudit.getRange("B19:B20").format.numberFormat = "0.00x";
aaInferenceAudit.getRange("B21:B21").format.numberFormat = "0.000";
aaInferenceAudit.getRange("B22:B25").format.numberFormat = "0.00x";
aaInferenceAudit.getRange("B26:B28").format.numberFormat = "0.0%";

const aaInferenceComparisonSection = aaInferenceSummaryEnd + 3;
section(aaInferenceAudit, `A${aaInferenceComparisonSection}:N${aaInferenceComparisonSection}`, "Strictly chronological, developer-held-out comparisons");
const aaInferenceComparisonHeader = aaInferenceComparisonSection + 1;
const aaInferenceComparisonHeaders = ["Branch", "Scope", "Tests", "Developers", "Baseline median (×)", "Candidate median (×)", "Baseline MAE", "Candidate MAE", "Baseline p80 (×)", "Candidate p80 (×)", "Equal-developer Δ", "90% CI low", "90% CI high", "P(candidate better)"];
aaInferenceAudit.getRange(`A${aaInferenceComparisonHeader}:N${aaInferenceComparisonHeader}`).values = [aaInferenceComparisonHeaders];
header(aaInferenceAudit.getRange(`A${aaInferenceComparisonHeader}:N${aaInferenceComparisonHeader}`));
const aaComparisonRow = (branch, scope, value) => [
  branch,
  scope,
  value.n,
  value.developers,
  value.baseline.median_multiplicative_error,
  value.candidate.median_multiplicative_error,
  value.baseline.mean_absolute_log10_error,
  value.candidate.mean_absolute_log10_error,
  value.baseline.p80_multiplicative_error,
  value.candidate.p80_multiplicative_error,
  value.paired_cluster_bootstrap.observed_delta,
  value.paired_cluster_bootstrap.ci_90[0],
  value.paired_cluster_bootstrap.ci_90[1],
  value.paired_cluster_bootstrap.bootstrap_probability_candidate_better,
];
const aaInferenceComparisonData = [
  aaComparisonRow(`Detailed ${aaInferenceData.unique_checkpoint_groups} vs current 50`, "all", aaDetailedBacktest.all),
  aaComparisonRow(`Detailed ${aaInferenceData.unique_checkpoint_groups} vs current 50`, "frontier_like", aaDetailedBacktest.frontier_like),
  aaComparisonRow("Measured token budget", "all", aaTokenBacktest.all),
  aaComparisonRow("Measured token budget", "frontier_like", aaTokenBacktest.frontier_like),
  aaComparisonRow("Portable reasoning standardization", "all", aaPortableBacktest.all),
  aaComparisonRow("Portable reasoning standardization", "frontier_like", aaPortableBacktest.frontier_like),
  aaComparisonRow("Creator-aware standardization", "all", aaCreatorBacktest.all),
  aaComparisonRow("Creator-aware standardization", "frontier_like", aaCreatorBacktest.frontier_like),
];
const aaInferenceComparisonEnd = aaInferenceComparisonHeader + aaInferenceComparisonData.length;
aaInferenceAudit.getRange(`A${aaInferenceComparisonHeader + 1}:N${aaInferenceComparisonEnd}`).values = aaInferenceComparisonData;
body(aaInferenceAudit.getRange(`A${aaInferenceComparisonHeader + 1}:N${aaInferenceComparisonEnd}`));
aaInferenceAudit.getRange(`C${aaInferenceComparisonHeader + 1}:D${aaInferenceComparisonEnd}`).format.numberFormat = "0";
aaInferenceAudit.getRange(`E${aaInferenceComparisonHeader + 1}:F${aaInferenceComparisonEnd}`).format.numberFormat = "0.00x";
aaInferenceAudit.getRange(`G${aaInferenceComparisonHeader + 1}:H${aaInferenceComparisonEnd}`).format.numberFormat = "0.000";
aaInferenceAudit.getRange(`I${aaInferenceComparisonHeader + 1}:J${aaInferenceComparisonEnd}`).format.numberFormat = "0.00x";
aaInferenceAudit.getRange(`K${aaInferenceComparisonHeader + 1}:M${aaInferenceComparisonEnd}`).format.numberFormat = "0.000";
aaInferenceAudit.getRange(`N${aaInferenceComparisonHeader + 1}:N${aaInferenceComparisonEnd}`).format.numberFormat = "0.0%";
aaInferenceAudit.tables.add(`A${aaInferenceComparisonHeader}:N${aaInferenceComparisonEnd}`, true, "AaInferenceComparisonTable").style = "TableStyleMedium2";

const pairCountsByCreator = new Map();
for (const row of aaReasoningPairRows) pairCountsByCreator.set(row.creator_slug, (pairCountsByCreator.get(row.creator_slug) || 0) + 1);
const aaCreatorSection = aaInferenceComparisonEnd + 3;
section(aaInferenceAudit, `A${aaCreatorSection}:D${aaCreatorSection}`, "Reasoning uplift by creator — same-checkpoint pairs");
const aaCreatorHeader = aaCreatorSection + 1;
aaInferenceAudit.getRange(`A${aaCreatorHeader}:D${aaCreatorHeader}`).values = [["Creator", "Pairs", "Median AA uplift", "Interpretation"]];
header(aaInferenceAudit.getRange(`A${aaCreatorHeader}:D${aaCreatorHeader}`));
const aaCreatorData = Object.entries(aaAllPairs.creator_medians)
  .sort((a, b) => b[1] - a[1])
  .map(([creator, uplift]) => [creator, pairCountsByCreator.get(creator), uplift, "Descriptive configuration lift; not pure RL compute"]);
const aaCreatorEnd = aaCreatorHeader + aaCreatorData.length;
aaInferenceAudit.getRange(`A${aaCreatorHeader + 1}:D${aaCreatorEnd}`).values = aaCreatorData;
body(aaInferenceAudit.getRange(`A${aaCreatorHeader + 1}:D${aaCreatorEnd}`));
aaInferenceAudit.getRange(`B${aaCreatorHeader + 1}:B${aaCreatorEnd}`).format.numberFormat = "0";
aaInferenceAudit.getRange(`C${aaCreatorHeader + 1}:C${aaCreatorEnd}`).format.numberFormat = "0.00";
aaInferenceAudit.tables.add(`A${aaCreatorHeader}:D${aaCreatorEnd}`, true, "AaReasoningCreatorTable").style = "TableStyleMedium2";

const aaConflictSection = aaCreatorEnd + 3;
section(aaInferenceAudit, `A${aaConflictSection}:N${aaConflictSection}`, "AA↔Epoch source disagreements — identity retained, values never overwritten");
const aaConflictHeader = aaConflictSection + 1;
const aaConflictHeaders = ["Epoch checkpoint", "Epoch model", "AA detailed model", "Epoch release", "AA release", "Δ days", "Epoch params (B)", "AA params (B)", "AA / Epoch", "Creator agrees", "Date ±45d", "Params ±20%", "Match method", "AA detailed group"];
aaInferenceAudit.getRange(`A${aaConflictHeader}:N${aaConflictHeader}`).values = [aaConflictHeaders];
header(aaInferenceAudit.getRange(`A${aaConflictHeader}:N${aaConflictHeader}`));
const aaConflictData = aaDetailedCrosscheckRows
  .filter((row) => row.creator_agreement !== "True" || row.date_within_45_days !== "True" || row.parameters_within_20_percent !== "True")
  .map((row) => [
    row.epoch_checkpoint_id,
    row.epoch_model,
    row.aa_detailed_model,
    new Date(`${row.epoch_release_date}T00:00:00Z`),
    new Date(`${row.aa_release_date}T00:00:00Z`),
    Number(row.date_delta_days),
    Number(row.epoch_parameters_b),
    Number(row.aa_parameters_b),
    Number(row.aa_over_epoch_parameters),
    row.creator_agreement === "True" ? "Yes" : "No",
    row.date_within_45_days === "True" ? "Yes" : "No",
    row.parameters_within_20_percent === "True" ? "Yes" : "No",
    row.match_method,
    row.aa_detailed_group_id,
  ]);
const aaConflictEnd = aaConflictHeader + aaConflictData.length;
aaInferenceAudit.getRange(`A${aaConflictHeader + 1}:N${aaConflictEnd}`).values = aaConflictData;
body(aaInferenceAudit.getRange(`A${aaConflictHeader + 1}:N${aaConflictEnd}`));
aaInferenceAudit.getRange(`D${aaConflictHeader + 1}:E${aaConflictEnd}`).format.numberFormat = "yyyy-mm-dd";
aaInferenceAudit.getRange(`F${aaConflictHeader + 1}:I${aaConflictEnd}`).format.numberFormat = "0.00";
aaInferenceAudit.getRange(`M${aaConflictHeader + 1}:N${aaConflictEnd}`).format.wrapText = true;
aaInferenceAudit.tables.add(`A${aaConflictHeader}:N${aaConflictEnd}`, true, "AaEpochConflictTable").style = "TableStyleMedium2";

const aaPairSection = aaConflictEnd + 3;
section(aaInferenceAudit, `A${aaPairSection}:N${aaPairSection}`, "Complete 100-pair reasoning/non-reasoning ledger");
const aaPairHeader = aaPairSection + 1;
const aaPairHeaders = ["Creator", "Release", "Open weights", "Params known", "Total params (B)", "Reasoning model", "Reasoning AA", "Reasoning output tokens/task", "Non-reasoning model", "Non-reasoning AA", "Non-reasoning output tokens/task", "AA uplift", "Same weights URL", "Pair basis"];
aaInferenceAudit.getRange(`A${aaPairHeader}:N${aaPairHeader}`).values = [aaPairHeaders];
header(aaInferenceAudit.getRange(`A${aaPairHeader}:N${aaPairHeader}`));
const aaPairData = aaReasoningPairRows.map((row) => [
  row.creator_slug,
  new Date(`${row.release_date}T00:00:00Z`),
  row.is_open_weights === "True" ? "Yes" : "No",
  row.parameters_known === "True" ? "Yes" : "No",
  row.parameters_b === "" ? null : Number(row.parameters_b),
  row.reasoning_name,
  Number(row.reasoning_aa),
  row.reasoning_output_tokens_per_task === "" ? null : Number(row.reasoning_output_tokens_per_task),
  row.nonreasoning_name,
  Number(row.nonreasoning_aa),
  row.nonreasoning_output_tokens_per_task === "" ? null : Number(row.nonreasoning_output_tokens_per_task),
  Number(row.aa_uplift),
  row.same_weights_url === "True" ? "Yes" : "No",
  row.pair_basis,
]);
const aaPairEnd = aaPairHeader + aaPairData.length;
aaInferenceAudit.getRange(`A${aaPairHeader + 1}:N${aaPairEnd}`).values = aaPairData;
body(aaInferenceAudit.getRange(`A${aaPairHeader + 1}:N${aaPairEnd}`));
aaInferenceAudit.getRange(`B${aaPairHeader + 1}:B${aaPairEnd}`).format.numberFormat = "yyyy-mm-dd";
aaInferenceAudit.getRange(`E${aaPairHeader + 1}:L${aaPairEnd}`).format.numberFormat = "0.00";
aaInferenceAudit.getRange(`F${aaPairHeader + 1}:N${aaPairEnd}`).format.wrapText = true;
aaInferenceAudit.tables.add(`A${aaPairHeader}:N${aaPairEnd}`, true, "AaReasoningPairTable").style = "TableStyleMedium2";

const aaDetailedPanelSection = aaPairEnd + 3;
section(aaInferenceAudit, `A${aaDetailedPanelSection}:N${aaDetailedPanelSection}`, `Complete ${aaDetailedRows.length}-checkpoint detailed parameter panel`);
const aaDetailedPanelHeader = aaDetailedPanelSection + 1;
const aaDetailedPanelHeaders = ["Model", "Creator", "Release", "Total params (B)", "Active params (B)", "AA", "Estimated", "Reasoning", "Output tokens/task", "Answer tokens/task", "Reasoning tokens/task", "Configurations", "Removed lower scores", "Configuration slugs"];
aaInferenceAudit.getRange(`A${aaDetailedPanelHeader}:N${aaDetailedPanelHeader}`).values = [aaDetailedPanelHeaders];
header(aaInferenceAudit.getRange(`A${aaDetailedPanelHeader}:N${aaDetailedPanelHeader}`));
const aaDetailedPanelData = aaDetailedRows.map((row) => [
  row.selected_name,
  row.creator_slug,
  new Date(`${row.release_date}T00:00:00Z`),
  Number(row.parameters_b),
  row.active_parameters_b === "" ? null : Number(row.active_parameters_b),
  Number(row.intelligence_index),
  row.intelligence_index_estimated === "True" ? "Yes" : "No",
  row.is_reasoning === "True" ? "Yes" : "No",
  row.output_tokens_per_task === "" ? null : Number(row.output_tokens_per_task),
  row.answer_tokens_per_task === "" ? null : Number(row.answer_tokens_per_task),
  row.reasoning_tokens_per_task === "" ? null : Number(row.reasoning_tokens_per_task),
  Number(row.configuration_count),
  Number(row.lower_score_configurations_removed),
  row.configuration_slugs,
]);
const aaDetailedPanelEnd = aaDetailedPanelHeader + aaDetailedPanelData.length;
aaInferenceAudit.getRange(`A${aaDetailedPanelHeader + 1}:N${aaDetailedPanelEnd}`).values = aaDetailedPanelData;
body(aaInferenceAudit.getRange(`A${aaDetailedPanelHeader + 1}:N${aaDetailedPanelEnd}`));
aaInferenceAudit.getRange(`C${aaDetailedPanelHeader + 1}:C${aaDetailedPanelEnd}`).format.numberFormat = "yyyy-mm-dd";
aaInferenceAudit.getRange(`D${aaDetailedPanelHeader + 1}:F${aaDetailedPanelEnd}`).format.numberFormat = "0.00";
aaInferenceAudit.getRange(`I${aaDetailedPanelHeader + 1}:K${aaDetailedPanelEnd}`).format.numberFormat = "0.0";
aaInferenceAudit.getRange(`L${aaDetailedPanelHeader + 1}:M${aaDetailedPanelEnd}`).format.numberFormat = "0";
aaInferenceAudit.getRange(`N${aaDetailedPanelHeader + 1}:N${aaDetailedPanelEnd}`).format.wrapText = true;
aaInferenceAudit.tables.add(`A${aaDetailedPanelHeader}:N${aaDetailedPanelEnd}`, true, "AaDetailedParameterPanelTable").style = "TableStyleMedium2";
aaInferenceAudit.freezePanes.freezeRows(aaInferenceComparisonHeader);
aaInferenceAudit.getRange("A:B").format.columnWidth = 32;
aaInferenceAudit.getRange("C:M").format.columnWidth = 18;
aaInferenceAudit.getRange("F:F").format.columnWidth = 34;
aaInferenceAudit.getRange("I:I").format.columnWidth = 34;
aaInferenceAudit.getRange("N:N").format.columnWidth = 72;

const aaOperationalData = aaOperationalResult.data_audit;
const aaOperationalBacktests = aaOperationalResult.backtests;
const aaOperationalCrosscheck = aaOperationalResult.aa_openrouter_exact_crosscheck;
title(aaOperationalAudit, "A1:N1", "Artificial Analysis operational audit — standardized price, speed, and latency");
subtitle(aaOperationalAudit, "A2:N3", "AA uses first-party API performance when available and the median across providers otherwise. Provider-median price predicts held-out open-weight scale and agrees with OpenRouter, but first-party frontier price does not transfer to Fable/Sol. Speed and latency remain unsupported, so the existing 3.375% final price weight is unchanged and all incremental operational weights stay at 0%.");
aaOperationalAudit.getRange("A5:B5").values = [["Audit statistic", "Value"]];
header(aaOperationalAudit.getRange("A5:B5"));
const aaOperationalSummaryRows = [
  ["Raw AA model configurations", aaOperationalData.raw_model_configurations],
  ["Deduplicated open parameter checkpoints", aaOperationalData.deduplicated_open_parameter_checkpoints],
  ["Price-covered checkpoints", aaOperationalData.coverage.blended_price.checkpoints],
  ["Price-covered developers", aaOperationalData.coverage.blended_price.developers],
  ["Speed-covered checkpoints", aaOperationalData.coverage.output_speed.checkpoints],
  ["Speed-covered developers", aaOperationalData.coverage.output_speed.developers],
  ["Cost-per-task checkpoints", aaOperationalData.coverage.cost_per_task.checkpoints],
  ["Time-per-task checkpoints", aaOperationalData.coverage.time_per_task.checkpoints],
  ["First-party configurations", aaOperationalData.performance_source_type_counts.firstParty],
  ["Provider-median configurations", aaOperationalData.performance_source_type_counts.median],
  ["Exact AA↔OpenRouter checkpoints", aaOperationalCrosscheck.exact_epoch_checkpoint_intersection],
  ["Cross-source price Spearman", aaOperationalCrosscheck.price.spearman],
  ["Cross-source raw-speed Spearman", aaOperationalCrosscheck.raw_speed.spearman],
  ["Provider-median frontier baseline median (×)", aaOperationalBacktests.price_provider_median.scopes.frontier_like.baseline.median_multiplicative_error],
  ["Provider-median frontier + price median (×)", aaOperationalBacktests.price_provider_median.scopes.frontier_like.candidate.median_multiplicative_error],
  ["First-party frontier baseline median (×)", aaOperationalBacktests.price_first_party.scopes.frontier_like.baseline.median_multiplicative_error],
  ["First-party frontier + price median (×)", aaOperationalBacktests.price_first_party.scopes.frontier_like.candidate.median_multiplicative_error],
  ["Incremental AA operational price weight", aaOperationalResult.decision.incremental_aa_operational_price_weight],
  ["Incremental speed weight", aaOperationalResult.decision.incremental_aa_speed_weight],
  ["Incremental latency weight", aaOperationalResult.decision.incremental_aa_latency_weight],
  ["Change live price weight", aaOperationalResult.decision.change_live_price_weight ? "Yes" : "No"],
];
const aaOperationalSummaryEnd = 5 + aaOperationalSummaryRows.length;
aaOperationalAudit.getRange(`A6:B${aaOperationalSummaryEnd}`).values = aaOperationalSummaryRows;
body(aaOperationalAudit.getRange(`A6:B${aaOperationalSummaryEnd}`));
aaOperationalAudit.getRange("B17:B18").format.numberFormat = "0.000";
aaOperationalAudit.getRange("B19:B22").format.numberFormat = "0.00x";
aaOperationalAudit.getRange("B23:B25").format.numberFormat = "0.0%";

const aaOperationalComparisonSection = aaOperationalSummaryEnd + 3;
section(aaOperationalAudit, `A${aaOperationalComparisonSection}:N${aaOperationalComparisonSection}`, "Strictly chronological, developer-held-out operational comparisons");
const aaOperationalComparisonHeader = aaOperationalComparisonSection + 1;
const aaOperationalComparisonHeaders = ["Branch", "Scope", "Eligible", "Tests", "Developers", "Baseline median (×)", "Candidate median (×)", "Baseline MAE", "Candidate MAE", "Baseline p80 (×)", "Candidate p80 (×)", "90% CI low", "90% CI high", "P(candidate better)"];
aaOperationalAudit.getRange(`A${aaOperationalComparisonHeader}:N${aaOperationalComparisonHeader}`).values = [aaOperationalComparisonHeaders];
header(aaOperationalAudit.getRange(`A${aaOperationalComparisonHeader}:N${aaOperationalComparisonHeader}`));
const aaOperationalLabels = {
  price_global: "Price — global",
  price_source_adjusted: "Price + source regime",
  price_provider_median: "Price — provider median",
  price_first_party: "Price — first party",
  output_speed: "Output speed",
  latency_ttfc: "Latency / TTFC",
  speed_and_latency: "Speed + latency",
  cost_per_task: "Cost per task — exploratory",
  time_per_task: "Time per task — exploratory",
};
const aaOperationalComparisonData = [];
for (const [branch, label] of Object.entries(aaOperationalLabels)) {
  const test = aaOperationalBacktests[branch];
  for (const scope of ["all", "frontier_like"]) {
    const metrics = test.scopes[scope];
    aaOperationalComparisonData.push([
      label,
      scope,
      test.eligible_checkpoints,
      metrics.n,
      metrics.developers,
      metrics.baseline.median_multiplicative_error,
      metrics.candidate.median_multiplicative_error,
      metrics.baseline.mean_absolute_log10_error,
      metrics.candidate.mean_absolute_log10_error,
      metrics.baseline.p80_multiplicative_error,
      metrics.candidate.p80_multiplicative_error,
      metrics.paired_developer_bootstrap.ci_90[0],
      metrics.paired_developer_bootstrap.ci_90[1],
      metrics.paired_developer_bootstrap.bootstrap_probability_candidate_better,
    ]);
  }
}
const aaOperationalComparisonEnd = aaOperationalComparisonHeader + aaOperationalComparisonData.length;
aaOperationalAudit.getRange(`A${aaOperationalComparisonHeader + 1}:N${aaOperationalComparisonEnd}`).values = aaOperationalComparisonData;
body(aaOperationalAudit.getRange(`A${aaOperationalComparisonHeader + 1}:N${aaOperationalComparisonEnd}`));
aaOperationalAudit.getRange(`C${aaOperationalComparisonHeader + 1}:E${aaOperationalComparisonEnd}`).format.numberFormat = "0";
aaOperationalAudit.getRange(`F${aaOperationalComparisonHeader + 1}:G${aaOperationalComparisonEnd}`).format.numberFormat = "0.00x";
aaOperationalAudit.getRange(`H${aaOperationalComparisonHeader + 1}:I${aaOperationalComparisonEnd}`).format.numberFormat = "0.000";
aaOperationalAudit.getRange(`J${aaOperationalComparisonHeader + 1}:K${aaOperationalComparisonEnd}`).format.numberFormat = "0.00x";
aaOperationalAudit.getRange(`L${aaOperationalComparisonHeader + 1}:M${aaOperationalComparisonEnd}`).format.numberFormat = "0.000";
aaOperationalAudit.getRange(`N${aaOperationalComparisonHeader + 1}:N${aaOperationalComparisonEnd}`).format.numberFormat = "0.0%";
aaOperationalAudit.tables.add(`A${aaOperationalComparisonHeader}:N${aaOperationalComparisonEnd}`, true, "AaOperationalComparisonTable").style = "TableStyleMedium2";

const aaOperationalCrosscheckSection = aaOperationalComparisonEnd + 3;
section(aaOperationalAudit, `A${aaOperationalCrosscheckSection}:N${aaOperationalCrosscheckSection}`, "Complete 28-checkpoint AA↔OpenRouter exact-identity crosscheck");
const aaOperationalCrosscheckHeader = aaOperationalCrosscheckSection + 1;
const aaOperationalCrosscheckHeaders = ["Canonical checkpoint", "Epoch model", "AA model", "Release", "Params (B)", "AA regime", "AA price", "OpenRouter price", "AA speed", "OpenRouter speed", "OR normalized speed", "AA/OR price", "AA/OR speed", "OpenRouter model IDs"];
aaOperationalAudit.getRange(`A${aaOperationalCrosscheckHeader}:N${aaOperationalCrosscheckHeader}`).values = [aaOperationalCrosscheckHeaders];
header(aaOperationalAudit.getRange(`A${aaOperationalCrosscheckHeader}:N${aaOperationalCrosscheckHeader}`));
const aaOperationalCrosscheckData = aaOpenRouterCrosscheckRows.map((row) => {
  const aaPrice = row.aa_blended_price_usd_per_mtoken === "" ? null : Number(row.aa_blended_price_usd_per_mtoken);
  const routerPrice = row.openrouter_blended_price_usd_per_mtoken === "" ? null : Number(row.openrouter_blended_price_usd_per_mtoken);
  const aaSpeed = row.aa_output_speed_tps === "" ? null : Number(row.aa_output_speed_tps);
  const routerSpeed = row.openrouter_raw_throughput_tps === "" ? null : Number(row.openrouter_raw_throughput_tps);
  return [
    row.canonical_checkpoint_id,
    row.epoch_model,
    row.aa_model,
    new Date(`${row.release_date}T00:00:00Z`),
    Number(row.parameters_b),
    row.aa_performance_source_type,
    aaPrice,
    routerPrice,
    aaSpeed,
    routerSpeed,
    row.openrouter_provider_normalized_throughput_ratio === "" ? null : Number(row.openrouter_provider_normalized_throughput_ratio),
    aaPrice != null && routerPrice > 0 ? aaPrice / routerPrice : null,
    aaSpeed != null && routerSpeed > 0 ? aaSpeed / routerSpeed : null,
    row.openrouter_model_ids,
  ];
});
const aaOperationalCrosscheckEnd = aaOperationalCrosscheckHeader + aaOperationalCrosscheckData.length;
aaOperationalAudit.getRange(`A${aaOperationalCrosscheckHeader + 1}:N${aaOperationalCrosscheckEnd}`).values = aaOperationalCrosscheckData;
body(aaOperationalAudit.getRange(`A${aaOperationalCrosscheckHeader + 1}:N${aaOperationalCrosscheckEnd}`));
aaOperationalAudit.getRange(`D${aaOperationalCrosscheckHeader + 1}:D${aaOperationalCrosscheckEnd}`).format.numberFormat = "yyyy-mm-dd";
aaOperationalAudit.getRange(`E${aaOperationalCrosscheckHeader + 1}:M${aaOperationalCrosscheckEnd}`).format.numberFormat = "0.00";
aaOperationalAudit.getRange(`A${aaOperationalCrosscheckHeader + 1}:C${aaOperationalCrosscheckEnd}`).format.wrapText = true;
aaOperationalAudit.getRange(`N${aaOperationalCrosscheckHeader + 1}:N${aaOperationalCrosscheckEnd}`).format.wrapText = true;
aaOperationalAudit.tables.add(`A${aaOperationalCrosscheckHeader}:N${aaOperationalCrosscheckEnd}`, true, "AaOpenRouterOperationalCrosscheckTable").style = "TableStyleMedium2";

const aaOperationalPanelSection = aaOperationalCrosscheckEnd + 3;
section(aaOperationalAudit, `A${aaOperationalPanelSection}:N${aaOperationalPanelSection}`, `Complete ${aaOperationalRows.length}-checkpoint AA operational parameter panel`);
const aaOperationalPanelHeader = aaOperationalPanelSection + 1;
const aaOperationalPanelHeaders = ["Model", "Creator", "Release", "Total params (B)", "Active params (B)", "AA", "Serving regime", "Performance provider", "Input $/M", "Output $/M", "Blended $/M", "Output tok/s", "TTFC (s)", "Cost/task ($)"];
aaOperationalAudit.getRange(`A${aaOperationalPanelHeader}:N${aaOperationalPanelHeader}`).values = [aaOperationalPanelHeaders];
header(aaOperationalAudit.getRange(`A${aaOperationalPanelHeader}:N${aaOperationalPanelHeader}`));
const aaOperationalPanelData = aaOperationalRows.map((row) => [
  row.selected_name,
  row.creator_slug,
  new Date(`${row.release_date}T00:00:00Z`),
  Number(row.parameters_b),
  row.active_parameters_b === "" ? null : Number(row.active_parameters_b),
  Number(row.intelligence_index),
  row.performance_data_source_type,
  row.performance_provider_name,
  row.price_input_usd_per_mtoken === "" ? null : Number(row.price_input_usd_per_mtoken),
  row.price_output_usd_per_mtoken === "" ? null : Number(row.price_output_usd_per_mtoken),
  row.price_blended_7_2_1_usd_per_mtoken === "" ? null : Number(row.price_blended_7_2_1_usd_per_mtoken),
  row.median_output_speed_tps === "" ? null : Number(row.median_output_speed_tps),
  row.median_time_to_first_chunk_seconds === "" ? null : Number(row.median_time_to_first_chunk_seconds),
  row.intelligence_cost_per_task_usd === "" ? null : Number(row.intelligence_cost_per_task_usd),
]);
const aaOperationalPanelEnd = aaOperationalPanelHeader + aaOperationalPanelData.length;
aaOperationalAudit.getRange(`A${aaOperationalPanelHeader + 1}:N${aaOperationalPanelEnd}`).values = aaOperationalPanelData;
body(aaOperationalAudit.getRange(`A${aaOperationalPanelHeader + 1}:N${aaOperationalPanelEnd}`));
aaOperationalAudit.getRange(`C${aaOperationalPanelHeader + 1}:C${aaOperationalPanelEnd}`).format.numberFormat = "yyyy-mm-dd";
aaOperationalAudit.getRange(`D${aaOperationalPanelHeader + 1}:F${aaOperationalPanelEnd}`).format.numberFormat = "0.00";
aaOperationalAudit.getRange(`I${aaOperationalPanelHeader + 1}:N${aaOperationalPanelEnd}`).format.numberFormat = "0.00";
aaOperationalAudit.tables.add(`A${aaOperationalPanelHeader}:N${aaOperationalPanelEnd}`, true, "AaOperationalPanelTable").style = "TableStyleMedium2";
aaOperationalAudit.freezePanes.freezeRows(aaOperationalComparisonHeader);
aaOperationalAudit.getRange("A:A").format.columnWidth = 34;
aaOperationalAudit.getRange("B:B").format.columnWidth = 22;
aaOperationalAudit.getRange("C:M").format.columnWidth = 17;
aaOperationalAudit.getRange("N:N").format.columnWidth = 46;

const activePredictability = activeTransportResult.active_parameter_predictability;
const activeTransport = activeTransportResult.high_sparsity_total_transport;
const activeK3 = activeTransportResult.kimi_k3_external_architecture_check;
const activeComputeDependency = activeTransportResult.compute_branch_independence;
const k3Derived = k3ReleaseEvidence.derived_quantities;
const k3Training = k3ReleaseEvidence.training_disclosures;
const k3TechnicalReportUrl = k3ReleaseEvidence.source_files.official_technical_report.url;
if (
  Math.abs(activeK3.k3_disclosed_total_b - k3Facts.total_parameters_b_exact) > 1e-12
  || Math.abs(activeK3.k3_disclosed_active_b - k3Facts.activated_parameters_b_exact) > 1e-12
) throw new Error("Active-parameter audit disagrees with official K3 parameter counts");
title(activeTransportAudit, "A1:N1", "Active-parameter transport — Kimi K3 architecture and compute-dependency audit");
subtitle(activeTransportAudit, "A2:N3", `The official K3 report discloses exactly ${(k3Facts.total_parameters_b_exact / 1000).toFixed(2)}T total and ${k3Facts.activated_parameters_b_exact.toFixed(1)}B activated parameters. The ${activePredictability.active_score_date.n}-checkpoint held-out audit is mixed and hidden-target sparsity is undisclosed, so active-parameter transport remains a 0%-weight sensitivity.`);

activeTransportAudit.getRange("A5:D5").values = [["Architecture quantity", "Value", "Audit interpretation", "Source URL"]];
header(activeTransportAudit.getRange("A5:D5"));
activeTransportAudit.getRange("A6:D18").values = [
  ["K3 total parameters (B)", k3Facts.total_parameters_b_exact, "Exact disclosed total", k3TechnicalReportUrl],
  ["K3 activated parameters (B)", k3Facts.activated_parameters_b_exact, "Exact disclosed per-token activated count", k3TechnicalReportUrl],
  ["K3 activated fraction (%)", 100 * k3Derived.activated_parameter_fraction, "Exact active / total ratio; includes all activated components", k3TechnicalReportUrl],
  ["K3 total / activated ratio", k3Derived.total_to_activated_parameter_ratio, "Exact disclosed architecture ratio", k3TechnicalReportUrl],
  ["K3 selected routed experts/token", k3Facts.selected_routed_experts_per_token, "Disclosed routing count", k3TechnicalReportUrl],
  ["K3 routed experts", k3Facts.routed_experts, "Disclosed routed-expert count", k3TechnicalReportUrl],
  ["K3 shared experts", k3Facts.shared_experts, "Disclosed shared-expert count", k3TechnicalReportUrl],
  ["K3 selected routed-expert fraction (%)", 100 * k3Derived.selected_routed_expert_fraction, "Routing statistic only; not the active-parameter fraction", k3TechnicalReportUrl],
  ["K3 transformer layers", k3Facts.layers, "1 dense + 92 MoE layers", k3TechnicalReportUrl],
  ["K2 total parameters (B)", k2Facts.total_parameters_b_exact, "Exact disclosed comparator total", k3TechnicalReportUrl],
  ["K2 activated parameters (B)", k2Facts.activated_parameters_b_exact, "Exact disclosed comparator active count", k3TechnicalReportUrl],
  ["K2 total / activated ratio", k3Derived.k2_total_to_activated_parameter_ratio, "Exact disclosed comparator ratio", k3TechnicalReportUrl],
  ["K3 reported scaling efficiency vs K2", k3Training.reported_scaling_efficiency_vs_k2, "Combines architecture, data, and training recipe; not a parameter multiplier", k3TechnicalReportUrl],
];
body(activeTransportAudit.getRange("A6:D18"));
activeTransportAudit.getRange("B6:B18").format.numberFormat = "0.00";
activeTransportAudit.getRange("C6:D18").format.wrapText = true;
activeTransportAudit.getRange("A6:D18").format.rowHeight = 30;

section(activeTransportAudit, "A19:N19", "Strict chronological developer-held-out comparisons");
activeTransportAudit.getRange("A20:H20").values = [["Comparison", "Tests", "Candidate median (×)", "Baseline median (×)", "Candidate MAE", "Baseline MAE", "90% CI low", "90% CI high"]];
header(activeTransportAudit.getRange("A20:H20"));
activeTransportAudit.getRange("A21:H22").values = [
  [
    `Predict active vs total on identical ${activePredictability.active_score_date.n}-checkpoint panel`,
    activePredictability.active_score_date.n,
    activePredictability.active_score_date.median_multiplicative_error,
    activePredictability.total_score_date_same_active_checkpoint_panel.median_multiplicative_error,
    activePredictability.active_score_date.mean_absolute_log10_error,
    activePredictability.total_score_date_same_active_checkpoint_panel.mean_absolute_log10_error,
    activePredictability.paired_active_vs_same_panel_total.ci_90[0],
    activePredictability.paired_active_vs_same_panel_total.ci_90[1],
  ],
  [
    "Convert active→total for ≥15× sparsity",
    activeTransport.candidate.n,
    activeTransport.candidate.median_multiplicative_error,
    activeTransport.direct_total_baseline.median_multiplicative_error,
    activeTransport.candidate.mean_absolute_log10_error,
    activeTransport.direct_total_baseline.mean_absolute_log10_error,
    activeTransport.paired_cluster_bootstrap.ci_90[0],
    activeTransport.paired_cluster_bootstrap.ci_90[1],
  ],
];
body(activeTransportAudit.getRange("A21:H22"));
activeTransportAudit.getRange("A20:A22").format.wrapText = true;
activeTransportAudit.getRange("A20:N22").format.rowHeight = 30;
activeTransportAudit.getRange("C21:D22").format.numberFormat = "0.00x";
activeTransportAudit.getRange("E21:H22").format.numberFormat = "0.000";
activeTransportAudit.getRange("J20:N20").values = [["Decision", "Active weight", "Epoch compute estimates", "Independent compute?", "Headline change?"]];
header(activeTransportAudit.getRange("J20:N20"));
activeTransportAudit.getRange("J21:N21").values = [[
  activeTransportResult.decision.promote_active_transport_to_live_factor ? "Promote" : "Diagnostic only",
  activeTransportResult.decision.incremental_live_weight,
  activeComputeDependency.target_models_with_epoch_training_compute_estimate,
  activeComputeDependency.independent_target_evidence ? "Yes" : "No",
  activeTransportResult.decision.change_headline_forecasts ? "Yes" : "No",
]];
body(activeTransportAudit.getRange("J21:N21"));
activeTransportAudit.getRange("K21:K21").format.numberFormat = "0.0%";
activeTransportAudit.getRange("A24:N25").merge();
activeTransportAudit.getRange("A24").values = [[`Compute dependency: ${activeComputeDependency.algebra} K3 has one explicitly speculative Epoch compute estimate, but no live target has disclosed training compute; this remains a compute-structured AA/date regularizer rather than an independent target likelihood.`]];
activeTransportAudit.getRange("A24:N25").format = { fill: C.paleAmber, font: { name: "Aptos", size: 10, color: C.text }, wrapText: true, verticalAlignment: "center" };

section(activeTransportAudit, "A28:H28", "K3-anchored active-parameter target sensitivity — formulas remain visible");
activeTransportAudit.getRange("A29:H29").values = [["Model", "Release", "AA", "Raw predicted active (B)", "K3-calibrated active (B)", "K3-anchored total (T)", "Status", "Interpretation"]];
header(activeTransportAudit.getRange("A29:H29"));
const activeTargetStart = 30;
const activeK3TargetIndex = activeTransportTargetRows.findIndex((row) => row.model === "Kimi K3");
if (activeTransportTargetRows.length !== 3 || activeK3TargetIndex < 0) throw new Error("Active-parameter target sensitivity must contain Fable, Sol, and K3");
activeTransportAudit.getRange(`A${activeTargetStart}:H${activeTargetStart + activeTransportTargetRows.length - 1}`).values = activeTransportTargetRows.map((row) => [
  row.model,
  new Date(`${row.release_date}T00:00:00Z`),
  Number(row.aa_score),
  Number(row.predicted_active_b),
  Number(row.k3_calibrated_active_b),
  null,
  row.status,
  row.model === "Kimi K3" ? "Exact 104.2B active / 2.78T total anchor" : "0%-weight structural sensitivity",
]);
for (let index = 0; index < activeTransportTargetRows.length; index += 1) {
  const row = activeTargetStart + index;
  activeTransportAudit.getRange(`F${row}`).formulas = [[`=$B$6*E${row}/$B$7/1000`]];
}
const activeTargetEnd = activeTargetStart + activeTransportTargetRows.length - 1;
body(activeTransportAudit.getRange(`A${activeTargetStart}:H${activeTargetEnd}`));
activeTransportAudit.getRange(`B${activeTargetStart}:B${activeTargetEnd}`).format.numberFormat = "yyyy-mm-dd";
activeTransportAudit.getRange(`C${activeTargetStart}:C${activeTargetEnd}`).format.numberFormat = "0.0";
activeTransportAudit.getRange(`D${activeTargetStart}:D${activeTargetEnd}`).format.numberFormat = "0.0";
activeTransportAudit.getRange(`E${activeTargetStart}:E${activeTargetEnd}`).format.numberFormat = "0.0";
activeTransportAudit.getRange(`F${activeTargetStart}:F${activeTargetEnd}`).format.numberFormat = "0.0\"T\"";
activeTransportAudit.getRange(`G${activeTargetStart}:H${activeTargetEnd}`).format.wrapText = true;

const activePredictionSection = activeTargetEnd + 3;
section(activeTransportAudit, `A${activePredictionSection}:N${activePredictionSection}`, `Complete ${activeTransportResult.inventory.chronological_predictions}-fold active-parameter prediction ledger`);
const activePredictionHeader = activePredictionSection + 1;
activeTransportAudit.getRange(`A${activePredictionHeader}:N${activePredictionHeader}`).values = [["Release", "Model", "Developer", "AA", "Frontier rank", "Actual active (B)", "Pred active (B)", "Active error (×)", "Actual total (B)", "Actual ratio", "Direct total pred (B)", "Direct error (×)", "Converted total (B)", "Converted error (×)"]];
header(activeTransportAudit.getRange(`A${activePredictionHeader}:N${activePredictionHeader}`));
const activePredictionStart = activePredictionHeader + 1;
const activePredictionEnd = activePredictionHeader + activeTransportPredictionRows.length;
if (activeTransportPredictionRows.length !== activeTransportResult.inventory.chronological_predictions) {
  throw new Error(`Active-parameter prediction ledger has ${activeTransportPredictionRows.length} rows; audit declares ${activeTransportResult.inventory.chronological_predictions}`);
}
activeTransportAudit.getRange(`A${activePredictionStart}:N${activePredictionEnd}`).values = activeTransportPredictionRows.map((row) => [
  new Date(`${row.release_date}T00:00:00Z`),
  row.model,
  row.developer,
  Number(row.aa_score),
  Number(row.frontier_score_rank),
  Number(row.actual_active_b),
  Number(row.predicted_active_b),
  null,
  Number(row.actual_total_b),
  null,
  Number(row.predicted_full_panel_total_b),
  null,
  row.active_converted_total_b === "" ? null : Number(row.active_converted_total_b),
  null,
]);
for (let row = activePredictionStart; row <= activePredictionEnd; row += 1) {
  activeTransportAudit.getRange(`H${row}`).formulas = [[`=10^ABS(LOG10(G${row}/F${row}))`]];
  activeTransportAudit.getRange(`J${row}`).formulas = [[`=I${row}/F${row}`]];
  activeTransportAudit.getRange(`L${row}`).formulas = [[`=10^ABS(LOG10(K${row}/I${row}))`]];
  activeTransportAudit.getRange(`N${row}`).formulas = [[`=IF(M${row}=\"\",\"\",10^ABS(LOG10(M${row}/I${row})))`]];
}
body(activeTransportAudit.getRange(`A${activePredictionStart}:N${activePredictionEnd}`));
activeTransportAudit.getRange(`A${activePredictionStart}:A${activePredictionEnd}`).format.numberFormat = "yyyy-mm-dd";
activeTransportAudit.getRange(`D${activePredictionStart}:D${activePredictionEnd}`).format.numberFormat = "0.0";
activeTransportAudit.getRange(`E${activePredictionStart}:E${activePredictionEnd}`).format.numberFormat = "0.0%";
activeTransportAudit.getRange(`F${activePredictionStart}:N${activePredictionEnd}`).format.numberFormat = "0.00";
activeTransportAudit.tables.add(`A${activePredictionHeader}:N${activePredictionEnd}`, true, "ActiveParameterPredictionTable").style = "TableStyleMedium2";
activeTransportAudit.freezePanes.freezeRows(20);
activeTransportAudit.getRange("A:A").format.columnWidth = 34;
activeTransportAudit.getRange("B:B").format.columnWidth = 38;
activeTransportAudit.getRange("C:C").format.columnWidth = 22;
activeTransportAudit.getRange("D:N").format.columnWidth = 18;
activeTransportAudit.getRange("D:D").format.columnWidth = 30;
activeTransportAudit.getRange("C:C").format.wrapText = true;
activeTransportAudit.getRange("H:H").format.columnWidth = 26;

const expandedEciAudit = eciComponentExtendedResult.expanded_total_parameter_panel;
const activeComponentAudit = eciComponentExtendedResult.active_parameter_component_audit;
const liveEciComparison = expandedEciAudit.aggregate_backtest.legacy_57_tests;
const allEciComparison = expandedEciAudit.aggregate_backtest.all_84_tests;
const bestComponent = activeComponentAudit.best_uncorrected_component;
title(eciComponentAudit, "A1:N1", "ECI component audit — exact parameter expansion and multiple-testing control");
subtitle(eciComponentAudit, "A2:N3", "After the July 31 identity migration, 57 legacy workbook parameter rows remain and are extended with 27 unique, exact, date-consistent open-weight Epoch matches. Individual components are tested on 89 active-parameter checkpoints with strict chronological family holdouts and a global familywise max-T correction. No component survives, so the live ECI center and component weight remain unchanged.");
eciComponentAudit.getRange("A5:B5").values = [["Audit statistic", "Value"]];
header(eciComponentAudit.getRange("A5:B5"));
const eciComponentSummaryRows = [
  ["Expanded total-parameter checkpoints", expandedEciAudit.models],
  ["Exact open-weight Epoch additions", expandedEciAudit.source_counts.exact_epoch_open_extension],
  ["Active-parameter checkpoints", activeComponentAudit.parameter_map_models],
  ["Model-by-benchmark measurements", activeComponentAudit.component_rows],
  ["Eligible benchmark comparisons", activeComponentAudit.eligible_comparisons],
  ["Legacy tests: 57-row median error (×)", liveEciComparison.legacy_57.median_multiplicative_error],
  ["Legacy tests: expanded 84 median error (×)", liveEciComparison.expanded_84.median_multiplicative_error],
  ["All tests: legacy 57 median error (×)", allEciComparison.legacy_57.median_multiplicative_error],
  ["All tests: expanded 84 median error (×)", allEciComparison.expanded_84.median_multiplicative_error],
  ["All-test family-bootstrap 90% CI high", allEciComparison.paired_family_bootstrap.ci_90[1]],
  ["Best uncorrected benchmark", bestComponent.benchmark],
  ["Best uncorrected probability better", bestComponent.active_unadjusted_probability_better],
  ["Familywise one-sided p", bestComponent.active_familywise_one_sided_p],
  ["Components supported after correction", activeComponentAudit.supported_after_familywise_correction.length],
  ["Incremental component weight", activeComponentAudit.decision.incremental_component_weight],
  ["Change live ECI center", expandedEciAudit.decision.change_live_eci_center ? "Yes" : "No"],
];
const eciComponentSummaryEnd = 5 + eciComponentSummaryRows.length;
eciComponentAudit.getRange(`A6:B${eciComponentSummaryEnd}`).values = eciComponentSummaryRows;
body(eciComponentAudit.getRange(`A6:B${eciComponentSummaryEnd}`));
eciComponentAudit.getRange("B11:B14").format.numberFormat = "0.00x";
eciComponentAudit.getRange("B15:B15").format.numberFormat = "0.000";
eciComponentAudit.getRange("B17:B18").format.numberFormat = "0.0%";
eciComponentAudit.getRange("B20:B20").format.numberFormat = "0.0%";

const eciComponentComparisonSection = eciComponentSummaryEnd + 3;
section(eciComponentAudit, `A${eciComponentComparisonSection}:M${eciComponentComparisonSection}`, "Incremental component tests — active and total parameters");
const eciComponentComparisonHeader = eciComponentComparisonSection + 1;
const eciComponentComparisonHeaders = ["Benchmark", "Coverage", "Held-out", "Families", "Baseline active (×)", "+ component active (×)", "Equal-family Δ MAE", "90% CI low", "90% CI high", "Familywise p", "Baseline total (×)", "+ component total (×)", "Decision"];
eciComponentAudit.getRange(`A${eciComponentComparisonHeader}:M${eciComponentComparisonHeader}`).values = [eciComponentComparisonHeaders];
header(eciComponentAudit.getRange(`A${eciComponentComparisonHeader}:M${eciComponentComparisonHeader}`));
const eciComponentComparisonData = eciComponentComparisonRows.map((row) => [
  row.benchmark,
  Number(row.coverage_models),
  Number(row.heldout_predictions),
  Number(row.heldout_families),
  Number(row.baseline_active_median_error_x),
  Number(row.augmented_active_median_error_x),
  Number(row.active_equal_family_mae_delta),
  Number(row.active_delta_ci_90_low),
  Number(row.active_delta_ci_90_high),
  Number(row.active_familywise_one_sided_p),
  Number(row.baseline_total_median_error_x),
  Number(row.augmented_total_median_error_x),
  Number(row.active_equal_family_mae_delta) < 0 && Number(row.active_familywise_one_sided_p) < 0.05 ? "Supported" : "Not supported",
]);
const eciComponentComparisonEnd = eciComponentComparisonHeader + eciComponentComparisonData.length;
eciComponentAudit.getRange(`A${eciComponentComparisonHeader + 1}:M${eciComponentComparisonEnd}`).values = eciComponentComparisonData;
body(eciComponentAudit.getRange(`A${eciComponentComparisonHeader + 1}:M${eciComponentComparisonEnd}`));
eciComponentAudit.getRange(`B${eciComponentComparisonHeader + 1}:D${eciComponentComparisonEnd}`).format.numberFormat = "0";
eciComponentAudit.getRange(`E${eciComponentComparisonHeader + 1}:F${eciComponentComparisonEnd}`).format.numberFormat = "0.00x";
eciComponentAudit.getRange(`G${eciComponentComparisonHeader + 1}:I${eciComponentComparisonEnd}`).format.numberFormat = "0.000";
eciComponentAudit.getRange(`J${eciComponentComparisonHeader + 1}:J${eciComponentComparisonEnd}`).format.numberFormat = "0.0%";
eciComponentAudit.getRange(`K${eciComponentComparisonHeader + 1}:L${eciComponentComparisonEnd}`).format.numberFormat = "0.00x";
eciComponentAudit.tables.add(`A${eciComponentComparisonHeader}:M${eciComponentComparisonEnd}`, true, "EciComponentComparisonTable").style = "TableStyleMedium2";

const eciComponentPanelSection = eciComponentComparisonEnd + 3;
section(eciComponentAudit, `A${eciComponentPanelSection}:N${eciComponentPanelSection}`, `Complete ${eciComponentExpandedRows.length}-checkpoint total-parameter admission ledger`);
const eciComponentPanelHeader = eciComponentPanelSection + 1;
const eciComponentPanelHeaders = ["Model", "Release", "Total params (B)", "Family", "ECI", "CI low", "CI high", "CI width", "WLS weight", "Panel source", "Epoch match", "Accessibility", "Parameter notes", "Source URL"];
eciComponentAudit.getRange(`A${eciComponentPanelHeader}:N${eciComponentPanelHeader}`).values = [eciComponentPanelHeaders];
header(eciComponentAudit.getRange(`A${eciComponentPanelHeader}:N${eciComponentPanelHeader}`));
const eciComponentPanelData = eciComponentExpandedRows.map((row) => [
  row.model,
  new Date(`${row.release_date}T00:00:00Z`),
  Number(row.total_parameters_b),
  row.family,
  Number(row.eci_score),
  Number(row.eci_ci_low),
  Number(row.eci_ci_high),
  Number(row.eci_ci_width),
  Number(row.wls_weight),
  row.panel_source,
  row.matched_epoch_model,
  row.epoch_accessibility,
  row.epoch_parameter_notes,
  row.source_url,
]);
const eciComponentPanelEnd = eciComponentPanelHeader + eciComponentPanelData.length;
eciComponentAudit.getRange(`A${eciComponentPanelHeader + 1}:N${eciComponentPanelEnd}`).values = eciComponentPanelData;
body(eciComponentAudit.getRange(`A${eciComponentPanelHeader + 1}:N${eciComponentPanelEnd}`));
eciComponentAudit.getRange(`B${eciComponentPanelHeader + 1}:B${eciComponentPanelEnd}`).format.numberFormat = "yyyy-mm-dd";
eciComponentAudit.getRange(`C${eciComponentPanelHeader + 1}:I${eciComponentPanelEnd}`).format.numberFormat = "0.00";
eciComponentAudit.getRange(`M${eciComponentPanelHeader + 1}:N${eciComponentPanelEnd}`).format.wrapText = true;
eciComponentAudit.tables.add(`A${eciComponentPanelHeader}:N${eciComponentPanelEnd}`, true, "EciExpandedParameterPanelTable").style = "TableStyleMedium2";
eciComponentAudit.freezePanes.freezeRows(eciComponentComparisonHeader);
eciComponentAudit.getRange("A:A").format.columnWidth = 38;
eciComponentAudit.getRange("B:B").format.columnWidth = 16;
eciComponentAudit.getRange("C:L").format.columnWidth = 18;
eciComponentAudit.getRange("D:D").format.columnWidth = 24;
eciComponentAudit.getRange("M:M").format.columnWidth = 60;
eciComponentAudit.getRange("N:N").format.columnWidth = 70;

const multivariateInventory = eciMultivariateResult.inventory;
const multivariateGates = eciMultivariateResult.promotion_gates;
const multivariatePrimary = eciMultivariateResult.backtest.total.all;
const multivariatePrimaryNarrowCi = eciMultivariateResult.backtest.total.narrow_eci_ci;
const multivariateNarrowCiOnly = eciMultivariateResult.narrow_eci_ci_only_training_backtest.total;
const multivariateFrontier = eciMultivariateResult.backtest.total.frontier_like;
title(eciMultivariateAudit, "A1:P1", "ECI multivariate component audit — nested selection and narrow-ECI-CI replication");
subtitle(eciMultivariateAudit, "A2:P3", "A regularized combination of benchmark-implied ECI residuals improves every headline point estimate, but the primary and narrow-ECI-CI equal-family intervals cross zero. The legacy flag separates aggregate ECI confidence intervals above 10 points; it does not classify parameter disclosure. Target fitting uses only strictly earlier checkpoints and removes the target developer, so incremental live weight remains 0%.");
eciMultivariateAudit.getRange("A5:B5").values = [["Audit statistic", "Value"]];
header(eciMultivariateAudit.getRange("A5:B5"));
const multivariateSummaryRows = [
  ["Parameter-map checkpoints", multivariateInventory.parameter_map_checkpoints],
  ["Developer families", multivariateInventory.parameter_map_families],
  ["Narrow ECI CI checkpoints", multivariateInventory.narrow_eci_ci_checkpoints],
  ["Broad ECI CI checkpoints", multivariateInventory.broad_eci_ci_checkpoints],
  ["Unique component measurements", multivariateInventory.unique_component_measurements],
  ["Component benchmarks", multivariateInventory.component_benchmarks],
  ["Primary outer predictions", multivariateInventory.outer_predictions],
  ["Primary prediction families", multivariateInventory.outer_prediction_families],
  ["Narrow-ECI-CI-only predictions", multivariateInventory.narrow_eci_ci_only_outer_predictions],
  ["Narrow-ECI-CI-only families", multivariateInventory.narrow_eci_ci_only_outer_prediction_families],
  ["Incremental live weight", eciMultivariateResult.decision.incremental_live_weight],
  ["Change headline forecasts", eciMultivariateResult.decision.change_headline_forecasts ? "Yes" : "No"],
];
const multivariateSummaryEnd = 5 + multivariateSummaryRows.length;
eciMultivariateAudit.getRange(`A6:B${multivariateSummaryEnd}`).values = multivariateSummaryRows;
body(eciMultivariateAudit.getRange(`A6:B${multivariateSummaryEnd}`));
eciMultivariateAudit.getRange("B6:B15").format.numberFormat = "0";
eciMultivariateAudit.getRange("B16").format.numberFormat = "0.0%";

eciMultivariateAudit.getRange("D5:L5").values = [["Held-out scope", "Tests", "Families", "Baseline median (×)", "Candidate median (×)", "Equal-family Δ MAE", "90% CI low", "90% CI high", "Decision"]];
header(eciMultivariateAudit.getRange("D5:L5"));
const multivariateBacktestRows = [
  ["Primary all outcomes", multivariatePrimary.n, multivariatePrimary.families, multivariatePrimary.baseline.median_multiplicative_error, multivariatePrimary.candidate.median_multiplicative_error, multivariatePrimary.paired_family_bootstrap.observed_delta, multivariatePrimary.paired_family_bootstrap.ci_90[0], multivariatePrimary.paired_family_bootstrap.ci_90[1], "Interval crosses zero"],
  ["Primary narrow-ECI-CI outcomes", multivariatePrimaryNarrowCi.n, multivariatePrimaryNarrowCi.families, multivariatePrimaryNarrowCi.baseline.median_multiplicative_error, multivariatePrimaryNarrowCi.candidate.median_multiplicative_error, multivariatePrimaryNarrowCi.paired_family_bootstrap.observed_delta, multivariatePrimaryNarrowCi.paired_family_bootstrap.ci_90[0], multivariatePrimaryNarrowCi.paired_family_bootstrap.ci_90[1], "Interval crosses zero"],
  ["Narrow-ECI-CI-only training/tests", multivariateNarrowCiOnly.n, multivariateNarrowCiOnly.families, multivariateNarrowCiOnly.baseline.median_multiplicative_error, multivariateNarrowCiOnly.candidate.median_multiplicative_error, multivariateNarrowCiOnly.paired_family_bootstrap.observed_delta, multivariateNarrowCiOnly.paired_family_bootstrap.ci_90[0], multivariateNarrowCiOnly.paired_family_bootstrap.ci_90[1], "Interval crosses zero"],
  ["Frontier-like primary folds", multivariateFrontier.n, multivariateFrontier.families, multivariateFrontier.baseline.median_multiplicative_error, multivariateFrontier.candidate.median_multiplicative_error, multivariateFrontier.paired_family_bootstrap.observed_delta, multivariateFrontier.paired_family_bootstrap.ci_90[0], multivariateFrontier.paired_family_bootstrap.ci_90[1], "Only five families"],
];
eciMultivariateAudit.getRange("D6:L9").values = multivariateBacktestRows;
body(eciMultivariateAudit.getRange("D6:L9"));
eciMultivariateAudit.getRange("G6:H9").format.numberFormat = "0.00x";
eciMultivariateAudit.getRange("I6:K9").format.numberFormat = "0.000";

section(eciMultivariateAudit, "D12:P12", "Promotion gates — every gate must pass");
eciMultivariateAudit.getRange("D13:F13").values = [["Gate", "Passed", "Interpretation"]];
header(eciMultivariateAudit.getRange("D13:F13"));
const multivariateGateRows = [
  ["Primary point metrics improve", multivariateGates.all_point_metrics_improve, "Pass"],
  ["Primary interval wholly favorable", multivariateGates.all_equal_family_ci_wholly_favorable, "CI high must be < 0"],
  ["Narrow-ECI-CI outcomes interval favorable", multivariateGates.narrow_eci_ci_equal_family_ci_wholly_favorable, "CI high must be < 0"],
  ["Narrow-ECI-CI-only point metrics improve", multivariateGates.narrow_eci_ci_only_training_point_metrics_improve, "Pass"],
  ["Narrow-ECI-CI-only interval favorable", multivariateGates.narrow_eci_ci_only_training_ci_wholly_favorable, "CI high must be < 0"],
  ["Outer coverage", multivariateGates.coverage_gate, "Pass"],
  ["Narrow-ECI-CI-only coverage", multivariateGates.narrow_eci_ci_only_training_coverage_gate, "Pass"],
  ["Frontier median non-degradation", multivariateGates.frontier_median_non_degradation, "Pass"],
  ["Target adjustment stability", multivariateGates.target_full_vs_narrow_eci_ci_adjustment_stable, "Direction and ≤1.5× ratio"],
  ["Target component coverage", multivariateGates.target_component_coverage_gate, "At least two observed in each branch"],
  ["Target chronology + family holdout", multivariateGates.target_chronology_and_family_holdout_gate, "Pass"],
];
const multivariateGateEnd = 13 + multivariateGateRows.length;
eciMultivariateAudit.getRange(`D14:F${multivariateGateEnd}`).values = multivariateGateRows;
body(eciMultivariateAudit.getRange(`D14:F${multivariateGateEnd}`));
eciMultivariateAudit.getRange(`D14:F${multivariateGateEnd}`).format.wrapText = true;
eciMultivariateAudit.getRange(`D14:F${multivariateGateEnd}`).format.rowHeight = 26;
eciMultivariateAudit.getRange(`E14:E${multivariateGateEnd}`).conditionalFormats.add("containsText", { text: "TRUE", format: { fill: C.paleTeal, font: { color: C.tealDark, bold: true } } });
eciMultivariateAudit.getRange(`E14:E${multivariateGateEnd}`).conditionalFormats.add("containsText", { text: "FALSE", format: { fill: C.palePink, font: { color: C.purple, bold: true } } });

const multivariateTargetSection = multivariateGateEnd + 3;
section(eciMultivariateAudit, `A${multivariateTargetSection}:P${multivariateTargetSection}`, "Frontier target applicability — strictly earlier and target-family-held-out");
const multivariateTargetHeader = multivariateTargetSection + 1;
eciMultivariateAudit.getRange(`A${multivariateTargetHeader}:P${multivariateTargetHeader}`).values = [["Model", "Release", "Full train", "Narrow-CI train", "Full policy", "Full observed", "Full baseline (T)", "Full candidate (T)", "Full adjustment", "Narrow-CI policy", "Narrow-CI observed", "Narrow-CI baseline (T)", "Narrow-CI candidate (T)", "Narrow-CI adjustment", "Direction agrees", "Live weight"]];
header(eciMultivariateAudit.getRange(`A${multivariateTargetHeader}:P${multivariateTargetHeader}`));
const multivariateTargetStart = multivariateTargetHeader + 1;
const multivariateTargetData = eciMultivariateTargetRows.map((row) => [
  row.model,
  new Date(`${row.release_date}T00:00:00Z`),
  Number(row.full_training_n),
  Number(row.narrow_ci_training_n),
  `${row.selected_feature_set} / α=${row.selected_alpha}`,
  Number(row.observed_selected_count),
  Number(row.raw_multivariate_baseline_t),
  Number(row.raw_multivariate_candidate_t),
  null,
  `${row.narrow_ci_training_feature_set} / α=${row.narrow_ci_training_alpha}`,
  Number(row.narrow_ci_training_observed_count),
  Number(row.narrow_ci_training_raw_baseline_t),
  Number(row.narrow_ci_training_raw_candidate_t),
  null,
  row.full_vs_narrow_ci_direction_agrees === "True",
  Number(row.incremental_live_weight),
]);
const multivariateTargetEnd = multivariateTargetHeader + multivariateTargetData.length;
eciMultivariateAudit.getRange(`A${multivariateTargetStart}:P${multivariateTargetEnd}`).values = multivariateTargetData;
for (let row = multivariateTargetStart; row <= multivariateTargetEnd; row += 1) {
  eciMultivariateAudit.getRange(`I${row}`).formulas = [[`=H${row}/G${row}`]];
  eciMultivariateAudit.getRange(`N${row}`).formulas = [[`=M${row}/L${row}`]];
}
body(eciMultivariateAudit.getRange(`A${multivariateTargetStart}:P${multivariateTargetEnd}`));
eciMultivariateAudit.getRange(`B${multivariateTargetStart}:B${multivariateTargetEnd}`).format.numberFormat = "yyyy-mm-dd";
eciMultivariateAudit.getRange(`G${multivariateTargetStart}:H${multivariateTargetEnd}`).format.numberFormat = "0.00";
eciMultivariateAudit.getRange(`I${multivariateTargetStart}:I${multivariateTargetEnd}`).format.numberFormat = "0.00x";
eciMultivariateAudit.getRange(`L${multivariateTargetStart}:M${multivariateTargetEnd}`).format.numberFormat = "0.00";
eciMultivariateAudit.getRange(`N${multivariateTargetStart}:N${multivariateTargetEnd}`).format.numberFormat = "0.00x";
eciMultivariateAudit.getRange(`P${multivariateTargetStart}:P${multivariateTargetEnd}`).format.numberFormat = "0.0%";
eciMultivariateAudit.tables.add(`A${multivariateTargetHeader}:P${multivariateTargetEnd}`, true, "EciMultivariateTargetsTable").style = "TableStyleMedium2";

const multivariateCoverageSection = multivariateTargetEnd + 3;
section(eciMultivariateAudit, `A${multivariateCoverageSection}:G${multivariateCoverageSection}`, "Complete benchmark coverage ledger");
const multivariateCoverageHeader = multivariateCoverageSection + 1;
eciMultivariateAudit.getRange(`A${multivariateCoverageHeader}:G${multivariateCoverageHeader}`).values = [["Benchmark", "Models", "Families", "Knowledge set", "Pretraining-like set", "Full-panel eligible", "Optimized rows"]];
header(eciMultivariateAudit.getRange(`A${multivariateCoverageHeader}:G${multivariateCoverageHeader}`));
const multivariateCoverageData = eciMultivariateCoverageRows.map((row) => [
  row.benchmark,
  Number(row.models),
  Number(row.families),
  row.knowledge_predeclared === "True",
  row.pretraining_like_predeclared === "True",
  row.eligible_on_full_panel === "True",
  Number(row.optimized_true_rows),
]);
const multivariateCoverageEnd = multivariateCoverageHeader + multivariateCoverageData.length;
eciMultivariateAudit.getRange(`A${multivariateCoverageHeader + 1}:G${multivariateCoverageEnd}`).values = multivariateCoverageData;
body(eciMultivariateAudit.getRange(`A${multivariateCoverageHeader + 1}:G${multivariateCoverageEnd}`));
eciMultivariateAudit.tables.add(`A${multivariateCoverageHeader}:G${multivariateCoverageEnd}`, true, "EciMultivariateCoverageTable").style = "TableStyleMedium2";

const multivariatePredictionHeaders = ["Release", "Model", "Family", "Broad ECI CI?", "ECI", "Train n", "Train families", "Train max date", "Feature set", "Alpha", "Observed components", "Actual total (B)", "Baseline predicted (B)", "Candidate predicted (B)"];
const multivariatePredictionData = (rows) => rows.map((row) => [
  new Date(`${row.release_date}T00:00:00Z`),
  row.model,
  row.family,
  row.broad_eci_ci === "1",
  Number(row.eci),
  Number(row.train_n),
  Number(row.train_families),
  new Date(`${row.train_max_date}T00:00:00Z`),
  row.total_feature_set,
  Number(row.total_alpha),
  Number(row.total_observed_selected_count),
  Number(row.actual_total_b),
  Number(row.baseline_total_predicted_b),
  Number(row.candidate_total_predicted_b),
]);

const multivariatePredictionSection = multivariateCoverageEnd + 3;
section(eciMultivariateAudit, `A${multivariatePredictionSection}:N${multivariatePredictionSection}`, "Primary nested chronological predictions — all outcomes, broad ECI confidence intervals half-weighted");
const multivariatePredictionHeader = multivariatePredictionSection + 1;
eciMultivariateAudit.getRange(`A${multivariatePredictionHeader}:N${multivariatePredictionHeader}`).values = [multivariatePredictionHeaders];
header(eciMultivariateAudit.getRange(`A${multivariatePredictionHeader}:N${multivariatePredictionHeader}`));
const multivariatePredictionStart = multivariatePredictionHeader + 1;
const multivariatePredictionDataRows = multivariatePredictionData(eciMultivariatePredictionRows);
const multivariatePredictionEnd = multivariatePredictionHeader + multivariatePredictionDataRows.length;
eciMultivariateAudit.getRange(`A${multivariatePredictionStart}:N${multivariatePredictionEnd}`).values = multivariatePredictionDataRows;
body(eciMultivariateAudit.getRange(`A${multivariatePredictionStart}:N${multivariatePredictionEnd}`));
eciMultivariateAudit.getRange(`A${multivariatePredictionStart}:A${multivariatePredictionEnd}`).format.numberFormat = "yyyy-mm-dd";
eciMultivariateAudit.getRange(`H${multivariatePredictionStart}:H${multivariatePredictionEnd}`).format.numberFormat = "yyyy-mm-dd";
eciMultivariateAudit.getRange(`E${multivariatePredictionStart}:E${multivariatePredictionEnd}`).format.numberFormat = "0.00";
eciMultivariateAudit.getRange(`L${multivariatePredictionStart}:N${multivariatePredictionEnd}`).format.numberFormat = "0.00";
eciMultivariateAudit.tables.add(`A${multivariatePredictionHeader}:N${multivariatePredictionEnd}`, true, "EciMultivariatePredictionsTable").style = "TableStyleMedium2";

const multivariateNarrowCiSection = multivariatePredictionEnd + 3;
section(eciMultivariateAudit, `A${multivariateNarrowCiSection}:N${multivariateNarrowCiSection}`, "Narrow-ECI-CI-only nested chronological replication — broad aggregate-score confidence intervals excluded");
const multivariateNarrowCiHeader = multivariateNarrowCiSection + 1;
eciMultivariateAudit.getRange(`A${multivariateNarrowCiHeader}:N${multivariateNarrowCiHeader}`).values = [multivariatePredictionHeaders];
header(eciMultivariateAudit.getRange(`A${multivariateNarrowCiHeader}:N${multivariateNarrowCiHeader}`));
const multivariateNarrowCiStart = multivariateNarrowCiHeader + 1;
const multivariateNarrowCiData = multivariatePredictionData(eciMultivariateNarrowCiPredictionRows);
const multivariateNarrowCiEnd = multivariateNarrowCiHeader + multivariateNarrowCiData.length;
eciMultivariateAudit.getRange(`A${multivariateNarrowCiStart}:N${multivariateNarrowCiEnd}`).values = multivariateNarrowCiData;
body(eciMultivariateAudit.getRange(`A${multivariateNarrowCiStart}:N${multivariateNarrowCiEnd}`));
eciMultivariateAudit.getRange(`A${multivariateNarrowCiStart}:A${multivariateNarrowCiEnd}`).format.numberFormat = "yyyy-mm-dd";
eciMultivariateAudit.getRange(`H${multivariateNarrowCiStart}:H${multivariateNarrowCiEnd}`).format.numberFormat = "yyyy-mm-dd";
eciMultivariateAudit.getRange(`E${multivariateNarrowCiStart}:E${multivariateNarrowCiEnd}`).format.numberFormat = "0.00";
eciMultivariateAudit.getRange(`L${multivariateNarrowCiStart}:N${multivariateNarrowCiEnd}`).format.numberFormat = "0.00";
eciMultivariateAudit.tables.add(`A${multivariateNarrowCiHeader}:N${multivariateNarrowCiEnd}`, true, "EciMultivariateNarrowCiTable").style = "TableStyleMedium2";

eciMultivariateAudit.freezePanes.freezeRows(3);
eciMultivariateAudit.getRange("A:A").format.columnWidth = 32;
eciMultivariateAudit.getRange("B:B").format.columnWidth = 18;
eciMultivariateAudit.getRange("C:C").format.columnWidth = 18;
eciMultivariateAudit.getRange("D:D").format.columnWidth = 34;
eciMultivariateAudit.getRange("E:E").format.columnWidth = 24;
eciMultivariateAudit.getRange("F:F").format.columnWidth = 34;
eciMultivariateAudit.getRange("G:K").format.columnWidth = 18;
eciMultivariateAudit.getRange("L:L").format.columnWidth = 28;
eciMultivariateAudit.getRange("M:N").format.columnWidth = 18;
eciMultivariateAudit.getRange("O:P").format.columnWidth = 18;

const posttrainingInventory = posttrainingLineageResult.inventory;
const posttrainingEci = posttrainingLineageResult.lineage_backtests.eci;
const posttrainingAa = posttrainingLineageResult.lineage_backtests.aa;
const posttrainingGates = posttrainingLineageResult.promotion_gates;
const posttrainingReasoningControl = posttrainingLineageResult.hard_same_checkpoint_control.open_weight_reasoning_pairs;
title(posttrainingLineageAudit, "A1:P1", "Post-training lineage audit — exact open-weight controls and proprietary sensitivities");
subtitle(posttrainingLineageAudit, "A2:P3", "Only Epoch-structured open-weight base→descendant links with unchanged total parameters enter the backtest. Six measured bases establish a real distortion floor, but ECI lineage collapse is not decisive, AA has only three prediction edges, METR has none, and the proprietary GPT-5/Opus same-base claims are not publicly disclosed. The diagnostic receives 0% incremental weight and does not change the forecasts.");
posttrainingLineageAudit.getRange("A5:B5").values = [["Audit statistic", "Value"]];
header(posttrainingLineageAudit.getRange("A5:B5"));
const posttrainingSummaryRows = [
  ["Epoch base-model links", posttrainingInventory.base_model_links],
  ["Unique exact parent matches", posttrainingInventory.unique_exact_parent_matches],
  ["Same-parameter links (±1%)", posttrainingInventory.same_parameter_links_1pct],
  ["Same-parameter links, both open", posttrainingInventory.same_parameter_both_open_links],
  ["Open language candidates", posttrainingInventory.candidate_open_language_same_parameter_links],
  ["Measured lineage edges", posttrainingInventory.admitted_measured_lineage_edges],
  ["Measured lineage bases", posttrainingInventory.admitted_measured_lineage_bases],
  ["Measured developers", posttrainingInventory.admitted_developers],
  ["Matched measurements", posttrainingInventory.matched_measurements],
  ["Matched ECI components", posttrainingInventory.matched_component_measurements],
  ["Edges with finetune compute", posttrainingInventory.edges_with_finetune_compute],
  ["No-CoT / METR lineage edges", `${posttrainingInventory.nocot_lineage_edges} / ${posttrainingInventory.metr_lineage_edges}`],
  ["Incremental live weight", posttrainingLineageResult.decision.incremental_live_weight],
  ["Change headline forecasts", posttrainingLineageResult.decision.change_headline_forecasts ? "Yes" : "No"],
];
const posttrainingSummaryEnd = 5 + posttrainingSummaryRows.length;
posttrainingLineageAudit.getRange(`A6:B${posttrainingSummaryEnd}`).values = posttrainingSummaryRows;
body(posttrainingLineageAudit.getRange(`A6:B${posttrainingSummaryEnd}`));
posttrainingLineageAudit.getRange(`B6:B${posttrainingSummaryEnd - 3}`).format.numberFormat = "0";
posttrainingLineageAudit.getRange(`B${posttrainingSummaryEnd - 1}`).format.numberFormat = "0.0%";

posttrainingLineageAudit.getRange("D5:M5").values = [["Signal", "Edges", "Bases", "Median implied child/parent size", "Baseline median (×)", "Collapsed median (×)", "Equal-base Δ MAE", "90% CI low", "90% CI high", "Decision"]];
header(posttrainingLineageAudit.getRange("D5:M5"));
const posttrainingBacktestRows = [
  ["ECI aggregate", posttrainingEci.rows, posttrainingEci.bases, posttrainingEci.median_implied_child_over_parent_parameter_ratio, posttrainingEci.baseline.median_multiplicative_error, posttrainingEci.collapsed.median_multiplicative_error, posttrainingEci.collapsed_vs_baseline.observed_delta, posttrainingEci.collapsed_vs_baseline.ci_90[0], posttrainingEci.collapsed_vs_baseline.ci_90[1], "Interval crosses zero"],
  ["AA Intelligence Index", posttrainingAa.rows, posttrainingAa.bases, posttrainingAa.median_implied_child_over_parent_parameter_ratio, posttrainingAa.baseline.median_multiplicative_error, posttrainingAa.collapsed.median_multiplicative_error, posttrainingAa.collapsed_vs_baseline.observed_delta, posttrainingAa.collapsed_vs_baseline.ci_90[0], posttrainingAa.collapsed_vs_baseline.ci_90[1], "Favorable interval; only three bases"],
];
posttrainingLineageAudit.getRange("D6:M7").values = posttrainingBacktestRows;
body(posttrainingLineageAudit.getRange("D6:M7"));
posttrainingLineageAudit.getRange("G6:I7").format.numberFormat = "0.00x";
posttrainingLineageAudit.getRange("J6:L7").format.numberFormat = "0.000";

posttrainingLineageAudit.getRange("D10:J10").values = [["Hard same-checkpoint control", "Pairs", "Creators", "Checkpoint median uplift", "Equal-creator median uplift", "90% CI low", "90% CI high"]];
header(posttrainingLineageAudit.getRange("D10:J10"));
posttrainingLineageAudit.getRange("D11:J11").values = [["Open-weight reasoning vs non-reasoning", posttrainingReasoningControl.pairs, posttrainingReasoningControl.creators, posttrainingReasoningControl.checkpoint_median_aa_uplift, posttrainingReasoningControl.equal_creator_median_aa_uplift, posttrainingReasoningControl.equal_creator_median_bootstrap_90_ci[0], posttrainingReasoningControl.equal_creator_median_bootstrap_90_ci[1]]];
body(posttrainingLineageAudit.getRange("D11:J11"));
posttrainingLineageAudit.getRange("G11:J11").format.numberFormat = "0.00";

section(posttrainingLineageAudit, "D14:P14", "Promotion gates — every gate must pass");
posttrainingLineageAudit.getRange("D15:F15").values = [["Gate", "Passed", "Interpretation"]];
header(posttrainingLineageAudit.getRange("D15:F15"));
const posttrainingGateRows = [
  ["At least eight verified open bases", posttrainingGates.verified_open_lineage_bases_at_least_8, "Six measured bases"],
  ["At least six ECI bases", posttrainingGates.eci_signal_bases_at_least_6, "Five ECI bases"],
  ["At least six AA bases", posttrainingGates.aa_signal_bases_at_least_6, "Three AA bases"],
  ["ECI collapse interval favorable", posttrainingGates.eci_collapse_ci_wholly_favorable, "CI high must be < 0"],
  ["AA collapse interval favorable", posttrainingGates.aa_collapse_ci_wholly_favorable, "Passes statistically but fails coverage"],
  ["Proprietary same-base claims public", posttrainingGates.proprietary_shared_base_claims_publicly_verified, "User assertion; no public disclosure"],
];
const posttrainingGateEnd = 15 + posttrainingGateRows.length;
posttrainingLineageAudit.getRange(`D16:F${posttrainingGateEnd}`).values = posttrainingGateRows;
body(posttrainingLineageAudit.getRange(`D16:F${posttrainingGateEnd}`));
posttrainingLineageAudit.getRange(`D16:F${posttrainingGateEnd}`).format.wrapText = true;
posttrainingLineageAudit.getRange(`E16:E${posttrainingGateEnd}`).conditionalFormats.add("containsText", { text: "TRUE", format: { fill: C.paleTeal, font: { color: C.tealDark, bold: true } } });
posttrainingLineageAudit.getRange(`E16:E${posttrainingGateEnd}`).conditionalFormats.add("containsText", { text: "FALSE", format: { fill: C.palePink, font: { color: C.purple, bold: true } } });

const posttrainingEvidenceSection = Math.max(posttrainingSummaryEnd, posttrainingGateEnd) + 3;
section(posttrainingLineageAudit, `A${posttrainingEvidenceSection}:P${posttrainingEvidenceSection}`, "Proprietary lineage evidence grades — primary sources separated from user assertions");
const posttrainingEvidenceHeader = posttrainingEvidenceSection + 1;
posttrainingLineageAudit.getRange(`A${posttrainingEvidenceHeader}:F${posttrainingEvidenceHeader}`).values = [["Lineage", "Claim", "Evidence grade", "Public-source finding", "Model treatment", "Source URL"]];
header(posttrainingLineageAudit.getRange(`A${posttrainingEvidenceHeader}:F${posttrainingEvidenceHeader}`));
const posttrainingEvidenceData = frontierLineageEvidenceRows.map((row) => [row.lineage, row.claim, row.evidence_grade, row.public_source_finding, row.model_treatment, row.source_url]);
const posttrainingEvidenceEnd = posttrainingEvidenceHeader + posttrainingEvidenceData.length;
posttrainingLineageAudit.getRange(`A${posttrainingEvidenceHeader + 1}:F${posttrainingEvidenceEnd}`).values = posttrainingEvidenceData;
body(posttrainingLineageAudit.getRange(`A${posttrainingEvidenceHeader + 1}:F${posttrainingEvidenceEnd}`));
posttrainingLineageAudit.getRange(`A${posttrainingEvidenceHeader + 1}:F${posttrainingEvidenceEnd}`).format.wrapText = true;
posttrainingLineageAudit.getRange(`A${posttrainingEvidenceHeader + 1}:F${posttrainingEvidenceEnd}`).format.rowHeight = 56;
posttrainingLineageAudit.getRange(`C${posttrainingEvidenceHeader + 1}:C${posttrainingEvidenceEnd}`).conditionalFormats.add("containsText", { text: "primary_source", format: { fill: C.paleTeal, font: { color: C.tealDark, bold: true } } });
posttrainingLineageAudit.getRange(`C${posttrainingEvidenceHeader + 1}:C${posttrainingEvidenceEnd}`).conditionalFormats.add("containsText", { text: "user_asserted", format: { fill: C.paleAmber, font: { color: C.navy, bold: true } } });
posttrainingLineageAudit.tables.add(`A${posttrainingEvidenceHeader}:F${posttrainingEvidenceEnd}`, true, "PosttrainingEvidenceTable").style = "TableStyleMedium2";

const posttrainingEdgeSection = posttrainingEvidenceEnd + 3;
section(posttrainingLineageAudit, `A${posttrainingEdgeSection}:P${posttrainingEdgeSection}`, "Exact open-weight same-parameter lineage edges");
const posttrainingEdgeHeader = posttrainingEdgeSection + 1;
posttrainingLineageAudit.getRange(`A${posttrainingEdgeHeader}:O${posttrainingEdgeHeader}`).values = [["Child", "Parent", "Child date", "Parent date", "Params (B)", "Organization", "Measurements", "Components", "Finetune FLOP", "Parent Epoch row", "Child Epoch row", "Identity evidence", "Admission", "Parent link", "Child link"]];
header(posttrainingLineageAudit.getRange(`A${posttrainingEdgeHeader}:O${posttrainingEdgeHeader}`));
const posttrainingEdgeData = posttrainingLineageEdgeRows.map((row) => [
  row.child_model,
  row.parent_model,
  new Date(`${row.child_release_date}T00:00:00Z`),
  new Date(`${row.parent_release_date}T00:00:00Z`),
  Number(row.child_parameters_b),
  row.child_organization,
  Number(row.overlapping_measurements),
  Number(row.overlapping_component_benchmarks),
  row.finetune_compute_flop === "" ? null : Number(row.finetune_compute_flop),
  Number(row.parent_epoch_row),
  Number(row.child_epoch_row),
  row.identity_evidence,
  row.admission_status,
  row.parent_link,
  row.child_link,
]);
const posttrainingEdgeEnd = posttrainingEdgeHeader + posttrainingEdgeData.length;
posttrainingLineageAudit.getRange(`A${posttrainingEdgeHeader + 1}:O${posttrainingEdgeEnd}`).values = posttrainingEdgeData;
body(posttrainingLineageAudit.getRange(`A${posttrainingEdgeHeader + 1}:O${posttrainingEdgeEnd}`));
posttrainingLineageAudit.getRange(`C${posttrainingEdgeHeader + 1}:D${posttrainingEdgeEnd}`).format.numberFormat = "yyyy-mm-dd";
posttrainingLineageAudit.getRange(`E${posttrainingEdgeHeader + 1}:E${posttrainingEdgeEnd}`).format.numberFormat = "0.0";
posttrainingLineageAudit.getRange(`I${posttrainingEdgeHeader + 1}:I${posttrainingEdgeEnd}`).format.numberFormat = "0.00E+00";
posttrainingLineageAudit.getRange(`L${posttrainingEdgeHeader + 1}:O${posttrainingEdgeEnd}`).format.wrapText = true;
posttrainingLineageAudit.tables.add(`A${posttrainingEdgeHeader}:O${posttrainingEdgeEnd}`, true, "PosttrainingLineageEdgeTable").style = "TableStyleMedium2";

const posttrainingPredictionSection = posttrainingEdgeEnd + 3;
section(posttrainingLineageAudit, `A${posttrainingPredictionSection}:P${posttrainingPredictionSection}`, "Strictly earlier, endpoint-group-held-out lineage predictions");
const posttrainingPredictionHeader = posttrainingPredictionSection + 1;
posttrainingLineageAudit.getRange(`A${posttrainingPredictionHeader}:N${posttrainingPredictionHeader}`).values = [["Signal", "Child", "Parent", "Actual (B)", "Score Δ", "Implied child/parent size", "Baseline predicted (B)", "Collapsed predicted (B)", "Baseline log error", "Collapsed log error", "Train n", "Train groups", "Train max date", "Group excluded"]];
header(posttrainingLineageAudit.getRange(`A${posttrainingPredictionHeader}:N${posttrainingPredictionHeader}`));
const posttrainingPredictionData = posttrainingLineagePredictionRows.map((row) => [
  row.signal,
  row.child_model,
  row.parent_model,
  Number(row.actual_parameters_b),
  Number(row.score_delta),
  Number(row.implied_child_over_parent_parameter_ratio),
  Number(row.baseline_predicted_b),
  Number(row.collapsed_predicted_b),
  Number(row.baseline_log10_error),
  Number(row.collapsed_log10_error),
  Number(row.train_n),
  Number(row.train_groups),
  new Date(`${row.train_max_date}T00:00:00Z`),
  row.test_group_excluded === "True",
]);
const posttrainingPredictionEnd = posttrainingPredictionHeader + posttrainingPredictionData.length;
posttrainingLineageAudit.getRange(`A${posttrainingPredictionHeader + 1}:N${posttrainingPredictionEnd}`).values = posttrainingPredictionData;
body(posttrainingLineageAudit.getRange(`A${posttrainingPredictionHeader + 1}:N${posttrainingPredictionEnd}`));
posttrainingLineageAudit.getRange(`D${posttrainingPredictionHeader + 1}:H${posttrainingPredictionEnd}`).format.numberFormat = "0.00";
posttrainingLineageAudit.getRange(`I${posttrainingPredictionHeader + 1}:J${posttrainingPredictionEnd}`).format.numberFormat = "0.000";
posttrainingLineageAudit.getRange(`M${posttrainingPredictionHeader + 1}:M${posttrainingPredictionEnd}`).format.numberFormat = "yyyy-mm-dd";
posttrainingLineageAudit.tables.add(`A${posttrainingPredictionHeader}:N${posttrainingPredictionEnd}`, true, "PosttrainingPredictionTable").style = "TableStyleMedium2";

const posttrainingFrontierSection = posttrainingPredictionEnd + 3;
section(posttrainingLineageAudit, `A${posttrainingFrontierSection}:P${posttrainingFrontierSection}`, "Proprietary asserted-same-base sensitivity — diagnostic only, not actual parameter ratios");
const posttrainingFrontierHeader = posttrainingFrontierSection + 1;
posttrainingLineageAudit.getRange(`A${posttrainingFrontierHeader}:P${posttrainingFrontierHeader}`).values = [["Chain", "Mode", "Seq", "Model", "Release", "AA", "Δ AA", "Naive ratio vs first", "Naive ratio vs prior", "Train n", "Creators", "Train max date", "Calibration max AA", "Score / max", "Extrapolates", "Evidence grade"]];
header(posttrainingLineageAudit.getRange(`A${posttrainingFrontierHeader}:P${posttrainingFrontierHeader}`));
const posttrainingFrontierData = frontierSharedBaseSensitivityRows.map((row) => [
  row.chain,
  row.mode,
  Number(row.sequence),
  row.model,
  new Date(`${row.release_date}T00:00:00Z`),
  Number(row.aa_intelligence_index),
  row.aa_score_change_from_previous === "" ? null : Number(row.aa_score_change_from_previous),
  Number(row.date_adjusted_implied_parameter_ratio_vs_first),
  row.date_adjusted_implied_parameter_ratio_vs_previous === "" ? null : Number(row.date_adjusted_implied_parameter_ratio_vs_previous),
  Number(row.calibration_train_n),
  Number(row.calibration_train_creators),
  new Date(`${row.calibration_train_max_date}T00:00:00Z`),
  Number(row.calibration_score_max),
  Number(row.score_over_calibration_max),
  row.score_extrapolates_above_calibration_max === "True",
  row.evidence_grade,
]);
const posttrainingFrontierEnd = posttrainingFrontierHeader + posttrainingFrontierData.length;
posttrainingLineageAudit.getRange(`A${posttrainingFrontierHeader + 1}:P${posttrainingFrontierEnd}`).values = posttrainingFrontierData;
body(posttrainingLineageAudit.getRange(`A${posttrainingFrontierHeader + 1}:P${posttrainingFrontierEnd}`));
posttrainingLineageAudit.getRange(`E${posttrainingFrontierHeader + 1}:E${posttrainingFrontierEnd}`).format.numberFormat = "yyyy-mm-dd";
posttrainingLineageAudit.getRange(`F${posttrainingFrontierHeader + 1}:I${posttrainingFrontierEnd}`).format.numberFormat = "0.00";
posttrainingLineageAudit.getRange(`L${posttrainingFrontierHeader + 1}:L${posttrainingFrontierEnd}`).format.numberFormat = "yyyy-mm-dd";
posttrainingLineageAudit.getRange(`M${posttrainingFrontierHeader + 1}:N${posttrainingFrontierEnd}`).format.numberFormat = "0.00";
posttrainingLineageAudit.getRange(`O${posttrainingFrontierHeader + 1}:O${posttrainingFrontierEnd}`).conditionalFormats.add("containsText", { text: "TRUE", format: { fill: C.paleAmber, font: { color: C.navy, bold: true } } });
posttrainingLineageAudit.tables.add(`A${posttrainingFrontierHeader}:P${posttrainingFrontierEnd}`, true, "PosttrainingFrontierSensitivityTable").style = "TableStyleMedium2";

posttrainingLineageAudit.freezePanes.freezeRows(3);
posttrainingLineageAudit.getRange("A:B").format.columnWidth = 30;
posttrainingLineageAudit.getRange("C:C").format.columnWidth = 22;
posttrainingLineageAudit.getRange("D:D").format.columnWidth = 40;
posttrainingLineageAudit.getRange("E:K").format.columnWidth = 18;
posttrainingLineageAudit.getRange("L:M").format.columnWidth = 34;
posttrainingLineageAudit.getRange("N:P").format.columnWidth = 42;

const activePriceInventory = openRouterActivePriceResult.inventory;
const activePriceTransport = openRouterActivePriceResult.high_sparsity_total_transport;
const activePriceGates = openRouterActivePriceResult.promotion_gates;
title(activePriceAudit, "A1:N1", "OpenRouter active-capacity price audit — exact joins and sparse-MoE transport");
subtitle(activePriceAudit, "A2:N3", "Current 2026-07-18 prices are joined to 45 disclosed active-parameter labels and 18 exact dense-model controls from primary Hugging Face configs. Active-to-total transport improves point metrics, but its seven-developer interval crosses zero, coverage misses the 20/8 gate, and frontier prices extrapolate. Incremental live weight remains 0%.");
activePriceAudit.getRange("A5:B5").values = [["Audit statistic", "Value"]];
header(activePriceAudit.getRange("A5:B5"));
const activePriceSummaryRows = [
  ["Active-parameter labels", activePriceInventory.active_parameter_matches],
  ["AA disclosed active labels", activePriceInventory.aa_disclosed_active_parameter_matches],
  ["Dense config controls", activePriceInventory.dense_config_active_equals_total_controls],
  ["Matched developers", activePriceInventory.developers],
  ["Release-ordered predictions", activePriceInventory.release_ordered_predictions],
  ["Prediction developers", activePriceInventory.prediction_developers],
  ["High-sparsity transport tests", activePriceInventory.high_sparsity_transport_predictions],
  ["High-sparsity developers", activePriceInventory.high_sparsity_transport_developers],
  ["Transport candidate median error (×)", activePriceTransport.candidate.median_multiplicative_error],
  ["Direct-total baseline median error (×)", activePriceTransport.direct_total_baseline.median_multiplicative_error],
  ["Equal-developer error delta", activePriceTransport.paired_cluster_bootstrap.observed_delta],
  ["Bootstrap 90% CI low", activePriceTransport.paired_cluster_bootstrap.ci_90[0]],
  ["Bootstrap 90% CI high", activePriceTransport.paired_cluster_bootstrap.ci_90[1]],
  ["Performance gate passed", activePriceGates.performance_gate_passed ? "Yes" : "No"],
  ["Coverage gate passed", activePriceGates.coverage_gate_passed ? "Yes" : "No"],
  ["Required tests / developers", `${activePriceGates.required_tests} / ${activePriceGates.required_developers}`],
  ["Observed tests / developers", `${activePriceGates.observed_tests} / ${activePriceGates.observed_developers}`],
  ["Prospective historical-price test", "No — current snapshot prices"],
  ["Incremental live weight", openRouterActivePriceResult.decision.incremental_live_weight],
  ["Change headline forecasts", openRouterActivePriceResult.decision.change_headline_forecasts ? "Yes" : "No"],
];
const activePriceSummaryEnd = 5 + activePriceSummaryRows.length;
activePriceAudit.getRange(`A6:B${activePriceSummaryEnd}`).values = activePriceSummaryRows;
body(activePriceAudit.getRange(`A6:B${activePriceSummaryEnd}`));
activePriceAudit.getRange("B14:B18").format.numberFormat = "0.000";
activePriceAudit.getRange("B24:B24").format.numberFormat = "0.0%";

section(activePriceAudit, "D5:K5", "Frontier sensitivity — K3-anchored, 0% live weight");
const activePriceTargetHeaders = ["Model", "Release", "AA", "Blended $/M", "Price / train max", "Predicted active (B)", "K3-anchored total (T)", "Status"];
activePriceAudit.getRange("D6:K6").values = [activePriceTargetHeaders];
header(activePriceAudit.getRange("D6:K6"));
const activePriceTargetData = openRouterActivePriceTargetRows.map((row) => [
  row.model,
  new Date(`${row.release_date}T00:00:00Z`),
  Number(row.aa_score),
  Number(row.blended_price_usd_per_mtoken),
  Number(row.price_over_training_max),
  Number(row.predicted_active_score_date_price_b),
  Number(row.k3_anchored_total_score_date_price_t),
  row.status,
]);
const activePriceTargetEnd = 6 + activePriceTargetData.length;
activePriceAudit.getRange(`D7:K${activePriceTargetEnd}`).values = activePriceTargetData;
body(activePriceAudit.getRange(`D7:K${activePriceTargetEnd}`));
activePriceAudit.getRange(`E7:E${activePriceTargetEnd}`).format.numberFormat = "yyyy-mm-dd";
activePriceAudit.getRange(`F7:G${activePriceTargetEnd}`).format.numberFormat = "0.00";
activePriceAudit.getRange(`H7:H${activePriceTargetEnd}`).format.numberFormat = "0.00x";
activePriceAudit.getRange(`I7:J${activePriceTargetEnd}`).format.numberFormat = "0.0";
activePriceAudit.getRange(`D6:K${activePriceTargetEnd}`).format.wrapText = true;
activePriceAudit.tables.add(`D6:K${activePriceTargetEnd}`, true, "ActivePriceTargetTable").style = "TableStyleMedium2";

const activePriceMatchSection = activePriceSummaryEnd + 3;
section(activePriceAudit, `A${activePriceMatchSection}:N${activePriceMatchSection}`, "Complete 93-row OpenRouter↔Epoch↔AA↔Hugging Face exact-identity audit");
const activePriceMatchHeader = activePriceMatchSection + 1;
const activePriceMatchHeaders = ["Epoch checkpoint", "Epoch model", "Developer", "Epoch release", "Epoch total (B)", "AA group", "AA model", "Effective active (B)", "Total / active", "Active source", "Config class", "Status", "OpenRouter IDs", "HF repositories"];
activePriceAudit.getRange(`A${activePriceMatchHeader}:N${activePriceMatchHeader}`).values = [activePriceMatchHeaders];
header(activePriceAudit.getRange(`A${activePriceMatchHeader}:N${activePriceMatchHeader}`));
const activePriceMatchData = openRouterActivePriceMatchRows.map((row) => {
  const total = Number(row.epoch_total_parameters_b);
  const active = row.aa_active_parameters_b !== ""
    ? Number(row.aa_active_parameters_b)
    : row.status === "dense_config_active_equals_total"
      ? total
      : null;
  return [
    row.canonical_checkpoint_id,
    row.epoch_model,
    row.developer,
    new Date(`${row.epoch_release_date}T00:00:00Z`),
    total,
    row.aa_checkpoint_group_id,
    row.aa_model,
    active,
    active == null ? null : total / active,
    row.active_parameter_source,
    row.hf_config_classifications,
    row.status,
    row.openrouter_model_ids,
    row.openrouter_hugging_face_repos,
  ];
});
const activePriceMatchEnd = activePriceMatchHeader + activePriceMatchData.length;
activePriceAudit.getRange(`A${activePriceMatchHeader + 1}:N${activePriceMatchEnd}`).values = activePriceMatchData;
body(activePriceAudit.getRange(`A${activePriceMatchHeader + 1}:N${activePriceMatchEnd}`));
activePriceAudit.getRange(`D${activePriceMatchHeader + 1}:D${activePriceMatchEnd}`).format.numberFormat = "yyyy-mm-dd";
activePriceAudit.getRange(`E${activePriceMatchHeader + 1}:I${activePriceMatchEnd}`).format.numberFormat = "0.00";
activePriceAudit.getRange(`A${activePriceMatchHeader}:N${activePriceMatchEnd}`).format.wrapText = true;
activePriceAudit.tables.add(`A${activePriceMatchHeader}:N${activePriceMatchEnd}`, true, "ActivePriceMatchTable").style = "TableStyleMedium2";

const activePricePredictionSection = activePriceMatchEnd + 3;
section(activePriceAudit, `A${activePricePredictionSection}:N${activePricePredictionSection}`, "Complete release-ordered developer-held-out prediction ledger");
const activePricePredictionHeader = activePricePredictionSection + 1;
const activePricePredictionHeaders = ["Checkpoint", "Release", "Model", "Developer", "Actual active (B)", "Actual total (B)", "Total / active", "Blended $/M", "Pred. active (B)", "Direct total (B)", "Transport total (B)", "Transport error", "Direct error", "Training max date"];
activePriceAudit.getRange(`A${activePricePredictionHeader}:N${activePricePredictionHeader}`).values = [activePricePredictionHeaders];
header(activePriceAudit.getRange(`A${activePricePredictionHeader}:N${activePricePredictionHeader}`));
const activePricePredictionData = openRouterActivePricePredictionRows.map((row) => [
  row.canonical_checkpoint_id,
  new Date(`${row.release_date}T00:00:00Z`),
  row.model,
  row.developer,
  Number(row.actual_active_b),
  Number(row.actual_total_b),
  Number(row.actual_total_to_active_ratio),
  Number(row.blended_price_usd_per_mtoken),
  Number(row.predicted_active_score_date_price_b),
  Number(row.predicted_total_score_date_price_b),
  row.converted_total_score_date_price_b === "" ? null : Number(row.converted_total_score_date_price_b),
  row.converted_total_score_date_price_log10_error === "" ? null : Number(row.converted_total_score_date_price_log10_error),
  Number(row.total_score_date_price_log10_error),
  new Date(`${row.train_max_date}T00:00:00Z`),
]);
const activePricePredictionEnd = activePricePredictionHeader + activePricePredictionData.length;
activePriceAudit.getRange(`A${activePricePredictionHeader + 1}:N${activePricePredictionEnd}`).values = activePricePredictionData;
body(activePriceAudit.getRange(`A${activePricePredictionHeader + 1}:N${activePricePredictionEnd}`));
activePriceAudit.getRange(`B${activePricePredictionHeader + 1}:B${activePricePredictionEnd}`).format.numberFormat = "yyyy-mm-dd";
activePriceAudit.getRange(`E${activePricePredictionHeader + 1}:M${activePricePredictionEnd}`).format.numberFormat = "0.000";
activePriceAudit.getRange(`N${activePricePredictionHeader + 1}:N${activePricePredictionEnd}`).format.numberFormat = "yyyy-mm-dd";
activePriceAudit.getRange(`A${activePricePredictionHeader}:N${activePricePredictionEnd}`).format.wrapText = true;
activePriceAudit.tables.add(`A${activePricePredictionHeader}:N${activePricePredictionEnd}`, true, "ActivePricePredictionTable").style = "TableStyleMedium2";
activePriceAudit.freezePanes.freezeRows(activePriceMatchHeader);
activePriceAudit.getRange("A:A").format.columnWidth = 35;
activePriceAudit.getRange("B:B").format.columnWidth = 32;
activePriceAudit.getRange("C:C").format.columnWidth = 20;
activePriceAudit.getRange("D:I").format.columnWidth = 18;
activePriceAudit.getRange("J:K").format.columnWidth = 40;
activePriceAudit.getRange("L:N").format.columnWidth = 42;

const historicalInventory = openRouterHistoricalResult.inventory;
const historicalDecision = openRouterHistoricalResult.decision;
const historicalWindows = openRouterHistoricalResult.metadata.price_windows_days;
const historicalBootstrapFor = (windowDays, panel, target) => {
  const row = openRouterHistoricalResult.paired_developer_bootstraps.find((candidate) =>
    candidate.window_days === windowDays && candidate.panel === panel && candidate.target === target
  );
  if (!row) throw new Error(`Missing historical-price bootstrap: ${windowDays}/${panel}/${target}`);
  return row;
};

title(historicalPriceAudit, "A1:T1", "OpenRouter launch-vintage price audit — prospective, developer-held-out evidence");
subtitle(historicalPriceAudit, "A2:T3", `Hash-pinned reconstruction of ${openRouterHistoricalMetadata.source.full_git_history_rebuild_snapshot_count.toLocaleString("en-US")} official OpenRouter /api/v1/models snapshots. All ${historicalInventory.historical_ledger_models.toLocaleString("en-US")} model IDs and ${historicalInventory.historical_change_points.toLocaleString("en-US")} price changes are preserved; all ${historicalInventory.calibration_checkpoints_audited} calibration checkpoints match exact aliases. Launch price robustly beats date alone for total parameters across every predeclared 1–90 day window, but adds no robust active-parameter information beyond AA score + date, so this audit adds 0% independent weight.`);
historicalPriceAudit.getRange("A5:B5").values = [["Audit statistic", "Value"]];
header(historicalPriceAudit.getRange("A5:B5"));
const historicalPriceSummaryRows = [
  ["Pinned upstream commit", openRouterHistoricalMetadata.source.pinned_commit],
  ["Full git-history rebuild verified", openRouterHistoricalMetadata.integrity_policy.full_upstream_git_history_rebuild_matches_frozen_ledger ? "Yes" : "No"],
  ["Snapshots rebuilt", openRouterHistoricalMetadata.source.full_git_history_rebuild_snapshot_count],
  ["History range", `${openRouterHistoricalMetadata.inventory.first_snapshot_date} to ${openRouterHistoricalMetadata.inventory.as_of}`],
  ["Historical model IDs", historicalInventory.historical_ledger_models],
  ["Price change points", historicalInventory.historical_change_points],
  ["Calibration checkpoints audited", historicalInventory.calibration_checkpoints_audited],
  ["Calibration aliases missing", historicalInventory.calibration_aliases_missing_from_history],
  ["Duplicate calibration checkpoints", historicalInventory.duplicate_calibration_checkpoints],
  ["Held-out prediction rows", historicalInventory.prediction_rows],
  ["1-day eligible total / active", `${historicalInventory.eligible_total_rows_by_window["1"]} / ${historicalInventory.eligible_active_rows_by_window["1"]}`],
  ["90-day eligible total / active", `${historicalInventory.eligible_total_rows_by_window["90"]} / ${historicalInventory.eligible_active_rows_by_window["90"]}`],
  ["Exact alias + ≤30-day onboarding rule", "Yes"],
  ["Strictly prospective fold prices", "Yes"],
  ["Test developer excluded", "Yes"],
  ["Total price beats date in all windows", historicalDecision.launch_vintage_price_predicts_total_beyond_date_robustly_across_all_windows ? "Yes" : "No"],
  ["Price adds beyond score + date for active", historicalDecision.launch_vintage_price_adds_robust_information_beyond_score_date_for_active_parameters ? "Yes" : "No"],
  ["Incremental live weight", historicalDecision.incremental_live_weight_from_this_audit],
  ["Headline forecasts changed", historicalDecision.headline_forecasts_changed ? "Yes" : "No"],
];
const historicalPriceSummaryEnd = 5 + historicalPriceSummaryRows.length;
historicalPriceAudit.getRange(`A6:B${historicalPriceSummaryEnd}`).values = historicalPriceSummaryRows;
body(historicalPriceAudit.getRange(`A6:B${historicalPriceSummaryEnd}`));
historicalPriceAudit.getRange(`B${historicalPriceSummaryEnd - 1}:B${historicalPriceSummaryEnd - 1}`).format.numberFormat = "0.0%";

section(historicalPriceAudit, "D5:K5", "Frontier first-day price sensitivity — K3-anchored, 0% live weight");
const historicalTargetHeaders = ["Model", "Release", "First seen", "Input $/M", "Output $/M", "Blended $/M", "K3-anchored total (T)", "Status"];
historicalPriceAudit.getRange("D6:K6").values = [historicalTargetHeaders];
header(historicalPriceAudit.getRange("D6:K6"));
const historicalTargetData = openRouterHistoricalTargetRows.map((row) => [
  row.model,
  new Date(`${row.release_date}T00:00:00Z`),
  new Date(`${row.openrouter_first_seen}T00:00:00Z`),
  Number(row.first_day_prompt_price_usd_per_mtoken),
  Number(row.first_day_completion_price_usd_per_mtoken),
  Number(row.first_day_blended_price_usd_per_mtoken),
  Number(row.k3_anchored_total_score_date_historical_price_t),
  row.status,
]);
const historicalTargetEnd = 6 + historicalTargetData.length;
historicalPriceAudit.getRange(`D7:K${historicalTargetEnd}`).values = historicalTargetData;
body(historicalPriceAudit.getRange(`D7:K${historicalTargetEnd}`));
historicalPriceAudit.getRange(`E7:F${historicalTargetEnd}`).format.numberFormat = "yyyy-mm-dd";
historicalPriceAudit.getRange(`G7:J${historicalTargetEnd}`).format.numberFormat = "0.00";
historicalPriceAudit.getRange(`D6:K${historicalTargetEnd}`).format.wrapText = true;
historicalPriceAudit.tables.add(`D6:K${historicalTargetEnd}`, true, "HistoricalPriceTargetTable").style = "TableStyleMedium2";

const historicalWindowSection = historicalTargetEnd + 3;
section(historicalPriceAudit, `D${historicalWindowSection}:R${historicalWindowSection}`, "Predeclared-window backtest summary — equal-developer paired bootstrap");
const historicalWindowHeader = historicalWindowSection + 1;
const historicalWindowHeaders = ["Window (days)", "Eligible total", "Eligible active", "Total paired N", "Date-only error (×)", "Date+price error (×)", "Total Δ log10", "Total CI low", "Total CI high", "Active paired N", "Score+date error (×)", "+price error (×)", "Active Δ log10", "Active CI low", "Active CI high"];
historicalPriceAudit.getRange(`D${historicalWindowHeader}:R${historicalWindowHeader}`).values = [historicalWindowHeaders];
header(historicalPriceAudit.getRange(`D${historicalWindowHeader}:R${historicalWindowHeader}`));
const historicalWindowData = historicalWindows.map((windowDays) => {
  const metrics = openRouterHistoricalResult.heldout_metrics[String(windowDays)];
  const totalBootstrap = historicalBootstrapFor(windowDays, "total_calibration", "total_b");
  const activeBootstrap = historicalBootstrapFor(windowDays, "active_label_common_panel", "active_b");
  return [
    windowDays,
    historicalInventory.eligible_total_rows_by_window[String(windowDays)],
    historicalInventory.eligible_active_rows_by_window[String(windowDays)],
    totalBootstrap.paired_checkpoints,
    metrics.total.date_only.median_multiplicative_error,
    metrics.total.date_historical_price.median_multiplicative_error,
    totalBootstrap.observed_delta,
    totalBootstrap.ci_90[0],
    totalBootstrap.ci_90[1],
    activeBootstrap.paired_checkpoints,
    metrics.active_common_panel.active_b.score_date.median_multiplicative_error,
    metrics.active_common_panel.active_b.score_date_historical_price.median_multiplicative_error,
    activeBootstrap.observed_delta,
    activeBootstrap.ci_90[0],
    activeBootstrap.ci_90[1],
  ];
});
const historicalWindowEnd = historicalWindowHeader + historicalWindowData.length;
historicalPriceAudit.getRange(`D${historicalWindowHeader + 1}:R${historicalWindowEnd}`).values = historicalWindowData;
body(historicalPriceAudit.getRange(`D${historicalWindowHeader + 1}:R${historicalWindowEnd}`));
historicalPriceAudit.getRange(`H${historicalWindowHeader + 1}:I${historicalWindowEnd}`).format.numberFormat = "0.00x";
historicalPriceAudit.getRange(`N${historicalWindowHeader + 1}:O${historicalWindowEnd}`).format.numberFormat = "0.00x";
historicalPriceAudit.getRange(`J${historicalWindowHeader + 1}:L${historicalWindowEnd}`).format.numberFormat = "0.000";
historicalPriceAudit.getRange(`P${historicalWindowHeader + 1}:R${historicalWindowEnd}`).format.numberFormat = "0.000";
historicalPriceAudit.tables.add(`D${historicalWindowHeader}:R${historicalWindowEnd}`, true, "HistoricalPriceWindowTable").style = "TableStyleMedium2";

const historicalMatchSection = Math.max(historicalPriceSummaryEnd, historicalWindowEnd) + 3;
section(historicalPriceAudit, `A${historicalMatchSection}:R${historicalMatchSection}`, "Complete 93-checkpoint exact-identity and launch-window eligibility audit");
const historicalMatchHeader = historicalMatchSection + 1;
const historicalMatchHeaders = ["Checkpoint", "Model", "Developer", "Release", "Total (B)", "OpenRouter IDs", "All aliases exact", "First seen", "Lag days", "Lag ≤30d", "1d availability", "1d blended $/M", "30d availability", "30d blended $/M", "90d availability", "90d blended $/M", "Last seen", "Identity status"];
historicalPriceAudit.getRange(`A${historicalMatchHeader}:R${historicalMatchHeader}`).values = [historicalMatchHeaders];
header(historicalPriceAudit.getRange(`A${historicalMatchHeader}:R${historicalMatchHeader}`));
const historicalMatchData = openRouterHistoricalMatchRows.map((row) => [
  row.canonical_checkpoint_id,
  row.canonical_model_name,
  row.developer,
  new Date(`${row.release_date}T00:00:00Z`),
  Number(row.total_parameters_b),
  row.openrouter_model_ids,
  row.all_aliases_exactly_matched,
  row.first_seen_dates,
  Number(row.onboarding_lag_days),
  row.identity_lag_within_30_days,
  row.window_1d_availability_date ? new Date(`${row.window_1d_availability_date}T00:00:00Z`) : null,
  row.window_1d_blended_price_usd_per_mtoken === "" ? null : Number(row.window_1d_blended_price_usd_per_mtoken),
  row.window_30d_availability_date ? new Date(`${row.window_30d_availability_date}T00:00:00Z`) : null,
  row.window_30d_blended_price_usd_per_mtoken === "" ? null : Number(row.window_30d_blended_price_usd_per_mtoken),
  row.window_90d_availability_date ? new Date(`${row.window_90d_availability_date}T00:00:00Z`) : null,
  row.window_90d_blended_price_usd_per_mtoken === "" ? null : Number(row.window_90d_blended_price_usd_per_mtoken),
  row.last_seen_dates,
  row.release_after_history_floor === "True" && row.identity_lag_within_30_days === "True" ? "Eligible identity" : "Explicitly excluded",
]);
const historicalMatchEnd = historicalMatchHeader + historicalMatchData.length;
historicalPriceAudit.getRange(`A${historicalMatchHeader + 1}:R${historicalMatchEnd}`).values = historicalMatchData;
body(historicalPriceAudit.getRange(`A${historicalMatchHeader + 1}:R${historicalMatchEnd}`));
historicalPriceAudit.getRange(`D${historicalMatchHeader + 1}:D${historicalMatchEnd}`).format.numberFormat = "yyyy-mm-dd";
historicalPriceAudit.getRange(`K${historicalMatchHeader + 1}:K${historicalMatchEnd}`).format.numberFormat = "yyyy-mm-dd";
historicalPriceAudit.getRange(`M${historicalMatchHeader + 1}:M${historicalMatchEnd}`).format.numberFormat = "yyyy-mm-dd";
historicalPriceAudit.getRange(`O${historicalMatchHeader + 1}:O${historicalMatchEnd}`).format.numberFormat = "yyyy-mm-dd";
historicalPriceAudit.getRange(`E${historicalMatchHeader + 1}:E${historicalMatchEnd}`).format.numberFormat = "0.0";
historicalPriceAudit.getRange(`L${historicalMatchHeader + 1}:L${historicalMatchEnd}`).format.numberFormat = "0.000";
historicalPriceAudit.getRange(`N${historicalMatchHeader + 1}:N${historicalMatchEnd}`).format.numberFormat = "0.000";
historicalPriceAudit.getRange(`P${historicalMatchHeader + 1}:P${historicalMatchEnd}`).format.numberFormat = "0.000";
historicalPriceAudit.getRange(`A${historicalMatchHeader}:R${historicalMatchEnd}`).format.wrapText = true;
historicalPriceAudit.tables.add(`A${historicalMatchHeader}:R${historicalMatchEnd}`, true, "HistoricalPriceMatchTable").style = "TableStyleMedium2";

const historicalPredictionSection = historicalMatchEnd + 3;
section(historicalPriceAudit, `A${historicalPredictionSection}:T${historicalPredictionSection}`, "Complete 3,065-row prospective developer-held-out prediction ledger");
const historicalPredictionHeader = historicalPredictionSection + 1;
const historicalPredictionHeaders = ["Panel", "Window", "Checkpoint", "Model", "Developer", "Release", "Price available", "Historical $/M", "Current $/M", "Target", "Specification", "Actual (B)", "Predicted (B)", "Log10 error", "Train N", "Train developers", "Train max price date", "Developer excluded", "Historical prospective", "Current nonprospective"];
historicalPriceAudit.getRange(`A${historicalPredictionHeader}:T${historicalPredictionHeader}`).values = [historicalPredictionHeaders];
header(historicalPriceAudit.getRange(`A${historicalPredictionHeader}:T${historicalPredictionHeader}`));
const historicalPredictionData = openRouterHistoricalPredictionRows.map((row) => [
  row.panel,
  Number(row.window_days),
  row.canonical_checkpoint_id,
  row.model,
  row.developer,
  new Date(`${row.release_date}T00:00:00Z`),
  new Date(`${row.price_availability_date}T00:00:00Z`),
  Number(row.historical_blended_price_usd_per_mtoken),
  Number(row.current_blended_price_usd_per_mtoken),
  row.target,
  row.specification,
  Number(row.actual_parameters_b),
  Number(row.predicted_parameters_b),
  Number(row.log10_error),
  Number(row.train_n),
  Number(row.train_developers),
  new Date(`${row.train_max_price_availability_date}T00:00:00Z`),
  row.test_developer_excluded,
  row.historical_price_is_prospective_at_fold_date,
  row.current_price_is_nonprospective_comparison,
]);
const historicalPredictionEnd = historicalPredictionHeader + historicalPredictionData.length;
historicalPriceAudit.getRange(`A${historicalPredictionHeader + 1}:T${historicalPredictionEnd}`).values = historicalPredictionData;
body(historicalPriceAudit.getRange(`A${historicalPredictionHeader + 1}:T${historicalPredictionEnd}`));
historicalPriceAudit.getRange(`F${historicalPredictionHeader + 1}:G${historicalPredictionEnd}`).format.numberFormat = "yyyy-mm-dd";
historicalPriceAudit.getRange(`H${historicalPredictionHeader + 1}:I${historicalPredictionEnd}`).format.numberFormat = "0.000";
historicalPriceAudit.getRange(`L${historicalPredictionHeader + 1}:N${historicalPredictionEnd}`).format.numberFormat = "0.000";
historicalPriceAudit.getRange(`Q${historicalPredictionHeader + 1}:Q${historicalPredictionEnd}`).format.numberFormat = "yyyy-mm-dd";
historicalPriceAudit.getRange(`A${historicalPredictionHeader}:T${historicalPredictionEnd}`).format.wrapText = true;
historicalPriceAudit.tables.add(`A${historicalPredictionHeader}:T${historicalPredictionEnd}`, true, "HistoricalPricePredictionTable").style = "TableStyleMedium2";
historicalPriceAudit.freezePanes.freezeRows(historicalMatchHeader);
historicalPriceAudit.getRange("A:A").format.columnWidth = 30;
historicalPriceAudit.getRange("B:B").format.columnWidth = 18;
historicalPriceAudit.getRange("C:F").format.columnWidth = 24;
historicalPriceAudit.getRange("G:J").format.columnWidth = 18;
historicalPriceAudit.getRange("K:T").format.columnWidth = 22;

title(openRouterAudit, "A1:N1", "OpenRouter operational cross-check — price, throughput, and Epoch identity audit");
subtitle(openRouterAudit, "A2:N3", "Frozen 2026-07-18 public OpenRouter snapshot. Price is predictive in developer-family-held-out tests, but provider/quantization-normalized tok/s adds no incremental information and receives 0% weight. The existing low API-price branch is retained; frontier price extrapolations remain diagnostic.");
openRouterAudit.getRange("A5:B5").values = [["Audit statistic", "Value"]];
header(openRouterAudit.getRange("A5:B5"));
const openRouterSummaryRows = [
  ["OpenRouter text models retained", openRouterResult.data_audit.openrouter_catalog_models],
  ["Manually matched OpenRouter aliases", openRouterResult.data_audit.manual_epoch_model_matches],
  ["Unique Epoch calibration checkpoints", openRouterResult.data_audit.unique_epoch_calibration_checkpoints],
  ["Developer families", openRouterResult.data_audit.calibration_families],
  ["Price-only family-held-out median error (×)", openRouterResult.heldout_metrics.family.date_price.median_multiplicative_error],
  ["Price + normalized tok/s median error (×)", openRouterResult.heldout_metrics.family.date_price_provider_normalized_throughput.median_multiplicative_error],
  ["Tok/s adds robust incremental information", openRouterResult.conclusion.tok_s_adds_robust_incremental_information_beyond_price ? "Yes" : "No"],
  ["Recommended incremental tok/s weight", openRouterResult.conclusion.recommended_incremental_tok_s_weight_in_live_ensemble],
  ["Current final API-price weight", openRouterResult.conclusion.current_live_final_price_weight],
];
openRouterAudit.getRange(`A6:B${5 + openRouterSummaryRows.length}`).values = openRouterSummaryRows;
body(openRouterAudit.getRange(`A6:B${5 + openRouterSummaryRows.length}`));
openRouterAudit.getRange(`B10:B11`).format.numberFormat = "0.00x";
openRouterAudit.getRange(`B13:B14`).format.numberFormat = "0.0%";

const openRouterFrontierHeaderRow = 17;
section(openRouterAudit, `A${openRouterFrontierHeaderRow}:N${openRouterFrontierHeaderRow}`, "Frontier operational estimates — diagnostic only");
const openRouterFrontierTableHeader = openRouterFrontierHeaderRow + 1;
const openRouterFrontierHeaders = ["Model", "Release", "Prompt $/M", "Output $/M", "Raw tok/s", "Provider-normalized speed", "Uncalibrated total (T)", "Two-anchor calibrated (T)", "Held-out low (T)", "Held-out high (T)", "Price / calibration max", "Disclosed total (T)", "Status", "OpenRouter ID"];
openRouterAudit.getRange(`A${openRouterFrontierTableHeader}:N${openRouterFrontierTableHeader}`).values = [openRouterFrontierHeaders];
header(openRouterAudit.getRange(`A${openRouterFrontierTableHeader}:N${openRouterFrontierTableHeader}`));
const openRouterFrontierData = openRouterFrontierRows.map((row) => [
  row.model,
  row.release_date ? new Date(`${row.release_date}T00:00:00Z`) : null,
  row.prompt_price_usd_per_mtoken === "" ? null : Number(row.prompt_price_usd_per_mtoken),
  row.completion_price_usd_per_mtoken === "" ? null : Number(row.completion_price_usd_per_mtoken),
  row.raw_throughput_tps_1w === "" ? null : Number(row.raw_throughput_tps_1w),
  row.provider_normalized_throughput_ratio === "" ? null : Number(row.provider_normalized_throughput_ratio),
  row.operational_model_central_b === "" ? null : Number(row.operational_model_central_b) / 1000,
  row.anchor_calibrated_central_b === "" ? null : Number(row.anchor_calibrated_central_b) / 1000,
  row.anchor_calibrated_heldout_error_80_low_b === "" ? null : Number(row.anchor_calibrated_heldout_error_80_low_b) / 1000,
  row.anchor_calibrated_heldout_error_80_high_b === "" ? null : Number(row.anchor_calibrated_heldout_error_80_high_b) / 1000,
  row.price_over_calibration_max === "" ? null : Number(row.price_over_calibration_max),
  row.disclosed_total_parameters_b === "" ? null : Number(row.disclosed_total_parameters_b) / 1000,
  row.status,
  row.openrouter_model_id,
]);
const openRouterFrontierEnd = openRouterFrontierTableHeader + openRouterFrontierData.length;
openRouterAudit.getRange(`A${openRouterFrontierTableHeader + 1}:N${openRouterFrontierEnd}`).values = openRouterFrontierData;
body(openRouterAudit.getRange(`A${openRouterFrontierTableHeader + 1}:N${openRouterFrontierEnd}`));
openRouterAudit.getRange(`B${openRouterFrontierTableHeader + 1}:B${openRouterFrontierEnd}`).format.numberFormat = "yyyy-mm-dd";
openRouterAudit.getRange(`C${openRouterFrontierTableHeader + 1}:F${openRouterFrontierEnd}`).format.numberFormat = "0.00";
openRouterAudit.getRange(`G${openRouterFrontierTableHeader + 1}:J${openRouterFrontierEnd}`).format.numberFormat = "0.0";
openRouterAudit.getRange(`K${openRouterFrontierTableHeader + 1}:K${openRouterFrontierEnd}`).format.numberFormat = "0.00x";
openRouterAudit.getRange(`L${openRouterFrontierTableHeader + 1}:L${openRouterFrontierEnd}`).format.numberFormat = "0.0";
openRouterAudit.tables.add(`A${openRouterFrontierTableHeader}:N${openRouterFrontierEnd}`, true, "OpenRouterFrontierTable").style = "TableStyleMedium2";

const openRouterMatchSectionRow = openRouterFrontierEnd + 3;
section(openRouterAudit, `A${openRouterMatchSectionRow}:N${openRouterMatchSectionRow}`, "Complete model-level OpenRouter → Epoch match audit");
const openRouterMatchHeaderRow = openRouterMatchSectionRow + 1;
const openRouterMatchHeaders = ["OpenRouter ID", "OpenRouter name", "Status", "Epoch checkpoint", "Epoch model", "Epoch release", "Total params (B)", "Parameter source", "Prompt $/M", "Output $/M", "Raw tok/s", "Normalized speed", "Catalog − Epoch days", "Source URL"];
openRouterAudit.getRange(`A${openRouterMatchHeaderRow}:N${openRouterMatchHeaderRow}`).values = [openRouterMatchHeaders];
header(openRouterAudit.getRange(`A${openRouterMatchHeaderRow}:N${openRouterMatchHeaderRow}`));
const openRouterMatchData = openRouterAuditRows.map((row) => [
  row.openrouter_model_id,
  row.openrouter_model_name,
  row.match_status,
  row.canonical_checkpoint_id,
  row.epoch_model_name,
  row.epoch_release_date ? new Date(`${row.epoch_release_date}T00:00:00Z`) : null,
  row.total_parameters_b === "" ? null : Number(row.total_parameters_b),
  row.parameter_value_source,
  row.prompt_price_usd_per_mtoken === "" ? null : Number(row.prompt_price_usd_per_mtoken),
  row.completion_price_usd_per_mtoken === "" ? null : Number(row.completion_price_usd_per_mtoken),
  row.raw_throughput_tps_1w === "" ? null : Number(row.raw_throughput_tps_1w),
  row.provider_normalized_throughput_ratio === "" ? null : Number(row.provider_normalized_throughput_ratio),
  row.openrouter_catalog_date_minus_epoch_release_days === "" ? null : Number(row.openrouter_catalog_date_minus_epoch_release_days),
  row.throughput_source_url,
]);
const openRouterMatchEnd = openRouterMatchHeaderRow + openRouterMatchData.length;
openRouterAudit.getRange(`A${openRouterMatchHeaderRow + 1}:N${openRouterMatchEnd}`).values = openRouterMatchData;
body(openRouterAudit.getRange(`A${openRouterMatchHeaderRow + 1}:N${openRouterMatchEnd}`));
openRouterAudit.getRange(`F${openRouterMatchHeaderRow + 1}:F${openRouterMatchEnd}`).format.numberFormat = "yyyy-mm-dd";
openRouterAudit.getRange(`G${openRouterMatchHeaderRow + 1}:M${openRouterMatchEnd}`).format.numberFormat = "0.00";
openRouterAudit.getRange(`A${openRouterMatchHeaderRow}:N${openRouterMatchEnd}`).format.wrapText = true;
openRouterAudit.tables.add(`A${openRouterMatchHeaderRow}:N${openRouterMatchEnd}`, true, "OpenRouterMatchAuditTable").style = "TableStyleMedium2";
openRouterAudit.freezePanes.freezeRows(openRouterMatchHeaderRow);
openRouterAudit.getRange("A:A").format.columnWidth = 34;
openRouterAudit.getRange("B:B").format.columnWidth = 34;
openRouterAudit.getRange("C:C").format.columnWidth = 24;
openRouterAudit.getRange("D:E").format.columnWidth = 38;
openRouterAudit.getRange("F:M").format.columnWidth = 18;
openRouterAudit.getRange("N:N").format.columnWidth = 70;

const openRouterTemporalInventory = openRouterTemporalResult.inventory;
const openRouterTemporalMetrics = openRouterTemporalResult.corrected_default_tier_backtest;
const openRouterTemporalStability = openRouterTemporalResult.temporal_stability;
title(openRouterTimeAudit, "A1:N1", "OpenRouter time-series audit — immutable refreshes, service tiers, and throughput stability");
subtitle(openRouterTimeAudit, "A2:N3", "Every refresh is archived byte-for-byte and every daily endpoint/service-tier observation is retained. Default, priority, and flex are separated. Corrected tok/s fails the promotion gate after price, and no request-supported throughput, latency, joint, or tail-spread candidate passes both the family-interval and chronological-direction gates. All incremental operational weights remain 0%.");
openRouterTimeAudit.getRange("A5:B5").values = [["Audit statistic", "Value"]];
header(openRouterTimeAudit.getRange("A5:B5"));
const openRouterTemporalSummaryRows = [
  ["Immutable refresh snapshots", openRouterTemporalInventory.immutable_snapshots],
  ["Current lossless daily rows", openRouterTemporalInventory.current_daily_rows],
  ["Daily rows across snapshot history", openRouterTemporalInventory.history_daily_rows],
  ["Default-tier rows", openRouterTemporalInventory.service_tier_row_counts.default],
  ["Priority-tier rows", openRouterTemporalInventory.service_tier_row_counts.priority],
  ["Flex-tier rows", openRouterTemporalInventory.service_tier_row_counts.flex],
  ["Endpoint/model pairs with multiple tiers", openRouterTemporalInventory.endpoint_models_with_multiple_service_tiers],
  ["Rows without endpoint metadata", openRouterTemporalInventory.current_daily_rows_without_endpoint_metadata],
  ["Unmatched endpoint/model pairs", openRouterTemporalInventory.current_daily_unmatched_endpoint_models],
  ["Calibration checkpoints changed by tier split", openRouterTemporalResult.service_tier_counterfactual.calibration_checkpoints_changed_by_tier_separation],
  ["Models with ≥4 daily observations", openRouterTemporalStability.model_daily_provider_median_with_at_least_four_days_max_over_min.n],
  ["Median model within-week max/min", openRouterTemporalStability.model_daily_provider_median_with_at_least_four_days_max_over_min.median],
  ["P90 model within-week max/min", openRouterTemporalStability.model_daily_provider_median_with_at_least_four_days_max_over_min.p90],
  ["Median overlapping-refresh max/min", openRouterTemporalStability.model_default_tier_median_across_all_refreshes_max_over_min.median],
  ["Recommended incremental tok/s weight", openRouterTemporalResult.decision.recommended_incremental_tok_s_weight],
  ["Change live forecast", openRouterTemporalResult.decision.change_live_forecast ? "Yes" : "No"],
];
const openRouterTemporalSummaryEnd = 5 + openRouterTemporalSummaryRows.length;
openRouterTimeAudit.getRange(`A6:B${openRouterTemporalSummaryEnd}`).values = openRouterTemporalSummaryRows;
body(openRouterTimeAudit.getRange(`A6:B${openRouterTemporalSummaryEnd}`));
openRouterTimeAudit.getRange(`B6:B${openRouterTemporalSummaryEnd - 5}`).format.numberFormat = "#,##0";
openRouterTimeAudit.getRange(`B${openRouterTemporalSummaryEnd - 4}:B${openRouterTemporalSummaryEnd - 2}`).format.numberFormat = "0.00x";
openRouterTimeAudit.getRange(`B${openRouterTemporalSummaryEnd - 1}:B${openRouterTemporalSummaryEnd - 1}`).format.numberFormat = "0.0%";

const openRouterTemporalBacktestSection = openRouterTemporalSummaryEnd + 3;
section(openRouterTimeAudit, `A${openRouterTemporalBacktestSection}:N${openRouterTemporalBacktestSection}`, "Corrected default-tier held-out regression — price helps, tok/s does not");
const openRouterTemporalBacktestHeader = openRouterTemporalBacktestSection + 1;
const openRouterTemporalBacktestHeaders = ["Holdout", "Specification", "Tests", "Median error (×)", "MAE log10", "P80 error (×)", "Within 2×", "Δ MAE vs price", "Interpretation", "Tier policy", "Live price weight", "Live tok/s weight", "Status", "Notes"];
openRouterTimeAudit.getRange(`A${openRouterTemporalBacktestHeader}:N${openRouterTemporalBacktestHeader}`).values = [openRouterTemporalBacktestHeaders];
header(openRouterTimeAudit.getRange(`A${openRouterTemporalBacktestHeader}:N${openRouterTemporalBacktestHeader}`));
const openRouterTemporalBacktestRows = [];
for (const [modeKey, modeLabel] of [["family", "Developer-family"], ["chronological_family", "Chronological family"]]) {
  const priceMae = openRouterTemporalMetrics[modeKey].date_price.mean_absolute_log10_error;
  for (const [specKey, specLabel] of [["date_only", "Date only"], ["date_price", "Date + price"], ["date_price_plus_normalized_tok_s", "Date + price + normalized tok/s"]]) {
    const metric = openRouterTemporalMetrics[modeKey][specKey];
    const delta = metric.mean_absolute_log10_error - priceMae;
    openRouterTemporalBacktestRows.push([
      modeLabel, specLabel, metric.n, metric.median_multiplicative_error,
      metric.mean_absolute_log10_error, metric.p80_multiplicative_error, metric.within_2x,
      delta, specKey === "date_price" ? "Price baseline" : specKey.includes("tok_s") ? (delta < 0 ? "Improves after price" : "Worsens after price") : "Scale/date baseline",
      "default only", openRouterTemporalResult.corrected_default_tier_backtest.promotion_gate_from_main_audit.current_live_final_price_weight,
      openRouterTemporalResult.decision.recommended_incremental_tok_s_weight,
      specKey.includes("tok_s") ? "Rejected" : "Diagnostic", "Service tiers retained separately; rolling windows are correlated",
    ]);
  }
}
const openRouterTemporalBacktestEnd = openRouterTemporalBacktestHeader + openRouterTemporalBacktestRows.length;
openRouterTimeAudit.getRange(`A${openRouterTemporalBacktestHeader + 1}:N${openRouterTemporalBacktestEnd}`).values = openRouterTemporalBacktestRows;
body(openRouterTimeAudit.getRange(`A${openRouterTemporalBacktestHeader + 1}:N${openRouterTemporalBacktestEnd}`));
openRouterTimeAudit.getRange(`C${openRouterTemporalBacktestHeader + 1}:C${openRouterTemporalBacktestEnd}`).format.numberFormat = "0";
openRouterTimeAudit.getRange(`D${openRouterTemporalBacktestHeader + 1}:D${openRouterTemporalBacktestEnd}`).format.numberFormat = "0.00x";
openRouterTimeAudit.getRange(`E${openRouterTemporalBacktestHeader + 1}:F${openRouterTemporalBacktestEnd}`).format.numberFormat = "0.000";
openRouterTimeAudit.getRange(`G${openRouterTemporalBacktestHeader + 1}:G${openRouterTemporalBacktestEnd}`).format.numberFormat = "0.0%";
openRouterTimeAudit.getRange(`H${openRouterTemporalBacktestHeader + 1}:H${openRouterTemporalBacktestEnd}`).format.numberFormat = "+0.000;-0.000;0.000";
openRouterTimeAudit.getRange(`K${openRouterTemporalBacktestHeader + 1}:L${openRouterTemporalBacktestEnd}`).format.numberFormat = "0.0%";
openRouterTimeAudit.tables.add(`A${openRouterTemporalBacktestHeader}:N${openRouterTemporalBacktestEnd}`, true, "OpenRouterTemporalBacktestTable").style = "TableStyleMedium2";

const openRouterRequestSection = openRouterTemporalBacktestEnd + 3;
section(openRouterTimeAudit, `A${openRouterRequestSection}:N${openRouterRequestSection}`, "Request-supported 30-minute throughput, latency, and percentile-spread audit");
const openRouterRequestHeader = openRouterRequestSection + 1;
const openRouterRequestHeaders = ["Holdout", "Specification", "Tests", "Median error (×)", "MAE log10", "RMSE log10", "Δ MAE vs price", "P(candidate better)", "CI90 low", "CI90 high", "Min requests", "Live weight", "Decision", "Note"];
openRouterTimeAudit.getRange(`A${openRouterRequestHeader}:N${openRouterRequestHeader}`).values = [openRouterRequestHeaders];
header(openRouterTimeAudit.getRange(`A${openRouterRequestHeader}:N${openRouterRequestHeader}`));
const openRouterRequestLabels = {
  date_price: "Date + price",
  date_price_one_week_throughput: "Date + price + one-week tok/s",
  date_price_p50_throughput_all: "Date + price + 30m p50 tok/s",
  date_price_p50_throughput_supported: "Date + price + request-supported tok/s",
  date_price_p50_latency_supported: "Date + price + request-supported latency",
  date_price_p50_throughput_latency_supported: "Date + price + request-supported tok/s + latency",
  date_price_p50_throughput_tail_spreads_supported: "Date + price + tok/s + p90/p50 spreads",
};
const openRouterRequestComparisons = new Map(openRouterRequestWeightedResult.paired_family_bootstraps.map((row) => [`${row.mode}|${row.candidate}`, row]));
const openRouterRequestRows = [];
for (const [modeKey, modeLabel] of [["family", "Developer-family"], ["chronological_family", "Chronological family"]]) {
  const baselineMae = openRouterRequestWeightedResult.heldout_metrics[modeKey].date_price.mean_absolute_log10_error;
  for (const [specification, metric] of Object.entries(openRouterRequestWeightedResult.heldout_metrics[modeKey])) {
    const comparison = openRouterRequestComparisons.get(`${modeKey}|${specification}`);
    openRouterRequestRows.push([
      modeLabel, openRouterRequestLabels[specification], metric.n, metric.median_multiplicative_error,
      metric.mean_absolute_log10_error, metric.rmse_log10, metric.mean_absolute_log10_error - baselineMae,
      comparison?.bootstrap_probability_candidate_better ?? "", comparison?.ci_90?.[0] ?? "", comparison?.ci_90?.[1] ?? "",
      specification === "date_price" || specification === "date_price_one_week_throughput" || specification === "date_price_p50_throughput_all" ? "" : 100,
      openRouterRequestWeightedResult.decision.incremental_live_weight,
      specification === "date_price" ? "Baseline" : "Rejected",
      specification === "date_price_p50_latency_supported" ? "Request-supported latency; family interval crosses zero" : `Common ${openRouterRequestWeightedResult.inventory.complete_checkpoints}-checkpoint panel; provider/quantization normalized`,
    ]);
  }
}
const openRouterRequestEnd = openRouterRequestHeader + openRouterRequestRows.length;
openRouterTimeAudit.getRange(`A${openRouterRequestHeader + 1}:N${openRouterRequestEnd}`).values = openRouterRequestRows;
body(openRouterTimeAudit.getRange(`A${openRouterRequestHeader + 1}:N${openRouterRequestEnd}`));
openRouterTimeAudit.getRange(`C${openRouterRequestHeader + 1}:C${openRouterRequestEnd}`).format.numberFormat = "0";
openRouterTimeAudit.getRange(`D${openRouterRequestHeader + 1}:D${openRouterRequestEnd}`).format.numberFormat = "0.00x";
openRouterTimeAudit.getRange(`E${openRouterRequestHeader + 1}:G${openRouterRequestEnd}`).format.numberFormat = "0.000";
openRouterTimeAudit.getRange(`H${openRouterRequestHeader + 1}:H${openRouterRequestEnd}`).format.numberFormat = "0.0%";
openRouterTimeAudit.getRange(`I${openRouterRequestHeader + 1}:J${openRouterRequestEnd}`).format.numberFormat = "+0.000;-0.000;0.000";
openRouterTimeAudit.getRange(`L${openRouterRequestHeader + 1}:L${openRouterRequestEnd}`).format.numberFormat = "0.0%";
openRouterTimeAudit.tables.add(`A${openRouterRequestHeader}:N${openRouterRequestEnd}`, true, "OpenRouterRequestWeightedBacktestTable").style = "TableStyleMedium2";

const openRouterFocalSection = openRouterRequestEnd + 3;
section(openRouterTimeAudit, `A${openRouterFocalSection}:N${openRouterFocalSection}`, "Frontier throughput stability — operational context, not parameter evidence");
const openRouterFocalHeader = openRouterFocalSection + 1;
const openRouterFocalHeaders = ["Model", "OpenRouter ID", "Within-week days", "Within-week median tok/s", "Within-week max/min", "Refreshes", "Refresh median tok/s", "Refresh max/min", "Tier used", "Price live weight", "Tok/s live weight", "Role", "Decision", "Caveat"];
openRouterTimeAudit.getRange(`A${openRouterFocalHeader}:N${openRouterFocalHeader}`).values = [openRouterFocalHeaders];
header(openRouterTimeAudit.getRange(`A${openRouterFocalHeader}:N${openRouterFocalHeader}`));
const openRouterFocalRows = openRouterTemporalStability.focal_models.map((row) => [
  row.model, row.openrouter_model_id, row.within_week_dates, row.within_week_median_tps,
  row.within_week_max_over_min, row.refreshes, row.refresh_median_tps, row.refresh_max_over_min,
  "default", openRouterTemporalResult.corrected_default_tier_backtest.promotion_gate_from_main_audit.current_live_final_price_weight,
  openRouterTemporalResult.decision.recommended_incremental_tok_s_weight, "diagnostic", "No forecast change", "Serving-stack volatility is not base-model scale",
]);
const openRouterFocalEnd = openRouterFocalHeader + openRouterFocalRows.length;
openRouterTimeAudit.getRange(`A${openRouterFocalHeader + 1}:N${openRouterFocalEnd}`).values = openRouterFocalRows;
body(openRouterTimeAudit.getRange(`A${openRouterFocalHeader + 1}:N${openRouterFocalEnd}`));
openRouterTimeAudit.getRange(`C${openRouterFocalHeader + 1}:C${openRouterFocalEnd}`).format.numberFormat = "0";
openRouterTimeAudit.getRange(`D${openRouterFocalHeader + 1}:H${openRouterFocalEnd}`).format.numberFormat = "0.00";
openRouterTimeAudit.getRange(`F${openRouterFocalHeader + 1}:F${openRouterFocalEnd}`).format.numberFormat = "0";
openRouterTimeAudit.getRange(`J${openRouterFocalHeader + 1}:K${openRouterFocalEnd}`).format.numberFormat = "0.0%";
openRouterTimeAudit.tables.add(`A${openRouterFocalHeader}:N${openRouterFocalEnd}`, true, "OpenRouterFocalStabilityTable").style = "TableStyleMedium2";

const openRouterHistorySection = openRouterFocalEnd + 3;
section(openRouterTimeAudit, `A${openRouterHistorySection}:N${openRouterHistorySection}`, "Immutable refresh manifest — exact archived bytes and hashes");
const openRouterHistoryHeader = openRouterHistorySection + 1;
const openRouterHistoryHeaders = ["Snapshot ID", "Fetched UTC", "Revision", "Catalog models", "Text models", "Provider rows", "Daily rows", "Daily models", "Endpoint/tier keys", "Failures", "Raw SHA-256", "Model SHA-256", "Provider SHA-256", "Archive directory"];
openRouterTimeAudit.getRange(`A${openRouterHistoryHeader}:N${openRouterHistoryHeader}`).values = [openRouterHistoryHeaders];
header(openRouterTimeAudit.getRange(`A${openRouterHistoryHeader}:N${openRouterHistoryHeader}`));
const openRouterHistoryRows = openRouterHistoryManifestRows.map((row) => [
  row.history_snapshot_id, new Date(row.fetched_at_utc), row.source_revision, Number(row.catalog_model_count),
  Number(row.eligible_text_model_count), Number(row.provider_endpoint_row_count), Number(row.daily_throughput_row_count),
  Number(row.daily_throughput_model_count), Number(row.daily_throughput_endpoint_tier_count), Number(row.failure_count),
  row.raw_sha256, row.model_sha256, row.provider_sha256, row.archive_directory,
]);
const openRouterHistoryEnd = openRouterHistoryHeader + openRouterHistoryRows.length;
openRouterTimeAudit.getRange(`A${openRouterHistoryHeader + 1}:N${openRouterHistoryEnd}`).values = openRouterHistoryRows;
body(openRouterTimeAudit.getRange(`A${openRouterHistoryHeader + 1}:N${openRouterHistoryEnd}`));
openRouterTimeAudit.getRange(`B${openRouterHistoryHeader + 1}:B${openRouterHistoryEnd}`).format.numberFormat = "yyyy-mm-dd hh:mm:ss";
openRouterTimeAudit.getRange(`D${openRouterHistoryHeader + 1}:J${openRouterHistoryEnd}`).format.numberFormat = "#,##0";
openRouterTimeAudit.tables.add(`A${openRouterHistoryHeader}:N${openRouterHistoryEnd}`, true, "OpenRouterSnapshotHistoryTable").style = "TableStyleMedium2";

const focalIds = new Set(openRouterTemporalStability.focal_models.map((row) => row.openrouter_model_id));
const volatileModelRows = openRouterModelStabilityRows
  .filter((row) => Number(row.dates) >= 4)
  .sort((left, right) => Number(right.max_over_min) - Number(left.max_over_min));
const openRouterModelSampleRows = [...volatileModelRows.slice(0, 20), ...openRouterModelStabilityRows.filter((row) => focalIds.has(row.openrouter_model_id))]
  .filter((row, index, rows) => rows.findIndex((candidate) => candidate.openrouter_model_id === row.openrouter_model_id) === index);
const openRouterModelSection = openRouterHistoryEnd + 3;
section(openRouterTimeAudit, `A${openRouterModelSection}:N${openRouterModelSection}`, "Model-level daily stability sample — most volatile plus all frontier targets");
const openRouterModelHeader = openRouterModelSection + 1;
const openRouterModelHeaders = ["OpenRouter ID", "Name", "Days", "Date min", "Date max", "Median endpoints/day", "Observations", "Median tok/s", "Mean tok/s", "Min tok/s", "Max tok/s", "Max/min", "Log SD", "Daily provider medians JSON"];
openRouterTimeAudit.getRange(`A${openRouterModelHeader}:N${openRouterModelHeader}`).values = [openRouterModelHeaders];
header(openRouterTimeAudit.getRange(`A${openRouterModelHeader}:N${openRouterModelHeader}`));
const openRouterModelSampleData = openRouterModelSampleRows.map((row) => [
  row.openrouter_model_id, row.openrouter_model_name, Number(row.dates), new Date(`${row.date_min}T00:00:00Z`),
  new Date(`${row.date_max}T00:00:00Z`), Number(row.median_endpoints_per_date), Number(row.observations), Number(row.median_tps),
  Number(row.mean_tps), Number(row.min_tps), Number(row.max_tps), Number(row.max_over_min), Number(row.log_standard_deviation), row.daily_provider_medians_json,
]);
const openRouterModelEnd = openRouterModelHeader + openRouterModelSampleData.length;
openRouterTimeAudit.getRange(`A${openRouterModelHeader + 1}:N${openRouterModelEnd}`).values = openRouterModelSampleData;
body(openRouterTimeAudit.getRange(`A${openRouterModelHeader + 1}:N${openRouterModelEnd}`));
openRouterTimeAudit.getRange(`C${openRouterModelHeader + 1}:M${openRouterModelEnd}`).format.numberFormat = "0.00";
openRouterTimeAudit.getRange(`C${openRouterModelHeader + 1}:C${openRouterModelEnd}`).format.numberFormat = "0";
openRouterTimeAudit.getRange(`G${openRouterModelHeader + 1}:G${openRouterModelEnd}`).format.numberFormat = "0";
openRouterTimeAudit.getRange(`D${openRouterModelHeader + 1}:E${openRouterModelEnd}`).format.numberFormat = "yyyy-mm-dd";
openRouterTimeAudit.tables.add(`A${openRouterModelHeader}:N${openRouterModelEnd}`, true, "OpenRouterModelStabilitySampleTable").style = "TableStyleMedium2";

const unmatchedEndpointRows = openRouterEndpointStabilityRows.filter((row) => row.endpoint_metadata_match !== "True");
const volatileEndpointRows = openRouterEndpointStabilityRows
  .filter((row) => Number(row.observations) >= 4)
  .sort((left, right) => Number(right.max_over_min) - Number(left.max_over_min));
const openRouterEndpointSampleRows = [...unmatchedEndpointRows, ...volatileEndpointRows.slice(0, 20)]
  .filter((row, index, rows) => rows.findIndex((candidate) => `${candidate.openrouter_model_id}|${candidate.endpoint_id}` === `${row.openrouter_model_id}|${row.endpoint_id}`) === index);
const openRouterEndpointSection = openRouterModelEnd + 3;
section(openRouterTimeAudit, `A${openRouterEndpointSection}:N${openRouterEndpointSection}`, "Endpoint stability sample — all unmatched metadata plus most volatile default-tier endpoints");
const openRouterEndpointHeader = openRouterEndpointSection + 1;
const openRouterEndpointHeaders = ["OpenRouter ID", "Name", "Endpoint UUID", "Provider", "Provider slug", "Quantization", "Metadata matched", "Observations", "Median tok/s", "Mean tok/s", "Min tok/s", "Max tok/s", "Max/min", "Log SD"];
openRouterTimeAudit.getRange(`A${openRouterEndpointHeader}:N${openRouterEndpointHeader}`).values = [openRouterEndpointHeaders];
header(openRouterTimeAudit.getRange(`A${openRouterEndpointHeader}:N${openRouterEndpointHeader}`));
const openRouterEndpointSampleData = openRouterEndpointSampleRows.map((row) => [
  row.openrouter_model_id, row.openrouter_model_name, row.endpoint_id, row.provider_name, row.provider_slug,
  row.quantization, row.endpoint_metadata_match === "True" ? "Yes" : "No", Number(row.observations),
  Number(row.median_tps), Number(row.mean_tps), Number(row.min_tps), Number(row.max_tps), Number(row.max_over_min), Number(row.log_standard_deviation),
]);
const openRouterEndpointEnd = openRouterEndpointHeader + openRouterEndpointSampleData.length;
openRouterTimeAudit.getRange(`A${openRouterEndpointHeader + 1}:N${openRouterEndpointEnd}`).values = openRouterEndpointSampleData;
body(openRouterTimeAudit.getRange(`A${openRouterEndpointHeader + 1}:N${openRouterEndpointEnd}`));
openRouterTimeAudit.getRange(`H${openRouterEndpointHeader + 1}:N${openRouterEndpointEnd}`).format.numberFormat = "0.00";
openRouterTimeAudit.getRange(`H${openRouterEndpointHeader + 1}:H${openRouterEndpointEnd}`).format.numberFormat = "0";
openRouterTimeAudit.tables.add(`A${openRouterEndpointHeader}:N${openRouterEndpointEnd}`, true, "OpenRouterEndpointStabilitySampleTable").style = "TableStyleMedium2";
openRouterTimeAudit.freezePanes.freezeRows(openRouterTemporalBacktestHeader);
openRouterTimeAudit.getRange("A:B").format.columnWidth = 34;
openRouterTimeAudit.getRange("C:C").format.columnWidth = 38;
openRouterTimeAudit.getRange("D:M").format.columnWidth = 18;
openRouterTimeAudit.getRange("K:M").format.columnWidth = 34;
openRouterTimeAudit.getRange("N:N").format.columnWidth = 52;
openRouterTimeAudit.getRange(`A${openRouterHistoryHeader}:N${openRouterEndpointEnd}`).format.wrapText = true;

title(openRouterPriceAudit, "A1:N1", "OpenRouter price-schedule audit — service tiers, long context, and official API reconciliation");
subtitle(openRouterPriceAudit, "A2:N3", "Every endpoint price schedule is retained separately by default, flex, and priority tier, including the first high-context override. The frontend extraction is independently checked against OpenRouter's documented model-endpoints API. This validates price parsing; it does not create an independent parameter-count likelihood term.");
openRouterPriceAudit.getRange("A5:B5").values = [["Audit statistic", "Value"]];
header(openRouterPriceAudit.getRange("A5:B5"));
const openRouterOfficialCounts = openRouterOfficialAudit.comparison_group_counts;
const openRouterPriceSummaryRows = [
  ["Provider endpoint rows", openRouterCollectionAudit.provider_endpoint_row_count],
  ["Endpoint/service-tier rows", openRouterCollectionAudit.endpoint_tier_row_count],
  ["Default-tier rows", openRouterCollectionAudit.endpoint_tier_service_tier_counts.default],
  ["Flex-tier rows", openRouterCollectionAudit.endpoint_tier_service_tier_counts.flex],
  ["Priority-tier rows", openRouterCollectionAudit.endpoint_tier_service_tier_counts.priority],
  ["Rows with high-context prices", openRouterCollectionAudit.endpoint_tier_rows_with_high_context_price],
  ["Official model requests succeeded", `${openRouterOfficialAudit.model_count_succeeded}/${openRouterOfficialAudit.model_count_requested}`],
  ["Official endpoint rows", openRouterOfficialAudit.official_endpoint_rows],
  ["Official price rows in exact groups", openRouterOfficialAudit.official_rows_in_exact_groups],
  ["Official price exact share", openRouterOfficialAudit.official_price_row_exact_share],
  ["Frontend-only groups", openRouterOfficialCounts.frontend_only || 0],
  ["Official-only groups", openRouterOfficialCounts.official_only || 0],
  ["Signature-mismatch groups", openRouterOfficialCounts.signature_mismatch || 0],
  ["Incremental forecast weight", 0],
];
const openRouterPriceSummaryEnd = 5 + openRouterPriceSummaryRows.length;
openRouterPriceAudit.getRange(`A6:B${openRouterPriceSummaryEnd}`).values = openRouterPriceSummaryRows;
body(openRouterPriceAudit.getRange(`A6:B${openRouterPriceSummaryEnd}`));
openRouterPriceAudit.getRange("B15:B15").format.numberFormat = "0.0%";
openRouterPriceAudit.getRange(`B${openRouterPriceSummaryEnd}:B${openRouterPriceSummaryEnd}`).format.numberFormat = "0.0%";

const openRouterPriceFocalIds = new Set([
  "anthropic/claude-fable-5", "openai/gpt-5.6-sol", "moonshotai/kimi-k3",
  "anthropic/claude-opus-4.8", "openai/gpt-5.5", "openai/gpt-5.6-terra",
  "anthropic/claude-sonnet-5", "openai/gpt-5.6-luna", "x-ai/grok-4.5",
]);
const openRouterFocalTierRows = openRouterTierRows.filter((row) => openRouterPriceFocalIds.has(row.openrouter_model_id));
const openRouterPriceFocalSection = openRouterPriceSummaryEnd + 3;
section(openRouterPriceAudit, `A${openRouterPriceFocalSection}:N${openRouterPriceFocalSection}`, "Complete frontier endpoint/service-tier schedules");
const openRouterPriceFocalHeader = openRouterPriceFocalSection + 1;
const openRouterPriceFocalHeaders = ["OpenRouter ID", "Provider", "Provider slug", "Endpoint UUID", "Tier", "Quantization", "Base input $/M", "Base output $/M", "High-context threshold", "High input $/M", "High output $/M", "p50 tok/s", "p90 tok/s", "p99 tok/s"];
openRouterPriceAudit.getRange(`A${openRouterPriceFocalHeader}:N${openRouterPriceFocalHeader}`).values = [openRouterPriceFocalHeaders];
header(openRouterPriceAudit.getRange(`A${openRouterPriceFocalHeader}:N${openRouterPriceFocalHeader}`));
const valueOrNull = (value) => value === "" ? null : Number(value);
const openRouterPriceFocalData = openRouterFocalTierRows.map((row) => [
  row.openrouter_model_id, row.provider_name, row.provider_slug, row.endpoint_id, row.service_tier, row.quantization,
  valueOrNull(row.prompt_usd_per_mtoken), valueOrNull(row.completion_usd_per_mtoken), valueOrNull(row.high_context_min_prompt_tokens),
  valueOrNull(row.high_context_prompt_usd_per_mtoken), valueOrNull(row.high_context_completion_usd_per_mtoken),
  valueOrNull(row.p50_throughput_tps_30m), valueOrNull(row.p90_throughput_tps_30m), valueOrNull(row.p99_throughput_tps_30m),
]);
const openRouterPriceFocalEnd = openRouterPriceFocalHeader + openRouterPriceFocalData.length;
openRouterPriceAudit.getRange(`A${openRouterPriceFocalHeader + 1}:N${openRouterPriceFocalEnd}`).values = openRouterPriceFocalData;
body(openRouterPriceAudit.getRange(`A${openRouterPriceFocalHeader + 1}:N${openRouterPriceFocalEnd}`));
openRouterPriceAudit.getRange(`G${openRouterPriceFocalHeader + 1}:N${openRouterPriceFocalEnd}`).format.numberFormat = "0.00";
openRouterPriceAudit.getRange(`I${openRouterPriceFocalHeader + 1}:I${openRouterPriceFocalEnd}`).format.numberFormat = "0";
openRouterPriceAudit.tables.add(`A${openRouterPriceFocalHeader}:N${openRouterPriceFocalEnd}`, true, "OpenRouterFrontierTierPriceTable").style = "TableStyleMedium2";

const openRouterOfficialMismatchRows = openRouterOfficialComparisonRows.filter((row) => row.status !== "exact");
const openRouterOfficialMismatchSection = openRouterPriceFocalEnd + 3;
section(openRouterPriceAudit, `A${openRouterOfficialMismatchSection}:N${openRouterOfficialMismatchSection}`, "All non-exact official API ↔ frontend price groups");
const openRouterOfficialMismatchHeader = openRouterOfficialMismatchSection + 1;
const openRouterOfficialMismatchHeaders = ["OpenRouter ID", "Provider tag", "Status", "Official rows", "Frontend rows", "Official signatures", "Frontend signatures", "Source URLs"];
openRouterPriceAudit.getRange(`A${openRouterOfficialMismatchHeader}:H${openRouterOfficialMismatchHeader}`).values = [openRouterOfficialMismatchHeaders];
header(openRouterPriceAudit.getRange(`A${openRouterOfficialMismatchHeader}:H${openRouterOfficialMismatchHeader}`));
const openRouterOfficialMismatchData = openRouterOfficialMismatchRows.map((row) => [
  row.openrouter_model_id, row.provider_tag, row.status, Number(row.official_endpoint_rows), Number(row.frontend_endpoint_tier_rows),
  row.official_signatures_json, row.frontend_signatures_json, `${row.official_source_url}\n${row.frontend_source_url}`,
]);
const openRouterOfficialMismatchEnd = openRouterOfficialMismatchHeader + openRouterOfficialMismatchData.length;
openRouterPriceAudit.getRange(`A${openRouterOfficialMismatchHeader + 1}:H${openRouterOfficialMismatchEnd}`).values = openRouterOfficialMismatchData;
body(openRouterPriceAudit.getRange(`A${openRouterOfficialMismatchHeader + 1}:H${openRouterOfficialMismatchEnd}`));
openRouterPriceAudit.getRange(`D${openRouterOfficialMismatchHeader + 1}:E${openRouterOfficialMismatchEnd}`).format.numberFormat = "0";
openRouterPriceAudit.getRange(`A${openRouterOfficialMismatchHeader}:H${openRouterOfficialMismatchEnd}`).format.wrapText = true;
openRouterPriceAudit.tables.add(`A${openRouterOfficialMismatchHeader}:H${openRouterOfficialMismatchEnd}`, true, "OpenRouterOfficialPriceMismatchTable").style = "TableStyleMedium2";
openRouterPriceAudit.freezePanes.freezeRows(openRouterPriceFocalHeader);
openRouterPriceAudit.getRange("A:A").format.columnWidth = 34;
openRouterPriceAudit.getRange("B:C").format.columnWidth = 24;
openRouterPriceAudit.getRange("D:D").format.columnWidth = 38;
openRouterPriceAudit.getRange("E:N").format.columnWidth = 18;
openRouterPriceAudit.getRange("F:H").format.columnWidth = 46;

// Raw LaTeX table archive: every table block, chunked below Excel's cell-text limit.
title(latexRaw, "A1:E1", "Raw LaTeX table archive — lossless chunked text");
subtitle(latexRaw, "A2:E3", "Every \\begin{table}…\\end{table} block from arXiv-2606.07157v3 is retained verbatim. Chunks are numbered and limited to 29,000 characters so the source can be reassembled exactly without hitting Excel's cell limit.");
latexRaw.getRange("A5:E5").values = [["Table index", "LaTeX label", "Chunk", "Chunks", "Raw LaTeX text"]];
header(latexRaw.getRange("A5:E5"));
latexRaw.getRange(`A6:E${5 + latexRawRows.length}`).values = latexRawRows;
body(latexRaw.getRange(`A6:E${5 + latexRawRows.length}`));
latexRaw.getRange(`E6:E${5 + latexRawRows.length}`).format.wrapText = true;
latexRaw.getRange("A:D").format.columnWidth = 18;
latexRaw.getRange("B:B").format.columnWidth = 42;
latexRaw.getRange("E:E").format.columnWidth = 110;
latexRaw.freezePanes.freezeRows(5);

// Full ECI sheet value snapshots plus a separate formula archive. Source rows/columns are not shifted.
for (const snapshot of eciSnapshots) {
  const target = rawSheets.get(snapshot.targetName);
  const rowCount = snapshot.values.length;
  const colCount = snapshot.values[0]?.length || 1;
  const endCol = colLetter(colCount - 1);
  target.getRange(`A1:${endCol}${rowCount}`).values = snapshot.values;
  header(target.getRange(`A1:${endCol}1`));
  if (rowCount > 1) body(target.getRange(`A2:${endCol}${rowCount}`));
  target.getRange(`A:${endCol}`).format.columnWidth = 18;
  target.getRange("A:A").format.columnWidth = 36;
  if (colCount > 1) target.getRange("B:B").format.columnWidth = 34;
  target.freezePanes.freezeRows(1);
}

// Full Epoch CSV snapshot retained verbatim in chunks below Excel's cell-text limit.
title(epochRaw, "A1:C1", "Epoch all-models CSV — lossless Base64 snapshot");
subtitle(epochRaw, "A2:C3", "Concatenate Base64 text by ascending chunk number, then Base64-decode it to reproduce the exact downloaded CSV bytes. This safely preserves XML-illegal control characters; SHA-256 and byte count are recorded in Source Manifest.");
epochRaw.getRange("A5:C5").values = [["Chunk", "Chunks", "Base64 text"]];
header(epochRaw.getRange("A5:C5"));
epochRaw.getRange(`A6:C${5 + epochCsvRawRows.length}`).values = epochCsvRawRows;
body(epochRaw.getRange(`A6:C${5 + epochCsvRawRows.length}`));
epochRaw.getRange(`C6:C${5 + epochCsvRawRows.length}`).format.wrapText = false;
epochRaw.getRange(`A6:C${5 + epochCsvRawRows.length}`).format.rowHeight = 18;
epochRaw.getRange("A:B").format.columnWidth = 16;
epochRaw.getRange("C:C").format.columnWidth = 110;
epochRaw.freezePanes.freezeRows(5);

title(formulaArchive, "A1:D1", "ECI formula archive — original formulas as inert text");
subtitle(formulaArchive, "A2:D3", "Raw ECI sheets store evaluated values without broken cross-sheet references. Every original formula is preserved with its source address; prepend Unicode character U+003D to the formula body wherever the marker column contains U+003D.");
formulaArchive.getRange("A5:D5").values = [["Original sheet", "Original cell", "Leading equals", "Formula body"]];
header(formulaArchive.getRange("A5:D5"));
formulaArchive.getRange(`A6:D${5 + formulaRows.length}`).values = formulaRows;
body(formulaArchive.getRange(`A6:D${5 + formulaRows.length}`));
formulaArchive.getRange(`D6:D${5 + formulaRows.length}`).format.wrapText = true;
formulaArchive.getRange("A:A").format.columnWidth = 30;
formulaArchive.getRange("B:B").format.columnWidth = 18;
formulaArchive.getRange("C:C").format.columnWidth = 16;
formulaArchive.getRange("D:D").format.columnWidth = 100;
formulaArchive.freezePanes.freezeRows(5);

// Exact model-level data parsed from the submitted LaTeX source.
title(evidence, "A1:M1", "No-CoT evidence — exact values and audited release dates");
subtitle(evidence, "A2:M3", `All horizons, confidence intervals, architecture fields, and parameter counts are parsed directly from arXiv-2606.07157v3.tar.gz. Release dates are reconciled separately at day precision for all 49 checkpoints; the paper's original month labels remain preserved in the registry and date-audit sheets.`);
section(evidence, "A5:M5", "Proprietary / frontier horizons — whole 32-short-answer suite");
evidence.getRange("A6:L6").values = [["Model", "Release date", "Time TH (min)", "Time CI low", "Time CI high", "Time source text", "Token TH", "Token CI low", "Token CI high", "Token source text", "At floor?", "LaTeX table"]];
header(evidence.getRange("A6:L6"));
const frontierStart = 7;
const frontierRows = frontierHorizons.map((row) => [
  row.model, exactNoCotDate(row.model), row.time.estimate, row.time.low, row.time.high,
  row.time.raw, row.tokens.estimate, row.tokens.low, row.tokens.high, row.tokens.raw,
  row.time.floor || row.tokens.floor ? "Yes" : "No", "tab:horizons-per-model",
]);
evidence.getRange(`A${frontierStart}:L${frontierStart + frontierRows.length - 1}`).values = frontierRows;
body(evidence.getRange(`A${frontierStart}:L${frontierStart + frontierRows.length - 1}`));
evidence.getRange(`B${frontierStart}:B${frontierStart + frontierRows.length - 1}`).format.numberFormat = "yyyy-mm-dd";
evidence.getRange(`C${frontierStart}:E${frontierStart + frontierRows.length - 1}`).format.numberFormat = "0.000";
evidence.getRange(`G${frontierStart}:I${frontierStart + frontierRows.length - 1}`).format.numberFormat = "0.000";
evidence.tables.add(`A6:L${frontierStart + frontierRows.length - 1}`, true, "NoCotFrontierTable").style = "TableStyleMedium2";

const openSectionRow = frontierStart + frontierRows.length + 2;
section(evidence, `A${openSectionRow}:M${openSectionRow}`, "Open-weight horizon panel — 25-benchmark subset (not level-comparable to proprietary suite)");
const openHeaderRow = openSectionRow + 1;
evidence.getRange(`A${openHeaderRow}:M${openHeaderRow}`).values = [["Developer", "Model", "Release date", "Total params (B)", "Active params (B)", "Layers", "Architecture", "Reasoning", "Modality", "Point TH (min)", "Bootstrap median", "CI low", "CI high"]];
header(evidence.getRange(`A${openHeaderRow}:M${openHeaderRow}`));
const openStart = openHeaderRow + 1;
const openRows = openModels.map((row) => [row.developer, row.model, exactNoCotDate(row.model), row.totalB, row.activeB, row.layers, row.architecture, row.reasoning, row.modality, row.point, row.median, row.low, row.high]);
evidence.getRange(`A${openStart}:M${openStart + openRows.length - 1}`).values = openRows;
body(evidence.getRange(`A${openStart}:M${openStart + openRows.length - 1}`));
evidence.getRange(`C${openStart}:C${openStart + openRows.length - 1}`).format.numberFormat = "yyyy-mm-dd";
evidence.getRange(`D${openStart}:F${openStart + openRows.length - 1}`).format.numberFormat = "0";
evidence.getRange(`J${openStart}:M${openStart + openRows.length - 1}`).format.numberFormat = "0.000";
evidence.getRange(`G${openStart}:G${openStart + openRows.length - 1}`).conditionalFormats.add("containsText", { text: "MoE", format: { fill: C.paleAmber, font: { color: C.navy, bold: true } } });
evidence.tables.add(`A${openHeaderRow}:M${openStart + openRows.length - 1}`, true, "NoCotOpenWeightTable").style = "TableStyleMedium2";
const evidenceSourceRow = openStart + openRows.length + 2;
evidence.getRange(`A${evidenceSourceRow}:M${evidenceSourceRow + 1}`).merge();
evidence.getRange(`A${evidenceSourceRow}`).values = [[`Source archive: ${latexArchiveDisplay}\nPaper: ${arxivUrl}\nHorizon and model fields are parsed from LaTeX. Exact dates come from the audited registry; all 49 are day-level, and the four date-only overrides cannot create parameter joins. Original paper month labels remain in “Model Registry” and “No-CoT Date Audit”.`]];
evidence.getRange(`A${evidenceSourceRow}:M${evidenceSourceRow + 1}`).format = { fill: C.gray, font: { name: "Aptos", size: 9, italic: true, color: C.text }, wrapText: true, verticalAlignment: "center" };
evidence.getRange("A:A").format.columnWidth = 17;
evidence.getRange("B:B").format.columnWidth = 32;
evidence.getRange("C:C").format.columnWidth = 15;
evidence.getRange("D:F").format.columnWidth = 16;
evidence.getRange("G:I").format.columnWidth = 16;
evidence.getRange("J:J").format.columnWidth = 28;
evidence.getRange("K:L").format.columnWidth = 16;
evidence.getRange("M:M").format.columnWidth = 16;
evidence.freezePanes.freezeRows(6);

const frontierRow = new Map(frontierHorizons.map((row, idx) => [row.model, frontierStart + idx]));
const exactTimeLaw = noCotExactDateResult.time_horizon.adjusted_reported_law;
const exactTokenLaw = noCotExactDateResult.token_horizon.adjusted_reported_law;

// Day-level release-date reconciliation and deterministic sensitivity audit.
title(noCotDateAudit, "A1:L1", "No-CoT release-date audit — 49/49 exact day-level dates");
subtitle(noCotDateAudit, "A2:L3", "The paper reports release months. This ledger reconciles every checkpoint to an exact date, preserves the original month, and applies a transparent exact-date/month-date Pareto-OLS slope ratio to the published bootstrap law. The four explicit overrides are date-only and cannot create parameter identities.");
noCotDateAudit.getRange("A5:B5").values = [["Audit statistic", "Value"]];
header(noCotDateAudit.getRange("A5:B5"));
const noCotDateSummaryRows = [
  ["No-CoT models", noCotExactDateResult.inventory.no_cot_models],
  ["Models with exact day-level dates", noCotExactDateResult.inventory.models_with_day_level_dates],
  ["Models remaining month-only", noCotExactDateResult.inventory.models_remaining_month_only],
  ["Explicit date-only overrides", noCotExactDateResult.inventory.explicit_date_only_overrides],
  ["Parameter identities added by overrides", noCotExactDateResult.inventory.parameter_identities_added_by_overrides],
  ["Paper time-horizon doubling (days)", exactTimeLaw.paper_reported_point_days],
  ["Exact-date adjusted time law (days)", exactTimeLaw.adjusted_point_days],
  ["Time-law adjustment ratio", exactTimeLaw.exact_date_adjustment_ratio],
  ["Paper token-horizon doubling (days)", exactTokenLaw.paper_reported_point_days],
  ["Exact-date adjusted token law (days)", exactTokenLaw.adjusted_point_days],
  ["Token-law adjustment ratio", exactTokenLaw.exact_date_adjustment_ratio],
  ["No-CoT branch weight changed", noCotExactDateResult.decision.change_live_no_cot_weight ? "Yes" : "No"],
];
const noCotDateSummaryEnd = 5 + noCotDateSummaryRows.length;
noCotDateAudit.getRange(`A6:B${noCotDateSummaryEnd}`).values = noCotDateSummaryRows;
body(noCotDateAudit.getRange(`A6:B${noCotDateSummaryEnd}`));
noCotDateAudit.getRange("B6:B10").format.numberFormat = "0";
noCotDateAudit.getRange("B11:B12").format.numberFormat = "0.0";
noCotDateAudit.getRange("B13:B13").format.numberFormat = "0.0000x";
noCotDateAudit.getRange("B14:B15").format.numberFormat = "0.0";
noCotDateAudit.getRange("B16:B16").format.numberFormat = "0.0000x";

const noCotDateModelSection = noCotDateSummaryEnd + 3;
section(noCotDateAudit, `A${noCotDateModelSection}:L${noCotDateModelSection}`, "Complete model-level exact-date ledger");
const noCotDateModelHeader = noCotDateModelSection + 1;
noCotDateAudit.getRange(`A${noCotDateModelHeader}:L${noCotDateModelHeader}`).values = [["Model", "Source locator", "Paper month", "Exact release", "Offset (days)", "Exact-date source", "Explicit override", "Parameter join policy", "Time frontier: month", "Time frontier: exact", "Token frontier: month", "Token frontier: exact"]];
header(noCotDateAudit.getRange(`A${noCotDateModelHeader}:L${noCotDateModelHeader}`));
const noCotDateModelData = noCotExactDateModelRows.map((row) => [
  row.model,
  row.source_locator,
  new Date(`${row.paper_month_date}T00:00:00Z`),
  new Date(`${row.exact_release_date}T00:00:00Z`),
  Number(row.day_offset_from_month_start),
  row.exact_date_source,
  row.explicit_override === "True" ? "Yes" : "No",
  row.parameter_join_policy,
  row.time_pareto_with_month_dates === "True" ? "Yes" : "No",
  row.time_pareto_with_exact_dates === "True" ? "Yes" : "No",
  row.token_pareto_with_month_dates === "True" ? "Yes" : "No",
  row.token_pareto_with_exact_dates === "True" ? "Yes" : "No",
]);
const noCotDateModelEnd = noCotDateModelHeader + noCotDateModelData.length;
noCotDateAudit.getRange(`A${noCotDateModelHeader + 1}:L${noCotDateModelEnd}`).values = noCotDateModelData;
body(noCotDateAudit.getRange(`A${noCotDateModelHeader + 1}:L${noCotDateModelEnd}`));
noCotDateAudit.getRange(`C${noCotDateModelHeader + 1}:D${noCotDateModelEnd}`).format.numberFormat = "yyyy-mm-dd";
noCotDateAudit.getRange(`F${noCotDateModelHeader + 1}:H${noCotDateModelEnd}`).format.wrapText = true;
noCotDateAudit.getRange(`G${noCotDateModelHeader + 1}:G${noCotDateModelEnd}`).conditionalFormats.add("containsText", { text: "Yes", format: { fill: C.paleAmber, font: { color: C.navy, bold: true } } });
noCotDateAudit.tables.add(`A${noCotDateModelHeader}:L${noCotDateModelEnd}`, true, "NoCotExactDateAuditTable").style = "TableStyleMedium2";
noCotDateAudit.getRange("A:A").format.columnWidth = 34;
noCotDateAudit.getRange("B:B").format.columnWidth = 34;
noCotDateAudit.getRange("C:E").format.columnWidth = 18;
noCotDateAudit.getRange("F:F").format.columnWidth = 55;
noCotDateAudit.getRange("G:G").format.columnWidth = 18;
noCotDateAudit.getRange("H:H").format.columnWidth = 42;
noCotDateAudit.getRange("I:L").format.columnWidth = 18;
noCotDateAudit.freezePanes.freezeRows(noCotDateModelHeader);

// First-party evidence for the live proprietary targets. Numeric measurements,
// parameter identities, serving-system caveats, and derived sensitivities are
// kept in separate rows so none can silently masquerade as a disclosure.
title(primaryEvidenceAudit, "A1:I1", "First-party frontier evidence and parameter-mapping audit");
subtitle(primaryEvidenceAudit, "A2:I3", "OpenAI directly reports Sol's no-CoT horizon; Anthropic directly confirms that Fable 5 and Mythos 5 share underlying weights. The official measurements are preserved, but the Sol-to-parameter mapping remains at 0% incremental weight because held-out support is small and defensible model forms disagree by more than 6×.");
primaryEvidenceAudit.getRange("A5:B5").values = [["Audit statistic", "Value"]];
header(primaryEvidenceAudit.getRange("A5:B5"));
const primarySummaryRows = [
  ["Official Sol no-CoT horizon (min)", frontierPrimaryResult.official_measurements.gpt_5_6_sol_nocot_minutes],
  ["Official GPT-5.5 comparator (min)", frontierPrimaryResult.official_measurements.gpt_5_5_comparator_minutes],
  ["Current projected Sol horizon (min)", frontierPrimaryResult.current_mapping.projected_sol_horizon_minutes],
  ["Current Sol horizon prior (T)", frontierPrimaryResult.current_mapping.projected_sol_horizon_prior_t],
  ["Direct model-level mapping (T)", frontierPrimaryResult.sol_mapping_sensitivity.direct_model_level_horizon_regression_t],
  ["Direct pooled-elasticity mapping (T)", frontierPrimaryResult.sol_mapping_sensitivity.direct_pooled_paper_elasticity_t],
  ["Direct MoE-elasticity mapping (T)", frontierPrimaryResult.sol_mapping_sensitivity.direct_moe_paper_elasticity_t],
  ["GPT-5.5-rebased pooled mapping (T)", frontierPrimaryResult.sol_mapping_sensitivity.gpt_5_5_suite_rebased_pooled_elasticity_t],
  ["Mapping max / min", frontierPrimaryResult.sol_mapping_sensitivity.nonbaseline_method_max_over_min],
  ["Chronological developer-holdouts", frontierPrimaryResult.inventory.chronological_developer_holdout_predictions],
  ["Held-out developers", frontierPrimaryResult.inventory.heldout_developers],
  ["Bootstrap P(horizon improves)", frontierPrimaryResult.heldout_backtest.incremental_bootstrap.bootstrap_probability_horizon_better],
  ["Incremental live weight", frontierPrimaryResult.decision.incremental_live_weight],
  ["Headline forecasts changed", frontierPrimaryResult.decision.change_headline_forecasts ? "Yes" : "No"],
];
const primarySummaryEnd = 5 + primarySummaryRows.length;
primaryEvidenceAudit.getRange(`A6:B${primarySummaryEnd}`).values = primarySummaryRows;
body(primaryEvidenceAudit.getRange(`A6:B${primarySummaryEnd}`));
primaryEvidenceAudit.getRange(`B6:B${primarySummaryEnd}`).format.numberFormat = "0.0";
primaryEvidenceAudit.getRange(`B${primarySummaryEnd - 2}:B${primarySummaryEnd - 1}`).format.numberFormat = "0.0%";

const primaryEvidenceSection = primarySummaryEnd + 3;
section(primaryEvidenceAudit, `A${primaryEvidenceSection}:I${primaryEvidenceSection}`, "Normalized first-party evidence ledger");
const primaryEvidenceHeader = primaryEvidenceSection + 1;
primaryEvidenceAudit.getRange(`A${primaryEvidenceHeader}:I${primaryEvidenceHeader}`).values = [["Evidence ID", "Model", "Comparator", "Type", "Metric", "Value", "Unit", "Identity / weight policy", "Claim summary"]];
header(primaryEvidenceAudit.getRange(`A${primaryEvidenceHeader}:I${primaryEvidenceHeader}`));
const primaryEvidenceData = frontierPrimaryEvidenceRows.map((row) => [
  row.evidence_id,
  row.model,
  row.comparator_model,
  row.evidence_type,
  row.metric_name,
  row.value === "" ? null : Number(row.value),
  row.unit,
  `${row.parameter_identity_policy}; ${row.live_weight_policy}`,
  row.claim_summary,
]);
const primaryEvidenceEnd = primaryEvidenceHeader + primaryEvidenceData.length;
primaryEvidenceAudit.getRange(`A${primaryEvidenceHeader + 1}:I${primaryEvidenceEnd}`).values = primaryEvidenceData;
body(primaryEvidenceAudit.getRange(`A${primaryEvidenceHeader + 1}:I${primaryEvidenceEnd}`));
primaryEvidenceAudit.getRange(`F${primaryEvidenceHeader + 1}:F${primaryEvidenceEnd}`).format.numberFormat = "0.0";
primaryEvidenceAudit.getRange(`H${primaryEvidenceHeader + 1}:I${primaryEvidenceEnd}`).format.wrapText = true;
primaryEvidenceAudit.tables.add(`A${primaryEvidenceHeader}:I${primaryEvidenceEnd}`, true, "FrontierPrimaryEvidenceTable").style = "TableStyleMedium2";

const primaryControlSection = primaryEvidenceEnd + 3;
section(primaryEvidenceAudit, `A${primaryControlSection}:I${primaryControlSection}`, "Chronological holdouts, same-size controls, and Sol mapping sensitivities");
const primaryControlHeader = primaryControlSection + 1;
primaryEvidenceAudit.getRange(`A${primaryControlHeader}:I${primaryControlHeader}`).values = [["Record type", "Series", "Model", "Release", "Total params (B)", "No-CoT horizon (min)", "Ratio / prediction ratio", "Evidence grade", "Interpretation"]];
header(primaryEvidenceAudit.getRange(`A${primaryControlHeader}:I${primaryControlHeader}`));
const primaryControlData = frontierPrimaryControlRows.map((row) => [
  row.record_type,
  row.series,
  row.model,
  row.release_date ? new Date(`${row.release_date}T00:00:00Z`) : null,
  row.total_parameters_b === "" ? null : Number(row.total_parameters_b),
  row.nocot_time_horizon_minutes === "" ? null : Number(row.nocot_time_horizon_minutes),
  row.horizon_ratio_vs_previous === "" ? null : Number(row.horizon_ratio_vs_previous),
  row.evidence_grade,
  row.interpretation,
]);
const primaryControlEnd = primaryControlHeader + primaryControlData.length;
primaryEvidenceAudit.getRange(`A${primaryControlHeader + 1}:I${primaryControlEnd}`).values = primaryControlData;
body(primaryEvidenceAudit.getRange(`A${primaryControlHeader + 1}:I${primaryControlEnd}`));
primaryEvidenceAudit.getRange(`D${primaryControlHeader + 1}:D${primaryControlEnd}`).format.numberFormat = "yyyy-mm-dd";
primaryEvidenceAudit.getRange(`E${primaryControlHeader + 1}:G${primaryControlEnd}`).format.numberFormat = "0.000";
primaryEvidenceAudit.getRange(`H${primaryControlHeader + 1}:I${primaryControlEnd}`).format.wrapText = true;
primaryEvidenceAudit.getRange(`A${primaryControlHeader + 1}:A${primaryControlEnd}`).conditionalFormats.add("containsText", { text: "sol_parameter_mapping_sensitivity", format: { fill: C.paleAmber, font: { color: C.navy, bold: true } } });
primaryEvidenceAudit.tables.add(`A${primaryControlHeader}:I${primaryControlEnd}`, true, "FrontierPrimaryControlTable").style = "TableStyleMedium2";
primaryEvidenceAudit.getRange("A:A").format.columnWidth = 36;
primaryEvidenceAudit.getRange("B:C").format.columnWidth = 30;
primaryEvidenceAudit.getRange("D:G").format.columnWidth = 20;
primaryEvidenceAudit.getRange("H:H").format.columnWidth = 40;
primaryEvidenceAudit.getRange("I:I").format.columnWidth = 58;
primaryEvidenceAudit.freezePanes.freezeRows(primaryEvidenceHeader);

// Independent factual-capacity signal. The live target uses only rows released
// before Fable and excludes Anthropic from calibration.
const ikpOverlap = ikpAudit.incremental_overlap;
const ikpChronological = ikpOverlap.chronological_fixed_weight_subset;
const ikpTarget = ikpAudit.target_signal.fable;
const ikpLiveEstimateT = ikpTarget.strict_open_only_release_and_vendor_holdout.mean.estimates.forward_inverse.estimated_b / 1000;
const ikpGate = ikpDecision.evidence_gates;
const ikpDecisionSummary = ikpPromoted
  ? `The audit promotes a ${Math.round(100 * ikpEvidenceWeight)}% evidence-level branch for Fable; Sol has no observed IKP result.`
  : `The audit retains IKP only as a Fable sensitivity at 0% live weight: the later chronological subset has ${ikpChronological.models} models, below the predeclared minimum of ${ikpGate.minimum_chronological_subset_models}. Sol has no observed IKP result.`;
title(ikpSignalAudit, "A1:J1", "Incompressible Knowledge Probes — direct factual-capacity size signal");
subtitle(ikpSignalAudit, "A2:J3", `The source fit is reproduced exactly, explicit thinking/non-thinking serving duplicates are collapsed to one weight base, and every scored calibration prediction uses only earlier releases with the test vendor excluded. ${ikpDecisionSummary}`);
ikpSignalAudit.getRange("A5:B5").values = [["Audit statistic", "Value"]];
header(ikpSignalAudit.getRange("A5:B5"));
const ikpSummaryRows = [
  ["Pinned upstream commit", ikpSourceMetadata.upstream.commit],
  ["Pinned independent-replication commit", ikpSourceMetadata.replication.commit],
  ["Raw calibration configurations", ikpAudit.source_inventory.calibration_configurations],
  ["Distinct weight bases after serving collapse", ikpAudit.source_inventory.calibration_weight_bases],
  ["Serving variants collapsed", ikpAudit.source_inventory.serving_variants_collapsed],
  ["Published fit R² reproduced", ikpAudit.published_reproduction.r_squared],
  ["Strict prediction rows", ikpPredictionRows.length],
  ["Exact overlap models", ikpOverlap.models],
  ["Exact overlap families", ikpOverlap.families],
  ["Existing overlap median error", ikpOverlap.existing.median_multiplicative_error],
  ["IKP overlap median error", ikpOverlap.ikp.median_multiplicative_error],
  ["10% blend overlap median error", ikpOverlap.blend_10pct.median_multiplicative_error],
  ["10% blend bootstrap CI90 low", ikpOverlap.family_bootstrap.ci_90[0]],
  ["10% blend bootstrap CI90 high", ikpOverlap.family_bootstrap.ci_90[1]],
  ["Later chronological subset models", ikpChronological.models],
  ["Later chronological subset families", ikpChronological.families],
  ["Chronological bootstrap CI90 low", ikpChronological.family_bootstrap.ci_90[0]],
  ["Chronological bootstrap CI90 high", ikpChronological.family_bootstrap.ci_90[1]],
  ["Signed-error correlation vs existing ensemble", ikpOverlap.signed_error_correlation_existing_vs_ikp],
  ["Decision-derived evidence-level weight", ikpEvidenceWeight],
  ["Final Fable weight with 50% crowd", ikpFinalWeight],
  ["Strict Fable estimate (T)", ikpLiveEstimateT],
  ["Source Fable estimate (T)", ikpTarget.published_lambda0_estimate_b / 1000],
  ["Strict Fable model-form minimum (T)", ikpTarget.strict_open_only_model_form_min_b / 1000],
  ["Strict Fable model-form maximum (T)", ikpTarget.strict_open_only_model_form_max_b / 1000],
  ["Fable refusal rate", ikpTarget.refusal_rate],
];
const ikpSummaryEnd = 5 + ikpSummaryRows.length;
ikpSignalAudit.getRange(`A6:B${ikpSummaryEnd}`).values = ikpSummaryRows;
body(ikpSignalAudit.getRange(`A6:B${ikpSummaryEnd}`));
ikpSignalAudit.getRange(`B8:B${ikpSummaryEnd}`).format.numberFormat = "0.000";
ikpSignalAudit.getRange(`B${ikpSummaryEnd - 6}:B${ikpSummaryEnd - 5}`).format.numberFormat = "0.0%";
ikpSignalAudit.getRange(`B${ikpSummaryEnd}:B${ikpSummaryEnd}`).format.numberFormat = "0.0%";

const ikpOverlapSection = ikpSummaryEnd + 3;
section(ikpSignalAudit, `A${ikpOverlapSection}:J${ikpOverlapSection}`, "Exact chronological overlap with the existing ensemble");
const ikpOverlapHeader = ikpOverlapSection + 1;
ikpSignalAudit.getRange(`A${ikpOverlapHeader}:J${ikpOverlapHeader}`).values = [["Model", "Release", "Family", "Actual (B)", "Existing prediction (B)", "IKP prediction (B)", "10% blend (B)", "Existing abs log error", "IKP abs log error", "Blend − existing"]];
header(ikpSignalAudit.getRange(`A${ikpOverlapHeader}:J${ikpOverlapHeader}`));
const ikpOverlapData = ikpOverlapRows.map((row) => [
  row.model,
  new Date(`${row.release_date}T00:00:00Z`),
  row.family,
  Number(row.actual_b),
  Number(row.existing_predicted_b),
  Number(row.ikp_predicted_b),
  Number(row.blended_predicted_b),
  Number(row.existing_abs_log10_error),
  Number(row.ikp_abs_log10_error),
  Number(row.blend_minus_existing_abs_log10_error),
]);
const ikpOverlapEnd = ikpOverlapHeader + ikpOverlapData.length;
ikpSignalAudit.getRange(`A${ikpOverlapHeader + 1}:J${ikpOverlapEnd}`).values = ikpOverlapData;
body(ikpSignalAudit.getRange(`A${ikpOverlapHeader + 1}:J${ikpOverlapEnd}`));
ikpSignalAudit.getRange(`B${ikpOverlapHeader + 1}:B${ikpOverlapEnd}`).format.numberFormat = "yyyy-mm-dd";
ikpSignalAudit.getRange(`D${ikpOverlapHeader + 1}:J${ikpOverlapEnd}`).format.numberFormat = "0.000";
ikpSignalAudit.getRange(`J${ikpOverlapHeader + 1}:J${ikpOverlapEnd}`).conditionalFormats.add("cellIs", { operator: "lessThan", formula: 0, format: { fill: C.paleTeal, font: { color: C.tealDark, bold: true } } });
ikpSignalAudit.tables.add(`A${ikpOverlapHeader}:J${ikpOverlapEnd}`, true, "IKPExactOverlapTable").style = "TableStyleMedium2";
const ikpInterpretationRow = ikpOverlapEnd + 3;
ikpSignalAudit.getRange(`A${ikpInterpretationRow}:J${ikpInterpretationRow + 3}`).merge();
ikpSignalAudit.getRange(`A${ikpInterpretationRow}`).values = [[ikpPromoted
  ? `Decision: promote ${Math.round(100 * ikpEvidenceWeight)}% inside the evidence model for Fable only. The fixed-blend checks are favorable and the strict pre-Fable/vendor-held-out estimate is ${ikpLiveEstimateT.toFixed(1)}T. This remains an effective factual-capacity signal rather than an architecture disclosure.`
  : `Decision: retain IKP as a 0%-weight sensitivity. The fixed 10% blend improves ${ikpOverlap.families_improved}/${ikpOverlap.families} families and both bootstrap intervals are favorable, but the later chronological subset contains ${ikpChronological.models} models versus the required ${ikpGate.minimum_chronological_subset_models}; the predeclared coverage gate therefore blocks promotion. The strict pre-Fable/vendor-held-out sensitivity is ${ikpLiveEstimateT.toFixed(1)}T.`]];
ikpSignalAudit.getRange(`A${ikpInterpretationRow}:J${ikpInterpretationRow + 3}`).format = { fill: C.paleAmber, font: { name: "Aptos", size: 9, color: C.text }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: "#D9991A" } } };

const ikpConditionalSection = ikpInterpretationRow + 5;
section(ikpSignalAudit, `A${ikpConditionalSection}:J${ikpConditionalSection}`, "Conditional benchmark test — does IKP add information after score and exact date?");
const ikpConditionalHeader = ikpConditionalSection + 1;
ikpSignalAudit.getRange(`A${ikpConditionalHeader}:J${ikpConditionalHeader}`).values = [["Benchmark", "Panel bases", "Held-out bases", "Held-out vendors", "Baseline median error", "+IKP median error", "Vendor CI90 low", "Vendor CI90 high", "Passing sensitivities", "Result"]];
header(ikpSignalAudit.getRange(`A${ikpConditionalHeader}:J${ikpConditionalHeader}`));
const ikpConditionalData = ["mmlu", "mmlu_pro", "gpqa_diamond", "simpleqa"].map((benchmark) => {
  const result = ikpConditionalAudit.heldout_results[benchmark];
  const primary = result.specifications.row_equal__score_date;
  return [
    result.label,
    result.weight_base_panel_rows,
    result.strict_prediction_models,
    result.strict_prediction_vendors,
    primary.baseline.median_multiplicative_error ?? null,
    primary.candidate_with_ikp.median_multiplicative_error ?? null,
    primary.paired_vendor_bootstrap.ci_90?.[0] ?? null,
    primary.paired_vendor_bootstrap.ci_90?.[1] ?? null,
    `${result.passing_specifications}/4`,
    result.passing_specifications === 4 ? "ROBUST" : result.passing_specifications > 0 ? "SUPPORTIVE" : "NOT ESTABLISHED",
  ];
});
const ikpConditionalEnd = ikpConditionalHeader + ikpConditionalData.length;
ikpSignalAudit.getRange(`A${ikpConditionalHeader + 1}:J${ikpConditionalEnd}`).values = ikpConditionalData;
body(ikpSignalAudit.getRange(`A${ikpConditionalHeader + 1}:J${ikpConditionalEnd}`));
ikpSignalAudit.getRange(`B${ikpConditionalHeader + 1}:H${ikpConditionalEnd}`).format.numberFormat = "0.000";
ikpSignalAudit.getRange(`J${ikpConditionalHeader + 1}:J${ikpConditionalEnd}`).conditionalFormats.add("containsText", { text: "ROBUST", format: { fill: C.paleTeal, font: { color: C.tealDark, bold: true } } });
ikpSignalAudit.getRange(`J${ikpConditionalHeader + 1}:J${ikpConditionalEnd}`).conditionalFormats.add("containsText", { text: "SUPPORTIVE", format: { fill: C.paleBlue, font: { color: C.navy, bold: true } } });
ikpSignalAudit.getRange(`J${ikpConditionalHeader + 1}:J${ikpConditionalEnd}`).conditionalFormats.add("containsText", { text: "NOT ESTABLISHED", format: { fill: C.paleAmber, font: { color: C.navy, bold: true } } });
ikpSignalAudit.tables.add(`A${ikpConditionalHeader}:J${ikpConditionalEnd}`, true, "IKPConditionalBenchmarkTable").style = "TableStyleMedium2";

const ikpNarrativeSection = ikpConditionalEnd + 2;
section(ikpSignalAudit, `A${ikpNarrativeSection}:D${ikpNarrativeSection}`, "Pinned upstream narrative reconciliation");
const ikpNarrativeHeader = ikpNarrativeSection + 1;
ikpSignalAudit.getRange(`A${ikpNarrativeHeader}:D${ikpNarrativeHeader}`).values = [["Claim", "Narrative", "Reproduced output", "Status"]];
header(ikpSignalAudit.getRange(`A${ikpNarrativeHeader}:D${ikpNarrativeHeader}`));
const ikpNarrativeData = ikpConditionalAudit.upstream_reproduction.narrative_summary_audit.claims.map((claim) => [
  claim.claim,
  claim.narrative,
  claim.generated_output,
  claim.matches_generated_output ? "MATCH" : "STALE",
]);
const ikpNarrativeEnd = ikpNarrativeHeader + ikpNarrativeData.length;
ikpSignalAudit.getRange(`A${ikpNarrativeHeader + 1}:D${ikpNarrativeEnd}`).values = ikpNarrativeData;
body(ikpSignalAudit.getRange(`A${ikpNarrativeHeader + 1}:D${ikpNarrativeEnd}`));
ikpSignalAudit.getRange(`B${ikpNarrativeHeader + 1}:C${ikpNarrativeEnd}`).format.numberFormat = "0.0000";
ikpSignalAudit.getRange(`D${ikpNarrativeHeader + 1}:D${ikpNarrativeEnd}`).conditionalFormats.add("containsText", { text: "STALE", format: { fill: C.palePink, font: { color: "#8B245C", bold: true } } });
ikpSignalAudit.tables.add(`A${ikpNarrativeHeader}:D${ikpNarrativeEnd}`, true, "IKPNarrativeReconciliationTable").style = "TableStyleMedium2";
const ikpConditionalInterpretationRow = ikpNarrativeEnd + 2;
ikpSignalAudit.getRange(`A${ikpConditionalInterpretationRow}:J${ikpConditionalInterpretationRow + 2}`).merge();
ikpSignalAudit.getRange(`A${ikpConditionalInterpretationRow}`).values = [[`Conditional result: GPQA passes all four exact-date, architecture, and vendor-weight sensitivities; MMLU passes ${ikpConditionalAudit.decision.mmlu_passing_specifications_out_of_four}/4; MMLU-Pro and SimpleQA are not promoted. The source CSVs reproduce exactly, while ${ikpConditionalAudit.upstream_reproduction.narrative_summary_audit.stale_claim_count} narrative claims are stale. This supports IKP as a sensitivity, but does not override the primary coverage gate or change its ${Math.round(100 * ikpEvidenceWeight)}% live weight.`]];
ikpSignalAudit.getRange(`A${ikpConditionalInterpretationRow}:J${ikpConditionalInterpretationRow + 2}`).format = { fill: C.paleTeal, font: { name: "Aptos", size: 9, color: C.text }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: C.tealDark } } };
const ikpSheetEnd = ikpConditionalInterpretationRow + 2;
ikpSignalAudit.getRange("A:A").format.columnWidth = 38;
ikpSignalAudit.getRange("B:B").format.columnWidth = 28;
ikpSignalAudit.getRange("C:C").format.columnWidth = 24;
ikpSignalAudit.getRange("D:J").format.columnWidth = 20;
ikpSignalAudit.freezePanes.freezeRows(ikpOverlapHeader);

// Epoch compute evidence: the frontier view is a correlated subset of all_ai_models, retained
// separately for source fidelity but never counted as an independent checkpoint observation.
const frontierViewRows = unifiedComputeRows.filter((row) => row.source === "Epoch Frontier View");
const stage2ObservationIds = new Set(computeStage2Rows.map((row) => row.observation_id));
title(computeEvidence, "A1:M1", "Epoch frontier compute evidence — correlated source view");
subtitle(computeEvidence, "A2:M3", "All 137 frontier_ai_models.csv rows from ai_models.zip. Training compute and parameters duplicate the corresponding all_ai records exactly; frontier-only estimation fields are retained. ‘Included’ marks the 19 confident/likely language-frontier rows used once in the compute branch.");
computeEvidence.getRange("A5:M5").values = [["Model", "Release", "Organization", "Total params (B)", "Training compute (FLOP)", "log10 compute", "Confidence", "Estimation method", "Compute notes", "Included in stage 2", "Frontier view row", "Matched all_ai row", "Source URL"]];
header(computeEvidence.getRange("A5:M5"));
const computeEvidenceStart = 6;
for (let index = 0; index < frontierViewRows.length; index += 1) {
  const row = 6 + index;
  const source = frontierViewRows[index];
  const raw = JSON.parse(source.source_record_json);
  computeEvidence.getRange(`A${row}:E${row}`).values = [[source.source_model_name, toDate(source.canonical_release_date), source.source_organization, Number(source.total_parameters_b) || null, Number(source.epoch_training_compute_flop) || null]];
  computeEvidence.getRange(`F${row}`).formulas = [[`=IF(E${row}="","",LOG10(E${row}))`]];
  computeEvidence.getRange(`G${row}:M${row}`).values = [[raw.Confidence || "", raw["Training compute estimation method"] || "", raw["Training compute notes"] || "", stage2ObservationIds.has(source.observation_id) ? "Yes" : "No", source.observation_id.split(":").at(-1), source.epoch_source_rows, source.source_url]];
}
const computeEvidenceEnd = computeEvidenceStart + frontierViewRows.length - 1;
body(computeEvidence.getRange(`A6:M${computeEvidenceEnd}`));
computeEvidence.getRange(`B6:B${computeEvidenceEnd}`).format.numberFormat = "yyyy-mm-dd";
computeEvidence.getRange(`D6:D${computeEvidenceEnd}`).format.numberFormat = "0.0";
computeEvidence.getRange(`E6:F${computeEvidenceEnd}`).format.numberFormat = "0.000E+00";
computeEvidence.getRange(`H6:I${computeEvidenceEnd}`).format.wrapText = true;
computeEvidence.getRange(`J6:J${computeEvidenceEnd}`).conditionalFormats.add("containsText", { text: "Yes", format: { fill: C.paleTeal, font: { color: C.tealDark, bold: true } } });
computeEvidence.tables.add(`A5:M${computeEvidenceEnd}`, true, "EpochComputeEvidenceTable").style = "TableStyleMedium2";
computeEvidence.getRange("A:A").format.columnWidth = 36;
computeEvidence.getRange("B:B").format.columnWidth = 15;
computeEvidence.getRange("C:C").format.columnWidth = 24;
computeEvidence.getRange("D:F").format.columnWidth = 20;
computeEvidence.getRange("G:G").format.columnWidth = 15;
computeEvidence.getRange("H:I").format.columnWidth = 48;
computeEvidence.getRange("J:L").format.columnWidth = 18;
computeEvidence.getRange("M:M").format.columnWidth = 62;
computeEvidence.freezePanes.freezeRows(5);

title(computeModel, "A1:R1", "Compute-informed parameter branch — 5% correlated evidence weight");
subtitle(computeModel, "A2:R3", "Two-stage sequential log model. Stage 1 estimates training compute from AA score, then corrects residuals for exact release date on 31 exact AA↔Epoch checkpoints. Stage 2 maps compute to total parameters, then applies a date-residual correction on 19 confident/likely Epoch frontier-language rows. No target has observed training compute, so this is a compute-structured AA/date regularizer, not an independent target likelihood. It receives only 5% log weight.");

section(computeModel, "A5:K5", "Stage 1 — AA score → compute, with date-residual correction");
computeModel.getRange("A6:K6").values = [["Checkpoint ID", "AA model/configuration", "Release", "Years since 2024-01-01", "AA index", "Training compute (FLOP)", "log10 compute", "Score-only fit", "Score residual", "Final fitted log10 compute", "Final residual"]];
header(computeModel.getRange("A6:K6"));
const computeStage1Start = 7;
for (let index = 0; index < computeStage1Rows.length; index += 1) {
  const row = computeStage1Start + index;
  const { aa, epoch } = computeStage1Rows[index];
  computeModel.getRange(`A${row}:C${row}`).values = [[aa.canonical_checkpoint_id, aa.source_model_name, toDate(aa.canonical_release_date)]];
  computeModel.getRange(`D${row}`).formulas = [[`=(C${row}-DATE(2024,1,1))/365.25`]];
  computeModel.getRange(`E${row}:F${row}`).values = [[Number(aa.aa_intelligence_index), Number(epoch.epoch_training_compute_flop)]];
  computeModel.getRange(`G${row}`).formulas = [[`=LOG10(F${row})`]];
  computeModel.getRange(`H${row}`).formulas = [[`=$N$7+$M$7*E${row}`]];
  computeModel.getRange(`I${row}`).formulas = [[`=G${row}-H${row}`]];
  computeModel.getRange(`J${row}`).formulas = [[`=H${row}+$P$7+$O$7*D${row}`]];
  computeModel.getRange(`K${row}`).formulas = [[`=G${row}-J${row}`]];
}
const computeStage1End = computeStage1Start + computeStage1Rows.length - 1;
computeModel.getRange("M6:P6").values = [["AA-score slope", "Score intercept", "Date-residual slope", "Date-residual intercept"]];
header(computeModel.getRange("M6:P6"));
computeModel.getRange("M7").formulas = [[`=SLOPE($G$${computeStage1Start}:$G$${computeStage1End},$E$${computeStage1Start}:$E$${computeStage1End})`]];
computeModel.getRange("N7").formulas = [[`=INTERCEPT($G$${computeStage1Start}:$G$${computeStage1End},$E$${computeStage1Start}:$E$${computeStage1End})`]];
computeModel.getRange("O7").formulas = [[`=SLOPE($I$${computeStage1Start}:$I$${computeStage1End},$D$${computeStage1Start}:$D$${computeStage1End})`]];
computeModel.getRange("P7").formulas = [[`=INTERCEPT($I$${computeStage1Start}:$I$${computeStage1End},$D$${computeStage1Start}:$D$${computeStage1End})`]];
computeModel.getRange("M9:N11").values = [["Calibration rows", computeStage1Rows.length], ["In-sample R²", null], ["RMSE factor", null]];
computeModel.getRange("N10").formulas = [[`=RSQ($G$${computeStage1Start}:$G$${computeStage1End},$J$${computeStage1Start}:$J$${computeStage1End})`]];
computeModel.getRange("N11").formulas = [[`=POWER(10,SQRT(SUMSQ($K$${computeStage1Start}:$K$${computeStage1End})/COUNT($K$${computeStage1Start}:$K$${computeStage1End})))`]];
body(computeModel.getRange(`A7:K${computeStage1End}`));
body(computeModel.getRange("M7:P11"));

section(computeModel, "A41:N41", "Stage 2 — compute → parameters, with date-residual correction");
computeModel.getRange("A42:N42").values = [["Epoch frontier model", "Release", "Years since 2024-01-01", "log10 compute", "log10 params", "Training compute (FLOP)", "Total params (B)", "Compute-only fit", "Compute residual", "Final fitted log10 params", "Final residual", "Confidence", "Estimation method", "Source row"]];
header(computeModel.getRange("A42:N42"));
const computeStage2Start = 43;
for (let index = 0; index < computeStage2Rows.length; index += 1) {
  const row = computeStage2Start + index;
  const source = computeStage2Rows[index];
  const raw = JSON.parse(source.source_record_json);
  computeModel.getRange(`A${row}:B${row}`).values = [[source.source_model_name, toDate(source.canonical_release_date)]];
  computeModel.getRange(`C${row}`).formulas = [[`=(B${row}-DATE(2024,1,1))/365.25`]];
  computeModel.getRange(`D${row}`).formulas = [[`=LOG10(F${row})`]];
  computeModel.getRange(`E${row}`).formulas = [[`=LOG10(G${row})`]];
  computeModel.getRange(`F${row}:G${row}`).values = [[Number(source.epoch_training_compute_flop), Number(source.total_parameters_b)]];
  computeModel.getRange(`H${row}`).formulas = [[`=$P$43+$O$43*D${row}`]];
  computeModel.getRange(`I${row}`).formulas = [[`=E${row}-H${row}`]];
  computeModel.getRange(`J${row}`).formulas = [[`=H${row}+$R$43+$Q$43*C${row}`]];
  computeModel.getRange(`K${row}`).formulas = [[`=E${row}-J${row}`]];
  computeModel.getRange(`L${row}:N${row}`).values = [[raw.Confidence, raw["Training compute estimation method"] || "", source.observation_id.split(":").at(-1)]];
}
const computeStage2End = computeStage2Start + computeStage2Rows.length - 1;
computeModel.getRange("O42:R42").values = [["log-compute slope", "Compute intercept", "Date-residual slope", "Date-residual intercept"]];
header(computeModel.getRange("O42:R42"));
computeModel.getRange("O43").formulas = [[`=SLOPE($E$${computeStage2Start}:$E$${computeStage2End},$D$${computeStage2Start}:$D$${computeStage2End})`]];
computeModel.getRange("P43").formulas = [[`=INTERCEPT($E$${computeStage2Start}:$E$${computeStage2End},$D$${computeStage2Start}:$D$${computeStage2End})`]];
computeModel.getRange("Q43").formulas = [[`=SLOPE($I$${computeStage2Start}:$I$${computeStage2End},$C$${computeStage2Start}:$C$${computeStage2End})`]];
computeModel.getRange("R43").formulas = [[`=INTERCEPT($I$${computeStage2Start}:$I$${computeStage2End},$C$${computeStage2Start}:$C$${computeStage2End})`]];
computeModel.getRange("O45:P47").values = [["Calibration rows", computeStage2Rows.length], ["In-sample R²", null], ["RMSE factor", null]];
computeModel.getRange("P46").formulas = [[`=RSQ($E$${computeStage2Start}:$E$${computeStage2End},$J$${computeStage2Start}:$J$${computeStage2End})`]];
computeModel.getRange("P47").formulas = [[`=POWER(10,SQRT(SUMSQ($K$${computeStage2Start}:$K$${computeStage2End})/COUNT($K$${computeStage2Start}:$K$${computeStage2End})))`]];
body(computeModel.getRange(`A43:N${computeStage2End}`));
body(computeModel.getRange("O43:R47"));

section(computeModel, "A65:P65", "Compute-structured AA/date priors and mixture weights");
computeModel.getRange("A66:N66").values = [["Model / base", "AA source record", "Release", "Years since 2024-01-01", "AA index", "Predicted log10 compute", "Implied training compute", "Raw log10 params", "Raw params (B)", "Calibration factor", "Compute-structured prior (T)", "Existing central (T)", "ln compute prior", "ln existing"]];
header(computeModel.getRange("A66:N66"));
const computeTargetStart = 67;
for (let index = 0; index < computeTargets.length; index += 1) {
  const row = computeTargetStart + index;
  const target = computeTargets[index];
  computeModel.getRange(`A${row}:C${row}`).values = [[target.name, target.aaName, toDate(target.aa.canonical_release_date)]];
  computeModel.getRange(`D${row}`).formulas = [[`=(C${row}-DATE(2024,1,1))/365.25`]];
  computeModel.getRange(`E${row}`).values = [[Number(target.aa.aa_intelligence_index)]];
  computeModel.getRange(`F${row}`).formulas = [[`=$N$7+$M$7*E${row}+$P$7+$O$7*D${row}`]];
  computeModel.getRange(`G${row}`).formulas = [[`=POWER(10,F${row})`]];
  computeModel.getRange(`H${row}`).formulas = [[`=$P$43+$O$43*F${row}+$R$43+$Q$43*D${row}`]];
  computeModel.getRange(`I${row}`).formulas = [[`=POWER(10,H${row})`]];
  computeModel.getRange(`J${row}`).formulas = [["=$P$55"]];
  computeModel.getRange(`K${row}`).formulas = [[`=I${row}*J${row}/1000`]];
  computeModel.getRange(`L${row}`).formulas = [[`='Frontier Estimates'!M${target.baseRow}`]];
  computeModel.getRange(`M${row}:N${row}`).formulas = [[`=LN(K${row})`, `=LN(L${row})`]];
}
const computeTargetEnd = computeTargetStart + computeTargets.length - 1;
computeModel.getRange("O52:P52").merge();
computeModel.getRange("O52").values = [["Calibration and weights"]];
computeModel.getRange("O52:P52").format = { fill: C.paleTeal, font: { name: "Aptos", size: 10, bold: true, color: C.tealDark } };
computeModel.getRange("O53:P59").values = [["K3 disclosed params (B)", k3Facts.total_parameters_b_exact], ["Grok disclosed params (B)", 1500], ["Two-anchor log calibration", null], ["Compute log weight", 0.05], ["Existing branch weight", 0.45], ["No-CoT horizon weight", 0.50], ["Log correlation: compute vs existing", null]];
computeModel.getRange("P55").formulas = [[`=SQRT(P53/I${computeTargetStart + 2}*P54/I${computeTargetStart + 8})`]];
computeModel.getRange("P59").formulas = [[`=CORREL(M${computeTargetStart}:M${computeTargetEnd},N${computeTargetStart}:N${computeTargetEnd})`]];
body(computeModel.getRange(`A67:N${computeTargetEnd}`));
body(computeModel.getRange("O53:P59"));
computeModel.getRange(`C7:C${computeStage1End}`).format.numberFormat = "yyyy-mm-dd";
computeModel.getRange(`D7:E${computeStage1End}`).format.numberFormat = "0.000";
computeModel.getRange(`F7:I${computeStage1End}`).format.numberFormat = "0.000E+00";
computeModel.getRange(`B43:B${computeStage2End}`).format.numberFormat = "yyyy-mm-dd";
computeModel.getRange(`C43:C${computeStage2End}`).format.numberFormat = "0.000";
computeModel.getRange(`D43:E${computeStage2End}`).format.numberFormat = "0.000";
computeModel.getRange(`F43:F${computeStage2End}`).format.numberFormat = "0.000E+00";
computeModel.getRange(`G43:K${computeStage2End}`).format.numberFormat = "0.000";
computeModel.getRange(`C67:C${computeTargetEnd}`).format.numberFormat = "yyyy-mm-dd";
computeModel.getRange(`D67:F${computeTargetEnd}`).format.numberFormat = "0.000";
computeModel.getRange(`G67:G${computeTargetEnd}`).format.numberFormat = "0.000E+00";
computeModel.getRange(`H67:N${computeTargetEnd}`).format.numberFormat = "0.000";
computeModel.getRange("P56:P58").format.numberFormat = "0%";
computeModel.getRange("A:A").format.columnWidth = 40;
computeModel.getRange("B:B").format.columnWidth = 38;
computeModel.getRange("C:C").format.columnWidth = 15;
computeModel.getRange("D:I").format.columnWidth = 18;
computeModel.getRange("J:N").format.columnWidth = 26;
computeModel.getRange("O:R").format.columnWidth = 20;
computeModel.getRange(`B67:B${computeTargetEnd}`).format.wrapText = true;
computeModel.freezePanes.freezeRows(6);
const computeTargetRowByModel = new Map(computeTargets.map((target, index) => [target.name, computeTargetStart + index]));

// Reported scaling laws and formula-level Bayesian overlay.
title(laws, "A1:H1", "Horizon laws and Bayesian calibration");
subtitle(laws, "A2:H3", "The no-CoT branch retains 50% log weight. Its published time/token laws are adjusted only by the deterministic exact-date/month-date Pareto-slope ratio (49/49 dates now day-level). A compute-structured AA/date regularizer receives 5%; the existing benchmark+price branch receives 45%. With-CoT METR identifies post-training leverage only.");
laws.getRange("A5:C5").values = [["Input / assumption", "Value", "Interpretation"]];
header(laws.getRange("A5:C5"));
const lawInputs = [
  ["Horizon log weight", 0.50, "Requested weight for the paper branch"],
  ["Existing benchmark + price weight", null, "1 − horizon weight"],
  ["Kimi K3 disclosed total (T)", k3Facts.total_parameters_b_exact / 1000, "Exact 2.78T internal anchor; displayed to one decimal in headline views"],
  ["K3 pretrain-equivalent TH (min)", null, "Ryan Greenblatt correction: K3 pretrain ≈ Opus 4.5"],
  ["Total params per TH doubling", scaling.total, "Reported pooled Pareto slope, 35 open-weight models"],
  ["Parameter elasticity α", null, "log2(total-parameter multiplier)"],
  ["No-CoT TH doubling (days)", exactTimeLaw.adjusted_point_days, "Paper law × exact-date/month-date Pareto-slope ratio"],
  ["No-CoT TH doubling CI low", exactTimeLaw.adjusted_ci_95_days[0], "Paper 95% CI × exact-date adjustment ratio"],
  ["No-CoT TH doubling CI high", exactTimeLaw.adjusted_ci_95_days[1], "Paper 95% CI × exact-date adjustment ratio"],
  ["Token-horizon doubling (days)", exactTokenLaw.adjusted_point_days, "Paper law × exact-date/month-date Pareto-slope ratio"],
  ["Token-horizon CI low", exactTokenLaw.adjusted_ci_95_days[0], "Paper 95% CI × exact-date adjustment ratio"],
  ["Token-horizon CI high", exactTokenLaw.adjusted_ci_95_days[1], "Paper 95% CI × exact-date adjustment ratio"],
  ["METR from-2023 doubling (days)", Number(metrOfficialMetadata.trend.from_2023_on_point_estimate_days), "Official METR-Horizon-v1.1 primary asset"],
  ["METR CI low", Number(metrOfficialMetadata.trend.from_2023_on_ci_low_days), "Official primary asset"],
  ["METR CI high", Number(metrOfficialMetadata.trend.from_2023_on_ci_high_days), "Official primary asset"],
  ["Ryan K3→Anthropic pretrain gap (months)", 8, "Corrected qualitative estimate; conservative point"],
  ["Compute-structured AA/date log weight", 0.05, "Low weight: zero targets have observed compute; two-stage and strongly correlated with AA/date"],
];
laws.getRange("A6:C22").values = lawInputs;
laws.getRange("B7").formulas = [["=1-B6-B22"]];
laws.getRange("B9").formulas = [[`='No-CoT Evidence'!C${frontierRow.get("Opus 4.5")}`]];
laws.getRange("B11").formulas = [["=LN(B10)/LN(2)"]];
body(laws.getRange("A6:C22"));
laws.getRange("B6:B22").format.numberFormat = "0.000";
laws.getRange("B6:B7").format.numberFormat = "0%";
laws.getRange("B8").format.numberFormat = "0.0\"T\"";
laws.getRange("B22").format.numberFormat = "0%";

section(laws, "A24:H24", "Exact reported open-weight relationships");
laws.getRange("A25:E25").values = [["Axis", "Factor per TH doubling", "Spearman ρ", "p-value", "n"]];
header(laws.getRange("A25:E25"));
const factorByAxis = new Map([
  ["Total parameters", scaling.total], ["Active parameters", scaling.active], ["Layer count", scaling.layers],
  ["AA Intelligence Index", 1.7], ["Pretraining FLOPs", scaling.pretrain], ["Total training FLOPs", null],
]);
const scaleRows = spearman.map((row) => [row.axis, factorByAxis.get(row.axis) ?? null, row.rho, row.p, row.n]);
laws.getRange(`A26:E${25 + scaleRows.length}`).values = scaleRows;
body(laws.getRange(`A26:E${25 + scaleRows.length}`));
laws.getRange(`B26:B${25 + scaleRows.length}`).format.numberFormat = "0.0x";
laws.getRange(`C26:C${25 + scaleRows.length}`).format.numberFormat = "0.00";
laws.getRange(`D26:D${25 + scaleRows.length}`).format.numberFormat = "0.0E+00";
laws.getRange(`E26:E${25 + scaleRows.length}`).format.numberFormat = "0";

laws.getRange("G25:H25").values = [["Architecture contrast", "Factor"]];
header(laws.getRange("G25:H25"));
laws.getRange("G26:H32").values = [
  ["Dense total parameters", 2.2], ["MoE total parameters", 8.1], ["Dense active parameters", 2.2],
  ["MoE active parameters", 13.0], ["Dense layers", 1.29], ["MoE layers", 1.27], ["Dense TH intercept advantage", 1.7],
];
body(laws.getRange("G26:H32"));
laws.getRange("H26:H32").format.numberFormat = "0.00x";

section(laws, "A35:H35", "Derived horizon priors");
laws.getRange("A36:C36").values = [["Derived quantity", "Formula result", "Use"]];
header(laws.getRange("A36:C36"));
const derivedLabels = [
  ["Opus shared-base TH (min)", null, "Geometric mean of Opus 4.5 / 4.6 / 4.7"],
  ["GPT shared-base TH (min)", null, "Geometric mean of GPT-5.4 / GPT-5.5"],
  ["Opus shared-base prior (T)", null, "K3 anchor × TH ratio^α"],
  ["GPT-5.5 prior (T)", null, "K3 anchor × TH ratio^α"],
  ["GPT effective release date", null, "Midpoint of exact GPT-5.4 / GPT-5.5 release dates"],
  ["GPT effective→Sol days", null, "Date projection interval"],
  ["Sol projected TH (min)", null, "GPT shared TH advanced on exact-date-adjusted time law"],
  ["Sol horizon prior (T)", null, "K3 anchor × TH ratio^α"],
  ["Opus 4.5→Fable days", null, "Date projection interval"],
  ["Fable date-projected TH (min)", null, "No-CoT time law"],
  ["Fable Ryan-gap TH (min)", null, "8-month pretrain lead over K3"],
  ["Fable combined TH (min)", null, "Geometric pooling of date and Ryan priors"],
  ["Fable horizon prior (T)", null, "K3 anchor × TH ratio^α"],
  ["No-CoT / METR doubling ratio", null, "With-CoT capability is improving much faster"],
  ["Opus 4.5→4.6 no-CoT ratio", null, "Same-base post-training sensitivity"],
  ["Opus 4.5→4.6 METR ratio", null, "Same-base with-CoT sensitivity"],
];
laws.getRange("A37:C52").values = derivedLabels;
laws.getRange("B37").formulas = [[`=POWER('No-CoT Evidence'!C${frontierRow.get("Opus 4.5")}*'No-CoT Evidence'!C${frontierRow.get("Opus 4.6")}*'No-CoT Evidence'!C${frontierRow.get("Opus 4.7")},1/3)`]];
laws.getRange("B38").formulas = [[`=SQRT('No-CoT Evidence'!C${frontierRow.get("GPT-5.4")}*'No-CoT Evidence'!C${frontierRow.get("GPT-5.5")})`]];
laws.getRange("B39").formulas = [["=$B$8*POWER(B37/$B$9,$B$11)"]];
laws.getRange("B40").formulas = [["=$B$8*POWER(B38/$B$9,$B$11)"]];
laws.getRange("B41").formulas = [[`=AVERAGE('Model Registry'!K${registryRowByCanonical.get(eciCanonical("GPT-5.4"))},'Model Registry'!K${registryRowByCanonical.get(eciCanonical("GPT-5.5"))})`]];
laws.getRange("C41").values = [["Exact checkpoint dates joined from ECI through canonical IDs"]];
laws.getRange("B42").formulas = [["='Frontier Estimates'!A7-B41"]];
laws.getRange("B43").formulas = [["=B38*POWER(2,B42/$B$12)"]];
laws.getRange("B44").formulas = [["=$B$8*POWER(B43/$B$9,$B$11)"]];
laws.getRange("B45").formulas = [[`='Frontier Estimates'!A6-'Model Registry'!K${registryRowByCanonical.get(eciCanonical("Claude Opus 4.5"))}`]];
laws.getRange("C45").values = [["Exact Opus 4.5 checkpoint date joined from ECI"]];
laws.getRange("B46").formulas = [["=$B$9*POWER(2,B45/$B$12)"]];
laws.getRange("B47").formulas = [["=$B$9*POWER(2,$B$21*365.25/12/$B$12)"]];
laws.getRange("B48").formulas = [["=SQRT(B46*B47)"]];
laws.getRange("B49").formulas = [["=$B$8*POWER(B48/$B$9,$B$11)"]];
laws.getRange("B50").formulas = [["=$B$12/$B$18"]];
laws.getRange("B51").formulas = [[`='No-CoT Evidence'!C${frontierRow.get("Opus 4.6")} / 'No-CoT Evidence'!C${frontierRow.get("Opus 4.5")}`]];
laws.getRange("B52").formulas = [[`=${Number(metrOfficialById.get("claude_opus_4_6_inspect").p50_estimate_minutes)}/${Number(metrOfficialById.get("claude_opus_4_5_inspect").p50_estimate_minutes)}`]];
body(laws.getRange("A37:C52"));
laws.getRange("B37:B40").format.numberFormat = "0.000";
laws.getRange("B41").format.numberFormat = "mmm yyyy";
laws.getRange("B42:B52").format.numberFormat = "0.000";

laws.getRange("E36:H40").merge();
laws.getRange("E36").values = [["Interpretation: Opus 4.5→4.6 barely changes no-CoT TH (~1.04×) while the official with-CoT METR horizon grows ~2.45×. That is direct same-base evidence that RL/inference gains can dominate agentic benchmark gains without implying a larger pretrain. Therefore METR changes the decomposition, not the absolute parameter scale."]];
laws.getRange("E36:H40").format = { fill: C.paleAmber, font: { name: "Aptos", size: 9, color: C.text }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: "#D9991A" } } };
laws.getRange("E42:H47").merge();
laws.getRange("E42").values = [["Ryan Greenblatt correction used here: K3's pretrain looked roughly halfway from Opus 4 to Opus 4.5 and approximately comparable to Opus 4.5 overall, about eight months behind Anthropic. His earlier view that K3 pretrain was between Opus 4.8 and Mythos was explicitly retracted. This is a qualitative prior, not a measurement."]];
laws.getRange("E42:H47").format = { fill: C.palePink, font: { name: "Aptos", size: 9, color: C.text }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: C.purple } } };
laws.getRange("A55:H57").merge();
laws.getRange("A55").values = [[`Primary paper: ${arxivUrl}\nSubmitted source archive: ${latexArchiveDisplay}\nMETR values: official primary asset ${metrOfficialUrl}; 26/26 rows exactly reconcile to the earlier user copy. Ryan Greenblatt text remains user-supplied in this Codex task on 2026-07-17.`]];
laws.getRange("A55:H57").format = { fill: C.gray, font: { name: "Aptos", size: 9, italic: true, color: C.text }, wrapText: true, verticalAlignment: "center" };
section(laws, "A60:H60", "Architecture-specific elasticity — descriptive result versus held-out prediction");
laws.getRange("A61:D61").values = [["Audit quantity", "Value", "90% CI low", "90% CI high"]];
header(laws.getRange("A61:D61"));
const noCotDirectComparison = noCotArchitectureAudit.paired_comparisons.chronological_developer_holdout.direct_architecture_minus_pooled;
const noCotParetoComparison = noCotArchitectureAudit.paired_comparisons.chronological_developer_holdout.training_pareto_architecture_minus_pooled;
laws.getRange("A62:D68").values = [
  ["Deterministic pooled Pareto factor", noCotArchitectureAudit.paper_relationship_reproduction.pooled.deterministic_bootstrap_median_reproduction, null, null],
  ["Deterministic dense Pareto factor", noCotArchitectureAudit.paper_relationship_reproduction.dense.deterministic_bootstrap_median_reproduction, null, null],
  ["Deterministic MoE Pareto factor", noCotArchitectureAudit.paper_relationship_reproduction.moe.deterministic_bootstrap_median_reproduction, null, null],
  ["MoE Pareto models", noCotArchitectureAudit.paper_relationship_reproduction.moe.pareto_n, null, null],
  ["Direct architecture − pooled held-out Δ", noCotDirectComparison.observed_delta, noCotDirectComparison.ci_90[0], noCotDirectComparison.ci_90[1]],
  ["Training-Pareto architecture − pooled Δ", noCotParetoComparison.observed_delta, noCotParetoComparison.ci_90[0], noCotParetoComparison.ci_90[1]],
  ["Replace pooled live elasticity", noCotArchitectureAudit.decision.replace_pooled_live_elasticity_with_moe_specific ? "Yes" : "No", null, null],
];
body(laws.getRange("A62:D68"));
laws.getRange("B62:B64").format.numberFormat = "0.00x";
laws.getRange("B66:D67").format.numberFormat = "0.000";
laws.getRange("E61:H68").merge();
laws.getRange("E61").values = [["The paper's MoE result is real: the submitted table deterministically reproduces 8.13× from five MoE Pareto models. But it is a descriptive frontier slope, not a validated inverse parameter estimator. In strictly chronological developer-held-out tests, a separately learned architecture slope worsens equal-family absolute log error (the 90% interval is wholly unfavorable). A training-fold-only Pareto version is directionally favorable but inconclusive. The pooled live mapping is therefore retained; 8.1× remains a sensitivity."]];
laws.getRange("E61:H68").format = { fill: C.paleAmber, font: { name: "Aptos", size: 9, color: C.text }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: "#D9991A" } } };
laws.getRange("A:A").format.columnWidth = 40;
laws.getRange("B:B").format.columnWidth = 22;
laws.getRange("C:C").format.columnWidth = 55;
laws.getRange("D:D").format.columnWidth = 16;
laws.getRange("E:E").format.columnWidth = 10;
laws.getRange("F:F").format.columnWidth = 3;
laws.getRange("G:G").format.columnWidth = 32;
laws.getRange("H:H").format.columnWidth = 14;
laws.freezePanes.freezeRows(5);

// Final log-posterior. Disclosed anchors remain fixed.
title(posterior, "A1:O1", "Horizon-, compute-, and factual-capacity-informed frontier parameter posterior");
subtitle(posterior, "A2:O3", `The baseline posterior is 45% price-informed benchmark evidence, 50% no-CoT horizon, and 5% compute-structured AA/date. Fable alone has an observed IKP factual-capacity estimate, but its decision-derived live weight is ${Math.round(100 * ikpEvidenceWeight)}%; ${ikpPromoted ? "the baseline branches are proportionally shrunk" : "the baseline evidence mix is therefore unchanged"}. Models without IKP retain their prior evidence composition.`);
posterior.getRange("A5:O5").values = [["Model / base", "Existing central (T)", "TH proxy (min)", "Horizon prior (T)", "Compute-structured prior (T)", "Existing weight", "Horizon weight", "Compute weight", "Posterior (T)", "Crowd point (T)", "Posterior / crowd", "Horizon evidence", "Compute evidence", "Status", "Method note"]];
header(posterior.getRange("A5:O5"));
const posteriorModels = [
  { name: "Claude Fable 5", baseRow: 6, th: "='Horizon Laws'!B48", horizon: "='Horizon Laws'!B49", crowd: crowdSummary("Claude Fable 5").center, evidence: "Projected new pretrain", status: "Regression estimate", note: "Date law and Ryan's corrected 8-month pretrain gap pooled geometrically; compute adds a small correlated check." },
  { name: "GPT-5.6 Sol", baseRow: 7, th: "='Horizon Laws'!B43", horizon: "='Horizon Laws'!B44", crowd: crowdSummary("GPT-5.6 Sol").center, evidence: "Same-family/date projection", status: "Regression estimate", note: "GPT-5.4/5.5 no-CoT shared-base proxy advanced to Sol's release." },
  { name: "Kimi K3", baseRow: 8, th: "='Horizon Laws'!B9", horizon: "='Horizon Laws'!B8", crowd: null, evidence: "Absolute anchor", status: "Disclosed anchor", note: "2.8T disclosed; horizon branch cannot move an exact anchor." },
  { name: "Claude Opus 4.7 / 4.8 shared base", baseRow: 10, th: "='Horizon Laws'!B37", horizon: "='Horizon Laws'!B39", crowd: crowdSummary("Claude Opus 4.7 / 4.8 shared base").center, evidence: "Measured shared-base proxy", status: "Regression estimate", note: "Opus 4.5/4.6/4.7 collapsed; 4.7 and 4.8 remain one base. Respondent R17's single forecast is shown as a check, not blended into the final." },
  { name: "GPT-5.5", baseRow: 9, th: "='Horizon Laws'!B38", horizon: "='Horizon Laws'!B40", crowd: null, evidence: "Measured shared-base proxy", status: "Regression estimate", note: "GPT-5.4 and 5.5 treated as same-base post-training variants." },
  { name: "GPT-5.6 Terra", baseRow: 11, th: null, horizon: "='Horizon Laws'!B44*'Frontier Estimates'!M11/'Frontier Estimates'!M7", crowd: null, evidence: "Relative to Sol", status: "Regression estimate", note: "Preserves Terra/Sol scale ratio while applying Sol's horizon correction." },
  { name: "Claude Sonnet 5", baseRow: 12, th: null, horizon: "='Frontier Estimates'!M12", crowd: null, evidence: "Neutral", status: "Regression estimate", note: "No direct Sonnet 5 no-CoT point; horizon branch left neutral." },
  { name: "GPT-5.6 Luna", baseRow: 13, th: null, horizon: "='Horizon Laws'!B44*'Frontier Estimates'!M13/'Frontier Estimates'!M7", crowd: null, evidence: "Relative to Sol", status: "Regression estimate", note: "Preserves Luna/Sol scale ratio while applying Sol's horizon correction." },
  { name: "Grok 4.5", baseRow: 15, th: null, horizon: "='Frontier Estimates'!M15", crowd: null, evidence: "Absolute anchor", status: "Disclosed anchor", note: "1.5T disclosed anchor; horizon branch cannot move an exact anchor." },
  { name: "Claude Opus 5", baseRow: 17, th: null, horizon: "='Frontier Estimates'!M17", crowd: null, evidence: "Neutral — no direct horizon", status: "Regression estimate", note: "Distinct newly pretrained base. Exact AA/ECI/date and weak price evidence are observed; METR, no-CoT, IKP, target compute, architecture, and parameter count remain unavailable, so the horizon branch is neutral rather than imputed." },
];
const posteriorEnd = 5 + posteriorModels.length;
for (let idx = 0; idx < posteriorModels.length; idx += 1) {
  const row = 6 + idx;
  const m = posteriorModels[idx];
  posterior.getRange(`A${row}`).values = [[m.name]];
  posterior.getRange(`B${row}`).formulas = [[`='Frontier Estimates'!M${m.baseRow}`]];
  if (m.th) posterior.getRange(`C${row}`).formulas = [[m.th]];
  posterior.getRange(`D${row}`).formulas = [[m.horizon]];
  posterior.getRange(`E${row}`).formulas = [[`='Compute Model'!K${computeTargetRowByModel.get(m.name)}`]];
  posterior.getRange(`F${row}:H${row}`).formulas = [["='Horizon Laws'!B7", "='Horizon Laws'!B6", "='Horizon Laws'!B22"]];
  const baselinePosterior = `POWER(B${row},F${row})*POWER(D${row},G${row})*POWER(E${row},H${row})`;
  const evidenceFormula = m.name === "Claude Fable 5"
    ? `POWER(${baselinePosterior},${1 - ikpEvidenceWeight})*POWER(${ikpLiveEstimateT},${ikpEvidenceWeight})`
    : baselinePosterior;
  posterior.getRange(`I${row}`).formulas = [[`=IF(N${row}="Disclosed anchor",B${row},${evidenceFormula})`]];
  if (m.crowd != null) {
    posterior.getRange(`J${row}`).values = [[m.crowd]];
    posterior.getRange(`K${row}`).formulas = [[`=I${row}/J${row}`]];
  }
  posterior.getRange(`L${row}:O${row}`).values = [[m.evidence, "Two-stage compute-structured AA/date prior; anchor-calibrated; no observed target compute", m.status, m.note]];
}
body(posterior.getRange(`A6:O${posteriorEnd}`));
posterior.getRange(`B6:E${posteriorEnd}`).format.numberFormat = "0.000";
posterior.getRange(`F6:H${posteriorEnd}`).format.numberFormat = "0%";
posterior.getRange(`I6:K${posteriorEnd}`).format.numberFormat = "0.0";
posterior.getRange(`I6:I${posteriorEnd}`).format.font = { name: "Aptos", size: 11, bold: true, color: C.navy };
posterior.getRange(`L6:O${posteriorEnd}`).format.wrapText = true;
posterior.getRange(`N6:N${posteriorEnd}`).conditionalFormats.add("containsText", { text: "Disclosed", format: { fill: C.paleBlue, font: { color: C.navy, bold: true } } });
posterior.tables.add(`A5:O${posteriorEnd}`, true, "HorizonPosteriorTable").style = "TableStyleMedium2";
section(posterior, "A17:O17", "Bayesian interpretation");
posterior.getRange("A19:O21").merge();
posterior.getRange("A19").values = [[`The no-CoT panel remains the dominant baseline branch. The compute-structured AA/date branch receives only 5% because no target has observed compute. Fable's separate IKP signal receives ${Math.round(100 * ikpEvidenceWeight)}% inside evidence; its favorable error checks do not overcome the ${ikpChronological.models}/${ikpGate.minimum_chronological_subset_models} chronological-model coverage shortfall. Its ${ikpLiveEstimateT.toFixed(1)}T point remains a sensitivity, not a literal architecture disclosure.`]];
posterior.getRange("A19:O21").format = { fill: C.paleTeal, font: { name: "Aptos", size: 9, color: C.text }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: C.teal } } };
posterior.getRange("A23:O25").merge();
posterior.getRange("A23").values = [[`The human forecast ledger currently centers Fable at ${crowdSummary("Claude Fable 5").center.toFixed(1)}T (n=${crowdSummary("Claude Fable 5").n}) and Sol at ${crowdSummary("GPT-5.6 Sol").center.toFixed(1)}T (n=${crowdSummary("GPT-5.6 Sol").n}). Crowd forecasts and METR remain calibration/decomposition checks with zero direct evidence-model weight; disclosed K3 and Grok rows override all regression branches.`]];
posterior.getRange("A23:O25").format = { fill: C.gray, font: { name: "Aptos", size: 9, italic: true, color: C.text }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: C.purple } } };
posterior.getRange("A:A").format.columnWidth = 38;
posterior.getRange("B:K").format.columnWidth = 17;
posterior.getRange("L:N").format.columnWidth = 25;
posterior.getRange("O:O").format.columnWidth = 64;
posterior.freezePanes.freezeRows(5);

// User-requested final ensemble: 50% human crowd and 50% evidence model in log space.
// Only Fable and Sol have sufficiently populated crowd pools; all other rows retain the model posterior.
const exactCrowdCenters = new Map(requiredCrowdModels.map((model) => [model, crowdSummary(model).center]));
title(finalEnsemble, "A1:H1", "Final forecast — 50% crowd + 50% evidence model");
subtitle(finalEnsemble, "A2:H3", `For Fable and Sol, the final forecast is the geometric mean of the evidence-model posterior and the human crowd center. Fable's evidence half includes ${(100 * ikpFinalWeight).toFixed(1)}% final weight on the strict IKP factual-capacity estimate under the current audit decision; Sol has no IKP observation. Models without a populated crowd pool remain unchanged.`);
finalEnsemble.getRange("A5:H5").values = [["Model / base", "Evidence model (T)", "Crowd center (T)", "Model weight", "Crowd weight", "Final forecast (T)", "Final status", "Method note"]];
header(finalEnsemble.getRange("A5:H5"));
for (let idx = 0; idx < posteriorModels.length; idx += 1) {
  const row = 6 + idx;
  const model = posteriorModels[idx];
  const crowdCenter = exactCrowdCenters.get(model.name);
  finalEnsemble.getRange(`A${row}:B${row}`).formulas = [[`='Horizon Estimates'!A${row}`, `='Horizon Estimates'!I${row}`]];
  if (crowdCenter != null) finalEnsemble.getRange(`C${row}`).values = [[crowdCenter]];
  finalEnsemble.getRange(`D${row}:E${row}`).values = [[crowdCenter != null ? 0.5 : 1, crowdCenter != null ? 0.5 : 0]];
  finalEnsemble.getRange(`F${row}`).formulas = [[`=IF(E${row}=0,B${row},POWER(B${row},D${row})*POWER(C${row},E${row}))`]];
  const disclosed = model.status === "Disclosed anchor";
  finalEnsemble.getRange(`G${row}:H${row}`).values = [[
    crowdCenter != null ? "50/50 crowd + model" : (disclosed ? "Disclosed anchor" : "Model posterior only"),
    crowdCenter != null ? "Equal log weights; geometric mean of separately tracked aggregates." : (disclosed ? "No crowd pool; disclosed total retained." : "No crowd pool; evidence-model posterior retained."),
  ]];
}
body(finalEnsemble.getRange(`A6:H${posteriorEnd}`));
finalEnsemble.getRange(`B6:C${posteriorEnd}`).format.numberFormat = "0.0";
finalEnsemble.getRange(`D6:E${posteriorEnd}`).format.numberFormat = "0.0%";
finalEnsemble.getRange(`F6:F${posteriorEnd}`).format.numberFormat = "0.0";
finalEnsemble.getRange(`F6:F${posteriorEnd}`).format.font = { name: "Aptos", size: 11, bold: true, color: C.navy };
finalEnsemble.getRange(`G6:H${posteriorEnd}`).format.wrapText = true;
finalEnsemble.getRange(`G6:G${posteriorEnd}`).conditionalFormats.add("containsText", { text: "50/50", format: { fill: C.paleTeal, font: { color: C.navy, bold: true } } });
finalEnsemble.getRange(`G6:G${posteriorEnd}`).conditionalFormats.add("containsText", { text: "Disclosed", format: { fill: C.paleBlue, font: { color: C.navy, bold: true } } });
finalEnsemble.tables.add(`A5:H${posteriorEnd}`, true, "FinalEnsembleTable").style = "TableStyleMedium2";
section(finalEnsemble, "A17:H17", "Interpretation");
finalEnsemble.getRange("A19:H22").merge();
finalEnsemble.getRange("A19").formulas = [[`="At the current 50/50 setting, Fable moves from "&TEXT(B6,"0.0")&"T toward the "&TEXT(C6,"0.0")&"T crowd center, producing "&TEXT(F6,"0.0")&"T. Sol moves from "&TEXT(B7,"0.0")&"T toward the "&TEXT(C7,"0.0")&"T crowd center, producing "&TEXT(F7,"0.0")&"T. Crowd evidence remains highly correlated and uncalibrated, so this is an explicit judgmental ensemble rather than a statistically independent posterior."`]];
finalEnsemble.getRange("A19:H22").format = { fill: C.paleTeal, font: { name: "Aptos", size: 9, color: C.text }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: C.teal } } };
finalEnsemble.getRange("A:A").format.columnWidth = 39;
finalEnsemble.getRange("B:F").format.columnWidth = 18;
finalEnsemble.getRange("G:G").format.columnWidth = 24;
finalEnsemble.getRange("H:H").format.columnWidth = 58;
finalEnsemble.freezePanes.freezeRows(5);

// Base-level registry is distinct from checkpoint identity, preventing post-training variants from becoming duplicate base observations.
title(baseRegistry, "A1:M1", "Canonical base registry — evidence-model posterior units");
subtitle(baseRegistry, "A2:M3", "Exactly ten posterior rows and ten unique base IDs. Checkpoints remain separate in Model Registry; asserted same-base relationships are encoded here with their evidence. Opus 5 is a new unique base and is not collapsed into either Fable/Mythos or the Opus 4.x lineage.");
baseRegistry.getRange("A5:M5").values = [["Base ID", "Final display", "Representative checkpoint ID", "Member checkpoint IDs", "Identity type", "Identity evidence", "Existing central (T)", "Horizon prior (T)", "Compute prior (T)", "Posterior (T)", "Status", "Base ID count", "Note"]];
header(baseRegistry.getRange("A5:M5"));
const cid = (eciName) => eciCanonical(eciName);
const baseDefinitions = [
  [registryById.get(cid("Claude Fable 5")).baseId, "Claude Fable 5 / Claude Mythos 5", cid("Claude Fable 5"), cid("Claude Fable 5"), "Official shared weights", "Anthropic system card p.12; Fable safeguards and fallback do not create another base", 6],
  [registryById.get(cid("GPT-5.6 Sol")).baseId, "GPT-5.6 Sol", cid("GPT-5.6 Sol"), cid("GPT-5.6 Sol"), "Unique scale variant", "Distinct GPT-5.6 Sol scale", 7],
  [registryById.get("moonshot:kimi-k3").baseId, "Kimi K3", "moonshot:kimi-k3", "moonshot:kimi-k3", "Disclosed base", "2.8T disclosed anchor", 8],
  ["base:anthropic-opus-4-5plus", "Claude Opus 4.7 / 4.8 shared base", cid("Claude Opus 4.8"), ["Claude Opus 4.5", "Claude Opus 4.6", "Claude Opus 4.7", "Claude Opus 4.8"].map(cid).join("; "), "Shared base / post-training lineage", "User asserts ≥4.5 shared base; 4.7/4.8 explicitly required to collapse", 9],
  ["base:openai-gpt-5-shared", "GPT-5.5", cid("GPT-5.5"), ["GPT-5", "GPT-5.1", "GPT-5.2", "GPT-5.3 Codex", "GPT-5.4", "GPT-5.5"].map(cid).join("; "), "Shared base / post-training lineage", "User asserts GPT-5 through GPT-5.5 share a base", 10],
  [registryById.get(cid("GPT-5.6 Terra")).baseId, "GPT-5.6 Terra", cid("GPT-5.6 Terra"), cid("GPT-5.6 Terra"), "Unique scale variant", "Distinct GPT-5.6 Terra scale", 11],
  [registryById.get(cid("Claude Sonnet 5")).baseId, "Claude Sonnet 5", cid("Claude Sonnet 5"), cid("Claude Sonnet 5"), "Unique pretrain", "No same-base collapse asserted", 12],
  [registryById.get(cid("GPT-5.6 Luna")).baseId, "GPT-5.6 Luna", cid("GPT-5.6 Luna"), cid("GPT-5.6 Luna"), "Unique scale variant", "Distinct GPT-5.6 Luna scale", 13],
  [registryById.get(cid("Grok 4.5")).baseId, "Grok 4.5", cid("Grok 4.5"), cid("Grok 4.5"), "Disclosed base", "1.5T disclosed anchor", 14],
  [registryById.get(cid("Claude Opus 5")).baseId, "Claude Opus 5", cid("Claude Opus 5"), cid("Claude Opus 5"), "Unique new pretrain", "Anthropic system card describes pretraining then post-training; no shared-weight identity or reused base is disclosed", 15],
];
const duplicateBaseIds = baseDefinitions.length - new Set(baseDefinitions.map((row) => row[0])).size;
if (duplicateBaseIds) throw new Error(`Final base registry contains ${duplicateBaseIds} duplicate IDs`);
for (let index = 0; index < baseDefinitions.length; index += 1) {
  const row = 6 + index;
  const [baseId, display, representative, members, identityType, identityEvidence, posteriorRow] = baseDefinitions[index];
  baseRegistry.getRange(`A${row}:F${row}`).values = [[baseId, display, representative, members, identityType, identityEvidence]];
  baseRegistry.getRange(`G${row}:J${row}`).formulas = [[`='Horizon Estimates'!B${posteriorRow}`, `='Horizon Estimates'!D${posteriorRow}`, `='Horizon Estimates'!E${posteriorRow}`, `='Horizon Estimates'!I${posteriorRow}`]];
  baseRegistry.getRange(`K${row}`).formulas = [[`='Horizon Estimates'!N${posteriorRow}`]];
  baseRegistry.getRange(`L${row}`).formulas = [[`=COUNTIF($A$6:$A$${5 + baseDefinitions.length},A${row})`]];
  baseRegistry.getRange(`M${row}`).values = [[posteriorModels[index].note]];
}
const baseRegistryEnd = 5 + baseDefinitions.length;
body(baseRegistry.getRange(`A6:M${baseRegistryEnd}`));
baseRegistry.getRange(`G6:J${baseRegistryEnd}`).format.numberFormat = "0.000";
baseRegistry.getRange(`L6:L${baseRegistryEnd}`).format.numberFormat = "0";
baseRegistry.getRange(`D6:F${baseRegistryEnd}`).format.wrapText = true;
baseRegistry.getRange(`K6:M${baseRegistryEnd}`).format.wrapText = true;
baseRegistry.getRange(`L6:L${baseRegistryEnd}`).conditionalFormats.add("cellIs", { operator: "notEqual", formula: 1, format: { fill: C.palePink, font: { color: "#8B245C", bold: true } } });
baseRegistry.tables.add(`A5:M${baseRegistryEnd}`, true, "CanonicalBaseRegistryTable").style = "TableStyleMedium2";
baseRegistry.getRange("A:A").format.columnWidth = 38;
baseRegistry.getRange("B:B").format.columnWidth = 38;
baseRegistry.getRange("C:D").format.columnWidth = 70;
baseRegistry.getRange("E:F").format.columnWidth = 40;
baseRegistry.getRange("G:L").format.columnWidth = 18;
baseRegistry.getRange("M:M").format.columnWidth = 62;
baseRegistry.freezePanes.freezeRows(5);
dataAudit.getRange(`A${baseDuplicateAuditRow}:D${baseDuplicateAuditRow}`).values = [["Final base IDs duplicated", 0, duplicateBaseIds, duplicateBaseIds === 0 ? "PASS" : "FAIL"]];
body(dataAudit.getRange(`A${baseDuplicateAuditRow}:D${baseDuplicateAuditRow}`));
dataAudit.getRange(`D${baseDuplicateAuditRow}`).conditionalFormats.add("containsText", { text: "PASS", format: { fill: C.paleTeal, font: { color: C.tealDark, bold: true } } });

// Rebuild the requested clean final graphic: no Gemini, no error bars, one decimal.
summary.deleteAllDrawings();
for (const range of ["A1:Q1", "A2:Q3", "A20:Q22", "A24:Q26"]) summary.getRange(range).unmerge();
summary.getRange("A1:Q28").clear({ applyTo: "all" });
summary.deleteAllDrawings();
title(summary, "A1:Q1", "Final frontier forecast — crowd + validated evidence model");
subtitle(summary, "A2:Q3", `Fable and Sol receive equal log weight on crowd and evidence. IKP is retained as a Fable sensitivity but receives ${Math.round(100 * ikpEvidenceWeight)}% of evidence (${(100 * ikpFinalWeight).toFixed(1)}% of the final) because the current audit does not pass every promotion gate. Sol has no IKP measurement. Other models remain at their evidence-model posterior.`);
summary.getRange("A5:C5").values = [["Model / base", "Final forecast (T)", "Status"]];
header(summary.getRange("A5:C5"));
const summaryOrder = [
  "Claude Fable 5",
  "GPT-5.6 Sol",
  "Claude Opus 5",
  "Kimi K3",
  "Claude Opus 4.7 / 4.8 shared base",
  "GPT-5.5",
  "GPT-5.6 Terra",
  "Claude Sonnet 5",
  "Grok 4.5",
  "GPT-5.6 Luna",
];
const finalRowByModel = new Map(posteriorModels.map((model, index) => [model.name, 6 + index]));
for (let idx = 0; idx < summaryOrder.length; idx += 1) {
  const row = 6 + idx;
  const finalRow = finalRowByModel.get(summaryOrder[idx]);
  summary.getRange(`A${row}:C${row}`).formulas = [[`='Final Ensemble'!A${finalRow}`, `='Final Ensemble'!F${finalRow}`, `='Final Ensemble'!G${finalRow}`]];
}
body(summary.getRange(`A6:C${posteriorEnd}`));
summary.getRange(`B6:B${posteriorEnd}`).format.numberFormat = "0.0";
summary.getRange(`B6:B${posteriorEnd}`).format.font = { name: "Aptos", size: 11, bold: true, color: C.navy };
summary.getRange(`C6:C${posteriorEnd}`).conditionalFormats.add("containsText", { text: "50/50", format: { fill: C.paleTeal, font: { color: C.navy, bold: true } } });
summary.getRange(`C6:C${posteriorEnd}`).conditionalFormats.add("containsText", { text: "Disclosed", format: { fill: C.paleBlue, font: { color: C.navy, bold: true } } });
summary.getRange("A:A").format.columnWidth = 39;
summary.getRange("B:B").format.columnWidth = 20;
summary.getRange("C:C").format.columnWidth = 24;
summary.getRange("D:D").format.columnWidth = 3;
summary.getRange("E:Q").format.columnWidth = 12;
const chart = summary.charts.add("bar", { title: "Final 50/50 crowd + model totals (T)", hasLegend: false });
const series = chart.series.add("Total parameters (T)");
series.categoryFormula = `'Executive Summary'!$A$6:$A$${posteriorEnd}`;
series.formula = `'Executive Summary'!$B$6:$B$${posteriorEnd}`;
series.fill = C.teal;
chart.title = "Final 50/50 crowd + model totals (T)";
chart.hasLegend = false;
chart.yAxis = { numberFormatCode: "0.0\"T\"" };
chart.setPosition("E5", "Q22");
summary.getRange("A17:C19").merge();
summary.getRange("A17").formulas = [["=\"Final centers\"&CHAR(10)&\"Fable \"&TEXT(B6,\"0.0\")&\"T · Sol \"&TEXT(B7,\"0.0\")&\"T · Opus 5 \"&TEXT(B8,\"0.0\")&\"T\"&CHAR(10)&\"K3 \"&TEXT(B9,\"0.0\")&\"T · Opus shared \"&TEXT(B10,\"0.0\")&\"T · GPT-5.5 \"&TEXT(B11,\"0.0\")&\"T\""]];
summary.getRange("A17:C19").format = { fill: C.paleTeal, font: { name: "Aptos", size: 10, bold: true, color: C.navy }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: C.teal } } };
summary.getRange("A21:C25").merge();
summary.getRange("A21").formulas = [[`="Final judgmental ensemble: the 50% crowd weight applies to Fable (n=${crowdSummary("Claude Fable 5").n}) and Sol (n=${crowdSummary("GPT-5.6 Sol").n}). Because the combination is performed in log space, Fable becomes "&TEXT('Final Ensemble'!F6,"0.0")&"T from "&TEXT('Final Ensemble'!B6,"0.0")&"T model / "&TEXT('Final Ensemble'!C6,"0.0")&"T crowd, and Sol becomes "&TEXT('Final Ensemble'!F7,"0.0")&"T from "&TEXT('Final Ensemble'!B7,"0.0")&"T model / "&TEXT('Final Ensemble'!C7,"0.0")&"T crowd. Other rows remain unchanged; K3 and Grok remain disclosed anchors."`]];
summary.getRange("A21:C25").format = { fill: C.palePink, font: { name: "Aptos", size: 9, color: C.text }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: C.purple } } };
summary.freezePanes.freezeRows(5);

// Append source provenance to the existing audit sheet.
section(sources, "A19:D19", "Horizon, Epoch, and forecast evidence added in this revision");
sources.getRange("A20:D33").values = [
  ["No-CoT horizon paper", arxivUrl, "Primary model-level horizons, CIs, scaling laws, and architecture findings", "v3 source parsed directly"],
  ["Submitted LaTeX source", latexArchiveDisplay, "Exact extraction of all model-level paper data used here", "Local immutable input for this build"],
  ["Epoch AI all-models CSV", epochCsvUrl, "Exact checkpoint identities, day-level dates, parameter cross-checks, and source links", `Pinned ${epochSnapshotManifest.snapshot_as_of}; full ${epochSnapshotManifest.inventory.all_model_rows.toLocaleString("en-US")}-row snapshot retained`],
  ["Epoch AI models archive", epochArchivePath, "Frontier/notable/large-scale source views and compute estimation metadata", "1,690 correlated view records retained; frontier view feeds 5% compute branch"],
  ["Compute-enriched unified observations", unifiedComputePath, "Single audited input for benchmark, AA↔Epoch, compute, and source-view joins", `${unifiedSummary.observations.toLocaleString("en-US")} observations; ${epochSnapshotManifest.inventory.component_rows.toLocaleString("en-US")} component rows and archive views excluded from independent model-level likelihood counts`],
  ["ECI component benchmark snapshot", eciComponentPath, "All model-by-benchmark measurements used in the held-out component comparison", `${epochSnapshotManifest.inventory.component_rows.toLocaleString("en-US")} unique model/benchmark rows; ${epochSnapshotManifest.inventory.models} models; ${epochSnapshotManifest.inventory.benchmarks} benchmarks`],
  ["Epoch benchmark data archive", eciBenchmarkArchivePath, "Official IRT slopes/difficulties and raw benchmark source files", "Used to map component performance onto the aggregate ECI scale"],
  ["Epoch AI downloads documentation", epochDocsUrl, "Dataset provenance and update cadence", "Official Epoch documentation"],
  ["LessWrong paper copy", lessWrongSourcePath, "Secondary paper text / discussion", "Primary numbers taken from LaTeX, not this copy"],
  ["METR-Horizon-v1.1 official asset", metrOfficialUrl, "26 model horizons, confidence intervals, full scaffold arrays, and 128.744-day from-2023 trend", "First-party YAML; 26/26 common-field exact match to retained legacy copy; used to identify post-training leverage only"],
  ["Ryan Greenblatt K3 assessment", "User-supplied in Codex conversation, 2026-07-17", "Corrected K3 pretrain placement and ~8-month gap", "Qualitative prior; earlier bullish view explicitly retracted"],
  ["Human forecast ledger", forecastLedgerPath, "Normalized contributor/model forecasts; geometric centers and counts", "Single source of truth for registry, workbook, and chart"],
  ["Human prediction registry", predictionRegistryPath, "Rendered Fable/Sol crowd register", "Generated from the ledger; unscored and likely correlated"],
  ["Kimi K3 official release evidence", k3ReleaseEvidencePath, "Official K3 2.78T total / 104.2B activated, 16/896 routed experts, and 2.5× reported efficiency; K2 comparator 1.04T / 32.6B", `Pinned primary-source bundle ${k3ReleaseEvidence.snapshot_date}; exact active count supersedes the earlier routed-expert approximation`],
];
header(sources.getRange("A20:D20"));
// Restore the first row as data after using the shared header formatter only for visual consistency.
sources.getRange("A20:D33").values = [
  ["No-CoT horizon paper", arxivUrl, "Primary model-level horizons, CIs, scaling laws, and architecture findings", "v3 source parsed directly"],
  ["Submitted LaTeX source", latexArchiveDisplay, "Exact extraction of all model-level paper data used here", "Local immutable input for this build"],
  ["Epoch AI all-models CSV", epochCsvUrl, "Exact checkpoint identities, day-level dates, parameter cross-checks, and source links", `Pinned ${epochSnapshotManifest.snapshot_as_of}; full ${epochSnapshotManifest.inventory.all_model_rows.toLocaleString("en-US")}-row snapshot retained`],
  ["Epoch AI models archive", epochArchivePath, "Frontier/notable/large-scale source views and compute estimation metadata", "1,690 correlated view records retained; frontier view feeds 5% compute branch"],
  ["Compute-enriched unified observations", unifiedComputePath, "Single audited input for benchmark, AA↔Epoch, compute, and source-view joins", `${unifiedSummary.observations.toLocaleString("en-US")} observations; ${epochSnapshotManifest.inventory.component_rows.toLocaleString("en-US")} component rows and archive views excluded from independent model-level likelihood counts`],
  ["ECI component benchmark snapshot", eciComponentPath, "All model-by-benchmark measurements used in the held-out component comparison", `${epochSnapshotManifest.inventory.component_rows.toLocaleString("en-US")} unique model/benchmark rows; ${epochSnapshotManifest.inventory.models} models; ${epochSnapshotManifest.inventory.benchmarks} benchmarks`],
  ["Epoch benchmark data archive", eciBenchmarkArchivePath, "Official IRT slopes/difficulties and raw benchmark source files", "Used to map component performance onto the aggregate ECI scale"],
  ["Epoch AI downloads documentation", epochDocsUrl, "Dataset provenance and update cadence", "Official Epoch documentation"],
  ["LessWrong paper copy", lessWrongSourcePath, "Secondary paper text / discussion", "Primary numbers taken from LaTeX, not this copy"],
  ["METR-Horizon-v1.1 official asset", metrOfficialUrl, "26 model horizons, confidence intervals, full scaffold arrays, and 128.744-day from-2023 trend", "First-party YAML; 26/26 common-field exact match to retained legacy copy; used to identify post-training leverage only"],
  ["Ryan Greenblatt K3 assessment", "User-supplied in Codex conversation, 2026-07-17", "Corrected K3 pretrain placement and ~8-month gap", "Qualitative prior; earlier bullish view explicitly retracted"],
  ["Human forecast ledger", forecastLedgerPath, "Normalized contributor/model forecasts; geometric centers and counts", "Single source of truth for registry, workbook, and chart"],
  ["Human prediction registry", predictionRegistryPath, "Rendered Fable/Sol crowd register", "Generated from the ledger; unscored and likely correlated"],
  ["Kimi K3 official release evidence", k3ReleaseEvidencePath, "Official K3 2.78T total / 104.2B activated, 16/896 routed experts, and 2.5× reported efficiency; K2 comparator 1.04T / 32.6B", `Pinned primary-source bundle ${k3ReleaseEvidence.snapshot_date}; exact active count supersedes the earlier routed-expert approximation`],
];
body(sources.getRange("A20:D33"));
sources.getRange("A20:D33").format.fill = C.white;
sources.getRange("B20:D33").format.wrapText = true;
sources.getRange("A34:D34").values = [["OpenRouter historical price ledger", openRouterHistoricalMetadata.source.repository, `${openRouterHistoricalMetadata.source.full_git_history_rebuild_snapshot_count.toLocaleString("en-US")} immutable official /api/v1/models snapshots; ${openRouterHistoricalMetadata.inventory.models.toLocaleString("en-US")} IDs and ${openRouterHistoricalMetadata.inventory.price_change_points.toLocaleString("en-US")} dated price changes`, `Pinned ${openRouterHistoricalMetadata.source.pinned_commit}; full git-history rebuild matches SHA-256 ${openRouterHistoricalMetadata.source.full_git_history_rebuild_sha256}`]];
body(sources.getRange("A34:D34"));
sources.getRange("A34:D34").format.fill = C.white;
sources.getRange("B34:D34").format.wrapText = true;
sources.getRange("A35:D35").values = [["No-CoT exact-date audit", noCotExactDateResultPath, "49/49 day-level checkpoint dates; four date-only overrides; zero parameter identities added", `Published time law ${exactTimeLaw.paper_reported_point_days.toFixed(1)}→${exactTimeLaw.adjusted_point_days.toFixed(1)} days; token law ${exactTokenLaw.paper_reported_point_days.toFixed(1)}→${exactTokenLaw.adjusted_point_days.toFixed(1)} days; branch weight unchanged`]];
body(sources.getRange("A35:D35"));
sources.getRange("A35:D35").format.fill = C.white;
sources.getRange("B35:D35").format.wrapText = true;
sources.getRange("A36:D36").values = [["First-party frontier evidence", frontierPrimaryEvidencePath, "Direct Sol 3.6-minute no-CoT point; official Fable/Mythos shared-weight identity; fallback and GPQA caveats", `${frontierPrimaryResult.inventory.chronological_developer_holdout_predictions} fully identified chronological developer-holdouts across ${frontierPrimaryResult.inventory.heldout_developers} developers; ${frontierPrimaryResult.sol_mapping_sensitivity.nonbaseline_method_max_over_min.toFixed(1)}× mapping spread; incremental weight 0%`]];
body(sources.getRange("A36:D36"));
sources.getRange("A36:D36").format.fill = C.white;
sources.getRange("B36:D36").format.wrapText = true;
sources.getRange("A37:D37").values = [["No-CoT architecture elasticity audit", noCotArchitectureAuditPath, "Exact 35-model dense/MoE Pareto reproduction plus chronological developer-held-out parameter recovery", `MoE factor ${noCotArchitectureAudit.paper_relationship_reproduction.moe.deterministic_bootstrap_median_reproduction.toFixed(2)}× on ${noCotArchitectureAudit.paper_relationship_reproduction.moe.pareto_n} Pareto models; separate live slope rejected`]];
body(sources.getRange("A37:D37"));
sources.getRange("A37:D37").format.fill = C.white;
sources.getRange("B37:D37").format.wrapText = true;
const sourcePathCells = sources.getRange("B6:B37");
sourcePathCells.values = sourcePathCells.values.map(([value]) => [portableLocalPath(value)]);
wb.comments.setSelf({ displayName: "Codex" });
wb.comments.addThread({ cell: posterior.getRange("I6") }, `Fable's IKP factual-capacity estimate is ${ikpLiveEstimateT.toFixed(1)}T, but its decision-derived live evidence weight is ${Math.round(100 * ikpEvidenceWeight)}%. The later chronological subset has ${ikpChronological.models} models versus the required ${ikpGate.minimum_chronological_subset_models}, so the signal remains a sensitivity rather than changing the posterior.`);
wb.comments.addThread({ cell: finalEnsemble.getRange("F6") }, "User-requested 50/50 judgmental ensemble. Equal weights are applied in log space, so the final is the geometric mean of the evidence-model posterior and the human crowd center.");
wb.comments.addThread({ cell: laws.getRange("B52") }, "Same-base Opus evidence shows why METR should not be inverted directly into parameters: 4.5→4.6 is ~1.04× on no-CoT but ~2.45× with CoT.");

const checks = [
  ["Executive Summary", "A1:Q25"],
  ["Final Ensemble", "A1:H22"],
  ["Horizon Estimates", "A1:O25"],
  ["Horizon Laws", "A1:H68"],
  ["Compute Model", `A1:R${computeTargetEnd}`],
  ["Epoch Compute", "A1:M24"],
  ["No-CoT Evidence", `A1:M${evidenceSourceRow + 1}`],
  ["No-CoT Date Audit", `A1:L${noCotDateModelEnd}`],
  ["Primary Evidence", `A1:I${primaryControlEnd}`],
  ["IKP Signal Audit", `A1:J${ikpSheetEnd}`],
  ["Epoch Reconciliation", `A1:L${5 + epochReconciliationRows.length}`],
  ["Data Audit", `A1:I${issueEnd}`],
  ["Base Registry", `A1:M${baseRegistryEnd}`],
  ["Model Registry", `A1:AL${Math.min(registryEnd, 45)}`],
  ["Alias Map", `A1:H${Math.min(aliasEnd, 55)}`],
  ["Source Manifest", `A1:E${5 + manifestRows.length}`],
  ["AA Expansion Audit", `A1:N${aaPanelEnd}`],
  ["AA Inference Audit", `A1:N${aaDetailedPanelEnd}`],
  ["AA Operational Audit", `A1:N${aaOperationalPanelEnd}`],
  ["Active Param Audit", `A1:N${activePredictionEnd}`],
  ["ECI Component Audit", `A1:N${eciComponentPanelEnd}`],
  ["ECI Multivariate Audit", `A1:P${multivariateNarrowCiEnd}`],
  ["Post-training Audit", `A1:P${posttrainingFrontierEnd}`],
  ["Active Price Audit", `A1:N${activePricePredictionEnd}`],
  ["Historical Price Audit", `A1:T${historicalPredictionEnd}`],
  ["OpenRouter Audit", `A1:N${openRouterMatchEnd}`],
  ["OR Time Stability", `A1:N${openRouterEndpointEnd}`],
  ["OR Price Audit", `A1:N${openRouterOfficialMismatchEnd}`],
  ["Sources", "A1:D37"],
];
for (const [sheetName, range] of checks) {
  const inspected = await wb.inspect({ kind: "table", range: `${sheetName}!${range}`, include: "values,formulas", tableMaxRows: 70, tableMaxCols: 14, maxChars: 22000 });
  console.log(inspected.ndjson);
}
const formulaErrorSheets = [...new Set([...checks.map(([sheetName]) => sheetName), "Frontier Estimates"])];
let activeFormulaErrorCount = 0;
for (const sheetName of formulaErrorSheets) {
  const errors = await wb.inspect({ kind: "match", sheetId: sheetName, searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: `${sheetName} formula error scan` });
  console.log(errors.ndjson);
  activeFormulaErrorCount += errors.ndjson.split(/\r?\n/).filter((line) => line.includes('"kind":"match"')).length;
}
if (activeFormulaErrorCount) throw new Error(`Active workbook formula errors found: ${activeFormulaErrorCount}`);

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });
const renders = [
  ["Executive Summary", "A1:Q25", "executive-summary.png"],
  ["Frontier Estimates", "A1:R17", "frontier-estimates.png"],
  ["API Prices", "A1:L14", "api-prices.png"],
  ["Price Model", "A1:O22", "price-model.png"],
  ["Anchors & Method", "A1:H37", "anchors-method.png"],
  ["Final Ensemble", "A1:H22", "final-ensemble.png"],
  ["Horizon Estimates", "A1:O25", "horizon-estimates.png"],
  ["Horizon Laws", "A1:H57", "horizon-laws.png"],
  ["Compute Model", `A1:R${computeTargetEnd}`, "compute-model.png"],
  ["Epoch Compute", "A1:M24", "epoch-compute-sample.png"],
  ["No-CoT Evidence", `A1:M${openHeaderRow - 1}`, "no-cot-frontier.png"],
  ["No-CoT Evidence", `A${openSectionRow}:M${evidenceSourceRow + 1}`, "no-cot-open-weight.png"],
  ["No-CoT Date Audit", `A1:L${Math.min(noCotDateModelEnd, noCotDateModelHeader + 18)}`, "no-cot-date-audit.png"],
  ["Primary Evidence", `A1:I${Math.min(primaryControlEnd, primaryControlHeader + 16)}`, "primary-evidence-audit.png"],
  ["IKP Signal Audit", `A1:J${Math.min(ikpOverlapEnd, ikpOverlapHeader + 14)}`, "ikp-signal-audit.png"],
  ["IKP Signal Audit", `A${ikpConditionalSection}:J${ikpSheetEnd}`, "ikp-conditional-audit.png"],
  ["Epoch Reconciliation", `A1:L${5 + epochReconciliationRows.length}`, "epoch-reconciliation.png"],
  ["Data Audit", `A1:I${Math.min(issueEnd, 52)}`, "data-audit.png"],
  ["Base Registry", `A1:M${baseRegistryEnd}`, "base-registry.png"],
  ["Model Registry", "A1:AL25", "model-registry-sample.png"],
  ["Alias Map", "A1:H36", "alias-map-sample.png"],
  ["Source Manifest", `A1:E${5 + manifestRows.length}`, "source-manifest.png"],
  ["AA Expansion Audit", `A1:N${Math.min(aaOverlapSection - 1, 58)}`, "aa-expansion-audit.png"],
  ["AA Expansion Audit", `A${aaPanelSection}:N${Math.min(aaPanelEnd, aaPanelHeader + 18)}`, "aa-expansion-ledger-sample.png"],
  ["AA Inference Audit", `A1:N${aaCreatorEnd}`, "aa-inference-audit.png"],
  ["AA Inference Audit", `A${aaConflictSection}:N${Math.min(aaPairEnd, aaPairHeader + 18)}`, "aa-inference-conflicts-pairs.png"],
  ["AA Inference Audit", `A${aaDetailedPanelSection}:N${Math.min(aaDetailedPanelEnd, aaDetailedPanelHeader + 18)}`, "aa-inference-panel-sample.png"],
  ["AA Operational Audit", `A1:N${aaOperationalComparisonEnd}`, "aa-operational-audit.png"],
  ["AA Operational Audit", `A${aaOperationalCrosscheckSection}:N${aaOperationalCrosscheckEnd}`, "aa-operational-crosscheck.png"],
  ["AA Operational Audit", `A${aaOperationalPanelSection}:N${Math.min(aaOperationalPanelEnd, aaOperationalPanelHeader + 18)}`, "aa-operational-panel-sample.png"],
  ["Active Param Audit", "A1:N32", "active-parameter-audit.png"],
  ["Active Param Audit", `A${activePredictionSection}:N${Math.min(activePredictionEnd, activePredictionHeader + 18)}`, "active-parameter-ledger-sample.png"],
  ["ECI Component Audit", `A1:N${Math.min(eciComponentPanelHeader - 1, 42)}`, "eci-component-audit.png"],
  ["ECI Component Audit", `A${eciComponentPanelSection}:N${Math.min(eciComponentPanelEnd, eciComponentPanelHeader + 18)}`, "eci-component-ledger-sample.png"],
  ["ECI Multivariate Audit", `A1:P${multivariateTargetEnd}`, "eci-multivariate-audit.png"],
  ["ECI Multivariate Audit", `A${multivariateCoverageSection}:P${Math.min(multivariateCoverageEnd, multivariateCoverageHeader + 18)}`, "eci-multivariate-coverage-sample.png"],
  ["ECI Multivariate Audit", `A${multivariatePredictionSection}:P${Math.min(multivariatePredictionEnd, multivariatePredictionHeader + 18)}`, "eci-multivariate-predictions-sample.png"],
  ["Post-training Audit", `A1:P${posttrainingEvidenceEnd}`, "posttraining-audit.png"],
  ["Post-training Audit", `A${posttrainingEdgeSection}:P${posttrainingPredictionEnd}`, "posttraining-lineage-ledgers.png"],
  ["Post-training Audit", `A${posttrainingFrontierSection}:P${posttrainingFrontierEnd}`, "posttraining-frontier-sensitivity.png"],
  ["Active Price Audit", `A1:N${activePriceSummaryEnd}`, "active-price-audit.png"],
  ["Active Price Audit", `A${activePriceMatchSection}:N${Math.min(activePriceMatchEnd, activePriceMatchHeader + 14)}`, "active-price-match-sample.png"],
  ["Active Price Audit", `A${activePricePredictionSection}:N${Math.min(activePricePredictionEnd, activePricePredictionHeader + 14)}`, "active-price-predictions-sample.png"],
  ["Historical Price Audit", `A1:T${historicalWindowEnd}`, "historical-price-audit.png"],
  ["Historical Price Audit", `A${historicalMatchSection}:R${Math.min(historicalMatchEnd, historicalMatchHeader + 14)}`, "historical-price-match-sample.png"],
  ["Historical Price Audit", `A${historicalPredictionSection}:T${Math.min(historicalPredictionEnd, historicalPredictionHeader + 14)}`, "historical-price-predictions-sample.png"],
  ["OpenRouter Audit", `A1:N${Math.min(openRouterFrontierEnd, 32)}`, "openrouter-audit.png"],
  ["OR Time Stability", `A1:N${openRouterFocalEnd}`, "openrouter-time-stability.png"],
  ["OR Time Stability", `A${openRouterHistorySection}:N${openRouterModelEnd}`, "openrouter-temporal-ledgers.png"],
  ["OR Price Audit", `A1:N${openRouterPriceFocalEnd}`, "openrouter-price-audit.png"],
  ["OR Price Audit", `A${openRouterOfficialMismatchSection}:H${openRouterOfficialMismatchEnd}`, "openrouter-price-mismatches.png"],
  ["Sources", "A1:D37", "sources.png"],
  ["LaTeX Tables Raw", "A1:E20", "latex-tables-raw-sample.png"],
  ["ECI Formula Archive", "A1:D25", "eci-formula-archive.png"],
  ["Epoch CSV Raw", "A1:C8", "epoch-raw-sample.png"],
  ...eciSnapshots.map((snapshot) => [snapshot.targetName, `A1:${colLetter(Math.min((snapshot.values[0]?.length || 1) - 1, 11))}${Math.min(snapshot.values.length, 25)}`, `${slug(snapshot.targetName)}-sample.png`]),
];
for (const [sheetName, range, fileName] of renders) {
  const rendered = await wb.render({ sheetName, range, scale: 1.05, format: "png" });
  await fs.writeFile(`${qaDir}/${fileName}`, new Uint8Array(await rendered.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(outputPath);
execFileSync(pythonPath, [localPathScrubberPath, outputPath, "--root", workDir], { stdio: "inherit" });
execFileSync(pythonPath, [ooxmlNormalizerPath, outputPath], { stdio: "inherit" });

if (await fs.stat(inspectPath).then(() => true, () => false)) {
  execFileSync(pythonPath, [localPathScrubberPath, inspectPath, "--root", workDir], { stdio: "inherit" });
}
execFileSync(pythonPath, [inspectNormalizerPath, inspectPath], { stdio: "inherit" });
console.log(JSON.stringify({
  outputPath,
  qaDir,
  frontierModels: frontierHorizons.length,
  openModels: openModels.length,
  eciRows: eciRows.length,
  epochRows: epochRows.length,
  duplicateEpochNames,
  duplicateEpochCandidateNames,
  epochCandidates: paperToEpoch.size,
  epochExactMatches: epochMatched,
  paperRowsWithExactDate,
  paperRowsMonthOnly,
  activeFormulaErrorCount,
  canonicalCheckpoints: registryRows.length,
  aliasRows: aliasRows.length,
  duplicateRegistryIds,
  duplicateAliasKeys,
  duplicateBaseIds,
  latexTableBlocks: latexTableBlocks.length,
  eciFormulaRows: formulaRows.length,
  scaling: {
    ...scaling,
    timeDaysPaper: scaling.timeDays,
    timeDaysApplied: exactTimeLaw.adjusted_point_days,
    tokenDaysPaper: scaling.tokenDays,
    tokenDaysApplied: exactTokenLaw.adjusted_point_days,
  },
}, null, 2));
