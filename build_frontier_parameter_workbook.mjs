import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const workDir = path.dirname(fileURLToPath(import.meta.url));
const outputDir = `${workDir}/outputs/019f6c42-2d53-7743-ab07-6293e2618dd7`;
const qaDir = `${workDir}/qa/frontier_parameter`;
const outputPath = `${outputDir}/frontier_model_parameter_equivalents_2026-07-16.xlsx`;
const results = JSON.parse(await fs.readFile(`${workDir}/regression_results.json`, "utf8"));

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const frontier = workbook.worksheets.add("Frontier Estimates");
const recent = workbook.worksheets.add("Recent Open Models");
const eci = workbook.worksheets.add("ECI Calibration");
const scenarios = workbook.worksheets.add("MoE Scenarios");
const regression = workbook.worksheets.add("Regression Model");
const sources = workbook.worksheets.add("Sources");

const navy = "#243838";
const teal = "#00A5A6";
const tealDark = "#087879";
const pink = "#E03D90";
const paleTeal = "#E8F4F4";
const paleGray = "#F3F6F6";
const palePink = "#FFF4FA";
const midGray = "#D2DCDC";
const charcoal = "#3A4848";
const white = "#FFFFFF";
const amber = "#B76E00";

const asDate = (iso) => new Date(`${iso}T00:00:00Z`);
const setTitle = (sheet, range, textValue) => {
  sheet.getRange(range).merge();
  const cell = sheet.getRange(range.split(":")[0]);
  cell.values = [[textValue]];
  sheet.getRange(range).format = {
    fill: navy,
    font: { name: "Aptos Display", size: 18, bold: true, color: white },
    verticalAlignment: "center",
  };
  sheet.getRange(range).format.rowHeight = 36;
};
const setSubtitle = (sheet, range, textValue) => {
  sheet.getRange(range).merge();
  sheet.getRange(range.split(":")[0]).values = [[textValue]];
  sheet.getRange(range).format = {
    font: { name: "Aptos", size: 10, color: charcoal },
    wrapText: true,
    verticalAlignment: "center",
  };
};
const styleHeaders = (range) => {
  range.format = {
    fill: tealDark,
    font: { name: "Aptos", size: 9, bold: true, color: white },
    wrapText: true,
    verticalAlignment: "center",
    borders: { bottom: { style: "medium", color: teal } },
  };
  range.format.rowHeight = 34;
};
const styleBody = (range) => {
  range.format = {
    font: { name: "Aptos", size: 9, color: charcoal },
    verticalAlignment: "center",
    borders: { insideHorizontal: { style: "thin", color: midGray } },
  };
};
const sectionHeader = (sheet, range, textValue) => {
  sheet.getRange(range).merge();
  sheet.getRange(range.split(":")[0]).values = [[textValue]];
  sheet.getRange(range).format = {
    fill: paleTeal,
    font: { name: "Aptos", size: 11, bold: true, color: tealDark },
    borders: { bottom: { style: "medium", color: teal } },
  };
};

for (const sheet of [summary, frontier, recent, eci, scenarios, regression, sources]) {
  sheet.showGridLines = false;
}

// Regression Model sheet first because the other sheets reference its cells.
setTitle(regression, "A1:J1", "Date-aware parameter-equivalence model");
setSubtitle(
  regression,
  "A3:J4",
  "Two frontier regressions are fit separately: Artificial Analysis Intelligence Index v4.1 on 50 open-weight releases in the last 12 months, and Epoch ECI on an 88-model 2023–2026 calibration panel. The reported estimate is their geometric ensemble. It is an open-weight-equivalent scale, not a claim about undisclosed physical weights.",
);

regression.getRange("A6:C6").values = [["AA v4.1 coefficient", "Value", "Interpretation"]];
regression.getRange("E6:G6").values = [["ECI coefficient", "Value", "Interpretation"]];
styleHeaders(regression.getRange("A6:C6"));
styleHeaders(regression.getRange("E6:G6"));

const aaFit = results.fit;
const eciFit = results.eci.fit;
const aaCv = results.model_comparisons[0];
const eciCv = results.eci.model_comparisons[0];
const aaMoe = aaFit.beta.length > 4 ? aaFit.beta[4] : 0;
const eciMoe = eciFit.beta.length > 4 ? eciFit.beta[4] : 0;
const coefficientLabels = [
  "κ: MoE dormant-capacity weight",
  "Raw intercept",
  "90th-frontier residual shift",
  "Frontier intercept",
  "Score points / parameter doubling",
  "Vintage points / year",
  "Reasoning uplift (fixed)",
  "MoE indicator uplift",
  "Residual σ (score points)",
  "Equivalent parameter doublings / year",
  "Vintage-equivalent doubling time (months)",
  "LOFO median absolute error (doublings)",
  "LOFO 80th-percentile error (doublings)",
];
const interpretations = [
  "0 = active count only; 1 = total count only",
  "Average-open-model intercept at 2026-01-01",
  "Raises the regression to the open frontier",
  "Raw intercept + frontier shift",
  "Slope on log2 effective parameters",
  "Release-vintage effect; not purely causal algorithms",
  "Median paired reasoning/non-reasoning uplift",
  "Architecture intercept for MoE (target scenario = 1)",
  "In-sample residual dispersion",
  "Vintage coefficient divided by scale coefficient",
  "12 divided by equivalent doublings/year",
  "Leave-one-family-out inversion error",
  "Used as the structural stress-band half-width",
];

const aaValues = [
  aaFit.kappa,
  aaFit.beta[0],
  aaFit.frontier_residual_shift_p90,
  aaFit.beta[0] + aaFit.frontier_residual_shift_p90,
  aaFit.beta[1],
  aaFit.beta[2],
  aaFit.beta[3],
  aaMoe,
  aaFit.sigma_score,
  aaFit.parameter_doublings_per_year,
  aaFit.algorithmic_doubling_months,
  aaCv.median_abs_log2_scale,
  aaCv.p80_abs_log2_scale,
];
const eciValues = [
  eciFit.kappa,
  eciFit.beta[0],
  eciFit.frontier_residual_shift_p90,
  eciFit.beta[0] + eciFit.frontier_residual_shift_p90,
  eciFit.beta[1],
  eciFit.beta[2],
  eciFit.beta[3],
  eciMoe,
  eciFit.sigma_score,
  eciFit.parameter_doublings_per_year,
  eciFit.algorithmic_doubling_months,
  eciCv.median_abs_log2_scale,
  eciCv.p80_abs_log2_scale,
];
regression.getRange("A7:C19").values = coefficientLabels.map((label, i) => [label, aaValues[i], interpretations[i]]);
regression.getRange("E7:G19").values = coefficientLabels.map((label, i) => [label, eciValues[i], interpretations[i]]);
styleBody(regression.getRange("A7:C19"));
styleBody(regression.getRange("E7:G19"));
regression.getRange("B7:B19").format.numberFormat = "0.000";
regression.getRange("F7:F19").format.numberFormat = "0.000";
regression.getRange("A7:A19").format.font = { name: "Aptos", size: 9, bold: true, color: charcoal };
regression.getRange("E7:E19").format.font = { name: "Aptos", size: 9, bold: true, color: charcoal };

regression.getRange("I6:J6").values = [["Core assumption", "Value"]];
styleHeaders(regression.getRange("I6:J6"));
regression.getRange("I7:J14").values = [
  ["As-of date", asDate(results.as_of)],
  ["Vintage origin", asDate(results.date_origin)],
  ["Open-frontier quantile", 0.90],
  ["Recent-open median total/active ratio", aaFit.observed_moe_ratio_median],
  ["Cluster bootstrap draws", results.bootstrap_draws],
  ["Structural half-width (doublings)", results.structural_log2_halfwidth],
  ["Fixed reasoning uplift (points)", results.fixed_reasoning_effect],
  ["Primary estimate", "Geometric mean of AA and ECI frontier equivalents"],
];
styleBody(regression.getRange("I7:J14"));
regression.getRange("J7:J8").format.numberFormat = "yyyy-mm-dd";
regression.getRange("J9").format.numberFormat = "0%";
regression.getRange("J10").format.numberFormat = "0.0x";
regression.getRange("J11").format.numberFormat = "#,##0";
regression.getRange("J12:J13").format.numberFormat = "0.00";

sectionHeader(regression, "A22:J22", "Model-selection and validation");
const compHeaders = ["Index", "Extra controls", "κ", "Score RMSE", "Score MAE", "Median abs error (doublings)", "80th abs error", "90th abs error", "N"];
regression.getRange("A24:I24").values = [compHeaders];
styleHeaders(regression.getRange("A24:I24"));
const comparisonRows = [
  ...results.model_comparisons.map((r) => ["AA v4.1", r.spec.length ? r.spec.join(" + ") : "None", r.kappa, r.rmse_score, r.mae_score, r.median_abs_log2_scale, r.p80_abs_log2_scale, r.p90_abs_log2_scale, r.n]),
  ...results.eci.model_comparisons.map((r) => ["ECI", r.spec.length ? r.spec.join(" + ") : "None", r.kappa, r.rmse_score, r.mae_score, r.median_abs_log2_scale, r.p80_abs_log2_scale, r.p90_abs_log2_scale, r.n]),
];
regression.getRange(`A25:I${24 + comparisonRows.length}`).values = comparisonRows;
styleBody(regression.getRange(`A25:I${24 + comparisonRows.length}`));
regression.getRange(`C25:H${24 + comparisonRows.length}`).format.numberFormat = "0.00";
regression.getRange(`I25:I${24 + comparisonRows.length}`).format.numberFormat = "#,##0";

sectionHeader(regression, "A34:J34", "Equations and interpretation");
regression.getRange("A36:J36").merge();
regression.getRange("A36").values = [["Effective scale: P_eff = P_active × (P_total / P_active)^κ. Capability = frontier intercept + β·log2(P_eff) + γ·vintage_years + 6·reasoning + δ_MoE."]];
regression.getRange("A37:J38").merge();
regression.getRange("A37").values = [["Inversion returns the parameter scale of a 90th-percentile open-weight model of the same release vintage and reasoning status. The MoE scenario converts that dense-equivalent scale into active and total counts using the observed 18.6:1 median sparsity ratio."]];
regression.getRange("A39:J40").merge();
regression.getRange("A39").values = [["The 80% fit interval reflects score uncertainty and family-clustered coefficient bootstrap. The much wider structural stress band uses the leave-one-family-out 80th-percentile inversion error and is the more honest guide for unknown proprietary architectures."]];
regression.getRange("A36:J40").format = { fill: paleGray, font: { name: "Aptos", size: 9, color: charcoal }, wrapText: true, verticalAlignment: "center" };

regression.getRange("A:A").format.columnWidth = 34;
regression.getRange("B:B").format.columnWidth = 15;
regression.getRange("C:C").format.columnWidth = 38;
regression.getRange("D:D").format.columnWidth = 14;
regression.getRange("E:E").format.columnWidth = 34;
regression.getRange("F:F").format.columnWidth = 15;
regression.getRange("G:G").format.columnWidth = 38;
regression.getRange("H:H").format.columnWidth = 14;
regression.getRange("I:I").format.columnWidth = 34;
regression.getRange("J:J").format.columnWidth = 28;
regression.getRange("A24:I24").format.rowHeight = 44;
regression.freezePanes.freezeRows(6);

// Frontier estimates.
setTitle(frontier, "A1:V1", "Current frontier-model parameter equivalents");
setSubtitle(frontier, "A2:V3", "Central values are a geometric ensemble of the AA v4.1 and Epoch ECI date-aware frontier regressions. Active/total columns assume an 18.6:1 MoE ratio; use the MoE Scenarios sheet for other architectures. Fable is a model cascade and should not be read as one model's size.");
const frontierHeaders = [
  "Release Date", "Model", "AA v4.1", "AA Output Tokens (M)", "Reasoning Config", "ECI", "ECI 90% Low", "ECI 90% High", "Classification",
  "AA Dense-Equivalent (B)", "ECI Dense-Equivalent (B)", "Ensemble Dense-Equivalent (B)", "MoE Ratio Scenario", "Implied Active (B)", "Implied Total (B)",
  "Fit 80% Total Low (B)", "Fit 80% Total High (B)", "Structural Low (B)", "Structural High (B)", "AA Source", "Epoch Source", "Notes",
];
frontier.getRange("A4:V4").values = [frontierHeaders];
styleHeaders(frontier.getRange("A4:V4"));
const frontierStart = 5;
const frontierRows = results.frontier_predictions.map((p) => [
  asDate(p.release_date), p.model, p.aa_score, p.aa_output_m, p.config, p.eci, p.eci_low, p.eci_high, p.classification,
  null, null, null, null, null, null,
  p.moe_total_interval.p10, p.moe_total_interval.p90, null, null,
  p.aa_source, p.epoch_source,
  p.classification.includes("cascade") ? "Exclude from literal single-model interpretation." : (p.model === "Grok 4.5" ? "Founder-attributed 1.5T total parameter claim is an external validation point; active count is unknown." : "Undisclosed physical count; regression equivalent only."),
]);
const frontierEnd = frontierStart + frontierRows.length - 1;
frontier.getRange(`A${frontierStart}:V${frontierEnd}`).values = frontierRows;
for (let row = frontierStart; row <= frontierEnd; row += 1) {
  frontier.getRange(`J${row}`).formulas = [[`=POWER(2,(C${row}-'Regression Model'!$B$10-'Regression Model'!$B$12*((A${row}-'Regression Model'!$J$8)/365.25)-'Regression Model'!$B$13-'Regression Model'!$B$14)/'Regression Model'!$B$11)`]];
  frontier.getRange(`K${row}`).formulas = [[`=POWER(2,(F${row}-'Regression Model'!$F$10-'Regression Model'!$F$12*((A${row}-'Regression Model'!$J$8)/365.25)-'Regression Model'!$F$13-'Regression Model'!$F$14)/'Regression Model'!$F$11)`]];
  frontier.getRange(`L${row}`).formulas = [[`=SQRT(J${row}*K${row})`]];
  frontier.getRange(`M${row}`).formulas = [[`='Regression Model'!$J$10`]];
  frontier.getRange(`N${row}`).formulas = [[`=SQRT((J${row}/POWER(M${row},'Regression Model'!$B$7))*(K${row}/POWER(M${row},'Regression Model'!$F$7)))`]];
  frontier.getRange(`O${row}`).formulas = [[`=N${row}*M${row}`]];
  frontier.getRange(`R${row}`).formulas = [[`=O${row}/POWER(2,'Regression Model'!$J$12)`]];
  frontier.getRange(`S${row}`).formulas = [[`=O${row}*POWER(2,'Regression Model'!$J$12)`]];
}
styleBody(frontier.getRange(`A${frontierStart}:V${frontierEnd}`));
frontier.getRange(`A${frontierStart}:A${frontierEnd}`).format.numberFormat = "yyyy-mm-dd";
frontier.getRange(`C${frontierStart}:C${frontierEnd}`).format.numberFormat = "0";
frontier.getRange(`D${frontierStart}:D${frontierEnd}`).format.numberFormat = "#,##0";
frontier.getRange(`F${frontierStart}:H${frontierEnd}`).format.numberFormat = "0.0";
frontier.getRange(`J${frontierStart}:S${frontierEnd}`).format.numberFormat = "#,##0.0";
frontier.getRange(`M${frontierStart}:M${frontierEnd}`).format.numberFormat = "0.0x";
frontier.getRange(`E${frontierStart}:I${frontierEnd}`).format.wrapText = true;
frontier.getRange(`T${frontierStart}:V${frontierEnd}`).format.wrapText = true;
frontier.getRange(`O${frontierStart}:O${frontierEnd}`).format.font = { name: "Aptos", size: 9, bold: true, color: navy };
frontier.getRange(`I${frontierStart}:I${frontierEnd}`).conditionalFormats.add("containsText", { text: "cascade", format: { fill: "#FFF0D9", font: { color: amber, bold: true } } });
frontier.getRange("A:A").format.columnWidth = 13;
frontier.getRange("B:B").format.columnWidth = 24;
frontier.getRange("C:D").format.columnWidth = 12;
frontier.getRange("E:E").format.columnWidth = 22;
frontier.getRange("F:H").format.columnWidth = 12;
frontier.getRange("I:I").format.columnWidth = 26;
frontier.getRange("J:S").format.columnWidth = 16;
frontier.getRange("T:U").format.columnWidth = 39;
frontier.getRange("V:V").format.columnWidth = 38;
frontier.freezePanes.freezeRows(4);
frontier.tables.add(`A4:V${frontierEnd}`, true, "FrontierEstimatesTable").style = "TableStyleMedium2";

// Recent open-weight data.
setTitle(recent, "A1:Q1", "Open-weight model parameter sizes: last 12 months");
setSubtitle(recent, "A2:Q3", "One canonical AA Intelligence Index v4.1 endpoint per checkpoint, released 2025-07-16 through 2026-07-16. Total and active parameters are separate for MoE models. Estimated-score rows receive half weight; models sharing a base lineage are clustered in validation and bootstrap.");
const recentHeaders = ["Release Date", "Model", "Total Params (B)", "Active Params (B)", "Total / Active", "Architecture", "Reasoning", "Lineage Cluster", "AA v4.1", "AA Score Estimated", "AA Output Tokens (M)", "Parameter Source", "AA Source", "Effective Scale (B)", "As-of Adjusted Score", "Fitted Score", "Residual"];
recent.getRange("A4:Q4").values = [recentHeaders];
styleHeaders(recent.getRange("A4:Q4"));
const recentStart = 5;
const recentRows = results.open_models.map((r) => [
  asDate(r.release_date), r.model, r.total_b, r.active_b, null, r.architecture, r.reasoning ? "Yes" : "No", r.family,
  r.score, r.estimated ? "Yes" : "No", r.output_m, r.parameter_source, r.source, null, null, null, null,
]);
const recentEnd = recentStart + recentRows.length - 1;
recent.getRange(`A${recentStart}:Q${recentEnd}`).values = recentRows;
for (let row = recentStart; row <= recentEnd; row += 1) {
  recent.getRange(`E${row}`).formulas = [[`=C${row}/D${row}`]];
  recent.getRange(`N${row}`).formulas = [[`=D${row}*POWER(E${row},'Regression Model'!$B$7)`]];
  recent.getRange(`O${row}`).formulas = [[`=I${row}+'Regression Model'!$B$12*('Regression Model'!$J$7-A${row})/365.25`]];
  recent.getRange(`P${row}`).formulas = [[`='Regression Model'!$B$8+'Regression Model'!$B$11*LOG(N${row},2)+'Regression Model'!$B$12*((A${row}-'Regression Model'!$J$8)/365.25)+'Regression Model'!$B$13*IF(G${row}="Yes",1,0)+'Regression Model'!$B$14*IF(E${row}>1.05,1,0)`]];
  recent.getRange(`Q${row}`).formulas = [[`=I${row}-P${row}`]];
}
styleBody(recent.getRange(`A${recentStart}:Q${recentEnd}`));
recent.getRange(`A${recentStart}:A${recentEnd}`).format.numberFormat = "yyyy-mm-dd";
recent.getRange(`C${recentStart}:E${recentEnd}`).format.numberFormat = "#,##0.00";
recent.getRange(`I${recentStart}:I${recentEnd}`).format.numberFormat = "0";
recent.getRange(`K${recentStart}:K${recentEnd}`).format.numberFormat = "#,##0";
recent.getRange(`N${recentStart}:N${recentEnd}`).format.numberFormat = "#,##0.00";
recent.getRange(`O${recentStart}:Q${recentEnd}`).format.numberFormat = "0.00";
recent.getRange(`L${recentStart}:M${recentEnd}`).format.wrapText = true;
recent.getRange(`J${recentStart}:J${recentEnd}`).conditionalFormats.add("containsText", { text: "Yes", format: { fill: "#FFF0D9", font: { color: amber } } });
recent.getRange("A:A").format.columnWidth = 13;
recent.getRange("B:B").format.columnWidth = 34;
recent.getRange("C:E").format.columnWidth = 13;
recent.getRange("F:F").format.columnWidth = 26;
recent.getRange("G:J").format.columnWidth = 14;
recent.getRange("K:K").format.columnWidth = 16;
recent.getRange("L:M").format.columnWidth = 44;
recent.getRange("N:Q").format.columnWidth = 16;
recent.freezePanes.freezeRows(4);
recent.tables.add(`A4:Q${recentEnd}`, true, "RecentOpenModelsTable").style = "TableStyleMedium2";

// Extended ECI calibration data.
setTitle(eci, "A1:Q1", "Extended open-weight calibration panel: Epoch ECI");
setSubtitle(eci, "A2:Q3", "The 2023–2026 panel identifies the vintage effect over multiple generations. ECI is a broad latent-capability index and an upper envelope across available evaluation settings; it is used as a robustness model, not mixed directly into AA scores.");
const eciHeaders = ["Release Date", "Model", "Total Params (B)", "Active Params (B)", "Total / Active", "Architecture", "Reasoning", "Lineage Cluster", "ECI", "ECI 90% Low", "ECI 90% High", "CI Width", "Epoch Model Source", "Effective Scale (B)", "Fitted Score", "Residual", "Parameter Source Note"];
eci.getRange("A4:Q4").values = [eciHeaders];
styleHeaders(eci.getRange("A4:Q4"));
const eciStart = 5;
const sortedEciModels = [...results.eci.open_models].sort((a, b) => b.release_date.localeCompare(a.release_date) || a.model.localeCompare(b.model));
const eciRows = sortedEciModels.map((r) => [
  asDate(r.release_date), r.model, r.total_b, r.active_b, null, r.architecture, r.reasoning ? "Yes" : "No", r.family,
  r.score, r.ci_low, r.ci_high, r.ci_width, r.source, null, null, null,
  "Parameter mapping from official model names/cards; Epoch AI Models dataset provides the common metadata cross-check.",
]);
const eciEnd = eciStart + eciRows.length - 1;
eci.getRange(`A${eciStart}:Q${eciEnd}`).values = eciRows;
for (let row = eciStart; row <= eciEnd; row += 1) {
  eci.getRange(`E${row}`).formulas = [[`=C${row}/D${row}`]];
  eci.getRange(`N${row}`).formulas = [[`=D${row}*POWER(E${row},'Regression Model'!$F$7)`]];
  eci.getRange(`O${row}`).formulas = [[`='Regression Model'!$F$8+'Regression Model'!$F$11*LOG(N${row},2)+'Regression Model'!$F$12*((A${row}-'Regression Model'!$J$8)/365.25)+'Regression Model'!$F$13*IF(G${row}="Yes",1,0)+'Regression Model'!$F$14*IF(E${row}>1.05,1,0)`]];
  eci.getRange(`P${row}`).formulas = [[`=I${row}-O${row}`]];
}
styleBody(eci.getRange(`A${eciStart}:Q${eciEnd}`));
eci.getRange(`A${eciStart}:A${eciEnd}`).format.numberFormat = "yyyy-mm-dd";
eci.getRange(`C${eciStart}:E${eciEnd}`).format.numberFormat = "#,##0.00";
eci.getRange(`I${eciStart}:P${eciEnd}`).format.numberFormat = "0.00";
eci.getRange(`M${eciStart}:M${eciEnd}`).format.wrapText = true;
eci.getRange(`Q${eciStart}:Q${eciEnd}`).format.wrapText = true;
eci.getRange("A:A").format.columnWidth = 13;
eci.getRange("B:B").format.columnWidth = 38;
eci.getRange("C:E").format.columnWidth = 13;
eci.getRange("F:H").format.columnWidth = 18;
eci.getRange("I:L").format.columnWidth = 13;
eci.getRange("M:M").format.columnWidth = 44;
eci.getRange("N:P").format.columnWidth = 16;
eci.getRange("Q:Q").format.columnWidth = 42;
eci.freezePanes.freezeRows(4);
eci.tables.add(`A4:Q${eciEnd}`, true, "ECICalibrationTable").style = "TableStyleMedium2";

// MoE architecture scenarios.
setTitle(scenarios, "A1:G1", "Architecture scenarios for frontier equivalents");
setSubtitle(scenarios, "A2:G3", "Physical parameter counts cannot be recovered from benchmark scores alone. This table converts the ensemble dense-equivalent scale into active and total parameters at several hypothetical total-to-active ratios. Ratio 1 is a dense model; 18.6 is the median among the recent open MoE comparison set.");
const scenarioHeaders = ["Model", "AA v4.1", "Release Date", "Total / Active Ratio", "Implied Active (B)", "Implied Total (B)", "Interpretation"];
scenarios.getRange("A4:G4").values = [scenarioHeaders];
styleHeaders(scenarios.getRange("A4:G4"));
const ratios = [1, 8, aaFit.observed_moe_ratio_median, 32, 64];
const scenarioRows = [];
for (let modelIndex = 0; modelIndex < results.frontier_predictions.length; modelIndex += 1) {
  const p = results.frontier_predictions[modelIndex];
  for (const ratio of ratios) {
    scenarioRows.push([p.model, p.aa_score, asDate(p.release_date), ratio, null, null, ratio === 1 ? "Dense-equivalent" : (ratio === aaFit.observed_moe_ratio_median ? "Observed recent-open median MoE ratio" : "Hypothetical MoE scenario")]);
  }
}
const scenarioStart = 5;
const scenarioEnd = scenarioStart + scenarioRows.length - 1;
scenarios.getRange(`A${scenarioStart}:G${scenarioEnd}`).values = scenarioRows;
for (let i = 0; i < scenarioRows.length; i += 1) {
  const row = scenarioStart + i;
  const frontierRow = frontierStart + Math.floor(i / ratios.length);
  scenarios.getRange(`E${row}`).formulas = [[`=SQRT(('Frontier Estimates'!$J$${frontierRow}/POWER(D${row},'Regression Model'!$B$7))*('Frontier Estimates'!$K$${frontierRow}/POWER(D${row},'Regression Model'!$F$7)))`]];
  scenarios.getRange(`F${row}`).formulas = [[`=E${row}*D${row}`]];
}
styleBody(scenarios.getRange(`A${scenarioStart}:G${scenarioEnd}`));
scenarios.getRange(`B${scenarioStart}:B${scenarioEnd}`).format.numberFormat = "0";
scenarios.getRange(`C${scenarioStart}:C${scenarioEnd}`).format.numberFormat = "yyyy-mm-dd";
scenarios.getRange(`D${scenarioStart}:D${scenarioEnd}`).format.numberFormat = "0.0x";
scenarios.getRange(`E${scenarioStart}:F${scenarioEnd}`).format.numberFormat = "#,##0.0";
scenarios.getRange(`G${scenarioStart}:G${scenarioEnd}`).format.wrapText = true;
scenarios.getRange(`D${scenarioStart}:D${scenarioEnd}`).conditionalFormats.add("cellIs", { operator: "equal", formula: aaFit.observed_moe_ratio_median, format: { fill: paleTeal, font: { bold: true, color: tealDark } } });
scenarios.getRange("A:A").format.columnWidth = 28;
scenarios.getRange("B:F").format.columnWidth = 16;
scenarios.getRange("G:G").format.columnWidth = 38;
scenarios.freezePanes.freezeRows(4);
scenarios.tables.add(`A4:G${scenarioEnd}`, true, "MoEScenariosTable").style = "TableStyleMedium2";

// Summary dashboard.
setTitle(summary, "A1:P1", "Frontier model parameter equivalents — 2026-07-16");
setSubtitle(summary, "A3:P4", "Performance does not uniquely identify literal hidden weights. These are date-, reasoning-, and MoE-sparsity-conditioned open-weight equivalents, estimated from 50 recent open models on AA v4.1 and cross-checked against an 88-model Epoch ECI panel.");

summary.getRange("A6:B6").values = [["Dataset / assumption", "Value"]];
styleHeaders(summary.getRange("A6:B6"));
summary.getRange("A7:A12").values = [
  ["As-of date"], ["Recent open checkpoints"], ["Extended ECI calibration"], ["Current open AA ceiling"], ["Recent-open median MoE ratio"], ["Structural stress half-width"],
];
summary.getRange("B7:B12").values = [[asDate(results.as_of)], [results.n_open_models], [results.eci.n_open_models], [51], [aaFit.observed_moe_ratio_median], [results.structural_log2_halfwidth]];
styleBody(summary.getRange("A7:B12"));
summary.getRange("A7:A12").format.font = { name: "Aptos", size: 9, bold: true, color: charcoal };
summary.getRange("B7:B12").format.font = { name: "Aptos", size: 10, bold: true, color: navy };
summary.getRange("B7").format.numberFormat = "yyyy-mm-dd";
summary.getRange("B8:B10").format.numberFormat = "#,##0";
summary.getRange("B11").format.numberFormat = "0.0x";
summary.getRange("B12").format.numberFormat = "0.00 \"doublings\"";

sectionHeader(summary, "A15:F15", "Selected pure-model equivalents at the 18.6:1 MoE scenario");
summary.getRange("A16:F16").values = [["Model", "AA v4.1", "Implied Active (B)", "Implied Total (B)", "Fit 80% Total (B)", "Structural Stress Band (B)"]];
styleHeaders(summary.getRange("A16:F16"));
const selectedModels = ["GPT-5.6 Sol", "Claude Opus 4.8", "GPT-5.5", "Claude Opus 4.7", "Grok 4.5", "Gemini 3.5 Flash"];
for (let i = 0; i < selectedModels.length; i += 1) {
  const model = selectedModels[i];
  const sourceIndex = results.frontier_predictions.findIndex((p) => p.model === model);
  const sourceRow = frontierStart + sourceIndex;
  const row = 17 + i;
  summary.getRange(`A${row}`).formulas = [[`='Frontier Estimates'!B${sourceRow}`]];
  summary.getRange(`B${row}`).formulas = [[`='Frontier Estimates'!C${sourceRow}`]];
  summary.getRange(`C${row}`).formulas = [[`='Frontier Estimates'!N${sourceRow}`]];
  summary.getRange(`D${row}`).formulas = [[`='Frontier Estimates'!O${sourceRow}`]];
  summary.getRange(`E${row}`).formulas = [[`=TEXT('Frontier Estimates'!P${sourceRow},"#,##0")&"–"&TEXT('Frontier Estimates'!Q${sourceRow},"#,##0")`]];
  summary.getRange(`F${row}`).formulas = [[`=TEXT('Frontier Estimates'!R${sourceRow},"#,##0")&"–"&TEXT('Frontier Estimates'!S${sourceRow},"#,##0")`]];
}
styleBody(summary.getRange("A17:F22"));
summary.getRange("B17:B22").format.numberFormat = "0";
summary.getRange("C17:D22").format.numberFormat = "#,##0";
summary.getRange("D17:D22").format.font = { name: "Aptos", size: 10, bold: true, color: navy };

const chart = summary.charts.add("bar", { title: "Open-weight-equivalent total parameters (B)", hasLegend: false });
const chartSeries = chart.series.add("Implied total (B)");
chartSeries.categoryFormula = "'Summary'!$A$17:$A$22";
chartSeries.formula = "'Summary'!$D$17:$D$22";
chartSeries.fill = teal;
chart.title = "Frontier open-weight-equivalent total parameters (B)";
chart.hasLegend = false;
chart.setPosition("H6", "P22");

summary.getRange("A25:P26").merge();
summary.getRange("A25").values = [["Headline: under the median recent-open MoE sparsity scenario, GPT-5.6 Sol is about 133B active / 2.5T total open-weight-equivalent parameters. Claude Opus 4.8 is about 95B active / 1.8T total. These are system-equivalent scales at the tested reasoning budgets, not physical architecture disclosures."]];
summary.getRange("A25:P26").format = { fill: paleTeal, font: { name: "Aptos", size: 10, bold: true, color: navy }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: teal } } };
summary.getRange("A28:P30").merge();
summary.getRange("A28").values = [["Uncertainty: the fit-only 80% intervals come from 2,500 family-clustered bootstraps plus benchmark-score uncertainty. Leave-one-family-out validation is much harsher: its 80th-percentile error is ±2.52 parameter doublings (about ×5.7). The structural stress band reflects that model risk and should be preferred for decisions."]];
summary.getRange("A28:P30").format = { fill: palePink, font: { name: "Aptos", size: 9, italic: true, color: charcoal }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: pink } } };
summary.getRange("A32:P34").merge();
summary.getRange("A32").values = [["Validation: xAI's founder-attributed claim that Grok 4.5 is based on a 1.5T-parameter foundation model exceeds the 0.6T central regression equivalent but lies inside the 0.1–3.5T structural band. This is evidence that benchmark inversion is informative only at order-of-magnitude scale."]];
summary.getRange("A32:P34").format = { fill: paleGray, font: { name: "Aptos", size: 9, color: charcoal }, wrapText: true, verticalAlignment: "center" };
summary.getRange("A:A").format.columnWidth = 30;
summary.getRange("B:B").format.columnWidth = 16;
summary.getRange("C:D").format.columnWidth = 17;
summary.getRange("E:F").format.columnWidth = 22;
summary.getRange("G:G").format.columnWidth = 3;
summary.getRange("H:P").format.columnWidth = 14;
summary.freezePanes.freezeRows(4);

// Source register.
setTitle(sources, "A1:D1", "Source register and scope notes");
setSubtitle(sources, "A2:D3", "URLs are plain text for auditability. Model-by-model parameter and score sources also appear on the Recent Open Models and Frontier Estimates sheets.");
sources.getRange("A5:D5").values = [["Source", "URL", "Role in analysis", "Snapshot / note"]];
styleHeaders(sources.getRange("A5:D5"));
const sourceRows = [
  ["Artificial Analysis Intelligence Index methodology", "https://artificialanalysis.ai/methodology/intelligence-benchmarking", "Primary recent-model outcome; v4.1, nine evaluations", "Snapshot 2026-07-16"],
  ["Artificial Analysis model leaderboard", "https://artificialanalysis.ai/leaderboards/models", "Current AA v4.1 model scores", "Do not mix with launch-era older index versions"],
  ["Epoch ECI methodology", "https://epoch.ai/data/eci-documentation/methodology", "Robustness outcome and latent-capability methodology", "Snapshot 2026-07-16"],
  ["Epoch ECI preprocessing / benchmarks", "https://epoch.ai/data/eci-documentation/data", "Documents upper-envelope score selection across settings", "More than 50 benchmarks in current data"],
  ["Epoch benchmarked-model CSV", "https://epoch.ai/data/benchmarked_models.csv", "Release dates, ECI scores and 90% CIs", "Snapshot 2026-07-16"],
  ["Epoch AI Models dataset", "https://epoch.ai/data/ai-models", "Parameter, training-compute and metadata cross-check", "Updated 2026-07-14"],
  ["Algorithmic progress in language models", "https://arxiv.org/abs/2403.05812", "Historical sensitivity prior; not imposed on the regression", "Perplexity-based historical result, not modern reasoning benchmarks"],
  ["Observational Scaling Laws", "https://openreview.net/forum?id=On5WIN7xyD", "Supports low-dimensional capability scaling with family efficiency differences", "Used as conceptual support only"],
  ["OpenAI GPT-5.5 release", "https://openai.com/index/introducing-gpt-5-5/", "Same underlying model for GPT-5.5 and Pro; Pro adds parallel test-time compute", "Shows why performance cannot identify weights without inference budget"],
  ["Anthropic Fable 5 / Mythos 5", "https://www.anthropic.com/news/claude-fable-5-mythos-5", "Confirms Fable fallback/cascade behavior", "Fable excluded from literal single-model reading"],
  ["Grok 4.5 founder claim", "https://x.com/elonmusk/status/2071184354756477041", "External validation point: 1.5T V9 foundation model", "Founder-attributed; total only, active unknown"],
  ["DeepSeek V4 Pro model card", "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro", "1.6T total / 49B active parameter anchor", "Primary model card"],
  ["GLM-5 model card", "https://huggingface.co/zai-org/GLM-5", "744B total / 40B active parameter anchor", "Primary model card"],
  ["Kimi K2.6 model card", "https://huggingface.co/moonshotai/Kimi-K2.6", "1T-class total / 32B active parameter anchor", "Primary model card"],
  ["OpenAI gpt-oss announcement", "https://openai.com/index/introducing-gpt-oss/", "Exact total and active parameter counts", "Primary source"],
  ["Qwen3.5 model card", "https://huggingface.co/Qwen/Qwen3.5-397B-A17B", "397B total / 17B active parameter anchor", "Primary model card"],
  ["MiniMax M2 model card", "https://huggingface.co/MiniMaxAI/MiniMax-M2", "230B total / 10B active parameter anchor", "Primary model card"],
];
sources.getRange(`A6:D${5 + sourceRows.length}`).values = sourceRows;
styleBody(sources.getRange(`A6:D${5 + sourceRows.length}`));
sources.getRange(`B6:D${5 + sourceRows.length}`).format.wrapText = true;
sources.getRange("A:A").format.columnWidth = 38;
sources.getRange("B:B").format.columnWidth = 54;
sources.getRange("C:C").format.columnWidth = 50;
sources.getRange("D:D").format.columnWidth = 42;
sources.freezePanes.freezeRows(5);
sources.tables.add(`A5:D${5 + sourceRows.length}`, true, "SourcesTable").style = "TableStyleMedium2";

// Comments on key judgment cells.
workbook.comments.setSelf({ displayName: "User" });
workbook.comments.addThread({ cell: regression.getRange("J10") }, "Observed median total-to-active parameter ratio among MoE rows in the 12-month open-weight comparison set. Change this assumption through the MoE Scenarios sheet rather than treating it as a disclosure about proprietary models.");
workbook.comments.addThread({ cell: regression.getRange("J12") }, "Structural half-width is the worse of the AA and ECI leave-one-family-out 80th-percentile absolute inversion errors. It captures model-family misspecification that ordinary coefficient intervals miss.");
workbook.comments.addThread({ cell: regression.getRange("B13") }, "Reasoning uplift is fixed at the robust median of paired reasoning versus non-reasoning AA configurations, reducing collinearity between release date and the rise of reasoning models.");

// Compact verification before export.
const summaryInspect = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:P34",
  include: "values,formulas",
  tableMaxRows: 34,
  tableMaxCols: 16,
  maxChars: 12000,
});
console.log(summaryInspect.ndjson);
const frontierInspect = await workbook.inspect({
  kind: "table",
  range: `Frontier Estimates!A4:S${frontierEnd}`,
  include: "values,formulas",
  tableMaxRows: 16,
  tableMaxCols: 19,
  maxChars: 12000,
});
console.log(frontierInspect.ndjson);
const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errorScan.ndjson);

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });
const previewSpecs = [
  ["Summary", "A1:P34", "summary.png"],
  ["Frontier Estimates", `A1:V${frontierEnd}`, "frontier.png"],
  ["Recent Open Models", "A1:Q22", "recent-open.png"],
  ["ECI Calibration", "A1:Q22", "eci-calibration.png"],
  ["MoE Scenarios", "A1:G24", "moe-scenarios.png"],
  ["Regression Model", "A1:J40", "regression-model.png"],
  ["Sources", `A1:D${5 + sourceRows.length}`, "sources.png"],
];
for (const [sheetName, range, fileName] of previewSpecs) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(`${qaDir}/${fileName}`, new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, qaDir, frontierEnd, recentEnd, eciEnd, scenarioEnd }, null, 2));
