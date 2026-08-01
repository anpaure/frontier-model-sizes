import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const workDir = path.dirname(fileURLToPath(import.meta.url));
const outputDir = `${workDir}/outputs/019f6c42-2d53-7743-ab07-6293e2618dd7`;
const resultPath = `${outputDir}/frontier_parameter_chronological_backtest_2026-07-17.json`;
const outputPath = `${outputDir}/frontier_parameter_chronological_backtest_2026-07-17.xlsx`;
const qaDir = `${workDir}/qa/parameter-backtest`;
const data = JSON.parse(await fs.readFile(resultPath, "utf8"));
const currentEnsembleMetrics = data.current_like_metrics["Available-components ensemble"];
const equalEnsembleMetrics = data.model_comparisons.find(
  (row) => row.panel === "Available-components ensemble" && row.spec === "equal_available_weights",
);
const k3Ensemble = data.ensemble_predictions.find((row) => row.model === "Kimi K3");
const k3StrictAa = data.external_checks.find(
  (row) => row.anchor === "Kimi K3" && row.method.includes("all Kimi"),
);
const grokExternal = data.external_checks.find(
  (row) => row.anchor === "Grok 4.5" && row.method.startsWith("AA/ECI"),
);

const wb = Workbook.create();
const summary = wb.worksheets.add("Executive Summary");
const comparison = wb.worksheets.add("Model Comparison");
const predictions = wb.worksheets.add("OOS Predictions");
const ensemble = wb.worksheets.add("Ensemble Backtest");
const external = wb.worksheets.add("External Checks");
const method = wb.worksheets.add("Method and Limits");

const C = {
  navy: "#213A3A",
  teal: "#00A5A6",
  tealDark: "#087879",
  paleTeal: "#E7F4F4",
  paleBlue: "#EDF4FF",
  paleAmber: "#FFF3D8",
  palePink: "#FFF0F5",
  gray: "#F3F6F6",
  mid: "#D2DCDC",
  text: "#354646",
  white: "#FFFFFF",
  red: "#A33A3A",
  green: "#1F6B4F",
};

const asDate = (iso) => new Date(`${iso}T00:00:00Z`);
const clean = (value) => (value === undefined ? null : value);
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
  sheet.getRange(range).format = {
    font: { name: "Aptos", size: 10, color: C.text },
    wrapText: true,
    verticalAlignment: "center",
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
const header = (range) => {
  range.format = {
    fill: C.tealDark,
    font: { name: "Aptos", size: 9, bold: true, color: C.white },
    wrapText: true,
    verticalAlignment: "center",
    borders: { bottom: { style: "medium", color: C.teal } },
  };
  range.format.rowHeight = 32;
};
const body = (range) => {
  range.format = {
    font: { name: "Aptos", size: 9, color: C.text },
    verticalAlignment: "center",
    borders: { insideHorizontal: { style: "thin", color: C.mid } },
  };
};

for (const sheet of [summary, comparison, predictions, ensemble, external, method]) sheet.showGridLines = false;

// Executive summary.
title(summary, "A1:P1", "Frontier parameter model — chronological backtest");
subtitle(summary, "A2:P4", "Each held-out checkpoint is predicted using only strictly earlier releases. The preferred test also removes the checkpoint's entire family/lab from training. Target: disclosed total parameters. Scores are from current snapshots, so this is a pseudo-chronological mapping test rather than a fully vintage real-time forecast.");
section(summary, "A6:G6", "Current-like specifications, strict family-held-out split");
summary.getRange("A7:G7").values = [["Component", "OOS n", "Median error", "Within 2x", "80th pct error", "Bias ratio", "Frontier-like median"]];
header(summary.getRange("A7:G7"));
const summaryPanels = ["AA", "ECI", "No-CoT", "Compute", "Available-components ensemble"];
const summaryRows = summaryPanels.map((panel) => {
  const m = data.current_like_metrics[panel];
  const f = data.frontier_like_metrics[panel];
  return [panel, m.n, m.median_multiplicative_error, m.within_2x, m.p80_multiplicative_error, m.signed_bias_factor, f.median_multiplicative_error];
});
summary.getRange("A8:G12").values = summaryRows;
body(summary.getRange("A8:G12"));
summary.getRange("A8:A12").format.font = { name: "Aptos", size: 9, bold: true, color: C.text };
summary.getRange("B8:B12").format.numberFormat = "0";
summary.getRange("C8:C12").format.numberFormat = "0.0x";
summary.getRange("D8:D12").format.numberFormat = "0%";
summary.getRange("E8:G12").format.numberFormat = "0.0x";
summary.getRange("A12:G12").format.fill = C.paleTeal;
summary.getRange("C8:C12").conditionalFormats.add("cellValue", { operator: "greaterThan", formula: 3, format: { fill: C.palePink, font: { color: C.red, bold: true } } });

const chart = summary.charts.add("bar", { title: "Strict OOS median multiplicative error", hasLegend: false });
const chartSeries = chart.series.add("Median error factor");
chartSeries.categoryFormula = "'Executive Summary'!$A$8:$A$12";
chartSeries.formula = "'Executive Summary'!$C$8:$C$12";
chartSeries.fill = C.teal;
chart.title = "Strict chronological median error (lower is better)";
chart.hasLegend = false;
chart.yAxis = { numberFormatCode: "0.0x", min: 0, max: 3.0 };
chart.setPosition("I6", "P20");

section(summary, "A15:G15", "External anchors not used as locked predictions");
summary.getRange("A16:E16").values = [["Anchor", "Method", "Actual (T)", "Predicted (T)", "Error"]];
header(summary.getRange("A16:E16"));
const externalSummary = [
  data.external_checks.find((row) => row.anchor === "Kimi K3" && row.method === "AA expanding fit"),
  data.external_checks.find((row) => row.anchor === "Kimi K3" && row.method.includes("all Kimi")),
  data.external_checks.find((row) => row.anchor === "Grok 4.5" && row.method.startsWith("AA/ECI")),
];
summary.getRange("A17:E19").values = externalSummary.map((row) => [row.anchor, row.method, row.actual_b / 1000, row.predicted_b / 1000, row.multiplicative_error]);
body(summary.getRange("A17:E19"));
summary.getRange("C17:D19").format.numberFormat = "0.0 \"T\"";
summary.getRange("E17:E19").format.numberFormat = "0.0x";

summary.getRange("A22:P25").merge();
summary.getRange("A22").values = [[`Verdict: there is real signal, but not literal-weight precision. Individual components recover held-out parameter counts with roughly 2.5–2.7× median error. The current available-component ensemble reaches ${currentEnsembleMetrics.median_multiplicative_error.toFixed(1)}× median error on ${currentEnsembleMetrics.n} matched checkpoints, with a ${currentEnsembleMetrics.p80_multiplicative_error.toFixed(1)}× 80th-percentile miss. K3's all-Kimi-held-out AA check is ${(k3StrictAa.predicted_b / 1000).toFixed(2)}T versus 2.78T; after the much lower ECI/compute proxies are included, its corrected incomplete-component ensemble is ${(k3Ensemble.predicted_b / 1000).toFixed(2)}T / ${k3Ensemble.multiplicative_error.toFixed(2)}×. Grok's AA/ECI external check is ${(grokExternal.predicted_b / 1000).toFixed(2)}T versus 1.5T. These are stress checks, not proof of tight calibration.`]];
summary.getRange("A22:P25").format = { fill: C.paleTeal, font: { name: "Aptos", size: 10, bold: true, color: C.navy }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: C.teal } } };
summary.getRange("A27:P30").merge();
summary.getRange("A27").values = [["Implication for Fable/Sol: the point estimates remain plausible, but the backtest does not justify interpreting one-decimal outputs as precise physical counts. The date term is not consistently helpful out of sample, and a 50% No-CoT evidence weight is not decisively validated: current weights have a slightly better median, while equal available-component weights have better RMSE, bias, and tail error. Keep the date law shrunk/model-averaged and report broad scenario ranges."]];
summary.getRange("A27:P30").format = { fill: C.paleAmber, font: { name: "Aptos", size: 9, color: C.text }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: "#C88400" } } };
summary.getRange("A32:P35").merge();
summary.getRange("A32").values = [["What this does not validate: the human crowd (targets still undisclosed), API price (too few clean disclosed bases), or independence among AA, ECI, No-CoT, and compute. These signals share model quality, training compute, and benchmark construction, so multiplying them as independent likelihoods would overstate confidence."]];
summary.getRange("A32:P35").format = { fill: C.palePink, font: { name: "Aptos", size: 9, italic: true, color: C.text }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: "#D33A7C" } } };
summary.getRange("A:A").format.columnWidth = 34;
summary.getRange("B:B").format.columnWidth = 34;
summary.getRange("C:G").format.columnWidth = 16;
summary.getRange("H:H").format.columnWidth = 3;
summary.getRange("I:P").format.columnWidth = 13;
summary.freezePanes.freezeRows(4);

// Model comparison.
title(comparison, "A1:N1", "Model specification comparison");
subtitle(comparison, "A2:N3", "All rows use expanding windows and strict date ordering. 'Chronological' may retain older siblings; 'chronological_family_holdout' removes the test family and is the preferred leakage-resistant estimate. Current-like specifications were fixed by the live pipeline before this backtest was read.");
const comparisonHeaders = ["Panel", "Split", "Specification", "Current-like", "n", "Median error", "Geo-mean error", "RMSE log10", "Bias ratio", "Within 1.5x", "Within 2x", "Within 3x", "P80 error", "P90 error"];
comparison.getRange("A5:N5").values = [comparisonHeaders];
header(comparison.getRange("A5:N5"));
const comparisonRows = [...data.model_comparisons].sort((a, b) => a.panel.localeCompare(b.panel) || a.split.localeCompare(b.split) || Number(b.is_current_like) - Number(a.is_current_like) || a.spec.localeCompare(b.spec));
const comparisonEnd = 5 + comparisonRows.length;
comparison.getRange(`A6:N${comparisonEnd}`).values = comparisonRows.map((row) => [
  row.panel, row.split, row.spec, row.is_current_like ? "Yes" : "No", row.n,
  clean(row.median_multiplicative_error), clean(row.geomean_multiplicative_error), clean(row.rmse_log10), clean(row.signed_bias_factor),
  clean(row.within_1_5x), clean(row.within_2x), clean(row.within_3x), clean(row.p80_multiplicative_error), clean(row.p90_multiplicative_error),
]);
body(comparison.getRange(`A6:N${comparisonEnd}`));
comparison.getRange(`E6:E${comparisonEnd}`).format.numberFormat = "0";
comparison.getRange(`F6:G${comparisonEnd}`).format.numberFormat = "0.0x";
comparison.getRange(`H6:H${comparisonEnd}`).format.numberFormat = "0.000";
comparison.getRange(`I6:I${comparisonEnd}`).format.numberFormat = "0.0x";
comparison.getRange(`J6:L${comparisonEnd}`).format.numberFormat = "0%";
comparison.getRange(`M6:N${comparisonEnd}`).format.numberFormat = "0.0x";
comparison.getRange(`D6:D${comparisonEnd}`).conditionalFormats.add("containsText", { text: "Yes", format: { fill: C.paleTeal, font: { color: C.green, bold: true } } });
comparison.getRange("A:A").format.columnWidth = 28;
comparison.getRange("B:B").format.columnWidth = 34;
comparison.getRange("C:C").format.columnWidth = 36;
comparison.getRange("D:E").format.columnWidth = 14;
comparison.getRange("F:N").format.columnWidth = 15;
comparison.freezePanes.freezeRows(5);
comparison.tables.add(`A5:N${comparisonEnd}`, true, "ModelComparisonTable").style = "TableStyleMedium2";

// Detailed preferred OOS predictions.
title(predictions, "A1:M1", "Preferred out-of-sample predictions");
subtitle(predictions, "A2:M3", "Current-like specification for each component plus the current-weight available-component ensemble. Every row is strictly chronological and family-held-out. Factor error and signed ratio are workbook formulas so the arithmetic remains inspectable.");
const predictionHeaders = ["Panel", "Release", "Model", "Family", "Actual (B)", "Predicted (B)", "Factor error", "Pred / actual", "Train n", "Train families", "Train max date", "Frontier rank", "Components"];
predictions.getRange("A5:M5").values = [predictionHeaders];
header(predictions.getRange("A5:M5"));
const predictionRows = [...data.predictions].sort((a, b) => a.panel.localeCompare(b.panel) || a.release_date.localeCompare(b.release_date) || a.model.localeCompare(b.model));
const predictionEnd = 5 + predictionRows.length;
predictions.getRange(`A6:M${predictionEnd}`).values = predictionRows.map((row) => [
  row.panel, asDate(row.release_date), row.model, row.family, row.actual_b, row.predicted_b, null, null,
  row.train_n, row.train_family_n, asDate(row.train_max_date), row.frontier_signal_rank, row.component_count ?? 1,
]);
for (let row = 6; row <= predictionEnd; row += 1) {
  predictions.getRange(`G${row}`).formulas = [[`=MAX(F${row}/E${row},E${row}/F${row})`]];
  predictions.getRange(`H${row}`).formulas = [[`=F${row}/E${row}`]];
}
body(predictions.getRange(`A6:M${predictionEnd}`));
predictions.getRange(`B6:B${predictionEnd}`).format.numberFormat = "yyyy-mm-dd";
predictions.getRange(`E6:F${predictionEnd}`).format.numberFormat = "#,##0.0";
predictions.getRange(`G6:H${predictionEnd}`).format.numberFormat = "0.0x";
predictions.getRange(`I6:J${predictionEnd}`).format.numberFormat = "0";
predictions.getRange(`K6:K${predictionEnd}`).format.numberFormat = "yyyy-mm-dd";
predictions.getRange(`L6:L${predictionEnd}`).format.numberFormat = "0%";
predictions.getRange(`M6:M${predictionEnd}`).format.numberFormat = "0";
predictions.getRange(`G6:G${predictionEnd}`).conditionalFormats.add("cellValue", { operator: "greaterThan", formula: 3, format: { fill: C.palePink, font: { color: C.red } } });
predictions.getRange("A:A").format.columnWidth = 30;
predictions.getRange("B:B").format.columnWidth = 13;
predictions.getRange("C:C").format.columnWidth = 42;
predictions.getRange("D:D").format.columnWidth = 25;
predictions.getRange("E:H").format.columnWidth = 15;
predictions.getRange("I:M").format.columnWidth = 15;
predictions.freezePanes.freezeRows(5);
predictions.tables.add(`A5:M${predictionEnd}`, true, "OOSPredictionTable").style = "TableStyleMedium2";

// Ensemble details.
title(ensemble, "A1:J1", "Available-component ensemble backtest");
subtitle(ensemble, "A2:J3", "Conservative exact-ish name matching, compatible disclosed totals, release dates within 62 days, and at least two independently backtested component predictions. Missing components are omitted and remaining weights renormalized, mirroring the live visualizer.");
ensemble.getRange("A5:F5").values = [["Metric", "Current weights", "Equal weights", "Interpretation", null, null]];
header(ensemble.getRange("A5:F5"));
const currentEnsembleMetric = data.current_like_metrics["Available-components ensemble"];
const equalMetric = data.model_comparisons.find((row) => row.panel === "Available-components ensemble" && row.spec === "equal_available_weights");
const ensembleMetricRows = [
  ["n", currentEnsembleMetric.n, equalMetric.n, "Same matched checkpoints"],
  ["Median multiplicative error", currentEnsembleMetric.median_multiplicative_error, equalMetric.median_multiplicative_error, "Current weights slightly better"],
  ["RMSE log10", currentEnsembleMetric.rmse_log10, equalMetric.rmse_log10, "Equal weights slightly better"],
  ["Bias ratio", currentEnsembleMetric.signed_bias_factor, equalMetric.signed_bias_factor, "1.0 is calibrated"],
  ["Within 2x", currentEnsembleMetric.within_2x, equalMetric.within_2x, "Current weights slightly better"],
  ["P80 multiplicative error", currentEnsembleMetric.p80_multiplicative_error, equalMetric.p80_multiplicative_error, "Equal weights has smaller tail"],
];
ensemble.getRange("A6:D11").values = ensembleMetricRows;
body(ensemble.getRange("A6:D11"));
ensemble.getRange("B6:C6").format.numberFormat = "0";
ensemble.getRange("B7:C7").format.numberFormat = "0.0x";
ensemble.getRange("B8:C8").format.numberFormat = "0.000";
ensemble.getRange("B9:C9").format.numberFormat = "0.0x";
ensemble.getRange("B10:C10").format.numberFormat = "0%";
ensemble.getRange("B11:C11").format.numberFormat = "0.0x";

section(ensemble, "A14:J14", "Current-weight ensemble predictions");
ensemble.getRange("A15:J15").values = [["Release", "Model", "Actual (B)", "Predicted (B)", "Factor error", "Pred / actual", "Components", "Component predictions", "Train max date", "Frontier rank"]];
header(ensemble.getRange("A15:J15"));
const ensembleRows = data.ensemble_predictions;
const ensembleEnd = 15 + ensembleRows.length;
ensemble.getRange(`A16:J${ensembleEnd}`).values = ensembleRows.map((row) => [
  asDate(row.release_date), row.model, row.actual_b, row.predicted_b, null, null, row.component_count,
  row.components.map((component) => `${component.panel}: ${(component.predicted_b / 1000).toFixed(2)}T`).join(" | "),
  asDate(row.train_max_date), row.frontier_signal_rank,
]);
for (let row = 16; row <= ensembleEnd; row += 1) {
  ensemble.getRange(`E${row}`).formulas = [[`=MAX(D${row}/C${row},C${row}/D${row})`]];
  ensemble.getRange(`F${row}`).formulas = [[`=D${row}/C${row}`]];
}
body(ensemble.getRange(`A16:J${ensembleEnd}`));
ensemble.getRange(`A16:A${ensembleEnd}`).format.numberFormat = "yyyy-mm-dd";
ensemble.getRange(`C16:D${ensembleEnd}`).format.numberFormat = "#,##0.0";
ensemble.getRange(`E16:F${ensembleEnd}`).format.numberFormat = "0.0x";
ensemble.getRange(`G16:G${ensembleEnd}`).format.numberFormat = "0";
ensemble.getRange(`H16:H${ensembleEnd}`).format.wrapText = true;
ensemble.getRange(`I16:I${ensembleEnd}`).format.numberFormat = "yyyy-mm-dd";
ensemble.getRange(`J16:J${ensembleEnd}`).format.numberFormat = "0%";
ensemble.getRange(`E16:E${ensembleEnd}`).conditionalFormats.add("cellValue", { operator: "greaterThan", formula: 3, format: { fill: C.palePink, font: { color: C.red } } });
ensemble.getRange("A:A").format.columnWidth = 13;
ensemble.getRange("B:B").format.columnWidth = 42;
ensemble.getRange("C:G").format.columnWidth = 15;
ensemble.getRange("H:H").format.columnWidth = 60;
ensemble.getRange("I:J").format.columnWidth = 15;
ensemble.freezePanes.freezeRows(15);
ensemble.tables.add(`A15:J${ensembleEnd}`, true, "EnsemblePredictionTable").style = "TableStyleMedium2";

// External anchor checks.
title(external, "A1:H1", "External anchor checks");
subtitle(external, "A2:H3", "These checks compare pre-lock component outputs with subsequently supplied disclosed totals. They are not part of the expanding-window panel and should be read as two encouraging case studies, not an estimated generalization rate.");
external.getRange("A5:H5").values = [["Anchor", "Method", "Actual (B)", "Predicted (B)", "Factor error", "Train n", "Coefficients", "Note"]];
header(external.getRange("A5:H5"));
const externalEnd = 5 + data.external_checks.length;
external.getRange(`A6:H${externalEnd}`).values = data.external_checks.map((row) => [
  row.anchor, row.method, row.actual_b, row.predicted_b, null, row.train_n,
  row.coefficients.length ? row.coefficients.map((value) => value.toFixed(4)).join(", ") : "", row.note,
]);
for (let row = 6; row <= externalEnd; row += 1) external.getRange(`E${row}`).formulas = [[`=MAX(D${row}/C${row},C${row}/D${row})`]];
body(external.getRange(`A6:H${externalEnd}`));
external.getRange(`C6:D${externalEnd}`).format.numberFormat = "#,##0.0";
external.getRange(`E6:E${externalEnd}`).format.numberFormat = "0.0x";
external.getRange(`F6:F${externalEnd}`).format.numberFormat = "0";
external.getRange(`G6:H${externalEnd}`).format.wrapText = true;
external.getRange("A:A").format.columnWidth = 18;
external.getRange("B:B").format.columnWidth = 46;
external.getRange("C:F").format.columnWidth = 16;
external.getRange("G:G").format.columnWidth = 44;
external.getRange("H:H").format.columnWidth = 68;
external.freezePanes.freezeRows(5);
external.tables.add(`A5:H${externalEnd}`, true, "ExternalChecksTable").style = "TableStyleMedium2";
external.getRange("A14:H17").merge();
external.getRange("A14").values = [[`Interpretation: K3's expanding AA fit has ${data.external_checks.find((row) => row.anchor === "Kimi K3" && row.method === "AA expanding fit").multiplicative_error.toFixed(2)}× error; excluding every earlier Kimi lineage gives ${k3StrictAa.multiplicative_error.toFixed(2)}×. Grok's AA/ECI geometric estimate has ${grokExternal.multiplicative_error.toFixed(2)}× error. These checks are materially better than the broad panel average, but they are only two frontier-scale observations and may be lucky or lab-correlated.`]];
external.getRange("A14:H17").format = { fill: C.paleTeal, font: { name: "Aptos", size: 10, bold: true, color: C.navy }, wrapText: true, verticalAlignment: "center" };

// Method, inventory, and limits.
title(method, "A1:H1", "Method, data inventory, and limits");
subtitle(method, "A2:H3", "This sheet records the exact evaluation contract. It is deliberately stricter than ordinary random cross-validation and deliberately narrower than a claim of real-time historical forecasting.");
section(method, "A5:H5", "Evaluation contract");
method.getRange("A6:C6").values = [["Item", "Rule", "Why"]];
header(method.getRange("A6:C6"));
const contractRows = [
  ["Target", data.metadata.target, "Directly tests literal total-parameter recovery"],
  ["Date split", data.metadata.date_rule, "Blocks future-checkpoint leakage"],
  ["Preferred split", data.metadata.preferred_split, "Blocks older sibling/lab-family leakage"],
  ["Weights", data.metadata.family_weighting, "Prevents prolific families from dominating"],
  ["Current-like status", data.metadata.selection_caveat, "Avoids selecting a winner after seeing test results"],
  ["Benchmark vintage", data.metadata.benchmark_vintage_caveat, "Limits the causal interpretation"],
];
method.getRange("A7:C12").values = contractRows;
body(method.getRange("A7:C12"));
method.getRange("B7:C12").format.wrapText = true;

section(method, "A15:H15", "Panel inventory");
method.getRange("A16:F16").values = [["Panel", "Rows", "Families", "First release", "Last release", "Current-like specification"]];
header(method.getRange("A16:F16"));
const currentSpec = { AA: "score + exact date", ECI: "60% score-only / 40% score+date", "No-CoT": "log horizon + date + MoE", Compute: "log compute + date" };
const inventoryRows = Object.entries(data.inventory).map(([panel, row]) => [panel, row.rows, row.families, asDate(row.min_date), asDate(row.max_date), currentSpec[panel]]);
method.getRange("A17:F20").values = inventoryRows;
body(method.getRange("A17:F20"));
method.getRange("B17:C20").format.numberFormat = "0";
method.getRange("D17:E20").format.numberFormat = "yyyy-mm-dd";

section(method, "A23:H23", "Current evidence weights and testable implication");
method.getRange("A24:D24").values = [["Component", "Evidence-only live weight", "Backtest median error", "Backtest status"]];
header(method.getRange("A24:D24"));
method.getRange("A25:D28").values = [
  ["AA", 19.125 / 93.25, data.current_like_metrics.AA.median_multiplicative_error, "Useful, high tail risk"],
  ["ECI", 19.125 / 93.25, data.current_like_metrics.ECI.median_multiplicative_error, "Useful, best large panel after ensemble"],
  ["No-CoT", 50 / 93.25, data.current_like_metrics["No-CoT"].median_multiplicative_error, "Useful, not precise enough to establish majority weight"],
  ["Compute", 5 / 93.25, data.current_like_metrics.Compute.median_multiplicative_error, "Correctly treated as weak/correlated"],
];
body(method.getRange("A25:D28"));
method.getRange("B25:B28").format.numberFormat = "0%";
method.getRange("C25:C28").format.numberFormat = "0.0x";
method.getRange("A30:H33").merge();
method.getRange("A30").values = [[`Weight conclusion: the current available-component mix is not dominated by the equal-weight sensitivity, but neither does it dominate it. Current weights: ${currentEnsembleMetrics.median_multiplicative_error.toFixed(1)}× median, ${currentEnsembleMetrics.rmse_log10.toFixed(3)} log10 RMSE, ${currentEnsembleMetrics.p80_multiplicative_error.toFixed(1)}× P80, ${currentEnsembleMetrics.signed_bias_factor.toFixed(2)} bias. Equal weights: ${equalEnsembleMetrics.median_multiplicative_error.toFixed(1)}× median, ${equalEnsembleMetrics.rmse_log10.toFixed(3)} RMSE, ${equalEnsembleMetrics.p80_multiplicative_error.toFixed(1)}× P80, ${equalEnsembleMetrics.signed_bias_factor.toFixed(2)} bias. Treat 50% No-CoT as a prior judgment, not a backtest-derived optimum.`]];
method.getRange("A30:H33").format = { fill: C.paleAmber, font: { name: "Aptos", size: 9, color: C.text }, wrapText: true, verticalAlignment: "center" };

section(method, "A36:H36", "Limits and next validation steps");
method.getRange("A37:C37").values = [["Priority", "Limitation", "Required improvement"]];
header(method.getRange("A37:C37"));
const limitRows = [
  ["High", "Current benchmark snapshots, not historical vintages", "Archive dated AA/ECI/No-CoT snapshots and rerun true vintage forecasts"],
  ["High", "Only K3 and Grok supply clean external frontier anchors", "Pre-register predictions before the next open-weight or closed-model disclosure"],
  ["High", "Crowd forecasts cannot yet be scored", "Timestamp every forecast and freeze it before disclosure"],
  ["High", "Components are correlated", "Estimate an OOS residual covariance matrix on a larger exact-overlap panel"],
  ["Medium", "Family definitions are manual proxies", "Maintain a base-model lineage graph and hold out shared pretrained bases"],
  ["Medium", "No-CoT architecture is known for open models but hidden for closed targets", "Run explicit dense/MoE scenario forecasts"],
  ["Medium", "API price branch has too few disclosed proprietary bases", "Do not raise its weight until there are more independent disclosures"],
  ["Medium", "Total parameters are not active parameters or training compute", "Report total, active, and compute-equivalent posteriors separately"],
];
method.getRange("A38:C45").values = limitRows;
body(method.getRange("A38:C45"));
method.getRange("B38:C45").format.wrapText = true;
method.getRange("A38:A45").conditionalFormats.add("containsText", { text: "High", format: { fill: C.palePink, font: { color: C.red, bold: true } } });

section(method, "A48:H48", "Immutable source files");
method.getRange("A49:C49").values = [["Path", "SHA-256", "Use"]];
header(method.getRange("A49:C49"));
const sourceRows = Object.entries(data.source_files).map(([path, hash]) => [path, hash, path.endsWith("regression_results.json") ? "Curated AA/ECI panels" : path.endsWith(".csv") ? "Unified No-CoT/Epoch compute rows" : "Live factor checks"]);
method.getRange(`A50:C${49 + sourceRows.length}`).values = sourceRows;
body(method.getRange(`A50:C${49 + sourceRows.length}`));
method.getRange(`A50:C${49 + sourceRows.length}`).format.wrapText = true;
method.getRange("A:A").format.columnWidth = 34;
method.getRange("B:B").format.columnWidth = 72;
method.getRange("C:C").format.columnWidth = 58;
method.getRange("D:F").format.columnWidth = 19;
method.getRange("G:H").format.columnWidth = 14;
method.freezePanes.freezeRows(3);

// Comments make the highest-judgment choices explicit.
wb.comments.setSelf({ displayName: "Codex" });
wb.comments.addThread({ cell: summary.getRange("C12") }, "The ensemble row is not a full-pipeline backtest: price and crowd are unavailable, and models contribute only where at least two component OOS predictions match conservatively.");
wb.comments.addThread({ cell: method.getRange("B26") }, "The ECI live branch is a 60/40 score-only/dated log blend. The date coefficient remains because chronology matters, but the backtest does not support treating it as a stable law.");
wb.comments.addThread({ cell: method.getRange("B27") }, "No-CoT retains the user's 50% evidence-level prior in the live model. This backtest shows useful signal but does not identify that weight as optimal.");

// Verification, render, and export.
const checks = [
  ["Executive Summary", "A1:P35"],
  ["Model Comparison", `A1:N${comparisonEnd}`],
  ["OOS Predictions", `A1:M${Math.min(predictionEnd, 60)}`],
  ["Ensemble Backtest", `A1:J${ensembleEnd}`],
  ["External Checks", "A1:H17"],
  ["Method and Limits", `A1:H${49 + sourceRows.length}`],
];
for (const [sheetName, range] of checks) {
  const inspected = await wb.inspect({ kind: "table", range: `${sheetName}!${range}`, include: "values,formulas", tableMaxRows: 70, tableMaxCols: 16, maxChars: 18000 });
  console.log(inspected.ndjson);
}
const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 500 }, summary: "final formula error scan" });
console.log(errors.ndjson);

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });
for (const [sheetName, range] of checks) {
  const rendered = await wb.render({ sheetName, range, scale: 1.0, format: "png" });
  const file = sheetName.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  await fs.writeFile(`${qaDir}/${file}.png`, new Uint8Array(await rendered.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, qaDir, predictions: predictionRows.length, ensemble: ensembleRows.length }, null, 2));
