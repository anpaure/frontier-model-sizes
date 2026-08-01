import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";
import {
  k3ActiveB,
  k3EvidencePath,
  k3ParameterSource,
  k3TotalB,
  k3TotalT,
} from "./k3_primary_evidence.mjs";

const workDir = path.dirname(fileURLToPath(import.meta.url));
const homeDir = process.env.HOME ? path.resolve(process.env.HOME) : null;
const portableLocalPath = (value) => {
  if (typeof value !== "string" || !path.isAbsolute(value)) return value;
  const resolved = path.resolve(value);
  if (resolved === workDir || resolved.startsWith(`${workDir}${path.sep}`)) {
    return `./${path.relative(workDir, resolved).split(path.sep).join("/")}`;
  }
  if (homeDir && (resolved === homeDir || resolved.startsWith(`${homeDir}${path.sep}`))) {
    return `~/${path.relative(homeDir, resolved).split(path.sep).join("/")}`;
  }
  return value;
};
const outputDir = `${workDir}/outputs/019f6c42-2d53-7743-ab07-6293e2618dd7`;
const qaDir = `${workDir}/qa/k3-crosscheck`;
const outputPath = `${outputDir}/k3_calibrated_frontier_parameter_crosscheck_2026-07-17.xlsx`;
const results = JSON.parse(await fs.readFile(`${workDir}/regression_results.json`, "utf8"));
const backtestPath = `${outputDir}/frontier_parameter_chronological_backtest_2026-07-17.json`;
const backtest = JSON.parse(await fs.readFile(backtestPath, "utf8"));

const exactlyOne = (items, description) => {
  if (items.length !== 1) {
    throw new Error(`Expected exactly one ${description}; found ${items.length}`);
  }
  return items[0];
};

const k3EciRecord = exactlyOne(
  results.eci.open_models.filter((row) => row.model === "Kimi K3"),
  "current Kimi K3 ECI record",
);
const k3AaCheck = exactlyOne(
  backtest.external_checks.filter(
    (row) => row.anchor === "Kimi K3" && row.method === "AA expanding fit",
  ),
  "Kimi K3 AA external check",
);
const k3AaKimiHeldoutCheck = exactlyOne(
  backtest.external_checks.filter(
    (row) => row.anchor === "Kimi K3" && row.method.includes("all Kimi held out"),
  ),
  "Kimi K3 all-Kimi-held-out AA external check",
);

if (k3EciRecord.release_date !== "2026-07-16" || k3EciRecord.total_b !== k3TotalB) {
  throw new Error("Kimi K3 ECI identity or exact parameter truth does not match primary evidence");
}
if (k3AaCheck.actual_b !== k3TotalB || k3AaKimiHeldoutCheck.actual_b !== k3TotalB) {
  throw new Error("Kimi K3 external checks do not use the exact primary parameter truth");
}

const eciDatedIntercept = 3.4585141554296785;
const eciDatedScoreSlope = 0.06429752658522685;
const eciDateCoefficient = -0.35092795129752774;
const eciNoDateIntercept = 5.704969833163;
const eciNoDateScoreSlope = 0.041808992552;
const eciNoDateWeight = 0.60;
const eciDatedWeight = 0.40;
const k3YearsFromEciOrigin = (
  asDateUtc("2026-07-16").getTime() - asDateUtc("2023-01-01").getTime()
) / (365.25 * 86_400_000);
const k3EciDatedB = 10 ** (
  eciDatedIntercept
  + eciDatedScoreSlope * k3EciRecord.score
  + eciDateCoefficient * k3YearsFromEciOrigin
) / 1e9;
const k3EciNoDateB = 10 ** (
  eciNoDateIntercept + eciNoDateScoreSlope * k3EciRecord.score
) / 1e9;
const k3EciBlendB = (
  k3EciNoDateB ** eciNoDateWeight * k3EciDatedB ** eciDatedWeight
);
const k3EciErrorFactor = Math.max(k3EciBlendB / k3TotalB, k3TotalB / k3EciBlendB);

const K3 = {
  release_date: "2026-07-16",
  model: "Kimi K3",
  aa_score: 57.1123394372091,
  eci: k3EciRecord.score,
  classification: "Disclosed calibration anchor",
  aa_source: "https://artificialanalysis.ai/models/kimi-k3",
  epoch_source: k3EciRecord.source,
};

const rows = [...results.frontier_predictions, K3]
  .sort((a, b) => b.aa_score - a.aa_score || a.model.localeCompare(b.model));

const wb = Workbook.create();
const summary = wb.worksheets.add("Executive Summary");
const estimates = wb.worksheets.add("Revised Estimates");
const validation = wb.worksheets.add("K3 Validation");
const audit = wb.worksheets.add("Method Audit");
const sources = wb.worksheets.add("Sources");

const C = {
  navy: "#243838",
  teal: "#00A5A6",
  tealDark: "#087879",
  paleTeal: "#E8F4F4",
  paleBlue: "#EDF4FF",
  palePink: "#FFF4FA",
  paleAmber: "#FFF4DF",
  gray: "#F3F6F6",
  mid: "#D2DCDC",
  text: "#3A4848",
  white: "#FFFFFF",
  red: "#A33A3A",
};

function asDateUtc(iso) {
  return new Date(`${iso}T00:00:00Z`);
}
const asDate = asDateUtc;
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

for (const sheet of [summary, estimates, validation, audit, sources]) sheet.showGridLines = false;

// Method sheet is built first because estimate formulas reference its cells.
title(audit, "A1:H1", "Method audit and recalibration specification");
subtitle(audit, "A3:H4", "The preferred estimate combines two direct total-parameter regressions on separate score scales. The ECI branch model-averages dated and no-date specifications; the AA branch is anchored to Kimi K3 after K3 is withheld from fitting. Results are system-equivalent total-parameter scales, not claims about proprietary weight tensors.");

audit.getRange("A6:C6").values = [["Parameter", "Value", "Interpretation"]];
header(audit.getRange("A6:C6"));
audit.getRange("A7:C20").values = [
  ["K3 exact AA v4.1", K3.aa_score, "UI displays 57"],
  ["K3 disclosed total (B)", k3TotalB, "Exact technical-report Table 1 anchor"],
  ["AA log10(P) score slope", 0.04520595908212863, "Family-balanced direct total-parameter fit"],
  ["AA / ECI dated coefficient", eciDateCoefficient, "log10 parameter change per later year at fixed score"],
  ["Conservative LOFO RMSE", 0.48132362596390643, "log10 total-parameter error"],
  ["Conservative uncertainty factor", null, "10^LOFO RMSE"],
  ["ECI dated intercept", eciDatedIntercept, "log10 raw parameter count"],
  ["ECI dated score slope", eciDatedScoreSlope, "per ECI point"],
  ["ECI dated year coefficient", eciDateCoefficient, "base date 2023-01-01"],
  ["ECI no-date intercept", eciNoDateIntercept, "refit during workbook audit"],
  ["ECI no-date score slope", eciNoDateScoreSlope, "refit during workbook audit"],
  ["ECI no-date blend weight", eciNoDateWeight, "favored by row-LOO"],
  ["ECI dated blend weight", eciDatedWeight, "retains calendar information"],
  ["K3 release date", asDate(K3.release_date), "AA release date"],
];
audit.getRange("B12").formulas = [["=POWER(10,B11)"]];
body(audit.getRange("A7:C20"));
audit.getRange("A7:A20").format.font = { name: "Aptos", size: 9, bold: true, color: C.text };
audit.getRange("B8").format.numberFormat = "#,##0.0";
audit.getRange("B7:B7").format.numberFormat = "0.000";
audit.getRange("B9:B11").format.numberFormat = "0.000";
audit.getRange("B12").format.numberFormat = "0.00x";
audit.getRange("B13:B17").format.numberFormat = "0.000";
audit.getRange("B18:B19").format.numberFormat = "0%";
audit.getRange("B20").format.numberFormat = "yyyy-mm-dd";

section(audit, "A23:H23", "Equations and validation logic");
audit.getRange("A25:H26").merge();
audit.getRange("A25").values = [[`AA branch, K3-relative: P_total(target) = ${k3TotalT.toFixed(2)}T × 10^[0.045206 × (AA_target − 57.1123) − 0.350928 × (release_target − 2026-07-16)/365.25]. This keeps K3 as the intercept anchor while retaining the open-model score slope and a shrunk calendar effect.`]];
audit.getRange("A27:H28").merge();
audit.getRange("A27").values = [["ECI branch: 60% log weight on the no-date direct fit and 40% on the workbook's dated direct fit. Final estimate = geometric mean of the AA and ECI branches. AA and ECI are never pooled as if they were the same score scale."]];
audit.getRange("A29:H30").merge();
audit.getRange("A29").values = [[`K3 check: the current family-balanced AA fit predicts ${(k3AaCheck.predicted_b / 1000).toFixed(3)}T before calibration; removing the entire Kimi family predicts ${(k3AaKimiHeldoutCheck.predicted_b / 1000).toFixed(3)}T. The frozen ECI equations predict ${(k3EciBlendB / 1000).toFixed(3)}T from the subsequently observed ECI. The exact disclosed ${k3TotalT.toFixed(2)}T is useful validation, but these same-release observations are not iid evidence.`]];
audit.getRange("A25:H30").format = { fill: C.gray, font: { name: "Aptos", size: 9, color: C.text }, wrapText: true, verticalAlignment: "center" };

section(audit, "A33:H33", "Material findings in the supplied workbook");
audit.getRange("A35:C35").values = [["Priority", "Finding", "Implication / action"]];
header(audit.getRange("A35:C35"));
const findings = [
  ["High", "Date effect is not identified robustly", "Central fit implies 2.24x/year, but the last-12-month coefficient reverses sign; no-date row-LOO RMSE is 0.431 versus 0.479. Model-average rather than treat 2.24x as a law."],
  ["High", "Sibling checkpoints are treated as independent", "DeepSeek, Kimi, GLM, Qwen and MiniMax rows leak lineage information through row-LOO. Use family/lab clusters and rolling-origin validation."],
  ["High", "Reasoning and test-time compute are omitted", "Same underlying weights can receive different implied parameter counts. Interpret results as product/system equivalents at tested reasoning budgets."],
  ["High", "gpt-oss-120b active count is wrong", "Workbook records 116.83B active/dense; official value is about 116.83B total and 5.13B active MoE. Correct before any active-parameter regression."],
  ["Medium", "Parameter conventions are mixed", "Some rows use backbone totals while others include MTP/checkpoint components. Harmonize one definition and retain both fields where possible."],
  ["Medium", "Blank ECI bounds produce numeric artifacts", "GPT-5 blank CI endpoints evaluate as zero in parameter-bound formulas. Return blank when CI is unavailable."],
  ["Medium", "Recent ECI display has only 17 rows", "Useful transparent subset, but not a complete census of last-year open models. Keep AA and ECI panels separate."],
  ["Note", "Terminology", "The inclusion rule is open weights, including restricted/non-commercial licenses; label it open-weight rather than open-source."],
];
audit.getRange(`A36:C${35 + findings.length}`).values = findings;
body(audit.getRange(`A36:C${35 + findings.length}`));
audit.getRange(`A36:C${35 + findings.length}`).format.wrapText = true;
audit.getRange(`A36:C${35 + findings.length}`).format.rowHeight = 34;
audit.getRange(`A36:A${35 + findings.length}`).conditionalFormats.add("containsText", { text: "High", format: { fill: C.palePink, font: { color: C.red, bold: true } } });
audit.getRange("A:A").format.columnWidth = 34;
audit.getRange("B:B").format.columnWidth = 44;
audit.getRange("C:C").format.columnWidth = 78;
audit.getRange("D:H").format.columnWidth = 14;
audit.freezePanes.freezeRows(6);

// Revised estimates.
title(estimates, "A1:N1", "K3-calibrated frontier parameter cross-check");
subtitle(estimates, "A2:N3", "Central total is the geometric mean of a K3-anchored direct AA regression and a 60/40 no-date/dated ECI model average. The factor band uses the worse leave-family-out RMSE (×3.03) and still does not cover every architecture or inference-budget uncertainty. Opus 5 is a distinct fallback-enabled regression target; K3 is a disclosed anchor.");
const estimateHeaders = ["Release", "Model", "AA v4.1", "ECI", "AA direct, K3-anchored (B)", "ECI dated (B)", "ECI no-date (B)", "ECI 60/40 blend (B)", "Revised total (B)", "Factor low (B)", "Factor high (B)", "Classification", "AA source", "ECI source / note"];
estimates.getRange("A5:N5").values = [estimateHeaders];
header(estimates.getRange("A5:N5"));
const start = 6;
const estimateRows = rows.map((r) => [
  asDate(r.release_date), r.model, r.aa_score, r.eci ?? null,
  null, null, null, null, null, null, null,
  r.classification || "pure", r.aa_source || "", r.epoch_source || "Epoch ECI not yet available",
]);
const end = start + estimateRows.length - 1;
estimates.getRange(`A${start}:N${end}`).values = estimateRows;
for (let row = start; row <= end; row += 1) {
  const model = estimateRows[row - start][1];
  if (model === "Kimi K3") {
    estimates.getRange(`E${row}`).formulas = [["='Method Audit'!$B$8"]];
    estimates.getRange(`F${row}`).formulas = [[`=POWER(10,'Method Audit'!$B$13+'Method Audit'!$B$14*D${row}+'Method Audit'!$B$15*((A${row}-DATE(2023,1,1))/365.25))/1E9`]];
    estimates.getRange(`G${row}`).formulas = [[`=POWER(10,'Method Audit'!$B$16+'Method Audit'!$B$17*D${row})/1E9`]];
    estimates.getRange(`H${row}`).formulas = [[`=POWER(G${row},'Method Audit'!$B$18)*POWER(F${row},'Method Audit'!$B$19)`]];
    estimates.getRange(`I${row}`).formulas = [["='Method Audit'!$B$8"]];
  } else {
    estimates.getRange(`E${row}`).formulas = [[`='Method Audit'!$B$8*POWER(10,'Method Audit'!$B$9*(C${row}-'Method Audit'!$B$7)+'Method Audit'!$B$10*((A${row}-'Method Audit'!$B$20)/365.25))`]];
    estimates.getRange(`F${row}`).formulas = [[`=POWER(10,'Method Audit'!$B$13+'Method Audit'!$B$14*D${row}+'Method Audit'!$B$15*((A${row}-DATE(2023,1,1))/365.25))/1E9`]];
    estimates.getRange(`G${row}`).formulas = [[`=POWER(10,'Method Audit'!$B$16+'Method Audit'!$B$17*D${row})/1E9`]];
    estimates.getRange(`H${row}`).formulas = [[`=POWER(G${row},'Method Audit'!$B$18)*POWER(F${row},'Method Audit'!$B$19)`]];
    estimates.getRange(`I${row}`).formulas = [[`=SQRT(E${row}*H${row})`]];
    estimates.getRange(`J${row}`).formulas = [[`=I${row}/'Method Audit'!$B$12`]];
    estimates.getRange(`K${row}`).formulas = [[`=I${row}*'Method Audit'!$B$12`]];
  }
}
body(estimates.getRange(`A${start}:N${end}`));
estimates.getRange(`A${start}:A${end}`).format.numberFormat = "yyyy-mm-dd";
estimates.getRange(`C${start}:D${end}`).format.numberFormat = "0.0";
estimates.getRange(`E${start}:K${end}`).format.numberFormat = "#,##0.0";
estimates.getRange(`I${start}:I${end}`).format.font = { name: "Aptos", size: 10, bold: true, color: C.navy };
estimates.getRange(`L${start}:N${end}`).format.wrapText = true;
estimates.getRange(`L${start}:L${end}`).conditionalFormats.add("containsText", { text: "cascade", format: { fill: C.paleAmber, font: { color: "#8A5A00", bold: true } } });
estimates.getRange(`B${start}:B${end}`).conditionalFormats.add("containsText", { text: "Kimi K3", format: { fill: C.paleBlue, font: { color: C.navy, bold: true } } });
estimates.getRange("A:A").format.columnWidth = 13;
estimates.getRange("B:B").format.columnWidth = 25;
estimates.getRange("C:D").format.columnWidth = 12;
estimates.getRange("E:K").format.columnWidth = 17;
estimates.getRange("L:L").format.columnWidth = 28;
estimates.getRange("M:N").format.columnWidth = 42;
estimates.freezePanes.freezeRows(5);
estimates.tables.add(`A5:N${end}`, true, "RevisedEstimatesTable").style = "TableStyleMedium2";

// K3 validation sheet.
title(validation, "A1:H1", "Kimi K3 calibration and holdout audit");
subtitle(validation, "A3:H4", "K3 is withheld from the AA fit and used to calibrate only the final AA intercept. Its July 31 Epoch ECI observation postdates the supplied July 17 workbook, so the workbook's frozen ECI equations can now be checked out of sample. The exact total and activated counts remain one paired architecture fact, not two independent likelihood terms.");
validation.getRange("A6:C6").values = [["Check", "Result", "Interpretation"]];
header(validation.getRange("A6:C6"));
validation.getRange("A7:C20").values = [
  ["AA UI score", 57, "User screenshot; rounded display"],
  ["AA exact score", K3.aa_score, "v4.1 underlying value"],
  ["Release date", asDate(K3.release_date), "Chronological holdout"],
  ["Official total parameters (B)", k3TotalB, "Exact technical-report Table 1 total"],
  ["Official active parameters (B)", k3ActiveB, "Exact technical-report Table 1 activated count"],
  ["Architecture", "Stable LatentMoE; 16/896 experts", "KDA + Attention Residuals; native vision"],
  ["Current family-balanced AA prediction (B)", k3AaCheck.predicted_b, "Before K3 calibration; generated by the current backtest"],
  ["Leave-entire-Kimi-family-out AA prediction (B)", k3AaKimiHeldoutCheck.predicted_b, "Stricter lab-correlated check"],
  ["Current AA prediction / actual", k3AaCheck.predicted_b / k3TotalB, `${(100 * (1 - k3AaCheck.predicted_b / k3TotalB)).toFixed(1)}% low`],
  ["Leave-Kimi AA prediction / actual", k3AaKimiHeldoutCheck.predicted_b / k3TotalB, `${(100 * (1 - k3AaKimiHeldoutCheck.predicted_b / k3TotalB)).toFixed(1)}% low`],
  ["Current reproduced Epoch ECI", k3EciRecord.score, "July 31 aggregate; not present in the supplied July 17 workbook"],
  ["Current Epoch ECI confidence interval", `${k3EciRecord.ci_low.toFixed(3)}–${k3EciRecord.ci_high.toFixed(3)}`, "Reproduced from Epoch benchmark data"],
  ["Frozen ECI 60/40 prediction (B)", k3EciBlendB, "Original workbook equations; K3 not used to fit them"],
  ["Frozen ECI multiplicative error", k3EciErrorFactor, `${(100 * (1 - k3EciBlendB / k3TotalB)).toFixed(1)}% low`],
];
body(validation.getRange("A7:C20"));
validation.getRange("A7:A20").format.font = { name: "Aptos", size: 9, bold: true, color: C.text };
validation.getRange("B7:B20").format.numberFormat = "0.000";
validation.getRange("B9").format.numberFormat = "yyyy-mm-dd";
validation.getRange("B10:B10").format.numberFormat = "#,##0";
validation.getRange("B13:B14").format.numberFormat = "#,##0";
validation.getRange("B15:B16").format.numberFormat = "0.0%";
validation.getRange("B17").format.numberFormat = "0.000";
validation.getRange("B19").format.numberFormat = "#,##0";
validation.getRange("B20").format.numberFormat = "0.00x";
validation.getRange("A23:H25").merge();
validation.getRange("A23").values = [[`Verdict: K3's stricter AA holdout is off by ×${k3AaKimiHeldoutCheck.multiplicative_error.toFixed(2)}, while the frozen direct ECI equations are off by ×${k3EciErrorFactor.toFixed(2)}. Both support direct log-parameter modeling, but a single same-lab/same-release target cannot justify reweighting the full ensemble or identify the calendar coefficient.`]];
validation.getRange("A23:H25").format = { fill: C.paleTeal, font: { name: "Aptos", size: 10, bold: true, color: C.navy }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: C.teal } } };
validation.getRange("A:A").format.columnWidth = 42;
validation.getRange("B:B").format.columnWidth = 24;
validation.getRange("C:C").format.columnWidth = 56;
validation.getRange("D:H").format.columnWidth = 13;
validation.freezePanes.freezeRows(6);

// Executive summary.
title(summary, "A1:P1", "Frontier parameter estimates — K3-calibrated cross-check");
subtitle(summary, "A3:P4", "The supplied ECI workbook is directionally useful, and Kimi K3 validates its direct-regression orientation. The revised headline model predicts total parameters directly, model-averages the unstable date effect, clusters interpretation by family, and keeps AA and ECI on separate score scales.");
summary.getRange("A6:B6").values = [["Calibration / validation", "Value"]];
header(summary.getRange("A6:B6"));
summary.getRange("A7:A13").values = [["As of"], ["K3 displayed AA"], ["K3 exact AA"], ["K3 disclosed total"], ["AA prediction before K3"], ["AA prediction, Kimi held out"], ["Conservative multiplicative band"]];
summary.getRange("B7:B13").values = [[asDate("2026-07-31")], [57], [K3.aa_score], [k3TotalB], [k3AaCheck.predicted_b], [k3AaKimiHeldoutCheck.predicted_b], [null]];
summary.getRange("B13").formulas = [["='Method Audit'!B12"]];
body(summary.getRange("A7:B13"));
summary.getRange("A7:A13").format.font = { name: "Aptos", size: 9, bold: true, color: C.text };
summary.getRange("B7").format.numberFormat = "yyyy-mm-dd";
summary.getRange("B8").format.numberFormat = "0";
summary.getRange("B9").format.numberFormat = "0.000";
summary.getRange("B10:B12").format.numberFormat = "#,##0 \"B\"";
summary.getRange("B13").format.numberFormat = "0.00x";

section(summary, "A16:F16", "Preferred pure-model totals");
summary.getRange("A17:F17").values = [["Model", "AA v4.1", "Revised total (B)", "Factor low (B)", "Factor high (B)", "Status"]];
header(summary.getRange("A17:F17"));
const selected = ["Claude Opus 5", "GPT-5.6 Sol", "Kimi K3", "GPT-5.5", "Claude Opus 4.8", "GPT-5.6 Terra", "Claude Opus 4.7", "Grok 4.5", "Claude Sonnet 5", "GPT-5.6 Luna", "Gemini 3.5 Flash"];
for (let i = 0; i < selected.length; i += 1) {
  const model = selected[i];
  const sourceRow = start + rows.findIndex((r) => r.model === model);
  const row = 18 + i;
  summary.getRange(`A${row}`).formulas = [[`='Revised Estimates'!B${sourceRow}`]];
  summary.getRange(`B${row}`).formulas = [[`='Revised Estimates'!C${sourceRow}`]];
  summary.getRange(`C${row}`).formulas = [[`='Revised Estimates'!I${sourceRow}`]];
  if (model !== "Kimi K3") {
    summary.getRange(`D${row}`).formulas = [[`='Revised Estimates'!J${sourceRow}`]];
    summary.getRange(`E${row}`).formulas = [[`='Revised Estimates'!K${sourceRow}`]];
  }
  summary.getRange(`F${row}`).values = [[model === "Kimi K3" ? "Disclosed anchor" : "Regression equivalent"]];
}
const selectedEnd = 17 + selected.length;
body(summary.getRange(`A18:F${selectedEnd}`));
summary.getRange(`B18:B${selectedEnd}`).format.numberFormat = "0.0";
summary.getRange(`C18:E${selectedEnd}`).format.numberFormat = "#,##0";
summary.getRange(`C18:C${selectedEnd}`).format.font = { name: "Aptos", size: 10, bold: true, color: C.navy };

const chart = summary.charts.add("bar", { title: "Revised total parameters (B)", hasLegend: false });
const series = chart.series.add("Total (B)");
series.categoryFormula = `'Executive Summary'!$A$18:$A$${selectedEnd}`;
series.formula = `'Executive Summary'!$C$18:$C$${selectedEnd}`;
series.fill = C.teal;
chart.title = "K3-calibrated open-weight-equivalent totals (B)";
chart.hasLegend = false;
chart.setPosition("H6", "P27");

summary.getRange("A30:P32").merge();
summary.getRange("A30").values = [[`Headline: the direct benchmark model places Opus 5 at about 3.2T and Sol at about 3.4T; K3 is exactly disclosed at ${k3TotalT.toFixed(2)}T total / ${k3ActiveB.toFixed(1)}B activated; Opus 4.8 is about 2.5T. These are system-equivalent scales at observed reasoning budgets, not physical disclosures.`]];
summary.getRange("A30:P32").format = { fill: C.paleTeal, font: { name: "Aptos", size: 10, bold: true, color: C.navy }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: C.teal } } };
summary.getRange("A34:P36").merge();
summary.getRange("A34").values = [["Uncertainty: the conservative LOFO scale is roughly ×/÷3.0. Architecture, active-expert count, training compute, data quality, post-training, output-token budget, and provider orchestration can make literal hidden-weight counts differ further. Fallback and service configurations are not separate base-model sizes."]];
summary.getRange("A34:P36").format = { fill: C.palePink, font: { name: "Aptos", size: 9, italic: true, color: C.text }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: "#E03D90" } } };
summary.getRange("A:A").format.columnWidth = 29;
summary.getRange("B:B").format.columnWidth = 16;
summary.getRange("C:E").format.columnWidth = 17;
summary.getRange("F:F").format.columnWidth = 24;
summary.getRange("G:G").format.columnWidth = 3;
summary.getRange("H:P").format.columnWidth = 13;
summary.freezePanes.freezeRows(4);

// Sources.
title(sources, "A1:D1", "Sources and audit trail");
subtitle(sources, "A2:D3", "URLs are plain text for portability. The supplied workbook was inspected read-only and remains unchanged.");
sources.getRange("A5:D5").values = [["Source", "URL / path", "Use", "Snapshot / caveat"]];
header(sources.getRange("A5:D5"));
const sourceRows = [
  ["Supplied ECI regression workbook", `${workDir}/sources/input_eci_parameter_regression_workbook_2026-07-17.xlsx`, "Direct ECI regression, sensitivity specifications, frontier ECIs, raw audit", "Read-only cross-check, 2026-07-17"],
  ["Kimi K3 technical report evidence", k3EvidencePath, `${k3TotalT.toFixed(2)}T total / ${k3ActiveB.toFixed(1)}B activated; Stable LatentMoE`, k3ParameterSource],
  ["Kimi K3 launch", "https://www.kimi.com/blog/kimi-k3", "Rounded 2.8T launch display and release identity", "Exact arithmetic uses the later technical report"],
  ["Artificial Analysis K3", "https://artificialanalysis.ai/models/kimi-k3", "AA v4.1 score 57.1123 and release date", "UI displays 57"],
  ["Artificial Analysis methodology", "https://artificialanalysis.ai/methodology/intelligence-benchmarking", "AA score definition", "v4.1"],
  ["Epoch ECI methodology", "https://epoch.ai/data/eci-documentation/methodology", "ECI measurement context", `K3 was absent from the supplied July 17 workbook; current July 31 reproduction is ${k3EciRecord.score.toFixed(4)}`],
  ["Epoch reproduced ECI scores", `${workDir}/${results.eci_reproduction.scores_path}`, "Exact current K3 ECI and confidence interval", `Epoch data commit ${results.eci_reproduction.source_commit}`],
  ["Chronological parameter backtest", backtestPath, "Current K3 AA external checks; no copied prediction constants", "Generated before this workbook in the pipeline"],
  ["OpenAI gpt-oss announcement", "https://openai.com/index/introducing-gpt-oss/", "Correct total/active parameters for gpt-oss", "About 117B total / 5.1B active for gpt-oss-120b"],
  ["Prior AA/ECI analysis", `${workDir}/regression_results.json`, `${results.n_open_models}-row AA and ${results.eci.open_models.length}-row ECI comparison panels`, "Current regenerated input"],
].map(([source, location, use, caveat]) => [
  source,
  portableLocalPath(location),
  use,
  caveat,
]);
sources.getRange(`A6:D${5 + sourceRows.length}`).values = sourceRows;
body(sources.getRange(`A6:D${5 + sourceRows.length}`));
sources.getRange(`B6:D${5 + sourceRows.length}`).format.wrapText = true;
sources.getRange("A:A").format.columnWidth = 34;
sources.getRange("B:B").format.columnWidth = 58;
sources.getRange("C:C").format.columnWidth = 52;
sources.getRange("D:D").format.columnWidth = 40;
sources.freezePanes.freezeRows(5);
sources.tables.add(`A5:D${5 + sourceRows.length}`, true, "SourcesTable").style = "TableStyleMedium2";

// Judgment comments.
wb.comments.setSelf({ displayName: "Codex" });
wb.comments.addThread({ cell: audit.getRange("B18") }, "No-date receives more weight because the date coefficient is unstable and the no-date row-LOO RMSE is better. The dated model remains in the ensemble so chronology is considered rather than discarded.");
wb.comments.addThread({ cell: validation.getRange("B14") }, "This stricter prediction refits after removing all earlier Kimi-family rows, reducing same-lab leakage. It is still not fully independent of broader MoE design trends.");
wb.comments.addThread({ cell: validation.getRange("B19") }, "This is a genuine check of the supplied workbook's frozen direct ECI equations against an ECI observation that was not present when those equations were recorded. It remains only one target and is not used to tune the branch weight.");
wb.comments.addThread({ cell: estimates.getRange(`I${start + rows.findIndex((r) => r.model === "Kimi K3")}`) }, "K3 is the disclosed calibration anchor. It is not a regression estimate and therefore has no factor band in this workbook.");

// Verification, render, and export.
const checks = [
  ["Executive Summary", "A1:P36"],
  ["Revised Estimates", `A1:N${end}`],
  ["K3 Validation", "A1:H25"],
  ["Method Audit", `A1:H${35 + findings.length}`],
  ["Sources", `A1:D${5 + sourceRows.length}`],
];
for (const [sheetName, range] of checks) {
  const inspected = await wb.inspect({ kind: "table", range: `${sheetName}!${range}`, include: "values,formulas", tableMaxRows: 50, tableMaxCols: 16, maxChars: 12000 });
  console.log(inspected.ndjson);
}
const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
console.log(errors.ndjson);

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });
for (const [sheetName, range] of checks) {
  const rendered = await wb.render({ sheetName, range, scale: 1.15, format: "png" });
  const file = sheetName.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  await fs.writeFile(`${qaDir}/${file}.png`, new Uint8Array(await rendered.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, qaDir, rows: rows.length }, null, 2));
