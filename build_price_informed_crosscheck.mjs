import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";
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
const threadId = "019f6c42-2d53-7743-ab07-6293e2618dd7";
const outputDir = `${workDir}/outputs/${threadId}`;
const qaDir = `${workDir}/qa/price-informed`;
const outputPath = `${outputDir}/price_informed_frontier_parameter_crosscheck_2026-07-17.xlsx`;
const k3InputPath = `${outputDir}/k3_calibrated_frontier_parameter_crosscheck_2026-07-17.xlsx`;
const opus5EvidencePath = `${workDir}/sources/claude_opus_5_evidence_2026-07-31.json`;
const opus5Evidence = JSON.parse(await fs.readFile(opus5EvidencePath, "utf8"));

const LOFO_LOG10_RMSE = 0.48132362596390643;
const PRICE_WEIGHT = 0.15;
const ANTHROPIC_TOKEN_FACTOR = 1.3;

const daysBetween = (left, right) => (new Date(`${left}T00:00:00Z`) - new Date(`${right}T00:00:00Z`)) / 86400000;
const opus5Aa = Number(opus5Evidence.artificial_analysis.selected.score);
const opus5Eci = Number(opus5Evidence.epoch.eci_exact);
const opus5Release = opus5Evidence.identity.release_date;
const opus5AaDirectB = k3TotalB * 10 ** (
  0.04520595908212863 * (opus5Aa - 57.1123394372091)
  - 0.35092795129752774 * daysBetween(opus5Release, "2026-07-16") / 365.25
);
const opus5YearsFrom2023 = daysBetween(opus5Release, "2023-01-01") / 365.25;
const opus5EciDatedB = 10 ** (
  3.4585141554296785 + 0.06429752658522685 * opus5Eci
  - 0.35092795129752774 * opus5YearsFrom2023
) / 1e9;
const opus5EciNoDateB = 10 ** (5.704969833163 + 0.041808992552 * opus5Eci) / 1e9;
const opus5EciBlendB = opus5EciNoDateB ** 0.60 * opus5EciDatedB ** 0.40;
const opus5BenchmarkT = Math.sqrt(opus5AaDirectB * opus5EciBlendB) / 1000;

const k3Workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(k3InputPath));
const k3Values = k3Workbook.worksheets.getItem("Revised Estimates").getUsedRange().values;
const k3Headers = Object.fromEntries(k3Values[4].map((value, index) => [String(value), index]));
const k3BenchmarkByModel = new Map(k3Values.slice(5)
  .filter((row) => row[k3Headers.Model])
  .map((row) => [String(row[k3Headers.Model]), Number(row[k3Headers["Revised total (B)"]]) / 1000]));

const modelRows = [
  {
    release: "2026-06-09",
    model: "Claude Fable 5",
    aa: 60,
    eci: 160.44,
    benchmarkT: 3.5009724758824072,
    provider: "Anthropic",
    input: 10,
    output: 50,
    tokenFactor: ANTHROPIC_TOKEN_FACTOR,
    status: "Regression estimate",
    note: "Regular model per user instruction; price-smoothed from the dated benchmark ensemble.",
    priceSource: "https://platform.claude.com/docs/en/about-claude/pricing",
  },
  {
    release: "2026-07-09",
    model: "GPT-5.6 Sol",
    aa: 59,
    eci: 161.77,
    benchmarkT: 3.4292100769491034,
    provider: "OpenAI",
    input: 5,
    output: 30,
    tokenFactor: 1,
    status: "Regression estimate",
    note: "Standard short-context price; max benchmark configuration.",
    priceSource: "https://developers.openai.com/api/docs/pricing",
  },
  {
    release: "2026-07-16",
    model: "Kimi K3",
    aa: 57.1123394372091,
    eci: null,
    benchmarkT: k3TotalT,
    provider: "Moonshot AI",
    input: null,
    output: null,
    tokenFactor: 1,
    status: "Disclosed anchor",
    note: `${k3TotalT.toFixed(2)}T total / ${k3ActiveB.toFixed(1)}B activated disclosed in the technical report. No regression whisker.`,
    priceSource: "",
  },
  {
    release: "2026-04-23",
    model: "GPT-5.5",
    aa: 55,
    eci: 158.33,
    benchmarkT: 2.5655772005746638,
    provider: "OpenAI",
    input: 5,
    output: 30,
    tokenFactor: 1,
    status: "Regression estimate",
    note: "Base service price only; Pro service tiers are not separate base-model observations.",
    priceSource: "https://developers.openai.com/api/docs/pricing",
  },
  {
    release: "2026-05-28",
    model: "Claude Opus 4.7 / 4.8 shared base",
    aa: 55,
    eci: 156.93,
    benchmarkT: Math.sqrt(2.4891765134599178 * 2.1495960719305183),
    provider: "Anthropic",
    input: 5,
    output: 25,
    tokenFactor: ANTHROPIC_TOKEN_FACTOR,
    status: "Regression estimate",
    note: "One latent base observation; 4.7 and 4.8 checkpoint estimates are collapsed geometrically.",
    priceSource: "https://platform.claude.com/docs/en/about-claude/pricing",
  },
  {
    release: "2026-07-09",
    model: "GPT-5.6 Terra",
    aa: 55,
    eci: 158.25,
    benchmarkT: 2.2665558150589984,
    provider: "OpenAI",
    input: 2.5,
    output: 15,
    tokenFactor: 1,
    status: "Regression estimate",
    note: "Standard short-context price.",
    priceSource: "https://developers.openai.com/api/docs/pricing",
  },
  {
    release: "2026-06-30",
    model: "Claude Sonnet 5",
    aa: 53,
    eci: 153.2,
    benchmarkT: 1.5414601447886873,
    provider: "Anthropic",
    input: 2,
    output: 10,
    tokenFactor: ANTHROPIC_TOKEN_FACTOR,
    status: "Regression estimate",
    note: "Current introductory price through 2026-08-31; announced future $3/$15 excluded.",
    priceSource: "https://platform.claude.com/docs/en/about-claude/pricing",
  },
  {
    release: "2026-07-09",
    model: "GPT-5.6 Luna",
    aa: 51,
    eci: 155.62,
    benchmarkT: 1.5781442815133315,
    provider: "OpenAI",
    input: 1,
    output: 6,
    tokenFactor: 1,
    status: "Regression estimate",
    note: "Standard short-context price.",
    priceSource: "https://developers.openai.com/api/docs/pricing",
  },
  {
    release: "2026-05-19",
    model: "Gemini 3.5 Flash",
    aa: 50,
    eci: 154.58,
    benchmarkT: 1.5255469569762952,
    provider: "Google",
    input: null,
    output: null,
    tokenFactor: 1,
    status: "Regression estimate",
    note: "No OpenAI/Anthropic price signal; benchmark estimate is unchanged.",
    priceSource: "",
  },
  {
    release: "2026-07-08",
    model: "Grok 4.5",
    aa: 53.8265951657731,
    eci: 153.17,
    benchmarkT: 1.5,
    priorT: 1.601005407941694,
    provider: "SpaceXAI",
    input: 2,
    output: 6,
    tokenFactor: 1,
    status: "Disclosed anchor",
    note: "Pegged to the public 1.5T V9 foundation disclosure; active count and architecture undisclosed.",
    priceSource: "https://docs.x.ai/developers/models/grok-4.5",
  },
  {
    release: "2026-02-19",
    model: "Gemini 3.1 Pro",
    aa: 46,
    eci: 154.84,
    benchmarkT: 1.4436946593511875,
    provider: "Google",
    input: null,
    output: null,
    tokenFactor: 1,
    status: "Regression estimate",
    note: "No OpenAI/Anthropic price signal; benchmark estimate is unchanged.",
    priceSource: "",
  },
  {
    release: opus5Release,
    model: "Claude Opus 5",
    aa: opus5Aa,
    eci: opus5Eci,
    benchmarkT: opus5BenchmarkT,
    provider: "Anthropic",
    input: Number(opus5Evidence.api.input_usd_per_mtok),
    output: Number(opus5Evidence.api.output_usd_per_mtok),
    tokenFactor: ANTHROPIC_TOKEN_FACTOR,
    status: "Regression estimate",
    note: "Distinct newly pretrained base. AA max used Opus 4.8 fallback; parameter count, architecture, training compute, and shared-weight identity are undisclosed.",
    priceSource: "https://platform.claude.com/docs/en/about-claude/pricing",
  },
];

// The K3/AA/ECI workbook is the generated upstream source of truth.  Resolve
// every non-anchor benchmark input from it so a score/date change cannot leave
// stale copied constants in the price stage.
const benchmarkAliases = new Map([
  ["Claude Opus 4.7 / 4.8 shared base", ["Claude Opus 4.7", "Claude Opus 4.8"]],
  ["Gemini 3.1 Pro", ["Gemini 3.1 Pro Preview"]],
]);
for (const row of modelRows) {
  if (row.model === "Kimi K3" || row.model === "Grok 4.5") continue;
  const aliases = benchmarkAliases.get(row.model) || [row.model];
  const values = aliases.map((name) => k3BenchmarkByModel.get(name));
  if (values.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new Error(`Missing generated K3 benchmark input for ${row.model}: ${aliases.join(", ")}`);
  }
  row.benchmarkT = Math.exp(values.reduce((sum, value) => sum + Math.log(value), 0) / values.length);
}
const generatedOpus5BenchmarkT = k3BenchmarkByModel.get("Claude Opus 5");
if (!Number.isFinite(generatedOpus5BenchmarkT) || Math.abs(generatedOpus5BenchmarkT - opus5BenchmarkT) > 1e-12) {
  throw new Error(`Claude Opus 5 benchmark reproduction mismatch: ${generatedOpus5BenchmarkT} vs ${opus5BenchmarkT}`);
}
modelRows.find((row) => row.model === "Grok 4.5").priorT = k3BenchmarkByModel.get("Grok 4.5");

const priceRows = [
  ["Claude Fable 5", "Anthropic", 10, 50, 1, 12.5, 10, 50, ANTHROPIC_TOKEN_FACTOR, "Current standard", "https://platform.claude.com/docs/en/about-claude/pricing", "New tokenizer; approximately 30% more tokens for equivalent text."],
  ["GPT-5.6 Sol", "OpenAI", 5, 30, 0.5, 6.25, 10, 45, 1, "Short context", "https://developers.openai.com/api/docs/pricing", "Long-context tier excluded from calibration."],
  ["GPT-5.5", "OpenAI", 5, 30, 0.5, null, 10, 45, 1, "Short context", "https://developers.openai.com/api/docs/pricing", "Pro price tier excluded as a service configuration."],
  ["Claude Opus 4.7 / 4.8 shared base", "Anthropic", 5, 25, 0.5, 6.25, 5, 25, ANTHROPIC_TOKEN_FACTOR, "Current standard", "https://platform.claude.com/docs/en/about-claude/pricing", "Same base and same standard price; fast-mode premiums excluded."],
  ["GPT-5.6 Terra", "OpenAI", 2.5, 15, 0.25, 3.125, 5, 22.5, 1, "Short context", "https://developers.openai.com/api/docs/pricing", "Long-context tier excluded from calibration."],
  ["Claude Sonnet 5", "Anthropic", 2, 10, 0.2, 2.5, 2, 10, ANTHROPIC_TOKEN_FACTOR, "Intro through 2026-08-31", "https://platform.claude.com/docs/en/about-claude/pricing", "Future $3/$15 standard price is shown in source but not used."],
  ["GPT-5.6 Luna", "OpenAI", 1, 6, 0.1, 1.25, 2, 9, 1, "Short context", "https://developers.openai.com/api/docs/pricing", "Long-context tier excluded from calibration."],
  ["Grok 4.5", "SpaceXAI", 2, 6, 0.5, null, null, null, 1, "Standard", "https://docs.x.ai/developers/models/grok-4.5", "Validation only; not used to fit the OpenAI/Anthropic price model."],
  ["Claude Opus 5", "Anthropic", Number(opus5Evidence.api.input_usd_per_mtok), Number(opus5Evidence.api.output_usd_per_mtok), 0.5, 6.25, 5, 25, ANTHROPIC_TOKEN_FACTOR, "Current standard", "https://platform.claude.com/docs/en/about-claude/pricing", "Fast mode is a service tier and is excluded; Opus 5 is a distinct base observation."],
];

const wb = Workbook.create();
const summary = wb.worksheets.add("Executive Summary");
const estimates = wb.worksheets.add("Frontier Estimates");
const prices = wb.worksheets.add("API Prices");
const priceModel = wb.worksheets.add("Price Model");
const method = wb.worksheets.add("Anchors & Method");
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

const asDate = (iso) => new Date(`${iso}T00:00:00Z`);
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

for (const sheet of [summary, estimates, prices, priceModel, method, sources]) sheet.showGridLines = false;

// Raw API pricing table.
title(prices, "A1:L1", "API pricing inputs — current standard rates");
subtitle(prices, "A2:L3", "USD per million tokens as of 2026-07-31. The calibration uses only standard uncached input and output rates; cached, batch, fast, regional, and long-context modifiers are retained for audit context but excluded from the price signal.");
prices.getRange("A5:L5").values = [["Base model", "Provider", "Input $/MTok", "Output $/MTok", "Cached input $/MTok", "5m cache write $/MTok", "Long input $/MTok", "Long output $/MTok", "Tokenizer factor", "Rate used", "Official source", "Note"]];
header(prices.getRange("A5:L5"));
prices.getRange(`A6:L${5 + priceRows.length}`).values = priceRows;
body(prices.getRange(`A6:L${5 + priceRows.length}`));
prices.getRange(`C6:I${5 + priceRows.length}`).format.numberFormat = "0.000";
prices.getRange(`J6:L${5 + priceRows.length}`).format.wrapText = true;
prices.getRange("A:A").format.columnWidth = 34;
prices.getRange("B:B").format.columnWidth = 15;
prices.getRange("C:I").format.columnWidth = 18;
prices.getRange("J:J").format.columnWidth = 24;
prices.getRange("K:K").format.columnWidth = 55;
prices.getRange("L:L").format.columnWidth = 52;
prices.freezePanes.freezeRows(5);
prices.tables.add(`A5:L${5 + priceRows.length}`, true, "ApiPricesTable").style = "TableStyleMedium2";

// Price-model cross-fitting table. OpenAI and Anthropic only.
title(priceModel, "A1:O1", "Weak API-price calibration");
subtitle(priceModel, "A2:O3", "For eight unique OpenAI/Anthropic base models: z = sqrt(input price × output price) × tokenizer factor. A provider fixed effect prevents Anthropic and OpenAI commercial schedules from being treated as identical. Each model receives a leave-one-base-out price prediction, then only 15% log weight.");
priceModel.getRange("A5:B12").values = [
  ["Diagnostic / assumption", "Value"],
  ["Full-fit slope on log10(z)", null],
  ["Full-fit OpenAI intercept, T units", null],
  ["Full-fit Anthropic effect", null],
  ["Full-fit R²", null],
  ["LOBO RMSE, log10", null],
  ["LOBO multiplicative error", null],
  ["Price log weight", PRICE_WEIGHT],
];
header(priceModel.getRange("A5:B5"));
body(priceModel.getRange("A6:B12"));
priceModel.getRange("A14:O14").values = [["Model", "Provider", "Benchmark central (T)", "Input", "Output", "Tokenizer factor", "z", "log10(z)", "log10(P)", "Anthropic dummy", "LOBO slope", "LOBO intercept", "LOBO Anthropic effect", "LOBO price-implied (T)", "15% blended (T)"]];
header(priceModel.getRange("A14:O14"));
const priceFitModels = ["Claude Fable 5", "GPT-5.6 Sol", "GPT-5.5", "Claude Opus 4.7 / 4.8 shared base", "GPT-5.6 Terra", "Claude Sonnet 5", "GPT-5.6 Luna", "Claude Opus 5"];
const pmStart = 15;
const pmEnd = pmStart + priceFitModels.length - 1;
const priceSheetByModel = new Map(priceRows.map((row, idx) => [row[0], 6 + idx]));
const estimateSheetByModel = new Map(modelRows.map((row, idx) => [row.model, 6 + idx]));
for (let idx = 0; idx < priceFitModels.length; idx += 1) {
  const row = pmStart + idx;
  const model = priceFitModels[idx];
  const priceRow = priceSheetByModel.get(model);
  const estimateRow = estimateSheetByModel.get(model);
  priceModel.getRange(`A${row}:F${row}`).formulas = [[
    `='API Prices'!A${priceRow}`,
    `='API Prices'!B${priceRow}`,
    `='Frontier Estimates'!E${estimateRow}`,
    `='API Prices'!C${priceRow}`,
    `='API Prices'!D${priceRow}`,
    `='API Prices'!I${priceRow}`,
  ]];
  priceModel.getRange(`G${row}`).formulas = [[`=SQRT(D${row}*E${row})*F${row}`]];
  priceModel.getRange(`H${row}`).formulas = [[`=LOG10(G${row})`]];
  priceModel.getRange(`I${row}`).formulas = [[`=LOG10(C${row})`]];
  priceModel.getRange(`J${row}`).formulas = [[`=IF(B${row}="Anthropic",1,0)`]];

  // Provider-specific raw products used by the leave-one-base-out closed-form OLS.
  priceModel.getRange(`P${row}:W${row}`).formulas = [[
    `=IF(B${row}="OpenAI",H${row},0)`,
    `=IF(B${row}="OpenAI",I${row},0)`,
    `=IF(B${row}="OpenAI",H${row}*I${row},0)`,
    `=IF(B${row}="OpenAI",H${row}*H${row},0)`,
    `=IF(B${row}="Anthropic",H${row},0)`,
    `=IF(B${row}="Anthropic",I${row},0)`,
    `=IF(B${row}="Anthropic",H${row}*I${row},0)`,
    `=IF(B${row}="Anthropic",H${row}*H${row},0)`,
  ]];
  priceModel.getRange(`X${row}:AG${row}`).formulas = [[
    `=COUNTIF($B$${pmStart}:$B$${pmEnd},"OpenAI")-IF(B${row}="OpenAI",1,0)`,
    `=COUNTIF($B$${pmStart}:$B$${pmEnd},"Anthropic")-IF(B${row}="Anthropic",1,0)`,
    `=SUM($P$${pmStart}:$P$${pmEnd})-P${row}`,
    `=SUM($Q$${pmStart}:$Q$${pmEnd})-Q${row}`,
    `=SUM($R$${pmStart}:$R$${pmEnd})-R${row}`,
    `=SUM($S$${pmStart}:$S$${pmEnd})-S${row}`,
    `=SUM($T$${pmStart}:$T$${pmEnd})-T${row}`,
    `=SUM($U$${pmStart}:$U$${pmEnd})-U${row}`,
    `=SUM($V$${pmStart}:$V$${pmEnd})-V${row}`,
    `=SUM($W$${pmStart}:$W$${pmEnd})-W${row}`,
  ]];
  priceModel.getRange(`K${row}`).formulas = [[`=((AB${row}-Z${row}*AA${row}/X${row})+(AF${row}-AD${row}*AE${row}/Y${row}))/((AC${row}-Z${row}*Z${row}/X${row})+(AG${row}-AD${row}*AD${row}/Y${row}))`]];
  priceModel.getRange(`L${row}`).formulas = [[`=AA${row}/X${row}-K${row}*Z${row}/X${row}`]];
  priceModel.getRange(`M${row}`).formulas = [[`=AE${row}/Y${row}-L${row}-K${row}*AD${row}/Y${row}`]];
  priceModel.getRange(`N${row}`).formulas = [[`=POWER(10,L${row}+K${row}*H${row}+M${row}*J${row})`]];
  priceModel.getRange(`O${row}`).formulas = [[`=POWER(C${row},1-$B$12)*POWER(N${row},$B$12)`]];

  // Global-fit prediction and squared residuals for diagnostics.
  priceModel.getRange(`AH${row}`).formulas = [[`=$B$7+$B$6*H${row}+$B$8*J${row}`]];
  priceModel.getRange(`AI${row}`).formulas = [[`=POWER(I${row}-AH${row},2)`]];
  priceModel.getRange(`AJ${row}`).formulas = [[`=POWER(I${row}-AVERAGE($I$${pmStart}:$I$${pmEnd}),2)`]];
  priceModel.getRange(`AK${row}`).formulas = [[`=POWER(I${row}-LOG10(N${row}),2)`]];
}

// Full-fit coefficients, using within-provider demeaning.
priceModel.getRange("B6").formulas = [[`=((SUM(R${pmStart}:R${pmEnd})-SUM(P${pmStart}:P${pmEnd})*SUM(Q${pmStart}:Q${pmEnd})/COUNTIF(B${pmStart}:B${pmEnd},"OpenAI"))+(SUM(V${pmStart}:V${pmEnd})-SUM(T${pmStart}:T${pmEnd})*SUM(U${pmStart}:U${pmEnd})/COUNTIF(B${pmStart}:B${pmEnd},"Anthropic")))/((SUM(S${pmStart}:S${pmEnd})-POWER(SUM(P${pmStart}:P${pmEnd}),2)/COUNTIF(B${pmStart}:B${pmEnd},"OpenAI"))+(SUM(W${pmStart}:W${pmEnd})-POWER(SUM(T${pmStart}:T${pmEnd}),2)/COUNTIF(B${pmStart}:B${pmEnd},"Anthropic")))`]];
priceModel.getRange("B7").formulas = [[`=SUM(Q${pmStart}:Q${pmEnd})/COUNTIF(B${pmStart}:B${pmEnd},"OpenAI")-B6*SUM(P${pmStart}:P${pmEnd})/COUNTIF(B${pmStart}:B${pmEnd},"OpenAI")`]];
priceModel.getRange("B8").formulas = [[`=SUM(U${pmStart}:U${pmEnd})/COUNTIF(B${pmStart}:B${pmEnd},"Anthropic")-B7-B6*SUM(T${pmStart}:T${pmEnd})/COUNTIF(B${pmStart}:B${pmEnd},"Anthropic")`]];
priceModel.getRange("B9").formulas = [[`=1-SUM(AI${pmStart}:AI${pmEnd})/SUM(AJ${pmStart}:AJ${pmEnd})`]];
priceModel.getRange("B10").formulas = [[`=SQRT(AVERAGE(AK${pmStart}:AK${pmEnd}))`]];
priceModel.getRange("B11").formulas = [["=POWER(10,B10)"]];
body(priceModel.getRange(`A15:O${pmEnd}`));
priceModel.getRange(`C15:O${pmEnd}`).format.numberFormat = "0.000";
priceModel.getRange("B6:B8").format.numberFormat = "0.000000";
priceModel.getRange("B9:B10").format.numberFormat = "0.0000";
priceModel.getRange("B11").format.numberFormat = "0.000x";
priceModel.getRange("B12").format.numberFormat = "0%";
priceModel.getRange("A:A").format.columnWidth = 35;
priceModel.getRange("B:B").format.columnWidth = 18;
priceModel.getRange("C:O").format.columnWidth = 16;
priceModel.freezePanes.freezeRows(14);

// Final base-model estimates.
title(estimates, "A1:R1", "Frontier parameter estimates — price-informed and base-collapsed");
subtitle(estimates, "A2:R3", "Primary estimates retain the K3-anchored benchmark/date model. Grok 4.5 is fixed to the disclosed 1.5T scale. API price adds only 15% log weight through a leave-one-base-out OpenAI/Anthropic fit. The conservative ×/÷3.03 band is unchanged.");
estimates.getRange("A5:R5").values = [["Release", "Model / base", "AA v4.1", "ECI", "Primary benchmark (T)", "Provider", "Input $/MTok", "Output $/MTok", "Tokenizer factor", "Price index z", "LOBO price-implied (T)", "Price log weight", "Final central (T)", "Error low (T)", "Error high (T)", "Status", "Method note", "Price source"]];
header(estimates.getRange("A5:R5"));
const estStart = 6;
const estEnd = estStart + modelRows.length - 1;
const pmByModel = new Map(priceFitModels.map((model, idx) => [model, pmStart + idx]));
for (let idx = 0; idx < modelRows.length; idx += 1) {
  const row = estStart + idx;
  const m = modelRows[idx];
  estimates.getRange(`A${row}:I${row}`).values = [[asDate(m.release), m.model, m.aa, m.eci, m.benchmarkT, m.provider, m.input, m.output, m.tokenFactor]];
  if (m.input != null && m.output != null) estimates.getRange(`J${row}`).formulas = [[`=SQRT(G${row}*H${row})*I${row}`]];
  const pmRow = pmByModel.get(m.model);
  if (pmRow) {
    estimates.getRange(`K${row}`).formulas = [[`='Price Model'!N${pmRow}`]];
    estimates.getRange(`L${row}`).formulas = [[`='Price Model'!$B$12`]];
    estimates.getRange(`M${row}`).formulas = [[`=POWER(E${row},1-L${row})*POWER(K${row},L${row})`]];
  } else {
    estimates.getRange(`L${row}`).values = [[0]];
    estimates.getRange(`M${row}`).formulas = [[`=E${row}`]];
  }
  const disclosed = m.status.includes("anchor");
  if (!disclosed) {
    estimates.getRange(`N${row}`).formulas = [[`=M${row}/'Anchors & Method'!$B$12`]];
    estimates.getRange(`O${row}`).formulas = [[`=M${row}*'Anchors & Method'!$B$12`]];
  }
  estimates.getRange(`P${row}:R${row}`).values = [[m.status, m.note, m.priceSource]];
}
body(estimates.getRange(`A6:R${estEnd}`));
estimates.getRange(`A6:A${estEnd}`).format.numberFormat = "yyyy-mm-dd";
estimates.getRange(`C6:D${estEnd}`).format.numberFormat = "0.000";
estimates.getRange(`E6:O${estEnd}`).format.numberFormat = "0.000";
estimates.getRange(`L6:L${estEnd}`).format.numberFormat = "0%";
estimates.getRange(`M6:M${estEnd}`).format.font = { name: "Aptos", size: 10, bold: true, color: C.navy };
estimates.getRange(`P6:R${estEnd}`).format.wrapText = true;
estimates.getRange(`P6:P${estEnd}`).conditionalFormats.add("containsText", { text: "anchor", format: { fill: C.paleBlue, font: { color: C.navy, bold: true } } });
estimates.getRange("A:A").format.columnWidth = 13;
estimates.getRange("B:B").format.columnWidth = 36;
estimates.getRange("C:D").format.columnWidth = 12;
estimates.getRange("E:O").format.columnWidth = 16;
estimates.getRange("P:P").format.columnWidth = 26;
estimates.getRange("Q:Q").format.columnWidth = 65;
estimates.getRange("R:R").format.columnWidth = 50;
estimates.freezePanes.freezeRows(5);
estimates.tables.add(`A5:R${estEnd}`, true, "FrontierEstimatesTable").style = "TableStyleMedium2";

// Anchors and uncertainty method.
title(method, "A1:H1", "Anchors, uncertainty, and model-choice audit");
subtitle(method, "A2:H3", "The primary benchmark remains date-aware and K3-anchored. Grok is used as a fixed disclosed row and validation check, not to rotate the entire frontier scale. Price is commercial evidence and therefore receives only weak cross-fitted weight.");
method.getRange("A5:C5").values = [["Parameter / check", "Value", "Interpretation"]];
header(method.getRange("A5:C5"));
method.getRange("A6:C15").values = [
  ["K3 exact AA v4.1", 57.1123394372091, "UI displays 57"],
  ["K3 disclosed total (T)", k3TotalT, `Exact technical-report disclosure; ${k3ActiveB.toFixed(1)}B activated`],
  ["Grok exact AA v4.1", 53.8265951657731, "UI displays 54"],
  ["Grok disclosed foundation scale (T)", 1.5, "Medium-confidence first-party executive disclosure"],
  ["Grok prior regression estimate (T)", 1.601005407941694, "Only 6.7% above the 1.5T anchor"],
  ["Conservative LOFO log10 RMSE", LOFO_LOG10_RMSE, "Worse leave-family-out scale retained"],
  ["Conservative error factor", null, "10^LOFO RMSE"],
  ["Price log weight", PRICE_WEIGHT, "Weak secondary regularizer"],
  ["Anthropic tokenizer factor", ANTHROPIC_TOKEN_FACTOR, "Approximate same-text token-count adjustment"],
  ["As of", asDate("2026-07-31"), "Pricing and benchmark snapshot date"],
];
method.getRange("B12").formulas = [["=POWER(10,B11)"]];
body(method.getRange("A6:C15"));
method.getRange("A6:A15").format.font = { name: "Aptos", size: 9, bold: true, color: C.text };
method.getRange("B6:B11").format.numberFormat = "0.000";
method.getRange("B12").format.numberFormat = "0.000x";
method.getRange("B13").format.numberFormat = "0%";
method.getRange("B14").format.numberFormat = "0.0x";
method.getRange("B15").format.numberFormat = "yyyy-mm-dd";

section(method, "A18:H18", "Specification decisions");
method.getRange("A20:H21").merge();
method.getRange("A20").values = [["Central estimate: 85% log weight on the existing benchmark/date estimate and 15% on a leave-one-base-out API-price estimate. Standard uncached input/output prices enter through z = sqrt(input × output), with an Anthropic provider effect and a 1.3× same-text tokenizer factor."]];
method.getRange("A22:H23").merge();
method.getRange("A22").values = [["Grok choice: peg the row at 1.5T, but do not force the entire regression through K3 and Grok. The prior model was already within 6.7%; exact two-anchor rotation would add model risk and widen the uncertainty band."]];
method.getRange("A24:H25").merge();
method.getRange("A24").values = [["Identity choice: Opus 4.7 and 4.8 are one shared base observation. Opus 5 is a separate newly pretrained base because Anthropic discloses neither same weights nor a reused base. Fable is modeled as a regular frontier model. Service tiers, fast mode, and reasoning settings are not separate base sizes."]];
method.getRange("A20:H25").format = { fill: C.gray, font: { name: "Aptos", size: 9, color: C.text }, wrapText: true, verticalAlignment: "center" };

section(method, "A28:H28", "Price-model diagnostics");
method.getRange("A30:C30").values = [["Diagnostic", "Result", "Meaning"]];
header(method.getRange("A30:C30"));
method.getRange("A31:A37").values = [["Full-fit price slope"], ["OpenAI intercept, T units"], ["Anthropic provider effect"], ["Full-fit R²"], ["LOBO price RMSE"], ["LOBO multiplicative error"], ["Main parameter error factor"]];
method.getRange("B31:B37").formulas = [["='Price Model'!B6"], ["='Price Model'!B7"], ["='Price Model'!B8"], ["='Price Model'!B9"], ["='Price Model'!B10"], ["='Price Model'!B11"], ["=B12"]];
method.getRange("C31:C37").values = [["Price elasticity after provider adjustment"], ["OpenAI baseline"], ["Commercial/provider offset"], ["Coherence with benchmark estimates, not physical ground truth"], ["Leave-one-base-out log10 error"], ["Price-only cross-fit error"], ["Used for final whiskers; not narrowed by price"]];
body(method.getRange("A31:C37"));
method.getRange("B31:B35").format.numberFormat = "0.0000";
method.getRange("B36:B37").format.numberFormat = "0.000x";
method.getRange("A:A").format.columnWidth = 43;
method.getRange("B:B").format.columnWidth = 24;
method.getRange("C:C").format.columnWidth = 60;
method.getRange("D:H").format.columnWidth = 14;
method.freezePanes.freezeRows(5);

// Executive summary and chart.
title(summary, "A1:Q1", "Frontier parameter estimates — pricing cross-check");
subtitle(summary, "A2:Q3", "Official OpenAI and Anthropic API pricing is now a weak secondary signal. K3 remains the primary disclosed calibration anchor; Grok 4.5 is pegged at the public 1.5T foundation scale. Opus 4.7/4.8 is one shared base, Fable is shown as a regular model, and error bars retain the conservative leave-family-out factor.");
summary.getRange("A5:G5").values = [["Model / base", "Final central (T)", "Error low (T)", "Error high (T)", "Input $/MTok", "Output $/MTok", "Status"]];
header(summary.getRange("A5:G5"));
for (let idx = 0; idx < modelRows.length; idx += 1) {
  const row = 6 + idx;
  const sourceRow = estStart + idx;
  summary.getRange(`A${row}:G${row}`).formulas = [[
    `='Frontier Estimates'!B${sourceRow}`,
    `='Frontier Estimates'!M${sourceRow}`,
    `=IF('Frontier Estimates'!N${sourceRow}="","",'Frontier Estimates'!N${sourceRow})`,
    `=IF('Frontier Estimates'!O${sourceRow}="","",'Frontier Estimates'!O${sourceRow})`,
    `=IF('Frontier Estimates'!G${sourceRow}="","",'Frontier Estimates'!G${sourceRow})`,
    `=IF('Frontier Estimates'!H${sourceRow}="","",'Frontier Estimates'!H${sourceRow})`,
    `='Frontier Estimates'!P${sourceRow}`,
  ]];
}
body(summary.getRange(`A6:G${5 + modelRows.length}`));
summary.getRange(`B6:F${5 + modelRows.length}`).format.numberFormat = "0.000";
summary.getRange(`B6:B${5 + modelRows.length}`).format.font = { name: "Aptos", size: 10, bold: true, color: C.navy };
summary.getRange(`G6:G${5 + modelRows.length}`).conditionalFormats.add("containsText", { text: "anchor", format: { fill: C.paleBlue, font: { color: C.navy, bold: true } } });
summary.getRange("A:A").format.columnWidth = 38;
summary.getRange("B:F").format.columnWidth = 17;
summary.getRange("G:G").format.columnWidth = 27;
summary.getRange("H:H").format.columnWidth = 3;
summary.getRange("I:Q").format.columnWidth = 13;
const chart = summary.charts.add("bar", { title: "Price-informed total parameters (T)", hasLegend: false });
const series = chart.series.add("Total parameters (T)");
series.categoryFormula = `'Executive Summary'!$A$6:$A$${5 + modelRows.length}`;
series.formula = `'Executive Summary'!$B$6:$B$${5 + modelRows.length}`;
series.fill = C.teal;
chart.title = "Price-informed frontier totals (T)";
chart.hasLegend = false;
chart.yAxis = { numberFormatCode: "0.0\"T\"" };
chart.setPosition("I5", "Q25");

summary.getRange("A20:Q22").merge();
summary.getRange("A20").values = [[`Headline: Opus 5 is added as a distinct regression target using exact AA/ECI/date evidence and weak price smoothing. K3 =${k3TotalT.toFixed(2)}T and Grok 4.5 =1.50T remain disclosed anchors; Opus 4.7/4.8 remains a separate shared base.`]];
summary.getRange("A20:Q22").format = { fill: C.paleTeal, font: { name: "Aptos", size: 10, bold: true, color: C.navy }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: C.teal } } };
summary.getRange("A24:Q26").merge();
summary.getRange("A24").values = [["Error bars: ×/÷3.03 LOFO band, symmetric in log space. K3 and Grok are disclosed anchors, so they have no regression whiskers. Price shifts centers by at most 4.2% and does not narrow the band."]];
summary.getRange("A24:Q26").format = { fill: C.palePink, font: { name: "Aptos", size: 9, italic: true, color: C.text }, wrapText: true, verticalAlignment: "center", borders: { left: { style: "medium", color: "#E03D90" } } };
summary.freezePanes.freezeRows(5);

// Sources and audit trail.
title(sources, "A1:D1", "Sources and audit trail");
subtitle(sources, "A2:D3", "Plain-text URLs and local paths are retained for reproducibility. Opus 5 evidence and pricing were verified and frozen on 2026-07-31.");
sources.getRange("A5:D5").values = [["Source", "URL / path", "Use", "Caveat / snapshot"]];
header(sources.getRange("A5:D5"));
const sourceRows = [
  ["OpenAI official API pricing", "https://developers.openai.com/api/docs/pricing", "GPT-5.6 Sol/Terra/Luna and GPT-5.5 input/output prices", "Standard short-context rates as of 2026-07-17"],
  ["Anthropic official pricing", "https://platform.claude.com/docs/en/about-claude/pricing", "Fable, Opus, Sonnet prices and tokenizer note", "Sonnet introductory $2/$10 used; future price excluded"],
  ["SpaceXAI Grok 4.5 model page", "https://docs.x.ai/developers/models/grok-4.5", "Grok API $2/$6 validation price", "Not used in OpenAI/Anthropic fit"],
  ["Grok 4.5 1.5T disclosure", "https://x.com/elonmusk/status/2071184354756477041", "Public first-party 1.5T V9 foundation scale", "Executive disclosure; not model-card-grade exactness"],
  ["Claude Opus 5 evidence bundle", opus5EvidencePath, "Exact release, API identity/pricing, AA max score and fallback, public ECI with 90% CI, availability checks, and source hashes", "Frozen 2026-07-31; distinct undisclosed base; no METR, no-CoT, IKP, public training compute, architecture, or parameter-count observation"],
  ["Kimi K3 technical report evidence", k3EvidencePath, `${k3TotalT.toFixed(2)}T total and ${k3ActiveB.toFixed(1)}B activated`, k3ParameterSource],
  ["Artificial Analysis K3", "https://artificialanalysis.ai/models/kimi-k3", "K3 exact AA v4.1 score", "UI displays rounded 57"],
  ["Artificial Analysis Grok 4.5", "https://artificialanalysis.ai/models/grok-4-5", "Grok AA v4.1 score and release date", "AA parameter fields remain blank"],
  ["Supplied ECI workbook", `${workDir}/sources/input_eci_parameter_regression_workbook_2026-07-17.xlsx`, "Dated and no-date ECI cross-check", "Read-only reference"],
  ["Prior K3-calibrated workbook", `${outputDir}/k3_calibrated_frontier_parameter_crosscheck_2026-07-17.xlsx`, "Primary benchmark estimates before price smoothing", "Opus rows collapsed here; Fable treated as regular per user"],
  ["OpenAI pricing attachment", `${workDir}/sources/input_openai_api_pricing_docs_2026-07-17.txt`, "User-provided official-doc snapshot", "Cross-checked with live official page"],
  ["Anthropic pricing attachment", `${workDir}/sources/input_anthropic_api_pricing_docs_2026-07-17.txt`, "User-provided official-doc snapshot", "Cross-checked with live official page"],
].map(([source, location, use, caveat]) => [
  source,
  portableLocalPath(location),
  use,
  caveat,
]);
sources.getRange(`A6:D${5 + sourceRows.length}`).values = sourceRows;
body(sources.getRange(`A6:D${5 + sourceRows.length}`));
sources.getRange(`B6:D${5 + sourceRows.length}`).format.wrapText = true;
sources.getRange("A:A").format.columnWidth = 38;
sources.getRange("B:B").format.columnWidth = 65;
sources.getRange("C:C").format.columnWidth = 55;
sources.getRange("D:D").format.columnWidth = 48;
sources.freezePanes.freezeRows(5);
sources.tables.add(`A5:D${5 + sourceRows.length}`, true, "SourcesTable").style = "TableStyleMedium2";

wb.comments.setSelf({ displayName: "Codex" });
wb.comments.addThread({ cell: priceModel.getRange("B12") }, "Price is capped at 15% log weight because API prices reflect product strategy, latency, tokenizer choice, and inference service design—not only physical parameter count.");
wb.comments.addThread({ cell: method.getRange("B9") }, "The 1.5T figure is a public first-party executive disclosure of the V9 foundation scale. It is less formal than a technical report and should not be used as an active-parameter count.");
wb.comments.addThread({ cell: estimates.getRange(`M${estimateSheetByModel.get("Claude Opus 4.7 / 4.8 shared base")}`) }, "Opus 4.7 and 4.8 are treated as one base. Their separate benchmark-derived priors are collapsed geometrically before pricing is applied.");

// Verification, rendering, and export.
const checks = [
  ["Executive Summary", "A1:Q26"],
  ["Frontier Estimates", `A1:R${estEnd}`],
  ["API Prices", `A1:L${5 + priceRows.length}`],
  ["Price Model", `A1:O${pmEnd}`],
  ["Anchors & Method", "A1:H37"],
  ["Sources", `A1:D${5 + sourceRows.length}`],
];
for (const [sheetName, range] of checks) {
  const inspected = await wb.inspect({ kind: "table", range: `${sheetName}!${range}`, include: "values,formulas", tableMaxRows: 45, tableMaxCols: 18, maxChars: 14000 });
  console.log(inspected.ndjson);
}
const errors = await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
console.log(errors.ndjson);

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });
for (const [sheetName, range] of checks) {
  const rendered = await wb.render({ sheetName, range, scale: 1.1, format: "png" });
  const file = sheetName.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  await fs.writeFile(`${qaDir}/${file}.png`, new Uint8Array(await rendered.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, qaDir, models: modelRows.length }, null, 2));
