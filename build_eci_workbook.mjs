import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const workDir = path.dirname(fileURLToPath(import.meta.url));
const sourcePageUrl = "https://epoch.ai/eci";
const sourceCsvUrl = "https://epoch.ai/data/benchmarked_models.csv";
const chartCodeUrl = "https://epoch.ai/generated/benchmarks-viz.js";
const harPath = process.env.EPOCH_HAR_PATH || path.join(os.homedir(), "Downloads", "epoch.ai.har");
const outputDir = process.env.ECI_OUTPUT_DIR || path.resolve(workDir, "../../outputs/eci_graph_20260716");
const qaDir = path.join(workDir, "qa");
const outputPath = `${outputDir}/epoch_eci_graph_data_from_har.xlsx`;

const har = JSON.parse(await fs.readFile(harPath, "utf8"));
const sourceEntry = har.log.entries.find((entry) =>
  entry.request?.url === sourceCsvUrl && entry.response?.content?.text,
);
if (!sourceEntry) {
  throw new Error(`The HAR does not contain a response body for ${sourceCsvUrl}`);
}
const content = sourceEntry.response.content;
const csvText = content.encoding === "base64"
  ? Buffer.from(content.text, "base64").toString("utf8")
  : content.text;
const capturedAt = sourceEntry.startedDateTime;

const imported = await Workbook.fromCSV(csvText, { sheetName: "Raw Import" });
const rawSheet = imported.worksheets.getItem("Raw Import");
const rawValues = rawSheet.getUsedRange().values;
const headers = rawValues[0].map((value) => String(value ?? "").trim());
const index = Object.fromEntries(headers.map((header, column) => [header, column]));

const cell = (row, header) => row[index[header]];
const textValue = (value) => String(value ?? "").trim();
const numericValue = (value) => {
  if (value === null || value === undefined || textValue(value) === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
};
const dateValue = (value) => {
  if (value instanceof Date && !Number.isNaN(value.getTime())) return value;
  const valueText = textValue(value);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(valueText)) return null;
  const parsed = new Date(`${valueText}T00:00:00Z`);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
};

const seenModels = new Set();
const records = [];
for (let rowIndex = 1; rowIndex < rawValues.length; rowIndex += 1) {
  const row = rawValues[rowIndex];
  const modelGroup = textValue(cell(row, "model_group"));
  const model = textValue(cell(row, "Model"));
  const deduplicationKey = modelGroup || model;

  if (seenModels.has(deduplicationKey)) continue;
  seenModels.add(deduplicationKey);

  const eciScore = numericValue(cell(row, "eci"));
  const versionReleaseDate = dateValue(cell(row, "Version release date"));
  const publicationDate = dateValue(cell(row, "Publication date"));
  const releaseDate = versionReleaseDate || publicationDate;
  const modelVersionId = textValue(cell(row, "id_model_version"));

  if (eciScore === null || !releaseDate || !modelVersionId) continue;

  const slug = textValue(cell(row, "slug"));
  records.push({
    releaseDate,
    model: deduplicationKey || modelVersionId,
    modelVersionId,
    organization: textValue(cell(row, "Organization")),
    country: textValue(cell(row, "Country (of organization)")),
    accessibility: textValue(cell(row, "Model accessibility")),
    eciScore,
    ciLow: numericValue(cell(row, "eci_ci_low")),
    ciHigh: numericValue(cell(row, "eci_ci_high")),
    trainingCompute: numericValue(cell(row, "Training compute (FLOP)")),
    confidence: textValue(cell(row, "Confidence")),
    slug,
    modelUrl: slug ? `https://epoch.ai/models/${slug}` : "",
    sourceUrl: sourceCsvUrl,
    dateFieldUsed: versionReleaseDate ? "Version release date" : "Publication date",
    sourceRow: rowIndex + 1,
  });
}

records.sort((a, b) => a.releaseDate - b.releaseDate || a.eciScore - b.eciScore || a.model.localeCompare(b.model));

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const data = workbook.worksheets.add("ECI Graph Data");

const navy = "#243838";
const teal = "#00A5A6";
const tealDark = "#087879";
const pink = "#E03D90";
const paleTeal = "#E8F4F4";
const paleGray = "#F3F6F6";
const midGray = "#D2DCDC";
const charcoal = "#3A4848";
const white = "#FFFFFF";

summary.showGridLines = false;
summary.getRange("A1:H1").merge();
summary.getRange("A1").values = [["Epoch Capabilities Index: graph data"]];
summary.getRange("A1:H1").format = {
  fill: navy,
  font: { name: "Aptos Display", size: 20, bold: true, color: white },
  verticalAlignment: "center",
};
summary.getRange("A1:H1").format.rowHeight = 38;

summary.getRange("A3:H3").merge();
summary.getRange("A3").values = [["Exact records used by Epoch AI's main ECI Score vs Release Date chart, extracted from the supplied HAR capture."]];
summary.getRange("A3:H3").format = {
  font: { name: "Aptos", size: 11, color: charcoal },
  verticalAlignment: "center",
};
summary.getRange("A3:H3").format.rowHeight = 24;

summary.getRange("A5:B5").values = [["Dataset summary", "Value"]];
summary.getRange("A6:A11").values = [
  ["Plotted model records"],
  ["Earliest release"],
  ["Latest release"],
  ["Highest ECI score"],
  ["Lowest ECI score"],
  ["Records with training compute"],
];
const lastDataRow = records.length + 1;
summary.getRange("B6:B11").formulas = [
  [`=COUNTA('ECI Graph Data'!$A$2:$A$${lastDataRow})`],
  [`=MIN('ECI Graph Data'!$A$2:$A$${lastDataRow})`],
  [`=MAX('ECI Graph Data'!$A$2:$A$${lastDataRow})`],
  [`=MAX('ECI Graph Data'!$G$2:$G$${lastDataRow})`],
  [`=MIN('ECI Graph Data'!$G$2:$G$${lastDataRow})`],
  [`=COUNT('ECI Graph Data'!$K$2:$K$${lastDataRow})`],
];
summary.getRange("A5:B5").format = {
  fill: tealDark,
  font: { name: "Aptos", size: 10, bold: true, color: white },
  borders: { bottom: { style: "medium", color: tealDark } },
};
summary.getRange("A6:B11").format = {
  font: { name: "Aptos", size: 10, color: charcoal },
  borders: { insideHorizontal: { style: "thin", color: midGray } },
};
summary.getRange("B6:B11").format.font = { name: "Aptos", size: 10, bold: true, color: navy };
summary.getRange("B7:B8").format.numberFormat = "yyyy-mm-dd";
summary.getRange("B9:B10").format.numberFormat = "0.00";
summary.getRange("B6:B6").format.numberFormat = "#,##0";
summary.getRange("B11:B11").format.numberFormat = "#,##0";

summary.getRange("D5:H5").merge();
summary.getRange("D5").values = [["Chart extraction rules"]];
summary.getRange("D5:H5").format = {
  fill: paleTeal,
  font: { name: "Aptos", size: 11, bold: true, color: tealDark },
  borders: { bottom: { style: "medium", color: teal } },
};
summary.getRange("D6:H9").merge(true);
summary.getRange("D6:D9").values = [
  ["1. Deduplicate in source order by model_group, falling back to Model."],
  ["2. Keep rows with a numeric ECI score and a model-version identifier."],
  ["3. Use Version release date; fall back to Publication date when needed."],
  ["4. Keep the reported 90% confidence interval and metadata without rounding."],
];
summary.getRange("D6:H9").format = {
  fill: paleGray,
  font: { name: "Aptos", size: 10, color: charcoal },
  wrapText: true,
  verticalAlignment: "center",
  borders: { insideHorizontal: { style: "thin", color: midGray } },
};
summary.getRange("D6:H9").format.rowHeight = 28;

summary.getRange("A14:H14").merge();
summary.getRange("A14").values = [["Sources"]];
summary.getRange("A14:H14").format = {
  fill: navy,
  font: { name: "Aptos", size: 11, bold: true, color: white },
};
summary.getRange("A15:B19").values = [
  ["ECI page", sourcePageUrl],
  ["Source CSV", sourceCsvUrl],
  ["Chart implementation", chartCodeUrl],
  ["HAR file", harPath],
  ["Captured", new Date(capturedAt)],
];
summary.getRange("A15:A19").format = { font: { name: "Aptos", size: 10, bold: true, color: charcoal } };
summary.getRange("B15:B19").format = { font: { name: "Aptos", size: 10, color: tealDark }, wrapText: true };
summary.getRange("B19").format.numberFormat = "yyyy-mm-dd hh:mm:ss";
summary.getRange("A21:H22").merge();
summary.getRange("A21").values = [["The data sheet contains one row per plotted model record. Scores are the HAR snapshot and may move when Epoch refits ECI with new model and benchmark data."]];
summary.getRange("A21:H22").format = {
  fill: "#FFF4FA",
  font: { name: "Aptos", size: 10, italic: true, color: charcoal },
  wrapText: true,
  verticalAlignment: "center",
  borders: { left: { style: "medium", color: pink } },
};

summary.getRange("A1:H22").format.verticalAlignment = "center";
summary.getRange("A:A").format.columnWidth = 27;
summary.getRange("B:B").format.columnWidth = 44;
summary.getRange("C:C").format.columnWidth = 3;
summary.getRange("D:H").format.columnWidth = 18;
summary.freezePanes.freezeRows(1);

const dataHeaders = [
  "Release Date",
  "Model",
  "Model Version ID",
  "Organization",
  "Country",
  "Accessibility",
  "ECI Score",
  "90% CI Low",
  "90% CI High",
  "CI Width",
  "Training Compute (FLOP)",
  "Confidence",
  "Model Slug",
  "Model URL",
  "Source URL",
  "Date Field Used",
  "Source CSV Row",
];
const dataRows = records.map((record) => [
  record.releaseDate,
  record.model,
  record.modelVersionId,
  record.organization,
  record.country,
  record.accessibility,
  record.eciScore,
  record.ciLow,
  record.ciHigh,
  null,
  record.trainingCompute,
  record.confidence,
  record.slug,
  record.modelUrl,
  record.sourceUrl,
  record.dateFieldUsed,
  record.sourceRow,
]);

data.showGridLines = false;
data.getRange("A1:Q1").values = [dataHeaders];
if (dataRows.length > 0) data.getRange(`A2:Q${lastDataRow}`).values = dataRows;
data.getRange("A1:Q1").format = {
  fill: navy,
  font: { name: "Aptos", size: 10, bold: true, color: white },
  wrapText: true,
  verticalAlignment: "center",
  borders: { bottom: { style: "medium", color: teal } },
};
data.getRange("A1:Q1").format.rowHeight = 34;
data.getRange(`A2:Q${lastDataRow}`).format = {
  font: { name: "Aptos", size: 9, color: charcoal },
  verticalAlignment: "center",
  borders: { insideHorizontal: { style: "thin", color: "#E5ECEC" } },
};
data.getRange(`A2:A${lastDataRow}`).format.numberFormat = "yyyy-mm-dd";
data.getRange(`G2:J${lastDataRow}`).format.numberFormat = "0.00";
data.getRange(`K2:K${lastDataRow}`).format.numberFormat = "0.00E+00";
data.getRange(`Q2:Q${lastDataRow}`).format.numberFormat = "#,##0";
data.getRange(`N2:O${lastDataRow}`).format.font = { name: "Aptos", size: 9, color: tealDark };
data.getRange(`J2:J${lastDataRow}`).formulas = records.map((_, offset) => {
  const rowNumber = offset + 2;
  return [`=IF(OR(H${rowNumber}="",I${rowNumber}=""),"",I${rowNumber}-H${rowNumber})`];
});

data.getRange("A:A").format.columnWidth = 13;
data.getRange("B:B").format.columnWidth = 28;
data.getRange("C:C").format.columnWidth = 31;
data.getRange("D:D").format.columnWidth = 22;
data.getRange("E:E").format.columnWidth = 24;
data.getRange("F:F").format.columnWidth = 25;
data.getRange("G:J").format.columnWidth = 11;
data.getRange("K:K").format.columnWidth = 19;
data.getRange("L:L").format.columnWidth = 13;
data.getRange("M:M").format.columnWidth = 21;
data.getRange("N:O").format.columnWidth = 43;
data.getRange("P:P").format.columnWidth = 20;
data.getRange("Q:Q").format.columnWidth = 13;
data.getRange(`B2:F${lastDataRow}`).format.wrapText = true;
data.getRange(`N2:P${lastDataRow}`).format.wrapText = true;
data.getRange(`A2:Q${lastDataRow}`).format.rowHeight = 22;

const dataTable = data.tables.add(`A1:Q${lastDataRow}`, true, "ECIGraphDataTable");
dataTable.showFilterButton = true;
dataTable.showBandedRows = true;
data.freezePanes.freezeRows(1);
data.freezePanes.freezeColumns(2);

data.getRange(`G2:G${lastDataRow}`).conditionalFormats.add("colorScale", {
  colors: ["#F2F6F6", "#77CACA", "#E03D90"],
});

await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(qaDir, { recursive: true });

const summaryInspect = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:H22",
  include: "values,formulas",
  tableMaxRows: 24,
  tableMaxCols: 9,
  maxChars: 9000,
});
const dataInspect = await workbook.inspect({
  kind: "table",
  range: `ECI Graph Data!A1:Q8`,
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 18,
  maxChars: 9000,
});
const errorInspect = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});

const summaryPreview = await workbook.render({ sheetName: "Summary", range: "A1:H22", scale: 1.5, format: "png" });
const dataPreview = await workbook.render({ sheetName: "ECI Graph Data", range: "A1:Q14", scale: 1.25, format: "png" });
await fs.writeFile(`${qaDir}/summary.png`, new Uint8Array(await summaryPreview.arrayBuffer()));
await fs.writeFile(`${qaDir}/data.png`, new Uint8Array(await dataPreview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });

console.log(JSON.stringify({
  outputPath,
  harPath,
  capturedAt,
  sourceRows: rawValues.length - 1,
  plottedModelRows: records.length,
  dateRange: [records[0]?.releaseDate.toISOString().slice(0, 10), records.at(-1)?.releaseDate.toISOString().slice(0, 10)],
  eciRange: [Math.min(...records.map((record) => record.eciScore)), Math.max(...records.map((record) => record.eciScore))],
  recordsWithoutCI: records.filter((record) => record.ciLow === null || record.ciHigh === null).length,
}, null, 2));
console.log(summaryInspect.ndjson);
console.log(dataInspect.ndjson);
console.log(errorInspect.ndjson);
