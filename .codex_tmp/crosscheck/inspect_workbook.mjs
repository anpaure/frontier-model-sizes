import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const workDir = path.resolve(scriptDir, "../..");
const workbookPath = path.join(
  workDir,
  "sources",
  "input_eci_parameter_regression_workbook_2026-07-17.xlsx",
);
const previewDir = path.join(scriptDir, "previews");
const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const sheetList = await workbook.inspect({ kind: "sheet", include: "id,name,range,index", maxChars: 10000 });
console.log("SHEETS");
console.log(sheetList.ndjson);

const checks = [
  ["Dashboard", "A1:F28"],
  ["Frontier Estimates", "A1:R27"],
  ["Open Models 12mo", "A1:P18"],
  ["Regression Data", "A1:M59"],
  ["Regression Model", "A1:N31"],
  ["ECI Graph Data", "A1:K20"],
  ["ECI Graph Data", "A190:K212"],
  ["Benchmarked Models", "A1:R20"],
  ["Sources & Method", "A1:C19"],
];

for (const [sheetName, address] of checks) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const range = sheet.getRange(address);
  console.log(`RANGE\t${sheetName}\t${address}`);
  console.log(JSON.stringify({ values: range.values, formulas: range.formulas }));
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  maxChars: 10000,
  summary: "formula error scan",
});
console.log("ERRORS");
console.log(errors.ndjson);

await fs.mkdir(previewDir, { recursive: true });
const previewChecks = [
  ["Dashboard", "A1:P33"],
  ["Frontier Estimates", "A1:R27"],
  ["Open Models 12mo", "A1:P18"],
  ["Regression Data", "A1:M59"],
  ["Regression Model", "A1:N31"],
  ["Sources & Method", "A1:C19"],
];

for (const [sheetName, address] of previewChecks) {
  const image = await workbook.render({ sheetName, range: address, scale: 1.25, format: "png" });
  const safeName = sheetName.replace(/[^a-z0-9]+/gi, "_").replace(/^_+|_+$/g, "");
  await fs.writeFile(path.join(previewDir, `${safeName}.png`), new Uint8Array(await image.arrayBuffer()));
}

console.log("PREVIEWS");
console.log(JSON.stringify(previewChecks));
