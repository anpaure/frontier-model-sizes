import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
import path from "node:path";
import { fileURLToPath } from "node:url";

const runtimeDir = path.dirname(fileURLToPath(import.meta.url));
const workDir = path.resolve(runtimeDir, "..");
const inputPath =
  process.env.ECI_INPUT_WORKBOOK ??
  path.join(
    workDir,
    "sources",
    "input_eci_parameter_regression_workbook_2026-07-17.xlsx",
  );
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

for (const req of [
  { kind: "sheet", include: "id,name", maxChars: 6000 },
  { kind: "table", maxChars: 12000, tableMaxRows: 4, tableMaxCols: 18, tableMaxCellChars: 120 },
  { kind: "table", sheetId: "Frontier Estimates", range: "A1:R27", include: "values,formulas", maxChars: 30000, tableMaxRows: 30, tableMaxCols: 18, tableMaxCellChars: 120 },
  { kind: "table", sheetId: "Regression Model", range: "A1:N31", include: "values,formulas", maxChars: 20000, tableMaxRows: 35, tableMaxCols: 14, tableMaxCellChars: 160 },
  { kind: "table", sheetId: "Sources & Method", range: "A1:C19", include: "values,formulas", maxChars: 12000, tableMaxRows: 25, tableMaxCols: 3, tableMaxCellChars: 500 },
  { kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "formula errors", maxChars: 12000 },
]) {
  const out = await workbook.inspect(req);
  console.log(`\n### ${req.kind} ${req.sheetId ?? ""} ${req.range ?? ""}`);
  console.log(out.ndjson);
}
