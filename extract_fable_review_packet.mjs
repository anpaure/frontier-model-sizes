import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workDir = path.dirname(fileURLToPath(import.meta.url));
const threadId = "019f6c42-2d53-7743-ab07-6293e2618dd7";
const workbooks = [
  {
    label: "USER-SUPPLIED ECI WORKBOOK",
    path: `${workDir}/sources/input_eci_parameter_regression_workbook_2026-07-17.xlsx`,
  },
  {
    label: "FINAL PRICE-INFORMED CROSS-CHECK",
    path: `${workDir}/outputs/${threadId}/price_informed_frontier_parameter_crosscheck_2026-07-17.xlsx`,
  },
];

const chunks = [];
for (const item of workbooks) {
  const blob = await FileBlob.load(item.path);
  const workbook = await SpreadsheetFile.importXlsx(blob);
  const overview = await workbook.inspect({
    kind: "workbook,sheet,table",
    include: "id,name,values,formulas",
    tableMaxRows: 160,
    tableMaxCols: 40,
    tableMaxCellChars: 500,
    maxChars: 240000,
  });
  const formulas = await workbook.inspect({
    kind: "formula",
    include: "values,formulas",
    options: { maxResults: 5000 },
    maxChars: 180000,
  });
  chunks.push([
    `===== ${item.label} =====`,
    `PATH: ${item.path}`,
    "--- WORKBOOK / SHEET / TABLE INSPECTION ---",
    overview.ndjson,
    "--- FORMULA INSPECTION ---",
    formulas.ndjson,
  ].join("\n"));
}

const outputPath = `${workDir}/fable_review_workbook_dump.txt`;
await fs.writeFile(outputPath, chunks.join("\n\n"), "utf8");
console.log(outputPath);
