import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
import nodePath from "node:path";
import { fileURLToPath } from "node:url";

const workDir = nodePath.dirname(fileURLToPath(import.meta.url));
const workbookPath = `${workDir}/sources/input_eci_parameter_regression_workbook_2026-07-17.xlsx`;
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));

const sheets = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 12000,
});
console.log(sheets.ndjson);

for (const name of [
  "Benchmarked Models",
  "ECI Graph Data",
  "Regression Data",
  "Regression Model",
  "Sources & Method",
]) {
  try {
    const table = await workbook.inspect({
      kind: "table",
      sheetId: name,
      range: "A1:R30",
      include: "values,formulas",
      tableMaxRows: 30,
      tableMaxCols: 18,
      tableMaxCellChars: 160,
      maxChars: 30000,
    });
    console.log(JSON.stringify({ sheet: name }));
    console.log(table.ndjson);
  } catch (error) {
    console.log(JSON.stringify({ sheet: name, error: String(error) }));
  }
}
