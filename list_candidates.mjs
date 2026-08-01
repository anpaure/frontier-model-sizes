import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
import path from "node:path";
import { fileURLToPath } from "node:url";

const workDir = path.dirname(fileURLToPath(import.meta.url));
const workbookPath = process.env.ECI_WORKBOOK_PATH || path.resolve(workDir, "../../outputs/eci_graph_20260716/epoch_eci_graph_data_from_har.xlsx");
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const values = workbook.worksheets.getItem("ECI Graph Data").getUsedRange().values;
const headers = values[0];
const index = Object.fromEntries(headers.map((header, column) => [header, column]));
const isoDate = (value) => {
  if (value instanceof Date) return value.toISOString().slice(0, 10);
  if (typeof value === "number") {
    return new Date((value - 25569) * 86400000).toISOString().slice(0, 10);
  }
  return String(value ?? "").slice(0, 10);
};
const candidates = values.slice(1).filter((row) =>
  isoDate(row[index["Release Date"]]) >= "2025-07-16"
    && String(row[index.Accessibility] ?? "").includes("Open weights"),
);
for (const row of candidates) {
  console.log([
    isoDate(row[index["Release Date"]]),
    row[index.Model],
    row[index["ECI Score"]],
    row[index.Accessibility],
    row[index["Model Version ID"]],
    row[index["Source CSV Row"]],
  ].join("\t"));
}
console.log(`COUNT\t${candidates.length}`);

console.log("\nALL OPEN-WEIGHT MODELS");
const allOpen = values.slice(1)
  .filter((row) => String(row[index.Accessibility] ?? "").includes("Open weights"))
  .sort((a, b) => isoDate(a[index["Release Date"]]).localeCompare(isoDate(b[index["Release Date"]])));
for (const row of allOpen) {
  console.log([
    isoDate(row[index["Release Date"]]),
    row[index.Model],
    row[index["ECI Score"]],
    row[index["90% CI Low"]],
    row[index["90% CI High"]],
    row[index.Organization],
    row[index["Model Version ID"]],
  ].join("\t"));
}
console.log(`COUNT\t${allOpen.length}`);

console.log("\nCLOSED FRONTIER (ECI >= 150)");
const frontier = values.slice(1)
  .filter((row) => !String(row[index.Accessibility] ?? "").includes("Open weights")
    && Number(row[index["ECI Score"]]) >= 150)
  .sort((a, b) => Number(b[index["ECI Score"]]) - Number(a[index["ECI Score"]]));
for (const row of frontier) {
  console.log([
    isoDate(row[index["Release Date"]]),
    row[index.Model],
    row[index["ECI Score"]],
    row[index.Organization],
    row[index.Accessibility],
  ].join("\t"));
}
console.log(`COUNT\t${frontier.length}`);
