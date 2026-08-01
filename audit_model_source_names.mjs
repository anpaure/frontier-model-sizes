import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";
import path from "node:path";
import { fileURLToPath } from "node:url";

const workDir = path.dirname(fileURLToPath(import.meta.url));
const horizonPath = `${workDir}/outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/horizon_informed_frontier_parameter_model_2026-07-17.xlsx`;
const eciPath = `${workDir}/sources/input_eci_parameter_regression_workbook_2026-07-17.xlsx`;

const h = await SpreadsheetFile.importXlsx(await FileBlob.load(horizonPath));
const e = await SpreadsheetFile.importXlsx(await FileBlob.load(eciPath));
const noCot = h.worksheets.getItem("No-CoT Evidence").getUsedRange().values;
const eci = e.worksheets.getItem("ECI Graph Data").getUsedRange().values;
const eciHeaders = Object.fromEntries(eci[0].map((v, i) => [String(v), i]));
const eciNames = eci.slice(1).map((row) => String(row[eciHeaders.Model] ?? "")).filter(Boolean);

const norm = (s) => s.toLowerCase().replace(/instruct|it|terminus|\(0324\)|\(2507\)|[-_.()]/g, " ").replace(/\bclaude\b/g, "").replace(/\s+/g, " ").trim();
const tokens = (s) => new Set(norm(s).split(" ").filter(Boolean));
const score = (a, b) => {
  const A = tokens(a); const B = tokens(b);
  const intersection = [...A].filter((x) => B.has(x)).length;
  return intersection / Math.max(1, new Set([...A, ...B]).size);
};

const paperNames = [];
for (let r = 6; r < noCot.length; r += 1) {
  const name = String(noCot[r]?.[0] ?? "");
  if (name && name !== "Developer" && !name.startsWith("Source archive")) paperNames.push(name);
}
for (let r = 23; r < noCot.length; r += 1) {
  const name = String(noCot[r]?.[1] ?? "");
  if (name && name !== "Model") paperNames.push(name);
}

for (const name of [...new Set(paperNames)]) {
  const exact = eciNames.includes(name) ? name : "";
  const ranked = eciNames.map((candidate) => [candidate, score(name, candidate)]).sort((a, b) => b[1] - a[1]).slice(0, 4);
  console.log(JSON.stringify({ paper: name, exact, candidates: ranked }));
}
