import { readFile, readdir, writeFile } from "node:fs/promises";
import { extname } from "node:path";
import { fileURLToPath } from "node:url";

const output = fileURLToPath(new URL("../dist/client/", import.meta.url));
const projectPath = "/frontier-model-sizes";
const textExtensions = new Set([".css", ".html", ".js", ".json", ".rsc", ".txt"]);

async function rewrite(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  await Promise.all(entries.map(async (entry) => {
    const path = `${directory}/${entry.name}`;
    if (entry.isDirectory()) return rewrite(path);
    if (!textExtensions.has(extname(entry.name))) return;

    const source = await readFile(path, "utf8");
    const updated = source
      .replaceAll('"/assets/_vinext_fonts/', `"${projectPath}/assets/_vinext_fonts/`)
      .replaceAll("'/assets/_vinext_fonts/", `'${projectPath}/assets/_vinext_fonts/`)
      .replaceAll("(/assets/_vinext_fonts/", `(${projectPath}/assets/_vinext_fonts/`);
    if (updated !== source) await writeFile(path, updated);
  }));
}

await rewrite(output);
