import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const output = new URL("../dist/client/", import.meta.url);

test("GitHub Pages export is complete and repository-path aware", async () => {
  const html = await readFile(new URL("index.html", output), "utf8");
  const rsc = await readFile(new URL("index.rsc", output), "utf8");

  assert.match(html, /Frontier model-size estimates/);
  assert.match(html, /frontier-model-sizes\/assets\//);
  assert.match(html, /frontier-model-sizes\/assets\/_vinext_fonts\//);
  assert.match(html, /https:\/\/anpaure\.github\.io\/frontier-model-sizes\//);
  assert.match(html, /https:\/\/anpaure\.github\.io\/frontier-model-sizes\/favicon\.svg/);
  assert.match(html, /https:\/\/github\.com\/anpaure\/frontier-model-sizes/);
  assert.doesNotMatch(html, /(?:href=|url\()["']?\/assets\//);
  assert.doesNotMatch(rsc, /(?:href|src|url)[^\n]*["'(]\/assets\//);

  await access(new URL("og-v3.png", output));
  await access(new URL(".nojekyll", output));
});
