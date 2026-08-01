# IKP replication / audit

Replication and methodology audit of the **Incompressible Knowledge Probes (IKP)**
paper and codebase by Bojie Li.

- Original paper & code: https://github.com/19PINE-AI/ikp
- Write-up: https://www.lesswrong.com/posts/veFMEzDDyWaer2Sms/sanity-checking-incompressible-knowledge-probes

This repo holds the replication's own code, figures, and derived data. To make it
runnable out of the box it also bundles a few **unmodified** input files from the
upstream IKP repo (probe set and model configs); these are Li's data, redistributed
here verbatim for convenience — the upstream repo is the source of record.

## Quick start

```bash
pip install -r requirements.txt

# Regenerate the calibration figures (no API key, no large downloads)
python runs/plot_cliff.py          # -> runs/calibration_cliff.png   (n = 97)
python runs/plot_uncertainty.py    # -> runs/calibration_uncertainty.png

# Run the Wikidata ambiguity audit (hits the public Wikidata SPARQL endpoint)
python scripts/wikidata_sparql_recheck.py --help
python scripts/wikidata_dominance_check.py
```

## Layout

```
scripts/   Wikidata ambiguity-audit scripts + modified ikp_estimate.py
patches/   Diff of ikp_estimate.py against the upstream version (what we changed)
runs/      Model-run outputs, analysis scripts, and figures
data/      Bundled upstream probe set + our derived data products
configs/   Bundled upstream model configs (unmodified)
```

## What's ours vs Li's

| Path | Origin |
|---|---|
| `scripts/wikidata_*.py` | ours (original work) |
| `scripts/ikp_estimate.py` | Li's, **modified** by us (see `patches/ikp_estimate.patch` for the diff) |
| `runs/` | ours (run outputs, analysis, figures) |
| `data/wikidata_*.json` | ours (audit outputs) |
| `data/probes/final_probe_set_v9.json` | Li's, **unmodified** |
| `configs/all_models.json`, `configs/models.json` | Li's, **unmodified** |

## Scripts

| Script | What it does |
|---|---|
| `scripts/wikidata_sparql_recheck.py` | Sequential SPARQL ambiguity recheck for T5–T7 Wikidata probes — counts distinct entities sharing each probe's label across 10 languages; flags a probe if any (lang, label) yields ≥3 entities. |
| `scripts/wikidata_sparql_batch.py` | Faster batched variant (batched `wbgetentities` + POSTed SPARQL counts), safe to run alongside the sequential script. |
| `scripts/wikidata_dominance_check.py` | Second pass on flagged probes — compares the probe entity's sitelink count to its top namesakes; a collision is treated as noise when the probe entity dominates by ≥5×. |
| `scripts/ikp_estimate.py` | Li's parameter estimator with our methodology changes (flooring / thinking-variant handling). Queries models via OpenRouter; needs `OPENROUTER_API_KEY`. |

The Wikidata scripts identify themselves to the SPARQL endpoint with a User-Agent
pointing at this repo (Wikidata API etiquette). `wikidata_dominance_check.py` reads
`data/wikidata_sparql_recheck_records.json` (bundled) and writes
`data/wikidata_dominance_check.json` (bundled, for reference).

## Calibration figures

`runs/plot_cliff.py` and `runs/plot_uncertainty.py` reproduce the two calibration
figures over 97 models with known parameter counts. They read per-model penalized
accuracy from `runs/calibration_error_per_model.csv` (bundled), so **no large data
download is needed**. If you instead drop the full upstream `data/results/` tree
(~127 MB, available from the upstream repo) into place, the scripts will prefer
those raw outputs automatically — the values are identical.

## The estimator patch

`patches/ikp_estimate.patch` is the diff of our `scripts/ikp_estimate.py` against the
upstream version, isolating exactly what we changed. To apply it onto a fresh
upstream clone instead of using our copy:

```bash
git clone https://github.com/19PINE-AI/ikp
cd ikp
git apply /path/to/patches/ikp_estimate.patch
```

## License

MIT for this replication's own code (`scripts/wikidata_*.py`, `runs/`, our changes).
The bundled upstream files (`data/probes/`, `configs/`, and the base of
`scripts/ikp_estimate.py`) are Li's work, redistributed unmodified — governed by the
upstream repository's terms: https://github.com/19PINE-AI/ikp.
