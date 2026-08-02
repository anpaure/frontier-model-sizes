# Frontier parameter estimates site

This directory contains the interactive frontend for the audited frontier-parameter forecast. It is a presentation layer over pipeline-generated data, not a separate forecasting implementation.

## Local development

```bash
npm install
npm run dev
```

Open `http://localhost:3000/`.

The chart is a horizontal forest plot over the ten current frontier/base identities. Each row shows:

- the public estimate as a gray circle;
- the model estimate as a developer-colored diamond; and
- the range of active factor-implied values as a thin whisker.

All live evidence weights are adjustable in the right-hand control rail as logarithmic `0.1x–10x` multipliers around the audited defaults; every active factor starts at `1x`. Hovering a row previews its effective contribution weights; the whisker and model card report the weight-adjusted 10th–90th percentile of the active factor estimates. The axis defaults to 1–7T; if a model estimate crosses 7T, it grows continuously with modest headroom while retaining a visible 7T reference line.

## Data flow

Do not edit JSON under `public/data/` by hand. From the repository root, run:

```bash
python3 generate_forecast_site_data.py
python3 generate_parameter_scatter_data.py
```

The ordinary full rebuild runs both stages automatically:

```bash
./run_forecast_pipeline.py
```

`public/data/forecast-model.json` is the canonical site forecast contract. `public/data/parameter-scatter.json` is a deterministic ten-row visual contract derived from it. External screenshots and page captures are visual references only and never enter either data file.

## Verification

```bash
npm run build
node --test tests/rendered-html.test.mjs
```

The rendered test checks the ten-model identity set, locked Kimi/Grok anchors, shared Opus 4.7/4.8 base, fixed default axis, and pipeline-source policy.
