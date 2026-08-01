# OpenRouter operational-signal audit

Snapshot: frozen 2026-07-31 operational observation (fetched 2026-07-31T01:09:27Z). Generated filenames retain their original `2026-07-18` suffix for compatibility.

The embedded dates and source hashes, rather than the compatibility suffix, determine the observation vintage. Current Kimi K3 primary evidence discloses exactly 2.780T total and 104.2B activated parameters; the routed-expert fraction is not used as an activated-count estimate.

## Outcome

Current OpenRouter API price contains real parameter-count information, but output throughput does not add a reliable signal after price. The live ensemble therefore retains its existing low 3.4% final weight on API price and assigns 0% incremental weight to tok/s.

The collector preserves a lossless daily panel and every refresh. The current snapshot contains 8,569 daily endpoint/service-tier rows: 7,763 default, 414 priority, and 392 flex. Eight immutable refreshes contribute 60,307 archived daily rows. Service tiers are never pooled; only default-tier throughput enters the regression.

The current endpoint panel also contains 1,172 explicit service-tier rows: 1,036 default, 60 flex, and 76 priority. It preserves p50/p75/p90/p95/p99 throughput and latency, request counts, base prices, and the first high-context price schedule. There are 142 endpoint-tier rows with a high-context threshold. No assumed mixture over prompt lengths or service tiers is used in forecasting.

An additional active-capacity audit uses 63 active-parameter labels across 15 developers: 45 disclosed active counts plus 18 dense controls whose active count equals the Epoch total. The dense controls come only from successfully parsed primary Hugging Face `config.json` files; gated and unavailable configs remain unresolved. This expands the release-ordered held-out ledger from 31 to 45 predictions. Active prediction followed by sparse-MoE transport improves the tested high-sparsity point metrics, but only 16 models from seven developers qualify and the paired 90% interval crosses zero. Fable and Sol prices also extrapolate 6.5× and 3.5× beyond the common training maximum. This branch therefore receives 0% incremental weight; see `OPENROUTER_ACTIVE_PRICE_AUDIT.md`.

This is based on 93 unique Epoch parameter checkpoints from 19 developer families in the frozen audit. Every accepted OpenRouter-to-Epoch identity is explicitly whitelisted; fuzzy matching is prohibited. Kimi K3 (exactly 2.780T total, displayed as 2.8T) and Grok 4.5 (1.5T) are excluded from fitting and used as external disclosed checks.

| Held-out test | Date only | Date + price | Date + price + normalized tok/s |
|---|---:|---:|---:|
| Leave-one-developer-family-out median error | 4.1× | 2.2× | 2.3× |
| Strict chronological + family-held-out median error | 3.9× | 2.2× | 2.3× |

The paired family bootstrap gives price a 100.0% probability of beating the date-only model in the family holdout and 99.97% in the chronological-family holdout. Adding provider/quantization-normalized tok/s to price has only a 6.7% probability of improving the family-held-out mean absolute log error; its observed error is worse.

The service-tier correction removes an avoidable provider-normalization defect. The conclusion is unchanged: family-held-out mean absolute log10 error rises from 0.397 with date+price to 0.405 with tok/s. Chronological-family error moves only marginally from 0.419 to 0.417, while the clustered interval crosses zero, so the feature still fails the joint promotion gate.

An additional incremental test finds 11 checkpoints from 7 developers with both an existing strictly chronological family-held-out evidence prediction and an independently chronological OpenRouter-price prediction. Adding price at the current 6.75% within-evidence weight improves median error from 2.3× to 2.1×. A paired developer-cluster bootstrap favors the blend with 99.6% probability and a 90% interval for the mean absolute log-error change of −0.020 to −0.004. This validates retaining the current small price branch; the overlap is too small and selected to justify increasing it.

## Why tok/s receives zero weight

Throughput measures the serving system as well as the model: hardware, batching, quantization, speculative decoding, traffic, and routing all matter. The analysis controls each endpoint against its provider and quantization peers, then aggregates the residuals by model. That still does not improve held-out prediction.

The daily panel quantifies this noise. Among 327 models with at least four dates, the median within-week max/min ratio is 1.52× and the 90th percentile is 3.78×. Among 991 default-tier endpoints, the corresponding median is 1.58× and the 90th percentile is 4.57×. The median max/min ratio across the eight refresh snapshots is 1.14×; those rolling one-week windows are correlated audit views, not independent samples.

Same-base controls independently expose the noise:

- Claude Opus 4.5–4.8 has identical aggregate price but a 1.7× raw throughput range and a 1.6× provider-normalized range.
- Standard GPT-5–5.5 endpoints have a 1.4× throughput range while price moves 3.6× despite the asserted shared base.

These are product and serving changes without corresponding base-parameter changes.

## Request counts, latency, and percentile spreads

A separate follow-up tests whether the 30-minute endpoint panel becomes useful after imposing a minimum of 100 requests and aggregating endpoints with capped log request weights. The common panel contains 81 known-size checkpoints from 16 developer families. It compares date + price with one-week throughput, 30-minute p50 throughput, request-supported p50 throughput, request-supported p50 latency, a joint throughput/latency feature, and p90/p50 throughput and latency spreads.

No candidate passes the predeclared promotion rule. All six operational candidates worsen the family-held-out mean absolute log10 error point estimate after price; the request-supported latency interval still crosses zero. Latency, request counts, throughput, and percentile spreads therefore remain 0%-weight diagnostics.

## Independent Artificial Analysis cross-check

The frozen Artificial Analysis payload supplies a separately measured, standardized operational panel. It uses first-party APIs where available, provider medians otherwise, and OpenAI-token units for output speed. Exact Epoch checkpoint IDs yield 28 AA↔OpenRouter overlaps.

Across 26 positive-price pairs, log-price Pearson correlation is 0.831 and rank correlation is 0.834. Across 25 speed pairs, raw tok/s rank correlation is only 0.404; provider-normalized OpenRouter tok/s reaches only 0.441. This independently confirms that the two price implementations measure a common scale-related serving decision while throughput is dominated by operational noise.

AA's own chronological developer-held-out tests further separate the serving regimes. Provider-median price improves frontier-like median error from 2.77× to 2.51×, with the 90% developer-bootstrap interval entirely favorable. First-party price worsens frontier-like median error from 2.13× to 2.77×. Because Fable and Sol are first-party targets, this validates the sign of the existing OpenRouter branch but does not justify increasing its weight or adding AA price as a second correlated factor. See `AA_OPERATIONAL_SIGNAL_AUDIT.md` for the complete audit.

## Price is useful, but extrapolative

The matched calibration price range is $0.04–$5.00 per million tokens on the prompt/output geometric-mean scale. Several frontier products are beyond that range: Fable is 4.5× above the calibration maximum and Sol is 2.4× above it. The raw price regression therefore extrapolates strongly and cannot be treated as an independent parameter census.

After correcting the price-only model by the geometric residual on disclosed Kimi K3 and Grok 4.5, its diagnostic centers are about 10.6T for Fable and 5.7T for Sol. Its family-held-out 80th-percentile error is still about 4.2×, so these values are not substituted for the main 4.5T/3.1T ensemble. The defensible use is a small upward-correlated cross-check.

## Reproducible pipeline

Run the frozen snapshot:

```bash
python3 run_forecast_pipeline.py
```

Refresh OpenRouter first, then rebuild everything:

```bash
python3 run_forecast_pipeline.py --refresh-openrouter
```

`--refresh-openrouter` also refreshes the primary Hugging Face architecture configs after the OpenRouter↔Epoch calibration inventory has been rebuilt. To refresh only those configs, use `--refresh-hf-configs`. Ordinary builds revalidate the frozen snapshot without network access.

The collector uses OpenRouter's official [models API](https://openrouter.ai/docs/api/api-reference/models/get-models) and public first-party provider-stat endpoints used by its model pages. OpenRouter documents catalog pricing as USD/token and describes provider throughput percentiles in its [routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection). The frozen raw payload is retained so later OpenRouter changes cannot silently rewrite this analysis. Every refresh is additionally archived under its fetched-at timestamp with exact hashes. Eleven endpoint/model pairs currently appear in the daily panel without public endpoint metadata; all 86 affected rows remain in the ledger with an explicit unmatched flag.

Every price schedule is independently crosschecked against OpenRouter's documented `/api/v1/models/{model_id}/endpoints` API. All 334 model requests succeeded. The official API returned 1,051 endpoint rows; 995 rows lie in exact price-signature groups, for a 94.7% exact row share against the near-contemporaneous frontend extraction. All discrepancies are retained. Fable and K3 match exactly. Sol's default, OpenAI flex, and OpenAI priority schedules match; only an Azure priority tier visible in the frontend statistics response was absent from the official response. The official API currently returns null throughput values, so tok/s remains sourced from the model-page statistics payload and is independently crosschecked against Artificial Analysis.

Important generated artifacts:

Every dated path below retains the legacy `2026-07-18` compatibility filename. The rows and audit metadata identify the actual 2026-07-31 observation.

- `sources/openrouter_operational_snapshot_2026-07-18.json.gz`: complete raw catalog and model-stat responses.
- `sources/openrouter_provider_signals_2026-07-18.csv`: 1,036 provider endpoint observations.
- `sources/openrouter_endpoint_tier_signals_2026-07-18.csv`: 1,172 endpoint/service-tier price and p50–p99 performance rows.
- `sources/openrouter_model_signals_2026-07-18.csv`: 334 eligible text-model aggregates from the 364-model catalog.
- `sources/openrouter_throughput_daily_2026-07-18.csv`: 8,569 lossless daily model/date/endpoint/service-tier rows.
- `sources/openrouter_history/`: exact immutable raw/model/provider/audit files for all eight refreshes.
- `sources/openrouter_snapshot_history_manifest_2026-07-18.csv`: snapshot timestamps, counts, archive paths, and hashes.
- `sources/openrouter_endpoint_tier_snapshot_history_2026-07-18.csv`: 8,046 reconstructed endpoint/service-tier rows across all refreshes.
- `sources/openrouter_throughput_daily_history_2026-07-18.csv`: 60,307 daily rows across all refreshes; overlapping rolling windows are explicitly correlated.
- `sources/openrouter_official_endpoint_prices_2026-07-18.csv`: 1,051 documented-API endpoint price schedules.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/openrouter_official_endpoint_audit_2026-07-18.json`: official/frontend price reconciliation and hashes.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/openrouter_epoch_match_audit_2026-07-18.csv`: explicit match decision for every OpenRouter model.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/openrouter_parameter_signal_backtest_2026-07-18.json`: held-out metrics, bootstraps, same-base controls, limitations, and hashes.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/openrouter_temporal_stability_audit_2026-07-18.json`: service-tier counterfactual, daily and refresh volatility, focal-model diagnostics, decision, limitations, and hashes.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/openrouter_request_weighted_operational_audit_2026-07-18.json`: request-count-gated throughput, latency, percentile-spread, held-out, and family-bootstrap audit.
- `sources/huggingface_architecture_config_snapshot_2026-07-18.json.gz`: 87 primary Hugging Face repository responses with verbatim configs or explicit HTTP failures.
- `sources/huggingface_architecture_config_signals_2026-07-18.csv`: nested expert-routing fields and conservative dense/MoE/unavailable classifications.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/huggingface_architecture_config_collection_audit_2026-07-18.json`: 73 successful JSON configs, 14 explicit gated responses, inventory reconciliation, and hashes.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/openrouter_incremental_price_backtest_2026-07-18.json`: paired test of price on top of the existing evidence ensemble.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/unified_model_observations_operational_enriched_2026-07-18.csv`: complete operationally enriched observations; exact counts are recorded in the adjacent summary JSON.
- `outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/unified_model_measurements_long_operational_enriched_2026-07-18.csv`: complete long-format measurements; exact counts are recorded in the adjacent summary JSON.

Automated tests verify that every base observation and measurement is preserved field-for-field, every new ID is unique, every long-format value is numeric and linked to an observation, every source hash reconciles, and unmatched OpenRouter models never inherit an Epoch parameter count. The operational successor includes all 2,566 historical price changes, the 3,098-row prospective prediction ledger, the request-weighted audit, the complete active-price ledgers, and the primary Hugging Face configuration snapshot and audit; its generated summary is the count authority.
