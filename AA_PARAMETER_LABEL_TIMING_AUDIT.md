# AA parameter-label timing audit

## Result

Chronological validation now distinguishes three dates that were previously
collapsed into one:

1. the model/API checkpoint release;
2. the first public primary-source parameter label used as the regression
   target; and
3. the public weight/config release.

Only `max(checkpoint release, parameter-label availability)` controls whether a
row may train a historical fold.  The weight date remains separate because a
first-party launch card or technical report can disclose the architecture
before downloadable weights appear.  The current July 31 calibration fit is
unchanged: every label below was public by then.

| Checkpoint | AA release | Public label | Weights/config | Timing basis | Historical folds affected |
|---|---:|---:|---:|---|---:|
| LongCat Flash Lite | 2026-01-28 | 2026-01-29 | 2026-01-30 | First-party technical report | 0 |
| MiniMax-M2.7 | 2026-03-18 | 2026-04-09 | 2026-04-09 | Official Hugging Face config and weights; launch page does not state the count | 1 |
| MiMo-V2.5 | 2026-04-22 | 2026-04-22 | 2026-04-27 | Xiaomi launch model card states 310B total / 15B active | 0 |
| MiniMax-M3 | 2026-06-01 | 2026-06-12 | 2026-06-12 | Official Hugging Face config, model card, and weights | 4 |
| Nex-N2-Pro | 2026-06-02 | 2026-06-03 | 2026-06-03 | Official Hugging Face config, model card, and weights | 0 |
| LongCat 2.0 | 2026-06-29 | 2026-06-30 | 2026-07-05 | First-party Meituan release article states 1.6T total / about 48B active | 0 |

The five affected AA tests are GLM-5.1 (MiniMax-M2.7 was premature)
and Nex-N2-Pro, Nemotron 3 Ultra, North Mini Code, and Kimi K2.7 Code
(MiniMax-M3 was premature).  No row is deleted; it simply becomes eligible on
the correct date.

## Validation impact

The following table preserves the isolated label-timing comparison that
motivated the correction. It predates the later score-publication timing and
canonical parameter-truth overlays, so it is historical diagnostic evidence,
not the current end-to-end precision summary:

| Metric | Release-only chronology | Label-aware chronology |
|---|---:|---:|
| Test rows | 33 | 33 |
| Median multiplicative error | 2.6650x | 2.4954x |
| Log10 RMSE | 0.72824 | 0.72753 |
| P80 error | 11.5097x | 11.5097x |
| P90 error | 18.6661x | 18.6661x |

The five changed AA point predictions move by +3.7% to +15.2%.  This correction
does not make the model look artificially worse; it slightly improves the AA
median while fixing the information-set definition.

MiniMax-M2.7 is also present in the ECI panel.  In that isolated comparison,
its timing correction changes
one ECI prediction (Gemma 4 31B IT) by -1.48%, while the ECI median remains
2.5963x.  In the available-component ensemble, only that Gemma prediction
changes (-0.413%); the 40-row and 24-row frontier medians remain 2.0675x and
1.9349x respectively.

Under that isolated developer-cluster order-statistic calculation,
the Gemma residual is the median-ranked developer residual.  The 50% factor
therefore moves from 2.73445x to 2.72317x (-0.413%).  The 80% and 90% factors
remain exactly 6.18643x and 7.22881x.  Central Fable, Sol, and Opus 5 forecasts,
all live evidence weights, and the 80%/90% published tails are unchanged.

The current combined audit applies label timing, score timing, and canonical
parameter truth together. Its available-component ensemble has 44 held-out
rows and 2.1364x median error; the 27-row frontier subset has 2.0237x median
error, and whole-developer holdout gives 1.8948x on those frontier rows. The
conservative target 80% factor remains 6.186428x. These later values supersede
the isolated table for readiness reporting without erasing its audit trail.

## Enforcement

- `sources/aa_parameter_label_availability_2026-07-31.json` is the canonical
  six-record timing ledger.
- Twenty-nine official source artifacts are vendored under
  `sources/aa_parameter_label_availability_evidence_2026-07-31/`; every path and
  SHA-256 is validated before use.
- `collect_aa_parameter_label_availability.py` verifies offline by default.
  Network refresh requires `--refresh`.
- Exact name, date, parameter count, and (when present) official repository URL
  are checked fail-closed. The label-timing ledger accepts MiniMax-M2.7's
  source-level 229B value and AA's rounded 230B representation; the separate
  canonical truth overlay resolves both to the commit-pinned exact tensor
  inventory of 228.703644928B total / 10B active without mutating either raw
  source row.
- The primary chronological backtest, AA expanded audit, detailed inference
  audit, operational audit, and active-parameter audit all consume the same
  eligibility helper.
- Tests independently verify the five affected folds, both accepted M2.7
  representations, tampered-evidence rejection, and source-hash reconciliation.

## Limitations

Hugging Face repository creation does not prove the repository was public at
that instant, so the ledger uses an observable public model card, config,
technical report, release article, or weight commit rather than assuming that
an empty/private repository was available.  Conversely, Epoch's `Last
modified` field is not a first-publication timestamp and is not used as a label
date: a later edit does not establish that the parameter count was unavailable
earlier.  Epoch rows without a dated primary source remain a provenance gap,
not evidence of leakage.
