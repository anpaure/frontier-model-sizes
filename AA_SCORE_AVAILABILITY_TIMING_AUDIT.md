# Artificial Analysis score-publication timing audit

Status: exact changelog events applied to chronological validation; current fits and centers unchanged.

Artificial Analysis release dates are retained as the algorithmic-progress feature. For exact slugs with a dated non-null Intelligence Index event, chronological tests now use the score-publication date as the prediction cutoff and `max(release, parameter-label date, score date)` as training eligibility.

## Coverage

- Official API returned 1,250 events while advertising 1,561; this unresolved source discrepancy is preserved explicitly.
- 136 unique score-publication slugs are hash-pinned.
- Live AA panel: 28/50 verified, 16 later than nominal release; 22 retain explicit release-date fallback.
- Detailed AA panel: 64/275 verified, 36 later than nominal release.

## Validation effect

| Scope | Release-order baseline | Score-timing corrected |
|---|---:|---:|
| AA median error | 2.60x | 2.70x |
| Frontier AA median error | 1.91x | 1.98x |
| Ensemble median error | 2.15x | 2.14x |
| Frontier ensemble median error | 2.09x | 2.02x |

This is a validation correction, not new capability evidence. The published current AA score values still come from the July 31 snapshot, so the exercise remains pseudo-chronological with respect to index-version changes and score revisions.
