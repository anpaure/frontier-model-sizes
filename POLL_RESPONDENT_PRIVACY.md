# Poll respondent privacy

The public project uses stable anonymous labels `Respondent R01` through `Respondent R21`. These labels preserve one-person/one-model validation, supersession rules, and the 18 paired Fable/Sol observations without publishing names. Forecast record IDs are also opaque. Provenance is reduced to direct-versus-relayed submission classes, and the repository retains no name-to-ID mapping.

All forecast values are unchanged. The privacy migration preserves 42 active records, 20 Fable forecasts, 19 Sol forecasts, the 4.371847506331046T and 3.2116911484449235T crowd centers, and every final model forecast.

The public prospective freeze is a privacy-redacted derivative. It cites the prior artifact SHA-256 `0ed93f398ad2f80f8c7b76ce7d7add4b017ab45b53a0f14303d47613ae9ac785`, records a canonical numerical projection digest, and asserts that only respondent labels, name-bearing record IDs, personal provenance, and personal notes changed. The name-bearing original bytes are not retained in the current tree.

The publishable snapshot was checked against the historical private-label set before the history reset. The audit covered 373 text, CSV, JSON, and source files; 10 XLSX files; one DOCX file; 350 pages across three PDFs; 158 tracked images; and the packaged site. It found no poll-linked private label, legacy forecast ID, historical attribution phrase, local home path, or image-metadata association. Public-source author-name collisions were reviewed separately and were unrelated to the poll.

The active branch is published from a parentless privacy root, so earlier identity-bearing branch ancestry is not part of the current Git history. GitHub's immutable pull-request reference and cached diff are a separate server-side retention surface; their purge requires GitHub Support and remains pending until GitHub confirms removal. Old independent clones, previously downloaded artifacts, and conversation copies are outside this repository's control.

This is pseudonymization rather than a guarantee of irreversible anonymity: stable IDs and distinctive forecasts can still be linked against a copy obtained elsewhere. The current repository tree, generated artifacts, and active branch history retain no poll name mapping.
