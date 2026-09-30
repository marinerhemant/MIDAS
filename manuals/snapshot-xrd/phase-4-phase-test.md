# Phase 4 -- candidate-cell test

> Part of the snapshot-xrd doc set. Spine: [`README.md`](README.md).

## Candidates
- One kind of source, one selection rule for all (`midas_hkls.io.structure_sources.select_entry`:
  ambient, not pre-1947, most precise cell, newest). Record source, id and the rule.
- A local licensed CIF collection can be indexed once (`index_cif_directory`) and searched by
  elements, space group and cell range; COD is queried online (`cod_search`).
- Lines come from the structure with basis absences dropped (`allowed_d_lines`).

## The test (`midas_hkls.feature_phase.feature_phase_test`)
- Input: feature d-values (optionally only those present before and absent after on raw photons).
- Statistic: M = features within tolerance (2 sigma_matrix) of a scaled line, maximised over a scale
  window; L = distinct lines hit. Report `n_lines` and `chance_coverage` for every candidate.
- Null: surrogate features drawn with the detector's 2-theta coverage (valid pixels, margin, maximum
  angle), matrix bands removed; the identical maximisation.
- Negative controls: each candidate's pattern rescaled well outside its window. They must fail;
  if one passes the run is not readable.
- Pass: p below the corrected alpha, M >= 5, L >= 2 (prefer >= 3), M >= 3 x null mean.
- A best scale on the window edge is flagged: the cell lies outside; say so.

## Cell scan (exploratory)
`cell_scan` rescales one pattern across a range of cells with no free scale and compares the peak
with the null distribution of the maximum over the whole scan (look-elsewhere). It is exploratory:
confirm on held-out data with a fixed range, run once.

## Wording
A pass identifies a cell and a pattern. Structures of one type share the pattern; composition is
not determined. Single-feature coincidences are leads, never identifications.
