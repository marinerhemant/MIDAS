# Column content — which orientations are present in one interaction volume

**The question.** For every raster point (one beam column through the sample): which crystal orientations are
present, each with its **intensity share** and **orientation spread**, and **how completely** that list explains the
detected intensity. It is not "the grain at this point": a column holds several crystals, and they are the answer, not
contamination.

**Two implementations of one method.**

| data | implementation | status |
|---|---|---|
| White-beam (polychromatic) Laue, 34-ID-E-type reflection geometry | LaueMatching `pipeline/analysis/column_content/` (reference implementation) | tested on synthetic columns of known content: registered reads NOT VALIDATED (narrow: per-frame completeness; wide: discovery recall), fit gates pass, recall table in `ENVELOPE.md` §1; applied to one real sample (sampleH) |
| Monochromatic rotation (e.g. DAC rasters, area detector, omega wedge) | `midas_defect.column_content` | tested on 240 synthetic columns: registered read NOT VALIDATED (recall 0.887 vs the 0.90 bar), six of seven gates pass; recall table and where the misses are in `ENVELOPE.md` §2 (status PROVISIONAL: reproduced and not refuted, two lenses uncertain); not applied to real data |

Read `METHOD.md` for the method and the interface a new modality must supply, `ENVELOPE.md` before quoting any number,
`DIAGNOSIS.md` when something looks wrong, and `LAB_NOTEBOOK.md` for the evidence (including what was refuted).

## The order

1. **Survey** the data you have: the geometry, the wedge or energy band, the detector mask, and saturation.
2. **Measure the instrument kernel.** Do not assume it. Laue: use a thin single-crystal calibrant (for example Si) at
   the same beam settings. Mono: use the data's own isolated spots (`estimate_kernel`), and correct for its
   thresholded-moment bias.
3. **Round 0: the candidate orientations.** Laue: the C indexer at a measured null. Mono: `find_domains` on the known
   cell (with `seed_from_nominal=True` for narrow wedges).
4. **Joint fit** of all candidates on shared pixels or voxels (`ColumnFit`).
5. **Discovery rounds on the residual.** Accept a new orientation only above a gate MEASURED on scrambled residuals.
6. **Validate on synthetic columns of known content, with this sample's geometry, crowding and spread classes, BEFORE
   reading real data.** The recall table by share and by spread class *is* the completeness statement.
7. **Real data, descriptive.** Report per column: orientations, shares, spreads, the unexplained fraction, and the
   recall table that applies.
8. **What the fit leaves unexplained:** use the arc diagnostics (beads, a common axis, chance-chaining null) before
   interpreting it.

## Hard rules

- **Validate before reading real data**, and preregister the gates. Two designs of this method (one Laue, one
  spectrum-budget) looked right and failed their validation. The validation is what caught it.
- **Shares are intensity shares, not volume fractions.** Per-reflection brightness is free in the fit.
- **Report a spread as "orientation spread present for this crystal family in this column".** Deformation and several
  close grains are not distinguished.
- **Never select the spots you score with the model you are scoring** (`DIAGNOSIS.md` T6).
- **A parameter on its bound is a finding about the model,** not a result.
- **Every number carries its provenance** (the file and command).

## Halt and ask

- The kernel cannot be measured (no calibrant, and no isolated spots).
- The validation reads NOT VALIDATED. Report which gate failed; do not redesign silently.
- The real data sit outside the validated envelope: crowding, spread class, wedge, or detector.
