---
name: column-content
description: >-
  Determine the ORIENTATION CONTENT of each raster point / beam column: which crystal
  orientations (grains, domains) are present in the interaction volume, each with its intensity
  share and orientation spread, and how completely that list explains the detected intensity --
  with completeness stated as a recall table measured on synthetic columns of known content.
  Use when asked which domains or grains a raster point contains and in what proportion, how
  spread each is, whether a column holds one crystal or several, or what a fit leaves
  unexplained (streaks, beaded arcs). Two implementations of one method: monochromatic rotation
  data (DAC rasters, area detector, omega wedge) through midas_defect.column_content, and
  white-beam Laue through LaueMatching pipeline/analysis/column_content. Getting the domains in
  the first place is defect (find_domains) or laue; an unknown cell is solve-cell. Both
  registered validations read NOT VALIDATED on recall (monochromatic: 0.887 against a 0.90 bar,
  measured on synthetic columns only; no real data read), so quote the recall table, not
  "validated".
---

# Column content

**This skill is a pointer, not the procedure.** The procedure is a doc set in the repository.

## Start here

Read **`manuals/column-content/README.md`**, then `METHOD.md`, and read **`ENVELOPE.md`**
before quoting anything. Then give, or work out from the data:

```
Frames:      <ABSOLUTE PATH>   # one raster point's rotation stack (mono) or Laue frames
Geometry:    <paramstest / geometry file>, wedge (mono) or params file (Laue)
Cell:        a, c (, b), space group -- known; if not, stop and use solve-cell
Kernel:      measured -- a calibrant (Laue) or the data's isolated spots (mono)
Goal:        per-point orientations + shares + spreads, and the completeness that applies
```

## Things to know before you start

1. **Validate before reading real data.** Run the synthetic-column validation at YOUR geometry,
   cell, wedge, crowding and spread classes (mono: `midas_defect.column_content.validate`). The
   recall table by share and by spread class IS the completeness statement. There is no reliable
   per-column completeness number. Both registered validations read NOT VALIDATED on recall (mono: 0.887 vs a 0.90 bar;
   weak domains below ~10% share and partners within ~1 deg are the ones missed), so quote the table, not "validated".
2. **Shares are intensity shares, not volume fractions.** Per-reflection brightness is free (it
   absorbs |F|^2, Lorentz, absorption and spectrum). A physical brightness model was tested and
   does not close the budget.
3. **A spread is "orientation spread present for this domain in this column".** Deformation and
   several close domains are not distinguished. Domains closer than ~0.3 deg merge into one
   wider cloud.
4. **Measure the kernel.** A guessed kernel width invents spread. The moment estimate from
   isolated spots is biased narrow; use the calibrated estimator.
5. **Mask convention (mono): True = EXCLUDED.** Mask saturated voxels, because there is no
   count-rate model.
6. **Budget the compute.** The heaviest mono columns (40 frames, 4 domains, ~1e6 voxels) took over a day per column on one
   core before the structured linear step and about an hour on 16 threads after; peak memory 13-18 GB per process.
7. **Discovery needs its own measured gate** on scrambled residuals. Recall of SPREAD crystals is
   limited by the search, not the fit.
8. **What the fit leaves unexplained is not automatically the found crystals' spread.** Test it
   by fitting, not by coincidence. Beaded arcs and chance chaining have diagnostics and nulls in
   the Laue implementation.

## When something looks wrong

`manuals/column-content/DIAGNOSIS.md`: symptom -> test -> cause -> fix, each one paid for.
