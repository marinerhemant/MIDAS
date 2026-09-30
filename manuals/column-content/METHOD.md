# The column-content method, and the interface a modality must supply

## Model

For one column, with candidate orientations j = 1..J:

    model(v) = c + sum_j sum_h a[j,h] * sum_k w[j,k] * Kern(v - pos(j, k, h))

- **v** runs over the union of fit windows (pixels for Laue, (omega, row, col) voxels for mono), minus masked or
  saturated elements.
- **pos(j,k,h)** is where reflection h of orientation j lands when j is rotated by component offset k. Each
  orientation is a small cloud of K sub-orientations (tangent offsets). The weighted spread of that cloud is the
  reported spread.
- **Kern** is the MEASURED instrument kernel with unit flux and one global width scale fitted per column.
- **a[j,h] >= 0** is a free per-reflection brightness. It absorbs |F|², spectrum or Lorentz factors, and absorption.
- **w[j,k] >= 0** are the component weights. a and w come from alternating non-negative least squares over ALL
  orientations jointly, so orientations sharing pixels are fitted together and nothing is masked as contamination.
  The normal equations are assembled from the window structure (each orientation touches only its own windows), never
  from a dense design over all voxels; that is what makes a 1e6-voxel column tractable (`fast_linear=False` is the dense
  reference, for tests). Cost is still dominated by 300 optimiser steps x 2 inits per fit and one refit per discovery round.
- **The offsets** are optimised by Adam, with lr 3e-4 rad and two inits (0.05 and 0.4 deg); the lower loss wins.
  The final linear solve restarts from the best iterate, and (a, w) are reset if they collapse to zero.

**Fit windows** are the bounding boxes of the DETECTED blobs touching each predicted reflection, dilated and clipped.
They are never derived from the model itself.

**Outputs, per orientation:**
- share_j = flux_j / (modelled flux + UNEXPLAINED flux over ALL detected-blob elements), so a missed crystal lowers the
  others' shares instead of inflating them;
- the mean orientation;
- the spread (rms, and principal axes; Laue splits it by the blind axis);
- the number of reflections used.

**Outputs, per column:** the unexplained fraction, C_int, the fitted kernel scale, and the init that won.

## Discovery

The residual is re-indexed with the round-0 search. A solution counts as NEW if it lies more than 1 deg from every
current orientation AND scores above g_disc. g_disc is the smallest gate whose chance rate, measured on residuals
whose spots or pixels have been scrambled, has a Poisson 95% upper bound of at most 0.10 per column. The gate must be
measured on residuals, because they differ from raw frames.

## What a modality must supply (the interface)

| piece | Laue reference | mono (`midas_defect.column_content`) |
|---|---|---|
| geometry + differentiable projection: orientation -> reflection positions | `Geom.project` (params file via `laue_material.Phase`) | `forward.predict_torch` on `midas_defect.geometry` (tilts, 15-term distortion, wedge); Ewald branch fixed at the seed; `guard()` checks it against the numpy path (~1e-6 px) |
| observed reflections | predicted spot on a detected peak, harmonics merged | predicted voxel within +/-1 frame, +/-2 px of a `find_blobs_3d` label |
| measured kernel | `EmpKernel` from a Si calibrant (supersampled LSQ grid) | `GaussKernel3D(sig_frame, sig_rad, sig_tan)` about the beam centre; `estimate_kernel` from isolated spots, threshold-bias calibrated by simulation |
| peak / blob detection | `frame_peaks.detect_peaks` (5 MAD components) | `midas_defect.ingest` (mask, 8-sector background, `find_blobs_3d(return_labels=True)`) |
| round-0 + discovery search | the LaueMatching C indexer (injected `index_fn`) | `midas_defect.domains.find_domains` (tetragonal-type cells); any other search via `search_fn` |
| synthetic columns of known content | `synthetic.render_columns` (laue_torch renderer + measured kernel) | `synthetic.synthetic_column` (same geometry primitives as ingest; foreign crystal; powder rings) |
| validation harness + evaluator | `synthetic.render_columns` + `evaluate.evaluate` (frozen gates) | `validate.validate(ValidationSpec(...))`: stages A-D at YOUR geometry/cell/kernel; reproduces the registered columns bit for bit; reads INCOMPLETE if any column's result is missing |

**Conventions that bit:**
- raw Laue frames are [row = y, col = x];
- the midas_defect mask is True = EXCLUDED;
- midas_defect U maps crystal -> sample (q_sample = U B hkl);
- `midas_stress.orientation.misorientation_om(U1, U2, sg)` gives 0 for crystal-side symmetry and returns radians.
