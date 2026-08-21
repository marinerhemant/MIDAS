# Phase 4 — Per-voxel peak-shape strain (`midas_pf_odf`)

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).
> **This phase has no far-field / near-field sibling.** It is what makes pf-HEDM distinct.

Traditional pf-HEDM reduces each Bragg spot to its **centroid** and recovers a mean
orientation + mean elastic strain per voxel. `midas_pf_odf` fits the **full 3-D intensity
patch** of each spot — same warm-start, same optimiser, the *only* difference is the loss
(image MSE on the patch vs 3-DoF MSE on the first moment). The extra information density
recovers the sub-voxel distribution (orientation spread, micro-strain spread) — the peak's
*shape*, not just its centre.

**Read the envelope (§2, §3) before promising a strain map.** On an attenuated scan this is
the phase most likely to over-promise.

## 4.1 Inputs

pf-odf consumes the phase-3 outputs:

- `Output/voxel_grid.csv` — voxel→grain map (groups voxels for the per-grain joint fit).
- `Results/Result_OrientPos_voxel_<v>.csv` — per-voxel refined orientation + lattice
  (warm-start). **39 columns**, or **45** from midas-fit-grain 0.9.0, which appends
  `PosErrPre/OmeErrPre/InternalAnglePre` then the `*Post` triple at 39-44. Cols 0-38
  are unchanged, so `row[1:10]` (orientation) and `row[15:21]` (lattice) do not move.
  These arrive in PF for free: the CSV writer at `FitUnified.c:2270` runs in **both**
  modes and `fitbest_adapter` forwards every token.

  > **PF's `PosErr` pre/post is NOT the FF quantity.** `isFF` gates fit stages 2 and 4
  > (`FitUnified.c:1991`/`:2015`), which are the **position** fits — so **PF does not
  > refine position at all**. A PF `PosErr` moves only through the orientation and
  > lattice fit. Do not compare a PF improvement against an FF one, and do not read a
  > small PF change as a failed position fit: there was no position fit.
- `paramstest.txt` + `hkls.csv` — geometry + reflections for the forward model.
- The **raw detector frames** — the peak-shape fit needs the actual pixel patches.

```python
from midas_pf_odf.io import build_model_from_paramstest, load_pf_grain
model, ring_nr = build_model_from_paramstest(
    layer_dir, n_pixels_y=NY, n_pixels_z=NZ, n_frames=N_FRAMES, omega_step=OMEGA_STEP)
ds = load_pf_grain(layer_dir, grain_id, n_pixels_y=NY, n_pixels_z=NZ, model=model,
                   n_frames=N_FRAMES, omega_step=OMEGA_STEP)
```

**Pass `n_frames` (= the acquisition frame count) and `omega_step` explicitly.** The
fallback is the indexing bin size, which is the wrong frame step for cropping patches and
warns.

## 4.2 Read frames from the zarr, not the raw h5

The MIDAS zarr (`*.MIDAS.zip`, Blosc-lz4) is ~2× smaller and ~2× faster to read than the raw
h5, byte-identical frames. Use the helper:

```python
from midas_pf_odf.io import zarr_frame_reader
reader = zarr_frame_reader(sorted_per_scan_zarr_paths)   # frame_reader(scan_idx, frame_idx)
```

⚠️ The frames are chunked **one full frame per chunk**, so reading a 15×15 patch decompresses
a whole 2880² frame. For a whole layer that is terabytes read once — plan a single-pass
extraction (read each needed frame once, distribute crops to all grains) rather than
re-reading per grain.

## 4.3 Validate the geometry against the raw signal FIRST

Before any fit, confirm the predicted spot **anchors land on real diffraction**. Compute the
per-spot anchor (`_forward_anchors`), read the patch at a **bright** reflection, and check the
peak sits inside the patch:

- A bright ring-2/3 reflection landing within a few pixels of its anchor confirms the ω +
  detector convention (this is the practical re-check of phase 1.1).
- **Do not trust a naive "is there a peak at the centre" gate on weak data** — see the
  refuted entry in the notebook. On an attenuated scan most spots are near noise and a
  centre-pixel gate reads spot *weakness* as misalignment. Use a bright spot.

## 4.4 Dark subtraction is mandatory for raw frames

pf-odf's loss is `‖c·pred − meas‖²` with `c` a purely multiplicative per-spot scale — **no
additive background term.** A flat pedestal under every spot inflates `c` and dominates the
residual with unmatched background pixels.

```python
from midas_pf_odf.io import assemble_grain_patch_data
data = assemble_grain_patch_data(ds, frame_reader=reader,
                                 subtract_background=True)     # raw frames
```

`subtract_background=True` removes a per-(spot,scan) background estimated from the patch
border ring. Measured impact on a raw attenuated scan: median strain **−30 %** and a
materially different field (raw vs corrected only ~0.26-correlated). **Leave it OFF only for
already-dark-subtracted caches** — double-subtracting clips real signal.

## 4.5 Fit

```python
from midas_pf_odf.inversion import fit_grain_peakshape, IdentifiabilityMode
res = fit_grain_peakshape(
    data, model, voxel_pos=ds.voxel_pos, R_init=ds.R_init, eps_init=ds.eps_init,
    lattice_init=ds.lattice_init,
    identifiability=IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,   # lattice absorbs bulk
    optimizer="adam", inner_steps=60)
eps = res.eps_fit          # (n_vox, 6) Voigt strain, crystal frame
```

For a whole layer, run per grain on GPU. Big grains need voxel chunking (`chunk_size_g`) to
fit VRAM — the forward's `(G, S, Σ, F, P, P)` tensor is large because Σ (scans) is high; too
large a chunk OOMs, so shrink on `OutOfMemoryError`. For a full-layer strain map, the
extraction I/O (§4.2) dominates, not the fit.

## 4.6 Judge the result — spatial structure, not magnitude

Per-voxel strain on an attenuated scan is noisy and processing-sensitive. **Do not judge it
by median magnitude.** Judge it by **spatial structure**: real residual strain organises
across neighbouring voxels; noise scatters. Moran's I against a permutation null is the test
— `I ≫ 0` (p small) means the field is spatially real even if per-voxel values are provisional.
State magnitudes as **provisional / qualitative** when the scan is signal-limited.

## 4.7 Orientation-derived deformation (no strain fit needed)

Even without the peak-shape fit, the refined per-voxel orientations give the community
metrics directly:

- **KAM** (kernel average misorientation) — each voxel's mean misorientation to its
  nearest neighbours, excluding pairs above a boundary threshold (~5°). A local-gradient /
  GND proxy. Overlay grain boundaries or it looks like it has none (by design — HAGBs are
  excluded).
- **GROD** (grain reference orientation deviation) — each voxel's misorientation from its
  grain mean. Accumulated lattice rotation.

On a deformed / cracked sample these localise deformation robustly and are far less
processing-sensitive than the strain — often the headline result.

Then [`phase-5-read-report.md`](phase-5-read-report.md).
