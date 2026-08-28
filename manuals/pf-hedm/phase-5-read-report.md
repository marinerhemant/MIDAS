# Phase 5 — Read the outputs and report

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).

## 5.1 The output files and their formats

| File | Format | Reads |
|---|---|---|
| `Output/voxel_grid.csv` | header + `voxel_idx x_um y_um z_um grain_id` per voxel | the layer's voxel→grain map |
| `SpotMatrix.csv` | 17 cols, one row per **predicted** reflection per voxel. `Matched`=1 observed, **0 = predicted but never found** — the completeness deficit itself, which `Result_OrientPos_voxel` records only as a number. Un-found rows: `SpotID`/`ScanNr` = -1, observed columns and residuals NaN, prediction (`YExp`/`ZExp`/`OmegaExp`/`RingNr`/`theorEta`) intact. Carries `ScanNr`, which FF's SpotMatrix has no room for. New 2026-08-21 |
| `Results/Result_OrientPos_voxel_<v>.csv` | 2-line: header + a **39**-col row (**45** from midas-fit-grain 0.9.0: `PosErr/OmeErr/InternalAngle` x `Pre/Post` at 39-44; PF does not fit position, so its `PosErr` pre/post is not the FF quantity) | per-voxel OM (cols 1–9), position (11–13), lattice (15–20), completeness (26), refiner strain (27–35), Euler (36–38) |
| `Output/UniqueOrientations.csv` | 14-col: grainID + 4 pad + 9 OM | unique grain orientations |
| `pfodf_eps/eps_grain<g>.npy` (if strain run) | `(n_vox, 6)` Voigt strain, crystal frame | pf-odf peak-shape strain per grain |

The voxel grid is `n_scans × n_scans`; infer `N = round(sqrt(n_voxels))` and lay
`voxel_idx = i·N + j` onto `(x,y) = (pos_sorted[i], pos_sorted[j])`.

## 5.2 The maps to make

- **IPF orientation map** — per-voxel crystal orientation coloured by inverse pole figure
  (IPF-Z: ⟨001⟩→red, ⟨011⟩→green, ⟨111⟩→blue after cubic FZ reduction). Weight brightness by
  completeness so weak voxels read dark. This is the primary spatial result and is robust.
  **Colour from the third ROW of the orientation matrix, not the third column** — MIDAS
  stores `v_lab = OM · v_crystal`. Verified by measurement: within a single grain the rgb
  standard deviation is 0.000 / 0.0015 / 0.0009 by row against 0.257 / 0.348 / 0.318 by
  column. Getting it wrong produces a map that is colourful, spatially structured, and
  wrong — a within-grain colour gradient is the tell.
  Clean isolated **salt-and-pepper** wrong-pick voxels with a neighbour-consensus filter
  before segmentation (they are wrong solutions that scored marginally higher, not sub-grains).
- **Pole figure** — the layer's orientation distribution ({100} stereographic); compare
  against the far-field pole figure as an FF↔PF consistency check.
- **KAM** and **GROD** (phase 4.7) — with grain boundaries overlaid.
- **Strain map** (if run) — von Mises per voxel, with the provisional-magnitude caveat.

Save every figure with a scale bar (the voxel step is the µm/px) and keep the generator
script beside the figure.

## 5.3 Report — with provenance

Every quantitative claim names the file and command that produced it and is re-derivable
(this is the scientific-conduct floor, not a style preference). Structure the report around
what is **robust** vs **provisional** on this dataset:

- **Robust:** orientation map, pole figure, KAM/GROD deformation localisation, grain
  statistics. These hold even on attenuated data. Grain **positions** fitted from sinograms
  are robust too (1.3–2.1 µm rms).
- **Provisional:** per-voxel absolute strain magnitudes when the scan is signal-limited
  (envelope §2). Report the *pattern* (does it localise where deformation is expected?) and
  the spatial-structure statistic (Moran's I), not calibrated magnitudes.
- **Not reportable:** grain **shapes** from the sinogram reconstruction — a spine halt
  condition and an open problem (envelope §3b, phase 6 §6.7). If shapes were asked for, say
  what the measurement can give instead (positions, and absorption tomography for shape).

**Two things to state explicitly if they apply**, because their absence is invisible in the
figures: whether the **sample boundary was masked** (phase 1b — unmasked, every per-voxel
statistic is averaged over vacuum that scores ~0.92 completeness), and whether the **grain
map was compared against its majority-class null** (§5.4, hard rule 9).

Produce the report through the `beamreport` framework (separate repo; see its `SPEC.md`).
State the grain-segmentation path used (`find_grains` vs the fast-path), whether strain
patches were dark-subtracted, and the attenuation setting — all three change how the numbers
should be read.

## 5.4 Cross-modal validation

If tomography of the same sample exists, it is ground truth for morphology (cracks, voids,
porosity). Register it before overlaying: the tomo reconstruction software's frame
convention (flip / rotation) and its **rotation-centre pixel** rarely match the MIDAS lab
frame, and neither is usually recorded — you need both to place a tomo feature on the
diffraction map. Confirm the **layer height** (`samY` ↔ reconstructed slice) first; the
in-plane registration is the harder half and may need a fiducial visible in both modalities.
Report a cross-modal overlay only once the transform is pinned, not eyeballed.

**Score any map-vs-map agreement against its majority-class null** (hard rule 9). Grain-ID
maps are dominated by one or two large grains, so a raw voxel-wise percentage is mostly
measuring "both maps found the big grain". Measured: a tomographic grain-ID map agreed with
the point-by-point map on **60.1 %** of voxels against a constant-map null of **65.2 %** —
Cohen's κ = 0.399. That is not weak agreement, it is none, and the bare 60 % hides it.

⚠️ **A tomographic *grain-intensity* map is not a material map.** It answers "did one of the
listed grains reconstruct here", so real material carrying no listed grain reads as dark.
Do not use it to locate the sample boundary (phase 1b §1b.5) or to claim a void.

## 5.5 Update the runbook

Before finishing, update the runbook's pick-up point: what ran, the healthy numbers you
observed (with their conditions), and where you stopped. A stale pick-up point makes the
next session re-derive what you already know.
