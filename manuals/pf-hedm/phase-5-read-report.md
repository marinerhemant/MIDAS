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
  statistics. These hold even on attenuated data.
- **Provisional:** per-voxel absolute strain magnitudes when the scan is signal-limited
  (envelope §2). Report the *pattern* (does it localise where deformation is expected?) and
  the spatial-structure statistic (Moran's I), not calibrated magnitudes.

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

## 5.5 Update the runbook

Before finishing, update the runbook's pick-up point: what ran, the healthy numbers you
observed (with their conditions), and where you stopped. A stale pick-up point makes the
next session re-derive what you already know.
