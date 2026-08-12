# pf-HEDM parameter reference

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).
> Provenance for any value quoted in the procedure lives in the notebook ledger; this file
> explains what each parameter *does* and the ones with a PF-specific meaning or trap.

## Scan-mode parameters (`ScanGeometry` / pipeline config)

| Parameter | Meaning | Trap |
|---|---|---|
| `scan_mode` | `"pf"` selects the scanning `STAGE_ORDER`; `"ff"` is single-scan | PF requires `n_scans ≥ 2` |
| `n_scans` | number of translation positions (= scan files) | the voxel grid is `n_scans × n_scans`, not `n_scans` |
| `BeamSize` | in-plane beam width (µm) | sets `scan_pos_tol` when that is 0; the voxel/beam membership width |
| `scan_pos_tol_um` | tolerance for attributing a voxel to a scan | defaults to `BeamSize/2` |
| `Hbeam` / `BeamThickness` | per-layer beam height (µm) | **never the sample size** — an oversized value lets Z roam (hard rule) |
| `friedel_symmetric_scan_filter` | use Friedel symmetry in the scan filter | affects which reflections seed each voxel |

## Geometry (shared with FF)

`Lsd`, `BC`/`YBC`/`ZBC`, `ty`/`tz`, `tx`, `px`, `Wavelength`, `p0…` (distortion), `RhoD`,
`OmegaStart`, `OmegaStep`, `OmegaRange`. See `manuals/ff-hedm/` for calibration. PF notes:

- **`OmegaStep` sign** — reconcile against the raw `SMS/aero` encoder (phase 1.1); the param
  file can disagree.
- **`OmegaRange`** — one or more valid spans. Gaps (blocked ranges) reduce reflections per
  voxel and are a §2 envelope limit, not a defect.
- **`tx`** — not constrainable by a powder calibrant (rotation about the beam); hold fixed in
  powder calibration, refine from grains after.

## Indexing / refinement

| Parameter | Meaning | Trap |
|---|---|---|
| `RingNumbers` | rings to index/refine on | indexing rings ≠ best strain rings; prefer bright, on-detector rings for strain |
| `StepsizeOrient` | indexer orientation grid step | **overloaded** — also sets the binning ω-margin (`omemargin = MarginOme + 0.5·StepsizeOrient/|sin η|`); coarsening it to speed indexing can OOM binning |
| `StepsizePos` | indexer position grid step | PF fixes position to the voxel grid; less critical than in FF |
| `GrainsFile <path>` | FF seed for the c-omp indexer/refiner | **required** for the c-omp seeded path (`isGrainsInput=1`); absent → silent full-grid comb |
| `MinMatchesToAcceptFrac`, `MinNrSpots` | acceptance gates | a low-completeness scan can fall below these |

## Zip-baked analysis parameters (the trap)

`MaxNPeaks`, integration thresholds, and the ring set used by peakfit are written into each
`*.MIDAS.zip` at zip-convert time and read from there, **not** from the live `paramstest`.
Changing them requires regenerating the zips (phase 2.2).

## Seeding config (PF-only)

| Field | Meaning |
|---|---|
| `SeedingConfig.mode` | `"unseeded"` (intractable on scanning data), `"ff"` (from a supplied `Grains.csv`), `"merged-ff"` (synthesised) |
| `grains_file` | path to the FF `Grains.csv` for `mode="ff"` |
| `dedup_misorientation_deg` | collapse symmetry-equivalent seed orientations |

## The seed / adapter files (c-omp → pf-odf bridge)

| File | Format | Written by |
|---|---|---|
| `SpotsToIndex.csv` (PF) | 5-col: `voxNr SpId nSpotsBest _ bestSolIdx` | `midas_fit_grain.scan_seed.write_pf_seed_file` (from `IndexBest_all.bin`) |
| `FitBest_<vox>_<sp>.csv` | multi-block: header + result row + repeated header + per-spot rows | the c-omp refiner |
| `Result_OrientPos_voxel_<v>.csv` | 2-line clean 43-col | python refiner directly, or `midas_fit_grain.fitbest_adapter` from FitBest |

## pf-odf strain parameters

| Parameter | Meaning | Default |
|---|---|---|
| `subtract_background` | per-patch dark subtraction (raw frames) | `False` — set `True` for raw, off for dark-subtracted caches |
| `n_frames`, `omega_step` | acquisition frame mapping for patch cropping | pass explicitly; fallback (bin size) is wrong |
| `identifiability` | `PROJECT_EPS_MEAN_ZERO` (lattice absorbs bulk) vs `FREE` | project-mean-zero |
| `chunk_size_g` | voxel chunk for the forward (VRAM) | shrink on OOM |
| `inner_steps`, `optimizer`, `lr_*` | optimiser controls | adam, ~60 steps |
