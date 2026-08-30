# pf-HEDM parameter reference

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).
> Provenance for any value quoted in the procedure lives in the notebook ledger; this file
> explains what each parameter *does* and the ones with a PF-specific meaning or trap.

## Scan-mode parameters (`ScanGeometry` / pipeline config)

| Parameter | Meaning | Trap |
|---|---|---|
| `scan_mode` | `"pf"` selects the scanning `STAGE_ORDER`; `"ff"` is single-scan | PF requires `n_scans ≥ 2` |
| `n_scans` | number of translation positions (= scan files) | the voxel grid is `n_scans × n_scans`, not `n_scans` |
| `BeamSize` | in-plane beam width (µm) | **the C adds 0.1 µm to it on parse** (`IndexerUnified.c:2627`), so the `BeamSize/2` fallback below is `(BeamSize+0.1)/2`, not half the beam. Never rely on the fallback |
| `scan_pos_tol_um` / `ScanPosTol` | half-width of the beam-position gate (µm) | **the gate is applied in the MATCHING loop, not just seeding** (`IndexerUnified.c:1006` and `3447`). Falls back to `(BeamSize+0.1)/2` when absent — 0.80 µm at `BeamSize 1.5`, where the pipeline writes 0.75. **Always pass it explicitly on a hand-run**: the 6.7 % difference measured +14.7 % accepted solutions and a changed winner in 10.5 % of voxels (spine hard rule 12) |
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
| `MinMatchesToAcceptFrac`, `MinNrSpots` | acceptance gates | a low-completeness scan can fall below these, **and every value in common use sits at or below the measured chance ceiling** (measured on five layers: **none / 0.5333 / 0.6957 / 0.7500 / 0.8333** — and NOT ordered by spot density, so it must be measured on the layer in hand). The parameter registry ships `typical=0.8` (`midas_params/midas_params/registry.py:892`) — above the sparse ceiling, *below* the dense one. The reference campaign's own file used `Completeness 0.5` (`_comp_params.py:138`), which admits roughly one null voxel per two real ones. **Omitting the key is the worst case, not the safe one:** `FitSetupParamsAllZarr.c` then falls back to `0` — accept everything. Set it just above the ceiling **measured on that layer** (phase 7 §7.3) — the ceiling is not ordered by spot density and must never be ported between layers |
| `LatticeConstant` / `LatticeParameter` | the phase's cell | **it is also the ZERO of the strain measurement.** The refiner gauges `(dsObs−ds0)/ds0` against the `ds0` this implies, so a cell that is not the sample's own rails strain components and depresses completeness. Pin it from the observed rings (phase-2 §2.5) — never by averaging refined per-grain cells, which is a feedback loop |
| `MargStrain` | half-width of the per-component strain search box, absolute strain | default **0.01 = ±10000 µε** (a compiled-in constant before 2026-08-21). Railing here means the reference cell is wrong — **fix the cell, do not widen the box**. `0` keeps the default |
| `MargABC` / `MargABG` | lattice length / angle refinement tolerance | `MargABG` is applied as a **percent**, not degrees (`alpha*(1 − MargABG/100)`) despite reading like an angle |

## Zip-baked analysis parameters (the trap)

`MaxNPeaks`, integration thresholds, and the ring set used by peakfit are written into each
`*.MIDAS.zip` at zip-convert time and read from there, **not** from the live `paramstest`.
Changing them requires regenerating the zips (phase 2.2).

## Seeding config (PF-only)

| Field | Meaning |
|---|---|
| `SeedingConfig.mode` | `"unseeded"` (intractable on scanning data), `"ff"` (from a supplied `Grains.csv`), `"merged-ff"` (synthesised). ⚠ **merged-FF is a SEEDING route only — never a grain-counting one.** Its 1-row `positions.csv` sets `nScans_ == 1`, so `doScanFilter` is 0 and the beam gate is off in the matching loop; the ω-shuffle null *beat* the real arm on every statistic (phase 7 §7.6). It also measured 5.6× more core-hours than PF unseeded on the same layer |
| `grains_file` | path to the FF `Grains.csv` for `mode="ff"` |
| `dedup_misorientation_deg` | collapse symmetry-equivalent seed orientations |

## The seed / adapter files (c-omp → pf-odf bridge)

| File | Format | Written by |
|---|---|---|
| `SpotsToIndex.csv` (PF) | 5-col: `voxNr SpId nSpotsBest _ bestSolIdx` | `midas_fit_grain.scan_seed.write_pf_seed_file` (from `IndexBest_all.bin`) |
| `FitBest_<vox>_<sp>.csv` | multi-block: header + result row + repeated header + per-spot rows | the c-omp refiner |
| `Result_OrientPos_voxel_<v>.csv` | 2-line clean **39-col** (was documented here as 43 — wrong; verified 39 on the s5pf1/L2 reference layer). **45-col** from midas-fit-grain 0.9.0, which appends `PosErr/OmeErr/InternalAngle` x `Pre/Post` at 39-44 | python refiner directly, or `midas_fit_grain.fitbest_adapter` from FitBest |

## Reconstruction-space parameters (PF-only, `ReconConfig`)

Phase 6. All of these are no-ops for a grain-map / pf-odf run except the two diagnostics,
which are worth running anyway because they improve the **point-by-point** result.

| Parameter | CLI | Meaning | Trap |
|---|---|---|---|
| `do_tomo` | — | run the tomo/vmap tail | **Leave it OFF for a point-by-point map.** Tomo-seeded re-indexing gave 2433/2601 voxels and 367 below completeness 0.5, against 2601 and 11 direct |
| `sino_type` | — | which variant the reconstructor reads: `raw` / `norm` / `abs` / `normabs` / `softsum` / `clean` | `norm` divides each row by its own max and **destroys the volume information** — it is not a physical normalisation. `abs` came back degenerate on the reference run; check it is populated |
| `sino_conc_threshold` | `--sino-conc-threshold` | drop sino rows carrying less than this fraction of their intensity on the grain's own fitted sinusoid, into `sinos_clean_*.bin` | `0.0` = off. **0.35 is calibrated and transfers unchanged** — do not retune it. It fixes **position**, not shape: the reconstruction residual does not move |
| `sino_conc_min_band_um` | `--sino-conc-min-band` | floor on the acceptance band, µm | default 4.0. On a coarse scan this is **sub-bin** and the filter effectively works in whole bins |
| `out_of_field_occupancy` | `--out-of-field-occupancy` | warn when a grain's rows light up more than this fraction of the scan line | default 0.65, `0` disables. **Diagnostic only — never a filter.** Excluding flagged grains took map agreement 47.8 % → 11.0 % (hard rule 10) |
| `method`, `mlem_iter`, `osem_subsets` | — | reconstructor and its iterations | the residual was **invariant** across FBP/SIRT/MLEM on the reference campaign; changing these is not a lever on the shape problem |
| `cull_min_size` | `--cull-min-size` | drop connected components smaller than this | a *segmentation* knob, not a quality one; it changes the grain count without changing any per-voxel result |

**Version floor for anything in this table: `midas_pipeline ≥ 0.11.0`** — below it the FBP
crop is off by one voxel in both axes for every odd `n_scans` (phase 6 §6.2), a constant
offset that mis-registers every shape against the voxel map without looking wrong.

## pf-odf strain parameters

| Parameter | Meaning | Default |
|---|---|---|
| `subtract_background` | per-patch dark subtraction (raw frames) | `False` — set `True` for raw, off for dark-subtracted caches |
| `n_frames`, `omega_step` | acquisition frame mapping for patch cropping | pass explicitly; fallback (bin size) is wrong |
| `identifiability` | `PROJECT_EPS_MEAN_ZERO` (lattice absorbs bulk) vs `FREE` | project-mean-zero |
| `chunk_size_g` | voxel chunk for the forward (VRAM) | shrink on OOM |
| `inner_steps`, `optimizer`, `lr_*` | optimiser controls | adam, ~60 steps |
