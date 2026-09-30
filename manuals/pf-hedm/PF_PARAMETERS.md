# pf-HEDM parameter reference

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).
> Provenance for any value quoted in the procedure lives in the notebook ledger; this file
> explains what each parameter *does* and the ones with a PF-specific meaning or trap.

## Scan-mode parameters (`ScanGeometry` / pipeline config)

| Parameter | Meaning | Trap |
|---|---|---|
| `scan_mode` | `"pf"` selects the scanning `STAGE_ORDER`; `"ff"` is single-scan | PF requires `n_scans ≥ 2` |
| `n_scans` | number of translation positions (= scan files) | the voxel grid is `n_scans × n_scans`, not `n_scans` |
| `BeamSize` | in-plane beam width (µm) | **the C adds 0.1 µm to it on parse** (`IndexerUnified.c:2830`), so the `BeamSize/2` fallback below is `(BeamSize+0.1)/2`, not half the beam. Never rely on the fallback |
| `scan_pos_tol_um` / `ScanPosTol` | half-width of the beam-position gate (µm) | **the gate is applied in the MATCHING loop, not just seeding** (`IndexerUnified.c:1174` and `3447`). Falls back to `(BeamSize+0.1)/2` when absent — 0.80 µm at `BeamSize 1.5`, where the pipeline writes 0.75. **Always pass it explicitly on a hand-run**: the 6.7 % difference measured +14.7 % accepted solutions and a changed winner in 10.5 % of voxels (spine hard rule 12) |
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
| `MinMatchesToAcceptFrac`, `MinNrSpots` | acceptance gates | a low-completeness scan can fall below these, **and every value in common use sits at or below the measured chance ceiling** (measured on five layers: **none / 0.5333 / 0.6957 / 0.7500 / 0.8333** — and NOT ordered by spot density, so it must be measured on the layer in hand). The parameter registry ships `typical=0.8` (`midas_params/midas_params/registry.py:898`) — above the sparse ceiling, *below* the dense one. The reference campaign's own file used `Completeness 0.5` (`_comp_params.py:140`), which admits roughly one null voxel per two real ones. **Omitting the key is the worst case, not the safe one:** `FitSetupParamsAllZarr.c` then falls back to `0` — accept everything. Set it just above the ceiling **measured on that layer** (phase 7 §7.3) — the ceiling is not ordered by spot density and must never be ported between layers |
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
| `augment_ff_layer` / `--seed-augment-ff-layer` | merged-FF layer whose dropped orientations are added back to an `ff` seed, gated by an ω-shuffled re-index of that layer (phase 7 §7.10) | `ff` mode only. Adds a shuffle + re-bin + re-index of the merged-FF layer to the seeding stage. The gate screens noise, not twin ghosts; the PBP argmax is what keeps those out |
| `augment_null_runs` / `--seed-augment-null-runs` | number of shuffled runs; the gate is the max over all of them | default 1. The added count moved 68 → 94 → 119 with the null seed on ma5608, so use ≥2 when the count matters |

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
| `method`, `mlem_iter`, `osem_subsets` | `--recon-method`, `--mlem-iter`, `--osem-subsets` | reconstructor and its iterations | **no reconstructor wins everywhere: run `--recon-method all` (phase 6 §6.10) and score them on the sample.** The residual was invariant across FBP/SIRT/MLEM on the reference campaign; on ESRF ma5608 the rewritten MLEM beat FBP on every phantom and lost on the real layer (half-split 0.789 vs 0.864, 17/204 grains piled on the border) — phase 6 §6.7 |
| `candidate_brightness` | `--no-candidate-brightness` to skip | write `Output/CandidateBrightness.npz`: mean `ln(I / ring median)` of every candidate's matched spots, plus each voxel's best contender within 0.05 of the winner | default **on**; needs the per-scan CSVs and `IDsMergedScanning.csv`, skipped with a warning without them. A diagnostic: it changes no voxel. DIAGNOSIS: ambiguity.dim_contender |
| `fusion.sibling_merge_deg` / `--sibling-merge-deg` | find_grains, before the spot association | **opt-in, default 0 (C parity).** Merges unique grains whose representative orientations are within this many degrees. `process_spots` drops any spot shared by two grains (1.0/1.0 deg) from BOTH, so near-duplicate "sibling" grains (greedy clustering leaves representatives 0.1-1 deg apart) keep few sinogram rows. Measured on 20-ID-E Fe9Cr 15 N: 14 of 46 grains had nr < 30 (28 % of solved voxels); `--sibling-merge-deg 1.0` -> 38 grains, none < 30, median nr 71.5 -> 93.5 (Lab Notebook §10, bt_20id_sep26b S10). **Does NOT improve the starved voxels' reconstructions** (S12: agreement with the per-voxel map on them -0.05 / -0.04 15 N, -0.01 / -0.03 crack for MLEM / FBP; registered REFUTE); FBP label agreement on the other voxels rises (+0.17 15 N, +0.06 crack) only because the starved siblings stop winning the FBP argmax through streaks (verified, provisional: zeroing them without merging gives most or all of it; MLEM unchanged; size depends on the label rule). Untested: whether 1-2 deg merges join distinct crystals (a warning is logged if a chain spans > 2x) |
| `recon.sino_scan_tol_um` / `--sino-scan-tol` | indexing-mode sinograms (`--sino-source indexing`): a spot is kept if the voxel's projected scan position is within this of the spot's scan | default 1.5 µm is **too tight for coarse scans**: a voxel projects anywhere between scan positions, so use ~half the scan step (5 µm at 10 µm). The projection convention was y-mirrored before 2026-09-28 (kept ~37 % of each grain's signal; §1b.6) |
| `recon.method` / `--recon-method all` | fbp + mlem + voxelmap with `Recons/ReconQuality.json` and `Recons/labels_<m>.npy` | which reconstruction is better depends on the data (alumina FBP, Fe9Cr MLEM; phase 6 §6.10); the TIFs stay FBP's |
| `recon.sample_mask` / `--sample-mask` | `(n_scans, n_scans)` .npy/.tif, nonzero = sample | reconstructions zeroed outside it; vacuum voxels get -1; the quality report is restricted to it. From a tomogram: `python -m midas_pipeline.recon.sample_mask` |
| (binning diagnostic) | `omega_coverage.json` | blocked omega windows inside `OmegaRange` (load frames, furnaces); warns with paste-ready ranges |
| (find_grains diagnostic) | `Output/SineConsistency.csv` | one grain = one sine; flags grains > 5 scans |
| `brightness_tiebreak_margin` | `--brightness-tiebreak-margin` | among candidates within this completeness margin of a voxel's best, the brightest matched spots win | default `0.0` = off. **Opt-in and not validated on a phantom** — do not use it for a quoted map |
| `sino_tol_ome_deg`, `sino_tol_eta_deg` | `--sino-tol-ome`, `--sino-tol-eta` | half-width of the omega / eta window that puts a spot in a tolerance-mode sinogram | default −1 = 2 × \|OmegaStep\| (1.0° if there is no OmegaStep); it used to be a fixed 1.0°. A positive value restores any window. Narrower windows admit fewer spurious cells but were measured at only two eta widths (0.15° and 1°) |
| `cull_min_size` | `--cull-min-size` | drop connected components smaller than this | a *segmentation* knob, not a quality one; it changes the grain count without changing any per-voxel result |

**Version floor for the FBP crop: `midas_pipeline ≥ 0.11.0`** (`--recon-method all`, the diagnostics and the seed-augment keys need 0.19.0) — below 0.11.0 the FBP
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
