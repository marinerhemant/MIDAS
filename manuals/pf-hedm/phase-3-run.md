# Phase 3 — Run the PF pipeline

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).

## 3.1 The stage order

`midas_pipeline` selects `STAGE_ORDER` by `scan_mode`. For PF (`scan_mode='pf'`,
`n_scans≥2`):

```
zip_convert → hkl → peakfit → merge_overlaps → calc_radius → transforms →
cross_det_merge → global_powder → [merge_scans] → [seeding] → binning →
indexing → refinement → [find_grains] → (voxel_cleanup, sinogen, reconstruct,
                                          fuse, potts, em_refine — tomo/vmap tail)
```

`[…]` marks the **PF-only** stages. `merge_scans` + `seeding` sit between `transforms` and
`binning`; `refinement` runs **before** `find_grains`. The tomo/vmap tail
(`voxel_cleanup`…`em_refine`) is for reconstruction-space work and is normally **skipped**
for a grain-map / pf-odf run — `--skip` it. Do **not** reach for `--only <stages>`: it drops
the essential PF-only stages between the ones you name.

Run naturally (no `--only`), on GPU, under `nohup`/`setsid` with a log. Outputs go under the
run's own `LayerNr_<N>/` — never `/tmp`.

## 3.2 Binning — the memory wall

A busy scanning layer generates hundreds of millions of raw spots; assigning them to
(spot × η-bin × ω-bin) is billions of pairs and OOMs a naive implementation on both GPU and
CPU. Current `midas-transforms` chunks the pair assignment (`MIDAS_BIN_PAIR_CHUNK`, per-ring)
— if binning is kernel-killed, that is the lever, not a smaller GPU. `StepsizeOrient` is
**overloaded**: it sets both the indexer orientation grid *and* the binning ω-margin —
coarsening it to speed indexing can OOM binning. Keep it in the tested window and decouple
by hand if needed.

## 3.3 Indexing — seeded

With the FF seed wired (phase 2.4), the scanning indexer combs only each voxel's candidate
orientations. The **c-omp** indexer (`--indexer-backend c-omp`) is the fast path and writes
the consolidated `Output/IndexBest_all.bin` (tens of GB). Confirm it actually took the
seeded path: a seeded layer finishes in tens of minutes; if it runs for hours pegging every
core, the `GrainsFile` line did not reach the binary (phase 2.4).

## 3.4 Refinement — c-omp, and the two files it needs

**Use `--refine-backend c-omp` for speed** (a full layer refines in ~a minute or two vs a
GIL-bound multi-day python refine). The pipeline now handles the two things the C refiner
needs in PF mode automatically (shipped in `midas_pipeline.stages.refinement`):

1. It **synthesises the 5-column `SpotsToIndex.csv`** from `IndexBest_all.bin`
   (`midas_fit_grain.scan_seed.write_pf_seed_file`) — the C refiner picks each voxel's seed
   from it. Without it, the refiner exits 0 and refines nothing (hard rule / DIAGNOSIS).
2. It runs the refiner into a dedicated `FitBest_comp/` dir, then **adapts the multi-block
   `FitBest_<vox>_<sp>.csv` into clean `Result_OrientPos_voxel_<vox>.csv`**
   (`midas_fit_grain.fitbest_adapter`) — the form pf-odf and `consolidation_pf` read.

If you drive the C refiner by hand, do both yourself. The python refiner writes
`Result_OrientPos_voxel_*.csv` directly (no adapter) but is far slower; use it only to
cross-check c-omp on a few voxels (they agree to <0.2° orientation, <0.01 Å lattice).

## 3.5 find_grains — and when to fast-path it

`find_grains` clusters the indexer's per-voxel best orientations into grains and writes
`Output/voxel_grid.csv` (`voxel_idx x_um y_um z_um grain_id`) + `UniqueOrientations.csv`.

On a **high-spread or cracked** map its per-voxel clustering pass (many candidates per voxel,
O(N²) cross-voxel dedup) can run for hours and never finish — a documented failure mode
(DIAGNOSIS). Its GPU/parallel fast-paths help
(`MIDAS_FINDGRAINS_NJOBS`, `MIDAS_FINDGRAINS_CLUSTER=gpu`), but if it is still impractical you
can build `voxel_grid.csv` directly by **connected-components segmentation of the refined
per-voxel orientations** (neighbour-consensus cleanup of salt-and-pepper wrong picks, then
union-find at a misorientation tolerance). That grouping is exactly what pf-odf's
`load_pf_grain` needs, and it runs in seconds. Note in the report which grain-segmentation
path produced the map.

## 3.6 What you have at the end of phase 3

```
LayerNr_<N>/Output/voxel_grid.csv               voxel→grain map
LayerNr_<N>/Output/IndexBest_all.bin            consolidated indexer output
LayerNr_<N>/Output/UniqueOrientations.csv       unique grain orientations
LayerNr_<N>/Results/Result_OrientPos_voxel_*.csv  per-voxel refined orientation+lattice
```

For a **grain map only**, go to [`phase-5-read-report.md`](phase-5-read-report.md). For
**per-voxel strain**, go to [`phase-4-strain.md`](phase-4-strain.md).
