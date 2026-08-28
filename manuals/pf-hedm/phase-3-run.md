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
(`voxel_cleanup`…`em_refine`) is reconstruction-space work and is normally **skipped** for a
grain-map / pf-odf run — `--skip` it. Do **not** reach for `--only <stages>`: it drops the
essential PF-only stages between the ones you name.

> **The tail is documented, and it is not just "the part you skip":**
> [`phase-6-reconstruction.md`](phase-6-reconstruction.md). Two of its diagnostics improve
> the *point-by-point* result and are worth running even when shapes are not the goal — the
> concentration filter took one grain's fitted position from 5.59 µm to 1.11 µm. Its other
> job is to stop you quoting the shapes, which are a known open problem (§6.7). Note
> `do_tomo=True` makes the point map **worse** (§6.3), so skipping the tail is the right
> default for the map itself.

Run naturally (no `--only`), on GPU, under `nohup`/`setsid` with a log. Outputs go under the
run's own `LayerNr_<N>/` — never `/tmp`.

### 3.1a Three flags that fail silently (measured 2026-08-21)

All three finish with **exit 0** and a plausible-looking run.

1. **`--num-files-per-scan` defaults to 1.** For a per-frame TIFF/`.tif.bz2`
   series you MUST pass it (e.g. `--num-files-per-scan 1440`). The parameter
   file's `NrFilesPerSweep` is written into the zarr but does **not** drive the
   file list (`midas_pipeline/stages/zip_convert.py`, `num_files_per_scan: int = 1`).
   Symptom: `zip_convert` finishes suspiciously fast, `exchange/data` is
   `(1, ny, nz)`, every stage runs, the layer ends with **0 voxels**.
   Check: `zarr.open(zip)["exchange/data"].shape[0]` must equal your sweep length.

2. **`--scan-work-dir` must be an ABSOLUTE path.** A relative value is joined
   twice (`…/scans/167051/scans/167051/…`) and every zip fails. Omitting it
   writes per-scan zips and `Temp/` **into the raw data tree**, which you may not
   own and will not think to clean.

3. **Stale layer state silently poisons a re-run.** `transforms` caches on the
   per-layer `InputAllExtraInfoFittingAll*.csv`, and `midas_state.h5` records
   stage completion. After a failed attempt these survive, and the next run logs
   `transforms(PF): 13 ok` in **0.04 s** with no per-scan spot counts, then
   indexes the OLD spots. Two tells: transforms taking ~0 s, and
   `resume: 'find_grains' already complete, skipping`. **Prefer a fresh result
   directory**; if you must reuse one, delete `InputAllExtraInfoFittingAll*.csv`,
   `Data.bin`, `ExtraInfo.bin`, `Spots.bin`, `hkls.csv`, `Output/`, `Results/`,
   `Recons/`, `FitBest_comp/` **and `midas_state.h5`**.

### 3.1b Throughput: peakfit is I/O-bound, and the page cache is the lever

Measured on a 64-core host, 13 scans × 1440 frames: peakfit pegs only
**~3.5–4 of 64 cores** (load ~3), so it is **not** CPU-bound — do not "fix" it
with more workers. `--scan-workers 8` was no faster than 4.

What does matter is whether the zarr is still in the page cache: peakfit run
immediately after `zip_convert` took **467 s/scan**, the same work on cold zips
took **1009 s/scan**. So drive the run **layer-by-layer (zip → peakfit → …)**
rather than pre-building all the zips and then peakfitting. On a
disk-constrained host, delete each layer's `*.MIDAS.zip` once its `Results/` are
written — they rebuild from raw in ~100 s and cost ~2 GB per scan.

### 3.1c Indexing wants ALL the cores — do not fund concurrency from it

Peakfit being I/O-bound (§3.1b) tempts you to conclude the pipeline underuses
the box and to run several layers at once on a slice of the cores each. **Do not
generalise across stages.** Measured per stage on one 13-scan layer, varying
only `numProcs` on the c-omp indexer (same binned inputs, byte-identical
20.5 MB `IndexBest_all.bin` every time, so only speed changes):

| indexer cores | time |
|---|---|
| 64 | **27.9 min** (saturates 63.5/64 cores) |
| 32 | 40.9 min |
| ~20 | 51.6–68.1 min |

64 vs 32 is 1.47× for 2× the cores (~73 % efficiency) — sublinear but well worth
having, and it keeps scaling to the full box. Two index slots at 32 cores each
are **slower** than one at 64.

Running 3 whole layers at 20 cores each was measured **worse** than serial: it
starved the only CPU-bound stage and still left the machine ~48 % idle
(3019 % of 6400 %, 0 % iowait).

**The configuration that does work is a two-stage software pipeline**, because
the halves want disjoint resources and take about the same wall time:

| half | stages | cost | resource |
|---|---|---|---|
| prep | `zip_convert → peakfit → transforms → binning` | ~20–27 min | disk, ~4 cores |
| index | `indexing → refinement → find_grains` | ~28 min at 64 cores | all 64 cores |

Overlap `prep(N+1)` with `index(N)` and the wall time per layer is
`max(prep, index)` (~28 min) instead of the sum (~50–55 min), with indexing
still holding the whole box. Exactly one prep and one index live at a time, so
disk and CPU each have a single consumer. Both halves already exist in the CLI:
prep is `--skip indexing --skip refinement --skip find_grains …`, index is
`--resume from --from indexing`.

The index half never reads the zips (it reads `Data.bin`/`Spots.bin`/
`nData.bin`), so free them at the end of **prep** — only one layer's ~26 GB is
then ever live.

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
orientations. The **c-omp** indexer (the only one, and the default) writes
the consolidated `Output/IndexBest_all.bin` (tens of GB). Confirm it actually took the
seeded path: a seeded layer finishes in tens of minutes; if it runs for hours pegging every
core, the `GrainsFile` line did not reach the binary (phase 2.4).

## 3.4 Refinement — c-omp, and the two files it needs

**c-omp is now the only backend either stage accepts, in PF or FF.**
`--indexer-backend` and `--refine-backend` take `c-omp` and nothing else — argparse
restricts the choice list and `_require_comp_backends` (`midas_pipeline/config.py:666`)
re-checks it from `__post_init__`, so a notebook driving the config directly cannot bypass
it either. Both default to `c-omp`; passing them is optional now.

**What that closed.** The refiner used to fall back to **python + torch + CUDA** with no
flag, so a run whose log said `indexing(PF, c-omp)` could be silently half on the GPU path.
Measured on an FF layer (24 900 seeds, 471 k spots) it died after ~12 s with a bare
`subprocess.CalledProcessError … non-zero exit status 1` and **no child traceback** — the
wrapper swallows the subprocess's stderr — and because `transforms` re-runs on resume, each
retry cost a ~90-minute re-index. The Python refiner is known-broken; it is not a slower
fallback, and `--refine-solver` / `--refine-loss` / `--refine-mode` / `--pf-refine-mode` /
`--use-bounds` / `--bound-*` were removed with it. **Still check the log says `c-omp` on
both lines** — that now confirms the binaries were found and ran.

On speed, separately: a full layer refines in ~a minute or two on c-omp vs a GIL-bound
multi-day python refine. The pipeline handles the two things the C refiner needs in PF mode
automatically (shipped in `midas_pipeline.stages.refinement`):

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
LayerNr_<N>/SpotMatrix.csv                        per-voxel spots: observed AND
                                                  predicted, plus one row per
                                                  reflection that was NEVER FOUND
```

For a **grain map only**, go to [`phase-5-read-report.md`](phase-5-read-report.md). For
**per-voxel strain**, go to [`phase-4-strain.md`](phase-4-strain.md).
