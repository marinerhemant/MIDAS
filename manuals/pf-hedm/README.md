# pf-HEDM — scanning-3DXRD grain map + per-voxel peak-shape strain

**Use this doc to start a fresh session on a dataset this pipeline has never seen.**
Paste it in together with `LAB_NOTEBOOK.md`, then give, or work out from the data:

```
Data folder:     <ABSOLUTE PATH>   # per-scan detector files (one file per translation)
Metadata / geom: <ABSOLUTE PATH>   # calibration + scan positions, or "find it"
Sample material: <e.g. FCC Ni / unknown — tell me from the data>
Goal:            grain map only | grain map + per-voxel strain (pf-odf)
```

**Scope.** Every recipe here was measured on **point/line-focused scanning 3DXRD** (a.k.a.
pf-HEDM / scanning-HEDM): the sample is **translated across a narrow beam** at each of many
**ω rotations**, one detector panel, one reconstructed layer per run. The reconstruction is
a **2-D voxel grid** (translations × translations) of per-voxel orientation + lattice; the
optional strain extension (`midas_pf_odf`) fits the **full Bragg-peak shape** per voxel. If
your data is far-field (grains, not voxels — use `ff-hedm`), near-field (topological, box
beam — `nf-hedm`), multi-panel, or a genuine 3-D translation stack, **stop and ask** rather
than adapting these recipes — the scan geometry, the seeding path, and the strain
identifiability below assume this configuration throughout.

<!-- The scope gate is not boilerplate. The single most common silent failure is a
     far-field recipe applied to a scanning dataset: FF refines position, PF fixes it to
     the voxel grid; FF's SpotsToIndex is a spot list, PF's is a 5-col per-voxel seed. -->

**Two instrument configurations** are covered — **1-ID scanning** and **20-ID HT-HEDM
Varex**. They differ in the ω sign, the dark group, and the entry point, and each has its
own reference campaign. The split is in [`INSTRUMENT.md`](INSTRUMENT.md); where a recipe
diverges it is called out inline as "20-ID:". Anything else, **stop and ask**.

## The doc set — what to read when

| File | Holds | Read it |
|---|---|---|
| **`README.md`** (this) | scope gate, install gate, the order, hard rules, halt conditions | always |
| `INSTRUMENT.md` | the two configurations, the ω sign on each, **the two code generations** | before the first run on a new station |
| `phase-0-survey.md` | is it scanning 3DXRD? translations, ω, per-scan files, positions | first |
| `phase-1-geometry.md` | calibration, the ω sign, the `positions.csv` convention | before configuring |
| `phase-1b-sample-boundary.md` | **where the sample ends** — from the spot-count sinogram, not from completeness | when any part of the grid may be vacuum |
| `phase-2-configure.md` | params, the zip-baked param trap, ring selection, the FF seed | before running |
| `phase-3-run.md` | the MIDAS PF pipeline stage-by-stage | the reconstruction |
| `phase-4-strain.md` | `midas_pf_odf` peak-shape per-voxel strain (the novel extension) | if strain is the goal |
| `phase-5-read-report.md` | read `voxel_grid.csv` + Results, orientation/KAM/GROD/strain maps, report | at the end |
| `phase-6-reconstruction.md` | sinograms, the concentration filter, the occupancy flag, **why shapes are not quotable** | **before quoting any shape or grain-ID map**; for better fitted positions |
| `phase-7-validation.md` | **the ω-shuffle null, the chance ceiling, the spatial-coherence screen** | **before quoting any per-voxel result, grain count or acceptance threshold** |
| `PF_PARAMETERS.md` | the PF-specific parameter reference | when tuning |
| `DIAGNOSIS.md` | symptom → test → cause → lever | **when something looks wrong** |
| `RUNBOOK.md` | where it runs, what healthy looks like, pick-up point, **and §R4 multi-layer campaigns** | on resume; **§R4 before driving more than one layer** |
| `LAB_NOTEBOOK.md` | evidence, ledger, **retracted claims** — Lab Notebook §1–§6 the 1-ID campaign, **Lab Notebook §7 the 20-ID Varex campaign**, **Lab Notebook §8 the validation / null campaign** | before re-investigating |

## STOP — read this before touching anything

### When to stop and come back with a question

**"Get back to me if you get stuck" does not fire here.** Three failures finish and look
completely healthy: a **mirrored voxel map** (a sign error in the scan→voxel mapping), a
**c-omp refine that wrote nothing** (the refiner ran, exited 0, and refined zero voxels
because its seed file was the wrong format), and **strain fit on a raw pedestal** (the fit
converges to a spatially-structured but wrong field). The run completes; the output is
plausible.

So the trigger is not confusion. **Halt on these named conditions, whether or not anything
seems wrong:**

| Condition | Why you cannot decide it yourself |
|---|---|
| A **grain shape**, a `Recons/` image, or a tomographic grain-ID map is about to be quoted | Shapes render, look like grains, and some of them *are* right. On the reference campaign the residual sat at 0.82–0.84 invariant across every reconstructor, eleven mechanisms were tested and **the cause is still unknown**. Positions are fine; shapes are not. Phase 6 §6.7. |
| Per-voxel statistics are about to be reported over a grid that **may contain vacuum** | Vacuum voxels inherit a neighbouring grain's orientation and score ~0.92 completeness, so the completeness map reads "material everywhere" and is wrong. The boundary must come from the spot-count sinogram (phase 1b), and it is not recoverable from the outputs you would look at. |
| The **ω sign / rotation direction** is not confirmed against the raw encoder | A flipped ω mirrors the whole map and reflects every orientation; nothing downstream looks wrong. Established only in phase 1, un-checkable after. |
| **`positions.csv` order** (file-order vs sorted; sign) is not confirmed | The voxel grid is `translations × translations`; the wrong order/sign mirrors the map about a diagonal. Looks like a plausible microstructure. |
| Indexing is about to run **unseeded** on a scanning dataset | Blind scanning indexing over tens of millions of merged spots is intractable (days, often no output). It needs an FF orientation seed. This is a resource cliff, not a bug. |
| The pf-odf strain patches are **not dark-subtracted** and the frames are raw | The fit has no additive-background term; a flat pedestal biases magnitude ~30 % and reshapes the field. Silently plausible. |
| The **strain reference cell** (`LatticeConstant`) has not been pinned from this sample's own rings | The refiner's strain fit measures against it inside a **±10000 µε** box, so a reference wrong by ~0.7 % rails components silently — and it also costs completeness. Un-noticeable after the fact: the map looks fine. See §"the reference-cell trap". |
| A **per-voxel result, grain count, or acceptance threshold** is about to be quoted without a measured **chance ceiling** | On a dense layer completeness **saturates** — a quarter of voxels sit at exactly 1.0000 — so it cannot separate a grain from a coincidence, and nothing internal to the run exposes that. The shipped `MinMatchesToAcceptFrac 0.5` sat at or below the measured ceiling on **all five** layers tested, admitting up to one null voxel per 1.5 real ones. The ceiling is **not predictable from spot density and must be measured per layer**. Phase 7. |
| A **merged-FF grain count** from scanning data is about to be quoted | A 1-row `positions.csv` sets `nScans_ == 1`, so `doScanFilter` is 0 and the beam gate is **off in the matching loop**. Measured: the ω-shuffle null *beat* the real arm. Phase 7 §7.6. |

When you halt, say which row fired, what you measured, and what you would need to proceed.
Finish everything not blocked by it first.

### Hard rules

1. **Suspect success.** A finished PF run with a full-looking `voxel_grid.csv` proves the
   *pipeline ran*, not that it is right. A mirrored map, a wrong-Friedel seed, and a
   zero-strain "solved" grain all finish clean. Judge by the raw frames and the
   orientation/KAM structure, never by "it completed."
2. **Debug your own config before the physics.** Before blaming the indexer, the refiner,
   or the sample, confirm the ω sign, the `positions.csv` convention, the ring numbers, and
   that the parameters actually reaching the code are the ones you think (see the zip trap).
   Most "the reconstruction is wrong" turns out to be one of these.
3. **The c-omp refiner's seed is a PF-specific file.** In scanning mode the C refiner reads
   a **5-column** `SpotsToIndex.csv` (`voxNr SpId nSpotsBest _ bestSolIdx`) to pick each
   voxel's seed out of `IndexBest_all.bin`. The python indexer never emits it; without it
   the refiner silently refines nothing. `midas_pipeline` now synthesises it — but if you
   drive the refiner by hand, you must too (`midas_fit_grain.scan_seed`).
4. **pf-odf expects dark-subtracted patches.** The peak-shape loss is a purely
   multiplicative per-spot scale with no background term. Feed raw frames only with
   `subtract_background=True`; feed already-dark-subtracted caches only with it **off**.
5. **`Hbeam` / `Rsample` are generous SEARCH BOUNDS — never the physical beam or
   sample size.** *(Corrected 2026-08-21; earlier drafts of this file said the
   opposite and sent a session off chasing a non-bug.)* They bound where the
   indexer/refiner may place a grain. Tighten them to the true dimensions and
   solutions **plop onto the bounding box**, giving an artefactual pile-up of
   positions at ±`Rsample` and ±`Hbeam`/2 that looks like real microstructure.
   Template-ish values (800 / 1800 / 2000 µm) are correct as-is — leave them
   alone. In PF this is doubly moot: **PF does not fit position at all**, it
   fixes each voxel to the scan grid. The true beam height matters only when
   *stitching* layers. (Shared with FF/NF.)

6. **Pin the strain reference cell from the sample's own rings.** `LatticeConstant`
   is not just a starting guess — it is the **zero of the strain measurement**
   (hard rule 7 below). Fit it from observed ring positions before quoting any
   strain.

7. **Never recover the reference cell from refined per-grain cells.** That is a
   feedback loop: the refinement starts from `LatticeConstant` and only partly
   leaves it, so averaging the refined cells returns roughly what you fed in.
   Measured: iterating drifted a further −3740 µε in `a` and +6361 µε in `c`
   without converging. Use the powder route (`refine_lattice_from_d_spacings`),
   which takes **no starting cell**.

8. **Profile PER STAGE. Never generalise one stage's behaviour to "the
   pipeline."** A PF layer is two workloads with opposite bottlenecks: **prep**
   (`zip_convert→peakfit→transforms→binning`) is **disk**-bound and uses ~4 of 64
   cores however many you give it, while **index**
   (`indexing→refinement→find_grains`) is **CPU**-bound and saturates 63.5 of 64.
   Measuring peakfit, concluding "the pipeline underuses the box", and running 3
   layers at 20 cores each was measured **worse than serial** — it starved the
   only stage that could use the cores. Overlap the two halves instead of
   splitting the cores. Multi-layer campaigns: **RUNBOOK §R4**.

9. **Report the majority-class null with any map comparison.** Whenever a
   grain-ID map is scored against another labelling — tomographic vs
   point-by-point, filtered vs unfiltered, one layer vs its neighbour — quote
   the score you would get by calling **every voxel the most common grain**.
   Measured: a tomographic map agreed with the point map on **60.1 %** of
   voxels, against a constant-map null of **65.2 %** (κ = 0.399). Agreement
   below its own null is not agreement, and a bare percentage hides it
   completely. (Phase 6 §6.8.)

10. **A diagnostic flag is not a filter.** Grains flagged as filling the scanned
    field (occupancy > 0.65) have untrustworthy *shapes* and are still real
    grains. Deleting them took map agreement from **47.8 % to 11.0 %**, because
    the largest grain is most of the material and its voxels then go to whichever
    small grain wins by default. The code leaves them in the competition on
    purpose. (Phase 6 §6.6.)

11. **Measure the chance ceiling before setting or trusting any acceptance
    threshold.** Run the ω-shuffle null (phase 7) and take the highest
    best-completeness any *null* voxel reaches. Below that value, real and chance
    overlap. Measured across five layers as **none / 0.5333 / 0.6957 / 0.7500 /
    0.8333** — a sparse enough layer has no chance floor at all, and the order is
    NOT the order of spot density (the densest layer came in mid-table, against
    an explicit prediction). Formerly quoted as 0.6957 on a 936 k-spot layer and 0.8333 on a 1.29 M-spot
    layer — it is NOT ordered by spot density (the densest of five came in
    mid-table, against an explicit prediction), so one layer's ceiling must never
    be quoted for another. Where
    completeness has saturated, `CalcAvgIA` (separated 2.6×) and the threshold-free
    distinct-winners-per-voxel ratio (§7.5) still discriminate.

12. **Pass `ScanPosTol` explicitly whenever you drive the indexer or refiner by
    hand.** The C adds **0.1 µm to the parsed `BeamSize`** before the
    `BeamSize/2` fallback, so a hand-run without `ScanPosTol` silently uses a
    wider beam gate than the parameter file states — 0.80 µm instead of 0.75 at
    `BeamSize 1.5`, worth **+14.7 % accepted solutions** and a changed winner in
    10.5 % of voxels. The pipeline computes it in Python and writes it out; a hand
    invocation does not. (DIAGNOSIS: "Solution counts differ between a pipeline run
    and a hand-run".)

### Traps that silently corrupt results

| Trap | Symptom if missed | Where |
|---|---|---|
| `MaxNPeaks` (and other analysis params) are **baked into the zarr zip** at zip-convert time; peakfit reads the zip's stored params, not your live `paramstest` | Peaks silently capped; changing `paramstest` does nothing until the zips are regenerated | phase-2 |
| c-omp PF refine against a 1-column or mis-counted `SpotsToIndex.csv` | Refiner exits 0, `Results/` empty or garbage | phase-3, DIAGNOSIS |
| Raw (non-dark-subtracted) patches into pf-odf | Strain converges, magnitude inflated, field pattern wrong | phase-4 |
| `find_grains` on a high-spread / cracked map | Per-voxel clustering pass runs for hours, never completes | phase-3, DIAGNOSIS |
| Detector h5 chunked one-frame-per-chunk | pf-odf patch extraction reads whole frames → multi-TB, multi-hour I/O | phase-4 |
| **`--num-files-per-scan` defaults to 1** for a per-frame TIFF series. `NrFilesPerSweep` in the parameter file is stored in the zarr but does **not** drive the file list | `exchange/data` is `(1, ny, nz)`, all 23 stages run, layer finishes **0 voxels, exit 0** | phase-3 |
| **`--scan-work-dir` given a relative path** | joined twice (`…/scans/167051/scans/167051/…`), every zip fails. Omitted entirely, per-scan intermediates are written *into the raw data tree* | phase-3 |
| Stale layer outputs from an earlier attempt (`InputAllExtraInfoFittingAll*.csv`, `midas_state.h5`) | `transforms` reports "13 ok" in 0.04 s from cache and indexing runs on the OLD spots — 0 voxels, exit 0 | phase-3 |
| **Reference cell (`LatticeConstant`) not pinned to the sample** | per-voxel strain rails at the hardcoded ±10000 µε and completeness is depressed | phase-2 §2.5 |
| **`do_tomo` / `-doTomo 1` left on** for a point-by-point map | the re-index runs from a tomography-seeded map and gets **worse**: 2433/2601 voxels refined instead of all 2601, and 367 below completeness 0.5 instead of 11 | phase-6 §6.3 |
| **`midas_pipeline` older than 0.11.0** and a shape is compared to the voxel map | the FBP crop was off by one voxel in both axes for every odd `n_scans` — a *constant* offset, so nothing looks broken | phase-6 §6.2 |
| **Completeness read as a material map** | vacuum inherits orientations and scores ~0.92; the floor was 0.445 with nothing below 0.40, which reads as "material everywhere" | phase-1b §1b.1 |
| The tomographic **max-grain-intensity map used as a material map** | it answers "did a *listed grain* reconstruct here", not "is there material here" — material with no listed grain reads as dark. Produced a wrong edge placement on the reference campaign | phase-1b §1b.5 |
| Legacy-C `spotPositions_*.bin` read as if populated | **97.7 % of it is the `-1` initialiser**. The Python path writes `spotPos_*.bin` and is correct — reading by the old name silently finds nothing | phase-6 §6.9 |
| **`ScanPosTol` omitted on a hand-run**; the C's `BeamSize/2` fallback uses `BeamSize + 0.1` | beam gate 6.7 % wider than the parameter file says → **+14.7 % solutions**, winner changed in 10.5 % of voxels. Nothing errors | DIAGNOSIS |
| **`paramstest_comp.txt` is written by two stages under one name** — indexing (`_emit_c_omp_paramstest`, carries `ScanPosTol`) then refinement (`comp_backend_paramstest`, does not), which overwrites it | the file on disk is **not** what the indexer read; reconstructing a run from it reproduces the wrong gate | DIAGNOSIS |
| **`RingNumber == 0` placeholder rows counted as spots** — failed transforms are written as all-zero rows, not dropped | 20.1 % of rows on the reference campaign; counting them manufactures a fake "20 % collapsed on merge" against a real 0.09 % | DIAGNOSIS |
| **`argv[4]` (`nWork`) is ignored in PF mode** — `nVoxels = numScans²` from `positions.csv` (`IndexerUnified.c:3200` prints "argv ignored for PF") | a voxel-limited test run silently processes the whole layer; use `blockNr`/`nBlocks` to slice | phase-7 §7.8 |

## 0. Verify the install

The version floors exist to keep out versions that produce plausible wrong answers rather
than errors. Paste the output:

```bash
$PY -c "import midas_pipeline, midas_fit_grain, midas_index; \
        print('pipeline', midas_pipeline.__version__); \
        print('fit_grain', midas_fit_grain.__version__); \
        from midas_fit_grain import backend_c; \
        print('c-omp refiner:', backend_c.available(), backend_c.binary_path())"
# strain path only:
$PY -c "import midas_pf_odf; from midas_fit_grain import scan_seed, fitbest_adapter; \
        print('pf-odf + c-omp->pf-odf bridge present')"
```

The bridge (`midas_fit_grain.scan_seed` + `fitbest_adapter`) is required for a c-omp PF
refine to feed pf-odf. If `backend_c.available()` is `False`, **rebuild `midas-fit-grain`
with an OpenMP toolchain** — that is the only fix. There is no Python fallback any more:
`--refine-backend` accepts `c-omp` and nothing else (`midas_pipeline/config.py:666`), and
the Python refiner it used to name is known-broken.

## 0a. THE ORDER

| # | Step | Where | Why it is here and not later |
|---|---|---|---|
| 0 | Verify the install | §0 | version floors gate silent wrong answers |
| 0b | Identify the station and the code generation | INSTRUMENT | the ω sign, the dark group and the entry point all differ; the legacy C path lacks two diagnostics and has one 97.7 %-unwritten output |
| 1 | Survey the folder | phase-0 | confirms it is scanning 3DXRD before any recipe applies |
| 2 | ω sign + `positions.csv` convention | phase-1 | un-checkable after the fact; a wrong value mirrors the map |
| 3 | Calibrate geometry | phase-1 | every position/strain number is conditional on it |
| 4 | Configure params + regen zips | phase-2 | `MaxNPeaks` etc. are baked into the zips, not live |
| 5 | Obtain the FF orientation seed | phase-2 | unseeded scanning indexing is intractable |
| 6 | Run the PF pipeline | phase-3 | zip→peakfit→transforms→merge_scans→seeding→binning→indexing→refinement→find_grains |
| 6b | Locate the sample boundary | phase-1b | needs only the peak search, so it is available here — and every per-voxel statistic below is averaged over vacuum until it is done |
| 7 | (strain) per-voxel peak-shape fit | phase-4 | needs the grain map + dark-subtracted patches |
| 8 | Read + report | phase-5 | orientation/KAM/GROD/strain, with provenance |
| 9 | (shapes / positions) reconstruction space | phase-6 | the concentration filter and occupancy flag improve the *point* result; shapes are gated |
| 10 | **Run the ω-shuffle null; measure the chance ceiling** | phase-7 | completeness saturates on dense layers, so nothing internal to the run can tell a grain from a coincidence. Cheap: a re-index, no re-prep |

## Sibling doc sets

`manuals/ff-hedm/` (far-field, skill `ff-hedm`), `manuals/nf-hedm/` (near-field, skill
`nf-hedm`), `manuals/dfxm/` (dark-field microscopy, skill `dfxm`), and in the LaueMatching
repository the `laue` skill. pf-HEDM is the only one that carries a **per-voxel peak-shape
strain** phase; the far-field envelope explains why grains-not-voxels changes the strain
question.
