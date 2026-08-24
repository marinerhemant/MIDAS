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

## The doc set — what to read when

| File | Holds | Read it |
|---|---|---|
| **`README.md`** (this) | scope gate, install gate, the order, hard rules, halt conditions | always |
| `phase-0-survey.md` | is it scanning 3DXRD? translations, ω, per-scan files, positions | first |
| `phase-1-geometry.md` | calibration, the ω sign, the `positions.csv` convention | before configuring |
| `phase-2-configure.md` | params, the zip-baked param trap, ring selection, the FF seed | before running |
| `phase-3-run.md` | the MIDAS PF pipeline stage-by-stage | the reconstruction |
| `phase-4-strain.md` | `midas_pf_odf` peak-shape per-voxel strain (the novel extension) | if strain is the goal |
| `phase-5-read-report.md` | read `voxel_grid.csv` + Results, orientation/KAM/GROD/strain maps, report | at the end |
| `PF_PARAMETERS.md` | the PF-specific parameter reference | when tuning |
| `DIAGNOSIS.md` | symptom → test → cause → lever | **when something looks wrong** |
| `RUNBOOK.md` | where it runs, what healthy looks like, pick-up point, **and §R4 multi-layer campaigns** | on resume; **§R4 before driving more than one layer** |
| `LAB_NOTEBOOK.md` | evidence, ledger, **retracted claims** | before re-investigating |

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
| The **ω sign / rotation direction** is not confirmed against the raw encoder | A flipped ω mirrors the whole map and reflects every orientation; nothing downstream looks wrong. Established only in phase 1, un-checkable after. |
| **`positions.csv` order** (file-order vs sorted; sign) is not confirmed | The voxel grid is `translations × translations`; the wrong order/sign mirrors the map about a diagonal. Looks like a plausible microstructure. |
| Indexing is about to run **unseeded** on a scanning dataset | Blind scanning indexing over tens of millions of merged spots is intractable (days, often no output). It needs an FF orientation seed. This is a resource cliff, not a bug. |
| The pf-odf strain patches are **not dark-subtracted** and the frames are raw | The fit has no additive-background term; a flat pedestal biases magnitude ~30 % and reshapes the field. Silently plausible. |
| The **strain reference cell** (`LatticeConstant`) has not been pinned from this sample's own rings | The refiner's strain fit measures against it inside a **±10000 µε** box, so a reference wrong by ~0.7 % rails components silently — and it also costs completeness. Un-noticeable after the fact: the map looks fine. See §"the reference-cell trap". |

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
| 1 | Survey the folder | phase-0 | confirms it is scanning 3DXRD before any recipe applies |
| 2 | ω sign + `positions.csv` convention | phase-1 | un-checkable after the fact; a wrong value mirrors the map |
| 3 | Calibrate geometry | phase-1 | every position/strain number is conditional on it |
| 4 | Configure params + regen zips | phase-2 | `MaxNPeaks` etc. are baked into the zips, not live |
| 5 | Obtain the FF orientation seed | phase-2 | unseeded scanning indexing is intractable |
| 6 | Run the PF pipeline | phase-3 | zip→peakfit→transforms→merge_scans→seeding→binning→indexing→refinement→find_grains |
| 7 | (strain) per-voxel peak-shape fit | phase-4 | needs the grain map + dark-subtracted patches |
| 8 | Read + report | phase-5 | orientation/KAM/GROD/strain, with provenance |

## Sibling doc sets

`manuals/ff-hedm/` (far-field, skill `ff-hedm`), `manuals/nf-hedm/` (near-field, skill
`nf-hedm`), `manuals/dfxm/` (dark-field microscopy, skill `dfxm`), and in the LaueMatching
repository the `laue` skill. pf-HEDM is the only one that carries a **per-voxel peak-shape
strain** phase; the far-field envelope explains why grains-not-voxels changes the strain
question.
