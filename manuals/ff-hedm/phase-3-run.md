# Phase 3 — Run the pipeline

> Part of the **FF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

---

## 7. STEP 6 — Run the pipeline

```bash
midas-pipeline run --scan-mode ff \
    --params Parameters.txt \
    --result results/ \
    --layers 1-1 \
    --indexer-backend c-omp \
    --refine-backend c-omp
```

### Both backends are c-omp — now the only choice the code accepts

**This is no longer something you have to remember.** `--indexer-backend` and
`--refine-backend` each take **`c-omp` and nothing else**: argparse restricts the
choice list, and `_require_comp_backends` (`midas_pipeline/config.py:666`), called
from `PipelineConfig.__post_init__`, re-checks it — so a notebook or library caller
that builds a `PipelineConfig` directly cannot bypass it either. Both flags default
to `c-omp`, so the two lines above are optional — keep them if you like the run
command to be explicit.

The c-omp binaries (`midas_index/bin/midas_indexer`,
`midas_fit_grain/bin/midas_fitgrain`) have no GPU path at all, so CUDA
contention, OOM and driver mismatch cannot occur in either stage.

**What this closed.** The refiner used to default to **python + torch + CUDA**
while the indexer defaulted to `c-omp`, so a run whose log said
`indexing(FF, c-omp)` was silently half on the GPU path. Measured on a 20-ID
alumina FF layer (24 900 seeds, 471 k spots): refinement died after ~12 s with a
bare `subprocess.CalledProcessError … non-zero exit status 1` and **no child
traceback** — `run_checked_streamed` swallows the subprocess's stderr. It was
first misdiagnosed as GPU contention and "fixed" with `CUDA_VISIBLE_DEVICES=0`;
it failed again identically. Because `transforms` re-runs on resume, **each retry
cost a ~90-minute re-index** — two of them. The Python refiner is
known-broken independently of that; it is not a fallback.

**Refiner tuning flags are gone with it.** `--pf-refine-mode`, `--refine-solver`,
`--refine-loss`, `--refine-mode`, `--use-bounds` and `--bound-*` were **removed**
— every one configured the in-process PyTorch refiner. The c-omp refiner has no
configurable solver or loss, so those flags advertised tuning that does not
exist. An old script carrying them now fails at argument parsing, which is the
intended outcome: it was not doing what it claimed before.

**Still worth reading the log:** both stage lines should say `c-omp`. That is now
a check that the *binaries* were found and ran, not that the flags were right.

### The c-omp refiner has no `tx`, and that is correct — `tx` is applied in `transforms`

Grep either c-omp binary's source for `tx` and you find nothing that acts on it.
This has been read as "the c-omp backends ignore `tx`, so a `tx` in
`Parameters.txt` can never reach the result, therefore do not use c-omp."
**That conclusion is wrong, and it is the wrong reason to abandon the fast
path.** `tx` is consumed *upstream*. By the time either backend runs, the spots
already carry the correction:

| stage | what it does with `tx` |
|---|---|
| `zip_convert` | carries `tx` from `Parameters.txt` into the zarr — it is in the zipper's float key set (`midas_zipper/ff_zip.py:204`) |
| `transforms` | **applies it.** `apply_tilt_distortion` (`midas_transforms/fit_setup/core.py:376`) sends every raw pixel through `pixel_to_REta_torch` (`midas_transforms/fit_setup/transform.py:82`) and writes *corrected* lab-frame µm into `InputAll.csv` → `Spots.bin` / `ExtraInfo.bin` |
| `indexing` (c-omp) | reads those corrected spots. It parses a `DetTx` out of `DetParams` and then never reads it back (`midas_index/c_src/IndexerUnified.c:2920`) — a dead store, not a dropped correction |
| `refinement` (c-omp) | reads the same corrected spots — it mmaps `AllSpots` straight out of the transforms output (`midas_fit_grain/c_src/FitUnified.c:1348`). No `tx` in the geometry model, by design |
| `refinement` (python) | parses `tx` into its config and deliberately does **not** apply it — `apply_tilts` stays False because "the refined tilts live in the *observed* positions already" (`midas_fit_grain/driver.py:249`) |

The two refiner backends therefore agree: **observed spots are in the ideal
detector frame and the forward model predicts in the ideal frame.** Re-applying
`tx` inside the refiner would be a *double* correction, not a fix. A `tx` change
does reach the result — through `transforms`, which is why `transforms` and
everything downstream of it must re-run when you change it (§7 resume).

Two further traps in this diagnosis:

- **`FF_HEDM/src/FitPosOrStrainsOMP.c` is not the binary `--refine-backend c-omp`
  runs.** The shipped c-omp refiner is
  `packages/midas_fit_grain/c_src/FitUnified.c`; `FF_HEDM/` is soft-deprecated C
  (spine, "Maintained code"). Reading the deprecated tree to characterise a
  shipped backend gives a true statement about the wrong file.
- **Fitting `tx` is a separate tool, not a refiner setting.**
  `midas_joint_ff_calibrate.grain_refine` is the only thing that fits it; it
  works on **raw** `SpotMatrix` pixels and rotates them by a trial `tx`
  (`midas_joint_ff_calibrate/grain_refine.py:426`). Its output is the *residual*
  roll relative to whatever `tx` the reconstruction already ran with, so it must
  be **composed** (`tx_total = tx_applied + tx_reported`) and **iterated** to
  convergence — see [`ENVELOPE.md`](ENVELOPE.md) §5.

13 stages, each with a provenance entry in `<result>/LayerNr_N/midas_state.h5`:

```
zip_convert → hkl → peakfit → merge_overlaps → calc_radius → transforms
→ cross_det_merge → global_powder → binning → indexing → refinement
→ process_grains → consolidation
```

`zip_convert` is skipped when the zarr already exists; `cross_det_merge` and
`global_powder` are no-ops for single-detector runs; `consolidation` is gated by
`--generate-h5`. Auto-resolved knobs (`--dtype`, `--shard-gpus`, `--group-size`) are logged
at startup; explicit values always win.

> `midas-ff-pipeline` is **deprecated** as of 0.4.0 — use `midas-pipeline run --scan-mode ff`.
> Same orchestrator underneath.

### `--pg-mode`: `c_parity` is the default, and `spot_aware` is DISABLED

`--pg-mode` takes `legacy`, `paper_claim` or **`c_parity`** (the default).
**`spot_aware` has been removed from the choice list and is rejected in four
independent places** — `PipelineConfig.__post_init__` (`midas_pipeline/config.py:687`),
the `midas-process-grains` CLI, and the package's own dispatcher and pipeline
(`midas_process_grains/modes.py:69`, `pipeline.py:235`). Calling the library directly
does not get you around it; an old script or config that asks for it fails with the
reason rather than running.

**Why it was disabled — adjudicated against EBSD, not against taste.**
On `shade_LSHR` layer 1, one refiner output, `MinNrSpots 3` + `Completeness 0.7`,
one-to-one matched at 1° / 15 µm against 4328 segmented EBSD grains:

| mode | grains | precision | recall |
|---|---|---|---|
| C `ProcessGrains` | 3491 | 79.8 % | 64.3 % |
| **`c_parity`** | **3492** | **79.8 %** | **64.4 %** |
| `spot_aware` | 4128 | 68.2 % | 65.0 % |

Of the **691 grains `spot_aware` adds** over `c_parity`, only **7.2 % have an EBSD
partner** against **80.4 %** for the shared population, and their `DiffPos` median is
**387 µm against 121**. It buys **+0.1 pp of recall for −11.6 pp of precision**.

Confirmed independently on a 20-ID alumina rod (1 mm diameter, 100 µm beam):
`spot_aware` returned **1652 grains against `c_parity`'s 533**, placed **4.1 % of
them outside the physical sample** (out to r = 1290 µm in a 500 µm-radius rod, vs
0.6 %), and spread `|Z|` to a **p90 of 286 µm through a 50 µm beam half-height**
(vs 57). Grains outside the rod and above the beam are not a tuning preference.

`c_parity` reproduces the C reference: on datasetA Ni it returns **6150 grains vs
C's 6138**, and matched pairs agree to **0.0000°** and **0.000 µm**.

> **Open, not closed.** *Why* the `spot_aware` branch manufactures those grains is
> **not yet diagnosed** — it is disabled on its output, not on a root cause. Do not
> treat "spot_aware is wrong" as a finished explanation, and do not re-enable it on
> the strength of a single dataset looking better. See Lab Notebook §2e.

**`c_parity` writes the residual sidecar** as of `midas-process-grains` 0.9.2.
`processgrains_diagnostics.h5` — which carries `residuals/spot_table`, the
per-observation residuals every downstream diagnostic needs — is produced by a default
run, and `utils/midas_ff_report.py` renders its full figure set from it. It costs no
extra FitBest I/O (the rows are already in RAM for the strain solve); skip it with
`--no-diagnostics-h5`.

> **Before 0.9.2 it was not.** That branch returned without calling `result.write()`,
> so a default run produced no sidecar, with or without `--generate-h5` (the FF
> `process_grains` stage does not read that flag; FF `consolidation` is a no-op stub).
> Nothing errored and `Grains.csv` was correct, so **an older run's missing sidecar is
> not evidence of anything wrong with that run** — check the version before concluding.
> `mode=physics` still has no residuals: `v4_pipeline` never reads FitBest, so there is
> no obs-vs-predicted table to decompose.

### The residual columns describe two different geometries — use the new pairs

**`Grains.csv` cols 19-21 are a MIXTURE, and nothing in the file says so.** Col 19
(`DiffPos`) is evaluated at the **refined** parameters; cols 20-21 (`DiffOme`,
`DiffAngle`) at the **indexer seed**, before any fitting. So no two of the three
describe the same geometry, and `DiffPos` is *not* the mean of the per-spot
`DiffLen` in `FitBest.bin` — measured ratio **1.711** over all 55,593 seeds of a
from-scratch Ni layer, 0 exceptions. ESTABLISHED: the mechanism is that
`SpotsComp` is filled only by `CalcAngleErrors` at `Ini`
(`FitUnified.c:1804`/`:1828`), the post-fit re-match being env-gated and off, while
`ErrorFin[0]` is `FitErrors12D(FinalResult)/nSpotsComp`. Convention-free
confirmation: the theoretical ring radii in `FitBest.bin` vary by **3e-16** across
grains whose refined `a` spans 4.3e-3, and match `hkls.csv` (built at the seed
lattice) to <5e-07 µm.

**From `midas-fit-grain` 0.9.0 you do not have to live with that.** `Grains.csv`
gains cols **47-52** — `DiffPos/Ome/Angle` x `Pre/Post`, both from the *same*
estimator, so `post - pre` is a real improvement rather than partly an estimator
change. Cols 19-21 are left exactly as they were, so nothing silently changes
value. On the reference layer: **596.77 -> 351.50 µm**, omega 0.19750 -> 0.15806,
internal angle 0.22070 -> 0.18151, improving on **1951/1951** grains. Cols 47-52
are **NaN** on a run whose `OrientPosFit.bin` is the legacy 27-column form — NaN
rather than 0.0, so a missing value cannot be read as a measured one.

**New artifacts from the same release:**

| artifact | what it is |
|---|---|
| `Output/FitBestFinal.bin` | the **post-fit** twin of `FitBest.bin` — same layout, stride and short-final-slot behaviour, matched at the refined parameters. `FitBest.bin` stays pre-fit, so every existing consumer is bit-unchanged (verified: sha256 identical over 1.76 GB). The two are **not row-aligned** — join on SpotID, never by row index |
| `Results/OrientPosFit.bin` | **33** doubles per seed, not 27 (27-29 pre, 30-32 post). Readers sniff the width; a legacy file still reads |
| `SpotMatrix.csv` | **28** columns, and now carries one row per **un-found expected spot** — see phase 4 |
| `Results/SpotDiagnostics.bin` | version **2**. Written by default in FF too, despite an in-source comment saying PF-only |

**Two things to check in the log every time:**

1. `nFrames` in the peakfit banner = logged frames − `SkipFrame` (§3e).
2. **Stage resume is silent.** `peakfit(FF): …AllPeaks_PS.bin already exists; skip.`
   means the peak search did **not** run and results were inherited from a previous
   invocation — which may have used a different threshold, a different dark, or a broken
   config. It costs 0.3 s instead of 55 s, so it is easy to miss. **After changing any
   peak-search or dark parameter, delete `results/` entirely**, do not rely on resume.

   **Geometry used to bite hardest here, and `tx` most of all.** `zip_convert(FF)` reuses
   any existing `*.MIDAS.zip`, and `transforms` reads the geometry out of the **zarr**,
   not out of `Parameters.txt` (`midas_transforms/params.py:751`). So editing `tx`, `Lsd`,
   `BC` or a distortion coefficient and re-running into the same result folder silently
   kept the *old* value while the run reported success — the observation "changing `tx`
   does nothing", which is a **stale zarr** and not a backend that ignores `tx`.

   **`zip_convert` now re-reads the parameter file into an archive it reuses**
   (`midas_pipeline/stages/_param_refresh.py`, rewriting via
   `midas_zipper.param_refresh`). Check the log line — it names every key it changed,
   and the same list lands in `midas_state.h5` under `n_params_refreshed` /
   `params_refreshed`, so "which parameters did this run actually use" is answerable
   afterwards instead of from memory. Three things it deliberately will not do:

   | situation | what happens |
   |---|---|
   | a key the **stored frames** depend on changed (`SkipFrame`, `Padding`, `FileStem`, pixel layout) | **hard error.** Those were consumed when the frames were written, so patching the number would leave the data and the metadata describing different things — a silently shifted ω in `SkipFrame`'s case. Delete the zip and let it rebuild |
   | a changed key invalidates a stage output **that already exists** (`RingThresh` vs `Temp/AllPeaks_PS.bin`) | **hard error**, naming the key and the stage. Refreshing the zarr alone would move the staleness one stage downstream, not fix it. Delete those outputs, or pass `--force-param-refresh` and accept they are stale |
   | the rewrite did not actually take | **hard error.** `zip -u` replaces an entry only when the staged file is *newer* and otherwise exits 12 having changed nothing, so every refresh is verified by re-reading the archive |

   `--no-refresh-params` turns the whole thing off and restores the old
   inherit-whatever-the-zarr-has behaviour. To fix an archive outside a pipeline run:
   `midas-refresh-zarr-params --zip <f>.MIDAS.zip --params Parameters.txt [--dry-run]`.

Subprocess stages (`peakfit_torch`, `midas_indexer`, `midas_fit_grain`) are invoked by
bare name, so the env's `bin` must be on `PATH` — calling `midas-pipeline` by full path is
not enough and fails with `FileNotFoundError: 'peakfit_torch'`:

```bash
export PATH=/home/beams12/S1IDUSER/opt/envs/midas/bin:$PATH
```

For ≥ 5 k seeds also export `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Check the zipper's frame accounting in the log.** It prints
`HDF5 scan: N file(s), F frames/file. Skipping the first S frame(s) of every file. Total
frames to write: T`. Confirm `T` equals the frame count you derived from the par file in
§3b. If `T` is one larger, `SkipFrame` did not take effect (§3e version trap).

---


## 12. Check reproducibility on a new install

Run the identical pipeline twice into a **clean** result dir and checksum every stage in
pipeline order — not just `Grains.csv`. You want the *first* artifact that diverges,
because that is what names the guilty stage.

`rm -rf` the result dir between runs. The stages resume silently off existing files (§7),
so a "reproducible" result can just be a skipped stage.

```bash
#!/bin/bash
# usage: ff_repro.sh <paramfile> <scratch-dir>
# Runs the pipeline twice into separate trees and reports the FIRST divergence.
set -u
PARAMS="$1"; BASE="$2"
STAGES="Temp/AllPeaks_PS.bin Temp/AllPeaks_PX.bin InputAll.csv
        InputAllExtraInfoFittingAll.csv Spots.bin Data.bin nData.bin ExtraInfo.bin
        SpotsToIndex.csv Output/IndexBest_all.bin Output/FitBest.bin
        Results/OrientPosFit.bin Grains.csv SpotMatrix.csv"

for run in A B; do
  rm -rf "$BASE/$run"; mkdir -p "$BASE/$run"
  midas-pipeline run --scan-mode ff --params "$PARAMS" \
      --result "$BASE/$run" --layers 1-1 > "$BASE/$run.log" 2>&1
done

# md5sum is GNU; on macOS use `md5 -q`. Pick whichever exists.
MD5=$(command -v md5sum >/dev/null && echo "md5sum" || echo "md5 -q")
diverged=0
for f in $STAGES; do
  a="$BASE/A/LayerNr_1/$f"; b="$BASE/B/LayerNr_1/$f"
  [ -e "$a" ] && [ -e "$b" ] || { printf '%-40s MISSING\n' "$f"; continue; }
  ha=$($MD5 < "$a" | cut -d' ' -f1); hb=$($MD5 < "$b" | cut -d' ' -f1)
  if [ "$ha" = "$hb" ]; then printf '%-40s ok\n' "$f"
  else printf '%-40s *** DIVERGED ***\n' "$f"
       [ $diverged -eq 0 ] && echo ">>> FIRST DIVERGENCE: $f  <- this stage is guilty"
       diverged=1
  fi
done
[ $diverged -eq 0 ] && echo "bit-identical across both runs"
```

Glob-named artifacts (`Result_StartNr_*.csv`, `Radius_StartNr_*.csv`) are omitted because
their names carry the start number; add them explicitly once you know it.

With an install that passes §0 this is bit-identical across runs. If it is not, read Lab
Notebook §2 — two separate nondeterminism bugs are documented there with the signature
each produces. They were found exactly this way, and each masked the other until the first
was fixed. Expect to iterate.

---
