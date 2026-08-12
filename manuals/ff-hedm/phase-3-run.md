# Phase 3 — Run the pipeline

> Part of the **FF-HEDM doc set**. The spine — scope gate, install gate, hard rules,
> halt conditions and the order of operations — is [`README.md`](README.md). Section
> numbers (§n) are continuous across the set; the index in the spine says which file
> holds which.

---

## 7. STEP 6 — Run the pipeline

```bash
midas-ff-pipeline run \
    --params Parameters.txt \
    --result results/ \
    --layers 1-1 \
    --device cuda
```

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

### The default `--pg-mode` writes no residual sidecar

`--pg-mode` defaults to **`c_parity`**, and that branch returns without calling
`result.write()`. So `processgrains_diagnostics.h5` — which carries `residuals/spot_table`,
the per-observation residuals every downstream diagnostic needs — **is never produced by a
default run**, with or without `--generate-h5` (the FF `process_grains` stage does not read
that flag; FF `consolidation` is an unconditional no-op stub).

Nothing errors. The run completes, `Grains.csv` is correct, and the sidecar is simply
absent, which reads as "this pipeline version doesn't write one" rather than as a mode
choice.

If you want the sidecar, ask for a mode that writes it:

```bash
midas-pipeline run --scan-mode ff --params Parameters.txt --result results/ \
    --layers 1-1 --pg-mode spot_aware
```

`spot_aware`, `legacy` and `paper_claim` all write it; `c_parity` does not. Note the modes
are not interchangeable scientifically — on `Au3_cubes_ff_000008`, `c_parity` returned 5
grains and `spot_aware` returned 2 (the documented parent + Σ3 twin, matching Lab Notebook
§3a's C-cross-checked radii to five decimals). Read `--pg-mode --help` for the
accuracy trade-off before choosing on convenience.

**Two things to check in the log every time:**

1. `nFrames` in the peakfit banner = logged frames − `SkipFrame` (§3e).
2. **Stage resume is silent.** `peakfit(FF): …AllPeaks_PS.bin already exists; skip.`
   means the peak search did **not** run and results were inherited from a previous
   invocation — which may have used a different threshold, a different dark, or a broken
   config. It costs 0.3 s instead of 55 s, so it is easy to miss. **After changing any
   peak-search or dark parameter, delete `results/` entirely**, do not rely on resume.

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
