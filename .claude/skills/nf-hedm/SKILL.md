---
name: nf-hedm
description: >-
  Take a near-field HEDM (NF-HEDM) dataset from raw frames to a grain map: find
  the beamtime metadata, establish the omega sign, measure the beam centre from a
  DetZBeamPos scan, refine the geometry on a calibrant, reduce the images, fit
  orientations, and read the .mic. Use when asked to reconstruct, calibrate or
  diagnose an NF-HEDM / near-field 3DXRD beamtime, when handed a folder of NF
  TIFFs, a 20-ID-D HT-HEDM HDF5 scan, or a DetZBeamPos scan, or when an NF
  reconstruction has low confidence. Covers 1-ID (TIFF-per-frame) and 20-ID-D
  HT-HEDM (Bluesky/HDF5); any other beamline is gated.
---

# NF-HEDM reconstruction

**This skill is a pointer, not the procedure.** The procedure is a doc set in the
repository so it lives beside the code it cites, gets checked by the repo's own
pre-commit hooks, and stays usable without this skill.

## Start here

Read **`manuals/nf-hedm/README.md`** — the spine. It is the only file meant to stay
loaded: scope gate, install gate, the order of operations (confirmed with the instrument
scientist), the hard rules, and the halt conditions. It carries an index saying which file
holds which section; open those as you reach them.

## Nine things to know before you start

1. **Run the floor gate first** (spine §1). `SumFrames` **inverted** its unit convention:
   `NrFilesPerDistance` and `OmegaStep` are now RAW, and a mix of package versions reads
   them differently with **no error** — the reduction and the fit derive the same wrong
   frame count from the same key, agree with each other, and put every spot at the wrong ω.
   At 20-ID the gate carries a second job: the HDF5 reader first shipped in
   `midas-nf-preprocess` **0.7.0**, and below it `extOrig h5` cannot work at all.

2. **Fix the pixel encoding before you choose any threshold** (§5d, §3h). Encoding is
   **per scan, not per detector**: on one detector serial, `nfdev_jul26` is 10-bit stored
   ×64 (max 65472) while `NF_Au_cube_0802` and the SS316L NF scan are 12-bit unscaled
   (max 4092). Declare it as `PixelScale`; it defaults to 1, warns in both directions, and
   **never infers**. Run `np.unique` on one frame. Getting it wrong turns "threshold 2"
   into "threshold 128" and thresholds the pedestal, so the background becomes signal —
   it produced three wrong distance answers in a row before it was found.

3. **Confidence 1.0 does not mean the geometry is right.** It is a *plateau*: `ty` seeds
   2° apart all reach exactly 1.0000. The test that separates a real orientation field
   from a wrong plateau is **misorientation between spatial neighbours vs random pairs**
   (0.23° / 78 % under 5°, against 40.98° / 4.5 %). maxC and the median are blind to it.

4. **The order matters and was confirmed with the instrument scientist.** BC comes from
   `DetZBeamPos`; `Lsd` comes from spots; neither measurement can give the other's
   quantity. Getting the order wrong is itself a documented failure mode.
   With an alternating-distance file series (`z7and11`), **measure which file is which**
   from the spot radii (their ratio follows the distance ratio); on one beamtime the FIRST
   file of each pair was the short distance, and "even = short" was wrong for two samples.

5. **On weak signal, fix the reduction before the geometry.** Denoising and dropping the
   threshold was worth 3.6× the voxels at C ≥ 0.9; a converged geometry refinement was
   worth +0.005 FracOverlap. Set that threshold with `BlanketSigma`, not
   `BlanketSubtraction` — the latter was an int and could not express a sub-σ step.

6. **No DetZBeamPos scan and no beam on the detector? Check the "stripe" before using it.**
   A beam stripe is thousands of counts. A band a few counts above the floor is scatter,
   and one such band (zbc off by 138 px) made 47 gold reconstructions sit at noise. A
   calibrant measured at three or more distances gives Lsd, zbc and the per-distance BC
   drift from its own rays (spine hard rule 13; handbook §6i-quater; DIAGNOSIS "every
   geometry sits at the same noise floor"). Do not extend a grid scan over Lsd and ybc
   with a wrong zbc: it cannot help.

7. **Accept a geometry refinement only if the FULL MAP gets better** (hard rule 26). A
   multi-point fit on ~10 voxels raised its own overlap twice while the full map got
   worse (30 → 18 and 593 → 291 voxels at C ≥ 0.5). Compare voxel counts at one grid, plus
   the FF or neighbour check, before adopting refined Lsd / BC / tilts.

8. **Sparse dark-subtracted frames keep their static background** (hard rule 27): ~90 %
   of pixels have a temporal median of 0, so median subtraction removes nothing of a static
   pattern, and a low threshold lets it through as peakless, ω-fixed blobs. Test a
   detector by the overlap of its mask with the mask 180° away (real spots do not
   return). `SpotDetect poisson` sets a per-pixel threshold from the local mean and
   **robust** variance for a stated false-alarm budget; the budget is a dial (5 = strict,
   50000 = as many voxels as threshold 4 in a fifth of the time). The plain variance is
   hijacked by spots (22.8 against 2.7).

9. **The independent accuracy check is the FF of the same sample.** Match NF voxels to FF
   grains on orientation alone, crystal-side symmetry (`midas_stress`), print the chance
   rate, count per grain as well as per voxel, and run a cross-sample null. If a sample
   fails while the map looks good, it may be a different layer or region or a remount;
   `midas_stress.find_frame_rotation` searches for one rotation between the two grain
   sets (limits in its docstring: it is fooled by two unrelated sets with the same texture).

## When something looks wrong

Go to **`manuals/nf-hedm/DIAGNOSIS.md`** — symptom → discriminating test → cause → lever,
indexed by symptom rather than by step. Thirteen entries, each carrying a test that can come
back the other way; eight of them are 20-ID specific.

Before re-investigating anything, read **`manuals/nf-hedm/LAB_NOTEBOOK.md`** — several
attractive hypotheses are recorded there as *refuted*, with the measurement that killed
each one, and §5 lists the retractions specifically.

## Scope

**1-ID**, TIFF-per-frame, and **20-ID-D HT-HEDM**, Bluesky/HDF5 in DXchange layout. NF at
sector 20 is at **station D**; FF and PF run at both D and E, and everything reconstructed
so far is D data. 20-ID-D runs through the pipeline natively — set `extOrig h5` and the
reduction reads the HDF5 directly, streaming so a layer need not fit in RAM (§3h). The two code blockers that used
to close that door, and the ω-sign gate that outlived them, are all **closed**; the ω sign
at 20-ID-D is `aero`, negated, the same convention as 1-ID (hard rule 1). At 1-ID the RAMS-III
load-frame stage `ramsrot` is the exception: **counterclockwise, used as logged, never negated**.

**On any other beamline, stop and ask rather than adapting a recipe.** The array→lab
mapping must be re-derived, not inherited — getting it wrong **mirrors the microstructure
invisibly**, the same silent failure as the ω sign, with nothing in the `.mic` that shows
it. At 20-ID it *was* re-derived and the 1-ID flip survived by a margin that leaves no
doubt (maxC 0.000000 vs 0.6957 for the two candidates). **That method is what transfers —
build both masks from one reduction and let the calibrant decide — not the constant.**

## Sibling doc sets

`manuals/ff-hedm/` (far-field, skill `ff-hedm`), `manuals/defect/` (diffuse-scattering
defect metrology — what the far-field peak fit discards, skill `defect`), `manuals/dfxm/`
(dark-field X-ray microscopy, skill `dfxm`), `manuals/tomo/` (tomography and the
**coordinate-system reference**, skill `tomo`), and, in the LaueMatching repository,
`scripts/pipeline/laue/` (skill `laue`).
