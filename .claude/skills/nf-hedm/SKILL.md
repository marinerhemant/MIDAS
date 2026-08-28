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

## Five things to know before you start

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

5. **On weak signal, fix the reduction before the geometry.** Denoising and dropping the
   threshold was worth 3.6× the voxels at C ≥ 0.9; a converged geometry refinement was
   worth +0.005 FracOverlap. Set that threshold with `BlanketSigma`, not
   `BlanketSubtraction` — the latter was an int and could not express a sub-σ step.

## When something looks wrong

Go to **`manuals/nf-hedm/DIAGNOSIS.md`** — symptom → discriminating test → cause → lever,
indexed by symptom rather than by step. Ten entries, each carrying a test that can come
back the other way; five of them are 20-ID specific.

Before re-investigating anything, read **`manuals/nf-hedm/LAB_NOTEBOOK.md`** — several
attractive hypotheses are recorded there as *refuted*, with the measurement that killed
each one, and §5 lists the retractions specifically.

## Scope

**1-ID**, TIFF-per-frame, and **20-ID-D HT-HEDM**, Bluesky/HDF5 in DXchange layout. NF at
sector 20 is at **station D**; FF and PF run at both D and E, and everything reconstructed
so far is D data. 20-ID-D runs through the pipeline natively — set `extOrig h5` and the
reduction reads the HDF5 directly, streaming so a layer need not fit in RAM (§3h). The two code blockers that used
to close that door, and the ω-sign gate that outlived them, are all **closed**; the ω sign
at 20-ID is `aero`, negated, the same convention as 1-ID (hard rule 1).

**On any other beamline, stop and ask rather than adapting a recipe.** The array→lab
mapping must be re-derived, not inherited — getting it wrong **mirrors the microstructure
invisibly**, the same silent failure as the ω sign, with nothing in the `.mic` that shows
it. At 20-ID it *was* re-derived and the 1-ID flip survived by a margin that leaves no
doubt (maxC 0.000000 vs 0.6957 for the two candidates). **That method is what transfers —
build both masks from one reduction and let the calibrant decide — not the constant.**

## Sibling doc sets

`manuals/ff-hedm/` (far-field, skill `ff-hedm`), `manuals/dfxm/` (dark-field X-ray
microscopy, skill `dfxm`), `manuals/tomo/` (tomography and the **coordinate-system
reference**, skill `tomo`), and, in the LaueMatching repository,
`scripts/pipeline/laue/` (skill `laue`).
