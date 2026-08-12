---
name: nf-hedm
description: >-
  Take a near-field HEDM (NF-HEDM) dataset from raw frames to a grain map: find
  the beamtime metadata, establish the omega sign, measure the beam centre from a
  DetZBeamPos scan, refine the geometry on a calibrant, reduce the images, fit
  orientations, and read the .mic. Use when asked to reconstruct, calibrate or
  diagnose an NF-HEDM / near-field 3DXRD beamtime, when handed a folder of NF
  TIFFs or a DetZBeamPos scan, or when an NF reconstruction has low confidence.
  Covers 1-ID; 20-ID HT-HEDM is a different world and is gated.
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

## Four things to know before you start

1. **Run the floor gate first** (spine §1). `SumFrames` **inverted** its unit convention:
   `NrFilesPerDistance` and `OmegaStep` are now RAW, and a mix of package versions reads
   them differently with **no error** — the reduction and the fit derive the same wrong
   frame count from the same key, agree with each other, and put every spot at the wrong ω.

2. **Confidence 1.0 does not mean the geometry is right.** It is a *plateau*: `ty` seeds
   2° apart all reach exactly 1.0000. The test that separates a real orientation field
   from a wrong plateau is **misorientation between spatial neighbours vs random pairs**
   (0.23° / 78 % under 5°, against 40.98° / 4.5 %). maxC and the median are blind to it.

3. **The order matters and was confirmed with the instrument scientist.** BC comes from
   `DetZBeamPos`; `Lsd` comes from spots; neither measurement can give the other's
   quantity. Getting the order wrong is itself a documented failure mode.

4. **On weak signal, fix the reduction before the geometry.** Denoising and dropping the
   threshold was worth 3.6× the voxels at C ≥ 0.9; a converged geometry refinement was
   worth +0.005 FracOverlap. Set that threshold with `BlanketSigma`, not
   `BlanketSubtraction` — the latter was an int and could not express a sub-σ step.

## When something looks wrong

Go to **`manuals/nf-hedm/DIAGNOSIS.md`** — symptom → discriminating test → cause → lever,
indexed by symptom rather than by step. Four entries, each carrying a test that can come
back the other way.

Before re-investigating anything, read **`manuals/nf-hedm/LAB_NOTEBOOK.md`** — several
attractive hypotheses are recorded there as *refuted*, with the measurement that killed
each one, and §4 lists the retractions specifically.

## Scope

1-ID, TIFF-per-frame. **20-ID HT-HEDM is a different acquisition, detector and file
format**, with two open blockers before that data can enter the pipeline at all (§3h). On
any other beamline the array→lab mapping must be re-derived, not inherited — getting it
wrong mirrors the microstructure invisibly, the same silent failure as the ω sign.

## Sibling doc sets

`manuals/ff-hedm/` (far-field, skill `ff-hedm`), `manuals/dfxm/` (dark-field X-ray
microscopy, skill `dfxm`), and, in the LaueMatching repository, `scripts/pipeline/laue/`
(skill `laue`).
