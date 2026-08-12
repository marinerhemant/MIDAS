# Phase 0 — Survey: is this scanning 3DXRD?

> Part of the **pf-HEDM doc set**. Spine: [`README.md`](README.md).

Before any recipe applies, confirm the dataset is pf-HEDM (scanning 3DXRD) and not
far-field or near-field. The three techniques share a detector and a beamline and are easy
to confuse from a folder listing alone.

## 0.1 The signature of scanning 3DXRD

| Evidence | pf-HEDM | far-field (`ff-hedm`) | near-field (`nf-hedm`) |
|---|---|---|---|
| Files per layer | **many** — one per translation step | one ω sweep | one per detector-distance × ω |
| Each file | a full ω rotation at one translation | the whole measurement | a shadow image stack |
| A `positions.csv` (or a translation motor logged per file) | **yes** — the beam is stepped across the sample | no | no |
| Beam | narrow (point/line), scanned | line/box, static | box, static |
| Reconstruction | a **2-D voxel grid** per layer | a grain list (centroids) | a spatially-resolved orientation map |

If there is one file and no per-file translation, it is far-field — switch to `ff-hedm`. If
the frames are shadow images at multiple detector distances, it is near-field — `nf-hedm`.

## 0.2 Read one raw file

Do not trust the folder name. Open one detector file and read its structure. For the HDF5
container MIDAS ingests:

```python
import h5py
f = h5py.File("<one scan file>", "r")
f.visititems(lambda n, o: print(n, getattr(o, "shape", "")) )
```

Look for and record:

- **`exchange/data`** — shape `(n_frames, ny, nz)`. `n_frames` is the ω sweep length; `ny,nz`
  the detector size. **Check the chunk shape** (`f["exchange/data"].chunks`): if it is one
  full frame per chunk, note it — it makes pf-odf patch extraction I/O-heavy (phase 4).
- **`measurement/instrument/SMS/aero`** — the **actual ω encoder value per frame**. This is
  ground truth for the ω sign in phase 1; the `paramstest` `OmegaStep` can disagree with it.
- **`measurement/instrument/SMS/samY`** (or the layer/height motor) — the **layer height**.
  Matching it against the far-field layers is how you pick the right FF seed (phase 2).
- **`measurement/process/scan_parameters/{start,end,step}`** — the nominal ω scan.

Count the scan files. The number of translation positions **`n_scans`** and the voxel grid
are linked: PF reconstructs an `n_scans × n_scans` grid (translations along two in-plane
axes from the same 1-D position list). A 259-file scan → a 259×259 = 67 081-voxel layer.

## 0.3 Attenuation and exposure — read it now, it decides the strain

The filename and the `paramstest` often carry the attenuation setting (an `att<N>` token, an
attenuator readback). **Record it and look at a frame's dynamic range:**

```python
import numpy as np
a = f["exchange/data"][f["exchange/data"].shape[0]//2]
print("min/median/max", a.min(), np.median(a), a.max(), "  saturated px:", (a>=60000).sum())
```

Two things decide whether per-voxel **strain** is achievable at all (envelope §2):

- **Are the low rings saturated?** (clamped at the detector maximum). Saturated low rings +
  weak high rings means one exposure cannot serve both — a fixed limit for this run.
- **How far above background do the useful mid/high-angle peaks sit?** A few hundred counts
  over a ~1–2 k pedestal is *signal-limited*: orientation will be fine, per-voxel strain
  will be provisional. State this in the report before fitting, not after.

## 0.4 What to hand forward

Write these into the runbook's pick-up point:

```
n_scans (translations) = <N>          voxel grid = <N>×<N>
detector = <ny>×<nz>, chunks = <...>, n_frames = <M>, ω step ≈ <...>
layer height (samY) = <...>           attenuation = <...>
low rings saturated? <y/n>            useful-ring SNR ≈ <...>
goal = grain map | grain map + strain
```

Then go to [`phase-1-geometry.md`](phase-1-geometry.md) — the ω sign and position
convention must be fixed before anything else, because a wrong value mirrors the whole map
and cannot be detected afterward.
