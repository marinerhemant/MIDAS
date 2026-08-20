# Phase 1a — reduce: raw frames to an (η, R) cake

> Part of the **XRD-CT doc set**. Spine: [`README.md`](README.md).

**Goal:** every frame integrated once, into a cached cake that every later question is
answered from. The raw frames are never read twice.

**Entry:** phase 1 has confirmed the geometry — and on this technique that means the
**distance was refined from the data**, not taken from metadata. If you have not done that, go
back; a wrong distance is baked into every number from here on.

---

## 1a.1 Calibrate UNSEEDED, and bound in pixels

This is the step where the largest single error enters, and the fix is counter-intuitive:
**give the calibrator less, not more.**

```python
from midas_calibrate_v2 import calibrate
r = calibrate(
    img,                          # the calibrant frame, unaltered
    wavelength=0.11595, pxY=150.0, calibrant="CeO2",
    min_ring_radius_px=350.0,     # bounds in DETECTOR PIXELS only
    max_ring_radius_px=1200.0,
    refine_tilts=True, refine_distortion=True,
    build_residual_corr=True,
    output_dir=str(OUT), verbose=True,
)
# NO initial_Lsd. NO BC_guess. NO initial_BC_y/z.
```

**Never hand it a beam-centre guess.** A supplied beam centre **overrides the auto-seeder**, and
the fit then cannot travel to the right answer. Measured on a frame from this beamline: a
hand-supplied guess gave **1040 µε** residual strain against **47 µε** from the seeder on the
same frame. This is a hard rule, not a preference.

**Bounds in detector pixels are safe; bounds in 2θ or Å are not.** A pixel range is
geometry-free, so it cannot bias the distance it is helping to determine. Choose the range to
clear the low-R plateau and stay inside the useful field.

**Do not pass a distance you measured another way.** Keep it as an *independent check* on what
the seeder returns. On the CeO₂ set the survey's ring measurement said 1632 mm and the
unseeded fit returned 1632.2 mm — that agreement is only evidence because the two were kept
apart.

### The acceptance gate

| Quantity | CeO₂ 11-ID-C, measured | Gate |
|---|---|---|
| post-residual strain | **66.2 µε** | **< 100 µε — hard.** A calibrant is a strain-free powder; more than 100 µε means the geometry is wrong, not the sample |
| in-loop strain | 69.8 µε | same |
| refined Lsd | **1 632 201 µm** | compare against metadata *and* the beamline file, separately |
| `tx` | **0.0, held** | a powder standard **cannot see `tx`** — hold it fixed here and refine it from grains later if you need it |
| ty, tz | 0.0108°, 0.2051° | |

Provenance: `~/MIDAS/wd/dt_survey/calib/geometry_for_dt.json` on `11idc`, produced by
`calibrate_tomodata.py`.

**Always overlay the predicted rings on the frame and look at it.** `overlay_full.png` plus a
zoom on the innermost, a middle and the outermost ring. A fit can reach a good residual on a
subset of rings and be wrong at the field edge; the overlay is the only thing that shows it.

### The residual correction map

`build_residual_corr=True` writes `residual_corr.bin` (66 MB here) — a per-pixel map of what
the parametric model could not absorb.

**It is silently ignored below `midas-integrate-v2 0.3.2`.** That version declared the
`ResidualCorrectionMap` field and did not use it, so every integration discarded the residual
the calibration had just measured, and nothing reported a problem. The failure is a slightly
wrong radius, not an error. `midas_dt`'s floor carries this; check it if you are driving the
integrator yourself.

## 1a.2 Read the frames as they are

**qxrd "Subtracted Data" frames are already background-subtracted, `float32`, and carry
negative values.** Pass them through **unaltered**. Clipping the negatives is a silent edit to
the thing being fitted, and it biases every integrated intensity upward by roughly the noise
half-width.

Other formats: `midas_dt.scan.RawFormat` / `DTScan.from_stem` handle the `.raw` path and the
1-ID conventions (ω negation, dropped first frame, snake detection).

## 1a.3 Integrate — and cache the whole radial range

```python
from midas_dt import Channel, DTGeometry
from midas_dt.reduce import FrameReducer

geo = DTGeometry(lsd_um=..., bc_y_px=..., bc_z_px=..., px_um=150.0,
                 n_pixels_y=2880, n_pixels_z=2880, wavelength_a=0.11595,
                 tx_deg=0.0, ty_deg=..., tz_deg=..., distortion=...,
                 rho_d_um=...)
reducer = FrameReducer(geo, channel)
frame = reducer.reduce(img)         # ReducedFrame
```

**★ `ReducedFrame.intensity` is `(n_eta, n_r)` — azimuth BEFORE radius.**

This is the single most common indexing mistake on this data, because it is the *opposite* of
the 1-ID integrated `.bin` layout, which is `[nR][nEta]`. Both reshape cleanly from a flat
buffer, so a swap gives a transposed array and no error.

**Verify it on your own data rather than trusting either convention:** collapse each axis and
look. The radial axis shows sharp rings on a falling background; the azimuthal axis is smooth.
Measured on the CeO₂ cake, one (translation, ω) frame:

| axis | length | max/median | strong local maxima |
|---|---|---|---|
| 0 | 36 | **1.03** | — → this is **η** |
| 1 | 850 | **181** | 4+ → this is **R** |

That is a five-second check and it is unambiguous.

<!-- The script that built this cache has `n_r, n_eta = probe.shape`, which is backwards,
     and the file carries an attr saying the label arrays were written swapped and later
     fixed. The DATA was always fine. Anyone reading build_cache.py to learn the layout
     will get it wrong; read the reduce.py field comment or run the check above. -->

### Cache the whole useful radial range, not one ring

R = 350–1200 px at 1 px covers all nine CeO₂ reflections; 36 η bins (10°) rebins down to any
coarser azimuthal choice later, including a single-bin survey.

**157 GB of TIFFs → 0.62 GB of cake.** Caching only one ring's window would save 90 % of that
and force a full re-read the moment anyone asks about a second ring — which is exactly what
made the first survey slow. Once cached, fitting all 5054 projections takes **1.4 s** on 16
workers.

Write the geometry, the R/η/translation axes and the binning **into the file as datasets and
attrs**, so a cake is self-describing and its provenance travels with it.

**★ Cache the VARIANCE too.** `FrameReducer` returns `ReducedFrame.variance` alongside
`.intensity` — a closed-form Poisson variance propagated through the integration. Storing only the
intensity throws it away, and every later noise question then needs an *approximation*. Two traps
that follow from discarding it, both measured on the CeO₂ cache:

* the cake holds the **per-bin mean** (`Σw·I/Σw`), **not a sum** — so `sqrt(Σ cake values)` is
  **not** `sqrt(N_photons)`, it is off by `sqrt(gain / n_pix)`;
* on that dataset the two factors nearly cancel by coincidence (Varex ~100–200 ADU/photon at
  105 keV against 70–200 pixels per bin), so the approximation landed within ~35 % and *looked*
  fine. Do not rely on that cancelling.

### ★ The hard-binning integrator has its own azimuthal artefact

`integrate_hard` returns `sums / counts` — the **mean over whichever pixels land in each (R, η)
bin**. Each bin therefore reports intensity at the mean R of *its own* pixel set, and that
effective R jitters with the pixel lattice.

**Measured:** a perfectly uniform, perfectly noiseless synthetic image pushed through the reducer
comes out with **0.19–0.44 % azimuthal area RMS** — structure from a sample that has none. It is
fixed in the **detector** frame, so it is ω-locked and translation-invariant, and its ring-to-ring
correlation is only **+0.035** — meaning it passes straight through the usual "is it common across
rings?" flat-field test.

On the CeO₂ scan it accounts for ~25 % of the variance of an ω-locked floor that five analyses
attributed to the sample. **If you are chasing an azimuthal signal below ~0.5 %, push a uniform
synthetic through your own reducer first and measure what it invents.**

### ★ Test a counting process by sweeping the η BIN WIDTH

The clean positive test, and it costs nothing once the cake is built: relative azimuthal RMS
against azimuthal bin width. **A counting process scales as −0.5; a smooth systematic gives 0.00.**

On CeO₂, 45° → 2°: slope **−0.520** (range −0.447…−0.538) on 9 of 9 rings, noise floor 20–60×
below, lag-1 η autocorrelation ≈ 0. That is what established the random component as crystallite
counting — after a chord-length argument had been tried and refuted by three lenses.

**Cache fine in η and rebin down**, so this axis stays available. Every earlier number in this
project was taken at one fixed 10° binning and the scaling axis was free the whole time.

### Parallelism: pin threads BEFORE importing numpy

```python
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np          # AFTER, or the libraries cache the wrong thread count
```

Also `torch.set_num_threads(1)` in each worker.

**Build the integration matrix once per worker, in the pool initialiser.** That build is the
expensive setup; the per-frame integration is cheap.

**This work is I/O bound, not CPU bound** — ~260 MB/s, and 64 parallel workers gave 8.0
frames/s against 6.7 serial. So more workers buys very little, while *unpinned* workers each
spawning 20 threads oversubscribe the box and run **slower than serial**. On another host 15
unpinned processes drove load to **437** with nothing finishing in 40 minutes.

## 1a.4 The three conventions to fix now

Restated from phase 0/1 because this is the step that bakes them in, and none is recoverable
afterwards from a finished reconstruction:

| Convention | Where |
|---|---|
| **ω sign** — negated once for the 1-ID aerotech | `DTScan.from_stem`; `conventions.aps_1id_omega` |
| **First frame dropped** — 1-ID writes a throwaway every acquisition | `conventions`; check whether `HeadSize` already skips it |
| **Snake** — *detected from the data*, never read from a flag | `conventions.detect_snake` / `unsnake` |

A missed snake mirrors alternate rows of the voxel grid, and the result is a plausible-looking
microstructure.

## 1a.5 What to hand forward

```
Geometry:      lsd, bc_y, bc_z, tx (held?), ty, tz, distortion terms kept AND dropped
Gate:          post-residual strain (< 100 ue?), overlay images looked at?
Residual map:  built? and is midas-integrate-v2 >= 0.3.2?
Cake:          path, shape, and WHICH AXIS IS WHICH (verified, not assumed)
Axes:          r_px, eta_deg, translation positions -- stored in the file
Conventions:   omega sign, first-frame handling, snake detected
Cost:          frames read, wall time, frames/s
```

Then `phase-1b-reconstruct.md`.
