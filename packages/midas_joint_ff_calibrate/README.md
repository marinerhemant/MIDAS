# midas-joint-ff-calibrate

Joint differentiable powder + FF-HEDM detector calibration.

The single-image powder calibration problem (Wright, Giacobbe & Lawrence
Bright 2022 § 3; midas_calibrate_v2 paper § 9) is rank-deficient on
per-panel `(δy, δz)` shifts: each panel's η coverage is too narrow to break
the radial-vs-azimuthal degeneracy without translating the detector to
multiple distances. For a ~100 kg detector on a translation stage that's
impractical.

This package solves the same problem at a single distance by combining the
calibrant powder image with a co-located HEDM grain-fit dataset. HEDM spots
are determinate in `(R, η)` and distribute across all panels at varied
azimuth, so the joint Fisher block on per-panel shifts becomes full rank.
The same machinery generalises to refining any subset of the unified spec —
geometry, distortion, panels, wavelength, per-grain orientation/position/
strain, or arbitrary user-defined blocks.

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_joint_ff_calibrate/notebooks).

**Start with [`05_grain_tx_from_grains`](notebooks/05_grain_tx_from_grains.ipynb)**
if you have an FF reconstruction in hand — it covers the one geometry parameter
a powder calibration can never give you (`tx`), why it matters, and how to
refine it. Notebooks 00–04 build up the joint-calibration theory behind the
IUCrJ paper.

## Grain-based `tx` — the powder-blind geometry

Powder/calibrant calibration cannot determine `tx`, the in-plane detector
rotation about the beam: Debye–Scherrer rings are rotationally symmetric about
that axis, so every value of `tx` fits equally well. Only single-crystal spots
break the symmetry. An FF reconstruction calibrated on a standard therefore
runs with `tx = 0`, and correcting it is a genuine second calibration pass.

```bash
midas-joint-ff-calibrate grain-tx \
    --paramstest  master_ff_params.txt \
    --layer-dir   recon/LayerNr_1 \
    --refine tx --kind angular --max-grains 50 \
    --out recon/LayerNr_1/paramstest_graintx.txt
```

> **Pass the *master* FF parameter file**, not the stripped per-layer
> `paramstest.txt` the refiner consumes — the latter omits `OmegaStep`,
> `NrFilesPerSweep` and `NrPixelsY/Z`, which the forward model needs. Without
> them every predicted spot is invalid and you get `matched spots = 0`.
> (A guard raises on `OmegaStep == 0` rather than failing silently.)

Also available as `refine_geometry_from_grains()` (library) and as an optional
FF pipeline stage (`midas-pipeline run ... --grain-geometry-run`). `tx` enters
at the *transforms* step, so benefiting from it means re-running the
reconstruction with the corrected file. Only `tx` and `Wedge` can be refined
here; see the notebook's Limitations section.

## Architecture

```
                   midas_peakfit  (shared substrate)
                   ParameterSpec / pack / lm / laplace / TPSpline / Σ=0
                          ▲                       ▲
                          │                       │
              midas_calibrate_v2          midas_fit_grain
              (powder forward + loss)     (HEDM forward + loss)
                          ▲                       ▲
                          └──────────┬────────────┘
                                     │
                       midas_joint_ff_calibrate
                       (joint spec / loss / drivers)
```

Pure-Python, fully autograd-traced. No legacy C code (`FitMultipleGrains.c`
is not used).

## Drivers

- `pipelines.alternating.AlternatingDriver` — the recommended default.
  Outer loop alternates between (geometry + grain orientation/position) and
  (grain strain) passes. Cheap, robust.
- `pipelines.full_joint.FullJointDriver` — refine every refined parameter
  at once with a single LM call; report MAP plus Laplace covariance.
- `pipelines.identifiability.fisher_block_rank` — diagnostic that reports
  rank, condition number, and σ per parameter on a user-chosen Fisher
  block under powder-only, HEDM-only, or joint evidence.

## Status

Pre-alpha. Companion paper to the J. Appl. Cryst. submission of
midas_calibrate_v2 (paper 3); see `dev/paper/`.
