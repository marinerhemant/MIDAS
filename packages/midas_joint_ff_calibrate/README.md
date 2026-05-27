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
