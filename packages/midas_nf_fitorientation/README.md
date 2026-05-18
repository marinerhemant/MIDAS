# midas-nf-fitorientation

Differentiable, drop-in replacement for the three NF-HEDM orientation
and calibration C executables in [MIDAS](https://github.com/marinerhemant/MIDAS):

| C executable | Python equivalent | Console script |
|---|---|---|
| `FitOrientationOMP` | `fit_orientation_run` | `midas-nf-fit-orientation` |
| `FitOrientationParameters` | `fit_parameters_run` | `midas-nf-fit-parameters` |
| `FitOrientationParametersMultiPoint` | `fit_multipoint_run` | `midas-nf-fit-multipoint` |

Built on top of [`midas-diffract`](../midas_diffract/), so the forward
model is the same one validated to pixel-exact agreement against the C
simulators. Orientation refinement uses a **vectorised PyTorch
Nelder-Mead** running every `(voxel × winner)` fit problem in one
batched forward call per NM iteration; calibration refinement (in
`fit_parameters` and `fit_multipoint`) uses **L-BFGS over the soft
Gaussian-splat surrogate** because Nelder-Mead scales poorly past
~10 DoF.

## What changed vs. the C codes

- **Orientation optimiser**: NLopt Nelder-Mead → **vectorised PyTorch
  Nelder-Mead** (`midas_nf_fitorientation.torch_nm`). Same hard
  FracOverlap objective the C uses, but every `(voxel, winner)` fit
  is batched into one forward call per NM iteration. Converged
  simplices are trimmed out of the active set on the fly so each
  successive iteration is cheaper. ~22× faster than per-winner scipy
  NM on H100; bit-equivalent answers modulo NM convergence noise.
- **Calibration optimiser** (joint orientation + geometry refinement
  in `fit_parameters` / `fit_multipoint`): NLopt LN_NELDERMEAD →
  PyTorch **L-BFGS** over a soft Gaussian-splat surrogate with
  tanh-reparameterised bounds. NM is the wrong tool past ~10 DoF;
  L-BFGS scales well, at the cost of optimising a slightly smoothed
  basin floor.
- **Obs volume storage**: the 24 GB `SpotsInfo.bin` is loaded as
  `uint8` (6 GB) by default. The orientation kernel only needs 0/1
  values; the L-BFGS calibration paths request `float32` explicitly.
- **Bounds**: each refined parameter (Eulers, tilts, Lsd, ΔLsd, BC,
  optionally wedge) is reparameterised as
  `x = x0 + tol * tanh(u)`, so L-BFGS sees an unbounded variable but
  the physical parameter cannot leave its tolerance box.
- **Tikhonov regularisation** (opt-in, layered on top of tanh): a
  quadratic prior on the calibration block, useful in the multi-point
  joint fit where you want to drift only when many voxels' worth of
  evidence agree.
- **Wedge refinement** (opt-in, paramfile key `RefineWedge 1`): not
  available in the C code; the model accepts wedge as a calibration
  DoF.
- **Multi-start global search** for the multi-point joint fit: the C
  code's NM→CRS2→NM ladder is replaced with `NumIterations`
  independent L-BFGS attempts, each seeded with a Gaussian
  perturbation of the previous best within the tanh box. CRS2's true
  global behaviour is lost; for well-seeded calibration the
  multi-start gives the same answer in practice.
- **Output files** (`MicFileBinary`, `MicFileBinary.AllMatches`)
  match the C `pwrite` byte layout exactly.

## Installation

From the repository root:

```bash
pip install ./packages/midas_diffract
pip install ./packages/midas_nf_fitorientation
```

PyTorch ≥ 2.0 is required; CUDA / MPS is auto-detected.

## CLI usage

The argument signatures mirror the C executables, so existing wrapper
scripts swap binaries without changes:

```bash
midas-nf-fit-orientation  params.txt blockNr nBlocks nCPUs [--device cuda] [--fp32] [--screen-only]
midas-nf-fit-parameters   params.txt rowNr   [nCPUs]      [--device cuda] [--fp32]
midas-nf-fit-multipoint   params.txt         [nCPUs]      [--device cuda] [--fp32]
```

Common flags (parsed after the positional args):

- `--device {auto,cpu,cuda}` — defaults to `auto` (CUDA if available).
- `--fp32` — float32 forward (faster on CUDA, less bit-stable than
  float64 — the package default).
- `--screen-only` — stop after Phase 1 and dump `screen_cpu.csv`
  (mirrors `MIDAS_SCREEN_ONLY=1` in the C code).
- `--verbose` — chatty progress.
- `--lbfgs-max-iter N`, `--lbfgs-max-outer N` — L-BFGS step limits.

## Python API

```python
import midas_nf_fitorientation as fit

# Replaces FitOrientationOMP
fit.fit_orientation_run(
    paramfile="params.txt", block_nr=0, n_blocks=1, n_cpus=8,
    device="cuda", verbose=True,
)

# Replaces FitOrientationParameters (single-voxel calibration)
result = fit.fit_parameters_run(
    paramfile="params.txt", voxel_idx=42, n_cpus=4, verbose=True,
)
print(result["Lsd"], result["tilts"], result["frac_overlap"])

# Replaces FitOrientationParametersMultiPoint (joint multi-voxel)
result = fit.fit_multipoint_run(paramfile="params.txt", n_cpus=8)
```

## Paramfile keys

Every key consumed by the C executables is recognised, plus a small
set of new keys. See `midas_nf_fitorientation/params.py` for the full
schema. The new keys:

| Key | Default | Purpose |
|---|---|---|
| `RefineWedge 1` | off | Add wedge to the calibration DoF set |
| `WedgeTol 0.05` | 0.05° | Tanh-box width for wedge |
| `TikhonovCalibration 1.0` | 0 | Global λ for the calibration prior; 0 disables |
| `TikhonovSigmaLsd 100.0` | 100 µm | Prior σ for Lsd |
| `TikhonovSigmaTilts 0.05` | 0.05° | Prior σ for tilts |
| `TikhonovSigmaBC 1.0` | 1 px | Prior σ for beam centres |
| `TikhonovSigmaWedge 0.05` | 0.05° | Prior σ for wedge |
| `GaussianSplatSigmaPx 1.5` | auto | Override the auto-σ for the soft-overlap kernel |

`NumIterations` (existing) controls the number of multi-start trials
in `fit_multipoint`. Default 1 (single L-BFGS run); set to ≥ 8 for
true multi-start.

## Status

v0.3.x. The forward path is validated against `midas-diffract`
(pixel-exact vs. the C simulators); the fit drivers have unit-test
coverage at the module level. End-to-end agreement against the C
`MicFileBinary` on a real reconstruction dataset is the next milestone.

## Licence

BSD-3-Clause.
