# midas-uq

Cross-validation based uncertainty quantification for High-Energy
Diffraction Microscopy (HEDM) grain refinement. Three diagnostics, three
modalities (far-field, point-focused, near-field), all built on the
end-to-end differentiable forward model of
[`midas-diffract`](https://github.com/marinerhemant/MIDAS).

Companion paper: H. Sharma, J.-S. Park, P. Kenesei, N. Andrejevic
& M. Cherukara, *Cross-Validation Based Uncertainty Quantification
for HEDM Grain Refinement*, IUCrJ (in preparation, 2026).

## What it does

For each grain in a polycrystal HEDM refinement, `midas-uq` answers:

- **How reproducible is this grain's refined state under random
  resampling of its measured spots (or omega frames)?** — `half_half`
- **Which individual measurements are driving the fit, and which (if
  any) appear corrupted?** — `jackknife`
- **What is the local Gaussian-posterior covariance from the inverse
  Hessian?** — `laplace_covariance`
- **Is refinement overfitting in the low-spots-per-DOF regime?** —
  `rfree_gap`

No ground truth required. Half-half disagreement and jackknife influence
are post-hoc and label-free. The differentiable forward model from paper
I makes population-scale resampling (hundreds of grains × tens of
splits) a single-CPU minute, not a day.

## Installation

```bash
pip install midas-uq         # adds midas-diffract automatically
```

Optional CPU multi-processing for population studies:
```bash
pip install "midas-uq[mp]"
```

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_uq/notebooks).

## Quick start (Python)

```python
import torch
import midas_uq as muq
import midas_diffract as md

# Build the forward model the usual way (see midas-diffract docs)
geom  = md.HEDMGeometry(...)
model = md.HEDMForwardModel(hkls=..., thetas=..., geometry=geom)

# Grain seed (e.g., from a Grains.csv row)
init = muq.GrainState(
    euler_rad=torch.tensor([phi1, Phi, phi2], dtype=torch.float64),
    latc=torch.tensor([a, b, c, alpha, beta, gamma], dtype=torch.float64),
    pos=torch.tensor([x, y, z], dtype=torch.float64),
)

# Observed spots: (N, 3) of (2theta, eta, omega) in radians
obs = ...

# 1. Half-half UQ
uq = muq.half_half(model, init, obs, mode="ff", n_splits=5)
print(uq.misori_median_deg, uq.lattice_median_A)

# 2. Per-spot jackknife (only on a flagged grain)
jk = muq.jackknife(model, init, obs, mode="ff")
print(jk.top_k(10))  # 10 most influential spots

# 3. Laplace baseline for comparison
sigma_vec = torch.tensor([sigma_2theta, sigma_eta, sigma_omega],
                         dtype=torch.float64)
lp = muq.laplace_covariance(model, init, obs, sigma_vec, refine_first=True)
print(lp.misori_p95_deg, lp.condition_number)
```

For NF-HEDM, `observations` is a (F, H, W) detector volume and the API
is the same with `mode="nf"`; see `examples/nf_frame_split.py`.

## Quick start (CLI)

For a standard MIDAS dataset (Grains.csv + SpotMatrix.csv + paramstest +
hkls.csv):

```bash
# Population half-half UQ (writes one row per grain)
midas-uq half-half \
    --params  /path/to/paramstest.txt \
    --hkls    /path/to/hkls.csv \
    --grains  /path/to/Grains.csv \
    --spot-matrix /path/to/SpotMatrix.csv \
    --n-splits 5 \
    --out uq_half_half.csv

# Drill into a single grain
midas-uq jackknife --grain-id 5823 ...
midas-uq laplace   --grain-id 5823 ...
```

## API surface

| Symbol | Modality | Description |
|---|---|---|
| `half_half(model, init, obs, mode='ff'\|'pf'\|'nf', ...)` | all | K-split UQ dispatch |
| `jackknife(model, init, obs, mode=...)`                   | all | Leave-one-out dispatch |
| `half_half_spots`, `jackknife_spots`                       | FF/pf | Spot-based |
| `half_half_frames`, `jackknife_frames`                     | NF    | Frame-based |
| `laplace_covariance`                                        | FF/pf | Hessian baseline |
| `rfree_gap`                                                 | FF/pf | Train/holdout loss tracking |
| `GrainState`                                                | -    | Grain (euler, latc, pos) container |

Result dataclasses: `HalfHalfResult`, `JackknifeResult`, `LaplaceResult`,
`RFreeResult`.

## When to use what

| Diagnostic | Cost per grain | Surfaces |
|---|---|---|
| `rfree_gap` | 1× refine | overfitting at low n_obs/DOF (sparse / mosaic / pf-HEDM) |
| `half_half` (K=5) | 10× refine | noise + model misspec + local-minimum basin |
| `jackknife` | N_obs × refine | per-spot leverage and corruption candidates |
| `laplace_covariance` | 1× Hessian | local Gaussian-posterior baseline |

Half-half is the recommended population-scale diagnostic; jackknife is
the drill-down on grains flagged by half-half; Laplace gives a
complementary Gaussian baseline whose discrepancy from the empirical
half-half spread itself diagnoses non-Gaussian posterior structure.

## Reproducing the companion paper

The Ti-7Al population study, Park22 in-situ tensile sweep, synthetic
sweeps, and figure scripts live in `dev/paper/` of the repository.

## License

BSD-3-Clause.
