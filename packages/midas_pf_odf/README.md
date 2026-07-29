# midas-pf-odf

Joint per-grain peak-shape inversion of pf-HEDM (point-focused HEDM, also
known as scanning 3DXRD) data.

**Phase 1 (current):** for each grain, fit all of its voxels'
`(R_V, ε_V)` jointly to image-MSE on measured 3D peak patches, given
voxel→grain assignment as input. Differentiable PyTorch forward;
voxel-summed splat per (spot, scan); closed-form per-spot intensity
scale `c_s*`.

**Phase 2 (later):** per-voxel ODF (sub-grain mosaic / GND density) on
the same forward.

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_pf_odf/notebooks).

## Status

Pre-alpha; private. See [dev/RESTART.md](dev/RESTART.md) for the active
worklog and resume point.

## Where this sits relative to the literature

Henningsson et al. 2020 (J. Appl. Cryst.) introduced PCR / ASR — joint
multi-voxel intragranular strain reconstruction from scanning 3DXRD,
using **peak centroids only**. They explicitly call out peak-shape
inversion as the next direction (untapped at time of writing). This
package takes that step:

- Centroid → peak-shape (image-MSE on `(F, P, P)` patches).
- Hand-derived Jacobian → autograd / PyTorch.
- Single (R_V, ε_V) per voxel → per-voxel ODF (Phase 2).

The forward model is `midas_diffract.HEDMForwardModel`; reusable
infrastructure (sparse splatter, ODF parameterizations, held-out CV
selector, sparse-tile chunking) comes from `midas_grain_odf`.

## Quickstart

```bash
cd packages/midas_pf_odf
pip install -e .[dev]
python -m pytest tests/ -xvs
```

## Layout

```
midas_pf_odf/        — public library
    simulate.py       — synthetic plant (orientation+strain gradients, multi-grain, noise)
    forward.py        — joint per-grain forward (soft beam gate, voxel-summed splat)
    inversion.py      — joint inversion driver (Adam / L-BFGS, identifiability knobs)
    validation.py     — per-voxel RMSE vs plant, held-out R²
    io.py             — I/O for real data (deferred)
dev/                  — implementation plan, paper, notebooks, worklog
tests/                — synthetic round-trip tests
```

## Decisions locked

1. **Identifiability knob:** project ε to mean-zero per grain (default,
   recommended) or per-voxel-lattice over-parameterized (toggle).
2. **Smoothness regularizer:** off by default (λ_smooth = 0).
3. **Synthetic source:** torch-native simulator inside this package.
4. **Outlier voxels:** soft-mask by held-out R² in the loss.
5. **Multi-grain parallelism:** one grain × one GPU each via parsl
   (deferred until workflow integration).
6. **Output schema:** parallel HDF5; centroid path untouched (deferred).
