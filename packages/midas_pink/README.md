# midas-pink

Pink-beam (and white-beam) HEDM analysis via a spectrum-aware
differentiable forward model.

`midas-pink` extends [`midas-diffract`](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_diffract)
from monochromatic to arbitrary illumination spectra by integrating
`S(E)` as a discrete weighted sum over per-energy mono forward
evaluations. The construction unifies monochromatic, pink, and white
HEDM under a single differentiable computational graph with no change
to the loss, the optimiser, or the parameterisation.

> **Companion paper.** Sharma, Andrejevic, Cherukara, "Pink-Beam
> High-Energy Diffraction Microscopy via a Differentiable Forward
> Model with Spectrum-Aware Inversion," IUCrJ (submitted, 2026).
> Full text and reproducible scripts in
> [`dev/paper/`](dev/paper/).

## Install

```bash
pip install midas-pink   # post-publication
# or, for development:
pip install -e packages/midas_pink/
```

Requires `midas-diffract>=0.1.0` and `midas-hkls>=0.1.0`.

## Notebooks

Worked-example Jupyter notebooks live in `notebooks/`. They are **not shipped with `pip install`** — get them by cloning the [MIDAS repository](https://github.com/marinerhemant/MIDAS/tree/master/packages/midas_pink/notebooks).

## Quick start

```python
import torch
import midas_diffract as md
import midas_pink as mp

# 1. Spectrum: fixed Gaussian pink S(E) FWHM/E0 = 1e-2
spec = mp.ParameterisedSpectrum(
    E0_keV=71.6764, half_bw=0.03, n_samples=51,
    init_kind="gaussian", init_rel_bw=1e-2, fixed=True,
)

# 2. Per-energy mono model bank from a geometry factory
def geom(lam_A):
    return md.HEDMGeometry(
        Lsd=1_000_000.0, y_BC=1024.0, z_BC=1024.0, px=200.0,
        omega_start=-180.0, omega_step=0.25, n_frames=1440,
        n_pixels_y=2048, n_pixels_z=2048,
        min_eta=6.0, wavelength=lam_A,
    )

from midas_hkls import SpaceGroup, Lattice
bank = mp.build_pink_bank(
    spec,
    space_group=SpaceGroup.from_number(225),  # FCC
    lattice=Lattice.for_system("cubic", a=4.078),
    geom_factory=geom, two_theta_max_deg=8.0,
)

# 3. Pick ROIs from the ground-truth state, splat observation
plan = mp.plan_rois_from_state(
    bank, euler.unsqueeze(0), pos.unsqueeze(0),
    lattice_params=latc, roi_h=31, roi_w=31,
)
observed = mp.splat_rois(bank, plan, euler.unsqueeze(0), pos.unsqueeze(0),
                         latc, sigma_psf_px=1.5)

# 4. Recover (orientation, lattice) from a perturbed initial guess
result = mp.recover_grain_state(
    bank, plan, observed,
    init_euler=init_euler, init_position=init_pos, init_lattice=init_latc,
    cfg=mp.RecoveryConfig(sigma_psf_px=1.5),
)
print("misori (m-deg):", result["misori_mdeg"])
print("strain err max:", result["strain_err_max"])
```

## What's in the package

| Module | What it provides |
|---|---|
| `midas_pink.spectrum` | `ParameterisedSpectrum` -- learnable softmax-normalised energy weights over a fixed dense grid |
| `midas_pink.inverse`  | `build_pink_bank`, `plan_rois_from_state`, `splat_rois` (2D), `splat_rois_3d` (3D, with frame axis), `recover_grain_state` (mono/pink with known S(E)), `recover_joint` (joint S(E) + grain fit with optional centroid pin), `fit_spectrum_to_rois` (two-stage calibrant fit) |

Form factor (via `midas_hkls.form_factor`), Lorentz-polarization, and
mosaicity are optional per-spot weighting in `splat_rois` / `splat_rois_3d`.

## Reproducing the paper

All synthetic results in the companion paper are reproducible from
`dev/paper/scripts/`. See [`dev/paper/README.md`](dev/paper/README.md)
for the per-proto index.

## License

BSD-3-Clause.
