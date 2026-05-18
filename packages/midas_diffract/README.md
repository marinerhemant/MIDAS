# midas-diffract

End-to-end differentiable forward model for High-Energy Diffraction Microscopy (HEDM), covering far-field (FF), near-field (NF), and point-focused (pf-HEDM) geometries. Pixel-exact agreement with the canonical C reference simulators in [MIDAS](https://github.com/marinerhemant/MIDAS).

Companion paper: Sharma, Zhang, Andrejevic & Cherukara, *An End-to-End Differentiable Forward Model for High-Energy Diffraction Microscopy*, IUCrJ (in preparation, 2026).

## Installation

```bash
pip install midas-diffract           # core forward model + losses + optimizer
pip install midas-diffract[hkls]     # also installs midas-hkls for the
                                     # pure-Python reflection-list helper
```

Optional PyTorch CUDA or MPS back-ends are used automatically if available.

## Quick start

```python
import torch
import midas_diffract as md
from midas_hkls import Lattice, SpaceGroup           # optional, see [hkls]

# Detector + scan geometry
geom = md.HEDMGeometry(
    Lsd=1_000_000.0,              # um
    y_BC=1024.0, z_BC=1024.0,
    px=200.0,
    omega_start=0.0, omega_step=0.25, n_frames=1440,
    n_pixels_y=2048, n_pixels_z=2048,
    min_eta=6.0,
    wavelength=0.172979,           # Angstroms
)

# Reflection list: either compute from a SpaceGroup + Lattice via the
# midas-hkls helper, or supply (hkls_cart, thetas, hkls_int) yourself
# (e.g. parsed from MIDAS GetHKLList output).
sg = SpaceGroup.from_number(225)                        # FCC
lat = Lattice.for_system("cubic", a=4.08)               # Au
hkls_cart, thetas, hkls_int = md.hkls_for_forward_model(
    sg, lat, wavelength_A=geom.wavelength, two_theta_max_deg=15.0,
)

model = md.HEDMForwardModel(
    hkls=hkls_cart, thetas=thetas, geometry=geom, hkls_int=hkls_int,
)

# Forward pass: grain state -> predicted spots. All inputs are leaves
# of the autograd graph.
euler = torch.tensor([[45.0, 30.0, 60.0]], requires_grad=True) * (3.14159 / 180)
pos   = torch.tensor([[0.0, 0.0, 0.0]],  requires_grad=True)
latc  = torch.tensor([4.08, 4.08, 4.08, 90.0, 90.0, 90.0], requires_grad=True)
spots = model(euler, pos, lattice_params=latc)

# Scalar loss -> gradients w.r.t. orientation, position, lattice
loss = ((spots.omega * spots.valid) ** 2).sum()
loss.backward()
```

## Output modes

- `md.HEDMForwardModel.predict_spot_coords(spots, space="angular")` — returns
  `(2θ, η, ω)` in radians for each valid reflection (FF and pf-HEDM).
- `md.HEDMForwardModel.predict_spot_coords(spots, space="detector")` — returns
  `(y_pixel, z_pixel, frame_nr)` in fractional units (FF and pf-HEDM).
- `md.HEDMForwardModel.predict_images(spots, ...)` — renders a differentiable
  3D detector volume via Gaussian splatting (NF-HEDM output mode).

## Validation

The forward model has been validated to pixel-exact agreement against the
canonical C simulators `ForwardSimulationCompressed` and `simulateNF` in the
MIDAS distribution. See the companion paper and the MIDAS repository
`fwd_sim/paper/` directory for reproducibility scripts.

## Scope

`midas-diffract` v0.1.x is deliberately focused on the forward model and its
gradient chain. The following capabilities build on this substrate and are
released separately as they mature:

- Sub-voxel grain mixtures
- Physics-informed regularisation
- Bayesian uncertainty quantification via HMC / variational inference
- Temporal 4D-HEDM tracking
- Coupling to differentiable crystal plasticity (JAX-FEM)
- EM spot ownership for ambiguous FF patterns

## Citation

If you use `midas-diffract` in published work, please cite the companion paper.

## Licence

BSD-3-Clause.
