# midas-stress

Crystallographic stress-strain analysis with Voigt-Mandel notation and mechanical equilibrium constraints.

A pure-Python library (NumPy + SciPy) for computing stress from strain in polycrystalline materials measured by High-Energy Diffraction Microscopy (HEDM), neutron diffraction, synchrotron strain scanning, or EBSD. Optional PyTorch backend for differentiable analysis.

Part of the [MIDAS](https://github.com/marinerhemant/MIDAS) toolkit.

## Installation

```bash
pip install midas-stress
```

For the PyTorch backend (GPU-accelerated, differentiable):

```bash
pip install midas-stress[torch]
```

## Quick start — with any HEDM code

The library works with **numpy arrays from any source** — MIDAS, hexrd, ImageD11, DAXM, or your own scripts. Just provide orientations, strains, and volumes:

```python
import numpy as np
import midas_stress as ms

# Your data (from hexrd, ImageD11, or any source):
orientations = np.array(...)   # (N, 3, 3) orientation matrices
strains = np.array(...)        # (N, 3, 3) strain tensors (lab frame)
volumes = np.array(...)        # (N,) grain volumes

# One function does everything:
result = ms.compute_stress(
    strain=strains,
    stiffness=ms.get_stiffness("Cu"),
    orient=orientations,
    volumes=volumes,
)

print(f"Mean von Mises: {result['von_mises'].mean():.1f} GPa")
print(f"d0 correction:  {result['hydrostatic_shift']:.1f} GPa")
print(f"Correction SE:  {result['uncertainty']['hydrostatic_se_MPa']:.1f} MPa")
```

The returned `result` dict contains:
- `stress_raw` — per-grain stress before equilibrium correction
- `stress_corrected` — per-grain stress after correction
- `hydrostatic_corrected`, `deviatoric`, `von_mises` — decomposition
- `hydrostatic_shift` — the d0 correction that was applied
- `uncertainty` — statistical uncertainty of the correction

## Quick start — with MIDAS output

```python
import midas_stress as ms

grains = ms.read_grains("Grains.csv")

# Optional: convert MIDAS -> APS frame
sam = ms.grains_midas_to_sample(
    grains['orientations'], grains['positions'],
    grains['strain_fable'], target_frame="aps",
)

result = ms.compute_stress(
    strain=sam['strains'],
    stiffness=ms.get_stiffness("Cu"),
    orient=sam['orientations'],
    volumes=(4/3) * 3.14159 * grains['radii']**3,
    confidences=grains.get('confidences'),
    min_confidence=0.5,
)
```

## Why equilibrium constraints matter

Every HEDM experiment has an unknown strain-free lattice parameter (d0). A tiny error in d0 causes a large systematic error in hydrostatic stress — **identical for every grain** and therefore invisible in grain-to-grain comparisons:

| Material | d0 error (ppm) | Hydrostatic stress error (MPa) |
|----------|---------------|-------------------------------|
| Cu | 100 | 41 |
| Fe | 100 | 50 |
| Ni | 100 | 54 |
| W  | 100 | 93 |

`midas-stress` is the **only** library that fixes this via mechanical equilibrium:

- **FF-1**: Volume-average stress constraint (forces macroscopic balance)
- **FF-2**: Force-balance d0 (determines hydrostatic component from equilibrium, not from d0)
- **Confidence weighting**: handles incomplete grain populations
- **Uncertainty estimation**: reports how reliable the correction is

## Features

### Voigt-Mandel tensor algebra

```python
voigt = ms.tensor_to_voigt(strain_3x3)     # (3,3) -> (6,)
tensor = ms.voigt_to_tensor(voigt_6)        # (6,) -> (3,3)
M = ms.rotation_voigt_mandel(orient)        # 6x6 rotation in Voigt space
p = ms.hydrostatic(stress)                  # scalar pressure
s = ms.deviatoric(stress)                   # deviatoric tensor
vm = ms.von_mises(stress)                   # von Mises equivalent
```

All operations are vectorized: pass `(N, 3, 3)` arrays for batch computation.

### Hooke's law with stiffness database

```python
# Built-in stiffness for 9 materials: Au, Cu, Al, Fe, Ni, Ti, W, Si, CeO2
C = ms.get_stiffness("Fe")

# Or build your own
C = ms.cubic_stiffness(C11=231.4, C12=134.7, C44=116.4)
C = ms.hexagonal_stiffness(C11=162.4, C12=92.0, C13=69.0, C33=180.7, C44=46.7)

# d0 sensitivity analysis
sens = ms.d0_sensitivity("Cu")
print(f"Cu: {sens['sensitivity_MPa_per_100ppm']:.1f} MPa per 100 ppm d0 error")
```

### Coordinate frame conversions

```python
# MIDAS (X=beam, Y=OB, Z=up) <-> APS (X=OB, Y=up, Z=beam)
sam = ms.grains_midas_to_sample(orientations, positions, strains,
                                 target_frame="aps", omega_deg=0)
```

### Orientation and misorientation

```python
angle, axis = ms.misorientation(euler1, euler2, space_group=225)
# All 230 space groups supported
# C-accelerated when MIDAS is built; pure-Python fallback otherwise
```

### I/O

```python
grains = ms.read_grains("Grains.csv")      # MIDAS CSV format
grains = ms.read_grains("output.h5")       # Consolidated HDF5
```

### PyTorch backend (optional)

```python
import midas_stress.torch_backend as mst
stress = mst.hooke_stress(strain_tensor, stiffness, orient, frame="lab")
```

## Voigt-Mandel convention

```
v = [T_xx, T_yy, T_zz, sqrt(2)*T_xy, sqrt(2)*T_xz, sqrt(2)*T_yz]
```

The sqrt(2) scaling preserves the Frobenius norm: `||T||_F == ||v||_2`.

## Citation

```bibtex
@article{midas_stress,
  title   = {midas-stress: A Python Library for Crystallographic
             Stress-Strain Analysis with Mechanical Equilibrium Constraints},
  author  = {Sharma, Hemant and Park, Jun-Sang and Kenesei, Peter},
  journal = {Journal of Applied Crystallography},
  year    = {2026},
}
```

## License

BSD-3-Clause. See [LICENSE](../../LICENSE).
