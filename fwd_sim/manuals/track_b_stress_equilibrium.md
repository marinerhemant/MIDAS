# Track B: Stress Computation & Equilibrium Constraints

## What It Does

Computes stress tensors from MIDAS strain output using Hooke's law, then applies equilibrium constraints to enforce physically consistent stress fields across all grains.

Two implementations:
- **NumPy** (`utils/stress_strain.py`) -- for post-processing scripts
- **PyTorch** (`fwd_sim/hedm_losses.py`) -- for differentiable optimization

## Files

| File | Purpose |
|------|---------|
| `utils/stress_strain.py` | NumPy: Voigt, Hooke, stiffness library, equilibrium |
| `fwd_sim/hedm_losses.py` | PyTorch: differentiable versions of the above |

## Prerequisites

```bash
source /path/to/miniconda3/bin/activate midas_env
# Needs: numpy, h5py (for reading MIDAS output)
# For differentiable version: torch
```

---

## Quick Start: From consolidated_Output.h5 to Stress

This is the most common workflow -- you have a finished FF-HEDM analysis and want stress tensors.

```python
import numpy as np
import sys
sys.path.insert(0, "/Users/hsharma/opt/MIDAS/utils")
from stress_strain import (
    read_grains_h5,
    lattice_params_to_strain,
    strain_grain_to_lab,
    hooke_stress,
    get_stiffness,
    volume_average_stress_constraint,
    hydrostatic_deviatoric_decomposition,
)

# ─── Step 1: Load grain data ───────────────────────────────────────
grains = read_grains_h5("consolidated_Output.h5")

print(f"Loaded {len(grains['grain_ids'])} grains")
print(f"  Orientations:    {grains['orientations'].shape}")    # (N, 3, 3)
print(f"  Lattice params:  {grains['lattice_params'].shape}")  # (N, 6)
print(f"  Positions:       {grains['positions'].shape}")       # (N, 3)
print(f"  Strain (Fable):  {grains['strain_fable'].shape}")    # (N, 3, 3)
print(f"  Radii:           {grains['radii'].shape}")           # (N,)

# ─── Step 2: Get stiffness matrix ─────────────────────────────────
# For cubic materials (Au, Cu, Fe, Ni, Al, W, Si, Ti, CeO2):
C = get_stiffness("Au")  # returns (6, 6) in GPa, Voigt-Mandel notation

# For custom materials:
from stress_strain import cubic_stiffness
C_custom = cubic_stiffness(C11=250.0, C12=150.0, C44=120.0)  # GPa

# ─── Step 3: Compute stress tensors ───────────────────────────────
# The strain tensors are already in grains['strain_fable'] (lab frame).
# Apply Hooke's law:

N = len(grains['grain_ids'])
stresses = np.zeros((N, 3, 3))
for i in range(N):
    stresses[i] = hooke_stress(
        grains['strain_fable'][i],        # (3, 3) strain in lab frame
        C,                                 # (6, 6) stiffness in crystal frame
        orient=grains['orientations'][i],  # (3, 3) grain orientation
        frame="lab",                       # strain is in lab frame
    )

print(f"\nStress tensors computed: {stresses.shape}")  # (N, 3, 3)
print(f"  Example σ_xx: {stresses[0, 0, 0]:.1f} GPa")
# Note: if strain is ~1e-3, stress is ~1e-3 * 200 GPa = 0.2 GPa = 200 MPa
```

---

## Step 4: Apply Volume-Average Equilibrium Constraint (B1)

This enforces that the volume-weighted average stress equals the applied macroscopic stress. For an unloaded sample, the average should be zero.

```python
# Grain volumes (proportional to radius^3, or use MIDAS radius directly)
volumes = (4/3) * np.pi * grains['radii']**3  # spherical grains

# For unloaded sample (no external stress):
stresses_corrected = volume_average_stress_constraint(
    stresses,      # (N, 3, 3)
    volumes,       # (N,)
    applied_stress=None,  # default: zero (unloaded)
)

# For uniaxial tension of 100 MPa in x:
applied = np.diag([0.1, 0.0, 0.0])  # 0.1 GPa = 100 MPa
stresses_corrected = volume_average_stress_constraint(
    stresses, volumes, applied_stress=applied
)

# Verify: volume-weighted average should now match applied
V_total = volumes.sum()
w = volumes / V_total
avg = np.sum(w[:, None, None] * stresses_corrected, axis=0)
print(f"\nVolume-average stress after correction:")
print(f"  σ_xx = {avg[0,0]:.6f} GPa (should be {applied[0,0]:.1f})")
print(f"  σ_yy = {avg[1,1]:.6f} GPa (should be 0)")
```

---

## Step 5: Hydrostatic-Deviatoric Decomposition (B2)

Separates each grain's stress into:
- **Deviatoric** (shape change): well-determined from relative peak shifts
- **Hydrostatic** (pressure): poorly determined from absolute d-spacing; fixed by equilibrium

This removes the dependence on the ambiguous strain-free reference lattice parameter (d0).

```python
hydro, dev, corrected = hydrostatic_deviatoric_decomposition(
    stresses,       # (N, 3, 3) raw stresses
    volumes,        # (N,)
    applied_stress=None,  # unloaded sample
)

# hydro: (N,) per-grain hydrostatic stress
# dev:   (N, 3, 3) per-grain deviatoric stress tensors (traceless)
# corrected: (N, 3, 3) full stress = hydro*I + dev (equilibrium-enforced)

print(f"\nHydrostatic-deviatoric decomposition:")
for i in range(min(3, N)):
    gid = grains['grain_ids'][i]
    print(f"  Grain {gid}: hydro={hydro[i]:.4f} GPa, "
          f"|dev|={np.linalg.norm(dev[i]):.4f} GPa")

# Verify deviatoric is traceless:
traces = np.trace(dev, axis1=-2, axis2=-1)
print(f"  Max deviatoric trace: {np.max(np.abs(traces)):.2e} (should be ~0)")
```

---

## Alternative: Compute Strain From Scratch

If you have lattice parameters but not the pre-computed strain tensors:

```python
from stress_strain import lattice_params_to_strain, strain_grain_to_lab

# Unstrained reference lattice
latc0 = np.array([4.08, 4.08, 4.08, 90.0, 90.0, 90.0])

# Per-grain strained lattice parameters
latc_fit = grains['lattice_params']  # (N, 6)

# Compute strain in grain frame
strain_grain = lattice_params_to_strain(latc_fit, latc0)  # (N, 3, 3)

# Transform to lab frame
strain_lab = np.array([
    strain_grain_to_lab(strain_grain[i], grains['orientations'][i])
    for i in range(N)
])
```

---

## Voigt Notation Utilities

For interfacing with FEM codes or other conventions:

```python
from stress_strain import tensor_to_voigt, voigt_to_tensor

# 3x3 tensor -> 6-vector (Mandel: sqrt(2) on shear)
sig_voigt = tensor_to_voigt(stresses[0])
# [σ_xx, σ_yy, σ_zz, √2·σ_yz, √2·σ_xz, √2·σ_xy]

# Back to 3x3
sig_tensor = voigt_to_tensor(sig_voigt)

# The 6x6 rotation matrix for transforming Voigt vectors between frames:
from stress_strain import rotation_voigt_mandel
M = rotation_voigt_mandel(grains['orientations'][0])  # (6, 6)
sig_crystal = M @ tensor_to_voigt(stresses[0])  # stress in crystal frame
```

---

## Differentiable Version (PyTorch, for optimization)

Use this when you want to backpropagate through the stress computation:

```python
import torch
from hedm_losses import (
    hooke_stress,
    cubic_stiffness_tensor,
    volume_average_stress_constraint,
)

# Create stiffness tensor on GPU or CPU
C = cubic_stiffness_tensor(192.9, 163.8, 41.5, dtype=torch.float64)

# Strain as differentiable tensor
strain = torch.tensor(grains['strain_fable'][0], dtype=torch.float64,
                       requires_grad=True)
orient = torch.tensor(grains['orientations'][0], dtype=torch.float64)

# Forward: strain -> stress (differentiable)
stress = hooke_stress(strain, C, orient=orient, frame="lab")

# Use in a loss function
target_hydro = 0.0  # equilibrium: average hydrostatic should be zero
hydro = torch.trace(stress) / 3.0
loss = (hydro - target_hydro) ** 2
loss.backward()

print(f"d(loss)/d(strain) = {strain.grad}")  # gradients exist!
```

---

## Available Materials

```python
from stress_strain import STIFFNESS_LIBRARY
for mat, params in STIFFNESS_LIBRARY.items():
    print(f"  {mat:4s}: C11={params['C11']:6.1f}, C12={params['C12']:6.1f}, "
          f"C44={params['C44']:6.1f} GPa")
```

Output:
```
  Au  : C11= 192.9, C12= 163.8, C44=  41.5 GPa
  Cu  : C11= 168.4, C12= 121.4, C44=  75.4 GPa
  Al  : C11= 108.2, C12=  61.3, C44=  28.5 GPa
  Fe  : C11= 231.4, C12= 134.7, C44= 116.4 GPa
  Ni  : C11= 246.5, C12= 147.3, C44= 124.7 GPa
  Ti  : C11= 162.4, C12=  92.0, C44=  46.7 GPa
  W   : C11= 522.4, C12= 204.4, C44= 160.8 GPa
  Si  : C11= 165.7, C12=  63.9, C44=  79.6 GPa
  CeO2: C11= 403.0, C12= 105.0, C44=  60.0 GPa
```
