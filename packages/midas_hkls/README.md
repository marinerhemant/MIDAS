# midas-hkls

Crystallography toolkit for MIDAS: HKL list generation (sginfo-equivalent),
CIF I/O, and **differentiable structure factors** in PyTorch for
intensity-aware peak fitting in pf-HEDM, ff-HEDM, and powder diffraction.

## What it provides

### Always-on (numpy only)
- `SpaceGroup` — load by number, Hermann-Mauguin symbol, or Hall symbol; expose
  symmetry operations, systematic absences, equivalent reflections,
  multiplicities, Laue class, centering. All 230 SGs.
- `Lattice` — direct/reciprocal metric tensors, d-spacings, Bragg 2θ, with
  per-crystal-system convenience constructors.
- `generate_hkls()` — enumerate Laue-unique allowed reflections within a
  d-spacing or 2θ cutoff, sorted by d-descending, with multiplicities.
- `Atom`, `Crystal` — asymmetric-unit description with symmetry expansion to
  the full unit cell (special-position dedupe).
- Cromer-Mann (IT92) form factors `f(s)` for 98 neutral elements.
- CLI: `midas-hkls gen|info|list` (drop-in for `GetHKLList`).

### Optional: `[cif]` (gemmi) or `[cif-pure]` (pycifrw)
- `read_cif(path) -> Crystal` and `write_cif(crystal, path)` — full CIF1.1
  with anisotropic ADPs (gemmi) or isotropic-only (pycifrw fallback).
- Origin-choice and rhombohedral/hexagonal settings handled correctly via the
  resolved Hall symbol.

### Optional: `[torch]` — differentiable structure factors
- `structure_factors(crystal_t, hkl, *, anomalous=False)` returns complex
  `F_hkl` tensor, differentiable through:
  - atomic fractional coordinates,
  - occupancies,
  - isotropic B-factors and anisotropic U-tensors,
  - the six lattice parameters,
  - wavelength (when `anomalous=True`).
- `intensity_from_crystal(...)` and `powder_intensity(F, m, 2θ)` for
  Lorentz-polarization-weighted powder I_hkl.
- `anomalous_correction(elements, wavelength_A)` for f', f'' (Cromer-Liberman
  tables, 92 elements × 401 log-spaced energies, 100 eV–200 keV).
- Symmetry expansion is exact integer arithmetic; the autograd graph is
  rebuilt each forward call so gradients flow through ASU handles to UC atoms.

## Quick start

### 1. Generate HKL list
```python
from midas_hkls import SpaceGroup, Lattice, generate_hkls

sg  = SpaceGroup.from_number(225)              # CeO₂ / Cu / Au / NaCl  (Fm-3m)
lat = Lattice.for_system("cubic", a=5.411)
refs = generate_hkls(sg, lat, wavelength_A=0.173, two_theta_max_deg=15.0)
for r in refs:
    print(r.ring_nr, (r.h, r.k, r.l), r.d_spacing, r.two_theta_deg, r.multiplicity)
```

### 2. Read a structure & compute differentiable F_hkl
```python
import torch
from midas_hkls import read_cif, generate_hkls, structure_factors, intensity_from_crystal

xt = read_cif("ceo2.cif")
xt_t = xt.to_torch(requires_grad={"B_iso": True})         # mark B-factors trainable
refs = generate_hkls(xt.space_group, xt.lattice,
                     wavelength_A=0.173, two_theta_max_deg=20.0)

F, I = intensity_from_crystal(xt_t, refs, wavelength_A=0.173, polarization=0.5)

# Fit B-factors against an experimental I_obs (log-space residual)
opt = torch.optim.Adam([xt_t.B_iso_asu], lr=0.05)
for _ in range(300):
    opt.zero_grad()
    _, I = intensity_from_crystal(xt_t, refs, wavelength_A=0.173)
    loss = ((torch.log(I + 1e-3) - torch.log(I_obs + 1e-3)) ** 2).mean()
    loss.backward()
    opt.step()
```

### 3. Anomalous scattering (resonant f', f'')
```python
from midas_hkls import structure_factors, anomalous_correction

# Add f', f'' from Cromer-Liberman tables at the experimental wavelength
F_anomalous = structure_factors(xt_t, hkls,
                                wavelength_A=1.5418, anomalous=True)

# Or get f', f'' directly per element
fp, fpp = anomalous_correction(["Fe", "O"], wavelength_A=1.5418)
```

## Examples

Runnable notebooks live in [`examples/`](examples/). They use synthetic /
self-generated data only and run on CPU.

| Notebook | Topic | Extras |
|----------|-------|--------|
| [01_absorption.ipynb](examples/01_absorption.ipynb) | NIST mass / linear attenuation coefficients μ(element, λ); energy sweep, density override, differentiable in λ. | none (numpy) |
| [02_anomalous.ipynb](examples/02_anomalous.ipynb) | Cromer-Liberman resonant f', f''; effect on the complex structure factor; differentiable in wavelength. | `[torch]` |
| [03_cif_io.ipynb](examples/03_cif_io.ipynb) | CIF read / write round-trip and straight into HKL generation. | `[cif]` or `[cif-pure]` |

The notebooks are generated from `_build_*.py` scripts (content lives in
version-controlled Python):

```bash
cd examples
python _build_01_absorption.py && python _build_02_anomalous.py && python _build_03_cif_io.py
jupyter nbconvert --to notebook --execute --inplace 01_absorption.ipynb
```

## CLI

```
midas-hkls gen --sg 225 --lat 5.411 5.411 5.411 90 90 90 --wavelength 0.173 \
               --two-theta-max 15.0 -o ceo2.csv
midas-hkls info --sg "Fm-3m" --ops
midas-hkls list
```

## Install

```
pip install midas-hkls                       # base: numpy only
pip install "midas-hkls[cif]"                # + gemmi (CIF I/O)
pip install "midas-hkls[torch]"              # + torch (structure factors)
pip install "midas-hkls[all]"                # all of the above
```

## Parity & validation

- HKL generation: byte-for-byte parity vs. MIDAS's `GetHKLList` (sginfo) on
  CeO₂, LaB₆, Si, α-Fe, α-Ti, calcite, Pnma, P21/c.
- Structure factors: |F| matches `gemmi.StructureFactorCalculatorX` to
  <0.01% on CeO₂, Si, α-Fe, LaB₆, calcite (after applying gemmi's
  `change_occupancies_to_crystallographic` to align conventions).
- Anomalous f', f'' matches `gemmi.cromer_liberman` exactly on grid
  energies and within 0.05 between grid points.
- `torch.autograd.gradcheck` verified on |F|² w.r.t. lattice parameters and
  atomic positions in float64.

## Conventions

- Lengths in Å; angles in degrees.
- B-factor B = 8π² U (Å²); CIF U_ij stored in fractional basis.
- Wavelengths in Å; energies in eV (`E_eV = 12398.4 / λ_Å`).
- Symmetry operations stored as integer Seitz matrices over translation base
  STBF=12 — exact-arithmetic absence detection, no float fuzz.
- Equivalent HKLs include Friedel pairs (centric structure factor under X-ray
  Laue symmetry).

## Roadmap (post v0.4.0)

- Wyckoff special-position constraints during refinement.
- Aspherical / multipole atomic form factors.
- Magnetic structure factors.
- Expanded ion form factors (currently only neutral atoms).

## Origin

The 530-entry Hall-symbol table is extracted verbatim from sginfo
(© 1994-96 Ralf W. Grosse-Kunstleve, public domain). IT92 form factors and
Cromer-Liberman anomalous tables are exported from `gemmi` at packaging time
and ship as JSON.
