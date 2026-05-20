"""Builds example notebook 01: X-ray absorption (NIST mass attenuation)."""
from pathlib import Path
from _nb_helper import write_notebook


CELLS = [
    ("md", """\
# 01 — X-ray absorption coefficients (NIST mass attenuation)

`midas_hkls.absorption` provides the photon attenuation a beam experiences
passing through a material:

- `mass_attenuation_coefficient(element, λ)` → σ_mass [cm²/g]
- `linear_absorption_coefficient(element, λ)` → μ [cm⁻¹] = ρ · σ_mass

Values come from the NIST XCOM table, interpolated log-log in energy. This
module is **numpy-only** — it always runs, no optional extras. It is also
torch-differentiable through the wavelength when you pass a tensor, which is
useful for absorption-correction refinement.

Units follow the package convention: wavelengths in Å, μ in cm⁻¹.
"""),
    ("code", """\
import numpy as np

from midas_hkls.absorption import (
    available_elements_absorption,
    atomic_mass,
    element_density,
    mass_attenuation_coefficient,
    linear_absorption_coefficient,
)

elems = available_elements_absorption()
print(f"{len(elems)} elements tabulated; e.g. "
      f"Ti M={atomic_mass('Ti'):.3f} g/mol, ρ={element_density('Ti')} g/cm³")
"""),
    ("md", """\
## μ at a typical HEDM energy

CP-Ti at λ = 0.173 Å (≈ 71.7 keV). xraylib/NIST give μ/ρ ≈ 0.509 cm²/g, so
μ = ρ·σ_mass ≈ 2.30 cm⁻¹. Our log-log interpolation matches to ~1%.
"""),
    ("code", """\
lam = 0.173
sigma = mass_attenuation_coefficient("Ti", lam)
mu = linear_absorption_coefficient("Ti", lam)
print(f"σ_mass(Ti, {lam} Å) = {sigma:.4f} cm²/g")
print(f"μ(Ti, {lam} Å)      = {mu:.4f} cm⁻¹   (NIST ≈ 2.295)")
assert abs(mu - 2.295) < 0.05
"""),
    ("md", """\
## Energy dependence

Away from absorption edges, μ falls steeply with photon energy (rises with
wavelength). We sweep Ti across the HEDM band and confirm the monotonic trend.
"""),
    ("code", """\
energies_keV = np.array([100, 80, 60, 40, 20, 12], dtype=float)
lams = 12.398 / energies_keV                 # E[keV] = 12.398 / λ[Å]
mus = np.array([linear_absorption_coefficient("Ti", l) for l in lams])
for e, l, m in zip(energies_keV, lams, mus):
    print(f"  {e:5.0f} keV  (λ={l:.3f} Å)   μ = {m:8.3f} cm⁻¹")
# Larger λ (lower E) ⇒ larger μ
assert np.all(np.diff(mus) > 0)
"""),
    ("md", """\
## Density override

μ scales linearly with density. For porous or alloyed samples pass an explicit
`density_g_cm3` instead of the tabulated bulk value.
"""),
    ("code", """\
mu_full = linear_absorption_coefficient("Ti", lam)
mu_half = linear_absorption_coefficient("Ti", lam, density_g_cm3=element_density("Ti") / 2)
print(f"μ(full ρ) / μ(half ρ) = {mu_full / mu_half:.4f}  (expect 2.0)")
assert abs(mu_full / mu_half - 2.0) < 1e-6
"""),
    ("md", """\
## Differentiable in wavelength (torch)

When λ is a torch tensor, μ carries a gradient — handy if absorption is part
of a larger differentiable model. (If torch is not installed, this cell is
skipped.)
"""),
    ("code", """\
try:
    import torch
    lam_t = torch.tensor(0.173, dtype=torch.float64, requires_grad=True)
    mu_t = linear_absorption_coefficient("Ti", lam_t)
    mu_t.backward()
    print(f"μ = {float(mu_t):.4f} cm⁻¹   dμ/dλ = {float(lam_t.grad):.3f} cm⁻¹/Å")
    assert torch.isfinite(lam_t.grad)
except ImportError:
    print("(torch not installed — differentiable path skipped)")
"""),
    ("md", """\
That is the whole absorption surface: a NIST-backed μ(element, λ) that is a
plain float in the numpy path and a differentiable tensor in the torch path,
with an optional density override for non-bulk samples.
"""),
]


def main():
    out = Path(__file__).parent / "01_absorption.ipynb"
    write_notebook(out, CELLS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
