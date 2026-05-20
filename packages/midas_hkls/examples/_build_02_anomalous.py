"""Builds example notebook 02: anomalous (resonant) scattering f', f''."""
from pathlib import Path
from _nb_helper import write_notebook


CELLS = [
    ("md", """\
# 02 — Anomalous scattering (resonant f', f'')

Near an absorption edge the atomic form factor picks up a complex resonant
correction f = f₀(s) + f'(λ) + i·f''(λ). `midas_hkls.anomalous` provides the
Cromer-Liberman f', f'' from tables exported from `gemmi`, and
`structure_factors(..., anomalous=True)` folds them into the complex F_hkl.

This feature needs the **`[torch]`** extra (the structure-factor path is
PyTorch). If torch is missing, the notebook documents the API and stops.
"""),
    ("code", """\
import importlib.util

HAVE_TORCH = importlib.util.find_spec("torch") is not None
print("torch available:", HAVE_TORCH)
if not HAVE_TORCH:
    print("Install with: pip install 'midas-hkls[torch]' — feature deferred.")
"""),
    ("md", """\
## f', f'' per element at a chosen wavelength

`anomalous_correction(elements, λ)` returns matched arrays of f' and f''. At
Cu Kα (1.5418 Å), Fe sits just above its K-edge: f' is strongly negative and
f'' is large (~3.2), the classic resonant-Fe signature.
"""),
    ("code", """\
from midas_hkls.anomalous import (
    anomalous_correction, available_elements_anomalous,
    energy_eV_to_wavelength, wavelength_to_energy_eV,
)

print(f"{len(available_elements_anomalous())} elements tabulated")
lam_cu = 1.5418
fp, fpp = anomalous_correction(["Fe", "O"], lam_cu)
print(f"Cu Kα (λ={lam_cu} Å, E={wavelength_to_energy_eV(lam_cu):.0f} eV):")
print(f"  Fe:  f' = {float(fp[0]):+.3f}   f'' = {float(fpp[0]):.3f}")
print(f"  O :  f' = {float(fp[1]):+.3f}   f'' = {float(fpp[1]):.3f}")
"""),
    ("md", """\
## Effect on the structure factor

We build BCC α-Fe and compute |F| with and without the anomalous term. Turning
it on (a) gives F a non-zero imaginary part where f'' > 0, and (b) shrinks the
real part because Fe's f' is negative at Cu Kα.
"""),
    ("code", """\
if HAVE_TORCH:
    import torch
    from midas_hkls import (
        Atom, Crystal, Lattice, SpaceGroup, generate_hkls, structure_factors,
    )

    sg = SpaceGroup.from_number(229)                 # BCC
    lat = Lattice.for_system("cubic", a=2.8665)        # α-Fe
    xt = Crystal(lattice=lat, space_group=sg,
                 atoms=[Atom("Fe", (0.0, 0.0, 0.0), B_iso=0.35)])
    xt_t = xt.to_torch()

    refs = generate_hkls(sg, lat, wavelength_A=lam_cu, two_theta_max_deg=120.0)[:6]
    hkls = [(r.h, r.k, r.l) for r in refs]

    F_plain = structure_factors(xt_t, hkls).detach()
    F_anom = structure_factors(xt_t, hkls, wavelength_A=lam_cu, anomalous=True).detach()

    print("  hkl        |F| no-anom   |F| anom      Im(F) anom")
    for (h, k, l), fa, fb in zip(hkls, F_plain, F_anom):
        print(f"  {h}{k}{l}   {abs(fa):10.3f}   {abs(fb):10.3f}   {fb.imag:10.3f}")
    assert F_anom.imag.abs().max() > 0.5      # f'' creates a nonzero imag part
else:
    print("(deferred — torch not installed)")
"""),
    ("md", """\
## Differentiable in wavelength

Because the resonant term depends on λ, gradients flow from |F| back to the
wavelength — useful for energy-scan / DAFS-style refinements.
"""),
    ("code", """\
if HAVE_TORCH:
    import torch
    lam_t = torch.tensor(lam_cu, dtype=torch.float64, requires_grad=True)
    fp_t, fpp_t = anomalous_correction(["Fe"], lam_t)
    fp_t.sum().backward()
    print(f"f'(Fe) = {float(fp_t):+.3f}   df'/dλ = {float(lam_t.grad):+.3f} /Å")
    assert torch.isfinite(lam_t.grad).all()
else:
    print("(deferred — torch not installed)")
"""),
    ("md", """\
The resonant correction is exact against `gemmi.cromer_liberman` on grid
energies and within 0.05 between grid points, and it stays differentiable in λ
the whole way into the complex structure factor.
"""),
]


def main():
    out = Path(__file__).parent / "02_anomalous.ipynb"
    write_notebook(out, CELLS)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
