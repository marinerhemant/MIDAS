#!/usr/bin/env python
"""Cross-validate midas_dt's closed-form pole-figure operator against TexTOM.

TexTOM (Frewein et al., IUCrJ 11, 2024; code Zenodo 10.5281/zenodo.12543638) is an
independent, peer-reviewed texture-tomography implementation. It computes the
pole-figure projection **numerically** -- a diffractlet is
``dlets[l] = (sHSHs[l] * dV) @ Isc`` (``src/model.py:476``), i.e. a quadrature over
an orientation grid with the SO(3) Haar measure. :mod:`midas_dt.gsh` computes the
same integral in closed form. So the two are directly comparable, and TexTOM
supplies the third-party half: its orientation grid, its Haar volume element and
its cubic fundamental zone, none of which we wrote.

This is the third of three validation routes for the operator; the other two --
the closed form against brute-force fibre quadrature (5.6e-15) and against our own
Monte-Carlo pole figure -- are in ``tests/test_gsh.py`` and
``tests/test_texture_kernel.py`` and run without any external checkout. This
script needs TexTOM on disk, which is why it is a script and not a test.

Three checks:

1. **Haar measure.** TexTOM uses ``dV = sin^2(Ome/2) sin(Tta) dchi^3``
   (``src/model.py:131``, citing Mason/Patala arXiv 2019 App. C).
   ``texture_kernel.radial_coeffs`` integrates against ``p(omega) = (1 - cos
   omega)/pi``. These agree only if ``2 sin^2(w/2) == 1 - cos w``, which is an
   identity -- but the NORMALISATION is the part that silently breaks kernels, so
   both are integrated numerically rather than argued about.

2. **Cubic fundamental zone.** ``ressources/symmetries.py`` case ``'432'`` imposes
   the 6 cube-face constraints and SEVEN octahedral constraints. The eighth,
   ``-R1 + R2 - R3 <= 1``, is absent. If that omission is real rather than
   redundant, their zone is too large and their ``V_fz`` is overestimated. Measured
   against the analytic 1/24 of SO(3), against the fully-constrained zone, and by
   an independent test of whether the accepted set is a fundamental domain at all
   (a correct one contains exactly one of each orientation's 24 symmetry copies).

3. **Pole figure.** Our cubic-symmetrised kernel ODF is integrated over TexTOM's
   grid with TexTOM's ``dV``, {111} normals are accumulated into caps, and the
   result is compared against our closed-form operator. The ODF on TexTOM's grid
   is evaluated **directly from the radial profile** -- no harmonics anywhere on
   that side -- so this is a genuine cross-check of the harmonic expansion rather
   than the same algebra twice.

Usage::

    KMP_DUPLICATE_LIB_OK=TRUE python validate_gsh_vs_textom.py <path-to-textom>

where ``<path-to-textom>`` contains ``ressources/symmetries.py``.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from midas_dt.gsh import CubicGSH, cubic_rotations, hkl_family
from midas_dt.texture_kernel import (
    kappa_for_halfwidth,
    kernel_profile,
    kernel_to_gsh,
    radial_coeffs,
)


def load_textom_symmetries(root: Path):
    """Import TexTOM's ``ressources/symmetries.py`` directly from disk."""
    path = root / "ressources" / "symmetries.py"
    spec = importlib.util.spec_from_file_location("textom_sym", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def textom_grid(dchi_deg: float):
    """TexTOM's orientation grid, reproduced from ``model.py:_define_angles``."""
    dchi = np.radians(dchi_deg)
    ome = np.linspace(dchi / 2, np.pi - dchi / 2, int(np.pi / dchi), endpoint=True)
    tta = np.linspace(dchi / 2, np.pi - dchi / 2, int(np.pi / dchi), endpoint=True)
    phi = np.linspace(0, 2 * np.pi, int(2 * np.pi / dchi), endpoint=False)
    TTA, PHI, OME = np.meshgrid(tta, phi, ome)
    Gc = np.column_stack((OME.ravel(), TTA.ravel(), PHI.ravel()))
    dV = np.sin(Gc[:, 0] / 2) ** 2 * np.sin(Gc[:, 1]) * dchi ** 3
    return Gc, dV


def check_haar() -> bool:
    print("1. HAAR MEASURE")
    dchi = np.radians(2.0)
    w = np.linspace(dchi / 2, np.pi - dchi / 2, int(np.pi / dchi))
    tex = (np.sin(w / 2) ** 2 * dchi).sum() * 4 * np.pi
    ours = ((1 - np.cos(w)) / np.pi * dchi).sum()
    analytic = 2 * np.pi ** 2            # omega in [0, pi], TexTOM's convention
    print(f"   TexTOM total SO(3) volume  {tex:.6f}  (analytic {analytic:.6f}, "
          f"rel dev {abs(tex - analytic) / analytic:.2e})")
    print(f"   our p(omega) integrates to {ours:.6f}  (must be 1.0, "
          f"rel dev {abs(ours - 1):.2e})")
    ok = abs(tex / analytic - 1) < 1e-3 and abs(ours - 1) < 1e-3
    print(f"   -> {'AGREE' if ok else 'DISAGREE'}: both are the sin^2(w/2) Haar "
          f"density, differing only by a constant\n")
    return ok


def check_fundamental_zone(sym, n_random=4_000_000, n_probe=20_000) -> bool:
    print("2. CUBIC FUNDAMENTAL ZONE  (TexTOM ressources/symmetries.py, '432')")
    rots = Rotation.random(n_random, rng=np.random.default_rng(0))
    rv = rots.as_rotvec()
    ang = np.linalg.norm(rv, axis=1)
    axis = rv / np.maximum(ang, 1e-15)[:, None]
    g = np.column_stack((ang, np.arccos(np.clip(axis[:, 2], -1, 1)),
                         np.arctan2(axis[:, 1], axis[:, 0])))

    fz = sym.zone("432", g)
    frac = float(fz.mean())
    print(f"   uniform-random acceptance: {frac:.6f}   (exact FZ = 1/24 = "
          f"{1 / 24:.6f}), ratio {frac * 24:.4f}")

    tan_half = np.tan(g[:, 0] / 2)
    R1 = tan_half * np.sin(g[:, 1]) * np.cos(g[:, 2])
    R2 = tan_half * np.sin(g[:, 1]) * np.sin(g[:, 2])
    R3 = tan_half * np.cos(g[:, 1])
    missing = (-R1 + R2 - R3) <= 1.0
    full = fz & missing
    print(f"   with the missing 8th octahedral constraint -R1+R2-R3 <= 1 added: "
          f"{full.mean():.6f}, ratio {full.mean() * 24:.4f}")
    leak = (fz & ~missing).sum() / max(fz.sum(), 1)
    print(f"   orientations TexTOM accepts that the full zone rejects: "
          f"{leak * 100:.2f}% of its zone")

    # Independent check: is the accepted set a fundamental domain at all? Count how
    # many of an orientation's 24 symmetry copies land inside the claimed zone. A
    # correct FZ gives exactly 1 almost everywhere.
    probe = Rotation.random(n_probe, rng=np.random.default_rng(1))
    counts = np.zeros(len(probe), dtype=int)
    for s in cubic_rotations():
        rv2 = (probe * s).as_rotvec()
        a2 = np.linalg.norm(rv2, axis=1)
        ax2 = rv2 / np.maximum(a2, 1e-15)[:, None]
        g2 = np.column_stack((a2, np.arccos(np.clip(ax2[:, 2], -1, 1)),
                              np.arctan2(ax2[:, 1], ax2[:, 0])))
        counts += sym.zone("432", g2).astype(int)
    vals, cnt = np.unique(counts, return_counts=True)
    print("   copies of a random orientation inside the claimed zone "
          "(a correct FZ gives exactly 1):")
    for v, c in zip(vals, cnt):
        print(f"      {v} copies : {c / len(probe) * 100:6.2f}% of orientations")
    ok = abs(frac * 24 - 1.0) < 0.01
    print(f"   -> TexTOM cubic zone is {'CORRECT' if ok else 'TOO LARGE'}\n")
    return ok


def check_pole_figure(l_max: int, hw_deg: float, dchi_deg: float,
                      cap_deg: float = 10.0, n_dir: int = 40) -> bool:
    print(f"3. POLE FIGURE  L={l_max}, kernel halfwidth {hw_deg} deg, "
          f"TexTOM grid dchi={dchi_deg} deg")
    kappa = kappa_for_halfwidth(hw_deg)
    basis = CubicGSH(l_max)
    ax = np.array([0.3, -0.7, 0.65])
    centre = Rotation.from_rotvec(
        (np.radians(37.0) * ax / np.linalg.norm(ax))[None, :])
    T = kernel_to_gsh(basis, centre, radial_coeffs(l_max, kappa))[:, 0]

    Gc, dV = textom_grid(dchi_deg)
    axis = np.stack([np.sin(Gc[:, 1]) * np.cos(Gc[:, 2]),
                     np.sin(Gc[:, 1]) * np.sin(Gc[:, 2]),
                     np.cos(Gc[:, 1])], axis=1)
    grid = Rotation.from_rotvec(axis * Gc[:, 0][:, None])
    print(f"   grid: {len(Gc)} orientations over all of SO(3), "
          f"sum dV = {dV.sum():.4f}")

    # ODF on the grid, evaluated DIRECTLY from the radial profile: no harmonics
    # anywhere on this side of the comparison.
    c_inv = centre.inv()
    odf = np.zeros(len(grid))
    for s in cubic_rotations():
        odf += kernel_profile((c_inv * (grid * s)).magnitude(), kappa)
    odf /= 24.0
    # Normalise to Haar mean 1 using TEXTOM'S OWN dV. radial_coeffs divides the
    # profile by exactly this factor; leaving it out makes the quadrature low by
    # the Haar mean of cos^2k(w/2) (~0.03 at kappa=5.6) while leaving the SHAPE
    # untouched -- it appears as slope ~0.03 at corr ~1, which is easy to
    # misdiagnose as a broken operator.
    haar_mean = float((odf @ dV) / dV.sum())
    odf = odf / haar_mean
    print(f"   kernel Haar mean over TexTOM's measure {haar_mean:.6f} "
          f"(divided out, as radial_coeffs does)")

    fam = hkl_family((1, 1, 1))
    ns = np.einsum("kab,mb->kma", grid.as_matrix(), fam).reshape(-1, 3)
    wts = np.repeat(odf * dV, fam.shape[0])
    uni = np.repeat(dV, fam.shape[0])

    ys = Rotation.random(n_dir, rng=np.random.default_rng(9)).as_matrix()[:, :, 0]
    cos_cap = np.cos(np.radians(cap_deg))
    c_unif = np.zeros_like(T)
    c_unif[0] = T[0]
    quad, model = [], []
    for y in ys:
        m = (ns @ y) >= cos_cap
        quad.append(wts[m].sum() / uni[m].sum())          # ratio to uniform
        row = basis.pole_row(fam, y[None, :])
        model.append(float((row @ T).real / (row @ c_unif).real))
    quad, model = np.array(quad), np.array(model)
    corr = float(np.corrcoef(quad, model)[0, 1])
    slope = float(np.polyfit(model, quad, 1)[0])
    print(f"   contrast (range of the pole figure) {np.ptp(quad):.3f}")
    print(f"   corr {corr:.6f}   slope {slope:.4f}   "
          f"max|err| {np.abs(quad - model).max():.4f}")
    if np.ptp(quad) < 0.05:
        print("   -> VACUOUS: pole figure too flat to discriminate\n")
        return False
    ok = corr > 0.99 and 0.9 < slope < 1.1
    print(f"   -> closed form vs TexTOM-grid quadrature: "
          f"{'AGREE' if ok else 'DISAGREE'}\n")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("textom_root", type=Path,
                    help="TexTOM source directory (contains ressources/symmetries.py)")
    ap.add_argument("--quick", action="store_true",
                    help="coarser grids; for checking the script runs at all")
    args = ap.parse_args(argv)

    if not (args.textom_root / "ressources" / "symmetries.py").exists():
        print(f"TexTOM source not found under {args.textom_root}\n"
              f"Expected {args.textom_root / 'ressources' / 'symmetries.py'}\n"
              "Get it from Zenodo 10.5281/zenodo.12543638.", file=sys.stderr)
        return 2
    print(f"TexTOM source: {args.textom_root}\n")

    sym = load_textom_symmetries(args.textom_root)
    dchi = 6.0 if args.quick else 3.0
    n_rand = 200_000 if args.quick else 4_000_000
    n_probe = 2_000 if args.quick else 20_000

    r1 = check_haar()
    r2 = check_fundamental_zone(sym, n_random=n_rand, n_probe=n_probe)
    r3 = check_pole_figure(6, 40.0, dchi)
    r4 = check_pole_figure(12, 25.0, dchi)

    print("=" * 70)
    print(f"Haar measure        : {'agree' if r1 else 'DISAGREE'}")
    print(f"TexTOM cubic FZ     : {'correct' if r2 else 'TOO LARGE (their zone)'}")
    print(f"pole figure L=6     : {'agree' if r3 else 'DISAGREE'}")
    print(f"pole figure L=12    : {'agree' if r4 else 'DISAGREE'}")
    # r2 is a statement about TexTOM, not about us, so it does not gate our result.
    return 0 if (r1 and r3 and r4) else 1


if __name__ == "__main__":
    raise SystemExit(main())
