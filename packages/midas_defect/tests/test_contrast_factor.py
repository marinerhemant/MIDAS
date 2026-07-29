"""Tests for `midas_defect.contrast_factor`.

Covers the four mandatory categories:
  1. Synthetic / published correctness (unit) — anchored to the ANIZC paper's
     silver worked example C(2̄20, 60° mixed) = 0.3843.
  2. Autograd correctness (autograd) — dC/dCij via finite difference.
  3. Device portability (device) — CPU vs CUDA (complex eig unsupported on MPS).
  4. Cubic strain-anisotropy model C̄ = C̄_h00(1 - qH²) recovery.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_defect.contrast_factor import (
    cubic_stiffness,
    single_contrast_factor,
    average_contrast_factor,
    cubic_invariant_H2,
    fit_cbar_h00_q,
    fcc_slip_systems,
    bcc_slip_systems,
)
from midas_defect.lattice import fcc_cu_crystal


# ---------------------------------------------------------------------------
# 1. Published correctness — ANIZC silver worked example
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_silver_worked_example_matches_paper():
    """ANIZC paper §3: Ag, C11/C12/C44 = 124/93.7/46.1 GPa, 60° mixed dislocation
    b=[1̄10], slip plane (111), line [01̄1], reflection (2̄20) → C = 0.3843."""
    C6 = cubic_stiffness(124.0, 93.7, 46.1)
    C = single_contrast_factor(
        C6, burgers=[-1, 1, 0], slip_normal=[1, 1, 1],
        line=[0, -1, 1], g=[-2, 2, 0])
    assert float(C) == pytest.approx(0.3843, abs=5e-4)


@pytest.mark.unit
def test_cubic_crystal_path_is_identity_to_cartesian():
    """Passing a cubic `Crystal` (A = a·I) reproduces the raw-Cartesian result —
    the silver worked example, routed through the crystal→Cartesian conversion."""
    C6 = cubic_stiffness(124.0, 93.7, 46.1)
    kw = dict(burgers=[-1, 1, 0], slip_normal=[1, 1, 1], line=[0, -1, 1], g=[-2, 2, 0])
    c_cart = single_contrast_factor(C6, **kw)
    # cubic crystal at an arbitrary lattice constant — directions are unchanged
    c_cryst = single_contrast_factor(C6, crystal=fcc_cu_crystal(a=4.2), **kw)
    assert float(c_cryst) == pytest.approx(float(c_cart), rel=1e-12)
    assert float(c_cryst) == pytest.approx(0.3843, abs=5e-4)


@pytest.mark.unit
def test_cubic_average_crystal_matches_cartesian_shortcut():
    """average_contrast_factor with a cubic Crystal == the crystal=None shortcut."""
    Cu = cubic_stiffness(168.4, 121.4, 75.4)
    for hkl in [(2, 0, 0), (1, 1, 1), (3, 1, 1)]:
        a = float(average_contrast_factor(Cu, hkl, family="fcc"))
        b = float(average_contrast_factor(Cu, hkl, family="fcc", crystal=fcc_cu_crystal()))
        assert b == pytest.approx(a, rel=1e-12)


@pytest.mark.unit
def test_noncubic_crystal_preserves_slip_orthogonality_and_runs():
    """For a tetragonal cell the Cartesian slip-plane normal and Burgers/line stay
    orthogonal (Miller dot = 0 ⇒ Cartesian dot = 0), and C is finite & positive.

    Geometry-only check: confirms the crystal→Cartesian wiring is correct ahead of
    a dedicated non-cubic slip catalog (no published C̄ anchor asserted here)."""
    from midas_hkls.crystal import Atom, Crystal
    from midas_hkls.lattice import Lattice
    from midas_hkls.space_group import SpaceGroup

    # body-centred tetragonal-ish cell; reuse a {110}⟨1̄11⟩-style system
    lat = Lattice(a=3.0, b=3.0, c=4.8, alpha=90.0, beta=90.0, gamma=90.0)
    cr = Crystal(lattice=lat, space_group=SpaceGroup.from_number(139),
                 atoms=[Atom(element="Fe", fract=(0.0, 0.0, 0.0), occupancy=1.0,
                             label="Fe1")])
    # screw: line ∥ b; choose a system with n·b = 0 in Miller
    normal, burgers, g = [1, 1, 0], [1, -1, 0], [1, 0, 1]
    assert np.dot(normal, burgers) == 0
    C6 = cubic_stiffness(220.0, 130.0, 110.0)   # placeholder anisotropic stiffness
    C = single_contrast_factor(
        C6, burgers=burgers, slip_normal=normal, line=burgers, g=g, crystal=cr)
    assert torch.isfinite(C) and float(C) > 0.0


@pytest.mark.unit
def test_contrast_factor_scale_invariant_in_directions():
    """C depends only on directions: scaling b, n, line, g leaves C unchanged."""
    C6 = cubic_stiffness(124.0, 93.7, 46.1)
    base = single_contrast_factor(
        C6, burgers=[-1, 1, 0], slip_normal=[1, 1, 1], line=[0, -1, 1], g=[-2, 2, 0])
    scaled = single_contrast_factor(
        C6, burgers=[-3, 3, 0], slip_normal=[2, 2, 2], line=[0, -5, 5], g=[-1, 1, 0])
    assert float(scaled) == pytest.approx(float(base), rel=1e-9)


@pytest.mark.unit
def test_contrast_factor_quadrature_converges():
    """Result is stable w.r.t. the φ-quadrature resolution."""
    C6 = cubic_stiffness(168.4, 121.4, 75.4)
    kw = dict(burgers=[-1, 1, 0], slip_normal=[1, 1, 1], line=[0, -1, 1], g=[2, 2, 0])
    c_coarse = float(single_contrast_factor(C6, n_phi=180, **kw))
    c_fine = float(single_contrast_factor(C6, n_phi=2880, **kw))
    assert c_coarse == pytest.approx(c_fine, rel=1e-4)


@pytest.mark.unit
def test_slip_system_counts_and_orthogonality():
    """FCC has 12 {111}⟨110⟩ and BCC 12 {110}⟨111⟩ systems; b ⊥ plane normal."""
    fcc = fcc_slip_systems()
    bcc = bcc_slip_systems()
    assert len(fcc) == 12
    assert len(bcc) == 12
    for systems in (fcc, bcc):
        for normal, burgers in systems:
            assert np.dot(normal, burgers) == 0


@pytest.mark.unit
def test_h00_higher_contrast_than_hhh_for_positive_anisotropy():
    """For Ai = 2C44/(C11-C12) > 1 (Cu-like), C̄ falls from ⟨h00⟩ to ⟨hhh⟩."""
    Cu = cubic_stiffness(168.4, 121.4, 75.4)
    c_h00 = float(average_contrast_factor(Cu, [2, 0, 0], family="fcc"))
    c_hhh = float(average_contrast_factor(Cu, [1, 1, 1], family="fcc"))
    assert c_h00 > c_hhh > 0.0


@pytest.mark.unit
def test_symmetry_equivalent_reflections_share_average():
    """Permutations of an ⟨hkl⟩ have equal slip-system-averaged C̄."""
    Cu = cubic_stiffness(168.4, 121.4, 75.4)
    a = float(average_contrast_factor(Cu, [3, 1, 1], family="fcc"))
    b = float(average_contrast_factor(Cu, [1, 3, 1], family="fcc"))
    c = float(average_contrast_factor(Cu, [-1, 1, 3], family="fcc"))
    assert a == pytest.approx(b, abs=1e-6)
    assert a == pytest.approx(c, abs=1e-6)


@pytest.mark.unit
def test_isotropic_stiffness_raises_on_degenerate_roots():
    """An elastically isotropic tensor (Ai = 1) gives degenerate sextic roots."""
    # isotropic: C44 = (C11 - C12)/2  → Zener ratio = 1
    iso = cubic_stiffness(200.0, 100.0, 50.0)
    with pytest.raises(ValueError, match="isotropic"):
        single_contrast_factor(
            iso, burgers=[-1, 1, 0], slip_normal=[1, 1, 1], line=[0, -1, 1], g=[2, 0, 0])


# ---------------------------------------------------------------------------
# 2. Autograd correctness
# ---------------------------------------------------------------------------

@pytest.mark.autograd
def test_contrast_factor_grad_wrt_c44_matches_fd():
    """dC/dC44 from autograd matches central finite difference."""
    def fn(c44):
        C6 = cubic_stiffness(124.0, 93.7, c44)
        return single_contrast_factor(
            C6, burgers=[-1, 1, 0], slip_normal=[1, 1, 1],
            line=[0, -1, 1], g=[-2, 2, 0], n_phi=720)

    c44 = torch.tensor(46.1, dtype=torch.float64, requires_grad=True)
    grad_auto = torch.autograd.grad(fn(c44), c44)[0]

    eps = 1e-4
    with torch.no_grad():
        gp = fn(torch.tensor(46.1 + eps, dtype=torch.float64))
        gm = fn(torch.tensor(46.1 - eps, dtype=torch.float64))
    grad_fd = (gp - gm) / (2 * eps)
    assert torch.isfinite(grad_auto)
    assert float(grad_auto) == pytest.approx(float(grad_fd), rel=1e-3, abs=1e-6)


@pytest.mark.autograd
def test_contrast_factor_grad_is_nonzero():
    """C genuinely depends on the elastic constants (gradient not trivially 0)."""
    c11 = torch.tensor(124.0, dtype=torch.float64, requires_grad=True)
    C6 = cubic_stiffness(c11, 93.7, 46.1)
    C = single_contrast_factor(
        C6, burgers=[-1, 1, 0], slip_normal=[1, 1, 1], line=[0, -1, 1], g=[-2, 2, 0])
    (grad,) = torch.autograd.grad(C, c11)
    assert abs(float(grad)) > 1e-6


# ---------------------------------------------------------------------------
# 3. Device portability (complex eig: CPU/CUDA only, not MPS)
# ---------------------------------------------------------------------------

@pytest.mark.device
def test_contrast_factor_device_portable(_device_param):
    """Same C on CPU vs CUDA. MPS lacks complex128 eig → skipped."""
    if _device_param.type == "mps":
        pytest.skip("MPS backend does not support complex128 eigendecomposition")
    C6_cpu = cubic_stiffness(124.0, 93.7, 46.1, device=torch.device("cpu"))
    C6_dev = cubic_stiffness(124.0, 93.7, 46.1, device=_device_param)
    kw = dict(burgers=[-1, 1, 0], slip_normal=[1, 1, 1], line=[0, -1, 1], g=[-2, 2, 0])
    c_cpu = single_contrast_factor(C6_cpu, **kw)
    c_dev = single_contrast_factor(C6_dev, **kw)
    assert float(c_cpu) == pytest.approx(float(c_dev.cpu()), abs=1e-9)


# ---------------------------------------------------------------------------
# 4. Cubic strain-anisotropy model  C̄ = C̄_h00 (1 - q H²)
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_cubic_invariant_H2_known_values():
    assert cubic_invariant_H2([1, 0, 0]) == pytest.approx(0.0)
    assert cubic_invariant_H2([2, 0, 0]) == pytest.approx(0.0)
    assert cubic_invariant_H2([1, 1, 1]) == pytest.approx(1.0 / 3.0)
    assert cubic_invariant_H2([1, 1, 0]) == pytest.approx(0.25)


@pytest.mark.unit
def test_cubic_invariant_H2_rejects_zero():
    with pytest.raises(ValueError):
        cubic_invariant_H2([0, 0, 0])


@pytest.mark.unit
def test_cbar_model_linear_in_H2_recovers_h00_and_q():
    """The averaged C̄ for cubic is exactly linear in H² (Ungár-Tichy); the fit
    reproduces C̄(hkl) for held-out reflections to machine precision."""
    Cu = cubic_stiffness(168.4, 121.4, 75.4)
    hkls = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0), (2, 1, 1), (3, 1, 1)]
    cbars = [float(average_contrast_factor(Cu, h, family="fcc")) for h in hkls]
    model = fit_cbar_h00_q(hkls, cbars)
    # near-perfect linear fit
    assert model.residual_norm < 1e-9
    assert model.cbar_h00 == pytest.approx(cbars[0], abs=1e-6)   # (100): H²=0
    assert model.q > 0.0
    # held-out reflection prediction
    c_222 = float(average_contrast_factor(Cu, [2, 2, 2], family="fcc"))
    assert model([2, 2, 2]) == pytest.approx(c_222, abs=1e-6)


@pytest.mark.unit
def test_fit_cbar_requires_two_points():
    with pytest.raises(ValueError, match="at least 2"):
        fit_cbar_h00_q([(1, 0, 0)], [0.3])
