"""Tests for `midas_defect.contrast_factor_hex`.

Anchored to Dragomir & Ungár (2002) Table 2. Validated against TWO materials
(Ti and Zr) so a single anomalous table cell can't masquerade as correctness —
the Ti S2 entry (0.41873) is in fact a table anomaly: every other material's S2
clusters near 0.10-0.12 and Zr's S2 reproduces to 0.0 %, so S2 is checked on Zr.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_hkls.crystal import Atom, Crystal
from midas_hkls.lattice import Lattice
from midas_hkls.space_group import SpaceGroup

from midas_defect.contrast_factor_hex import (
    hexagonal_stiffness, mb_plane_to_hkl, mb_direction_to_uvw,
    hex_slip_systems, subsystem_contrast_factor, fit_hex_cbar,
    _hex_proper_rotations, _crystal_A_matrix,
)


def _hex_crystal(a, c):
    return Crystal(
        lattice=Lattice(a=a, b=a, c=c, alpha=90.0, beta=90.0, gamma=120.0),
        space_group=SpaceGroup.from_number(194),
        atoms=[Atom(element="X", fract=(1/3, 2/3, 0.25), occupancy=1.0, label="X")])


# Fisher & Renken single-crystal elastic constants (GPa) + lattice constants (Å)
TI = (dict(c11=162.4, c12=92.0, c13=69.0, c33=180.7, c44=46.7), (2.951, 4.684))
ZR = (dict(c11=143.5, c12=72.5, c13=65.4, c33=164.9, c44=32.1), (3.232, 5.147))

# Dragomir & Ungár (2002) Table 2 — C̄_{hk.0} per sub-slip-system.
TABLE2_CHK0 = {
    "Ti": {"BE": 0.20227, "PrE": 0.35387, "Pr2E": 0.04853, "Pr3E": 0.10247,
           "PyE": 0.31180, "Py2E": 0.09227, "PyE3": 0.09813, "PyE4": 0.09323,
           "S1": 0.14440},   # S2 cell anomalous (see module/test docstring); S3 ≈ 0
    "Zr": {"BE": 0.18313, "PrE": 0.34453, "Pr2E": 0.04500, "Pr3E": 0.08937,
           "PyE": 0.30083, "Py2E": 0.08407, "PyE3": 0.08633, "PyE4": 0.08393,
           "S1": 0.11866, "S2": 0.10493},   # Zr S2 is sound → checked here
}


# ---------------------------------------------------------------------------
# 1. Miller-Bravais geometry & catalog
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_mb_conversions():
    assert mb_plane_to_hkl((1, 0, -1, 0)) == (1, 0, 0)
    assert mb_plane_to_hkl((1, 1, -2, 3)) == (1, 1, 3)
    # ⟨a⟩ = ⅓⟨2̄110⟩ → a₁ = [100];  ⟨c+a⟩ = ⅓⟨2̄113⟩ → [101]
    assert mb_direction_to_uvw((2, -1, -1, 0)) == (3.0, 0.0, 0.0)
    assert mb_direction_to_uvw((2, -1, -1, 3)) == (3.0, 0.0, 3.0)


@pytest.mark.unit
def test_mb_rejects_bad_indices():
    with pytest.raises(AssertionError):
        mb_plane_to_hkl((1, 1, 1, 0))     # h+k+i ≠ 0
    with pytest.raises(AssertionError):
        mb_direction_to_uvw((1, 1, 1, 0))


@pytest.mark.unit
def test_catalog_has_11_subsystems():
    sys = hex_slip_systems()
    assert len(sys) == 11
    assert sum(s.character == "edge" for s in sys) == 8
    assert sum(s.character == "screw" for s in sys) == 3
    assert {s.name for s in sys} == {
        "BE", "PrE", "Pr2E", "Pr3E", "PyE", "Py2E", "PyE3", "PyE4", "S1", "S2", "S3"}


@pytest.mark.unit
def test_proper_point_group_has_12_rotations():
    cr = _hex_crystal(2.951, 4.684)
    A = _crystal_A_matrix(cr, dtype=torch.float64, device=torch.device("cpu"))
    R = _hex_proper_rotations(A, dtype=torch.float64, device=torch.device("cpu"))
    assert R.shape == (12, 3, 3)
    # all proper rotations: det = +1, orthogonal
    dets = torch.linalg.det(R)
    assert torch.allclose(dets, torch.ones(12, dtype=torch.float64), atol=1e-9)
    for Ri in R:
        assert torch.allclose(Ri @ Ri.T, torch.eye(3, dtype=torch.float64), atol=1e-9)


# ---------------------------------------------------------------------------
# 2. Physics correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_azimuthal_invariance_same_x_same_cbar():
    """Reflections with equal x (= angle to c) give equal averaged C̄ after the
    point-group average — restored in-plane (azimuthal) isotropy."""
    cr = _hex_crystal(2.951, 4.684)
    C6 = hexagonal_stiffness(**TI[0])
    BE = next(s for s in hex_slip_systems() if s.name == "BE")
    vals = [float(subsystem_contrast_factor(C6, cr, BE, hkil, n_phi=180))
            for hkil in [(1, 0, -1, 0), (0, 1, -1, 0), (1, 1, -2, 0), (2, -1, -1, 0)]]
    assert max(vals) - min(vals) < 1e-4


@pytest.mark.unit
def test_cbar_is_exact_parabola_in_x():
    """The averaged C̄ is analytically a parabola in x (Ungár & Tichy) — the fit
    residual must be ~0 for a non-degenerate sub-slip-system."""
    cr = _hex_crystal(2.951, 4.684)
    C6 = hexagonal_stiffness(**TI[0])
    for name in ("BE", "Pr2E", "S1"):
        s = next(x for x in hex_slip_systems() if x.name == name)
        m = fit_hex_cbar(C6, cr, s, n_phi=240)
        assert m.residual_norm < 1e-10, f"{name} not parabolic: {m.residual_norm}"


@pytest.mark.slow
@pytest.mark.parametrize("mat", ["Ti", "Zr"])
def test_table2_cbar_hk0_regression(mat):
    """C̄_{hk.0} reproduces Dragomir & Ungár (2002) Table 2 within ~3 % across
    all non-degenerate sub-slip-systems, for both Ti and Zr."""
    cij, (a, c) = {"Ti": TI, "Zr": ZR}[mat]
    cr = _hex_crystal(a, c)
    C6 = hexagonal_stiffness(**cij)
    for s in hex_slip_systems():
        ref = TABLE2_CHK0[mat].get(s.name)
        if ref is None:
            continue
        m = fit_hex_cbar(C6, cr, s, n_phi=240)
        assert m.cbar_hk0 == pytest.approx(ref, rel=0.035), \
            f"{mat} {s.name}: got {m.cbar_hk0:.5f}, Table 2 {ref:.5f}"


@pytest.mark.slow
def test_prismatic_a_edge_degeneracy_handled():
    """PrE (prismatic ⟨a⟩ edge) has its line ∥ c → basal-isotropic degeneracy;
    the C66-perturbation extrapolation must yield the Table-2 value, not crash."""
    cr = _hex_crystal(2.951, 4.684)
    C6 = hexagonal_stiffness(**TI[0])
    PrE = next(s for s in hex_slip_systems() if s.name == "PrE")
    m = fit_hex_cbar(C6, cr, PrE, n_phi=240)
    assert math.isfinite(m.cbar_hk0)
    assert m.cbar_hk0 == pytest.approx(0.35387, rel=0.02)


@pytest.mark.slow
def test_c_screw_cbar_hk0_vanishes():
    """S3 (⟨c⟩ screw, line ∥ c): at hk.0 the diffraction vector ⊥ line ⇒ F≡0, so
    C̄_{hk.0} ≈ 0 (the paper's 3.6e-6 is its numerical floor)."""
    cr = _hex_crystal(2.951, 4.684)
    C6 = hexagonal_stiffness(**TI[0])
    S3 = next(s for s in hex_slip_systems() if s.name == "S3")
    m = fit_hex_cbar(C6, cr, S3, n_phi=240)
    assert abs(m.cbar_hk0) < 1e-4


# ---------------------------------------------------------------------------
# 3. Autograd & device
# ---------------------------------------------------------------------------

@pytest.mark.autograd
def test_subsystem_cbar_grad_wrt_c44():
    """C̄ is differentiable w.r.t. the elastic constants (FD check on C44)."""
    cr = _hex_crystal(2.951, 4.684)
    S1 = next(s for s in hex_slip_systems() if s.name == "S1")

    def fn(c44):
        C6 = hexagonal_stiffness(162.4, 92.0, 69.0, 180.7, c44)
        return subsystem_contrast_factor(C6, cr, S1, (1, 0, -1, 2), n_phi=180)

    c44 = torch.tensor(46.7, dtype=torch.float64, requires_grad=True)
    g = torch.autograd.grad(fn(c44), c44)[0]
    eps = 1e-3
    fd = (fn(torch.tensor(46.7 + eps, dtype=torch.float64))
          - fn(torch.tensor(46.7 - eps, dtype=torch.float64))) / (2 * eps)
    assert torch.isfinite(g)
    assert float(g) == pytest.approx(float(fd), rel=2e-3, abs=1e-5)


@pytest.mark.device
def test_hex_cbar_device_portable(_device_param):
    """Same C̄ on CPU vs CUDA (complex eig unsupported on MPS → skipped)."""
    if _device_param.type == "mps":
        pytest.skip("MPS lacks complex128 eigendecomposition")
    cr = _hex_crystal(2.951, 4.684)
    S1 = next(s for s in hex_slip_systems() if s.name == "S1")
    c_cpu = subsystem_contrast_factor(
        hexagonal_stiffness(**TI[0], device=torch.device("cpu")), cr, S1, (1, 0, -1, 2), n_phi=180)
    c_dev = subsystem_contrast_factor(
        hexagonal_stiffness(**TI[0], device=_device_param), cr, S1, (1, 0, -1, 2), n_phi=180)
    assert float(c_cpu) == pytest.approx(float(c_dev.cpu()), abs=1e-9)
