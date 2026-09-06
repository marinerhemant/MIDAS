"""Anisotropic-elasticity primitives, and the re-export contract they carry.

These primitives moved here out of `midas_defect.contrast_factor`. Three
packages import them across a boundary (midas_defect, midas_dfxm, and the
pf-/grain-ODF dev scripts), so the move is only safe as long as the old paths
resolve to the *same objects*. `test_reexport_*` are the guards on that; if
someone re-ports a copy back into midas_defect they fail immediately, which is
the whole point.
"""
import math

import pytest
import torch

from midas_ddd.elasticity import (
    _crystal_A_matrix,
    _rotate_tensor,
    _slip_frame,
    _stroh_eig,
    _to_cartesian,
    _voigt_to_tensor,
    bcc_slip_systems,
    cubic_stiffness,
    fcc_slip_systems,
    hexagonal_stiffness,
)

# Cu, GPa. Zener ratio 2A/(C11-C12) = 2*75.4/47.0 = 3.2 -- comfortably anisotropic,
# which _stroh_eig requires (it refuses the near-isotropic degenerate limit).
CU = (168.4, 121.4, 75.4)


# --------------------------------------------------------------------------
# Stiffness construction
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_cubic_stiffness_symmetry_and_entries():
    C6 = cubic_stiffness(*CU)
    assert C6.shape == (6, 6)
    assert torch.allclose(C6, C6.T)
    c11, c12, c44 = CU
    assert float(C6[0, 0]) == pytest.approx(c11)
    assert float(C6[0, 1]) == pytest.approx(c12)
    assert float(C6[3, 3]) == pytest.approx(c44)
    # Cubic has no coupling between normal and shear blocks.
    assert torch.count_nonzero(C6[:3, 3:]) == 0


@pytest.mark.unit
def test_hexagonal_stiffness_c66_constraint():
    # Ti, GPa.
    C6 = hexagonal_stiffness(162.4, 92.0, 69.0, 180.7, 46.7)
    assert torch.allclose(C6, C6.T)
    # The defining hexagonal relation.
    assert float(C6[5, 5]) == pytest.approx((162.4 - 92.0) / 2.0)
    assert float(C6[2, 2]) == pytest.approx(180.7)


@pytest.mark.unit
def test_cubic_stiffness_is_differentiable_in_its_constants():
    c11 = torch.tensor(168.4, dtype=torch.float64, requires_grad=True)
    C6 = cubic_stiffness(c11, 121.4, 75.4)
    C6.sum().backward()
    # c11 appears on the three diagonal normal entries.
    assert float(c11.grad) == pytest.approx(3.0)


# --------------------------------------------------------------------------
# Voigt expansion and rotation
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_voigt_to_tensor_recovers_voigt_entries():
    C6 = cubic_stiffness(*CU)
    C4 = _voigt_to_tensor(C6)
    assert C4.shape == (3, 3, 3, 3)
    assert float(C4[0, 0, 0, 0]) == pytest.approx(CU[0])   # C1111 = C11
    assert float(C4[0, 0, 1, 1]) == pytest.approx(CU[1])   # C1122 = C12
    assert float(C4[1, 2, 1, 2]) == pytest.approx(CU[2])   # C2323 = C44
    # Minor symmetries.
    assert torch.allclose(C4, C4.permute(1, 0, 2, 3))
    assert torch.allclose(C4, C4.permute(0, 1, 3, 2))


@pytest.mark.unit
def test_rotate_tensor_by_identity_is_a_noop():
    C4 = _voigt_to_tensor(cubic_stiffness(*CU))
    eye = torch.eye(3, dtype=C4.dtype)
    assert torch.allclose(_rotate_tensor(C4, eye), C4)


@pytest.mark.unit
def test_rotate_tensor_preserves_cubic_invariance_under_90deg():
    """A 90 deg rotation about z is a cubic symmetry operation: C is unchanged."""
    C4 = _voigt_to_tensor(cubic_stiffness(*CU))
    R = torch.tensor([[0.0, -1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0]], dtype=C4.dtype)
    assert torch.allclose(_rotate_tensor(C4, R), C4, atol=1e-10)


# --------------------------------------------------------------------------
# Slip frame
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_slip_frame_is_orthonormal_and_right_handed():
    line = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)
    normal = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    M = _slip_frame(line, normal)
    assert torch.allclose(M @ M.T, torch.eye(3, dtype=M.dtype), atol=1e-12)
    assert float(torch.linalg.det(M)) == pytest.approx(1.0)
    # e3 is the unit line; e2 is the unit normal.
    assert torch.allclose(M[2], line / torch.linalg.norm(line))
    assert torch.allclose(M[1], normal / torch.linalg.norm(normal))


@pytest.mark.unit
def test_slip_frame_gram_schmidts_a_non_orthogonal_normal():
    """A normal with a component along the line is projected, not accepted."""
    line = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    normal = torch.tensor([0.3, 1.0, 0.0], dtype=torch.float64)   # not perpendicular
    M = _slip_frame(line, normal)
    assert float(M[1] @ M[2]) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Stroh sextic solution
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_stroh_eig_returns_three_roots_with_positive_imaginary_part():
    C6 = cubic_stiffness(*CU)
    line = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64)
    normal = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    M = _slip_frame(line, normal)
    C4 = _rotate_tensor(_voigt_to_tensor(C6), M)
    p, A, B = _stroh_eig(C4)
    assert p.shape == (3,) and A.shape == (3, 3) and B.shape == (3, 3)
    assert bool((p.imag > 0).all())


@pytest.mark.unit
def test_stroh_eig_normalisation_2aTb_equals_one():
    """Stroh orthonormality: 2 a_a^T b_a = 1 for each root (bilinear, no conjugate)."""
    C6 = cubic_stiffness(*CU)
    M = _slip_frame(torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64),
                    torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64))
    C4 = _rotate_tensor(_voigt_to_tensor(C6), M)
    _, A, B = _stroh_eig(C4)
    for k in range(3):
        assert complex(2.0 * (A[:, k] @ B[:, k])) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.unit
def test_stroh_eig_refuses_the_isotropic_degenerate_limit():
    """Zener ratio 1 collapses the roots to p = i; the simple-eigenvector path
    is unreliable there and must raise rather than return quiet garbage."""
    c11, c12 = 200.0, 100.0
    c44_iso = (c11 - c12) / 2.0            # Zener ratio exactly 1
    C4 = _voigt_to_tensor(cubic_stiffness(c11, c12, c44_iso))
    with pytest.raises(ValueError, match="isotropic|degenerate"):
        _stroh_eig(C4)


# --------------------------------------------------------------------------
# Crystal -> Cartesian
# --------------------------------------------------------------------------

class _FakeLattice:
    def __init__(self, a, b, c, alpha, beta, gamma):
        self.a, self.b, self.c = a, b, c
        self.alpha, self.beta, self.gamma = alpha, beta, gamma


class _FakeCrystal:
    def __init__(self, lattice):
        self.lattice = lattice


@pytest.mark.unit
def test_crystal_A_matrix_is_diagonal_for_cubic():
    xtal = _FakeCrystal(_FakeLattice(3.615, 3.615, 3.615, 90.0, 90.0, 90.0))
    A = _crystal_A_matrix(xtal, dtype=torch.float64, device=torch.device("cpu"))
    assert torch.allclose(A, torch.diag(torch.full((3,), 3.615, dtype=torch.float64)),
                          atol=1e-12)


@pytest.mark.unit
def test_to_cartesian_cubic_identity_up_to_scale():
    """For cubic, direct and reciprocal maps agree up to the a^2 scale -- the
    property that lets cubic callers pass raw Miller indices."""
    a = 3.615
    A = torch.diag(torch.full((3,), a, dtype=torch.float64))
    v = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    direct = _to_cartesian(v, A, "direct")
    recip = _to_cartesian(v, A, "reciprocal")
    assert torch.allclose(direct, v * a)
    assert torch.allclose(recip, v / a)
    assert torch.allclose(direct, recip * a * a)


@pytest.mark.unit
def test_to_cartesian_rejects_an_unknown_kind():
    A = torch.eye(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="direct.*reciprocal"):
        _to_cartesian(torch.zeros(3, dtype=torch.float64), A, "sideways")


# --------------------------------------------------------------------------
# Slip-system tables
# --------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("fn,n", [(fcc_slip_systems, 12), (bcc_slip_systems, 12)])
def test_slip_system_counts(fn, n):
    assert len(fn()) == n


@pytest.mark.unit
@pytest.mark.parametrize("fn", [fcc_slip_systems, bcc_slip_systems])
def test_every_burgers_vector_lies_in_its_slip_plane(fn):
    for normal, burgers in fn():
        assert sum(x * y for x, y in zip(normal, burgers)) == 0


@pytest.mark.unit
def test_slip_systems_are_sign_deduplicated():
    """b and -b are the same physical system and must not both appear."""
    systems = fcc_slip_systems()
    keys = {(n, tuple(sorted((b, tuple(-x for x in b))))) for n, b in systems}
    assert len(keys) == len(systems)


# --------------------------------------------------------------------------
# The re-export contract
# --------------------------------------------------------------------------

@pytest.mark.unit
def test_reexport_midas_defect_contrast_factor_is_the_same_object():
    cf = pytest.importorskip("midas_defect.contrast_factor")
    import midas_ddd.elasticity as el

    for name in ("cubic_stiffness", "fcc_slip_systems", "bcc_slip_systems",
                 "_stroh_eig", "_slip_frame", "_voigt_to_tensor",
                 "_rotate_tensor", "_crystal_A_matrix", "_to_cartesian",
                 "_gen_slip_systems"):
        assert getattr(cf, name) is getattr(el, name), (
            f"{name} in midas_defect.contrast_factor is NOT the midas_ddd object "
            "-- someone re-ported a copy; there must be exactly one definition."
        )


@pytest.mark.unit
def test_reexport_midas_defect_contrast_factor_hex_is_the_same_object():
    cfh = pytest.importorskip("midas_defect.contrast_factor_hex")
    import midas_ddd.elasticity as el

    assert cfh.hexagonal_stiffness is el.hexagonal_stiffness


@pytest.mark.unit
def test_reexport_midas_dfxm_dislocation_is_the_same_object():
    """midas_dfxm.dislocation imports these across a package boundary and has
    carried a 'do NOT re-port' comment since before the move."""
    disl = pytest.importorskip("midas_dfxm.dislocation")
    import midas_ddd.elasticity as el

    for name in ("cubic_stiffness", "fcc_slip_systems", "_stroh_eig"):
        assert getattr(disl, name) is getattr(el, name)
