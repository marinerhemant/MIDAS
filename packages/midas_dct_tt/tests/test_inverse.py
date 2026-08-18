"""Phase 3 tests: the deformation inverse, identifiability, and the refuted claim.

Pre-registered in ``dev/paper/PREREGISTER.md``; results in
``dev/paper/RESULTS_phase3.md``. The tests here lock the findings that a future
change could silently undo -- in particular the identifiability ranks (B1) and
the *size* of the linearisation bias, which is the number that refuted the
package's original novelty claim.
"""
import math

import pytest
import torch
from midas_dfxm.conventions import rotation_matrix
from midas_dfxm.field import deform_reflection, reciprocal_basis

from midas_dct_tt import (
    attach_uniform_field,
    identifiability_rank,
    local_Q,
    sphere_grain,
    tt_alignment,
    tt_resolution,
)

DT = torch.float64
LAMBDA_A = 0.172979
LATTICE = torch.tensor([3.6356] * 3 + [90.0] * 3, dtype=DT)
B = reciprocal_basis(LATTICE)
H_DIR = torch.tensor([[1.0, 0.6, -0.3], [-0.2, -0.7, 0.4], [0.5, -0.1, 0.8]], dtype=DT)
H_DIR = H_DIR / torch.linalg.matrix_norm(H_DIR)
I3 = torch.eye(3, dtype=DT)


def _G(hkl):
    return B @ torch.tensor(hkl, dtype=DT)


# ---------------------------------------------------------------------------
# the two models
# ---------------------------------------------------------------------------
@pytest.mark.contract
def test_exact_model_is_midas_dfxm_deform_reflection():
    """The exact arm must be the sibling primitive, not a local re-derivation."""
    F = I3 + 0.01 * H_DIR
    G0 = _G((1, 1, 1))
    assert torch.allclose(local_Q(F, G0, model="exact"), deform_reflection(F, G0), atol=1e-15)


@pytest.mark.unit
def test_linear_model_is_the_small_strain_form():
    F = I3 + 0.01 * H_DIR
    G0 = _G((1, 1, 1))
    want = G0 - (F - I3).T @ G0
    assert torch.allclose(local_Q(F, G0, model="linear"), want, atol=1e-15)


@pytest.mark.unit
def test_models_agree_to_first_order():
    """Both are correct to O(|H|); they must not differ at first order."""
    G0 = _G((1, 1, 1))
    for mag in (1e-6, 1e-5):
        F = I3 + mag * H_DIR
        d = torch.linalg.vector_norm(local_Q(F, G0, model="exact")
                                     - local_Q(F, G0, model="linear"))
        first_order = mag * float(torch.linalg.vector_norm(G0))
        assert float(d) / first_order < 1e-4


@pytest.mark.unit
def test_both_models_are_exact_for_no_deformation():
    G0 = _G((2, 0, 0))
    for model in ("exact", "linear"):
        assert torch.allclose(local_Q(I3, G0, model=model), G0, atol=1e-15)


@pytest.mark.unit
def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="exact.*linear"):
        local_Q(I3, _G((1, 1, 1)), model="quadratic")


@pytest.mark.unit
def test_model_difference_scales_as_H_squared():
    """The pre-registered discriminator: the gap IS the omitted second-order term.

    Measured ratios matched |H|^2 to better than 1% across two decades. This is
    the correctness check on the exact-finite-strain implementation -- if the
    difference ever stops scaling as |H|^2, one of the two models is wrong.
    """
    G0 = _G((1, 1, 1))
    def gap(mag):
        F = I3 + mag * H_DIR
        return float(torch.linalg.vector_norm(
            local_Q(F, G0, model="exact") - local_Q(F, G0, model="linear")))
    for a, b in ((1e-3, 3e-3), (3e-3, 1e-2), (1e-2, 2e-2)):
        assert abs(gap(b) / gap(a) / (b / a) ** 2 - 1.0) < 0.02


# ---------------------------------------------------------------------------
# B1 -- identifiability (CONFIRMED, with controls)
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("hkls,expected", [
    ([(1, 1, 1)], 3),
    ([(2, 0, 0)], 3),
    ([(1, 1, 1), (2, 0, 0)], 6),
    ([(1, 1, 1), (2, 0, 0), (2, 2, 0)], 9),
    ([(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1)], 9),
])
def test_identifiability_rank(hkls, expected):
    """One reflection constrains 3 of 9; three non-coplanar give all 9."""
    rank, _ = identifiability_rank(I3 + 1e-3 * H_DIR, [_G(h) for h in hkls])
    assert rank == expected


@pytest.mark.unit
def test_collinear_reflections_add_no_rank():
    """CONTROL: (111),(222),(333) is three reflections but one direction.

    Without this control "three reflections give rank 9" would be indistinguishable
    from "three *independent* reflections give rank 9", and a scan plan could be
    built on collinear reflections that determines nothing extra.
    """
    rank, _ = identifiability_rank(
        I3 + 1e-3 * H_DIR, [_G((1, 1, 1)), _G((2, 2, 2)), _G((3, 3, 3))])
    assert rank == 3


@pytest.mark.unit
def test_coplanar_reflections_give_rank_six():
    """CONTROL: three coplanar G span a 2-D subspace -> 3 x 2 = 6 components."""
    rank, _ = identifiability_rank(
        I3 + 1e-3 * H_DIR, [_G((1, 1, 0)), _G((1, -1, 0)), _G((2, 0, 0))])
    assert rank == 6


@pytest.mark.unit
def test_null_space_is_exactly_zero_not_merely_small():
    """The limit is structural: unconstrained directions have zero singular value."""
    rank, sv = identifiability_rank(I3 + 1e-3 * H_DIR, [_G((1, 1, 1))])
    assert rank == 3
    assert sv.numel() == 9                      # the full parameter space
    assert float(sv[rank]) == 0.0               # exactly zero, not merely small
    assert float(sv[-1]) == 0.0


# ---------------------------------------------------------------------------
# B2 -- REFUTED: position resolves the sign that intensity cannot
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_intensity_is_blind_to_the_longitudinal_sign():
    """Half of B2 holds: the acceptance is even in the longitudinal offset."""
    grain = attach_uniform_field(sphere_grain(3.0, spacing_um=1.0))
    G0 = grain.field.reference_G((1, 1, 1))
    al = tt_alignment(G0, LAMBDA_A)
    res = tt_resolution(al)
    w = []
    for sign in (+1, -1):
        F = (1.0 + sign * 2e-4) * I3
        w.append(float(res.weight(local_Q(F, G0) @ al.sample_to_lab.T)))
    assert abs(w[0] / w[1] - 1.0) < 1e-5


@pytest.mark.unit
def test_position_resolves_the_longitudinal_sign():
    """The other half fails: G is at 90 - theta to k_h, so the sign moves the spot.

    Refutes B2 as pre-registered, and is a positive result -- TT's spatial
    resolution carries strain-sign information that a rocking curve alone cannot.
    """
    grain = attach_uniform_field(sphere_grain(3.0, spacing_um=1.0))
    G0 = grain.field.reference_G((1, 1, 1))
    al = tt_alignment(G0, LAMBDA_A)
    khat = al.beam_direction()
    dirs = []
    for sign in (+1, -1):
        F = (1.0 + sign * 2e-4) * I3
        k = al.k_in + local_Q(F, G0) @ al.sample_to_lab.T
        dirs.append(k / torch.linalg.vector_norm(k))
    # the two beams tilt in opposite senses, transverse to the optical axis
    sep = dirs[0] - dirs[1]
    sep_perp = sep - khat * torch.dot(sep, khat)
    assert float(torch.linalg.vector_norm(sep_perp)) > 1e-6

    # and G is nearly perpendicular to k_h, which is why
    g_hat = al.G_lab / torch.linalg.vector_norm(al.G_lab)
    cos_perp = math.sqrt(1.0 - float(torch.dot(g_hat, khat)) ** 2)
    assert cos_perp > 0.99


# ---------------------------------------------------------------------------
# A -- the refuted claim, locked as a number
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_linearisation_bias_stays_small_in_the_accessible_range():
    """The measurement that refuted Hypothesis A, at the Q level.

    Relative model discrepancy `|Q_exact - Q_lin| / (|H| |G0|)` stays at the
    percent level for |H| <= 5%. Recovered-F bias tracked this at 2.2% for
    |H| = 5% (dev/paper/RESULTS_phase3.md), against a 10% pre-registered
    threshold reached only near |H| ~ 24%.
    """
    G0 = _G((1, 1, 1))
    g_mag = float(torch.linalg.vector_norm(G0))
    for mag, bound in ((1e-3, 2e-3), (1e-2, 2e-2), (5e-2, 1e-1)):
        F = I3 + mag * H_DIR
        d = float(torch.linalg.vector_norm(
            local_Q(F, G0, model="exact") - local_Q(F, G0, model="linear")))
        assert d / (mag * g_mag) < bound


@pytest.mark.unit
def test_acceptance_ceiling_precedes_the_linearisation_crossover():
    """Finding A0: a fixed setting goes dark long before linearisation matters.

    Half-intensity at |H| = 7.6e-3 (dilation) and 1.9e-3 (rotation), against a
    2.4-4.9% crossover -- 3x and 13x earlier.
    """
    grain = attach_uniform_field(sphere_grain(3.0, spacing_um=1.0))
    G0 = grain.field.reference_G((1, 1, 1))
    al = tt_alignment(G0, LAMBDA_A)
    res = tt_resolution(al)

    def weight(F):
        return float(res.weight(local_Q(F, G0) @ al.sample_to_lab.T))

    assert weight((1.0 + 7.6e-3) * I3) == pytest.approx(0.5, abs=0.05)
    rot = rotation_matrix((0.0, 1.0, 0.0), math.degrees(1.9e-3)).to(DT)
    assert weight(rot) == pytest.approx(0.5, abs=0.05)
    # and by the crossover the reflection is gone
    assert weight((1.0 + 0.024) * I3) < 1e-2


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_fit_rejects_mismatched_reflection_lists():
    from midas_dct_tt import PlaneDetector, fit_uniform_deformation, psi_scan
    grain = attach_uniform_field(sphere_grain(2.0, spacing_um=1.0))
    al = tt_alignment(grain.field.reference_G((1, 1, 1)), LAMBDA_A)
    with pytest.raises(ValueError, match="same length"):
        fit_uniform_deformation(
            [torch.zeros(2, 8, 8, dtype=DT)], grain, [al, al], psi_scan(2),
            detector=PlaneDetector(shape=(8, 8)), hkls=[(1, 1, 1)],
            resolutions=[None, None])
