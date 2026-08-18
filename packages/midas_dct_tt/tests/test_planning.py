"""Phase 4 tests: reflection-set planning.

The leakage index is validated against numbers measured independently by two
adversarial verification agents (dev/paper/RESULTS_phase3.md): a symmetry-closed
set nulls strain-to-orientation leakage, an asymmetric one does not.
"""
import math

import math

import pytest
import torch
from midas_dfxm.field import reciprocal_basis

from midas_dct_tt import (
    accessible_reflections,
    rank_reflection_sets,
    second_moment_tensor,
    strain_leakage_for_strain,
    strain_leakage_index,
)

DT = torch.float64
LATTICE = torch.tensor([3.6356] * 3 + [90.0] * 3, dtype=DT)
B = reciprocal_basis(LATTICE)
LAMBDA_A = 0.172979


def _G(hkl):
    return B @ torch.tensor(hkl, dtype=DT)


# ---------------------------------------------------------------------------
# the second-moment tensor and the leakage index
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_symmetry_closed_set_has_M_proportional_to_identity():
    """(200),(020),(002): M ∝ I, so the commutator [M,E] vanishes for ALL E."""
    M = second_moment_tensor([_G((2, 0, 0)), _G((0, 2, 0)), _G((0, 0, 2))])
    assert torch.allclose(M, torch.eye(3, dtype=DT) / 3.0, atol=1e-12)


@pytest.mark.unit
def test_leakage_index_is_zero_for_symmetry_closed_sets():
    """The design rule: these sets cannot convert strain into orientation."""
    for hkls in (
        [(2, 0, 0), (0, 2, 0), (0, 0, 2)],
        [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],   # full {111} family
    ):
        assert strain_leakage_index([_G(h) for h in hkls]) < 1e-12


@pytest.mark.unit
def test_leakage_index_is_nonzero_for_the_asymmetric_triplet():
    """(111),(200),(220) — the set that leaked ~18 deg/unit strain."""
    idx = strain_leakage_index([_G((1, 1, 1)), _G((2, 0, 0)), _G((2, 2, 0))])
    assert idx > 0.1


@pytest.mark.unit
def test_leakage_index_ranks_the_two_sets_as_measured():
    """Reproduces the ordering both verification agents measured independently."""
    good = strain_leakage_index([_G((2, 0, 0)), _G((0, 2, 0)), _G((0, 0, 2))])
    bad = strain_leakage_index([_G((1, 1, 1)), _G((2, 0, 0)), _G((2, 2, 0))])
    assert bad > 100 * max(good, 1e-15)


@pytest.mark.unit
def test_direction_resolved_leak_vanishes_for_every_strain_on_a_closed_set():
    """Not just for one strain: M ∝ I nulls the commutator identically."""
    torch.manual_seed(0)
    gs = [_G((2, 0, 0)), _G((0, 2, 0)), _G((0, 0, 2))]
    for _ in range(20):
        E = torch.randn(3, 3, dtype=DT)
        E = E + E.T
        E = E - torch.eye(3, dtype=DT) * torch.trace(E) / 3.0
        assert strain_leakage_for_strain(gs, E) < 1e-12


@pytest.mark.unit
def test_direction_resolved_leak_is_generally_nonzero_on_an_open_set():
    torch.manual_seed(0)
    gs = [_G((1, 1, 1)), _G((2, 0, 0)), _G((2, 2, 0))]
    n_leak = 0
    for _ in range(20):
        E = torch.randn(3, 3, dtype=DT)
        E = E + E.T
        E = E - torch.eye(3, dtype=DT) * torch.trace(E) / 3.0
        E = E / torch.linalg.matrix_norm(E)
        if strain_leakage_for_strain(gs, E) > 1e-3:
            n_leak += 1
    assert n_leak >= 18          # agents measured 98% of directions leaking


@pytest.mark.unit
def test_leakage_has_a_kernel_even_on_an_open_set():
    """axial([M,E]) is linear R^5 -> R^3, so a >=2-D null space always exists."""
    gs = [_G((1, 1, 1)), _G((2, 0, 0)), _G((2, 2, 0))]
    M = second_moment_tensor(gs)
    evals, evecs = torch.linalg.eigh(M)
    # a strain diagonal in M's eigenbasis commutes with M -> zero leak
    E = evecs @ torch.diag(torch.tensor([1.0, -0.5, -0.5], dtype=DT)) @ evecs.T
    assert strain_leakage_for_strain(gs, E) < 1e-12


# ---------------------------------------------------------------------------
# accessible reflections
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_accessible_reflections_respect_the_wavelength_limit():
    refl = accessible_reflections(B, LAMBDA_A, hkl_max=3)
    assert len(refl) > 0
    k = 2.0 * math.pi / LAMBDA_A
    for hkl, G, theta in refl:
        assert float(torch.linalg.vector_norm(G)) <= 2.0 * k + 1e-9
        assert 0.0 <= theta <= 90.0


@pytest.mark.unit
def test_long_wavelength_excludes_high_index_reflections():
    """d < lambda/2 is inaccessible at any setting — geometry, not a threshold."""
    few = accessible_reflections(B, 3.0, hkl_max=3)
    many = accessible_reflections(B, 0.172979, hkl_max=3)
    assert len(few) < len(many)


@pytest.mark.unit
def test_theta_window_filters():
    wide = accessible_reflections(B, LAMBDA_A, hkl_max=3)
    narrow = accessible_reflections(B, LAMBDA_A, hkl_max=3, max_theta_deg=3.0)
    assert 0 < len(narrow) < len(wide)
    assert all(t <= 3.0 for _, _, t in narrow)


# ---------------------------------------------------------------------------
# ranking
# ---------------------------------------------------------------------------
# At 71.7 keV nearly every low-index reflection is accessible (124 for hkl_max=2),
# so a theta window is mandatory before enumerating triplets -- which is also what
# you want physically, since low theta means a small missing cone.
def _planning_set():
    return accessible_reflections(B, LAMBDA_A, hkl_max=2, max_theta_deg=2.5)


@pytest.mark.unit
def test_ranking_puts_zero_leakage_sets_first():
    refl = _planning_set()
    best = rank_reflection_sets(refl, n_reflections=3, top=5)
    assert best, "no full-rank triplets found"
    assert best[0].leakage < 1e-9
    assert best[0].rank == 9


@pytest.mark.unit
def test_ranked_sets_are_full_rank_by_default():
    refl = _planning_set()
    for r in rank_reflection_sets(refl, n_reflections=3, top=10):
        assert r.rank == 9


@pytest.mark.unit
def test_ranking_refuses_a_combinatorial_blowup_rather_than_hanging():
    refl = accessible_reflections(B, LAMBDA_A, hkl_max=3)
    with pytest.raises(ValueError, match="max_candidates"):
        rank_reflection_sets(refl, n_reflections=3, max_candidates=10)


@pytest.mark.unit
def test_report_summary_is_printable():
    refl = _planning_set()
    r = rank_reflection_sets(refl, n_reflections=3, top=1)[0]
    s = r.summary()
    assert "rank" in s and "leak" in s and "cone" in s


def test_leakage_sort_is_available_and_correct_when_asked_for():
    """Leakage is no longer the DEFAULT sort -- the three criteria conflict, so the
    choice is explicit (default "cone"). But when asked for, it must still put a
    zero-leakage symmetry-closed set first.

    History worth keeping: this was briefly the default, restored after a wrong
    reading of C2, and finally demoted when the CRLB criterion showed the three
    axes pull in different directions. C2's final verdict is that the leakage
    benefit is real but ~5x under realistic noise, not the 120x the noiseless
    idealisation suggests -- worth reporting, not worth silently optimising.
    """
    reports = rank_reflection_sets(_planning_set(), top=600, sort_by="leakage")
    assert len(reports) > 50
    keys = [(round(r.leakage, 9), r.max_missing_cone_deg, r.crlb_trace)
            for r in reports]
    assert keys == sorted(keys), "leakage sort is not ordered"
    assert reports[0].leakage == pytest.approx(0.0, abs=1e-9)


def test_leakage_still_reported_as_a_diagnostic():
    """Demoted from a sort key, but still computed and still exactly zero for a
    symmetry-closed set -- it remains a correct statement about the POSITION
    channel, which is why it is kept rather than deleted."""
    B = torch.eye(3, dtype=torch.float64) * (2 * math.pi / 3.6)
    gs = [B @ torch.tensor(h, dtype=torch.float64)
          for h in [(2, 0, 0), (0, 2, 0), (0, 0, 2)]]
    assert strain_leakage_index(gs) == pytest.approx(0.0, abs=1e-12)


def test_the_three_criteria_genuinely_disagree():
    """Pinned because collapsing them into one sort key hides a real trade-off.

    CRLB falls as 1/|Q|^2 so it climbs to the highest accessible index, which is
    exactly where the missing cone is worst. If these ever agree, the candidate
    pool has become degenerate and the ranking is no longer informative.
    """
    refl = _planning_set()
    by_cone = rank_reflection_sets(refl, top=1, sort_by="cone")[0]
    by_crlb = rank_reflection_sets(refl, top=1, sort_by="crlb")[0]
    assert by_cone.hkls != by_crlb.hkls
    assert by_cone.max_missing_cone_deg <= by_crlb.max_missing_cone_deg
    assert by_crlb.crlb_trace <= by_cone.crlb_trace


def test_crlb_falls_as_inverse_q_squared():
    from midas_dfxm.field_inverse import crlb_trace
    c1 = crlb_trace([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    c2 = crlb_trace([(2, 0, 0), (0, 2, 0), (0, 0, 2)])
    assert c1 / c2 == pytest.approx(4.0, rel=1e-6)


def test_unknown_sort_key_is_rejected():
    with pytest.raises(ValueError, match="sort_by must be one of"):
        rank_reflection_sets(_planning_set(), top=1, sort_by="cheapest")


# --- systematic absences ------------------------------------------------
# Regression: accessible_reflections filtered only on d >= lambda/2, so it
# returned fcc-forbidden reflections. Because the TT missing cone has
# half-angle theta, the forbidden LOW-angle {100} set then *won* the ranking
# and was recommended as the optimal scan. A plan that cannot diffract.

def test_forbidden_reflections_are_dropped_when_a_crystal_is_given():
    from midas_dfxm.io import fcc_reference_crystal
    geom = accessible_reflections(B, LAMBDA_A, hkl_max=2, max_theta_deg=2.5)
    phys = accessible_reflections(B, LAMBDA_A, hkl_max=2, max_theta_deg=2.5,
                                  crystal=fcc_reference_crystal())
    assert len(geom) == 26, "geometry-only count changed; update the test"
    assert len(phys) == 8, f"fcc admits only the 8 {{111}}, got {len(phys)}"
    for hkl, _, _ in phys:
        h, k, l = (int(v) for v in hkl)
        parities = {h % 2, k % 2, l % 2}
        assert len(parities) == 1, f"{(h, k, l)} has mixed parity -> |F| = 0"


def test_forbidden_reflections_would_otherwise_win_the_ranking():
    """The bug was not cosmetic: the absences outrank the allowed set."""
    from midas_dfxm.io import fcc_reference_crystal
    geom = accessible_reflections(B, LAMBDA_A, hkl_max=2, max_theta_deg=2.5)
    top_unfiltered = rank_reflection_sets(geom, top=1, sort_by="cone")[0]
    assert any(len({int(h) % 2 for h in hkl}) > 1 for hkl in top_unfiltered.hkls), (
        "unfiltered ranking no longer prefers a forbidden set -- if the ranking "
        "changed, this regression test no longer guards anything"
    )
    phys = accessible_reflections(B, LAMBDA_A, hkl_max=2, max_theta_deg=2.5,
                                  crystal=fcc_reference_crystal())
    top_filtered = rank_reflection_sets(phys, top=1, sort_by="cone")[0]
    for hkl in top_filtered.hkls:
        assert len({int(h) % 2 for h in hkl}) == 1


def test_missing_crystal_warns_loudly():
    """crystal=None is still the default; it must not fail silently."""
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        accessible_reflections(B, LAMBDA_A, hkl_max=1, max_theta_deg=2.5)
    assert any(issubclass(c.category, RuntimeWarning)
               and "systematic" in str(c.message) for c in caught), \
        "no RuntimeWarning raised when the structure factor was skipped"


# --- physically-weighted A-optimality -----------------------------------
# The published |Q|^-2 law is an identity of the isotropic equal-photon error
# model, not a property of reflections. These pin both the identity and its
# reversal under the real noise, so the design claim cannot silently revert.

def _ortho_triplet(n, a=3.6356, lam=12.398 / 71.7):
    import math
    from midas_dfxm.field import reciprocal_basis
    Bm = reciprocal_basis(torch.tensor([a] * 3 + [90.0] * 3, dtype=torch.float64))
    out = []
    for hkl in [(n, 0, 0), (0, n, 0), (0, 0, n)]:
        G = Bm @ torch.tensor(hkl, dtype=torch.float64)
        d = a / math.sqrt(sum(v * v for v in hkl))
        out.append((hkl, G, math.degrees(math.asin(lam / (2 * d)))))
    return out


def test_isotropic_equal_photon_crlb_is_the_inverse_square_identity():
    from midas_dct_tt.planning import physical_crlb_trace
    kw = dict(wavelength_A=12.398 / 71.7, anisotropic=False,
              photon_weighted=False, lorentz=False)
    c2 = physical_crlb_trace(_ortho_triplet(2), **kw)
    c6 = physical_crlb_trace(_ortho_triplet(6), **kw)
    assert c2 / c6 == pytest.approx(9.0, rel=1e-9), "must reproduce |Q|^-2 exactly"


def test_anisotropic_acceptance_alone_softens_the_law_to_inverse_first_power():
    """sigma_rock scales AS |Q|, cancelling the gain in the narrow channel."""
    from midas_dct_tt.planning import physical_crlb_trace
    kw = dict(wavelength_A=12.398 / 71.7, anisotropic=True,
              photon_weighted=False, lorentz=False)
    ratio = (physical_crlb_trace(_ortho_triplet(2), **kw)
             / physical_crlb_trace(_ortho_triplet(6), **kw))
    assert 2.5 < ratio < 3.5, f"expected ~3 (i.e. |Q|^-1), got {ratio}"


def test_real_photon_counts_reverse_the_ranking():
    """The design conclusion inverts: low index wins once |F|^2 is honoured."""
    from midas_dfxm.io import fcc_reference_crystal
    from midas_dct_tt.planning import physical_crlb_trace
    kw = dict(wavelength_A=12.398 / 71.7, crystal=fcc_reference_crystal(),
              anisotropic=True, photon_weighted=True, lorentz=True)
    c2 = physical_crlb_trace(_ortho_triplet(2), **kw)
    c6 = physical_crlb_trace(_ortho_triplet(6), **kw)
    assert c6 > c2, "with real photons the HIGH-index set must be worse"
    assert c6 / c2 == pytest.approx(5.95, rel=0.05)


def test_photon_weighting_without_a_crystal_is_refused():
    from midas_dct_tt.planning import physical_crlb_trace
    with pytest.raises(ValueError, match="needs `crystal`"):
        physical_crlb_trace(_ortho_triplet(2), wavelength_A=12.398 / 71.7,
                            photon_weighted=True)
