"""Phase 2 tests: Friedel pairing and spot-to-grain assignment."""
import math

import pytest
import torch

from midas_dct_tt import (
    Spot,
    assign_spots,
    bragg_flashes,
    friedel_pairs,
    friedel_partner_omega,
    unassigned,
)

DT = torch.float64
LAMBDA_A = 0.172979
K_MAG = 2.0 * math.pi / LAMBDA_A


def _g(theta_deg, direction=(0.3, -0.5, 0.2)):
    d = torch.as_tensor(direction, dtype=DT)
    return 2.0 * K_MAG * math.sin(math.radians(theta_deg)) * d / torch.linalg.vector_norm(d)


# ---------------------------------------------------------------------------
# Friedel pairing
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_partner_omega_wraps():
    assert abs(friedel_partner_omega(10.0) - 190.0) < 1e-12
    assert abs(friedel_partner_omega(200.0) - 20.0) < 1e-12
    assert abs(friedel_partner_omega(359.0) - 179.0) < 1e-12


@pytest.mark.unit
def test_pairs_a_reflection_with_its_own_negative():
    """G and -G give four flashes: exactly two Friedel pairs."""
    g = _g(7.5)
    flashes = bragg_flashes(g, LAMBDA_A) + bragg_flashes(-g, LAMBDA_A)
    pairs = friedel_pairs(flashes)
    assert len(pairs) == 2
    assert sorted(i for p in pairs for i in p) == [0, 1, 2, 3]


@pytest.mark.unit
def test_paired_flashes_are_180_apart_and_antiparallel_in_the_sample_frame():
    g = _g(7.5)
    flashes = bragg_flashes(g, LAMBDA_A) + bragg_flashes(-g, LAMBDA_A)
    for i, j in friedel_pairs(flashes):
        d_om = abs((flashes[j].omega_deg - flashes[i].omega_deg) % 360.0 - 180.0)
        assert d_om < 1e-6
        gi = flashes[i].G_sample / torch.linalg.vector_norm(flashes[i].G_sample)
        gj = flashes[j].G_sample / torch.linalg.vector_norm(flashes[j].G_sample)
        assert float(torch.dot(gi, gj)) < -0.999999


@pytest.mark.unit
def test_lab_frame_partner_is_a_mirror_not_an_inversion():
    """The trap: in the lab, a Friedel partner is NOT the negative of G.

    Rotating half a turn negates the in-plane part of G but leaves z alone, and
    the partner reflection negates all three -- so the lab-frame vectors share
    ``(x, y)`` and have opposite ``z``. Measured cos between them is +0.79 for
    this reflection, nowhere near -1, so a lab-frame antiparallel test would find
    no pairs at all (it silently did, before this was pinned down).
    """
    g = _g(7.5)
    flashes = bragg_flashes(g, LAMBDA_A) + bragg_flashes(-g, LAMBDA_A)
    pairs = friedel_pairs(flashes)
    assert pairs
    for i, j in pairs:
        a, b = flashes[i].G_lab, flashes[j].G_lab
        mirror = torch.tensor([1.0, 1.0, -1.0], dtype=DT) * a
        assert torch.allclose(b, mirror, atol=1e-10)
        cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
        assert cos > 0.0                     # emphatically not antiparallel


@pytest.mark.unit
def test_paired_beams_are_mirrored_vertically_about_the_beam():
    """The geometry a pair actually gives you: equal horizontal, opposite vertical.

    This vertical mirror -- not a full inversion -- is what lets a Friedel pair
    localise the grain along the beam by triangulation.
    """
    g = _g(7.5)
    flashes = bragg_flashes(g, LAMBDA_A) + bragg_flashes(-g, LAMBDA_A)
    for i, j in friedel_pairs(flashes):
        a, b = flashes[i].k_out, flashes[j].k_out
        assert abs(float(a[0] - b[0])) < 1e-12          # same along the beam
        assert abs(float(a[1] - b[1])) < 1e-12          # same horizontal deflection
        assert abs(float(a[2] + b[2])) < 1e-12          # opposite vertical
        assert abs(float(a[2])) > 1e-3                  # and it is genuinely nonzero


@pytest.mark.unit
def test_unrelated_reflections_180_apart_are_not_paired():
    """The omega test alone is not enough; the direction test must veto.

    Two different reflections can easily flash half a turn apart. Pairing them
    would corrupt any triangulation built on the pair.
    """
    a = bragg_flashes(_g(7.5, (0.3, -0.5, 0.2)), LAMBDA_A)
    fake = [a[0]]
    # A flash at exactly +180 deg but with a direction that is NOT antiparallel.
    other = bragg_flashes(_g(7.5, (0.9, 0.2, -0.1)), LAMBDA_A)
    forced = type(other[0])(
        omega_deg=friedel_partner_omega(a[0].omega_deg),
        G_lab=other[0].G_lab, k_out=other[0].k_out, theta_deg=other[0].theta_deg,
        G_sample=other[0].G_sample,
    )
    assert friedel_pairs(fake + [forced]) == []


@pytest.mark.unit
def test_each_flash_is_used_at_most_once():
    g = _g(7.5)
    flashes = bragg_flashes(g, LAMBDA_A) + bragg_flashes(-g, LAMBDA_A)
    idx = [i for p in friedel_pairs(flashes) for i in p]
    assert len(idx) == len(set(idx))


@pytest.mark.unit
def test_no_pairs_when_only_one_side_is_present():
    assert friedel_pairs(bragg_flashes(_g(7.5), LAMBDA_A)) == []


# ---------------------------------------------------------------------------
# spot assignment
# ---------------------------------------------------------------------------
def _pred():
    return [
        Spot(omega_deg=10.0, u_px=100.0, v_px=100.0, grain=0, hkl=(1, 1, 1)),
        Spot(omega_deg=95.0, u_px=40.0, v_px=180.0, grain=0, hkl=(2, 0, 0)),
        Spot(omega_deg=200.0, u_px=150.0, v_px=60.0, grain=1, hkl=(1, 1, 1)),
    ]


@pytest.mark.unit
def test_assigns_exact_matches():
    pred = _pred()
    obs = [Spot(p.omega_deg, p.u_px, p.v_px) for p in pred]
    assert assign_spots(obs, pred) == [0, 1, 2]
    assert unassigned(assign_spots(obs, pred)) == 0


@pytest.mark.unit
def test_assigns_within_tolerance_and_recovers_grain_identity():
    pred = _pred()
    obs = [Spot(10.4, 102.0, 98.0), Spot(199.6, 148.0, 62.0)]
    a = assign_spots(obs, pred, omega_tol_deg=1.0, pixel_tol=5.0)
    assert [pred[i].grain for i in a] == [0, 1]


@pytest.mark.unit
def test_reports_unmatched_rather_than_forcing_a_match():
    """A spot from a grain not in the candidate list must come back None.

    An assignment that always succeeds cannot tell you the grain map is
    incomplete -- which is the thing you most need to know.
    """
    pred = _pred()
    obs = [Spot(10.0, 100.0, 100.0), Spot(300.0, 20.0, 20.0)]
    a = assign_spots(obs, pred)
    assert a[0] == 0 and a[1] is None
    assert unassigned(a) == 1


@pytest.mark.unit
def test_omega_tolerance_is_enforced():
    pred = _pred()
    obs = [Spot(13.0, 100.0, 100.0)]                    # 3 deg off
    assert assign_spots(obs, pred, omega_tol_deg=1.0)[0] is None
    assert assign_spots(obs, pred, omega_tol_deg=5.0)[0] == 0


@pytest.mark.unit
def test_pixel_tolerance_is_enforced():
    pred = _pred()
    obs = [Spot(10.0, 130.0, 100.0)]                    # 30 px off
    assert assign_spots(obs, pred, pixel_tol=5.0)[0] is None
    assert assign_spots(obs, pred, pixel_tol=40.0)[0] == 0


@pytest.mark.unit
def test_one_prediction_cannot_claim_two_spots():
    pred = _pred()
    obs = [Spot(10.0, 100.0, 100.0), Spot(10.2, 101.0, 100.5)]
    a = assign_spots(obs, pred, omega_tol_deg=2.0, pixel_tol=10.0)
    assert sorted(x for x in a if x is not None) == [0]
    assert unassigned(a) == 1


@pytest.mark.unit
def test_closest_spot_wins_the_contested_prediction():
    pred = _pred()
    far = Spot(11.0, 104.0, 104.0)
    near = Spot(10.05, 100.2, 100.1)
    a = assign_spots([far, near], pred, omega_tol_deg=2.0, pixel_tol=10.0)
    assert a[1] == 0 and a[0] is None


@pytest.mark.unit
def test_omega_wraparound_is_handled():
    """359.5 and 0.5 deg are 1 deg apart, not 359."""
    pred = [Spot(omega_deg=359.5, u_px=50.0, v_px=50.0, grain=3)]
    obs = [Spot(0.5, 50.0, 50.0)]
    assert assign_spots(obs, pred, omega_tol_deg=1.5)[0] == 0
