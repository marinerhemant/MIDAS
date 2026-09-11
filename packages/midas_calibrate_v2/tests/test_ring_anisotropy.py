"""ring_harmonics -- the cos(2 eta) floor on an a/b splitting, measured on a ring (2026-09-10)."""
import math
import numpy as np
from midas_calibrate_v2.ring_anisotropy import ring_harmonics


def _ring(n=240, q0=3.05, a1=0.0, a2=0.0, p1=30.0, p2=20.0, noise=2e-5, seed=0, span=360.0):
    rng = np.random.default_rng(seed)
    eta = rng.uniform(0.0, span, n)
    e = np.radians(eta)
    q = q0 * (1 + a1 * np.cos(e - math.radians(p1)) + a2 * np.cos(2 * (e - math.radians(p2))))
    return eta, q + rng.normal(0.0, noise, n)


def test_a_planted_cos2eta_is_recovered_as_the_equivalent_delta():
    eta, q = _ring(a2=1.42e-3)                          # 0.142 %, the on-frame anvil value
    r = ring_harmonics(eta, q, n_null=200)
    assert r.determined and abs(r.a2_rel - 1.42e-3) < 5e-5 and r.p_a2 < 0.01
    assert abs(((r.phase2_deg - 20.0 + 90.0) % 180.0) - 90.0) < 2.0


def test_an_isotropic_ring_is_not_called_anisotropic():
    eta, q = _ring()
    r = ring_harmonics(eta, q, n_null=200)
    assert r.a2_rel < 3e-5 and r.p_a2 > 0.01


def test_a_beam_centre_error_lands_in_a1_not_a2():
    eta, q = _ring(a1=2e-3)
    r = ring_harmonics(eta, q, n_null=200)
    assert r.p_a1 < 0.01 and abs(r.a1_rel - 2e-3) < 5e-5 and r.a2_rel < 5e-5


def test_clustered_azimuths_are_reported_undetermined():
    eta, q = _ring(n=40, a2=1.42e-3, span=50.0)
    assert not ring_harmonics(eta, q, n_null=50).determined
