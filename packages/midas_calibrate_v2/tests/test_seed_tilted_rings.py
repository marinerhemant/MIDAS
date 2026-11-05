"""make_seed on a TILTED detector, against an independent ray / plane renderer.

A ring on a tilted plane is an ellipse. The seeder used to drop every arc whose
circle fit missed by more than 1.5 px; at -10 deg that is every full-length arc
(rms 0.6 % of R), so only short fragments of the OUTER rings survived, were taken
for the innermost rings, and the seed distance came out 2.66x too large with an
rms of ~0 (measured on 5 of 43 frames of a real rotation series).

The image here is drawn by exact ray / plane geometry in this file (no MIDAS
forward model), so a convention error in the package cannot hide.
"""
import numpy as np
import pytest

from midas_calibrate_v2.seed.auto_seed import make_seed, _ring_table
from midas_calibrate_v2.seed.calibrant import resolve_calibrant

N = 1800
PX_UM = 150.0
WL = 12.398419843 / 63.0
L_PERP_UM = 900_000.0


def _tilted_rings(alpha_deg, phi_deg=0.0, seed=0):
    """CeO2 rings on a plane tilted by alpha about azimuth phi; beam hits the image centre."""
    a, p = np.radians(alpha_deg), np.radians(phi_deg)
    n = np.array([np.sin(a) * np.cos(p), np.sin(a) * np.sin(p), np.cos(a)])
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - n * (ref @ n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    c = np.array([0.0, 0.0, L_PERP_UM / np.cos(a)])
    j, i = np.meshgrid(np.arange(N) - (N - 1) / 2, np.arange(N) - (N - 1) / 2)
    pts = c + (j[..., None] * PX_UM) * e1 + (i[..., None] * PX_UM) * e2
    tth = np.degrees(np.arccos(pts[..., 2] / np.linalg.norm(pts, axis=-1)))
    cal = resolve_calibrant("CeO2")
    ring_tth = [r[2] for r in _ring_table(cal, WL, n_rings=12)]
    rng = np.random.default_rng(seed)
    img = np.full((N, N), 100.0, np.float32)
    for k, t in enumerate(ring_tth):
        if t < tth.max():
            img += (900.0 - 30 * k) * np.exp(-0.5 * ((tth - t) / 0.025) ** 2)
    img += rng.normal(0.0, 3.0, img.shape).astype(np.float32)
    return np.clip(img, 0, None), L_PERP_UM


@pytest.mark.parametrize("alpha", [0.0, 10.0])
def test_small_tilt_seeds_first_ring_and_the_right_distance(alpha):
    img, lperp = _tilted_rings(alpha)
    s = make_seed(img, wavelength_A=WL, px_um=PX_UM, calibrant="CeO2", use_diplib=False)
    assert s.first_ring == 1, s.notes
    assert s.n_measured >= 4, s.notes
    assert s.Lsd_um == pytest.approx(lperp / np.cos(np.radians(alpha)), rel=0.03), (s, s.notes)
    assert "WEAK" not in s.notes


def test_large_tilt_does_not_mistake_an_outer_ring_for_the_first():
    """At 30 deg the circle misfit is ~4 % of R: beyond the first tier, inside the second."""
    img, lperp = _tilted_rings(30.0, phi_deg=35.0)
    s = make_seed(img, wavelength_A=WL, px_um=PX_UM, calibrant="CeO2", use_diplib=False)
    assert s.first_ring == 1, s.notes
    assert s.n_measured >= 4, s.notes
    # a circle is the wrong model here, so the distance is only a seed: along-beam Lsd = Lperp/cos
    assert s.Lsd_um == pytest.approx(lperp / np.cos(np.radians(30.0)), rel=0.15), (s, s.notes)


def test_seed_on_two_radii_is_flagged_weak():
    """Two usable rings fit several assignments equally well; the seed must say so."""
    img, _ = _tilted_rings(0.0)
    yy, xx = np.indices(img.shape)
    r = np.hypot(yy - (N - 1) / 2, xx - (N - 1) / 2)
    img = img.copy()
    img[(r < 1100)] = 100.0                           # erase the inner rings
    s = make_seed(img, wavelength_A=WL, px_um=PX_UM, calibrant="CeO2", use_diplib=False)
    assert s.n_measured < 4
    assert "WEAK" in s.notes


def test_old_fixed_cutoff_is_what_broke_it():
    """Pin the cause: with the old 1.5 px absolute cutoff the 10 deg frame loses its full arcs."""
    img, lperp = _tilted_rings(10.0)
    old = make_seed(img, wavelength_A=WL, px_um=PX_UM, calibrant="CeO2", use_diplib=False,
                    max_rms_rel=0.0, max_rms_rel_tilted=0.0)
    new = make_seed(img, wavelength_A=WL, px_um=PX_UM, calibrant="CeO2", use_diplib=False)
    assert new.n_measured > old.n_measured
    assert abs(new.Lsd_um / lperp - 1.0) < 0.03


def test_calibrate_tilted_recovers_from_a_wrong_first_seed(monkeypatch):
    """If the first seed has the wrong distance (the 2.66x failure) and the gate FAILs, the
    alternative seed must be tried and kept. The first seed is forged here; the alternative is
    the real make_seed(rms_rel_force=...)."""
    import midas_calibrate_v2.seed.auto_seed as A
    from midas_calibrate_v2.pipelines.tilted import calibrate_tilted
    img, lperp = _tilted_rings(10.0, phi_deg=20.0)
    real = A.make_seed

    def forged(im, **kw):
        s = real(im, **kw)
        if kw.get("rms_rel_force") is None:
            s.Lsd_um = s.Lsd_um * 2.66
            s.first_ring = 1
        return s
    monkeypatch.setattr(A, "make_seed", forged)
    res = calibrate_tilted(img, wavelength_A=WL, px_um=PX_UM, calibrant="CeO2", polish=False,
                           fallback_pv=False)
    assert res.verdict in ("PASS", "WARN"), (res.verdict, res.notes)
    assert any("alternative seed" in n for n in res.notes), res.notes
    assert res.perp_distance_mm == pytest.approx(lperp / 1e3, rel=0.02), res.perp_distance_mm


def test_calibrate_tilted_tries_the_runner_up_assignment(monkeypatch):
    """Both seeds forged wrong (what 2 of 43 real frames did: the right ring assignment ranked
    second); the correct distance sits in ``seed.alternatives``. The gate must send us there."""
    import midas_calibrate_v2.seed.auto_seed as A
    from midas_calibrate_v2.pipelines.tilted import calibrate_tilted
    img, lperp = _tilted_rings(10.0, phi_deg=20.0)
    real = A.make_seed
    correct = lperp / np.cos(np.radians(10.0))

    def forged(im, **kw):
        s = real(im, **kw)
        s.Lsd_um = s.Lsd_um * 1.0 / 0.68
        s.first_ring = 2
        s.alternatives = [{"first_ring": 1, "Lsd_um": correct * 0.99, "n_match": 8, "rms_px": 3.0, "score": 5.0}]
        return s
    monkeypatch.setattr(A, "make_seed", forged)
    res = calibrate_tilted(img, wavelength_A=WL, px_um=PX_UM, calibrant="CeO2", polish=False,
                           fallback_pv=False)
    assert res.verdict in ("PASS", "WARN"), (res.verdict, res.notes)
    assert any("runner-up" in n for n in res.notes), res.notes


def test_seed_reports_distinct_runner_up_assignments():
    img, _ = _tilted_rings(0.0)
    s = make_seed(img, wavelength_A=WL, px_um=PX_UM, calibrant="CeO2", use_diplib=False)
    assert isinstance(s.alternatives, list) and len(s.alternatives) <= 3
    for h in s.alternatives:
        assert abs(h["Lsd_um"] / s.Lsd_um - 1.0) >= 0.03
        assert h["score"] <= 1e9 and "first_ring" in h
