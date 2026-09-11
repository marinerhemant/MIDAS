"""detector_angle_maps -- the per-pixel 2theta/eta maps the ingest chain consumes (2026-09-10).

Until this existed every caller of subtract_background / choose_sectors / detect_powder_rings built the
maps by hand; one project's local copy had 94 importers and took eta from untilted pixel offsets while
2theta came from the tilted pixel_to_qlab.
"""
import math
import numpy as np
from midas_defect.geometry import Geometry, detector_angle_maps, pixel_to_qlab
from midas_defect.indexing import observed_coords


def _geom(**kw):
    base = dict(lsd_um=349640.6, bcy_px=100.3, bcz_px=90.7, px_um=172.0, wavelength_A=0.42459,
                n_pix_y=200, n_pix_z=180, omega_first_deg=0.0, omega_step_deg=1.0, n_frames=1)
    base.update(kw)
    return Geometry(**base)


def test_shape_is_rows_by_columns():
    tth, eta = detector_angle_maps(_geom())
    assert tth.shape == (180, 200) and eta.shape == (180, 200)


def test_flat_detector_matches_the_forward_model_convention():
    g = _geom()
    tth, eta = detector_angle_maps(g)
    rng = np.random.default_rng(0)
    r, c = rng.integers(0, 180, 60), rng.integers(0, 200, 60)
    oc = observed_coords(r.astype(float), c.astype(float), np.zeros(60), g).numpy()
    np.testing.assert_allclose(tth[r, c], np.degrees(oc[:, 0]), atol=1e-6)
    np.testing.assert_allclose((eta[r, c] - np.degrees(oc[:, 1]) + 180.0) % 360.0 - 180.0, 0.0, atol=1e-6)


def test_tilts_reach_the_two_theta_map():
    flat, _ = detector_angle_maps(_geom())
    g = _geom(ty_deg=0.4, tz_deg=0.3)
    tilted, _ = detector_angle_maps(g)
    r, c = np.array([10, 170, 90]), np.array([15, 190, 100])
    qn = np.linalg.norm(pixel_to_qlab(r.astype(float), c.astype(float), g, device="cpu").numpy(), axis=1)
    np.testing.assert_allclose(tilted[r, c], np.degrees(2 * np.arcsin(qn * g.wavelength_A / (4 * math.pi))), atol=1e-6)
    assert np.max(np.abs(tilted - flat)) > 1e-3
