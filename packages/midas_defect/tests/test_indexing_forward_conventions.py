"""build_forward_model (midas_diffract.HEDMForwardModel) must predict the same spots as raster.predict_reflections, the
path checked against real DAC data: same pixels with tilts and distortion, and no reflection lost at the wedge edge.

Before 2026-09-27 it passed the frame-CENTRE omega as HEDMForwardModel's leading-edge omega_start (dropping every
reflection in the first half of frame 0), applied the NF ray-plane tilt convention instead of the FF one the calibrated
tilts mean (median 7.6 px off at tilts 0.3/0.15/0.37 deg), and ignored the distortion coefficients.

The wedge case needs a midas_diffract carrying the shared "Wedge convention" (after 0.8.1); with an older one the two
predictors disagree at Wedge != 0 only."""
import numpy as np
import pytest

from midas_defect.geometry import Geometry
from midas_defect.indexing import build_forward_model, predict_spots
from midas_defect.raster import predict_reflections
from midas_defect.synthetic import _b_matrix, _hkl_candidates

A, C = 3.6008, 19.2522


def _crystal():
    from midas_hkls.crystal import Atom, Crystal
    from midas_hkls.lattice import Lattice
    from midas_hkls.space_group import SpaceGroup
    return Crystal(Lattice(A, A, C, 90, 90, 90), SpaceGroup.from_number(139), [Atom("La", (0, 0, 0))])


@pytest.mark.parametrize("kw", [dict(), dict(tx_deg=0.3, ty_deg=0.15, tz_deg=0.37),
                                dict(tx_deg=0.3, ty_deg=0.15, tz_deg=0.37, p_coeffs=(1e-4,) + (0.0,) * 14),
                                dict(tx_deg=0.3, ty_deg=0.15, tz_deg=0.37, p_coeffs=(1e-4,) + (0.0,) * 14,
                                     wedge_deg=2.0)],
                         ids=["flat", "tilted", "tilted+distortion", "tilted+distortion+wedge"])
def test_forward_model_matches_reference_predictor(kw):
    g = Geometry(lsd_um=349680.0, bcy_px=737.0, bcz_px=839.0, px_um=172.0, wavelength_A=0.42459, n_pix_y=1475,
                 n_pix_z=1679, omega_first_deg=-6.5, omega_step_deg=1.0, n_frames=14, **kw)
    model, _, _ = build_forward_model(_crystal(), g, d_min=0.35)
    B = _b_matrix(A, A, C); hkl = _hkl_candidates(8, 8, 30, 139)
    rng = np.random.default_rng(3)
    d_px, d_om, n_first_half = [], [], 0
    for _ in range(6):
        U = np.linalg.qr(rng.normal(size=(3, 3)))[0]; U[:, 0] *= np.sign(np.linalg.det(U))
        r, c, om, _, _ = predict_reflections(U, B, hkl, g, omega_sign=1)
        s = predict_spots(model, U)
        v = s.valid.reshape(-1).numpy() > 0.5
        mr, mc = s.z_pixel.reshape(-1).numpy()[v], s.y_pixel.reshape(-1).numpy()[v]
        mo = model.omega_edge_deg + g.omega_step_deg * s.frame_nr.reshape(-1).numpy()[v]
        for rr, cc, oo in zip(r, c, om):
            k = int(np.argmin(np.hypot(mr - rr, mc - cc)))
            d_px.append(float(np.hypot(mr[k] - rr, mc[k] - cc))); d_om.append(abs(float(mo[k] - oo)))
            n_first_half += (oo - g.omega_first_deg) / g.omega_step_deg < 0
    d_px, d_om = np.asarray(d_px), np.asarray(d_om)
    assert n_first_half > 0                                  # the edge case is actually exercised
    assert d_px.max() < 1e-2, f"max pixel difference {d_px.max():.3g}"
    assert d_om.max() < 1e-2, f"max omega difference {d_om.max():.3g} deg"
