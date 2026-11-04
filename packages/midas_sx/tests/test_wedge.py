"""The omega-axis tilt (MIDAS wedge, rotation about lab y): Ewald crossings, predicted positions, hkl assignment and the refinement of the wedge and the detector tilts. The reference for the
convention is midas_defect's own qlab_to_qsample / pixel_to_qlab (q_lab = R_y(-W) R_z(omega) q_sample), not this package's formulas."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.spatial.transform import Rotation as Rot

from midas_defect.geometry import ewald_crossing_omegas, pixel_to_qlab, qlab_to_qsample
from midas_hkls import Lattice
from midas_sx.assign import assign_hkl, calculated_crossing, ewald_crossings, reciprocal_basis
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.orient import index_and_refine, predict_positions

LAT = Lattice(5.8033, 5.9595, 12.2978, 90, 90, 90)
LAM = 0.6889


def geom(**kw):
    g = make_geometry(wavelength_A=LAM, dist_mm=170.0, pixel_mm=0.172, centre_rc=(840.0, 400.0), shape=(1679, 1475), scan=Scan(-150.0, 0.2, 850))
    return replace(g, **kw)


def test_ewald_crossings_zero_wedge_is_the_midas_defect_function_itself():
    q = 2 * np.pi * np.array([0.31, -0.72, 0.44])
    assert np.array_equal(ewald_crossings(q, LAM, 0.0), ewald_crossing_omegas(q, LAM))


@pytest.mark.parametrize("W", [0.3, -0.5, 2.0])
def test_ewald_crossings_with_a_wedge_satisfy_the_diffraction_condition_in_midas_conventions(W):
    rng = np.random.default_rng(1)
    k = 2 * np.pi / LAM
    n_found = 0
    for _ in range(200):
        q = 2 * np.pi * rng.normal(size=3) * 0.5
        for w in ewald_crossings(q, LAM, W):
            n_found += 1
            q_lab = qlab_to_qsample  # noqa: F841  (the inverse is used below)
            # forward: q_lab = R_y(-W) R_z(w) q_sample, via midas_defect's qlab_to_qsample on the result:  q_sample = R_z(-w) R_y(W) q_lab
            c, s = np.cos(w), np.sin(w)
            x, y, z = c * q[0] - s * q[1], s * q[0] + c * q[1], q[2]
            Wr = np.radians(W)
            ql = np.array([np.cos(Wr) * x - np.sin(Wr) * z, y, np.sin(Wr) * x + np.cos(Wr) * z])
            back = qlab_to_qsample(torch.as_tensor(ql[None], dtype=torch.float64), torch.as_tensor([w], dtype=torch.float64), np.radians(W)).numpy()[0]
            assert np.allclose(back, q, atol=1e-10)                                # my forward rotation is the inverse of midas_defect's
            assert abs(np.linalg.norm(ql + np.array([k, 0, 0])) - k) < 1e-9        # on the Ewald sphere
    assert n_found > 50


def test_calculated_crossing_with_a_wedge_matches_the_analytic_one():
    q = np.array([0.31, -0.72, 0.44]) * 1.1                                       # 1/A, no 2 pi
    W = 0.4
    w_an = np.degrees(ewald_crossings(2 * np.pi * q, LAM, W))
    for w0 in w_an:
        w, s = calculated_crossing(q, float(w0) + 0.3, LAM, wedge_deg=W)
        assert w == pytest.approx(float(w0), abs=1e-6) and abs(np.linalg.norm(s) - 1.0) < 1e-12


def test_predicted_position_sits_on_the_ewald_sphere_for_midas_defects_own_inverse():
    W = 0.35
    g = geom(wedge_deg=W, tz_deg=-20.0)
    U = Rot.from_rotvec([0.4, -0.2, 0.7]).as_matrix()
    hkl = np.array([[1, 2, 3], [-3, 1, 4], [2, -2, 5], [0, 3, 6]])
    scan = Scan(-150.0, 0.2, 850)
    n_ok = 0
    for h in hkl:
        q_s = 2 * np.pi * (U @ reciprocal_basis(LAT) @ h.astype(float))
        for w in np.degrees(ewald_crossings(q_s, LAM, W)):
            fr, ro, co, ok = predict_positions(U, LAT, h[None], np.array([w]), g, scan)
            if not ok[0] or not (0 <= ro[0] < 1679 and 0 <= co[0] < 1475):
                continue
            n_ok += 1
            ql = pixel_to_qlab(ro, co, g, device="cpu", dtype=torch.float64)
            qs = qlab_to_qsample(ql, torch.as_tensor([np.radians(w)], dtype=torch.float64), np.radians(W)).numpy()[0]
            assert np.allclose(qs, q_s, atol=1e-6), (h, w)
            assert fr[0] == pytest.approx((w - scan.omega_start) / scan.omega_step - 0.5, abs=1e-9)
    assert n_ok >= 1


def _synthetic_spots(g_true, U, scan, rng, noise_px=0.2, n_max=1200):
    rows = []
    for h in range(-4, 5):
        for k in range(-4, 5):
            for l in range(-8, 9):
                if (h, k, l) == (0, 0, 0):
                    continue
                q_s = 2 * np.pi * (U @ reciprocal_basis(LAT) @ np.array([h, k, l], float))
                for w in np.degrees(ewald_crossings(q_s, LAM, g_true.wedge_deg)):
                    for shift in (0.0, 360.0, -360.0):
                        if min(scan.omega_start, scan.omega_end) <= w + shift <= max(scan.omega_start, scan.omega_end):
                            rows.append((h, k, l, w + shift))
    d = pd.DataFrame(rows, columns=["h", "k", "l", "omega"])
    fr, ro, co, ok = predict_positions(U, LAT, d[["h", "k", "l"]].to_numpy(), d.omega.to_numpy(), g_true, scan)
    d = d.assign(frame=fr, row=ro, col=co)[ok & (ro > 5) & (ro < 1673) & (co > 5) & (co < 1469)]
    d = d.sample(min(len(d), n_max), random_state=0)
    return d.assign(row=d.row + rng.normal(0, noise_px, len(d)), col=d.col + rng.normal(0, noise_px, len(d)), frame=d.frame + rng.normal(0, 0.1, len(d)))


def test_the_wedge_and_the_tilts_are_recovered_from_positions_and_neglecting_them_costs_pixels():
    rng = np.random.default_rng(3)
    scan = Scan(-150.0, 0.2, 850)
    U = Rot.from_rotvec([0.3, 0.2, -0.5]).as_matrix()
    g_true = geom(wedge_deg=0.30, tz_deg=-20.0 + 0.10, tx_deg=0.15, ty_deg=-0.08)
    spots = _synthetic_spots(g_true, U, scan, rng)
    assert len(spots) > 300
    g_start = replace(g_true, wedge_deg=0.0, tz_deg=-20.0, tx_deg=0.0, ty_deg=0.0)
    U_start = Rot.from_rotvec([0.0005, -0.0004, 0.0006]).as_matrix() @ U
    plain = index_and_refine(spots, LAT, U_start, g_start, scan)
    full = index_and_refine(spots, LAT, U_start, g_start, scan, refine_tilts=True, refine_wedge=True)
    assert full.converged and full.rms_px < 0.5, (full.reason, full.rms_px)
    assert full.geom.wedge_deg == pytest.approx(0.30, abs=0.03)
    assert full.geom.tx_deg == pytest.approx(0.15, abs=0.05) and full.geom.ty_deg == pytest.approx(-0.08, abs=0.05)
    assert plain.rms_px > 2.0 * full.rms_px or not plain.converged, (plain.rms_px, full.rms_px)
    wedge_only = index_and_refine(spots, LAT, U_start, replace(g_start, tx_deg=0.15, ty_deg=-0.08, tz_deg=-19.9), scan, refine_wedge=True)
    assert wedge_only.geom.wedge_deg == pytest.approx(0.30, abs=0.03) and wedge_only.rms_px < 0.5


def test_assign_hkl_uses_the_wedge():
    W = 0.4
    g = geom(wedge_deg=W)
    U = Rot.from_rotvec([0.3, 0.2, -0.5]).as_matrix()
    scan = Scan(-150.0, 0.2, 850)
    d = _synthetic_spots(g, U, scan, np.random.default_rng(0), noise_px=0.0, n_max=200)
    s, tth, _ = spot_directions(d.row.to_numpy(), d.col.to_numpy(), g)
    om = scan.omega_of(d.frame.to_numpy())
    hkl, _, r_w = assign_hkl(s, om, LAT, U, LAM, W)
    _, _, r_0 = assign_hkl(s, om, LAT, U, LAM, 0.0)
    assert np.median(r_w) < 0.02 and np.median(r_0) > 3 * np.median(r_w)
