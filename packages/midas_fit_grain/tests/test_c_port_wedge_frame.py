"""Issues #11/#17/#18: the C-parity port under a wedge, and midas_diffract.

``c_port._displacement_in_the_spot`` (port of C ``DisplacementInTheSpot``)
computes ``R_y(-W) @ R_z(omega) @ pos``: the rigid-body rotation about the
tilted axis ``n = R_y(-W) e_z = (-sin W, 0, cos W)`` of a position expressed
in the rotation-STAGE frame. ``_correct_for_ome`` maps G with the same
matrix, so G and position move together -- the grain is a rigid body.

That is THE MIDAS Wedge convention (``midas_diffract.forward`` module doc),
and since 2026-09 midas_diffract implements it too, so the relation is the
identity: same ``Wedge``, same stage-frame position and orientation. (It
used to be ``W_C = -W_md, p_stage = R_y(-W_md) p_md``.)

#18: ``_correct_for_ome`` must return the spot of the WEDGE-FREE geometry,
``(y, z, omega)`` of ``R_z(omega') g`` -- eta included. It kept the observed
eta, and the refiner turned that eta error into ~440 um of grain position.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_fit_grain import c_port

DEG2RAD = math.pi / 180.0
LSD = 1.0e6


def _Ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    kx, ky, kz = axis
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _ray_displacement(P: np.ndarray, y_obs: float, z_obs: float):
    """Undisplaced-spot offset for a grain at LAB position P, by explicit
    ray/plane geometry: the ray from P to (Lsd, y_obs, z_obs), extended back
    to the plane x = 0, crosses it at (P - P_x k/k_x); the spot of a grain at
    the origin with the same k lands P_yz - P_x k_yz/k_x lower."""
    k = np.array([LSD - P[0], y_obs - P[1], z_obs - P[2]])
    return P[1] - P[0] * k[1] / k[0], P[2] - P[0] * k[2] / k[0]


POS = [np.array([650.0, -300.0, 140.0]), np.array([-500.0, 450.0, -225.0]),
       np.array([120.0, 700.0, 60.0])]
SPOTS = [(-92217.8, 23566.1, -39.4), (81836.3, 47635.9, 23.4),
         (15000.0, -120000.0, 146.1), (-60000.0, -60000.0, -170.0)]


def test_w0_displacement_is_about_z():
    for p in POS:
        for (y, z, ome) in SPOTS:
            got = c_port._displacement_in_the_spot(p[0], p[1], p[2], LSD, y, z, ome, 0.0, 0.0)
            want = _ray_displacement(_Ry(0.0) @ _rodrigues(np.array([0.0, 0.0, 1.0]),
                                                            ome * DEG2RAD) @ p, y, z)
            assert abs(got[0] - want[0]) < 1e-9 and abs(got[1] - want[1]) < 1e-9


@pytest.mark.parametrize("w_c", [2.0, -3.0, 0.05])
def test_displacement_is_rigid_rotation_about_tilted_stage_axis(w_c):
    """Position rotates about n_C = (-sin W, 0, cos W) (Rodrigues), in the
    stage frame (fixed tilt R_y(-W)) -- the same matrix used for G."""
    W = w_c * DEG2RAD
    n_c = np.array([-math.sin(W), 0.0, math.cos(W)])
    worst_about_z = 0.0
    for p in POS:
        for (y, z, ome) in SPOTS:
            P_lab = _rodrigues(n_c, ome * DEG2RAD) @ (_Ry(-W) @ p)
            want = _ray_displacement(P_lab, y, z)
            got = c_port._displacement_in_the_spot(p[0], p[1], p[2], LSD, y, z, ome, w_c, 0.0)
            assert abs(got[0] - want[0]) < 1e-9 and abs(got[1] - want[1]) < 1e-9
            zonly = _ray_displacement(_rodrigues(np.array([0.0, 0.0, 1.0]), ome * DEG2RAD) @ p, y, z)
            worst_about_z = max(worst_about_z, abs(got[1] - zonly[1]))
    # Null can fail: an about-untilted-z rotation is measurably different.
    assert worst_about_z > 1e-3


def _md_model(wedge_deg):
    md = pytest.importorskip("midas_diffract.forward")
    WL, A, PX, BC = 0.172979, 4.08, 200.0, 1024.0
    ints = [(h, k, l) for h in range(-3, 4) for k in range(-3, 4) for l in range(-3, 4)
            if (h, k, l) != (0, 0, 0) and (h % 2, k % 2, l % 2) in ((0, 0, 0), (1, 1, 1))]
    hk = torch.tensor(ints, dtype=torch.float64) / A
    th = torch.asin(WL * hk.norm(dim=1) / 2)
    geom = md.HEDMGeometry(Lsd=[LSD], y_BC=[BC], z_BC=[BC], px=PX, omega_start=-180,
                           omega_step=0.25, n_frames=1440, n_pixels_y=2048, n_pixels_z=2048,
                           min_eta=6, wavelength=WL, flip_y=True, wedge=wedge_deg)
    return md.HEDMForwardModel(hk, th, geom), WL, PX, BC


def _md_spots(m, OM, pos, PX, BC):
    om, eta, tt, v = m.calc_bragg_geometry(OM)
    sp = m.project_to_detector(om, eta, tt, torch.tensor(pos, dtype=torch.float64).view(1, 1, 3), v)
    shape = tuple(om.shape)
    y = ((BC - sp.y_pixel) * PX).reshape(shape).numpy()
    z = ((sp.z_pixel - BC) * PX).reshape(shape).numpy()
    ok = (v.numpy() > 0.5) & (sp.valid.reshape(shape).numpy() > 0.5)
    return om.numpy() / DEG2RAD, y, z, ok


@pytest.mark.parametrize("w", [3.0, -3.0])
def test_wedge_sign_and_frame_match_midas_diffract(w):
    """c_port reproduces midas_diffract's (Rodrigues-verified) spot shifts
    exactly with the SAME Wedge and the SAME stage-frame position; the other
    sign, or a position in a frame tilted by R_y(W), is measurably wrong."""
    m, WL, PX, BC = _md_model(w)
    OM = m.euler2mat(torch.tensor([[0.3, 0.5, 0.7]], dtype=torch.float64))
    pA = POS[0]
    ome, y0, z0, ok0 = _md_spots(m, OM, np.zeros(3), PX, BC)
    _, y1, z1, ok1 = _md_spots(m, OM, pA, PX, BC)
    ok = ok0 & ok1
    assert ok.sum() > 50

    def worst(p_stage, w_c):
        e = 0.0
        for idx in zip(*np.nonzero(ok)):
            dY, dZ = c_port._displacement_in_the_spot(
                p_stage[0], p_stage[1], p_stage[2], LSD, float(y1[idx]), float(z1[idx]),
                float(ome[idx]), w_c, 0.0)
            e = max(e, abs(y1[idx] - dY - y0[idx]), abs(z1[idx] - dZ - z0[idx]))
        return e

    assert worst(pA, w) < 1e-6                           # um
    assert worst(pA, -w) > 1.0                           # the other sign is wrong
    assert worst(_Ry(-w * DEG2RAD) @ pA, w) > 1.0        # so is a tilted frame


@pytest.mark.parametrize("w", [2.0, -0.5])
def test_correct_for_ome_returns_the_wedge_free_spot(w):
    """#18. Raw spots of a grain at the origin simulated with Wedge W, pushed
    through ``_correct_for_ome(W)``, must land on the W = 0 spots of the same
    stage-frame orientation: omega AND (y, z), i.e. eta too; and the G it
    returns is O h. Before the fix (y, z) kept the raw eta and missed by
    up to ~W * Lsd * tan(2 theta)-scale amounts."""
    mW, WL, PX, BC = _md_model(w)
    m0, _, _, _ = _md_model(0.0)
    OM = mW.euler2mat(torch.tensor([[1.9, 1.1, 4.2]], dtype=torch.float64))
    omW, yW, zW, okW = _md_spots(mW, OM, np.zeros(3), PX, BC)
    om0, y0, z0, ok0 = _md_spots(m0, OM, np.zeros(3), PX, BC)
    hk = mW.hkls.to(torch.float64).numpy()
    G = (OM[0].numpy() @ hk.T).T                       # stage-frame G per hkl
    n = 0
    worst_yz, worst_om, worst_g, worst_null = 0.0, 0.0, 0.0, 0.0
    for idx in zip(*np.nonzero(okW)):
        ys, zs, oc, g1, g2, g3 = c_port._correct_for_ome(
            float(yW[idx]), float(zW[idx]), LSD, float(omW[idx]), WL, w)
        # same hkl, W = 0: the branch whose omega is nearest the corrected one
        mm = idx[-1]
        cands = [k for k in range(om0.shape[0]) if ok0[k, mm]]
        if not cands:
            continue
        k0 = min(cands, key=lambda k: abs(((om0[k, mm] - oc + 180) % 360) - 180))
        d_om = abs(((om0[k0, mm] - oc + 180) % 360) - 180)
        if d_om > 1.0:
            continue
        n += 1
        worst_om = max(worst_om, d_om)
        worst_yz = max(worst_yz, abs(ys - y0[k0, mm]), abs(zs - z0[k0, mm]))
        gv = np.array([g1, g2, g3]) / math.sqrt(g1 * g1 + g2 * g2 + g3 * g3)
        gt = G[mm] / np.linalg.norm(G[mm])
        worst_g = max(worst_g, float(np.degrees(np.linalg.norm(np.cross(gv, gt)))))
        # Null: the pre-fix answer kept the RAW eta at the corrected radius.
        r = math.hypot(ys, zs)
        eta_raw = math.atan2(-float(yW[idx]), float(zW[idx]))
        worst_null = max(worst_null, abs(-r * math.sin(eta_raw) - y0[k0, mm]),
                         abs(r * math.cos(eta_raw) - z0[k0, mm]))
    assert n > 40
    assert worst_om < 1e-6, worst_om                    # deg
    assert worst_yz < 1e-3, worst_yz                    # um
    assert worst_g < 1e-8, worst_g                      # deg
    assert worst_null > 100.0, worst_null               # um: the null fails
