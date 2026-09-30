"""Small-Wedge recovery with OFF-AXIS grains on the absolute-Wedge path.

``test_grain_refine._build_synth`` puts every grain at the origin and never
exercises ``make_residual(wedge_absolute=True)``: the raw-frame path that
predicts each spot as the ray from the grain's rotated position
``R_y(-W) R_z(omega) p``. This test forward-models grains spread over +-800 um
at a true Wedge of -0.03 deg (the magnitude seen on real AlON s6 data),
takes the predicted pixels as the RAW observation, and asks the grain-tx
objective -- seeded at Wedge 0, tx 0, poses held at truth -- to recover it.

The residual at the truth must also vanish: that pins the inline position
rotation in ``make_residual`` to ``HEDMForwardModel.project_to_detector``.
"""
import math

import numpy as np
import torch

from midas_diffract import HEDMForwardModel
from midas_diffract.forward import HEDMGeometry
from midas_diffract.hkls import hkls_for_forward_model
from midas_hkls import Lattice, SpaceGroup
from midas_fit_grain.matching import MatchResult
from midas_fit_grain.observations import ObservedSpots

import midas_peakfit as mp
from midas_peakfit import Parameter
from midas_joint_ff_calibrate.grain_refine import make_residual
from midas_joint_ff_calibrate.spec import build_joint_spec

DT = torch.float64
LSD, BCY, BCZ, PX = 1.0e6, 1024.0, 1024.0, 200.0
RHOD = 1024.0 * PX
NPIX = 2048
LAT = (3.6, 3.6, 3.6, 90.0, 90.0, 90.0)
WL = 0.2066


def _model(wedge):
    sg = SpaceGroup.from_number(225)
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg, Lattice(*LAT), wavelength_A=WL, two_theta_max_deg=14.0,
        expand_equivalents=True)
    geom = HEDMGeometry(
        Lsd=LSD, y_BC=BCY, z_BC=BCZ, px=PX, omega_start=-180.0, omega_step=0.25,
        n_frames=1440, n_pixels_y=NPIX, n_pixels_z=NPIX, min_eta=6.0,
        wavelength=WL, tx=0.0, ty=0.0, tz=0.0, wedge=wedge,
        flip_y=True, apply_tilts=False, multi_mode="layered")
    return HEDMForwardModel(hkls_cart, thetas, geom, hkls_int=hkls_int.float())


def _sq(t):
    while t.dim() > 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def _synth(w_true, n_grains=10, seed=3, max_spots=60, pos_scale=800.0):
    truth = _model(w_true)
    rng = np.random.default_rng(seed)
    eulers = rng.uniform(-math.pi, math.pi, size=(n_grains, 3))
    eulers[:, 1] = np.arccos(rng.uniform(-1, 1, n_grains))
    positions = rng.uniform(-pos_scale, pos_scale, size=(n_grains, 3))
    positions[:, 2] = rng.uniform(-50, 50, n_grains)
    lattices = np.tile(np.array(LAT), (n_grains, 1))
    obs, matches, raw_yz = [], [], []
    for g in range(n_grains):
        s = truth(torch.tensor(eulers[g], dtype=DT).view(1, 1, 3),
                  torch.tensor(positions[g], dtype=DT).view(1, 1, 3),
                  lattice_params=torch.tensor(LAT, dtype=DT).view(1, 6))
        valid = _sq(s.valid).bool()
        ks, ms = torch.where(valid)
        if ks.numel() > max_spots:
            sel = torch.from_numpy(rng.permutation(ks.numel())[:max_spots])
            ks, ms = ks[sel], ms[sel]
        M = valid.shape[1]
        flat = ks * M + ms
        pick = lambda t: _sq(t).double().reshape(-1)[flat]
        yp, zp = pick(s.y_pixel), pick(s.z_pixel)
        om, eta, tth = pick(s.omega), pick(s.eta), pick(s.two_theta)
        S = ks.numel()
        obs.append(ObservedSpots(
            spot_id=torch.arange(S), ring_nr=torch.zeros(S, dtype=torch.int64),
            y_lab=(BCY - yp) * PX, z_lab=(zp - BCZ) * PX,
            omega=om, eta=eta, two_theta=tth,
            grain_radius=torch.full((S,), 50.0, dtype=DT),
            fit_rmse=torch.zeros(S, dtype=DT), y_orig=torch.zeros(S, dtype=DT),
            z_orig=torch.zeros(S, dtype=DT), omega_ini=om.clone(),
            mask_touched=torch.zeros(S, dtype=torch.bool)))
        matches.append(MatchResult(
            k_idx=ks.long(), m_idx=ms.long(), mask=torch.ones(S, dtype=torch.bool),
            delta_omega=torch.zeros(S, dtype=DT), delta_eta=torch.zeros(S, dtype=DT)))
        raw_yz.append((yp, zp))
    return obs, matches, raw_yz, eulers, positions, lattices


def _fixed_geo():
    return dict(
        Lsd=torch.tensor(LSD, dtype=DT), BC_y=torch.tensor(BCY, dtype=DT),
        BC_z=torch.tensor(BCZ, dtype=DT), ty=torch.tensor(0.0, dtype=DT),
        tz=torch.tensor(0.0, dtype=DT), px=torch.tensor(PX, dtype=DT),
        RhoD=torch.tensor(RHOD, dtype=DT), p_coeffs=torch.zeros(15, dtype=DT))


def _spec(eulers, positions, lattices, w0=0.0):
    spec = mp.ParameterSpec()
    spec.add(Parameter("tx", init=torch.tensor(0.0, dtype=DT), refined=True,
                       bounds=(-5.0, 5.0)))
    spec.add(Parameter("Wedge", init=torch.tensor(w0, dtype=DT), refined=True,
                       bounds=(-5.0, 5.0)))
    return build_joint_spec(
        powder_spec=spec,
        grain_eulers_init=torch.from_numpy(eulers).to(DT),
        grain_positions_init=torch.from_numpy(positions).to(DT),
        grain_lattices_init=torch.from_numpy(lattices).to(DT),
        refine_grain_orientation=False, refine_grain_position=False,
        refine_grain_strain=False)


def _fit(w_true, w0):
    obs, matches, raw_yz, eul, pos, lat = _synth(w_true)
    model = _model(w0)            # built at the file's Wedge, as grain-tx does
    resid = make_residual(model, obs, matches, raw_yz, fixed_geo=_fixed_geo(),
                          observed_from_raw=True, wedge_absolute=True)
    spec = _spec(eul, pos, lat, w0=w0)
    u_truth = {n: spec.parameters[n].init_tensor() for n in spec.parameters}
    u_truth["Wedge"] = torch.tensor(w_true, dtype=DT)
    r_truth = resid(u_truth)
    u, cost, rc = mp.lm_minimise(
        spec, resid, config=mp.GenericLMConfig(max_iter=60, ftol_rel=1e-14,
                                               xtol_rel=1e-14),
        fallback_span=2.0)
    return float(u["Wedge"]), float(u["tx"]), r_truth, sum(int(m.mask.sum()) for m in matches)


def test_residual_vanishes_at_true_wedge_offaxis():
    *_, r_truth, n = _fit(-0.03, 0.0)
    assert n > 200
    # predicted spot = observed pixel to well under a micrometre at the truth
    assert float(r_truth.abs().max()) < 1e-3, float(r_truth.abs().max())


def test_recover_small_wedge_offaxis_from_zero():
    w_rec, tx_rec, _, _ = _fit(-0.03, 0.0)
    assert abs(w_rec - (-0.03)) < 1e-4, w_rec
    assert abs(tx_rec) < 1e-4, tx_rec


def test_recover_small_wedge_offaxis_from_other_side():
    w_rec, tx_rec, _, _ = _fit(-0.03, -0.1)
    assert abs(w_rec - (-0.03)) < 1e-4, w_rec
