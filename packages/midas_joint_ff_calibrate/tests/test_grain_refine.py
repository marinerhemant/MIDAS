"""Synthetic proof that grain spots recover the powder-blind tx.

We build a known geometry, forward-model several grains in ideal space, then
bake a known ``tx_true`` into the RAW spot pixels by applying the inverse of the
pure-tx detector rotation (verified to be an exact 2-D rotation when
ty=tz=0, distortion=0). The refiner — which re-DetCors raw pixels with a trial
tx via the same ``pixel_to_REta_torch`` and matches against the ideal forward
model — must recover ``tx_true`` starting from tx=0.

Also pins the refine-mask (only the requested geometry + grain blocks refine)
and that a radial-only signal can't see tx (so the loss must be η-sensitive).
"""
import math

import numpy as np
import pytest
import torch

from midas_calibrate.geometry_torch import build_tilt_matrix_torch
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
LSD, BCY, BCZ, PX = 1.0e6, 1024.0, 1024.0, 150.0
RHOD = 1024.0 * PX
NPIX = 2048


def _model():
    sg = SpaceGroup.from_number(225)            # Ni FCC
    lat = Lattice(3.6, 3.6, 3.6, 90.0, 90.0, 90.0)
    hkls_cart, thetas, hkls_int = hkls_for_forward_model(
        sg, lat, wavelength_A=0.2066, two_theta_max_deg=18.0, expand_equivalents=True)
    geom = HEDMGeometry(
        Lsd=LSD, y_BC=BCY, z_BC=BCZ, px=PX, omega_start=-180.0, omega_step=0.25,
        n_frames=1440, n_pixels_y=NPIX, n_pixels_z=NPIX, min_eta=6.0,
        wavelength=0.2066, tx=0.0, ty=0.0, tz=0.0, wedge=0.0,
        flip_y=True, apply_tilts=False, multi_mode="layered")
    return HEDMForwardModel(hkls_cart, thetas, geom, hkls_int=hkls_int.float())


def _bake_tx_into_raw(yp_ideal, zp_ideal, tx_true_deg):
    """raw pixels s.t. DetCor(raw, tx_true) == ideal. DetCor applies tilt(tx) to
    (Yc,Zc); so raw's (Yc,Zc) = tilt(tx_true)^{-1} · ideal (Yc,Zc)."""
    T = build_tilt_matrix_torch(torch.tensor(tx_true_deg, dtype=DT),
                                torch.tensor(0.0, dtype=DT), torch.tensor(0.0, dtype=DT))
    Rot = T[1:, 1:]                              # 2x2 rotation on (Yc, Zc)
    Yc = (-yp_ideal + BCY) * PX
    Zc = (zp_ideal - BCZ) * PX
    yzc = torch.stack([Yc, Zc], dim=-1) @ Rot    # = Rot^T·col? use inverse below
    # inverse rotation: raw = Rot^{-1} · ideal = Rot^T · ideal
    raw = torch.stack([Yc, Zc], dim=-1) @ Rot    # Rot is orthonormal; @Rot == ·Rot
    Yc_raw, Zc_raw = raw[..., 0], raw[..., 1]
    yp_raw = BCY - Yc_raw / PX
    zp_raw = BCZ + Zc_raw / PX
    return yp_raw, zp_raw


def _build_synth(tx_true_deg, n_grains=8, seed=0, max_spots=40):
    """Forward-model grains, select valid spots, bake tx_true into raw pixels,
    and assemble observations/matches/raw_yz (identity match: obs == pred)."""
    model = _model()
    rng = np.random.default_rng(seed)
    eulers = rng.uniform(-math.pi, math.pi, size=(n_grains, 3))
    eulers[:, 1] = rng.uniform(0, math.pi, size=n_grains)     # Phi ∈ [0, π]
    positions = np.zeros((n_grains, 3))
    lattices = np.tile(np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]), (n_grains, 1))

    observations, matches, raw_yz = [], [], []
    eul_t = torch.from_numpy(eulers).to(DT)
    pos_t = torch.from_numpy(positions).to(DT)
    lat_t = torch.from_numpy(lattices).to(DT)
    for g in range(n_grains):
        s = model(eul_t[g].view(1, 1, 3), pos_t[g].view(1, 1, 3),
                  lattice_params=lat_t[g].view(1, 6))

        def sq(t):
            while t.dim() > 2 and t.shape[0] == 1:
                t = t.squeeze(0)
            return t
        valid = sq(s.valid).bool()
        yp = sq(s.y_pixel).double(); zp = sq(s.z_pixel).double()
        om = sq(s.omega).double(); eta = sq(s.eta).double(); tth = sq(s.two_theta).double()
        K, M = valid.shape
        ks, msq = torch.where(valid)
        if ks.numel() == 0:
            continue
        if ks.numel() > max_spots:
            sel = torch.randperm(ks.numel())[:max_spots]
            ks, msq = ks[sel], msq[sel]
        flat = ks * M + msq
        yp_i = yp.reshape(-1)[flat]; zp_i = zp.reshape(-1)[flat]
        om_i = om.reshape(-1)[flat]; eta_i = eta.reshape(-1)[flat]; tth_i = tth.reshape(-1)[flat]
        yp_raw, zp_raw = _bake_tx_into_raw(yp_i, zp_i, tx_true_deg)

        # Observed lab (Y,Z) for the (Y,Z) position loss (make_residual): the
        # ideal predicted (Y,Z) = (-R sinη, R cosη), R = Lsd·tan(2θ), rotated
        # about the beam by -tx_true. make_residual rotates the observed by the
        # trial tx and matches to the ideal prediction, so this construction
        # has its (Y,Z) minimum exactly at tx = tx_true.
        R_id = LSD * torch.tan(tth_i)
        Yp_id = -R_id * torch.sin(eta_i)
        Zp_id = R_id * torch.cos(eta_i)
        T_tx = build_tilt_matrix_torch(torch.tensor(tx_true_deg, dtype=DT),
                                       torch.tensor(0.0, dtype=DT),
                                       torch.tensor(0.0, dtype=DT))
        Rot2 = T_tx[1:, 1:]                       # 2×2 rotation on (Y,Z)
        pred_vec = torch.stack([Yp_id, Zp_id], dim=-1)
        obs_vec = pred_vec @ Rot2                 # = Rot2^T · pred  ⇒ Rot2·obs = pred
        y_lab_i, z_lab_i = obs_vec[..., 0], obs_vec[..., 1]

        S = ks.numel()
        observations.append(ObservedSpots(
            spot_id=torch.arange(S), ring_nr=torch.zeros(S, dtype=torch.int64),
            y_lab=y_lab_i, z_lab=z_lab_i,
            omega=om_i, eta=eta_i, two_theta=tth_i,
            grain_radius=torch.full((S,), 50.0, dtype=DT),
            fit_rmse=torch.zeros(S, dtype=DT), y_orig=torch.zeros(S, dtype=DT),
            z_orig=torch.zeros(S, dtype=DT), omega_ini=om_i.clone(),
            mask_touched=torch.zeros(S, dtype=torch.bool)))
        matches.append(MatchResult(
            k_idx=ks.long(), m_idx=msq.long(), mask=torch.ones(S, dtype=torch.bool),
            delta_omega=torch.zeros(S, dtype=DT), delta_eta=torch.zeros(S, dtype=DT)))
        raw_yz.append((yp_raw, zp_raw))
    return model, observations, matches, raw_yz, eulers, positions, lattices


def _fixed_geo():
    return dict(
        Lsd=torch.tensor(LSD, dtype=DT), BC_y=torch.tensor(BCY, dtype=DT),
        BC_z=torch.tensor(BCZ, dtype=DT), ty=torch.tensor(0.0, dtype=DT),
        tz=torch.tensor(0.0, dtype=DT), px=torch.tensor(PX, dtype=DT),
        RhoD=torch.tensor(RHOD, dtype=DT), p_coeffs=torch.zeros(15, dtype=DT))


def _spec(eulers, positions, lattices, tx_init=0.0):
    spec = mp.ParameterSpec()
    spec.add(Parameter("tx", init=torch.tensor(tx_init, dtype=DT), refined=True,
                       bounds=(-5.0, 5.0)))
    # Mirror production refine_geometry_from_grains: grain pose is FIXED for the
    # tx step. A free grain orientation rotates the predicted pattern about the
    # beam exactly as tx rotates the observed one, re-absorbing tx (the
    # divergence we saw) — so only tx is refined here.
    spec = build_joint_spec(
        powder_spec=spec,
        grain_eulers_init=torch.from_numpy(eulers).to(DT),
        grain_positions_init=torch.from_numpy(positions).to(DT),
        grain_lattices_init=torch.from_numpy(lattices).to(DT),
        refine_grain_orientation=False, refine_grain_position=False,
        refine_grain_strain=False)
    return spec


def test_recover_tx_angular():
    tx_true = 0.40
    model, obs, matches, raw_yz, eulers, pos, lat = _build_synth(tx_true)
    assert sum(int(m.mask.sum()) for m in matches) > 50
    spec = _spec(eulers, pos, lat, tx_init=0.0)
    resid = make_residual(model, obs, matches, raw_yz, fixed_geo=_fixed_geo(), kind="angular")
    u0 = {n: spec.parameters[n].init_tensor() for n in spec.parameters}
    cost0 = float((resid(u0) ** 2).sum())
    u, cost, rc = mp.lm_minimise(spec, resid,
                                 config=mp.GenericLMConfig(max_iter=80, ftol_rel=1e-12,
                                                           xtol_rel=1e-12),
                                 fallback_span=2.0)
    tx_rec = float(u["tx"])
    assert abs(tx_rec - tx_true) < 1e-2, f"recovered tx={tx_rec:.4f}, true={tx_true}"
    assert cost < cost0 * 1e-2, f"cost {cost0:.3e} -> {cost:.3e} (insufficient drop)"


def test_recover_tx_internal_angle():
    """The g-vector-angle loss also recovers tx (η enters via the g-vector)."""
    tx_true = 0.35
    model, obs, matches, raw_yz, eulers, pos, lat = _build_synth(tx_true, seed=5)
    spec = _spec(eulers, pos, lat, tx_init=0.0)
    resid = make_residual(model, obs, matches, raw_yz, fixed_geo=_fixed_geo(),
                          kind="internal_angle")
    u0 = {n: spec.parameters[n].init_tensor() for n in spec.parameters}
    cost0 = float((resid(u0) ** 2).sum())
    u, cost, rc = mp.lm_minimise(spec, resid,
                                 config=mp.GenericLMConfig(max_iter=80, ftol_rel=1e-12,
                                                           xtol_rel=1e-12),
                                 fallback_span=2.0)
    assert abs(float(u["tx"]) - tx_true) < 1e-2
    assert cost < cost0 * 1e-2


def test_residual_is_tx_sensitive_and_kind_agnostic():
    """make_residual now uses the FitMultipleGrains (Y,Z) position loss for any
    ``kind`` (kind is cosmetic). The qualifying property is that the loss is
    tx-SENSITIVE: ~0 at the true tx, large at tx=0. (A radial/pixel loss would
    be blind to tx — the failure mode this guards against.)"""
    tx_true = 0.40
    model, obs, matches, raw_yz, eulers, pos, lat = _build_synth(tx_true, n_grains=3)
    fg = _fixed_geo()
    for kind in ("angular", "internal_angle", "pixel"):
        resid = make_residual(model, obs, matches, raw_yz, fixed_geo=fg, kind=kind)
        base = {"grain_euler": torch.from_numpy(eulers).to(DT),
                "grain_pos": torch.from_numpy(pos).to(DT),
                "grain_lattice": torch.from_numpy(lat).to(DT)}
        cost_true = float((resid({**base, "tx": torch.tensor(tx_true, dtype=DT)}) ** 2).sum())
        cost_zero = float((resid({**base, "tx": torch.tensor(0.0, dtype=DT)}) ** 2).sum())
        assert cost_true < 1e-6, f"kind={kind}: residual not ~0 at true tx ({cost_true:.2e})"
        assert cost_zero > 1e3 * max(cost_true, 1e-12), f"kind={kind}: loss not tx-sensitive"


def test_radial_only_is_blind_to_tx():
    """ΔR from tx is ~0: the radial (2θ) residual component barely moves with tx,
    confirming the loss must use η. (Guards against re-enabling a pixel loss.)"""
    tx_true = 0.40
    model, obs, matches, raw_yz, eulers, pos, lat = _build_synth(tx_true)
    fg = _fixed_geo()
    from midas_calibrate.geometry_torch import pixel_to_REta_torch
    Yr, Zr = raw_yz[0]
    R0, _ = pixel_to_REta_torch(Yr, Zr, Lsd=fg["Lsd"], BC_y=fg["BC_y"], BC_z=fg["BC_z"],
                                tx=torch.tensor(0.0, dtype=DT), ty=fg["ty"], tz=fg["tz"],
                                p_coeffs=fg["p_coeffs"], parallax=torch.zeros((), dtype=DT),
                                px=fg["px"], rho_d=fg["RhoD"])
    Rt, _ = pixel_to_REta_torch(Yr, Zr, Lsd=fg["Lsd"], BC_y=fg["BC_y"], BC_z=fg["BC_z"],
                                tx=torch.tensor(tx_true, dtype=DT), ty=fg["ty"], tz=fg["tz"],
                                p_coeffs=fg["p_coeffs"], parallax=torch.zeros((), dtype=DT),
                                px=fg["px"], rho_d=fg["RhoD"])
    assert (R0 - Rt).abs().max() * PX < 5.0  # < 5 µm radial shift for 0.4° tx


def test_refine_mask_only_requested():
    """Only tx is refined for the tx step; grain pose is held fixed (matches
    production refine_geometry_from_grains, which fixes pose to avoid the
    tx↔orientation degeneracy)."""
    _, _, _, _, eulers, pos, lat = _build_synth(0.1, n_grains=3)
    spec = _spec(eulers, pos, lat)
    assert set(spec.refined_names()) == {"tx"}


def test_stripped_paramstest_without_omegastep_errors(tmp_path):
    """Regression: passing the c-omp pipeline's STRIPPED per-layer paramstest
    (no OmegaStep / NrPixels) must raise a clear error, not silently default
    OmegaStep→0 and produce "matched spots=0" / tx=0.

    The guard fires before grains are loaded, so a bare layer dir is enough.
    """
    from midas_joint_ff_calibrate.grain_refine import refine_geometry_from_grains

    # A paramstest that has geometry but NO OmegaStep — exactly what the c-omp
    # refiner's paramstest.txt looks like.
    ps = tmp_path / "paramstest.txt"
    ps.write_text(
        "LatticeParameter 3.6 3.6 3.6 90 90 90;\n"
        "SpaceGroup 225;\n"
        "Wavelength 0.2066;\n"
        "Distance 959886.899;\n"
        "px 150.0;\n"
        "BC 1391.14 1422.36;\n"
        "tx 0;\nty -0.198;\ntz 0.324;\n"
        "RhoD 312493.0;\n"
        "MaxRingRad 312493.0;\n"
        "OmegaRange -180 180;\n"
    )
    with pytest.raises(ValueError, match="OmegaStep"):
        refine_geometry_from_grains(ps, tmp_path, refine_params=("tx",))
