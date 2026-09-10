"""End-to-end: does index_from_cloud predict from the CONVERGED cell?

INVERSE CRIME, and deliberately so: observations are generated with the same
forward model that is later fitted. This tests PLUMBING -- that the refined cell
reaches the prediction -- not physics. A physics test would need an independent
generator. Labelled here so nobody reads it as validation of the forward model.

Before 2026-09-07 `index_from_cloud` computed a refined cell in
`find_seed_orientation(refine_lattice=True)` and then built the forward model
from the NOMINAL crystal, discarding it. This is the regression test for that.
"""
from __future__ import annotations
import numpy as np
import pytest
from midas_hkls import Lattice, Crystal, SpaceGroup, Atom
from midas_defect.geometry import Geometry
from midas_defect.indexing import (index_from_cloud, build_forward_model,
                                   predict_spots)


def _crystal(a, c, sg=139):
    return Crystal(lattice=Lattice(a=a, b=a, c=c, alpha=90., beta=90., gamma=90.),
                   space_group=SpaceGroup.from_number(sg),
                   atoms=[Atom(element="La", fract=(0., 0., 0.5), label="La1"),
                          Atom(element="Ni", fract=(0., 0., 0.098), label="Ni1"),
                          Atom(element="O", fract=(0., 0.5, 0.096), label="O1")])


def _geom():
    return Geometry(lsd_um=349682.0, bcy_px=737.06, bcz_px=810.32, px_um=172.0,
                    wavelength_A=0.42459, n_pix_y=1475, n_pix_z=1679,
                    omega_first_deg=-19.5, omega_step_deg=1.0, n_frames=40,
                    ty_deg=-0.227, tz_deg=-0.308)


def _observations(cry, g, U, d_min):
    """Forward-predict, keep what lands on the detector -> (row, col, frame)."""
    model, hkls, _ = build_forward_model(cry, g, d_min=d_min, device="cpu")
    sp = predict_spots(model, U)
    y = sp.y_pixel.reshape(-1).detach().cpu().numpy()
    z = sp.z_pixel.reshape(-1).detach().cpu().numpy()
    f = sp.frame_nr.reshape(-1).detach().cpu().numpy()
    ok = sp.valid.reshape(-1).detach().cpu().numpy() > 0.5
    ok &= (y > 5) & (y < g.n_pix_y-5) & (z > 5) & (z < g.n_pix_z-5)
    ok &= (f >= 0) & (f < g.n_frames)
    return z[ok], y[ok], f[ok]


def test_index_from_cloud_predicts_from_the_converged_cell():
    from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
    import torch

    A_TRUE, C_TRUE = 3.6116, 19.2516
    A_NOM,  C_NOM = A_TRUE*1.010, C_TRUE*1.010      # nominal 1.0 % off
    g = _geom()
    U = np.array([[0.936, -0.152, 0.317],
                  [0.239, 0.941, -0.238],
                  [-0.258, 0.302, 0.918]])
    U, _ = np.linalg.qr(U)                           # make it a clean rotation
    if np.linalg.det(U) < 0:
        U[:, 0] *= -1

    row, col, frame = _observations(_crystal(A_TRUE, C_TRUE), g, U, d_min=1.2)
    if len(row) < 25:
        pytest.skip(f"synthetic scene produced only {len(row)} reflections")

    om = g.omega_first_deg + g.omega_step_deg*frame
    qlab = pixel_to_qlab(row, col, g, device="cpu")
    qs = qlab_to_qsample(qlab, torch.deg2rad(torch.as_tensor(om, dtype=qlab.dtype)))
    q = qs.detach().cpu().numpy().astype(float)
    inten = np.full(len(row), 1000.0)
    mask = np.zeros((g.n_pix_z, g.n_pix_y), bool)

    res = index_from_cloud(q, inten, row, col, frame,
                           _crystal(A_NOM, C_NOM), g, mask,     # NOMINAL cell in
                           d_min=1.2, max_two_theta_rad=np.radians(0.5),
                           max_eta_rad=np.radians(2.0), max_omega_rad=np.radians(2.0),
                           n_bright=30, device="cpu")

    assert res.cell_converged is not None, "the cell never converged; nothing was refined"
    a_fit, b_fit, c_fit = res.cell_converged
    err_true = abs(c_fit - C_TRUE)/C_TRUE
    err_nom = abs(c_fit - C_NOM)/C_NOM
    assert err_true < err_nom, (
        f"converged c={c_fit:.4f} is closer to the NOMINAL {C_NOM:.4f} than to the "
        f"TRUE {C_TRUE:.4f} -- the refined cell is not reaching the prediction")
    assert err_true < 2e-3, f"converged c={c_fit:.4f} vs true {C_TRUE:.4f}"
    assert abs(a_fit - A_TRUE)/A_TRUE < 5e-3, f"converged a={a_fit:.4f} vs true {A_TRUE:.4f}"
    assert res.n_cell_reflections >= 10


@pytest.mark.parametrize("offset,should_converge", [(0.002, True), (0.005, True),
                                                    (0.010, True), (0.020, False)])
def test_cell_convergence_capture_range(offset, should_converge):
    """The converged cell is SEED-INDEPENDENT inside ~1 %, and fails by 2 %.

    Measured (66 synthetic reflections, inverse crime, c_true = 19.2516):

        nominal offset   seed res.c err   converged err
            0.2 %            0.204 %          0.179 %
            0.5 %            0.504 %          0.179 %
            1.0 %            1.005 %          0.179 %
            2.0 %            2.007 %          FAILS, claims 0

    The seed's own `refine_lattice=True` barely leaves the nominal cell -- its
    error simply equals the offset -- which is why discarding the convergence
    left the prediction on the nominal cell. The converged value is identical at
    every offset inside the capture range, which is what makes it a convergence
    and not a nudge.

    CONSEQUENCE: a nominal cell more than ~1 % from the truth will not be
    recovered; it will claim nothing. Get the starting cell right to 1 %.
    """
    from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
    from midas_defect.seed_index import find_seed_orientation
    from midas_defect.rows import refine_to_convergence
    import torch

    A_TRUE, C_TRUE = 3.6116, 19.2516
    g = _geom()
    U = np.array([[0.936, -0.152, 0.317], [0.239, 0.941, -0.238],
                  [-0.258, 0.302, 0.918]])
    U, _ = np.linalg.qr(U)
    if np.linalg.det(U) < 0:
        U[:, 0] *= -1
    row, col, frame = _observations(_crystal(A_TRUE, C_TRUE), g, U, d_min=1.2)
    if len(row) < 25:
        pytest.skip("scene too small")
    om = g.omega_first_deg + g.omega_step_deg*frame
    ql = pixel_to_qlab(row, col, g, device="cpu")
    q = qlab_to_qsample(ql, torch.deg2rad(torch.as_tensor(om, dtype=ql.dtype)))
    q = q.detach().cpu().numpy().astype(float)
    I = np.full(len(row), 1000.0)

    cry = _crystal(A_TRUE*(1+offset), C_TRUE*(1+offset))
    res = find_seed_orientation(q[:, 0], q[:, 1], q[:, 2], I, crystal=cry,
                                n_bright=30, tol_q_rel=0.02, tol_angle_deg=3.0,
                                refine_lattice=True)
    lat0 = cry.lattice
    conv = refine_to_convergence(q, np.asarray(res.U), a0=lat0.a, b0=lat0.b,
                                 c0=lat0.c, alpha0=lat0.alpha, beta0=lat0.beta,
                                 gamma0=lat0.gamma, min_reflections=8)
    if not should_converge:
        assert conv is None, ("convergence unexpectedly survived a 2 % offset; the "
                              "documented capture range has changed")
        return
    assert conv is not None
    seed_err = abs(res.c - C_TRUE)/C_TRUE
    conv_err = abs(conv.lat.c - C_TRUE)/C_TRUE
    assert conv_err < seed_err, (
        f"convergence ({conv_err:.4%}) did not beat the seed ({seed_err:.4%}) "
        f"at a {offset:.1%} offset -- the seed refinement barely moves off nominal, "
        f"so this must improve on it")
    assert conv_err < 0.003
