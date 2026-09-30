"""column_content: forward guard, exact fast NNLS, and a known-content column recovered end to end."""
import math

import numpy as np
import pytest
import torch

import midas_defect.column_content as cc
from midas_defect.column_content.fit import _nnls
from midas_defect.geometry import Geometry
from midas_defect.synthetic import _b_matrix, _hkl_candidates


def _geom(n_frames=14, tilts=False):
    return Geometry(lsd_um=349680.0, bcy_px=737.0, bcz_px=839.0, px_um=172.0, wavelength_A=0.42459,
                    n_pix_y=1475, n_pix_z=1679, omega_first_deg=-6.5, omega_step_deg=1.0, n_frames=n_frames,
                    ty_deg=0.15 if tilts else 0.0, tz_deg=0.37 if tilts else 0.0,
                    p_coeffs=((1e-5,) + (0.0,) * 14) if tilts else (0.0,) * 15)


def _rot(rng):
    M = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(M) < 0:
        M[:, 0] *= -1
    return M


B = _b_matrix(3.6008, 3.6008, 19.2522)
HKL = _hkl_candidates(8, 8, 30, 139)


def test_forward_torch_matches_numpy_with_tilt_and_distortion():
    rng = np.random.default_rng(0)
    g = _geom(tilts=True)
    for _ in range(3):
        assert cc.guard(_rot(rng), B, HKL, g) < 1e-4


def test_small_offset_moves_spots_consistently():
    """A 0.05 deg rotation predicted by the torch path equals re-predicting the rotated U with the numpy path."""
    rng = np.random.default_rng(1); g = _geom()
    U = _rot(rng); w = np.array([0.3, -0.2, 0.9]); w = w / np.linalg.norm(w) * math.radians(0.05)
    obs = cc.observable_reflections(U, B, HKL, g)
    f, r, c = cc.predict_torch(U, B, obs, torch.as_tensor(w[None]), g)
    U2 = cc.rotvec_to_matrix(torch.as_tensor(w)).numpy() @ U
    obs2 = cc.observable_reflections(U2, B, HKL, g)
    pairs = {tuple(h): (fr, rr, cl) for h, fr, rr, cl in zip(map(tuple, obs2.hkl), obs2.frame, obs2.row, obs2.col)}
    err = [max(abs(f[0, i].item() - pairs[tuple(h)][0]), abs(r[0, i].item() - pairs[tuple(h)][1]), abs(c[0, i].item() - pairs[tuple(h)][2]))
           for i, h in enumerate(obs.hkl) if tuple(h) in pairs]
    assert len(err) > 10 and max(err) < 1e-3


def test_fast_nnls_is_exact():
    from scipy.optimize import nnls
    rng = np.random.default_rng(2)
    D = np.abs(rng.normal(size=(30, 4000))); D[1] = D[0]
    y = D.T @ np.abs(rng.normal(size=30)) + rng.normal(0, 0.1, 4000)
    x_ref, _ = nnls(D.T, y)
    x = _nnls(torch.as_tensor(D), torch.as_tensor(y)).numpy()
    assert abs(np.linalg.norm(D.T @ x - y) - np.linalg.norm(D.T @ x_ref - y)) < 1e-9 * np.linalg.norm(y)


@pytest.mark.slow
def test_two_domain_column_shares_recovered():
    rng = np.random.default_rng(1); g = _geom()
    kern = cc.GaussKernel3D(0.8, 1.6, 1.2, g.bcz_px, g.bcy_px)
    doms = [cc.DomainSpec(_rot(rng), 1.0), cc.DomainSpec(_rot(rng), 0.4)]
    frames, truth = cc.synthetic_column(doms, B=B, hkl_all=HKL, geom=g, kernel=kern, seed=3)
    ing = cc.ingest_column(frames, g)
    fit = cc.ColumnFit(ing.sub, ing.mask, ing.labels, truth.U, B, HKL, g, kern, K=8)
    fit.fit(n_iter=60, inits=(0.05,))
    rep = fit.report()
    shares = np.array([o["share"] for o in rep["orientations"]])
    assert np.all(np.abs(shares - truth.share) / truth.share < 0.25)
    assert rep["unexplained_flux_frac"] < 0.05


@pytest.mark.slow
def test_kernel_estimate_calibration_removes_threshold_bias():
    """Raw thresholded-blob moments read ~0.8-0.9x the true widths; the calibrated estimate is within 6%."""
    rng = np.random.default_rng(11); g = _geom()
    true = np.array([0.8, 1.6, 1.2])
    kt = cc.GaussKernel3D(*true, g.bcz_px, g.bcy_px)
    doms = [cc.DomainSpec(_rot(rng), 1.0), cc.DomainSpec(_rot(rng), 0.6), cc.DomainSpec(_rot(rng), 0.4)]
    frames, truth = cc.synthetic_column(doms, B=B, hkl_all=HKL, geom=g, kernel=kt, seed=11, peak_counts=20000)
    ing = cc.ingest_column(frames, g)
    pr = []
    for U in truth.U:
        ob = cc.observable_reflections(U, B, HKL, g); pr += list(zip(ob.frame, ob.row, ob.col))
    k0, _ = cc.estimate_kernel(ing.sub, ing.labels, np.asarray(pr), g.bcz_px, g.bcy_px, calibrate=False)
    k1, _ = cc.estimate_kernel(ing.sub, ing.labels, np.asarray(pr), g.bcz_px, g.bcy_px, calibrate=True)
    raw = np.array([k0.sig_frame, k0.sig_rad, k0.sig_tan]) / true
    cal = np.array([k1.sig_frame, k1.sig_rad, k1.sig_tan]) / true
    assert np.all(raw < 0.95)
    assert np.all(np.abs(cal - 1) < 0.06)


def test_validation_columns_are_deterministic():
    spec = cc.ValidationSpec(geometries=[_geom()], a=3.6008, c=19.2522, sg=139, kernel_sigmas=(0.8, 1.6, 1.2),
                             n_columns=8, n_domains=(1, 2), foreign_frac=0.0)
    f1, T1, *_ = cc.build_column(spec, 5)
    f2, T2, *_ = cc.build_column(spec, 5)
    assert np.array_equal(f1, f2) and T1["share"] == T2["share"] and T1["N"] == 2


def test_run_column_uses_injected_search():
    """A custom search_fn replaces find_domains in round 0 (here: the truth orientations, so the fit must find them)."""
    rng = np.random.default_rng(4); g = _geom()
    kern = cc.GaussKernel3D(0.8, 1.6, 1.2, g.bcz_px, g.bcy_px)
    U = _rot(rng)
    frames, truth = cc.synthetic_column([cc.DomainSpec(U, 1.0)], B=B, hkl_all=HKL, geom=g, kernel=kern, seed=4)
    calls = []

    def srch(df, geom, q, I):
        calls.append(len(q)); return [(U, 99)] if len(calls) == 1 else []
    r = cc.run_column(frames, g, a=3.6008, c=19.2522, sg=139, B=B, hkl_all=HKL, kernel=kern, g_disc=9, K=4,
                      fit_kw=dict(n_iter=20, inits=(0.05,)), search_fn=srch, max_rounds=1)
    assert len(calls) == 2 and len(r.U) == 1 and r.report["orientations"][0]["share"] > 0.8


def test_evaluate_refuses_to_validate_an_unfinished_run(tmp_path):
    """A column whose result file is missing (a killed or still-running worker) must make the read INCOMPLETE, not be
    silently dropped: the gates would otherwise be scored on whichever columns happened to finish."""
    import json
    from midas_defect.column_content.validate import evaluate
    spec = cc.ValidationSpec(geometries=[_geom()], a=3.6008, c=19.2522, sg=139, kernel_sigmas=(0.8, 1.6, 1.2),
                             n_columns=3)
    U = np.eye(3).tolist()
    truth = dict(U=[U], share=[0.9], spread=["point"], param=[0.0], rms_deg=[0.0], N=1, geom=0, foreign=False,
                 kind=["random"], partner=[-1], delta=[0.0])
    rep = dict(orientations=[dict(U=U, share=0.9, spread_rms_deg=0.01)], unexplained_flux_frac=0.05, init_won=0)
    for i in (0, 2):
        json.dump(dict(i=i, truth=truth, report=rep), open(tmp_path / f"C_{i:03d}.json", "w"))
    E = evaluate(spec, str(tmp_path))
    assert E["missing"] == [1]
    assert E["read"].startswith("INCOMPLETE")
    json.dump(dict(i=1, truth=truth, report=rep), open(tmp_path / "C_001.json", "w"))
    E = evaluate(spec, str(tmp_path))
    assert E["missing"] == [] and not E["read"].startswith("INCOMPLETE")


def test_structured_normal_equations_match_the_dense_design():
    """The fast linear step assembles G = D D^T and b = D y from the window structure instead of forming the dense
    (m x NP) design. Same numbers as the dense product, on a column where two orientations share voxels (a 0.4 deg
    partner), so the cross-orientation blocks and duplicated-window merging are exercised."""
    from midas_defect.column_content.fit import _nnls_gram
    from midas_defect.column_content.forward import rotvec_to_matrix
    rng = np.random.default_rng(5); g = _geom()
    kern = cc.GaussKernel3D(0.8, 1.6, 1.2, g.bcz_px, g.bcy_px)
    U1, U2 = _rot(rng), _rot(rng)
    U1b = rotvec_to_matrix(torch.tensor([0.004, -0.003, 0.002], dtype=torch.float64)).numpy() @ U1     # ~0.3 deg from U1
    frames, truth = cc.synthetic_column([cc.DomainSpec(U1, 1.0), cc.DomainSpec(U2, 0.6, spread="cloud", param_deg=0.2)],
                                        B=B, hkl_all=HKL, geom=g, kernel=kern, seed=4)
    ing = cc.ingest_column(frames, g)
    Us = [truth.U[0], truth.U[1], U1b]
    fit = cc.ColumnFit(ing.sub, ing.mask, ing.labels, Us, B, HKL, g, kern, K=6)
    fit._prepare_gram()
    assert len(fit._gram["inter"]) > 0, "test column must have orientations sharing voxels"
    om = [(torch.randn(6, 3, generator=torch.Generator().manual_seed(j), dtype=torch.float64) * 0.002) for j in range(3)]
    with torch.no_grad():
        Bk = fit.blocks(om)
        a = [torch.rand(n, dtype=torch.float64) + 0.5 for n in fit.nref]; w = [torch.rand(6, dtype=torch.float64) + 0.1 for _ in Us]
        yc = fit.y - 0.3
        act = [j for j in range(3) if Bk[j] is not None]
        Dw = torch.cat([fit._scatter_rows(j, Bk[j] * a[j][fit.pref[j]][None]) for j in act], 0)
        Gw, bw = fit._gram_w(Bk, a, yc)
        assert np.allclose(Gw, (Dw @ Dw.T).numpy(), rtol=1e-10, atol=1e-10 * np.abs(Gw).max())
        assert np.allclose(bw, (Dw @ yc).numpy(), rtol=1e-10, atol=1e-10 * np.abs(bw).max())
        Da = torch.cat([fit._per_refl(j, (w[j][:, None] * Bk[j]).sum(0)) for j in act], 0)
        Ga, ba = fit._gram_a(Bk, w, yc)
        assert np.allclose(Ga, (Da @ Da.T).numpy(), rtol=1e-10, atol=1e-10 * np.abs(Ga).max())
        assert np.allclose(ba, (Da @ yc).numpy(), rtol=1e-10, atol=1e-10 * np.abs(ba).max())
        # and the solved model agrees with the dense reference path (the minimiser, not the degenerate-direction weights)
        a1, w1, c1 = fit.linear(Bk, [x.clone() for x in a], [x.clone() for x in w])
        m1 = fit.model(Bk, a1, w1, c1)
        fit.fast_linear = False
        a2, w2, c2 = fit.linear(Bk, [x.clone() for x in a], [x.clone() for x in w])
        m2 = fit.model(Bk, a2, w2, c2)
    assert float(torch.linalg.norm(m1 - m2) / torch.linalg.norm(m2)) < 1e-6
