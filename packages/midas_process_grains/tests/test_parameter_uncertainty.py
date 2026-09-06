"""``compute.position_uncertainty`` — per-parameter, per-grain sigma.

The 12x12 Hessian over ``[euler(3), latc(6), pos(3)]`` was always inverted in
full and nine of the twelve marginals thrown away. These tests cover the
plumbing that now keeps them, with ``per_grain_hessian_blocks`` stubbed to a
known matrix so the assertions are about THIS code and not about the forward
model (which midas_propagate's own suite covers).

The load-bearing one is
:func:`test_position_wrapper_is_a_slice_not_a_second_implementation` — the
v4 pipeline consumes the position-only view and its numbers must not move.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from midas_process_grains.compute.position_uncertainty import (
    PerGrainParameterSigmaResult,
    compute_per_grain_parameter_sigma,
    compute_per_grain_position_sigma,
)

N_GRAINS = 3
PK_STRIDE = 5000


@pytest.fixture
def fake_inputs(tmp_path):
    """Minimal stand-ins for the pipeline artifacts the function reads."""
    import pandas as pd

    # ProcessKey.bin: one row per grain, each claiming 8 spots
    pk = np.zeros((N_GRAINS, PK_STRIDE), dtype=np.int32)
    for g in range(N_GRAINS):
        pk[g, :8] = np.arange(1, 9) + 100 * g
    pk_path = tmp_path / "ProcessKey.bin"
    pk.tofile(pk_path)

    sids = np.concatenate([np.arange(1, 9) + 100 * g for g in range(N_GRAINS)])
    rng = np.random.default_rng(0)
    inputall = pd.DataFrame(
        {"YLab": rng.uniform(-40000, 40000, sids.size),
         "ZLab": rng.uniform(-40000, 40000, sids.size),
         "Omega": rng.uniform(-180, 180, sids.size)},
        index=pd.Index(sids, name="SpotID"))

    hkls = types.SimpleNamespace(
        g_crystal=np.eye(3)[None].repeat(4, 0).reshape(12, 3).astype(float),
        theta_deg=np.linspace(2.0, 8.0, 12),
        h=np.ones(12, int), k=np.zeros(12, int), l=np.zeros(12, int))
    geometry = types.SimpleNamespace(
        y_BC=1022.0, z_BC=974.0, px=200.0, Lsd=767765.75, ty=0.0, tz=0.13)

    return dict(
        grain_OM=np.stack([np.eye(3)] * N_GRAINS),
        grain_pos_um=np.zeros((N_GRAINS, 3)),
        rep_cand_idx=np.arange(N_GRAINS),
        pk_path=pk_path, inputall_df=inputall, hkls=hkls, geometry=geometry,
    )


def _stub(H_gg):
    """Patch per_grain_hessian_blocks to return a fixed H_gg."""
    import torch
    import midas_propagate.joint_nll as jn

    def fake(grain_obs, **kw):
        fake.seen_latc.append(np.asarray(grain_obs.latc, dtype=float).copy())
        return types.SimpleNamespace(
            H_gg=torch.as_tensor(H_gg, dtype=torch.float64),
            H_gc=torch.zeros((12, 5), dtype=torch.float64),
            H_cc_data=torch.zeros((5, 5), dtype=torch.float64),
            n_spots_matched=8,
            residual_at_map=torch.zeros(24, dtype=torch.float64),
            sigma_r_used=1.0)
    fake.seen_latc = []
    return jn, fake


def test_returns_all_twelve_sigmas(fake_inputs, monkeypatch):
    H = np.diag(np.arange(1, 13, dtype=float))     # sigma_k = 1/sqrt(k)
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)

    res = compute_per_grain_parameter_sigma(
        latc=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]), **fake_inputs)

    assert res.ok.all(), "every stubbed grain should succeed"
    assert res.sigma_euler_rad.shape == (N_GRAINS, 3)
    assert res.sigma_latc.shape == (N_GRAINS, 6)
    assert res.sigma_pos_um.shape == (N_GRAINS, 3)
    expect = 1.0 / np.sqrt(np.arange(1, 13, dtype=float))
    got = np.concatenate([res.sigma_euler_rad[0], res.sigma_latc[0],
                          res.sigma_pos_um[0]])
    # rtol is looser than the 1e-9 ridge the implementation adds to
    # H_gg before inverting - that regularisation is deliberate.
    np.testing.assert_allclose(got, expect, rtol=1e-6)


def test_position_wrapper_is_a_slice_not_a_second_implementation(
        fake_inputs, monkeypatch):
    """v4_pipeline reads the position-only view; it must equal the full one."""
    H = np.diag(np.arange(1, 13, dtype=float))
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)
    latc = np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0])

    full = compute_per_grain_parameter_sigma(latc=latc, **fake_inputs)
    pos = compute_per_grain_position_sigma(latc=latc, **fake_inputs)

    np.testing.assert_allclose(pos.sigma_X_um, full.sigma_pos_um[:, 0])
    np.testing.assert_allclose(pos.sigma_Y_um, full.sigma_pos_um[:, 1])
    np.testing.assert_allclose(pos.sigma_Z_um, full.sigma_pos_um[:, 2])
    np.testing.assert_array_equal(pos.n_spots_matched, full.n_spots_matched)
    np.testing.assert_array_equal(pos.ok, full.ok)


def test_hydrostatic_strain_uses_the_full_abc_covariance(
        fake_inputs, monkeypatch):
    """a, b, c are strongly correlated through the common radial scale.

    Treating them as independent gets eps_hydro wrong in whichever direction
    the correlation points, so the propagation must use the off-diagonals.
    Here the (a,b,c) block is built with correlation +0.9, which INFLATES
    sigma(eps_hydro) well above the independent answer.
    """
    import numpy as _np
    S_abc = _np.array([[1.0, 0.9, 0.9],
                       [0.9, 1.0, 0.9],
                       [0.9, 0.9, 1.0]]) * (1e-3 ** 2)
    Sigma = _np.eye(12) * 1e-12
    Sigma[3:6, 3:6] = S_abc
    H = _np.linalg.inv(Sigma)
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)

    a = 3.6
    res = compute_per_grain_parameter_sigma(
        latc=_np.array([a, a, a, 90.0, 90.0, 90.0]), **fake_inputs)

    J = _np.full(3, 1.0 / (3.0 * a))
    expect = float(_np.sqrt(J @ S_abc @ J))
    independent = float(_np.sqrt(((J ** 2) * _np.diag(S_abc)).sum()))
    assert res.sigma_hydrostatic_strain[0] == pytest.approx(expect, rel=1e-8)
    assert expect > independent * 1.3, (
        "fixture no longer exercises the correlation; strengthen it")
    assert res.sigma_hydrostatic_strain[0] != pytest.approx(independent, rel=1e-3)


def test_per_grain_lattice_reaches_the_hessian(fake_inputs, monkeypatch):
    """An (N,6) latc must give each grain ITS OWN cell, not a shared one.

    device and MIDAS_PG_SIGMA_JOBS are pinned because this assertion is about
    what the PARENT process observed. Left to auto-detect, the device is
    whatever the machine has: on a Mac that is "mps", which forces _njobs=1 and
    the serial path, so the spy sees every call. On a plain CPU box -- CI, and
    every Linux workstation -- it resolves to "cpu", _njobs becomes
    min(cpu_count, 16) and the work forks. The children inherit the patched
    function and append to THEIR OWN copy of fake.seen_latc, which dies with
    them, so the parent's list is empty and the test failed with
    shape (0,) against (3,). It was passing here only because this laptop has
    a GPU backend.
    """
    monkeypatch.setenv("MIDAS_PG_SIGMA_JOBS", "1")
    H = np.eye(12)
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)

    latc = np.array([[3.60 + 0.01 * g, 3.60 + 0.01 * g, 3.60 + 0.01 * g,
                      90.0, 90.0, 90.0] for g in range(N_GRAINS)])
    compute_per_grain_parameter_sigma(latc=latc, device="cpu", **fake_inputs)

    seen = np.array(sorted(x[0] for x in fake.seen_latc))
    np.testing.assert_allclose(seen, sorted(latc[:, 0]), rtol=1e-12)


def test_scalar_latc_is_broadcast(fake_inputs, monkeypatch):
    """device/jobs pinned for the reason given in
    test_per_grain_lattice_reaches_the_hessian: an in-process spy cannot see
    what a forked child did."""
    monkeypatch.setenv("MIDAS_PG_SIGMA_JOBS", "1")
    H = np.eye(12)
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)
    compute_per_grain_parameter_sigma(
        latc=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]),
        device="cpu", **fake_inputs)
    seen = np.array(fake.seen_latc)
    assert seen.shape == (N_GRAINS, 6)
    assert np.allclose(seen[:, 0], 3.6)


def test_wrong_latc_shape_raises(fake_inputs, monkeypatch):
    H = np.eye(12)
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)
    with pytest.raises(ValueError, match=r"latc must be"):
        compute_per_grain_parameter_sigma(
            latc=np.zeros((N_GRAINS + 5, 6)), **fake_inputs)


def test_return_cov_gives_the_full_matrix(fake_inputs, monkeypatch):
    H = np.diag(np.arange(1, 13, dtype=float))
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)
    res = compute_per_grain_parameter_sigma(
        latc=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]),
        return_cov=True, **fake_inputs)
    assert res.cov.shape == (N_GRAINS, 12, 12)
    np.testing.assert_allclose(np.diag(res.cov[0]),
                               1.0 / np.arange(1, 13, dtype=float), rtol=1e-6)
    # the reported sigma must be the sqrt of that same diagonal
    got = np.concatenate([res.sigma_euler_rad[0], res.sigma_latc[0],
                          res.sigma_pos_um[0]])
    np.testing.assert_allclose(got, np.sqrt(np.diag(res.cov[0])), rtol=1e-12)


def test_as_columns_labels_every_parameter(fake_inputs, monkeypatch):
    H = np.eye(12)
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)
    res = compute_per_grain_parameter_sigma(
        latc=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]), **fake_inputs)
    cols = res.as_columns()
    for nm in ("sigma_euler0_rad", "sigma_a", "sigma_gamma", "sigma_X_um",
               "sigma_Z_um", "sigma_eps_hydro", "n_spots_matched", "ok"):
        assert nm in cols, f"missing {nm}"
    assert all(len(v) == N_GRAINS for v in cols.values())


def test_grain_with_too_few_spots_is_marked_not_ok(fake_inputs, monkeypatch):
    """Fewer than 4 matched spots cannot support a 12-parameter fit."""
    pk = np.zeros((N_GRAINS, PK_STRIDE), dtype=np.int32)
    pk[0, :8] = np.arange(1, 9)
    pk[1, :2] = np.array([101, 102])          # only 2 spots
    pk[2, :8] = np.arange(1, 9) + 200
    pk.tofile(fake_inputs["pk_path"])
    H = np.eye(12)
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)
    res = compute_per_grain_parameter_sigma(
        latc=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]), **fake_inputs)
    assert not res.ok[1]
    assert np.isnan(res.sigma_pos_um[1]).all()
    assert res.ok[0] and res.ok[2]


def test_rank_deficient_hessian_is_reported_as_failed_not_as_a_huge_sigma(
        fake_inputs, monkeypatch):
    """A singular direction must fail, not return sqrt(1/ridge).

    Without the conditioning gate, inv(H + 1e-9 I) returns 1e9 in the null
    direction and sigma comes back as 3.162e4 um — which is a property of the
    regulariser, not of the data. Measured on 1-ID LSHR layer 6, that affected
    about half the grains, and a caller filtering on `sigma < threshold` would
    silently have kept them.
    """
    H = np.diag(np.arange(1, 13, dtype=float))
    H[9, 9] = 0.0                     # position X carries no information
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)
    res = compute_per_grain_parameter_sigma(
        latc=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]), **fake_inputs)
    assert not res.ok.any(), "a singular Hessian must be marked failed"
    assert np.isnan(res.sigma_pos_um).all()
    # and specifically NOT the ridge value
    assert not np.any(np.isclose(np.nan_to_num(res.sigma_pos_um), np.sqrt(1e9)))


def test_well_conditioned_hessian_still_passes(fake_inputs, monkeypatch):
    """The gate must not reject healthy grains."""
    H = np.diag(np.arange(1, 13, dtype=float))
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)
    res = compute_per_grain_parameter_sigma(
        latc=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]), **fake_inputs)
    assert res.ok.all()


def test_forked_pool_gives_the_same_answer_as_serial(fake_inputs, monkeypatch):
    """Parallelism must be by PROCESS, and must not change the numbers.

    Threads are unusable here: torch forward-mode AD keeps its dual-level stack
    in global interpreter state, so concurrent jacfwd calls corrupt each other
    (measured 22/120 grains surviving a 96-thread pool). Forked children each
    get their own interpreter. Measured speedup on 400 real grains: 25.0 s
    serial -> 8.1 s at 8 jobs.
    """
    import multiprocessing
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("no fork start method on this platform")

    H = np.diag(np.arange(1, 13, dtype=float))
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)
    latc = np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0])

    monkeypatch.delenv("MIDAS_PG_SIGMA_JOBS", raising=False)
    monkeypatch.setenv("MIDAS_PG_SIGMA_JOBS", "1")
    a = compute_per_grain_parameter_sigma(latc=latc, **fake_inputs)
    monkeypatch.setenv("MIDAS_PG_SIGMA_JOBS", "3")
    b = compute_per_grain_parameter_sigma(latc=latc, **fake_inputs)

    np.testing.assert_array_equal(a.ok, b.ok)
    np.testing.assert_allclose(a.sigma_pos_um, b.sigma_pos_um, rtol=1e-12)
    np.testing.assert_allclose(a.sigma_latc, b.sigma_latc, rtol=1e-12)


def test_cuda_device_is_not_forked(fake_inputs, monkeypatch):
    """CUDA does not survive fork; the serial path is already fast on GPU."""
    H = np.eye(12)
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)
    monkeypatch.setenv("MIDAS_PG_SIGMA_JOBS", "8")
    msgs = []
    compute_per_grain_parameter_sigma(
        latc=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]),
        device="cpu", log=lambda *m, **k: msgs.append(" ".join(map(str, m))),
        **fake_inputs)
    # on an explicit CPU device the fork pool is allowed
    assert any("forked processes" in m or "serial" in m for m in msgs)


def test_euclidean_position_sigma_is_the_quadrature_sum(fake_inputs, monkeypatch):
    """sqrt(sx^2+sy^2+sz^2) — exact, because E[|dr|^2] = trace(Sigma) and the
    trace ignores the X/Y/Z correlations."""
    H = np.diag(np.arange(1, 13, dtype=float))
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)
    res = compute_per_grain_parameter_sigma(
        latc=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]), **fake_inputs)
    s3 = res.sigma_pos_3d_um
    assert s3.shape == (N_GRAINS,)
    np.testing.assert_allclose(
        s3, np.sqrt((res.sigma_pos_um ** 2).sum(axis=1)), rtol=1e-12)
    # for this stub sigma_k = 1/sqrt(k) with k = 10, 11, 12
    expect = np.sqrt(1/10 + 1/11 + 1/12)
    np.testing.assert_allclose(s3[0], expect, rtol=1e-6)


def test_euclidean_sigma_unaffected_by_position_correlations(fake_inputs,
                                                             monkeypatch):
    """Two covariances with the same diagonal but different off-diagonals must
    give the SAME Euclidean sigma — that is the point of using the trace."""
    base = np.eye(12) * 1e-6
    out = []
    for rho in (0.0, 0.85):
        S = base.copy()
        d = np.array([4.0, 9.0, 16.0])            # variances for X, Y, Z
        C = np.outer(np.sqrt(d), np.sqrt(d)) * rho
        np.fill_diagonal(C, d)
        S[9:12, 9:12] = C
        jn, fake = _stub(np.linalg.inv(S))
        monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)
        r = compute_per_grain_parameter_sigma(
            latc=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]), **fake_inputs)
        out.append(r.sigma_pos_3d_um[0])
    assert out[0] == pytest.approx(out[1], rel=1e-6)
    assert out[0] == pytest.approx(np.sqrt(4.0 + 9.0 + 16.0), rel=1e-4)


def test_as_columns_includes_the_euclidean_sigma(fake_inputs, monkeypatch):
    H = np.eye(12)
    jn, fake = _stub(H)
    monkeypatch.setattr(jn, "per_grain_hessian_blocks", fake)
    res = compute_per_grain_parameter_sigma(
        latc=np.array([3.6, 3.6, 3.6, 90.0, 90.0, 90.0]), **fake_inputs)
    assert "sigma_pos_3d_um" in res.as_columns()
