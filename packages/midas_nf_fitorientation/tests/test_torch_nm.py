"""Tests for the vectorised PyTorch Nelder-Mead.

We check three things:

1. **Toy convergence parity vs scipy NM.** On a smooth quadratic, both
   should land at the same minimum within a tight tolerance.
2. **Bounded behaviour.** When the unconstrained minimum is outside
   the box, the simplex must respect the bounds.
3. **Independence.** Running ``B`` problems together produces the
   same per-problem result as running each one alone — vectorisation
   must not couple them.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from midas_nf_fitorientation.torch_nm import batched_nelder_mead


def _quadratic(centers: torch.Tensor):
    """Return ``f(x, idx) = sum((x - centers[idx])^2, dim=-1)`` for
    a batch of points indexed into a fixed set of centres."""
    def fn(x, idx):
        return ((x - centers[idx]) ** 2).sum(dim=-1)
    return fn


def test_batched_nm_converges_to_known_minimum():
    """One simplex per centre; must land within xatol of each centre."""
    centers = torch.tensor([
        [0.5, 0.0, -0.3],
        [-1.2, 1.5, 0.2],
        [0.0, 0.0, 0.0],
    ], dtype=torch.float64)
    x0 = torch.zeros_like(centers)
    fn = _quadratic(centers)
    res = batched_nelder_mead(
        fn, x0, bounds=None, max_iter=400, xatol=1e-8, fatol=1e-8,
        init_step=0.5,
    )
    assert torch.allclose(res.x, centers, atol=1e-3), (
        f"got {res.x}, expected {centers}"
    )
    assert torch.all(res.fun < 1e-5)


def test_batched_nm_matches_scipy_nm_on_quadratic():
    """For one problem at a time, the batched optimiser and scipy NM
    should converge to the same point within numerical tolerance."""
    rng = np.random.default_rng(42)
    for _ in range(5):
        center = rng.standard_normal(3)
        x0 = rng.standard_normal(3) * 0.5
        # scipy NM
        sp = minimize(
            lambda x: float(((x - center) ** 2).sum()),
            x0, method="Nelder-Mead",
            options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 1000},
        )
        # batched NM with B=1
        x0_t = torch.tensor(x0, dtype=torch.float64).unsqueeze(0)
        center_t = torch.tensor(center, dtype=torch.float64).unsqueeze(0)
        fn = _quadratic(center_t)
        res = batched_nelder_mead(
            fn, x0_t, bounds=None, max_iter=1000,
            xatol=1e-9, fatol=1e-9,
        )
        assert np.allclose(
            res.x.numpy().flatten(), sp.x, atol=1e-3,
        ), f"batched: {res.x.numpy().flatten()}, scipy: {sp.x}"


def test_batched_nm_respects_bounds():
    """When the optimum is outside the box the optimiser must stay
    on the box boundary, not drift outside."""
    centers = torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float64)
    bounds = torch.tensor(
        [[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]], dtype=torch.float64,
    )
    x0 = torch.zeros(1, 3, dtype=torch.float64)
    res = batched_nelder_mead(
        _quadratic(centers), x0, bounds=bounds,
        max_iter=300, xatol=1e-8, fatol=1e-8, init_step=0.3,
    )
    assert torch.allclose(
        res.x, torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64),
        atol=1e-2,
    ), f"got {res.x}, expected [[1, 1, 1]]"


def test_batched_nm_problems_are_independent():
    """Running 4 problems together gives the same result as running
    each alone — batching must not couple them."""
    rng = np.random.default_rng(0)
    centers_np = rng.standard_normal((4, 3))
    centers = torch.tensor(centers_np, dtype=torch.float64)
    x0 = torch.zeros_like(centers)

    # Batched run
    res_batched = batched_nelder_mead(
        _quadratic(centers), x0, bounds=None,
        max_iter=300, xatol=1e-8, fatol=1e-8, init_step=0.5,
    )

    # Per-problem runs
    per = []
    for i in range(4):
        c = centers[i:i + 1]
        x0_i = x0[i:i + 1]
        r = batched_nelder_mead(
            _quadratic(c), x0_i, bounds=None,
            max_iter=300, xatol=1e-8, fatol=1e-8, init_step=0.5,
        )
        per.append(r.x[0])
    per_t = torch.stack(per, dim=0)

    # Same answer (mod NM convergence noise).
    assert torch.allclose(res_batched.x, per_t, atol=1e-3)


def test_batched_nm_trimming_preserves_per_problem_results():
    """Mixing easy + hard problems in one batch: easy ones should
    converge fast, hard ones should still converge to the right
    answer. The trimming optimisation must not corrupt either."""
    # Three problems with very different curvature → wildly different
    # convergence rates. Centres are all far apart.
    centers = torch.tensor([
        [0.1, -0.1, 0.0],     # close to x0 — converges fast
        [3.5, -2.0, 1.7],     # far from x0 — slower
        [0.0, 0.0, 0.0],      # x0 itself
    ], dtype=torch.float64)
    x0 = torch.zeros_like(centers)
    res = batched_nelder_mead(
        _quadratic(centers), x0, bounds=None,
        max_iter=500, xatol=1e-9, fatol=1e-9, init_step=0.5,
    )
    assert torch.allclose(res.x, centers, atol=1e-3)


def test_batched_nm_fn_sees_active_idx_in_order():
    """After trimming, the ``idx`` tensor passed to ``fn`` must
    correctly select the still-active subset of the caller's per-
    problem aux data."""
    # 4 problems; each problem has its own centre stored in a
    # tensor that ``fn`` indexes via ``idx``.
    centers = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float64)
    seen_idx = []

    def fn(x, idx):
        seen_idx.append(idx.detach().clone())
        return ((x - centers[idx]) ** 2).sum(dim=-1)

    x0 = torch.zeros_like(centers)
    res = batched_nelder_mead(
        fn, x0, bounds=None,
        max_iter=500, xatol=1e-9, fatol=1e-9, init_step=0.5,
    )
    # The problem with centre [0, 0, 0] is at x0 already → converges
    # immediately. Every subsequent fn call should be passed an idx
    # that no longer contains 0.
    # Just verify that converged problems get pulled out at some
    # point, i.e. some fn call sees a strictly smaller idx than
    # ``arange(4)``.
    assert any(t.numel() < 4 for t in seen_idx[1:])
    # And the final answers are still right.
    assert torch.allclose(res.x, centers, atol=1e-3)


def test_fixed_B_mode_matches_trimming_mode():
    """``fixed_B=True`` must produce the same per-problem optima as
    the trimming default — it just uses a different convergence
    bookkeeping path. Empirically the answers should be byte-identical
    on smooth objectives because both paths hit the convergence test
    on the same iteration."""
    rng = np.random.default_rng(11)
    centers = torch.tensor(rng.standard_normal((6, 3)), dtype=torch.float64)
    x0 = torch.zeros_like(centers)
    res_trim = batched_nelder_mead(
        _quadratic(centers), x0, bounds=None,
        max_iter=400, xatol=1e-9, fatol=1e-9, init_step=0.5,
        fixed_B=False,
    )
    res_fixed = batched_nelder_mead(
        _quadratic(centers), x0, bounds=None,
        max_iter=400, xatol=1e-9, fatol=1e-9, init_step=0.5,
        fixed_B=True,
    )
    # Both should converge to the centres within ~xatol.
    assert torch.allclose(res_trim.x, centers, atol=1e-3)
    assert torch.allclose(res_fixed.x, centers, atol=1e-3)
    # And to each other within numerical noise (their NM sequences
    # diverge once frozen simplices stay in the batch, so we're
    # generous on the per-element atol).
    assert torch.allclose(res_trim.x, res_fixed.x, atol=1e-3)


def test_batched_nm_handles_shared_bounds():
    """A single ``(n_dim, 2)`` bounds tensor should be broadcast to
    every problem in the batch."""
    centers = torch.tensor([
        [10.0, 10.0, 10.0],
        [-10.0, -10.0, -10.0],
    ], dtype=torch.float64)
    shared_bounds = torch.tensor(
        [[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]], dtype=torch.float64,
    )
    x0 = torch.zeros_like(centers)
    res = batched_nelder_mead(
        _quadratic(centers), x0, bounds=shared_bounds,
        max_iter=300, xatol=1e-8, fatol=1e-8, init_step=0.3,
    )
    # Each simplex pinned at its respective box corner.
    assert torch.allclose(
        res.x[0], torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64), atol=1e-2,
    )
    assert torch.allclose(
        res.x[1], torch.tensor([-2.0, -2.0, -2.0], dtype=torch.float64), atol=1e-2,
    )


def _rosenbrock(x, idx):
    return (100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2
            + (1.0 - x[:, :-1]) ** 2).sum(dim=-1)


def test_default_cap_lets_a_long_problem_converge():
    """Regression: the default cap used to be 200 iterations, which
    silently truncated anything that needed more (the C uses 5000).

    5-D Rosenbrock from the textbook start needs ~440 iterations here. With
    the old cap it stops far from the minimum and reports not-converged;
    with the default it must converge on its own, well BEFORE the cap, so
    it is the tolerance test and not the cap that ends the loop.
    """
    from midas_nf_fitorientation.torch_nm import NM_MAX_ITER_DEFAULT

    assert NM_MAX_ITER_DEFAULT == 5000
    x0 = torch.full((2, 5), -1.2, dtype=torch.float64)
    x0[1] = 0.0

    old = batched_nelder_mead(_rosenbrock, x0.clone(), None,
                              max_iter=200, init_step=0.5)
    assert not bool(old.converged.any()), "toy problem no longer needs >200"

    res = batched_nelder_mead(_rosenbrock, x0.clone(), None, init_step=0.5)
    assert bool(res.converged.all())
    assert 200 < res.n_iter < NM_MAX_ITER_DEFAULT
    assert torch.allclose(res.x, torch.ones_like(res.x), atol=1e-4)


def test_orientation_cap_defaults_from_paramfile(tmp_path):
    """``NMMaxIter`` is a paramfile key; absent, it is the C's 5000."""
    from midas_nf_fitorientation.params import parse_paramfile

    base = "nDistances 1\nLsd 1000\nBC 1 1\n"
    f = tmp_path / "p.txt"
    f.write_text(base)
    assert parse_paramfile(f).nm_max_iter == 5000
    f.write_text(base + "NMMaxIter 750\n")
    assert parse_paramfile(f).nm_max_iter == 750
