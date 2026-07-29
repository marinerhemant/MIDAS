"""Tests for `midas_defect.seed_index`."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from midas_defect.lattice import (
    CUAL2_A_DEFAULT,
    CUAL2_C_DEFAULT,
    cual2_crystal,
    tetragonal_shells,
)
from midas_defect.seed_index import (
    SeedIndexResult,
    find_seed_orientation,
    predict_q_from_U,
    refine_U_from_centroids,
    refine_U_from_friedel_pairs,
    refine_U_lattice,
    _discrete_extract_bright_cores,
    _hkl_to_g_cry,
    _matrix_to_rotvec,
    _rotvec_to_matrix,
)


CPU = torch.device("cpu")


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix (Shoemake's method)."""
    u1, u2, u3 = rng.random(3)
    q = np.array([
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1)     * math.sin(2 * math.pi * u3),
        math.sqrt(u1)     * math.cos(2 * math.pi * u3),
    ])  # (x, y, z, w)
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _synthetic_single_crystal_cloud(
    U_true: np.ndarray, *, a: float = CUAL2_A_DEFAULT, c: float = CUAL2_C_DEFAULT,
    n_shells: int = 15, noise_pos: float = 0.005,
    rng: np.random.Generator = None,
) -> dict:
    """Produce a sparse voxel cloud for a perfect CuAl₂ crystal at orientation U.

    For each (hkl) family, pick the (h,k,l) representative and place ~30 noisy
    points around the predicted sample-frame q-vector. The "intensity" of each
    point falls off so the brightest cluster centres are exactly at the
    predicted q-positions.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    cr = cual2_crystal(a=a, c=c)
    shells = tetragonal_shells(cr, q_max_inv_A=8.0)[:n_shells]
    xs, ys, zs, vs = [], [], [], []
    for sh in shells:
        for hkl in sh.hkls[:1]:  # one representative per shell
            g_cry = _hkl_to_g_cry(hkl, a, c)
            q_pred = U_true @ g_cry
            pts = q_pred[None, :] + rng.normal(scale=noise_pos, size=(30, 3))
            xs.append(pts[:, 0])
            ys.append(pts[:, 1])
            zs.append(pts[:, 2])
            vs.append(rng.uniform(80.0, 300.0, size=30))
    return dict(
        qx=np.concatenate(xs), qy=np.concatenate(ys), qz=np.concatenate(zs),
        intensity=np.concatenate(vs),
        shells=shells, U_true=U_true, a=a, c=c,
    )


# ---------------------------------------------------------------------------
# 1. Synthetic correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_extract_bright_cores_picks_top_intensities():
    rng = np.random.default_rng(0)
    qx = rng.uniform(-1, 1, 1000)
    qy = rng.uniform(-1, 1, 1000)
    qz = rng.uniform(-1, 1, 1000)
    inten = rng.uniform(1, 10, 1000)
    # plant 3 known bright spots well separated
    for i, q in enumerate([(0.5, 0.5, 0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, -0.5)]):
        qx[i] = q[0]; qy[i] = q[1]; qz[i] = q[2]
        inten[i] = 1000.0 + i
    cents, ints = _discrete_extract_bright_cores(
        qx, qy, qz, inten, n_bright=3, min_separation=0.1
    )
    assert cents.shape == (3, 3)
    # brightest should be (0.5, 0.5, 0.5) (intensity 1002, since i=2 has 1002)
    assert tuple(cents[0]) == pytest.approx((0.5, -0.5, -0.5), abs=1e-8)


@pytest.mark.unit
def test_rotvec_matrix_roundtrip():
    """`_matrix_to_rotvec` and `_rotvec_to_matrix` are inverses (within SO(3))."""
    rng = np.random.default_rng(42)
    for _ in range(5):
        U_true = _random_rotation(rng)
        rotvec = _matrix_to_rotvec(U_true)
        rv_t = torch.as_tensor(rotvec, dtype=torch.float64)
        U_back = _rotvec_to_matrix(rv_t).numpy()
        # SO(3) round-trip — should be exact to machine precision
        assert np.allclose(U_true, U_back, atol=1e-10)


@pytest.mark.unit
def test_find_seed_orientation_recovers_known_U():
    """Plant a known orientation, run find_seed_orientation, check |misorientation| < 1°."""
    rng = np.random.default_rng(7)
    U_true = _random_rotation(rng)
    data = _synthetic_single_crystal_cloud(U_true, n_shells=12, noise_pos=0.002, rng=rng)
    result = find_seed_orientation(
        data["qx"], data["qy"], data["qz"], data["intensity"],
        n_bright=12, tol_q_rel=0.05, tol_angle_deg=8.0,
        n_refine_steps=400, refine_lr=2e-2,
        refine_lattice=False,
    )
    assert isinstance(result, SeedIndexResult)
    # measure rotation difference
    delta = result.U @ U_true.T
    cos_a = max(-1.0, min(1.0, (np.trace(delta) - 1.0) / 2.0))
    misorientation_deg = math.degrees(math.acos(cos_a))
    assert misorientation_deg < 1.0, (
        f"recovered U is {misorientation_deg:.3f}° off from ground truth"
    )
    assert result.score >= 8, f"only matched {result.score}/12 shells"


@pytest.mark.unit
def test_lattice_refinement_pulls_a_c_toward_truth():
    """Start with a 1.5% wrong (a, c); refinement should pull them toward the truth."""
    rng = np.random.default_rng(11)
    a_true, c_true = 6.066, 4.874
    U_true = _random_rotation(rng)
    data = _synthetic_single_crystal_cloud(
        U_true, a=a_true, c=c_true, n_shells=12, noise_pos=0.001, rng=rng
    )
    # seed with deliberately wrong a, c
    cr_wrong = cual2_crystal(a=a_true * 1.015, c=c_true * 0.985)
    result = find_seed_orientation(
        data["qx"], data["qy"], data["qz"], data["intensity"],
        crystal=cr_wrong,
        n_bright=12, tol_q_rel=0.05, tol_angle_deg=8.0,
        n_refine_steps=600, refine_lr=2e-2,
        refine_lattice=True,
    )
    assert abs(result.a - a_true) / a_true < 0.005, (
        f"a not pulled back: got {result.a:.4f}, truth {a_true:.4f}"
    )
    assert abs(result.c - c_true) / c_true < 0.005, (
        f"c not pulled back: got {result.c:.4f}, truth {c_true:.4f}"
    )


# ---------------------------------------------------------------------------
# 2. Autograd correctness
# ---------------------------------------------------------------------------

@pytest.mark.autograd
def test_predict_q_from_U_gradient_wrt_U():
    """Gradient of `(predict_q - q_obs)²` w.r.t. U entries matches FD."""
    hkls = torch.tensor([[1, 1, 0], [2, 0, 0], [0, 0, 2], [2, 1, 1]],
                        dtype=torch.float64)
    a = torch.tensor(6.066, dtype=torch.float64)
    c = torch.tensor(4.874, dtype=torch.float64)
    rng = np.random.default_rng(0)
    U_true = _random_rotation(rng)
    q_obs = predict_q_from_U(
        torch.as_tensor(U_true, dtype=torch.float64), hkls, a, c
    )
    # perturb rotvec slightly and check gradient
    rotvec0 = _matrix_to_rotvec(U_true) + np.array([0.02, -0.03, 0.01])
    rv = torch.as_tensor(rotvec0, dtype=torch.float64).clone().requires_grad_(True)

    def loss(rv_):
        U = _rotvec_to_matrix(rv_)
        q = predict_q_from_U(U, hkls, a, c)
        return ((q - q_obs) ** 2).sum()

    L = loss(rv)
    g_auto = torch.autograd.grad(L, rv)[0]
    eps = 1e-6
    g_fd = torch.zeros_like(rv)
    for i in range(3):
        plus = rv.detach().clone(); plus[i] += eps
        minus = rv.detach().clone(); minus[i] -= eps
        g_fd[i] = (loss(plus) - loss(minus)) / (2 * eps)
    assert torch.allclose(g_auto, g_fd, atol=1e-4)


@pytest.mark.autograd
def test_predict_q_from_U_gradient_wrt_a_c():
    """Gradient w.r.t. lattice constants is correct."""
    hkls = torch.tensor([[1, 1, 0], [2, 0, 0], [0, 0, 2], [2, 1, 1]],
                        dtype=torch.float64)
    rng = np.random.default_rng(0)
    U_t = torch.as_tensor(_random_rotation(rng), dtype=torch.float64)
    a = torch.tensor(6.066, dtype=torch.float64, requires_grad=True)
    c = torch.tensor(4.874, dtype=torch.float64, requires_grad=True)
    q_target = predict_q_from_U(U_t, hkls, a.detach() * 1.01, c.detach() * 0.99)

    def loss(a_, c_):
        q = predict_q_from_U(U_t, hkls, a_, c_)
        return ((q - q_target) ** 2).sum()

    L = loss(a, c)
    g_auto_a, g_auto_c = torch.autograd.grad(L, (a, c))
    eps = 1e-6
    g_fd_a = (loss(a + eps, c.detach()) - loss(a - eps, c.detach())) / (2 * eps)
    g_fd_c = (loss(a.detach(), c + eps) - loss(a.detach(), c - eps)) / (2 * eps)
    assert g_auto_a.item() == pytest.approx(g_fd_a.item(), rel=1e-3, abs=1e-6)
    assert g_auto_c.item() == pytest.approx(g_fd_c.item(), rel=1e-3, abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Device portability
# ---------------------------------------------------------------------------

@pytest.mark.device
def test_refinement_device_portable(_device_param):
    """Refinement reaches a similar (U, a, c) on CPU and MPS within tolerance."""
    if _device_param.type == "mps":
        # MPS lacks float64 — use float32 for both sides on this test
        dtype = torch.float32
        atol_misorient = 0.5     # looser for float32
    else:
        dtype = torch.float64
        atol_misorient = 0.01
    rng = np.random.default_rng(3)
    U_true = _random_rotation(rng)
    data = _synthetic_single_crystal_cloud(U_true, n_shells=12, noise_pos=0.001, rng=rng)
    # Build the matched pairs deterministically using the truth (skip discrete pair-voting
    # so we isolate the refinement from any device-dependent ordering in the search).
    a, c = CUAL2_A_DEFAULT, CUAL2_C_DEFAULT
    centroids = []
    matched = []
    for sh in data["shells"][:8]:
        hkl = sh.hkls[0]
        centroids.append(U_true @ _hkl_to_g_cry(hkl, a, c))
        matched.append(hkl)
    centroids = np.stack(centroids)
    ints = np.ones(len(centroids))
    # perturb the initial U by ~3°
    U_init = U_true @ _rotvec_to_matrix(
        torch.as_tensor([0.05, -0.04, 0.03], dtype=torch.float64)
    ).numpy()

    U_cpu, _, _, _ = refine_U_lattice(
        U_init, centroids, ints, matched,
        a_init=a, c_init=c, refine_lattice=False,
        n_steps=300, lr=2e-2, device=CPU, dtype=dtype,
    )
    U_dev, _, _, _ = refine_U_lattice(
        U_init, centroids, ints, matched,
        a_init=a, c_init=c, refine_lattice=False,
        n_steps=300, lr=2e-2, device=_device_param, dtype=dtype,
    )
    # Compare the two refined matrices via misorientation angle
    delta = U_cpu @ U_dev.T
    cos_a = max(-1.0, min(1.0, (np.trace(delta) - 1.0) / 2.0))
    misorient_deg = math.degrees(math.acos(cos_a))
    assert misorient_deg < atol_misorient, (
        f"CPU vs {_device_param} refinement disagrees by {misorient_deg:.3f}°"
    )


# ---------------------------------------------------------------------------
# 4. Failure modes / robustness
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_find_seed_orientation_raises_when_too_sparse():
    """With <2 bright cores, the seed indexer must raise, not silently return."""
    qx = np.array([0.5]); qy = np.array([0.0]); qz = np.array([0.0])
    inten = np.array([1000.0])
    with pytest.raises(ValueError, match="at least 2 bright cores"):
        find_seed_orientation(qx, qy, qz, inten, n_bright=5)


@pytest.mark.unit
def test_refine_raises_with_one_match():
    """Refinement needs ≥2 matched pairs."""
    rng = np.random.default_rng(0)
    U_true = _random_rotation(rng)
    a, c = CUAL2_A_DEFAULT, CUAL2_C_DEFAULT
    # 1 centroid only
    cent = (U_true @ _hkl_to_g_cry((1, 1, 0), a, c))[None, :]
    with pytest.raises(ValueError, match="at least 2"):
        refine_U_lattice(
            U_true, cent, np.array([1.0]), [(1, 1, 0)],
            a_init=a, c_init=c, n_steps=10,
        )


# ---------------------------------------------------------------------------
# 5. Friedel-pair refinement
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_friedel_pair_refinement_matches_centroid_refinement_at_noise_zero():
    """With centrosymmetric truth and zero noise, Friedel-avg and plain refine
    must converge to the same U (within autograd-step tolerance)."""
    rng = np.random.default_rng(2)
    U_true = _random_rotation(rng)
    a, c = CUAL2_A_DEFAULT, CUAL2_C_DEFAULT
    # Build centrosymmetric (h, -h) pairs from the truth (no noise)
    hkl_list = [(1, 1, 0), (-1, -1, 0), (2, 0, 0), (-2, 0, 0),
                (0, 0, 2), (0, 0, -2), (2, 1, 1), (-2, -1, -1)]
    pairs = []
    for h in hkl_list:
        g_cry = _hkl_to_g_cry(h, a, c)
        pairs.append((h, U_true @ g_cry))
    # Start from a slightly perturbed U
    U_init = U_true @ _rotvec_to_matrix(
        torch.as_tensor([0.02, -0.015, 0.01], dtype=torch.float64)
    ).numpy()
    U_centroid, _, _, _ = refine_U_from_centroids(
        U_init, pairs, a=a, c=c, n_steps=600, lr=5e-3, device=CPU,
    )
    U_friedel, _, _, _, n_pairs = refine_U_from_friedel_pairs(
        U_init, pairs, a=a, c=c, n_steps=600, lr=5e-3, device=CPU,
    )
    assert n_pairs == 4, f"expected 4 Friedel pairs, got {n_pairs}"
    delta = U_centroid @ U_friedel.T
    cos_a = max(-1.0, min(1.0, (np.trace(delta) - 1.0) / 2.0))
    misor_deg = math.degrees(math.acos(cos_a))
    assert misor_deg < 0.05, (
        f"Friedel vs centroid refinement disagree by {misor_deg:.4f}°"
    )


@pytest.mark.unit
def test_friedel_pair_refinement_reduces_centroid_loss_under_noise():
    """When centrosymmetric pairs are present with independent noise, Friedel
    averaging should give a strictly lower final loss than refining on the raw
    noisy centroids (because the noise averages out)."""
    rng = np.random.default_rng(5)
    U_true = _random_rotation(rng)
    a, c = CUAL2_A_DEFAULT, CUAL2_C_DEFAULT
    hkl_pos = [(1, 1, 0), (2, 0, 0), (0, 0, 2), (2, 1, 1), (3, 1, 0), (1, 1, 2)]
    noise_scale = 0.005
    pairs_noisy = []
    for h in hkl_pos:
        g_cry = _hkl_to_g_cry(h, a, c)
        q = U_true @ g_cry
        # add INDEPENDENT noise to (h) and (-h)
        pairs_noisy.append((h,                    q + rng.normal(scale=noise_scale, size=3)))
        pairs_noisy.append((tuple(-x for x in h), -q + rng.normal(scale=noise_scale, size=3)))
    U_init = U_true @ _rotvec_to_matrix(
        torch.as_tensor([0.01, -0.02, 0.015], dtype=torch.float64)
    ).numpy()
    _, _, _, loss_centroid = refine_U_from_centroids(
        U_init, pairs_noisy, a=a, c=c, n_steps=600, lr=5e-3, device=CPU,
    )
    _, _, _, loss_friedel, n_pairs = refine_U_from_friedel_pairs(
        U_init, pairs_noisy, a=a, c=c, n_steps=600, lr=5e-3, device=CPU,
    )
    assert n_pairs == len(hkl_pos)
    assert loss_friedel < loss_centroid, (
        f"Friedel loss {loss_friedel:.3e} should be < raw {loss_centroid:.3e}"
    )


@pytest.mark.unit
def test_friedel_pair_refinement_handles_unpaired_centroids():
    """When some hkls have no Friedel partner, they should still be used."""
    rng = np.random.default_rng(9)
    U_true = _random_rotation(rng)
    a, c = CUAL2_A_DEFAULT, CUAL2_C_DEFAULT
    # one Friedel pair, three unpaired → 4 constraints after collapsing
    pairs = []
    for h in [(1, 1, 0), (-1, -1, 0)]:
        pairs.append((h, U_true @ _hkl_to_g_cry(h, a, c)))
    pairs.append(((2, 0, 0), U_true @ _hkl_to_g_cry((2, 0, 0), a, c)))
    pairs.append(((0, 0, 2), U_true @ _hkl_to_g_cry((0, 0, 2), a, c)))
    pairs.append(((2, 1, 1), U_true @ _hkl_to_g_cry((2, 1, 1), a, c)))
    U_init = U_true @ _rotvec_to_matrix(
        torch.as_tensor([0.01, 0.0, 0.0], dtype=torch.float64)
    ).numpy()
    U_out, _, _, _, n_pairs = refine_U_from_friedel_pairs(
        U_init, pairs, a=a, c=c, n_steps=400, lr=5e-3, device=CPU,
    )
    assert n_pairs == 1, f"expected 1 paired set + 2 unpaired, got {n_pairs}"
    delta = U_out @ U_true.T
    cos_a = max(-1.0, min(1.0, (np.trace(delta) - 1.0) / 2.0))
    assert math.degrees(math.acos(cos_a)) < 0.5
