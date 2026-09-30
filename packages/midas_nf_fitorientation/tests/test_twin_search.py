"""twin_search: the local dNLL must equal a full re-evaluation of the whole image, and commits must be exact."""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from midas_nf_fitorientation.intensity import ContributionTable, RecordedPixels, SplatBlurModel
from midas_nf_fitorientation.twin_search import (BaseImage, LocalSearch, Nuisance, coherent_strip_normals,
                                                 sigma3_variants, strip_fractions)

NF, NY, NZ = 30, 80, 80
NUIS = Nuisance(sigma_psf=(0.9, 1.8), sigma_omega=0.2, blur_radius=4, median_radius=1, blanket=5.0,
                sigma_noise=0.8, kappa=0.4, t_dof=4.0, p=2.0)


def _rows(rng, n_vox, per_vox, grain, n_refl=5):
    v = np.repeat(np.arange(n_vox), per_vox); R = v.size
    return dict(grain=np.full(R, grain), refl=rng.integers(0, n_refl, R), sol=np.zeros(R, np.int8),
                frame=rng.uniform(2, NF - 3, R), y=rng.uniform(6, NY - 7, R), z=rng.uniform(6, NZ - 7, R),
                w_geom=rng.uniform(0.5, 2.0, R), voxel=v)


def _table(parts, n_refl=5):
    cat = lambda k: np.concatenate([p[k] for p in parts])
    return ContributionTable(cat("grain"), cat("refl"), cat("sol"), cat("frame"), cat("y"), cat("z"), cat("w_geom"),
                             np.log(np.linspace(1.5, 3.0, n_refl)), int(cat("grain").max()) + 1)


def _scene(seed=3):
    rng = np.random.default_rng(seed)
    g0 = _rows(rng, 6, 20, 0); g1 = _rows(rng, 6, 20, 1)
    cand = _rows(rng, 6, 20, 0)                 # a candidate orientation for grain 0's voxels (new spot positions)
    log_s = np.log([6.0, 8.0])
    base_tab = _table([g0, g1])
    # recorded data: truth = grain 0 voxels 0-2 twinned (fraction 1), voxels 3-5 parent; grain 1 intact
    tv = np.array([1, 1, 1, 0, 0, 0], float)
    g0t = dict(g0); g0t["w_geom"] = g0["w_geom"] * (1 - tv[g0["voxel"]])
    ct = dict(cand); ct["w_geom"] = cand["w_geom"] * tv[cand["voxel"]]
    truth = _table([g0t, g1, ct])
    m = SplatBlurModel(truth, NF, NY, NZ, blur_radius=4, sigma_psf=NUIS.sigma_psf, p=2.0, median_radius=1,
                       sigma_omega=NUIS.sigma_omega)
    with torch.no_grad():
        m.log_s[:] = torch.tensor(np.log([6.0, 8.0, 6.0][:truth.n_grains]))
        mu = m().numpy()
    x = mu + rng.normal(0, 0.8, mu.size); rec = x > 5.0
    f, rem = np.divmod(m.support_key[rec], NY * NZ); yy, zz = np.divmod(rem, NZ)
    R = RecordedPixels.from_arrays(f, yy, zz, x[rec] - 5.0, NY, NZ)
    return g0, g1, cand, log_s, base_tab, R


def _full_nll(g0, g1, cand, log_s, t, R):
    """Reference: rebuild the whole image with grain-0 voxels re-weighted by t (map 1-t, candidate t)."""
    a = dict(g0); a["w_geom"] = g0["w_geom"] * (1 - t[g0["voxel"]])
    c = dict(cand); c["w_geom"] = cand["w_geom"] * t[cand["voxel"]]
    c["grain"] = np.full(c["grain"].size, 2)
    keep = c["w_geom"] != 0
    c = {k: v[keep] for k, v in c.items()}
    tab = _table([a, g1, c])
    ls = np.r_[log_s, log_s[0]][:tab.n_grains]
    return BaseImage(tab, ls, R, NF, NY, NZ, NUIS).total_nll()


def test_local_dnll_equals_full_recompute_and_commit_is_exact():
    g0, g1, cand, log_s, base_tab, R = _scene()
    base = BaseImage(base_tab, log_s, R, NF, NY, NZ, NUIS)
    nll_base = base.total_nll()
    # the full-rebuild reference with t = 0 must reproduce the base
    assert _full_nll(g0, g1, cand, log_s, np.zeros(6), R) == pytest.approx(nll_base, rel=1e-9, abs=1e-6)
    region = _table([g0, cand])
    block = np.r_[np.zeros(g0["voxel"].size, int), np.ones(cand["voxel"].size, int)]
    vox = np.r_[g0["voxel"], cand["voxel"]]
    ls_row = np.full(vox.size, log_s[0])
    ls = LocalSearch(base, region, ls_row, block, vox)
    rng = np.random.default_rng(0)
    T = np.vstack([np.array([1, 1, 1, 0, 0, 0.]), np.array([0, 0, 0, 1, 1, 1.]), rng.uniform(0, 1, 6),
                   np.array([0.5, 0, 0, 0, 0, 0.])])
    W = np.where(block[None] == 0, -T[:, vox], T[:, vox])
    d_local = ls.dnll(torch.tensor(W, dtype=torch.float32)).numpy()
    d_full = np.array([_full_nll(g0, g1, cand, log_s, t, R) - nll_base for t in T])
    np.testing.assert_allclose(d_local, d_full, rtol=2e-4, atol=2e-3)
    assert d_local[0] < d_local[1]                    # the true twin (voxels 0-2) beats the wrong placement
    # commit the true hypothesis: the base must then equal the full rebuild, and re-scoring it must give ~0 change
    ls.commit(torch.tensor(W[0], dtype=torch.float32))
    assert base.total_nll() == pytest.approx(_full_nll(g0, g1, cand, log_s, T[0], R), rel=1e-6, abs=1e-3)


def test_coherent_strip_normals_are_the_in_layer_111_projection():
    rng = np.random.default_rng(1)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    from midas_stress.orientation import quat_to_orient_mat
    om = np.asarray(quat_to_orient_mat(q)).reshape(3, 3)
    m, mag = coherent_strip_normals(om)
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)
    n = om @ (np.array([1, 1, 1]) / math.sqrt(3))
    assert np.allclose(m[0], n[:2] / np.linalg.norm(n[:2])) and mag[0] == pytest.approx(np.linalg.norm(n[:2]))
    assert len(sigma3_variants(om)) == 4


def test_strip_fractions_exact_for_axis_aligned_strip():
    sub = torch.tensor(np.array([[[0.0, 0], [1, 0], [2, 0], [3, 0]]]))           # one voxel, 4 sub-points on x
    f = strip_fractions(sub, torch.tensor([[1.0, 0.0]]), torch.tensor([0.5]), torch.tensor([2.0]))
    assert float(f[0, 0]) == pytest.approx(0.5)                                     # x = 1, 2 inside [0.5, 2.5)
