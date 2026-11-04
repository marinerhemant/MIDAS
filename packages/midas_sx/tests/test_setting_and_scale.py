"""match_setting (found lattice -> reference cell setting) and scale_groups (relative run scales)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

pytest.importorskip("gemmi")
from midas_hkls import Lattice, SpaceGroup                           # noqa: E402
from midas_sx.assign import reciprocal_basis                         # noqa: E402
from midas_sx.merge import merge_observations, scale_groups          # noqa: E402
from midas_sx.setting import match_setting                           # noqa: E402
import pandas as pd                                                  # noqa: E402

REF = Lattice(7.4962, 8.5211, 5.4695, 90, 90, 90)


def _found(U, perm, signs):
    """A 'found' basis: the reference reciprocal vectors reordered and flipped, as ab initio would hand them over."""
    B = reciprocal_basis(REF)
    cols = [signs[j] * (U @ B[:, perm[j]]) for j in range(3)]
    return np.stack(cols, axis=1)


@pytest.mark.parametrize("seed", range(6))
def test_match_setting_recovers_the_orientation_for_any_signed_permutation(seed):
    rng = np.random.default_rng(seed)
    U = Rot.random(random_state=seed).as_matrix()
    perm = rng.permutation(3)
    signs = rng.choice([-1, 1], 3)
    found = _found(U, perm, signs)
    Uf, P, err = match_setting(found, REF)
    assert err < 1e-9 and np.linalg.det(Uf) == pytest.approx(1.0)
    # the recovered U reproduces the reference reciprocal lattice (it can differ from U by a mirror-equivalent operation of mmm)
    B = reciprocal_basis(REF)
    pts = np.array([[h, k, l] for h in range(-3, 4) for k in range(-3, 4) for l in range(-3, 4)], float)
    a = (U @ B @ pts.T).T
    b = (Uf @ B @ pts.T).T
    ka = {tuple(np.round(v, 6)) for v in a}
    kb = {tuple(np.round(v, 6)) for v in b}
    assert ka == kb


@pytest.mark.parametrize("seed", range(4))
def test_a_left_handed_found_basis_still_gives_a_proper_rotation(seed):
    """ab initio hands over a basis of either handedness; the same orthorhombic lattice with the opposite handedness must give U with det +1 (the earlier version only required
    det(P) = +1 and returned a reflection, polar-decomposed into a wrong rotation: thiourea runs 03 and 05 failed to refine)."""
    rng = np.random.default_rng(100 + seed)
    U = Rot.random(random_state=100 + seed).as_matrix()
    found = _found(U, rng.permutation(3), rng.choice([-1, 1], 3))
    found = found * np.array([-1.0, 1.0, 1.0])                                  # flip one axis: left-handed relative to the proper version
    for cand in (found, found[:, [1, 0, 2]]):
        Uf, P, err = match_setting(cand, REF)
        assert np.linalg.det(Uf) == pytest.approx(1.0) and err < 1e-9


def test_match_setting_refuses_a_different_lattice():
    other = Lattice(5.0, 9.3, 12.0, 90, 90, 90)
    Bo = reciprocal_basis(other)
    with pytest.raises(ValueError, match="no signed permutation"):
        match_setting(Rot.random(random_state=1).as_matrix() @ Bo, REF)


def test_scale_groups_recovers_known_run_scales():
    rng = np.random.default_rng(0)
    sg = SpaceGroup.from_hm("P 21 m a")
    hkl = np.array([(h, k, l) for h in range(0, 6) for k in range(0, 6) for l in range(0, 4) if (h, k, l) != (0, 0, 0)])
    truth = 100.0 + 400.0 * rng.random(len(hkl))
    true_k = {"r1": 1.0, "r2": 0.85, "r3": 1.2}
    rows = []
    for run, kk in true_k.items():
        pick = rng.random(len(hkl)) < 0.8
        sig = 0.02 * truth[pick]
        for (h, k, l), t, s_ in zip(hkl[pick], truth[pick], sig):
            rows.append(dict(run=run, h=h, k=k, l=l, F2=kk * t + rng.normal() * s_ * kk, sigF2=s_ * kk))
    obs = pd.DataFrame(rows)
    scaled, tab = scale_groups(obs, sg)
    k = dict(zip(tab.group, tab.scale))
    ratio = np.array([k["r2"] / k["r1"], k["r3"] / k["r1"]])
    assert np.allclose(ratio, [1 / 0.85, 1 / 1.2], rtol=0.01), (ratio, tab)        # scale undoes the run factor
    assert np.exp(np.mean(np.log(list(k.values())))) == pytest.approx(1.0, abs=1e-6)
    r_before = merge_observations(obs, sg)
    r_after = merge_observations(scaled, sg)
    assert r_after.R_int_refl.mean() < 0.5 * r_before.R_int_refl.mean()


def test_scale_groups_single_group_and_disjoint_group():
    sg = SpaceGroup.from_hm("P 21 m a")
    obs = pd.DataFrame(dict(run=["a", "a", "b"], h=[1, 1, 5], k=[0, 0, 5], l=[1, 1, 3], F2=[10.0, 12.0, 7.0], sigF2=[1.0, 1.0, 1.0]))
    scaled, tab = scale_groups(obs, sg)
    assert tab.set_index("group").loc["b", "n_shared"] == 0 and tab.set_index("group").loc["b", "scale"] == 1.0
    one = scale_groups(obs[obs.run == "a"], sg)[1]
    assert one.scale.iloc[0] == pytest.approx(1.0)
