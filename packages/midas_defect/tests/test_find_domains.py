"""The per-position multi-domain search must find planted domains and nothing in structureless data."""
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from midas_defect.domains import find_domains

A, C = 3.6116, 19.2516
ROT422 = [np.eye(3)] + [Rotation.from_euler("z", k * 90, degrees=True).as_matrix() for k in (1, 2, 3)]
ROT422 += [R @ np.diag([1.0, -1.0, -1.0]) for R in ROT422]


def _miso(Ua, Ub):
    best = 180.0
    for R in ROT422:
        M = Ua.T @ (Ub @ R)
        best = min(best, math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(M) - 1.0) / 2.0)))))
    return best


def _domain(U, rng, keep=0.6, qmax=4.5, noise=0.003):
    B = np.diag([2 * math.pi / A, 2 * math.pi / A, 2 * math.pi / C])
    hkl = np.array([(h, k, l) for h in range(-4, 5) for k in range(-4, 5) for l in range(-22, 23)
                    if (h, k, l) != (0, 0, 0) and (h + k + l) % 2 == 0], float)
    qq = (U @ B @ hkl.T).T
    sel = (np.linalg.norm(qq, axis=1) < qmax) & (rng.random(len(hkl)) < keep)
    return qq[sel] + rng.normal(0, noise, (int(sel.sum()), 3))


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(7)
    U1 = Rotation.from_euler("zyx", [20.0, 35.0, -10.0], degrees=True).as_matrix()
    U2 = Rotation.from_euler("zyx", [-55.0, 12.0, 40.0], degrees=True).as_matrix()
    q1, q2 = _domain(U1, rng), _domain(U2, rng)
    dirs = rng.normal(size=(40, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    qn = dirs * rng.uniform(1.0, 4.0, (40, 1))
    q = np.vstack([q1, q2, qn])
    I = np.r_[1e4 + 1e5 * rng.random(len(q1)), 2e2 + 2e3 * rng.random(len(q2)), 5e2 * rng.random(40)]
    lab = np.r_[np.zeros(len(q1), int), np.ones(len(q2), int), np.full(40, -1)]
    n = len(q)
    return dict(q=q, I=I, lab=lab, U=(U1, U2), row=rng.uniform(0, 1679, n), col=rng.uniform(0, 1475, n),
                frame=rng.uniform(0, 40, n))


def _run(s, q=None, **kw):
    return find_domains(s["q"] if q is None else q, s["I"], s["row"], s["col"], s["frame"],
                        a=A, c=C, space_group_number=139, null_reps=5, **kw)


def test_finds_both_planted_domains_and_no_third(scene):
    res = _run(scene)
    assert len(res.domains) == 2, str(res)
    for U in scene["U"]:
        best = min(_miso(U, d.U) for d in res.domains)
        assert best < 0.5, f"planted orientation recovered only to {best:.2f} deg; {res}"
    for d in res.domains:
        assert abs(d.lat.a / A - 1) < 0.005 and abs(d.lat.c / C - 1) < 0.005, str(res)
        assert len(d.hkl) == d.n                                   # hkl aligned with the claim
    lab = scene["lab"]
    assert res.explained[lab >= 0].mean() > 0.8, str(res)
    assert res.explained[lab < 0].mean() < 0.25, str(res)


def test_structureless_data_of_the_same_kind_gives_no_domain(scene):
    rng = np.random.default_rng(11)
    d = rng.normal(size=scene["q"].shape); d /= np.linalg.norm(d, axis=1, keepdims=True)
    scrambled = d * np.linalg.norm(scene["q"], axis=1, keepdims=True)   # every |q| kept, directions destroyed
    res = _run(scene, q=scrambled)
    assert len(res.domains) == 0, str(res)


def test_nothing_seeds_from_spots_that_may_not_seed(scene):
    res = _run(scene, seedable=np.zeros(len(scene["q"]), bool))
    assert len(res.domains) == 0, str(res)


def test_the_cell_and_space_group_are_required(scene):
    with pytest.raises(TypeError):
        find_domains(scene["q"], scene["I"], scene["row"], scene["col"], scene["frame"])


def test_nominal_seed_runs_the_pair_branch_when_no_row_domain_exists(scene):
    """S5 at 30 K: c* near the beam, no lattice row, `find_domains` returned nothing (dry run 2026-09-10).
    Here every row is refused by an impossible match floor, which leaves the pair branch with no seed."""
    assert len(_run(scene, min_row_match=10**6).domains) == 0            # documented default behaviour
    res = _run(scene, min_row_match=10**6, seed_from_nominal=True)
    assert len(res.domains) == 2, str(res)
    for U in scene["U"]:
        assert min(_miso(U, d.U) for d in res.domains) < 0.5, str(res)
    for d in res.domains:
        assert d.branch == "pair" and d.seed_source == "nominal", str(res)
        assert abs(d.lat.a / A - 1) < 0.005 and abs(d.lat.c / C - 1) < 0.005, str(res)
    assert "nominal" in str(res)
