"""HCP twin operators must live in the same crystal frame as the orientations.

``Grains.csv`` orientation matrices (and everything ``midas-index`` /
``midas-fit-grain`` produce) are crystal->lab matrices whose crystal Cartesian
frame is the ``midas_hkls`` B-matrix frame: ``G_cart = B·(h, k, l)`` with a*
along x, b* in the xy plane, c* along z. That frame is rotated 30 deg about c
from the textbook a1-along-x frame. The 12 hexagonal symmetry operators and all
disorientation ANGLES are identical in the two frames, so a test that only looks
at an angle (or that plants twins with the package's own operator) cannot see a
frame error. These tests build the truth from ``midas_hkls`` instead.

History: until 2026-10-20 ``default_hcp_twin_relations`` used the textbook frame
and found 0 of 24 exact twins built in the MIDAS frame.
"""
import math

import numpy as np
import pytest

pytest.importorskip("midas_hkls")
from midas_hkls.lattice import Lattice                      # noqa: E402
from midas_hkls.nf_hkls import _b_matrix                    # noqa: E402
from midas_stress.orientation import orient_mat_to_quat     # noqa: E402

from midas_process_grains.compute.twins import (            # noqa: E402
    TwinRelation,
    default_hcp_twin_relations,
    find_twin_pairs,
    hcp_hkil_to_cartesian_normal,
)

CA = 1.587  # alpha-Ti
SYSTEMS = {
    # name: (3-index (h, k, l) of K1, expected disorientation deg, axis family)
    "tension_1012": ((1, 0, 2), 85.00, "a"),
    "compression_1011": ((1, 0, 1), 57.24, "a"),
    "tension_1121": ((1, 1, 1), 34.98, "astar"),
    "compression_1122": ((1, 1, 2), 64.43, "astar"),
}


def _B():
    return _b_matrix(Lattice(1.0, 1.0, CA, 90.0, 90.0, 120.0))


def _rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _d6():
    g = [_rz(math.radians(60 * k)) for k in range(6)]
    g += [m @ np.diag([1.0, -1.0, -1.0]) for m in g]
    return np.array(g)


def _rot180(n):
    return 2.0 * np.outer(n, n) - np.eye(3)


def _random_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat(g):
    return np.asarray(orient_mat_to_quat(g.reshape(9)), dtype=float)


def _disorientation(R):
    """(angle deg, unit axis) of R reduced over D6 x D6 (brute force)."""
    best = None
    for s in _d6():
        for t in _d6():
            m = s @ R @ t
            a = math.degrees(math.acos(np.clip((np.trace(m) - 1) / 2, -1, 1)))
            if best is None or a < best[0] - 1e-9:
                best = (a, m)
    a, m = best
    w, v = np.linalg.eig(m)
    ax = np.real(v[:, np.argmin(abs(w - 1))])
    return a, ax / np.linalg.norm(ax)


@pytest.mark.parametrize("hkil", [(1, 0, -1, 2), (1, 0, -1, 1), (1, 1, -2, 1),
                                  (1, 1, -2, 2), (2, -1, -1, 2), (0, 1, -1, 3)])
def test_normal_matches_midas_hkls_b_matrix(hkil):
    h, k, i, l = hkil
    expected = _B() @ np.array([h, k, l], float)
    expected /= np.linalg.norm(expected)
    n = hcp_hkil_to_cartesian_normal(h, k, i, l, CA)           # default = MIDAS frame
    np.testing.assert_allclose(n, expected, atol=1e-12)


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_operator_angle_and_axis_in_midas_frame(name):
    """Angle AND axis of the shipped operators, derived independently from B.

    The axis is checked against the real-space a axes (b*×c*, c*×a*, ...) and
    the reciprocal a* axes, both taken from the B matrix, so a 30 deg frame error
    fails here although the angle would still pass.
    """
    (hkl, angle, fam) = SYSTEMS[name]
    B = _B()
    a_star, b_star, c_star = B[:, 0], B[:, 1], B[:, 2]
    a_axes = [np.cross(b_star, c_star), np.cross(c_star, a_star)]
    a_axes.append(-(a_axes[0] + a_axes[1]))
    astar_axes = [a_star, b_star, a_star - b_star]
    targets = a_axes if fam == "a" else astar_axes
    targets = [t / np.linalg.norm(t) for t in targets]

    rels = default_hcp_twin_relations(CA, systems=(name,))
    assert len(rels) == 6
    for r in rels:
        q = r.quaternion                      # (0, n): 180 deg about the K1 normal
        n = np.asarray(q[1:], float)
        a, ax = _disorientation(_rot180(n / np.linalg.norm(n)))
        assert abs(a - angle) < 0.05, (name, a)
        assert max(abs(ax @ t) for t in targets) > 1 - 1e-6, (name, ax)


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_find_twin_pairs_recovers_exact_midas_frame_twins(name):
    """24 exact twins built from the B matrix, with random variant and random
    twin-side symmetry operator, must all be found (the textbook-frame operators
    found 0 of 24)."""
    (hkl, _, _) = SYSTEMS[name]
    B, G = _B(), _d6()
    n0 = B @ np.array(hkl, float)
    n0 /= np.linalg.norm(n0)
    rng = np.random.default_rng(31)
    rels = default_hcp_twin_relations(CA, systems=(name,))
    found = 0
    for _ in range(24):
        gp = _random_rot(rng)
        T = _rot180(G[rng.integers(12)] @ n0)                  # random K1 variant
        gt = gp @ T @ G[rng.integers(12)]
        q = np.stack([_quat(gp), _quat(gt)])
        found += bool(find_twin_pairs(q, 194, rels, tol_rad=math.radians(0.5)))
    assert found == 24, f"{name}: found {found}/24"


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_frame_is_a_real_switch_not_a_relabel(name):
    """Operators built in the other frame find none of the MIDAS-frame twins, and
    vice versa. Guards against the frame option silently doing nothing."""
    (hkl, _, _) = SYSTEMS[name]
    B, G = _B(), _d6()
    n0 = B @ np.array(hkl, float)
    n0 /= np.linalg.norm(n0)
    rng = np.random.default_rng(5)
    midas = default_hcp_twin_relations(CA, systems=(name,), frame="midas")
    textb = default_hcp_twin_relations(CA, systems=(name,), frame="crystallographic")
    hit_m = hit_t = 0
    for _ in range(12):
        gp = _random_rot(rng)
        gt = gp @ _rot180(G[rng.integers(12)] @ n0) @ G[rng.integers(12)]
        q = np.stack([_quat(gp), _quat(gt)])
        hit_m += bool(find_twin_pairs(q, 194, midas, tol_rad=math.radians(0.5)))
        hit_t += bool(find_twin_pairs(q, 194, textb, tol_rad=math.radians(0.5)))
    assert hit_m == 12 and hit_t == 0


def test_compression_2112_is_the_1122_plane_not_a_new_twin():
    """(2 -1 -1 2) is the {11-22} family: same operators up to symmetry, so it is
    kept selectable but out of the default list. (The old (2 1 -3 2) plane gave
    49.6 deg about an inclined axis, which is not a twin.)"""
    dflt = default_hcp_twin_relations(CA)
    assert not any("compression_2112" in r.name for r in dflt)
    r2112 = default_hcp_twin_relations(CA, systems=("compression_2112",))
    a, _ = _disorientation(_rot180(np.asarray(r2112[0].quaternion[1:], float)))
    assert abs(a - 64.43) < 0.05


# ---------------------------------------------------------------------------
# label_twins (Stage 5 / v4 'physics' mode) must be symmetry-aware
# ---------------------------------------------------------------------------
# Its bucket prefilter used raw quaternions, so a twin was only found when the
# stored representatives happened to line up: planted {10-12} pairs with a random
# twin-side symmetry operator were labelled 5/40 (raw) and 0/40 (fundamental-zone
# quaternions, which is what v4_pipeline passes). Real data always carries an
# arbitrary representative, so the prefilter has to see the symmetry equivalents.

from midas_process_grains.compute.twin_label import label_twins   # noqa: E402
from midas_stress.orientation import fundamental_zone             # noqa: E402


def _planted_hcp(rng, n, name, random_sym):
    (hkl, _, _) = SYSTEMS[name]
    B, G = _B(), _d6()
    n0 = B @ np.array(hkl, float)
    n0 /= np.linalg.norm(n0)
    P, Tq = [], []
    for _ in range(n):
        gp = _random_rot(rng)
        sj = G[rng.integers(12)] if random_sym else np.eye(3)
        gt = gp @ _rot180(G[rng.integers(12)] @ n0) @ sj
        P.append(_quat(gp))
        Tq.append(_quat(gt))
    return np.array(P + Tq)


def _n_found(partner, n):
    return sum(1 for i in range(n) if partner[i] == i + n or partner[i + n] == i)


@pytest.mark.parametrize("name", list(SYSTEMS))
@pytest.mark.parametrize("fz", [False, True], ids=["raw", "fundamental_zone"])
def test_label_twins_finds_hcp_twins_with_random_symmetry_representatives(name, fz):
    rng = np.random.default_rng(17)
    n = 30
    quats = _planted_hcp(rng, n, name, random_sym=True)
    if fz:
        quats = np.array([np.asarray(fundamental_zone(q, 194), float) for q in quats])
    partner, _, ttype, n_pairs = label_twins(
        grain_quats=quats, space_group=194, c_over_a=CA, tol_deg=0.5)
    assert _n_found(partner, n) == n, f"{name} fz={fz}: {_n_found(partner, n)}/{n}"
    assert all(name in ttype[i] for i in range(n) if ttype[i])


def test_label_twins_finds_fcc_sigma3_with_random_symmetry_representatives():
    from midas_stress.orientation import make_symmetries, quat_to_orient_mat
    from midas_process_grains.compute.twin_label import _qmul
    from midas_process_grains.compute.twins import default_fcc_twin_relations
    rng = np.random.default_rng(23)
    n_sym, sym = make_symmetries(225)
    sym = np.asarray(sym, float)[:n_sym]
    T = default_fcc_twin_relations()[0].quaternion
    n = 30
    P, Tq = [], []
    for _ in range(n):
        qp = _quat(_random_rot(rng))
        qt = _qmul(_qmul(qp[None], T[None])[0][None], sym[rng.integers(n_sym)][None])[0]
        P.append(qp)
        Tq.append(qt / np.linalg.norm(qt))
    quats = np.array(P + Tq)
    quats = np.array([np.asarray(fundamental_zone(q, 225), float) for q in quats])
    partner, _, _, _ = label_twins(grain_quats=quats, space_group=225, tol_deg=0.5)
    assert _n_found(partner, n) == n


def test_label_twins_false_positive_rate_on_random_orientations_is_small():
    rng = np.random.default_rng(41)
    quats = np.array([_quat(_random_rot(rng)) for _ in range(300)])
    _, _, _, n_pairs = label_twins(grain_quats=quats, space_group=194,
                                   c_over_a=CA, tol_deg=0.5)
    # 300 random grains, 4 systems x 6 variants at 0.5 deg: chance pairs are rare
    assert n_pairs <= 6, n_pairs
