"""The candidate-orientation sweep angle must be 360/n for every space group.

The indexer rotates each seed's candidate crystal about the seed reciprocal
vector G over ``[0, MaxAngle)``. Rotations about G that are symmetries of the
crystal predict identical spots, so the sweep must cover exactly ``360/n``
where ``n`` is the order of the stabilizer of G in the proper half of the Laue
group. Too small an angle skips orientations *silently*; too large only costs
time.

Why this file exists. The old hand-written branch table returned **0** for a
cubic (h,k,0) with h != k: seeding from garnet (420) generated zero candidates
and the run indexed 0 of 23 689 seeds while exiting 0 (20-ID nfdev_jul26
HPcat_P2, 2026-09-24). Auditing the table against the group theory found four
more wrong classes (m-3 treated as m-3m, 4/m as 4/mmm, monoclinic [100], 2-folds
perpendicular to c), and a C/Python disagreement on which hkls.csv row supplied
the indices.

Three independent layers, so that no single source of truth is trusted alone:

1. **Seitz reference, all 230 space groups.** midas_hkls carries every space
   group's operators; improper ones map to -R (Friedel), and ``n`` is counted
   exactly on integer Miller indices. Python and C must both equal it for every
   hkl in [-3, 3]^3 plus the families that have bitten (420), (642), ... -- in
   the default settings AND in the monoclinic c-/a-unique and rhombohedral-axis
   settings.
2. **Physics ground truth that uses no rotation table at all.** A generic atom
   gives a (G, |F|^2) set whose only symmetry is the Laue group; the largest k
   for which a 360/k rotation about G maps that set onto itself IS ``n``. One
   space group per Laue class and setting.
3. **Named regressions** for each class the old table got wrong.
"""
from __future__ import annotations

import functools
import itertools
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from midas_index.compute.rotation import calc_rotation_angle

hk = pytest.importorskip("midas_hkls")

_PKG = Path(__file__).resolve().parent.parent
_C_SRC = _PKG / "c_src"

# ----------------------------------------------------------------- lattices
# Generic cells (no accidental metric symmetry beyond the crystal system).
_TRI = (5.1, 6.3, 7.4, 81.0, 97.0, 103.0)
_MONO_B = (5.2, 6.1, 7.3, 90.0, 104.0, 90.0)
_MONO_C = (5.2, 6.1, 7.3, 90.0, 90.0, 104.0)
_MONO_A = (5.2, 6.1, 7.3, 104.0, 90.0, 90.0)
_ORTH = (5.1, 6.2, 7.3, 90.0, 90.0, 90.0)
_TET = (5.3, 5.3, 7.1, 90.0, 90.0, 90.0)
_HEX = (4.1, 4.1, 6.7, 90.0, 90.0, 120.0)
_RHOMB = (5.5, 5.5, 5.5, 55.0, 55.0, 55.0)
_CUB = (6.2, 6.2, 6.2, 90.0, 90.0, 90.0)


def _default_lattice(sg: int):
    if sg <= 2:
        return _TRI
    if sg <= 15:
        return _MONO_B
    if sg <= 74:
        return _ORTH
    if sg <= 142:
        return _TET
    if sg <= 194:
        return _HEX
    return _CUB


def _first_extension(sg: int, prefix: str):
    """First setting code of `sg` whose extension starts with `prefix`."""
    import json
    table = json.loads((Path(hk.__file__).parent / "data" / "space_groups.json").read_text())
    for e in table:
        if e["sg_number"] == sg and e["extension"].startswith(prefix):
            return e["extension"]
    return None


def _cases():
    """(label, sg, extension, lattice) for every SG + the non-default settings."""
    out = [(f"sg{n}", n, "", _default_lattice(n)) for n in range(1, 231)]
    for n in range(3, 16):
        for axis, lat in (("c", _MONO_C), ("a", _MONO_A)):
            ext = _first_extension(n, axis)
            if ext is not None:
                out.append((f"sg{n}{ext}", n, ext, lat))
    for n in (146, 148, 155, 160, 161, 166, 167):
        out.append((f"sg{n}R", n, "R", _RHOMB))
    return out


_HKLS = sorted(set(itertools.product(range(-3, 4), repeat=3)) - {(0, 0, 0)}
               | {(4, 2, 0), (2, 4, 0), (0, 4, 2), (6, 4, 2), (6, 4, 0), (8, 4, 0),
                  (5, 1, 1), (3, 3, 3), (6, 1, 1), (5, 3, 2), (4, 4, 4), (1, 2, 4)})


@functools.lru_cache(maxsize=None)
def _seitz_laue_rotations(sg: int, ext: str) -> np.ndarray:
    """Distinct proper-ized rotations (improper -> -R) of midas_hkls' Seitz ops."""
    g = hk.SpaceGroup.from_number(sg, ext)
    rots = set()
    for op in g.symmetry_operations():
        R = np.array(op.R, dtype=int).reshape(3, 3)
        if round(np.linalg.det(R)) == -1:
            R = -R
        rots.add(tuple(R.ravel()))
    return np.array(sorted(rots)).reshape(-1, 3, 3)


def _reference_angle(sg: int, ext: str, h) -> float:
    """360/n from midas_hkls' Seitz operators (independent of our generators)."""
    rots = _seitz_laue_rotations(sg, ext)
    hv = np.array(h, dtype=int)
    n = int(np.all(np.einsum("i,kij->kj", hv, rots) == hv, axis=1).sum())
    return 360.0 / n


def _reference_table():
    rows = []
    for label, sg, ext, lat in _cases():
        for h in _HKLS:
            rows.append((label, sg, ext, lat, h, _reference_angle(sg, ext, h)))
    return rows


@pytest.fixture(scope="module")
def reference_table():
    return _reference_table()


# ------------------------------------------------ 1a. Python vs Seitz, all SGs
def test_python_matches_seitz_reference_all_space_groups(reference_table):
    bad = []
    for label, sg, _ext, lat, h, ref in reference_table:
        got = calc_rotation_angle(0, sg, h, lat)
        if got != pytest.approx(ref):
            bad.append((label, h, got, ref))
    assert not bad, f"{len(bad)} mismatches, first 10: {bad[:10]}"


# ------------------------------------------------ 1b. C vs Seitz, all SGs
_VERSION_H = """#ifndef MIDAS_VERSION_H
#define MIDAS_VERSION_H
#define MIDAS_VERSION "test"
#define MIDAS_GIT_HASH ""
#define MIDAS_GIT_DATE ""
#define MIDAS_VERSION_STRING "midas-index v" MIDAS_VERSION
#endif
"""


def _build_c_harness(tmp_path: Path) -> Path:
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler available")
    (tmp_path / "midas_version.h").write_text(_VERSION_H)
    exe = tmp_path / "rot_angle"
    cmd = [cc, "-std=gnu99", "-fopenmp", "-O2",
           "-I", str(_C_SRC), "-I", str(tmp_path),
           str(_PKG / "tests" / "rotation_angle_test.c"),
           str(_C_SRC / "MIDAS_Math.c"), str(_C_SRC / "GetMisorientation.c"),
           str(_C_SRC / "forward.c"), "-lm", "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        if "omp.h" in r.stderr or "fopenmp" in r.stderr:
            pytest.skip(f"OpenMP unavailable: {r.stderr[:200]}")
        pytest.fail(f"compile failed:\n{r.stderr[-3000:]}")
    return exe


def test_c_matches_seitz_reference_all_space_groups(tmp_path, reference_table):
    exe = _build_c_harness(tmp_path)
    lines = "".join(
        f"{sg} {' '.join(f'{x:.6f}' for x in lat)} {h[0]} {h[1]} {h[2]}\n"
        for _l, sg, _e, lat, h, _r in reference_table)
    out = subprocess.run([str(exe)], input=lines, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    got = [float(x) for x in out.stdout.split()]
    assert len(got) == len(reference_table)
    bad = [(row[0], row[4], g, row[5]) for row, g in zip(reference_table, got)
           if abs(g - row[5]) > 1e-6]
    assert not bad, f"{len(bad)} C mismatches, first 10: {bad[:10]}"


# ------------------------------------------------ 2. physics ground truth
_PHYS_CASES = [
    (2, "", _TRI), (14, "", _MONO_B), (14, "c1", _MONO_C), (14, "a1", _MONO_A), (5, "c1", _MONO_C),
    (62, "", _ORTH), (87, "", _TET), (139, "", _TET),
    (148, "", _HEX), (148, "R", _RHOMB), (164, "", _HEX), (162, "", _HEX),
    (166, "R", _RHOMB), (176, "", _HEX), (194, "", _HEX),
    (205, "", _CUB), (206, "", _CUB), (225, "", _CUB), (230, "", _CUB),
]
_PHYS_SEEDS = [h for h in itertools.product((-1, 0, 1), repeat=3) if any(h)] + [
    (2, 1, 0), (4, 2, 0), (1, 2, 0), (2, 1, 1), (6, 4, 2), (1, 1, 2)]


def _rotation(axis, angle_deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(angle_deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


@functools.lru_cache(maxsize=None)
def _physics_set(sg, ext, lat, nbox=6):
    """(G, |F|^2, inside-mask, KD-tree, B*) for one generic-atom crystal."""
    import torch
    from midas_hkls.structure_factor import structure_factors
    lattice = hk.Lattice(*lat)
    cry = hk.Crystal(lattice, hk.SpaceGroup.from_number(sg, ext),
                     [hk.Atom("Fe", (0.1234, 0.2718, 0.3927))])
    H = np.array([h for h in itertools.product(range(-nbox, nbox + 1), repeat=3) if any(h)])
    F = structure_factors(cry.to_torch(), torch.as_tensor(H))
    f2 = np.abs(F.detach().cpu().numpy()) ** 2
    Bstar = lattice.reciprocal_cartesian_vectors()
    G = H @ Bstar
    # |G| <= R guarantees the rotated vector is still inside the enumerated box.
    R = nbox / max(np.linalg.norm(lattice.cartesian_vectors(), axis=1))
    inside = np.linalg.norm(G, axis=1) <= R
    from scipy.spatial import cKDTree
    return G, f2, inside, cKDTree(G), Bstar


def _physics_order(sg, ext, lat, seed):
    """Largest k such that a 360/k rotation about G(seed) maps (G, |F|^2) onto itself."""
    G, f2, inside, tree, Bstar = _physics_set(sg, ext, lat)
    scale = np.linalg.norm(Bstar, axis=1).min()
    fmax = f2.max()
    axis = np.asarray(seed) @ Bstar
    best = 1
    for k in (2, 3, 4, 6):
        Gr = G[inside] @ _rotation(axis, 360.0 / k).T
        d, j = tree.query(Gr)
        if np.all(d < 1e-6 * scale) and np.allclose(f2[inside], f2[j], atol=1e-6 * fmax):
            best = max(best, k)
    return best


@pytest.mark.parametrize("sg,ext,lat", _PHYS_CASES,
                         ids=[f"sg{s}{e}" for s, e, _ in _PHYS_CASES])
def test_sweep_angle_matches_structure_factor_invariance(sg, ext, lat):
    pytest.importorskip("scipy")
    pytest.importorskip("torch")
    bad = []
    for seed in _PHYS_SEEDS:
        n_true = _physics_order(sg, ext, lat, seed)
        got = calc_rotation_angle(0, sg, seed, lat)
        if got != pytest.approx(360.0 / n_true):
            bad.append((seed, got, 360.0 / n_true))
    assert not bad, f"SG {sg}{ext}: {bad}"


# ------------------------------------------------ 3. named regressions
@pytest.mark.parametrize("sg,lat,hkl,expected,why", [
    (230, _CUB, (4, 2, 0), 360.0, "garnet (420): old C returned 0 -> 0 seeds"),
    (230, _CUB, (-4, -2, 0), 360.0, "the row hkls.csv actually lists first"),
    (225, _CUB, (2, 0, 0), 90.0, "m-3m <100> 4-fold"),
    (225, _CUB, (2, 2, 0), 180.0, "m-3m <110> 2-fold"),
    (225, _CUB, (1, 1, 1), 120.0, "m-3m <111> 3-fold"),
    (205, _CUB, (2, 0, 0), 180.0, "m-3 <100> is only 2-fold (old: 90)"),
    (205, _CUB, (2, 2, 0), 360.0, "m-3 has no <110> axis (old: 180)"),
    (87, _TET, (1, 0, 0), 360.0, "4/m has no 2-fold on a (old: 180)"),
    (87, _TET, (1, 1, 0), 360.0, "4/m has no 2-fold on [110] (old: 180)"),
    (87, _TET, (0, 0, 1), 90.0, "4/m 4-fold on c"),
    (139, _TET, (1, 0, 0), 180.0, "4/mmm 2-fold on a"),
    (14, _MONO_B, (1, 0, 0), 360.0, "b-unique: [100] is not an axis (old: 180)"),
    (14, _MONO_B, (0, 1, 0), 180.0, "b-unique 2-fold"),
    (14, _MONO_C, (0, 0, 1), 180.0, "c-unique 2-fold"),
    (194, _HEX, (0, 0, 1), 60.0, "6-fold on c"),
    (176, _HEX, (1, 0, 0), 360.0, "6/m: no 2-fold perpendicular to c"),
    (148, _HEX, (0, 0, 3), 120.0, "-3 on hexagonal axes"),
    (148, _RHOMB, (1, 1, 1), 120.0, "-3 on rhombohedral axes: 3-fold on [111]"),
    (1, _TRI, (1, 0, 0), 360.0, "triclinic"),
])
def test_named_regressions(sg, lat, hkl, expected, why):
    assert calc_rotation_angle(0, sg, hkl, lat) == pytest.approx(expected), why
