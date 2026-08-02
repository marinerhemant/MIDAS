"""Tests for the Laue reader and plots.

The properties checked here are the ones whose failure would be silent: a
column read by the wrong index, a misorientation that ignores symmetry, a tilt
histogram without its reference, a pole density quoted without the chance level
that makes it comparable.
"""
import warnings

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from midas_plotting import laue
from midas_plotting.solutions import (
    COS45, read_solutions, read_spots, read_validated,
)

SG_HEX, SG_CUB = 194, 225

SOL_HEADER = (
    "%ImageNr\tGrainNr\tNumberOfSolutions\tIntensity\tNMatches*Intensity\t"
    "NMatches*sqrt(Intensity)\tNMatches\tNSpotsCalc\t"
    + "\t".join(f"Recip{i}" for i in range(1, 10)) + "\t"
    + "\t".join(["LatticeParameterFit[a]", "LatticeParameterFit[b]",
                 "LatticeParameterFit[c]", "LatticeParameterFit[alpha]",
                 "LatticeParameterFit[beta]", "LatticeParameterFit[gamma]"])
    + "\t" + "\t".join(f"OrientMatrix{i}" for i in range(9))
    + "\tCoarseNMatches*sqrt(Intensity)\tmisOrientationPostRefinement"
      "\torientationRowNr\n"
)


def _rot(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def _write_solutions(path, oms, n_matches, row_nrs, misos=None):
    misos = np.zeros(len(oms)) if misos is None else misos
    with open(path, "w") as fh:
        fh.write(SOL_HEADER)
        for i, (om, nm, rn, mo) in enumerate(zip(oms, n_matches, row_nrs, misos)):
            row = ([i + 1, 1, 5, 1000.0, 1.0, 1.0, nm, 40]
                   + [0.0] * 9
                   + [0.26649, 0.26649, 0.49468, 90.0, 90.0, 120.0]
                   + list(np.asarray(om, float).ravel())
                   + [1.0, mo, rn])
            fh.write("\t".join(f"{v}" for v in row) + "\n")


# ---------------------------------------------------------------- reader

def test_reads_columns_by_name_not_position(tmp_path):
    """orientationRowNr is column 34 and misOrientation is 33.

    Reading 33 for 34 does not raise -- it returns a float near zero for every
    row, which silently collapses every distinct-orientation count. The reader
    must pick them apart by name.
    """
    p = tmp_path / "solutions.txt"
    oms = [np.eye(3), _rot([0, 0, 1], 30.0), np.eye(3)]
    _write_solutions(p, oms, [20, 18, 17], row_nrs=[7, 99, 7],
                     misos=[0.01, 0.02, 0.03])
    s = read_solutions(p)
    assert len(s) == 3
    assert list(s.row_nr) == [7, 99, 7]
    assert s.n_distinct == 2                       # not 3, and not 1
    assert np.allclose(s.misorientation, [0.01, 0.02, 0.03])
    assert list(s.n_matches) == [20, 18, 17]


def test_missing_row_nr_warns_and_does_not_substitute(tmp_path):
    p = tmp_path / "solutions.txt"
    _write_solutions(p, [np.eye(3)], [15], [3])
    text = p.read_text().replace("\torientationRowNr", "")
    lines = text.splitlines()
    lines[1] = "\t".join(lines[1].split("\t")[:-1])
    p.write_text("\n".join(lines) + "\n")
    with pytest.warns(RuntimeWarning, match="orientationRowNr"):
        s = read_solutions(p)
    assert s.row_nr is None
    assert s.n_distinct is None                    # not silently misorientation


def test_non_rotation_matrix_warns(tmp_path):
    """A wrong column offset produces matrices that are not proper rotations."""
    p = tmp_path / "solutions.txt"
    bad = np.eye(3) * 2.0
    _write_solutions(p, [bad], [15], [1])
    with pytest.warns(RuntimeWarning, match="not proper rotations"):
        read_solutions(p)


def test_header_mismatch_raises(tmp_path):
    p = tmp_path / "solutions.txt"
    p.write_text("ImageNr GrainNr\n1 2\n")          # no leading %
    with pytest.raises(ValueError, match="header"):
        read_solutions(p)


def test_gate_is_strictly_greater(tmp_path):
    p = tmp_path / "solutions.txt"
    _write_solutions(p, [np.eye(3)] * 3, [11, 12, 13], [1, 2, 3])
    s = read_solutions(p)
    assert list(s.gate(11).n_matches) == [12, 13]   # 11 itself is excluded


def test_read_spots(tmp_path):
    p = tmp_path / "spots.txt"
    p.write_text("%ImageNr\tGrainNr\tSpotNr\th\tk\tl\tX\tY\tQhat[0]\tQhat[1]"
                 "\tQhat[2]\tIntensity\n"
                 "1\t1\t0\t0\t0\t2\t988\t788\t0.1\t0.2\t0.3\t1000\n"
                 "1\t2\t1\t1\t0\t1\t100\t200\t0.4\t0.5\t0.6\t50\n"
                 "2\t1\t0\t1\t1\t1\t300\t400\t0.7\t0.1\t0.2\t70\n")
    sp = read_spots(p)
    assert len(sp) == 3
    assert np.allclose(sp.for_frame(1).xy[0], [988, 788])
    assert len(sp.for_frame(2)) == 1


def test_validated_npz_applies_the_45_degree_correction(tmp_path):
    """Z in these files is a STAGE coordinate; true in-sample distance is Z/cos45.

    Quoting the raw extent understates the map by 1.41x -- a 200 x 100 um map
    reads as 200 x 71.
    """
    p = tmp_path / "v.npz"
    np.savez(p, oms=np.stack([np.eye(3)] * 4),
             X=np.array([0.0, 10.0, 0.0, 10.0]),
             Z=np.array([0.0, 0.0, 70.71067811865476, 70.71067811865476]),
             nhit=np.array([20, 20, 20, 20]))
    s = read_validated(p)
    assert np.ptp(s.pos[:, 1]) == pytest.approx(100.0, abs=1e-6)
    raw = read_validated(p, raw_z=False)
    assert np.ptp(raw.pos[:, 1]) == pytest.approx(70.7106781, abs=1e-6)


# ---------------------------------------------------------------- maths

def test_validated_npz_frames_may_be_filenames(tmp_path):
    """Some versions write ``frames`` as frame FILENAMES, not integers.

    Found on real data: int('scan100Cu_1.h5') raises. The names are kept and the
    image index is derived, because silently coercing them would be a crash at
    best and a wrong join at worst.
    """
    p = tmp_path / "v.npz"
    np.savez(p, oms=np.stack([np.eye(3)] * 3),
             X=np.array([0.0, 1.0, 2.0]), Z=np.zeros(3),
             nhit=np.array([20, 20, 20]),
             frames=np.array(["s_1.h5", "s_2.h5", "s_1.h5"]))
    s = read_validated(p)
    assert s.frame_name is not None
    assert list(s.frame_name) == ["s_1.h5", "s_2.h5", "s_1.h5"]
    assert np.unique(s.image).size == 2          # two distinct frames
    assert s[[0, 1]].frame_name is not None      # survives slicing


def test_validated_npz_frames_as_integers(tmp_path):
    p = tmp_path / "v.npz"
    np.savez(p, oms=np.stack([np.eye(3)] * 2), X=np.zeros(2), Z=np.zeros(2),
             nhit=np.array([20, 20]), frames=np.array([4, 9]))
    s = read_validated(p)
    assert s.frame_name is None
    assert list(s.image) == [4, 9]


def test_random_tilt_reference_is_exact():
    """Half a random population lies beyond 60 deg, purely from solid angle.

    This is the number that decides whether a tilt histogram reads as a texture
    or as nothing.
    """
    f = laue.random_tilt_fractions()
    assert f.sum() == pytest.approx(1.0)
    assert f[-1] == pytest.approx(0.5)              # the 60-90 band
    assert f[0] == pytest.approx(1 - np.cos(np.radians(15)))


def test_random_reference_matches_sampled_directions():
    rng = np.random.default_rng(3)
    v = rng.normal(size=(200000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    n = np.array([0.0, 0.0, 1.0])
    v *= np.sign(v @ n)[:, None]
    tilt = np.degrees(np.arccos(np.clip(v @ n, -1, 1)))
    got, _ = np.histogram(tilt, bins=[0, 15, 30, 45, 60, 90])
    assert np.allclose(got / got.sum(), laue.random_tilt_fractions(), atol=5e-3)


def test_effective_n():
    assert laue.effective_n([5, 5, 5, 5]) == pytest.approx(4.0)
    assert laue.effective_n([1000, 1, 1]) < 1.1     # one object dominating
    assert laue.effective_n([]) == 0.0


def test_misorientation_respects_symmetry():
    sym = laue.sym_matrices(SG_CUB)
    a = _rot([1, 2, 3], 37.0)
    equivalent = a @ sym[7]
    d = laue.misorientation_matrix(np.stack([a, equivalent]), sym)
    assert d[0, 1] < 1e-3                            # same crystal
    assert d[0, 0] == 0.0
    assert d[0, 1] == pytest.approx(d[1, 0], abs=1e-4)


def test_misorientation_recovers_a_known_angle():
    sym = laue.sym_matrices(SG_HEX)
    a = _rot([0, 1, 0], 12.0)
    d = laue.misorientation_matrix(np.stack([a, a @ _rot([0, 0, 1], 2.5)]), sym)
    assert d[0, 1] == pytest.approx(2.5, abs=1e-3)


# ---------------------------------------------------------------- clustering

def _solutions_from(oms, pos, n_matches=None):
    from midas_plotting.solutions import LaueSolutions
    oms = np.asarray(oms, float).reshape(-1, 3, 3)
    return LaueSolutions(
        image=np.arange(1, len(oms) + 1), grain=np.zeros(len(oms), int),
        n_matches=(np.full(len(oms), 20) if n_matches is None
                   else np.asarray(n_matches)),
        orient_mat=oms, pos=np.asarray(pos, float).reshape(-1, 2))


def test_cluster_groups_one_grain_and_separates_two():
    base = _rot([1, 0, 0], 5.0)
    close = [base @ _rot([0, 0, 1], d) for d in (0.0, 0.2, 0.4)]
    far = [base @ _rot([0, 1, 0], 20.0)]
    pos = [(0, 0), (1, 0), (2, 0), (3, 0)]
    c = laue.cluster(_solutions_from(close + far, pos), 1.0, space_group=SG_HEX)
    assert c.n_clusters == 2
    assert sorted(c.sizes.tolist()) == [1.0, 3.0]


def test_complete_linkage_refuses_a_drifting_chain():
    """Single linkage would merge a chain that spans far more than the tolerance."""
    base = np.eye(3)
    chain = [base @ _rot([0, 0, 1], d) for d in (0.0, 0.9, 1.8, 2.7)]
    pos = [(i, 0) for i in range(4)]
    c = laue.cluster(_solutions_from(chain, pos), 1.0, space_group=SG_HEX)
    assert c.n_clusters > 1                          # 0.0 and 2.7 are 2.7 apart


def test_full_field_object_is_flagged_not_deleted():
    """An orientation present across the whole map is not a grain."""
    everywhere = [np.eye(3)] * 8
    local = [_rot([0, 1, 0], 30.0)] * 2
    pos = [(x, 0.0) for x in range(8)] + [(0.0, 0.0), (1.0, 0.0)]
    c = laue.cluster(_solutions_from(everywhere + local, pos), 1.0,
                     space_group=SG_HEX)
    assert c.full_field.sum() == 1
    assert c.n_clusters == 2 and c.n_grains == 1
    assert c.n_eff_grains <= c.n_clusters
    assert "spanning >half the map" in repr(c)


def test_cluster_refuses_rather_than_hanging():
    n = laue.MAX_CLUSTER + 1
    sol = _solutions_from([np.eye(3)] * n, [(0, 0)] * n)
    with pytest.raises(ValueError, match="UNIFORMLY ACROSS POSITIONS"):
        laue.cluster(sol, 1.0, space_group=SG_HEX)


# ---------------------------------------------------------------- plots

def _textured(n=120, spread=6.0, seed=1):
    """c-axes clustered near the surface plane -- a prismatic-like texture."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(_rot([0, 1, 0], 90.0)
                   @ _rot([1, 0, 0], rng.normal(0, spread))
                   @ _rot([0, 0, 1], rng.uniform(0, 360)))
    return np.stack(out)


def test_texture_strength_normalises_by_its_own_chance_level():
    """A random population must come out at about 1x its chance level."""
    rng = np.random.default_rng(5)
    q = rng.normal(size=(300, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    rand = np.stack([
        np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1),
        np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], -1),
        np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)], -2)
    _, _, ratio = laue.texture_strength(rand, n_null=40, seed=0)
    assert 0.6 < ratio < 1.6, ratio

    _, _, tex = laue.texture_strength(_textured(), n_null=40, seed=0)
    assert tex > 1.8, tex                            # a real texture stands out


def test_tilt_histogram_draws_the_random_reference():
    ax = laue.tilt_histogram(_textured(), reference=True)
    # two bar series: the measurement and the reference
    containers = [c for c in ax.containers]
    assert len(containers) == 2
    ref = [b.get_height() for b in containers[0]]
    assert ref[-1] == pytest.approx(50.0, abs=0.1)   # the 60-90 band
    assert "random" in ax.get_legend().get_texts()[0].get_text()


def test_tilt_histogram_area_weighting_differs_from_grain_weighting():
    om = _textured(n=40)
    w = np.ones(40); w[0] = 500.0                    # one huge grain
    a1 = laue.tilt_histogram(om)
    a2 = laue.tilt_histogram(om, weights=w)
    assert "grains" in a1.get_ylabel() and "area" in a2.get_ylabel()


def test_orientation_map_requires_positions():
    from midas_plotting.solutions import LaueSolutions
    s = LaueSolutions(image=np.array([1]), grain=np.array([0]),
                      n_matches=np.array([20]), orient_mat=np.eye(3)[None])
    with pytest.raises(ValueError, match="no positions"):
        laue.orientation_map(s)


def test_orientation_map_runs():
    om = _textured(n=9)
    pos = [(x, y) for x in range(3) for y in range(3)]
    ax = laue.orientation_map(_solutions_from(om, pos))
    assert ax.images


def test_pole_figure_and_size_distribution_run():
    om = _textured(n=30)
    pos = [(i % 6, i // 6) for i in range(30)]
    ax = laue.pole_figure(om)
    assert ax.collections
    c = laue.cluster(_solutions_from(om, pos), 3.0, space_group=SG_HEX)
    ax2 = laue.grain_size_distribution(c)
    assert ax2 is not None


def test_tolerance_sweep_reports_each_tolerance():
    om = _textured(n=25)
    pos = [(i % 5, i // 5) for i in range(25)]
    ax, rows = laue.tolerance_sweep(_solutions_from(om, pos),
                                    tolerances=(1.0, 3.0, 5.0),
                                    space_group=SG_HEX)
    assert [r["tolerance"] for r in rows] == [1.0, 3.0, 5.0]
    # a looser tolerance can never yield more clusters
    assert rows[0]["clusters"] >= rows[-1]["clusters"]
    assert all("n_eff_grains" in r for r in rows)


def test_spot_overlay_rejects_a_row_instead_of_a_frame():
    """h5[...][0] is the first ROW of a 2-D dataset, not the first frame."""
    from midas_plotting.solutions import LaueSpots
    sp = LaueSpots(image=np.array([1]), grain=np.array([1]),
                   hkl=np.zeros((1, 3)), xy=np.array([[10.0, 20.0]]))
    with pytest.raises(ValueError, match="first ROW"):
        laue.spot_overlay(np.zeros(2048), sp)


def test_spot_overlay_runs():
    from midas_plotting.solutions import LaueSpots
    rng = np.random.default_rng(0)
    img = rng.normal(100, 5, size=(64, 64))
    sp = LaueSpots(image=np.array([1, 1]), grain=np.array([1, 1]),
                   hkl=np.zeros((2, 3)), xy=np.array([[10.0, 20.0], [30, 40]]))
    ax = laue.spot_overlay(img, sp)
    assert ax.collections


def test_occupancy_map_separates_invariant_from_moving():
    """A position firing on every frame is the substrate, not a grain."""
    rng = np.random.default_rng(2)
    frames = []
    for _ in range(20):
        fixed = np.array([[100.0, 100.0], [200.0, 300.0]])
        moving = rng.uniform(400, 1800, size=(8, 2))
        frames.append(np.vstack([fixed, moving]))
    ax, stats = laue.occupancy_map(frames, shape=(2048, 2048), bin_px=6)
    assert stats["n_invariant_bins"] == 2
    assert stats["frac_peaks_invariant"] == pytest.approx(2 / 10, abs=0.02)
    assert ax.images


def test_summary_runs():
    om = _textured(n=24)
    pos = [(i % 6, i // 6) for i in range(24)]
    fig = laue.summary(_solutions_from(om, pos), tolerance=3.0,
                       space_group=SG_HEX)
    assert len(fig.axes) >= 4
