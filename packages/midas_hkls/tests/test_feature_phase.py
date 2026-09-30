"""Tests for midas_hkls.feature_phase (synthetic structures and features only)."""
import numpy as np

from midas_hkls.crystal import Atom, Crystal
from midas_hkls.feature_phase import (
    allowed_d_lines,
    cell_scan,
    d_clusters,
    family_count,
    feature_phase_test,
    match_count,
    pixel_weighted_sampler,
)
from midas_hkls.lattice import Lattice
from midas_hkls.space_group import SpaceGroup

LAM = 0.124
D_MIN, D_MAX = 0.8, 3.0


def _rocksalt(a, A="Zr", B="C"):
    return Crystal(Lattice(a, a, a, 90, 90, 90), SpaceGroup.from_number(225),
                   [Atom(A, (0, 0, 0)), Atom(B, (0.5, 0.5, 0.5))])


def _sampler():
    tth = np.random.default_rng(0).uniform(2.4, 9.0, 200000)     # stand-in detector coverage
    return pixel_weighted_sampler(tth, LAM)


def _features(true_lines, n_true, n_noise, scale, sig, seed):
    rng = np.random.default_rng(seed)
    d_true = rng.choice(true_lines, n_true) * scale * (1 + rng.normal(0, sig, n_true))
    return np.r_[d_true, _sampler()(n_noise, rng)]


def test_basis_absences_are_dropped():
    # rock salt with identical atoms on both sites = simple cubic with a/2: odd
    # reflections vanish by the basis, not by the space-group conditions
    same = allowed_d_lines(_rocksalt(4.4, "Zr", "Zr"), D_MIN, D_MAX)
    diff = allowed_d_lines(_rocksalt(4.4, "Zr", "C"), D_MIN, D_MAX)
    assert len(same) < len(diff)
    assert not np.any(np.isclose(same, 4.4 / np.sqrt(3), atol=1e-4))


def test_true_population_passes_and_controls_fail():
    sig = 0.0015
    lines = allowed_d_lines(_rocksalt(4.40), D_MIN, D_MAX)
    d = _features(lines, 15, 25, 1.010, sig, seed=1)
    cands = {"cand": lines}
    ctrls = {"ctrl_small": allowed_d_lines(_rocksalt(4.10), D_MIN, D_MAX),
             "ctrl_large": allowed_d_lines(_rocksalt(4.75), D_MIN, D_MAX)}
    res = feature_phase_test(d, 2 * sig, cands, _sampler(), controls=ctrls, n_null=300)
    row = {r.name: r for r in res["rows"]}
    assert res["valid"]
    assert row["cand"].passed and row["cand"].L >= 3
    assert not row["ctrl_small"].passed and not row["ctrl_large"].passed


def test_noise_only_does_not_pass():
    sig = 0.0015
    lines = allowed_d_lines(_rocksalt(4.40), D_MIN, D_MAX)
    d = _sampler()(40, np.random.default_rng(7))
    res = feature_phase_test(d, 2 * sig, {"cand": lines}, _sampler(), n_null=300)
    assert not res["rows"][0].passed


def test_dense_candidate_reports_high_chance_coverage():
    sparse = allowed_d_lines(_rocksalt(4.40), D_MIN, D_MAX)
    dense = allowed_d_lines(Crystal(Lattice(10.6, 10.6, 10.6, 90, 90, 90),
                                    SpaceGroup.from_number(221), []), D_MIN, D_MAX)
    d = _sampler()(30, np.random.default_rng(3))
    res = feature_phase_test(d, 0.004, {"sparse": sparse, "dense": dense}, _sampler(), n_null=100)
    row = {r.name: r for r in res["rows"]}
    assert row["dense"].chance_coverage > 3 * row["sparse"].chance_coverage


def test_scale_edge_flag():
    lines = allowed_d_lines(_rocksalt(4.40), D_MIN, D_MAX)
    d = lines[:6] * 1.027                          # true scale just past the window [1.0, 1.025]
    M, L, s, edge = match_count(d, 0.003, lines, np.arange(1.0, 1.0251, 0.0005))
    assert M == 6 and edge and s > 1.02
    M2, L2, s2, edge2 = match_count(lines[:6] * 1.010, 0.003, lines, np.arange(1.0, 1.0251, 0.0005))
    assert M2 == 6 and not edge2


def test_cell_scan_finds_the_cell_with_look_elsewhere_null():
    sig = 0.0015
    ref = allowed_d_lines(_rocksalt(4.30), D_MIN, D_MAX)
    d = _features(allowed_d_lines(_rocksalt(4.43), D_MIN, D_MAX), 15, 20, 1.0, sig, seed=5)
    out = cell_scan(d, 2 * sig, ref, 4.30, np.arange(4.0, 4.8, 0.002), _sampler(), n_null=100)
    assert abs(out["a_best"] - 4.43) < 0.01
    assert out["p_look_elsewhere"] < 0.05
    assert out["M_best"] > out["null_max_p99"]


def test_many_spots_on_few_rings_do_not_pass():
    """30 spots on only two d-values that lie on the candidate's lines: the spot count is far
    beyond chance, but two distinct d-values on a cell's lines are not evidence for the cell."""
    sig = 0.0015
    lines = allowed_d_lines(_rocksalt(4.40), D_MIN, D_MAX)
    rng = np.random.default_rng(11)
    d = np.r_[lines[2] * 1.01 * (1 + rng.normal(0, 0.0003, 15)), lines[4] * 1.01 * (1 + rng.normal(0, 0.0003, 15))]
    reps, lab = d_clusters(d, 2 * sig)
    assert len(reps) == 2 and len(np.unique(lab)) == 2
    res = feature_phase_test(d, 2 * sig, {"cand": lines}, _sampler(), n_null=300)
    r = res["rows"][0]
    assert r.p < 0.01 and r.M == 30                 # the spot count alone would call it
    assert r.K == 2 and not r.passed                # the family-level test does not


def test_true_population_passes_at_family_level():
    sig = 0.0015
    lines = allowed_d_lines(_rocksalt(4.40), D_MIN, D_MAX)
    d = _features(lines, 15, 25, 1.010, sig, seed=1)
    r = feature_phase_test(d, 2 * sig, {"cand": lines}, _sampler(), n_null=300)["rows"][0]
    assert r.F >= 4 and r.p_family < 0.01 and r.passed


def test_two_clusters_either_side_of_one_line_count_once():
    """Two d-values 0.4 % apart (more than the merge tolerance, so two clusters) that both sit within
    tol of ONE line are one line, not two."""
    lines = allowed_d_lines(_rocksalt(4.40), D_MIN, D_MAX)
    L0 = lines[3]
    d = np.array([L0 * 0.998, L0 * 1.002])
    reps, _ = d_clusters(d, 0.003)
    assert len(reps) == 2
    F, s = family_count(reps, 0.003, lines, np.array([1.0]))
    assert F == 1
