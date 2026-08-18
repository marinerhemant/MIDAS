"""FF/NF grain map -> selection -> TT scan plan.

Two kinds of test here. Most run on a synthetic ``Grains.csv`` built in-process.
One -- :func:`test_real_grains_csv_om_convention` -- runs against a real
``ProcessGrains`` output if one is on this machine, and is the regression that
protects the convention finding the whole module rests on: that the ``O11..O33``
block is a ``midas_stress`` orientation matrix used *without* transposing.
"""
import math
import warnings
import os
from pathlib import Path

import pytest
import torch

from midas_dct_tt.chain import (
    GrainRecord,
    radius_is_suspect,
    read_grains_csv,
    select_tt_candidates,
    tt_scan_plan,
)

# 71.7 keV -- an ordinary HEXM energy. lambda = 12.398 / E.
LAMBDA_A = 12.398 / 71.7
CUBIC = (3.6, 3.6, 3.6, 90.0, 90.0, 90.0)

_HEADER = ("%GrainID\tO11\tO12\tO13\tO21\tO22\tO23\tO31\tO32\tO33\t"
           "X\tY\tZ\ta\tb\tc\talpha\tbeta\tgamma\t"
           "GrainRadius\tConfidence\tRMSErrorStrain\tPhaseNr")


def _row(gid, om, pos, cell=CUBIC, radius=50.0, conf=0.9, rms=1e-4, phase=1):
    vals = ([gid] + list(om) + list(pos) + list(cell) + [radius, conf, rms, phase])
    return "\t".join(f"{v}" for v in vals)


def _write_grains(tmp_path, rows, header=_HEADER, preamble=("%NumGrains 3",
                                                            "%BeamCenter 9.62")):
    p = tmp_path / "Grains.csv"
    p.write_text("\n".join(list(preamble) + [header] + rows) + "\n")
    return p


_IDENT = (1, 0, 0, 0, 1, 0, 0, 0, 1)
# A 90 deg rotation about z, so G_sample differs from G_crystal detectably.
_ROT_Z90 = (0, -1, 0, 1, 0, 0, 0, 0, 1)


@pytest.fixture
def grains_file(tmp_path):
    return _write_grains(tmp_path, [
        _row(7, _IDENT, (0.0, 0.0, 0.0), radius=80.0, conf=0.95),
        _row(11, _ROT_Z90, (100.0, 0.0, 0.0), radius=40.0, conf=0.60),
        _row(13, _IDENT, (10.0, 10.0, 10.0), radius=120.0, conf=0.30, phase=2),
    ])


# --- reading ---------------------------------------------------------------
def test_reads_records_and_metadata(grains_file):
    grains, meta = read_grains_csv(grains_file)
    assert [g.grain_id for g in grains] == [7, 11, 13]
    assert meta["NumGrains"] == "3"
    assert meta["BeamCenter"] == "9.62"
    g = grains[0]
    assert torch.allclose(g.orientation, torch.eye(3, dtype=torch.float64))
    assert g.radius_um == pytest.approx(80.0)
    assert g.confidence == pytest.approx(0.95)
    assert g.phase_nr == 1


def test_orientation_is_row_major(grains_file):
    """``O11 O12 O13`` is the first *row*, not the first column."""
    grains, _ = read_grains_csv(grains_file)
    om = grains[1].orientation
    expected = torch.tensor(_ROT_Z90, dtype=torch.float64).reshape(3, 3)
    assert torch.allclose(om, expected)
    assert om[0, 1] == pytest.approx(-1.0)      # O12
    assert om[1, 0] == pytest.approx(1.0)       # O21


def test_header_keyed_off_O11_not_grainid_literal(tmp_path):
    """ProcessGrains has spelled the ID column both ways; both must load."""
    p = _write_grains(tmp_path, [_row(3, _IDENT, (0, 0, 0))],
                      header=_HEADER.replace("%GrainID", "%ID"))
    grains, _ = read_grains_csv(p)
    assert grains[0].grain_id == 3


def test_missing_orientation_header_raises(tmp_path):
    p = tmp_path / "Grains.csv"
    p.write_text("%NumGrains 1\n1 2 3\n")
    with pytest.raises(ValueError, match="not a ProcessGrains output file"):
        read_grains_csv(p)


def test_missing_required_column_raises(tmp_path):
    p = _write_grains(tmp_path, [_row(1, _IDENT, (0, 0, 0))],
                      header=_HEADER.replace("\tZ\t", "\tZZ\t"))
    with pytest.raises(ValueError, match=r"missing required columns.*'Z'"):
        read_grains_csv(p)


def test_header_but_no_rows_raises(tmp_path):
    p = _write_grains(tmp_path, [])
    with pytest.raises(ValueError, match="no parsable data rows"):
        read_grains_csv(p)


def test_blank_and_comment_rows_skipped(tmp_path):
    p = _write_grains(tmp_path, [_row(1, _IDENT, (0, 0, 0)), "", "% a comment",
                                 _row(2, _IDENT, (0, 0, 0))])
    grains, _ = read_grains_csv(p)
    assert [g.grain_id for g in grains] == [1, 2]


def test_optional_columns_become_nan(tmp_path):
    trimmed = "%ID\t" + "\t".join(
        list(("O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33"))
        + ["X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma"])
    vals = [1] + list(_IDENT) + [0, 0, 0] + list(CUBIC)
    p = _write_grains(tmp_path, ["\t".join(str(v) for v in vals)], header=trimmed)
    grains, _ = read_grains_csv(p)
    assert math.isnan(grains[0].radius_um)
    assert math.isnan(grains[0].confidence)


# --- the physics composition ----------------------------------------------
def test_G_sample_magnitude_is_two_pi_over_d():
    """``|OM @ B @ hkl| = 2*pi/d``, and OM being a rotation cannot change it."""
    g = GrainRecord(grain_id=0, orientation=torch.eye(3, dtype=torch.float64),
                    position_um=torch.zeros(3, dtype=torch.float64),
                    lattice=torch.tensor(CUBIC, dtype=torch.float64))
    for hkl, n in [((1, 0, 0), 1), ((1, 1, 0), 2), ((1, 1, 1), 3), ((2, 0, 0), 4)]:
        d = CUBIC[0] / math.sqrt(n)
        assert float(torch.linalg.vector_norm(g.G_sample(hkl))) == pytest.approx(
            2 * math.pi / d, rel=1e-12)


def test_orientation_rotates_G_into_the_sample_frame():
    """A 90 deg rotation about z must take the (100) G onto +y."""
    g = GrainRecord(grain_id=0,
                    orientation=torch.tensor(_ROT_Z90, dtype=torch.float64).reshape(3, 3),
                    position_um=torch.zeros(3, dtype=torch.float64),
                    lattice=torch.tensor(CUBIC, dtype=torch.float64))
    G = g.G_sample((1, 0, 0))
    assert float(G[1]) == pytest.approx(2 * math.pi / 3.6, rel=1e-12)
    assert abs(float(G[0])) < 1e-12


def test_offcenter_is_full_norm_not_offaxis_radius():
    g = GrainRecord(grain_id=0, orientation=torch.eye(3, dtype=torch.float64),
                    position_um=torch.tensor([3.0, 4.0, 12.0], dtype=torch.float64),
                    lattice=torch.tensor(CUBIC, dtype=torch.float64))
    assert g.offcenter_um() == pytest.approx(13.0)     # not hypot(3,4)=5


# --- selection -------------------------------------------------------------
def test_selection_filters(grains_file):
    grains, _ = read_grains_csv(grains_file)
    assert [g.grain_id for g in select_tt_candidates(grains, min_radius_um=50.0)] \
        == [13, 7]
    assert [g.grain_id for g in select_tt_candidates(grains, max_radius_um=100.0)] \
        == [7, 11]
    assert [g.grain_id for g in select_tt_candidates(grains, min_confidence=0.9)] == [7]
    assert [g.grain_id for g in select_tt_candidates(grains, phase_nr=2)] == [13]
    assert [g.grain_id for g in select_tt_candidates(grains, max_offcenter_um=30.0)] \
        == [13, 7]


def test_selection_sorts_biggest_first_then_most_central(grains_file):
    grains, _ = read_grains_csv(grains_file)
    assert [g.grain_id for g in select_tt_candidates(grains)] == [13, 7, 11]
    assert [g.grain_id for g in select_tt_candidates(grains, top_n=2)] == [13, 7]


def test_selection_refuses_to_filter_on_an_absent_column(tmp_path):
    """A NaN comparison is False, so an unguarded filter would reject everything
    and look exactly like 'no candidates matched'."""
    trimmed = "%ID\t" + "\t".join(
        ["O11", "O12", "O13", "O21", "O22", "O23", "O31", "O32", "O33",
         "X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma"])
    vals = [1] + list(_IDENT) + [0, 0, 0] + list(CUBIC)
    p = _write_grains(tmp_path, ["\t".join(str(v) for v in vals)], header=trimmed)
    grains, _ = read_grains_csv(p)
    with pytest.warns(RuntimeWarning, match="no GrainRadius column"), \
            pytest.raises(ValueError, match="cannot filter on GrainRadius"):
        select_tt_candidates(grains, min_radius_um=10.0)
    with pytest.raises(ValueError, match="cannot filter on Confidence"):
        select_tt_candidates(grains, min_confidence=0.5)


# --- the scan plan ---------------------------------------------------------
def test_scan_plan_is_full_rank_and_physical(grains_file):
    """The default plan is full rank and every reflection can actually diffract.

    Renamed from ``..._zero_leakage``, which asserted leakage == 0 and passed for
    the wrong reason: without a crystal the cone ranking returned the forbidden
    orthogonal {100} triplet, which is symmetry-closed and therefore scored zero
    leakage -- a flawless plan that produces no photons. With absences filtered
    the cone-optimal set is {111}, which is full rank but does leak; zero leakage
    now requires ranking on leakage (see the next test).
    """
    from midas_dfxm.io import fcc_reference_crystal
    grains, _ = read_grains_csv(grains_file)
    plan = tt_scan_plan(grains[0], LAMBDA_A, crystal=fcc_reference_crystal())
    assert plan.grain_id == 7
    assert plan.report.rank == 9
    assert len(plan.alignments) == 3
    assert plan.n_accessible == 14          # {111} + {200} at theta <= 3.0 deg
    for hkl in plan.report.hkls:            # no mixed parity survives
        assert len({int(v) % 2 for v in hkl}) == 1


def test_default_theta_window_reaches_the_zero_leakage_family(grains_file):
    """The default must ADMIT {200}, or ranking on leakage cannot pay off.

    Regression for a real defect. The 2.5 deg default was calibrated against a
    candidate pool that still contained systematic absences. Once those are
    filtered, 2.5 deg admits only {111}, and the symmetry-closed {200} family --
    the zero-leakage set, the entire reason this module ranks on leakage -- sits
    at theta = 2.726 deg, just outside. The recommendation became unreachable at
    the module's own default. This pins the window open.
    """
    from midas_dfxm.io import fcc_reference_crystal
    from midas_dct_tt.planning import accessible_reflections, rank_reflection_sets
    grains, _ = read_grains_csv(grains_file)
    g, xtal = grains[0], fcc_reference_crystal()

    def families_and_best_leakage(max_theta):
        refl = accessible_reflections(g.reciprocal_basis(), LAMBDA_A, hkl_max=2,
                                      orientation=g.orientation,
                                      max_theta_deg=max_theta, crystal=xtal)
        fams = {tuple(sorted(abs(int(v)) for v in h)) for h, _, _ in refl}
        best = rank_reflection_sets(refl, n_reflections=3, top=1,
                                    sort_by="leakage")[0]
        return fams, best.leakage

    narrow_fams, narrow_leak = families_and_best_leakage(2.5)
    assert narrow_fams == {(1, 1, 1)}, "expected {111}-only inside 2.5 deg"
    assert narrow_leak > 0.1, "no zero-leakage set should be reachable at 2.5 deg"

    wide_fams, wide_leak = families_and_best_leakage(3.0)   # the shipped default
    assert (0, 0, 2) in wide_fams, "the default window must admit {200}"
    assert wide_leak == pytest.approx(0.0, abs=1e-9)


def test_scan_plan_alignments_satisfy_bragg_and_are_tt(grains_file):
    """Every returned setting must actually be a TT setting: Bragg satisfied and
    G parallel to the tomographic axis, or the plan is not executable."""
    grains, _ = read_grains_csv(grains_file)
    plan = tt_scan_plan(grains[0], LAMBDA_A)
    for hkl, al in plan.alignments:
        assert abs(float(al.bragg_residual())) < 1e-9
        axis = al.rotation_axis / torch.linalg.vector_norm(al.rotation_axis)
        gdir = al.G_lab / torch.linalg.vector_norm(al.G_lab)
        assert abs(abs(float(torch.sum(axis * gdir))) - 1.0) < 1e-12
        # the defining TT property: G is invariant over the whole psi sweep
        for psi in (0.0, 37.0, 180.0, 359.0):
            moved = al.psi_rotation(psi) @ al.G_lab
            assert torch.allclose(moved, al.G_lab, atol=1e-9)


def test_scan_plan_axis_beam_angle_is_ninety_minus_theta(grains_file):
    grains, _ = read_grains_csv(grains_file)
    plan = tt_scan_plan(grains[0], LAMBDA_A)
    for _, al in plan.alignments:
        assert float(al.axis_beam_angle_deg()) == pytest.approx(
            90.0 - float(al.theta_deg), abs=1e-8)


def test_scan_plan_reports_alternatives(grains_file):
    grains, _ = read_grains_csv(grains_file)
    plan = tt_scan_plan(grains[0], LAMBDA_A, n_alternatives=4)
    assert len(plan.alternatives) == 4
    assert plan.report.leakage <= plan.alternatives[0].leakage + 1e-12
    assert "grain 7" in plan.summary()


def test_scan_plan_survives_a_rotated_grain(grains_file):
    """The plan must be equally good for a grain in a general orientation --
    leakage is a property of the reflection set, not of the grain's pose."""
    grains, _ = read_grains_csv(grains_file)
    rotated = tt_scan_plan(grains[1], LAMBDA_A)
    assert rotated.report.rank == 9
    assert rotated.report.leakage == pytest.approx(0.0, abs=1e-9)


def test_scan_plan_raises_when_too_few_reflections(grains_file):
    grains, _ = read_grains_csv(grains_file)
    with pytest.raises(ValueError, match="Widen max_theta_deg"):
        tt_scan_plan(grains[0], LAMBDA_A, max_theta_deg=0.5)


def test_wide_theta_filter_trips_the_combinatorial_guard(grains_file):
    """Documents the guard: 5 deg admits all 124 reflections with hkl_max=2.

    Geometry-only, i.e. no crystal -- with the structure factor applied the
    surviving pool is small enough that the candidate guard is never reached.
    """
    grains, _ = read_grains_csv(grains_file)
    with pytest.raises(ValueError, match="exceeds max_candidates"):
        tt_scan_plan(grains[0], LAMBDA_A, max_theta_deg=5.0)


# --- real data -------------------------------------------------------------
# Real ProcessGrains output, kept OUT of the tree: it is a collaborator's data,
# and a hardcoded absolute path is both a disclosure and unrunnable for anyone
# else. Point MIDAS_DCT_TT_REAL_GRAINS at a Grains.csv to enable these two
# tests; without it they skip, which is the state on every machine but one.
# The dataset behind this is recorded in the private BEAMTIME_KEY.md.
_REAL = Path(os.environ.get("MIDAS_DCT_TT_REAL_GRAINS", "/nonexistent/Grains.csv"))


@pytest.mark.skipif(not _REAL.exists(), reason="real Grains.csv not on this machine")
def test_real_grains_csv_om_convention():
    """The load-bearing convention, checked on real ProcessGrains output.

    ``O11..O33`` must equal ``euler_to_orient_mat(Eul, radians)`` -- NOT its
    transpose. If this flips, every ``G_sample`` in the package is wrong and TT
    alignments will be solved for the wrong reciprocal vector.
    """
    from midas_stress.orientation import euler_to_orient_mat

    grains, meta = read_grains_csv(_REAL)
    assert len(grains) > 100
    assert "NumGrains" in meta

    lines = _REAL.read_text().splitlines()
    hdr = next(ln[1:].split() for ln in lines if ln.startswith("%") and "O11" in ln)
    eul_idx = [hdr.index(c) for c in ("Eul0", "Eul1", "Eul2")]
    rows = [ln.split() for ln in lines if ln.strip() and not ln.startswith("%")]

    worst_direct = worst_transpose = 0.0
    for g, toks in list(zip(grains, rows))[:200]:
        eul = torch.tensor([float(toks[i]) for i in eul_idx], dtype=torch.float64)
        R = torch.as_tensor(euler_to_orient_mat(eul), dtype=torch.float64).reshape(3, 3)
        worst_direct = max(worst_direct, float(torch.max(torch.abs(R - g.orientation))))
        worst_transpose = max(worst_transpose,
                              float(torch.max(torch.abs(R.T - g.orientation))))

    assert worst_direct < 1e-5, f"OM is not euler_to_orient_mat(Eul): {worst_direct:.3e}"
    assert worst_transpose > 1e-2, "OM and its transpose are indistinguishable here"


@pytest.mark.skipif(not _REAL.exists(), reason="real Grains.csv not on this machine")
def test_real_grains_are_orthonormal_and_plannable():
    grains, _ = read_grains_csv(_REAL)
    for g in grains[:50]:
        om = g.orientation
        assert torch.allclose(om @ om.T, torch.eye(3, dtype=torch.float64), atol=1e-5)
        assert float(torch.linalg.det(om)) == pytest.approx(1.0, abs=1e-5)

    best = select_tt_candidates(grains, min_confidence=0.5, top_n=1)
    assert best, "no grain in the real file passed a 0.5 confidence cut"
    plan = tt_scan_plan(best[0], LAMBDA_A)
    assert plan.report.rank == 9
    for _, al in plan.alignments:
        assert abs(float(al.bragg_residual())) < 1e-8


# --- row width: both real-world deviations ---------------------------------
def test_short_rows_carrying_only_the_required_block_load(tmp_path):
    """mpe_dec24/Grains.csv declares 47 columns and writes 19.

    Those 19 are exactly GrainID + OM + XYZ + cell, i.e. everything this module
    needs. Requiring the full header width discarded the entire file.
    """
    vals = [5] + list(_IDENT) + [1.0, 2.0, 3.0] + list(CUBIC)
    assert len(vals) == 19
    p = _write_grains(tmp_path, ["\t".join(str(v) for v in vals)])   # 23-col header
    grains, _ = read_grains_csv(p)
    assert len(grains) == 1 and grains[0].grain_id == 5
    assert math.isnan(grains[0].radius_um)          # absent -> NaN, not garbage
    assert grains[0].offcenter_um() == pytest.approx(math.sqrt(14.0))


def test_extra_unnamed_trailing_column_is_harmless(tmp_path):
    """bt_1id_nov25/Grains.csv writes 48 fields against a 47-name header."""
    p = _write_grains(tmp_path, [_row(9, _IDENT, (0, 0, 0)) + "\t999.0"])
    grains, _ = read_grains_csv(p)
    assert grains[0].grain_id == 9
    assert grains[0].radius_um == pytest.approx(50.0)


def test_rows_too_short_for_the_required_block_are_skipped(tmp_path):
    p = _write_grains(tmp_path, [_row(1, _IDENT, (0, 0, 0)),
                                 "\t".join(str(v) for v in [2] + list(_IDENT))])
    grains, _ = read_grains_csv(p)
    assert [g.grain_id for g in grains] == [1]


# --- GrainRadius trustworthiness -------------------------------------------
def test_radius_is_suspect_flags_submicron(tmp_path):
    p = _write_grains(tmp_path, [_row(i, _IDENT, (0, 0, 0), radius=0.83)
                                 for i in range(5)])
    grains, _ = read_grains_csv(p)
    suspect, reason = radius_is_suspect(grains)
    assert suspect and "0.83" in reason and "ID-space bug" in reason


def test_radius_is_suspect_accepts_plausible_sizes(tmp_path):
    p = _write_grains(tmp_path, [_row(i, _IDENT, (0, 0, 0), radius=80.0 + i)
                                 for i in range(5)])
    grains, _ = read_grains_csv(p)
    suspect, _ = radius_is_suspect(grains)
    assert not suspect


def test_radius_is_suspect_when_column_absent(tmp_path):
    vals = [1] + list(_IDENT) + [0, 0, 0] + list(CUBIC)
    p = _write_grains(tmp_path, ["\t".join(str(v) for v in vals)])
    grains, _ = read_grains_csv(p)
    assert radius_is_suspect(grains)[0]


def test_radius_filter_warns_on_suspect_column(tmp_path):
    p = _write_grains(tmp_path, [_row(i, _IDENT, (0, 0, 0), radius=0.83)
                                 for i in range(5)])
    grains, _ = read_grains_csv(p)
    with pytest.warns(RuntimeWarning, match="ID-space bug"):
        select_tt_candidates(grains, min_radius_um=0.1)


def test_no_warning_when_not_filtering_on_radius(tmp_path):
    p = _write_grains(tmp_path, [_row(i, _IDENT, (0, 0, 0), radius=0.83)
                                 for i in range(5)])
    grains, _ = read_grains_csv(p)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        select_tt_candidates(grains, min_confidence=0.5)


@pytest.mark.skipif(not Path("/Users/hsharma/Desktop/analysis/mpe_dec24/Grains.csv").exists(),
                    reason="mpe_dec24 not on this machine")
def test_real_truncated_file_loads_all_its_grains():
    grains, meta = read_grains_csv("/Users/hsharma/Desktop/analysis/mpe_dec24/Grains.csv")
    assert len(grains) == int(meta["NumGrains"])
    assert math.isnan(grains[0].radius_um)
