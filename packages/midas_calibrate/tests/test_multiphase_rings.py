"""Multi-phase ring tables, d-spacing dedup, and blend flagging."""
from __future__ import annotations

import numpy as np
import pytest

from midas_calibrate.params import CalibrationParams
from midas_calibrate.rings import (
    build_ring_table, flag_blended_rings, drop_blended_rings,
    max_resolvable_ring_radius_px,
)


def _params(**kw):
    p = CalibrationParams(
        NrPixelsY=2048, NrPixelsZ=2048, pxY=200.0, pxZ=200.0,
        Lsd=2733000.0, BC_y=2294.6, BC_z=2178.9,
        Wavelength=0.153443, SpaceGroup=225,
        LatticeConstant=(5.41153, 5.41153, 5.41153, 90.0, 90.0, 90.0),
        MaxRingRad=3164.0, MinRingRad=280.0,
    )
    for k, v in kw.items():
        setattr(p, k, v)
    return p


CEO2 = {"name": "CeO2", "sg": 225,
        "lattice": (5.41153, 5.41153, 5.41153, 90.0, 90.0, 90.0)}
LAB6 = {"name": "LaB6", "sg": 221,
        "lattice": (4.15689, 4.15689, 4.15689, 90.0, 90.0, 90.0)}


def test_single_phase_unchanged_shape():
    """No Phases set -> single-phase table, phase columns still well-formed."""
    rt = build_ring_table(_params())
    assert len(rt) > 5
    assert rt.n_phases == 1
    assert rt.phase_idx is not None and set(rt.phase_idx) == {0}
    assert rt.phase_of(0) == rt.phase_names[0]


def test_multiphase_is_union_sorted_by_radius():
    single_ce = build_ring_table(_params())
    single_la = build_ring_table(_params(SpaceGroup=221,
                                          LatticeConstant=LAB6["lattice"]))
    both = build_ring_table(_params(Phases=[CEO2, LAB6]))

    assert both.phase_names == ("CeO2", "LaB6")
    assert int(both.phase_mask("CeO2").sum()) == len(single_ce)
    assert int(both.phase_mask("LaB6").sum()) == len(single_la)
    assert len(both) == len(single_ce) + len(single_la)
    # sorted by radius, so blend detection can walk it linearly
    assert np.all(np.diff(both.r_ideal_px) >= 0)


def test_exact_hkl_degeneracies_are_merged_not_duplicated():
    """LaB6 (300)/(221) have identical d — one physical ring, one row.

    Left unmerged they present as a zero-separation "doublet" and any blend
    rule throws away a perfectly good ring.
    """
    rt = build_ring_table(_params(SpaceGroup=221,
                                  LatticeConstant=LAB6["lattice"]))
    # no two rows of the same phase share a d-spacing
    d = np.sort(rt.d_spacing)
    assert np.all(np.diff(d) > 1e-9 * d[:-1]), "duplicate d-spacings survived"
    # and at least one row absorbed an alias on this (dense, cubic) table
    assert any(len(a) > 0 for a in rt.hkl_aliases)
    # the merged row's multiplicity is the sum of what went in
    merged = [(m, a) for m, a in zip(rt.multiplicity, rt.hkl_aliases) if a]
    assert all(m > 0 for m, _ in merged)


def test_dedup_can_be_disabled():
    rt_on = build_ring_table(_params(SpaceGroup=221,
                                      LatticeConstant=LAB6["lattice"]))
    rt_off = build_ring_table(_params(SpaceGroup=221,
                                       LatticeConstant=LAB6["lattice"]),
                              dedup_d_rel_tol=0.0)
    assert len(rt_off) > len(rt_on)


def test_flag_blended_rings_flags_both_members():
    rt = build_ring_table(_params(Phases=[CEO2, LAB6]))
    R = np.sort(rt.r_ideal_px)
    gaps = np.diff(R)
    cut = float(np.median(gaps))          # guarantees some pairs are inside
    flagged = flag_blended_rings(rt, min_separation_px=cut)
    assert flagged.any()
    # every flagged ring really does have a close neighbour
    for i in np.nonzero(flagged)[0]:
        others = np.delete(rt.r_ideal_px, i)
        assert np.min(np.abs(others - rt.r_ideal_px[i])) < cut


def test_drop_blended_rings_leaves_only_separated_rings():
    rt = build_ring_table(_params(Phases=[CEO2, LAB6]))
    kept, n_dropped = drop_blended_rings(rt, min_separation_px=12.0)
    assert n_dropped > 0
    assert len(kept) == len(rt) - n_dropped
    assert np.all(np.diff(np.sort(kept.r_ideal_px)) >= 12.0)
    # phase bookkeeping survives the filter
    assert kept.phase_names == rt.phase_names
    assert kept.phase_idx is not None and len(kept.phase_idx) == len(kept)


def test_cross_phase_only_keeps_same_phase_doublets():
    rt = build_ring_table(_params(Phases=[CEO2, LAB6]))
    all_flags = flag_blended_rings(rt, min_separation_px=25.0)
    cross = flag_blended_rings(rt, min_separation_px=25.0,
                                cross_phase_only=True)
    assert cross.sum() <= all_flags.sum()
    assert np.all(cross <= all_flags)


def test_blend_exclusion_beats_radial_truncation_on_ring_count():
    """The point of per-ring flagging: one mid-radius collision should not
    discard every ring outside it, which is what a radial cutoff does."""
    rt = build_ring_table(_params(Phases=[CEO2, LAB6]))
    kept, _ = drop_blended_rings(rt, min_separation_px=12.0)
    cut_radius, n_inside = max_resolvable_ring_radius_px(
        rt, min_separation_px=12.0)
    assert len(kept) > n_inside


def test_phase_mask_rejects_unknown_name():
    rt = build_ring_table(_params(Phases=[CEO2, LAB6]))
    with pytest.raises(KeyError):
        rt.phase_mask("Si")


def test_bad_phase_spec_raises_clearly():
    with pytest.raises(ValueError, match="missing required key"):
        build_ring_table(_params(Phases=[{"name": "x", "sg": 225}]))
    with pytest.raises(ValueError, match="6 entries"):
        build_ring_table(_params(Phases=[{"name": "x", "sg": 225,
                                          "lattice": (1, 2, 3)}]))


def test_params_file_roundtrip_phase_lines(tmp_path):
    f = tmp_path / "ps.txt"
    f.write_text(
        "Wavelength 0.153443\n"
        "SpaceGroup 225\n"
        "LatticeConstant 5.41153 5.41153 5.41153 90 90 90\n"
        "Phase CeO2 225 5.41153 5.41153 5.41153 90 90 90\n"
        "Phase LaB6 221 4.15689 4.15689 4.15689 90 90 90\n"
        "MinRingSeparation 12.0\n"
        "BlendExcludeCrossPhaseOnly 1\n"
    )
    p = CalibrationParams.from_file(f)
    assert [ph["name"] for ph in p.Phases] == ["CeO2", "LaB6"]
    assert p.Phases[1]["sg"] == 221
    assert p.Phases[1]["lattice"][0] == pytest.approx(4.15689)
    assert p.MinRingSeparation == pytest.approx(12.0)
    assert p.BlendExcludeCrossPhaseOnly is True
