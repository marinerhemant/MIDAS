"""Unit tests for pipeline stage-order computation.

These tests pin the FF / PF stage lists so any reorder is caught
immediately. Parallel-stream developers may need to know the order
their stage lands in.
"""

from __future__ import annotations

import pytest

from midas_pipeline import all_stage_names, stage_order_for


_EXPECTED_FF = [
    "zip_convert", "hkl", "peakfit", "merge_overlaps", "calc_radius",
    "transforms", "cross_det_merge", "global_powder",
    "binning", "indexing", "refinement",
    "process_grains", "grain_geometry", "consolidation",
    # P8 V-map orchestration (clean no-op when vmap.run=False)
    "calc_radius_v", "refine_vmap",
]

_EXPECTED_PF = [
    "zip_convert", "hkl", "peakfit", "merge_overlaps", "calc_radius",
    "transforms", "cross_det_merge", "global_powder",
    "merge_scans", "seeding",
    "binning", "indexing", "refinement",
    "find_grains", "voxel_cleanup", "sinogen", "reconstruct",
    "fuse", "potts", "em_refine",
    "consolidation",
    # P8 V-map orchestration (clean no-op when vmap.run=False)
    "calc_radius_v", "refine_vmap",
]


def test_ff_stage_order():
    names = [n for n, _ in stage_order_for("ff")]
    assert names == _EXPECTED_FF


def test_pf_stage_order():
    names = [n for n, _ in stage_order_for("pf")]
    assert names == _EXPECTED_PF


def test_ff_excludes_pf_only_stages():
    """No PF-only stage should appear in FF mode."""
    ff_names = {n for n, _ in stage_order_for("ff")}
    pf_only = {"merge_scans", "seeding", "find_grains", "voxel_cleanup",
               "sinogen", "reconstruct", "fuse", "potts", "em_refine"}
    assert ff_names.isdisjoint(pf_only)


def test_pf_excludes_ff_only_stages():
    """process_grains is FF-only; PF uses consolidation directly."""
    pf_names = {n for n, _ in stage_order_for("pf")}
    assert "process_grains" not in pf_names
    assert "consolidation" in pf_names


def test_all_stage_names_is_union():
    """all_stage_names() should be the union of FF and PF names, no duplicates."""
    ff_names = [n for n, _ in stage_order_for("ff")]
    pf_names = [n for n, _ in stage_order_for("pf")]
    union = set(ff_names) | set(pf_names)
    assert set(all_stage_names()) == union
    # No duplicates in the master list
    assert len(all_stage_names()) == len(set(all_stage_names()))


def test_unknown_scan_mode_raises():
    with pytest.raises(ValueError):
        stage_order_for("invalid")  # type: ignore[arg-type]


def test_indexing_runs_after_binning_in_both_modes():
    """Sanity: binning must precede indexing in both modes (the kernel
    contract is unchanged by scan-mode)."""
    for mode in ("ff", "pf"):
        names = [n for n, _ in stage_order_for(mode)]
        assert names.index("binning") < names.index("indexing")
        assert names.index("indexing") < names.index("refinement")


def test_pf_find_grains_after_refinement():
    """Sanity: find_grains operates on refined per-voxel candidates."""
    names = [n for n, _ in stage_order_for("pf")]
    assert names.index("refinement") < names.index("find_grains")
    assert names.index("find_grains") < names.index("sinogen")
    assert names.index("sinogen") < names.index("reconstruct")
