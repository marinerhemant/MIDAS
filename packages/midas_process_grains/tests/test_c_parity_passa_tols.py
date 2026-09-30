"""c_parity Pass A tolerances are configurable, and the C defaults are unchanged.

Pass A (orientation + centroid dedup, ProcessGrains.c:836-874) used hard-coded
0.1 deg / 5 um. On 20-ID data grain centroids scatter by tens of um, so one grain
found twice survived: bt_20id_jul26b nf_sampleF layer 6 kept 246 pairs < 0.1 deg apart
(spot Jaccard median 0.61) at 7-40 um separation. The keys are opt-in
(``CParityPassAMisoriTol`` deg, ``CParityPassAPosTol`` um) and deliberately not
``PassAMisoriTol``, which means the spot-overlap merge in the other modes.

Pinned: resolution order; an unset file gives exactly the C values (so the
default output is untouched); the CLI forwards the flags and forwards ``None``
when absent; the pipeline propagates both keys; Pass A itself keeps the nf_sampleF
case at 5 um and merges it at a larger position tolerance.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_process_grains.compute.c_parity_run import (
    C_PASSA_MISORI_DEG, C_PASSA_POS_UM, resolve_passa_tols,
)
from midas_process_grains.params import ProcessGrainsParams


def _params(**raw):
    p = ProcessGrainsParams()
    for k, v in raw.items():
        p.raw[k] = [str(v)]
    return p


def test_unset_file_gives_the_c_values():
    m, d, src = resolve_passa_tols(None, None, _params())
    assert (m, d) == (0.1, 5.0) == (C_PASSA_MISORI_DEG, C_PASSA_POS_UM)
    assert "C default" in src


def test_file_values_are_used():
    m, d, src = resolve_passa_tols(None, None, _params(CParityPassAMisoriTol=0.08,
                                                       CParityPassAPosTol=30))
    assert (m, d) == (pytest.approx(0.08), pytest.approx(30.0))
    assert "parameter file" in src


def test_explicit_arguments_win():
    m, d, _ = resolve_passa_tols(0.05, 12.0, _params(CParityPassAMisoriTol=0.08,
                                                     CParityPassAPosTol=30))
    assert (m, d) == (pytest.approx(0.05), pytest.approx(12.0))


def test_other_modes_passa_key_does_not_leak_into_c_parity():
    """PassAMisoriTol (spot-overlap merge, other modes) must not move c_parity."""
    m, d, _ = resolve_passa_tols(None, None, _params(PassAMisoriTol=1.0))
    assert (m, d) == (0.1, 5.0)


@pytest.mark.parametrize("argv_extra,expected", [
    (["--passa-misori-tol", "0.08", "--passa-pos-tol", "30"], (0.08, 30.0)),
    ([], (None, None)),
])
def test_cli_forwards_the_flags(tmp_path, monkeypatch, argv_extra, expected):
    import midas_process_grains.compute.c_parity_run as cpr
    from midas_process_grains import cli

    seen = {}
    monkeypatch.setattr(cpr, "run_c_parity_pipeline_from_disk", lambda **kw: seen.update(kw))
    ps = tmp_path / "paramstest_pg.txt"
    ps.write_text("MinNrSpots 3\n")
    assert cli.main([str(ps), "1", "--mode", "c_parity", "--device", "cpu", *argv_extra]) == 0
    got = (seen.get("misori_tol_passa_deg"), seen.get("pos_tol_passa_um"))
    if expected == (None, None):
        assert got == (None, None)
    else:
        assert got == (pytest.approx(expected[0]), pytest.approx(expected[1]))


def test_pipeline_propagates_both_keys():
    mp = pytest.importorskip("midas_pipeline.stages._comp_params")
    assert "CParityPassAMisoriTol" in mp._PG_SELECTION_KEYS
    assert "CParityPassAPosTol" in mp._PG_SELECTION_KEYS


def _om(axis, ang):
    from midas_process_grains.compute.c_parity import pass_a_position_dedup  # noqa: F401
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def test_pass_a_nf_sampleF_case():
    """0.05 deg apart, 19 um apart: kept at C's 5 um, merged at 30 um."""
    from test_c_parity import _make_opf
    from midas_process_grains.compute.c_parity import pass_a_position_dedup
    om = np.stack([np.eye(3), _om([0, 0, 1], math.radians(0.05))])
    pos = np.array([[0, 0, 0], [19, 0, 0]], dtype=np.float64)
    opf, _ = _make_opf(om, pos)
    gp = np.array([0, 1], dtype=np.int64)
    kw = dict(grain_positions=gp, opf=opf, space_group=225, misori_tol_rad=math.radians(0.1))
    assert pass_a_position_dedup(pos_tol_um=5.0, **kw).tolist() == [False, False]
    assert pass_a_position_dedup(pos_tol_um=30.0, **kw).tolist() == [False, True]
