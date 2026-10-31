"""The strain-cap and pointing gates describe the iterate the run RETURNED, not the last one.

``autocalibrate`` adopts the best iterate by mean strain and discards a worse last one ("adopting best iterate
(380.5 ue) over the last (1755.7 ue)").  ``strain_cap_check`` and ``pointing_precision_check`` read
``history[-1]``, i.e. the rejected iterate: on a real GE frame the gate failed on 1.88 px that belonged to a geometry
the run had thrown away, while the returned one sat at 0.49 px.  Synthetic history, no fit needed.
"""
from __future__ import annotations

from types import SimpleNamespace

from midas_calibrate_v2.pipelines.diagnostics import (
    adopted_iterate, pointing_precision_check, strain_cap_check,
)


def _rec(i, strain, px):
    return SimpleNamespace(iteration=i, mean_strain_uE=strain, mean_abs_px=px)


# the GE pattern: iteration 0 is the best, later iterates drift away
GE_LIKE = [_rec(0, 380.5, 0.489), _rec(1, 700.0, 0.9), _rec(2, 1200.0, 1.4), _rec(3, 1755.7, 1.878)]


def test_adopted_is_the_best_iterate_not_the_last():
    assert adopted_iterate(GE_LIKE).iteration == 0


def test_last_is_kept_when_it_is_itself_the_best():
    h = [_rec(0, 50.0, 0.3), _rec(1, 30.0, 0.2), _rec(2, 30.0, 0.2)]
    assert adopted_iterate(h).iteration == 2          # ties: autocalibrate leaves the last in place


def test_pointing_gate_scores_the_returned_geometry():
    res = pointing_precision_check(GE_LIKE, fail_px=1.0, warn_px=0.5, use_adopted=True)
    assert res.severity != "fail"
    assert res.metrics["abs_px"] == 0.489


def test_strain_cap_gate_scores_the_returned_geometry():
    res = strain_cap_check(GE_LIKE, threshold_uE=1000.0, use_adopted=True)
    assert res.severity != "fail", res.message


def test_nan_iterates_do_not_win():
    h = [_rec(0, float("nan"), float("nan")), _rec(1, 40.0, 0.2)]
    assert adopted_iterate(h).iteration == 1
    assert adopted_iterate([]) is None


def test_default_keeps_the_last_iterate_for_pv_histories():
    """autocalibrate_pv (single_pv.py) RETURNS its last iterate, so the default must not move to an earlier one."""
    res = pointing_precision_check(GE_LIKE, fail_px=1.0, warn_px=0.5)
    assert res.severity == "fail" and res.metrics["abs_px"] == 1.878
    assert strain_cap_check(GE_LIKE, threshold_uE=1000.0).severity == "fail"


def test_run_all_gates_routes_use_adopted():
    import torch
    from midas_calibrate_v2.pipelines.diagnostics import run_all_gates
    v1 = SimpleNamespace(Lsd=1.0e6, BC_y=100.0, BC_z=100.0)
    u = {"Lsd": torch.tensor(1.0e6), "BC_y": torch.tensor(100.0), "BC_z": torch.tensor(100.0),
         "pxY": torch.tensor(200.0), "pxZ": torch.tensor(200.0)}
    last = {r.name: r for r in run_all_gates(v1_init=v1, unpacked=u, history=GE_LIKE)}["strain_cap"]
    best = {r.name: r for r in run_all_gates(v1_init=v1, unpacked=u, history=GE_LIKE, use_adopted=True)}["strain_cap"]
    assert last.severity == "fail" and best.severity != "fail"
