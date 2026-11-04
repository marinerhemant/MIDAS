"""reduce_twin90_scan: every branch of the composition, with the heavy stages stubbed (the end-to-end equivalence with validation/run_pm2.py is
validation/check_workflow_equiv.py: bit-identical observation tables on P11 seeds 3, 8, 21)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import midas_sx.workflow as W
from midas_sx.twin_orient import RZ90

SETUP = W.ScanSetup(0.4133, 100.0, 0.3, (255.5, 255.5), -40.0, 0.25)
CRYSTAL = SimpleNamespace(lattice="LAT", space_group="SG")
U0 = np.eye(3)


def _orient(converged=True, U=None, reason="ok"):
    return SimpleNamespace(converged=converged, U=np.eye(3) if U is None else U, geom=SETUP.geometry((64, 64), 4), reason=reason, n_used=30, n_spots=40,
                           rms_px=0.1, rms_frame=0.1)


def _targets():
    return pd.DataFrame(dict(frame=[1.0, 2.0, 3.0], row=[10.0, 20.0, 30.0], col=[11.0, 21.0, 31.0], extra_px=0.0, extra_frames=0.0,
                             kind=["blend", "c1", "c2"], f1=[1, 1, 0], f2=[1, 0, 1], p0=[1, 2, 3], p1=[0, 0, 0], p2=[4, 5, 6], sep_px=[0.5, 5.0, 5.0],
                             h=[1, 2, 3], k=[0, 0, 0], l=[4, 5, 6]))


@pytest.fixture
def stubs(monkeypatch):
    calls = SimpleNamespace(d_min=None, reduce_U=[], twin_U=None)
    monkeypatch.setattr(W, "find_spots", lambda raw, geom, **kw: (pd.DataFrame(dict(frame=[1.0], row=[1.0], col=[1.0])), np.zeros((64, 64), bool), {}))

    def _enum(lattice, d_min, sg):
        calls.d_min = d_min
        return np.array([[1, 0, 4]])
    monkeypatch.setattr(W, "enumerate_hkl", _enum)
    monkeypatch.setattr(W, "_profile", lambda raw, spots, geom, mask, background=None: SimpleNamespace(half_widths=lambda L: (2, 3)))

    def _tt(U1, U2, lattice, geom, scan, hkl, **kw):
        calls.twin_U = (U1, U2)
        return _targets()
    monkeypatch.setattr(W, "twin_targets", _tt)

    def _red(spots, raw, geom, scan, lattice, U, profile, **kw):
        calls.reduce_U.append(U)
        obs = spots[["frame", "row", "col"]].copy().reset_index(drop=True)
        obs["h"], obs["k"], obs["l"] = 1, 0, 4
        obs["F2"], obs["sigF2"] = 10.0, 1.0
        return obs, pd.DataFrame(), {"input": len(spots), "kept": len(spots)}
    monkeypatch.setattr(W, "reduce_spots", _red)
    return calls


def _run(monkeypatch, o0, o1, **kw):
    monkeypatch.setattr(W, "index_and_refine", lambda *a, **k: o0)
    monkeypatch.setattr(W, "index_and_refine_twin", lambda *a, **k: o1)
    return W.reduce_twin90_scan(np.zeros((4, 64, 64), np.float32), CRYSTAL, U0, SETUP, **kw)


def test_pair_model_path_builds_u2_by_the_twin_law_and_reduces_each_group_under_its_own_orientation(stubs, monkeypatch):
    U1 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = _run(monkeypatch, _orient(), _orient(U=U1))
    assert t.pair_orientation_used and np.array_equal(t.U1, U1) and np.allclose(t.U2, U1 @ RZ90)
    assert np.array_equal(stubs.twin_U[0], U1) and np.allclose(stubs.twin_U[1], U1 @ RZ90)
    assert len(stubs.reduce_U) == 2 and np.array_equal(stubs.reduce_U[0], U1) and np.allclose(stubs.reduce_U[1], U1 @ RZ90)      # blend + c1 under U1, c2 under U2
    assert list(t.obs.kind) == ["blend", "c1", "c2"] and list(t.obs.f1) == [1, 1, 0] and list(t.obs.f2) == [1, 0, 1]
    assert {"p0", "p1", "p2", "sep_px"} <= set(t.obs.columns)


def test_pair_model_failure_falls_back_to_the_centroid_orientation_with_a_warning(stubs, monkeypatch):
    U0c = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.warns(RuntimeWarning, match="pair-model orientation did not converge"):
        t = _run(monkeypatch, _orient(U=U0c), _orient(converged=False, reason="residuals above the gate"))
    assert not t.pair_orientation_used and np.array_equal(t.U1, U0c)


def test_unconverged_first_stage_raises_unless_allowed_and_then_warns(stubs, monkeypatch):
    bad = _orient(converged=False, reason="rms position residual 3.00 px above 2.0")
    with pytest.raises(W.OrientationNotConverged) as e:
        _run(monkeypatch, bad, _orient())
    assert "rms position residual" in str(e.value)
    with pytest.warns(RuntimeWarning, match="NON-converged orientation"):
        t = _run(monkeypatch, bad, _orient(), allow_unconverged=True)
    assert t.pair_orientation_used


def test_default_d_min_is_the_inscribed_edge_and_an_explicit_one_is_used_as_given(stubs, monkeypatch):
    _run(monkeypatch, _orient(), _orient())
    g = SETUP.geometry((64, 64), 4)
    assert stubs.d_min == pytest.approx(SETUP.wavelength_A / (2.0 * np.sin(np.radians(W._tth_max_deg(g) / 2.0))), rel=1e-12)
    _run(monkeypatch, _orient(), _orient(), d_min=0.7)
    assert stubs.d_min == 0.7


def test_nothing_reduced_returns_an_empty_table_and_the_refinement_refuses_it(stubs, monkeypatch):
    monkeypatch.setattr(W, "reduce_spots", lambda spots, *a, **k: (pd.DataFrame(columns=["frame", "row", "col"]), pd.DataFrame(), {"input": 0, "kept": 0}))
    t = _run(monkeypatch, _orient(), _orient())
    assert len(t.obs) == 0
    with pytest.raises(ValueError, match="nothing to refine"):
        W.refine_twin90_structure(CRYSTAL, t)
