"""Geometry multi-start for ``midas-nf-fit-multipoint`` (MIDAS issue #16).

On real AlON NF data the multipoint objective is multimodal in the wedge:
starts at -0.05/0/0.09/0.15/0.30 deg converged to 0.029/0.043/0.086/0.089/
0.103 with overlap 30.63/31.20/31.80/30.86/28.85, and the reported answer came
from starts near 0, which is not the best basin. Both drivers started the
geometry at the paramfile value only.

Covered here:

* HARD path, physical synthetic: an observation volume lit at the spots of
  TWO wedges -- 6 of 7 spots at ``WEDGE_GOOD`` and 2 of 3 at
  ``WEDGE_BAD`` next to the seed. The local search from the seed ends in the
  worse basin; the multi-start reports the better one and records every start.
* SOFT path, analytic two-basin objective in the wedge (the soft surrogate
  itself blurs these two basins into one on a 3-voxel synthetic, so the
  driver is exercised on a known landscape by replacing ``soft_overlap``):
  the seed start ends in the worse basin, the multi-start finds the better
  one; ``multimodal`` is set when two basins are within the margin and NOT
  set when the gap is large or there is only one basin.
* The pure helpers: start construction is deterministic, and the basin rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from midas_nf_fitorientation import fit_multipoint as fm
from midas_nf_fitorientation import geom_multistart as gm
from midas_nf_fitorientation.obs_volume import ObsVolume
from midas_nf_fitorientation.params import parse_paramfile
from midas_nf_fitorientation.soft_overlap import (
    build_forward_model, hkls_cart_thetas,
)

from .test_multipoint_hard import VOXELS, _fcc_hkls, _paramfile_text

# Wedge values in the MIDAS convention (midas_diffract.forward "Wedge
# convention"). With the orientation pinned, the wedge only tilts the rotation
# axis, so spots move ~4x less per degree than under the pre-2026-09 forward
# (which also rotated the crystal by R_y(-W)); the synthetic's two basins and
# the WedgeTol box are spaced accordingly. Measured landscape (hard frac vs
# wedge, this synthetic): 0.872 at -3.0, barrier 0.469 near -1.3, seed 0.568
# at 0, local max 0.738 at +0.6.
WEDGE_GOOD = -3.0        # every 7th spot dark: the better basin, unsaturated
WEDGE_BAD = 0.6          # every 3rd spot dark, uphill from the seed (0)
WEDGE_TOL = 4.0


def _lit(p, hkls, wedge, *, drop_every=0):
    """uint8 (1, F, H, W): one pixel per predicted spot at ``wedge``."""
    F, H, W = p.n_frames_per_distance, p.n_pixels_y, p.n_pixels_z
    arr = np.zeros((1, F, H, W), dtype=np.uint8)
    saved = p.wedge
    p.wedge = wedge
    model = build_forward_model(p, hkls, device="cpu", dtype=torch.float64)
    p.wedge = saved
    eul = torch.tensor([v[2] for v in VOXELS], dtype=torch.float64)
    pos = torch.tensor([(v[0], v[1], 0.0) for v in VOXELS],
                       dtype=torch.float64)
    with torch.no_grad():
        sp = model(eul, pos)
    fr = sp.frame_nr.numpy().reshape(-1)
    yp = sp.y_pixel.numpy().reshape(-1)
    zp = sp.z_pixel.numpy().reshape(-1)
    va = sp.valid.numpy().reshape(-1) > 0.5
    for j, (f_, y_, z_) in enumerate(zip(fr[va], yp[va], zp[va])):
        if drop_every and j % drop_every == 0:
            continue
        fi, yi, zi = int(f_), int(y_), int(z_)
        if 0 <= fi < F and 0 <= yi < H and 0 <= zi < W:
            arr[0, fi, yi, zi] = 1
    return arr


@pytest.fixture
def two_basin_hard(tmp_path, monkeypatch):
    hkls = _fcc_hkls()

    def make(extra: str = "", *, dense: bool = False, tol: float = 0.0):
        pf = tmp_path / "params.txt"
        # tol=0 pins everything but the wedge (WedgeTol WEDGE_TOL, seed 0).
        pf.write_text(_paramfile_text(
            wedge=0.0, refine_wedge=1, tol=tol,
            extra=f"OutputDirectory {tmp_path}\n" + extra).replace(
                "WedgeTol 1.0", f"WedgeTol {WEDGE_TOL}"))
        p = parse_paramfile(pf)
        arr = np.maximum(_lit(p, hkls, WEDGE_GOOD, drop_every=7),
                         _lit(p, hkls, WEDGE_BAD, drop_every=3))
        if dense:                       # the soft path wants dense floats
            obs = ObsVolume.from_dense_array(arr.astype(np.float32),
                                             dtype=torch.float32)
            monkeypatch.setattr(fm, "read_grid", lambda *a, **k: None)
        else:
            obs = ObsVolume.from_dense_array(arr, packed=True)
        cart, _ = hkls_cart_thetas(hkls, p.lattice_constant, p.wavelength)

        class _HKL:
            hkls_int = hkls
            hkls_cart = cart

            def filter_rings(self, rings):
                return self

        monkeypatch.setattr(fm, "read_hkls", lambda out_dir: _HKL())
        monkeypatch.setattr(ObsVolume, "from_spotsinfo",
                            classmethod(lambda cls, *a, **k: obs))
        return pf

    return make


def _hard(pf, **kw):
    kw.setdefault("max_iter", 300)
    kw.setdefault("global_iters", 1)
    kw.setdefault("compile_model", False)
    return fm.fit_multipoint_hard_run(
        str(pf), device="cpu", verbose=False, **kw)


def test_hard_seed_start_finds_worse_basin_multistart_finds_better(
        two_basin_hard, capsys):
    pf = two_basin_hard()
    res = _hard(pf)
    out = capsys.readouterr().out

    trials = res["geometry_trials"]
    # seed + 4 deterministic wedge scan points (+ the ladder's own end)
    labels = [t["label"] for t in trials]
    assert labels[0] == "seed"
    assert sum(l.startswith("scan wedge=") for l in labels) == 4
    assert labels[-1] == "ladder"
    scan_w = sorted(t["start_geometry"]["wedge"] for t in trials
                    if t["label"].startswith("scan"))
    assert scan_w == pytest.approx([WEDGE_TOL * f for f in (-0.9, -0.3, 0.3, 0.9)])

    seed_t = trials[0]
    assert seed_t["start_geometry"]["wedge"] == 0.0
    # Starting near 0: the local search ends in the WORSE basin ...
    assert abs(seed_t["end_geometry"]["wedge"] - WEDGE_BAD) < 0.3
    # ... and the multi-start reports the better one.
    assert abs(res["wedge"] - WEDGE_GOOD) < 0.3
    assert res["final_frac_overlap"] > seed_t["end_frac_overlap"] + 0.05
    assert res["saturated"] is False
    # NOTE: on this 16-parameter synthetic the ladder's global (DE) phase can
    # also reach WEDGE_GOOD from a single start; the point asserted here is
    # that every start and end is recorded and the better basin is reported.
    # Distinct basins, but far apart in objective: not multimodal.
    assert res["n_basins"] >= 2
    assert res["multimodal"] is False
    assert "NOT UNIQUELY DETERMINED" not in out
    assert "geometry multi-start: 5 starts" in out

    js = json.loads(Path(res["result_json"]).read_text())
    for k in ("geometry_trials", "multimodal", "basins",
              "multimodal_criteria", "geometry_names"):
        assert k in js, k
    for t in js["geometry_trials"]:
        for k in ("start", "end", "start_frac_overlap", "end_frac_overlap"):
            assert k in t


def test_hard_single_start_when_wedge_not_refined(synthetic_off):
    """RefineWedge 0, NumIterations 1: exactly one start (old behaviour)."""
    res = synthetic_off
    assert [t["label"] for t in res["geometry_trials"]] == ["seed"]
    assert res["multimodal"] is False


@pytest.fixture
def synthetic_off(tmp_path, monkeypatch, two_basin_hard):
    pf = two_basin_hard()
    pf.write_text(pf.read_text().replace("RefineWedge 1", "RefineWedge 0"))
    return _hard(pf, max_iter=50)


# ---------------------------------------------------------------------------
#  Soft path: analytic two-basin objective in the wedge
# ---------------------------------------------------------------------------

def _two_gauss(w, a_good, a_bad, w_good=0.6, w_bad=0.05, width=0.2):
    return (a_good * torch.exp(-((w - w_good) / width) ** 2)
            + a_bad * torch.exp(-((w - w_bad) / width) ** 2))


@pytest.fixture
def soft_analytic(tmp_path, monkeypatch):
    """Soft driver on an analytic overlap(wedge); only the objective is
    replaced -- starts, boxes, L-BFGS phases, selection and reporting are the
    real ones."""

    def make(a_good, a_bad, extra=""):
        pf = tmp_path / "params.txt"
        pf.write_text(_paramfile_text(
            wedge=0.0, refine_wedge=1, tol=1e-3,
            extra=f"OutputDirectory {tmp_path}\n" + extra))
        p = parse_paramfile(pf)
        hkls = _fcc_hkls()
        cart, _ = hkls_cart_thetas(hkls, p.lattice_constant, p.wavelength)

        class _HKL:
            hkls_int = hkls
            hkls_cart = cart

            def filter_rings(self, rings):
                return self

        dummy = ObsVolume.from_dense_array(
            np.zeros((1, 2, 2, 2), dtype=np.float32), dtype=torch.float32)
        monkeypatch.setattr(fm, "read_hkls", lambda out_dir: _HKL())
        monkeypatch.setattr(fm, "read_grid", lambda *a, **k: None)
        monkeypatch.setattr(ObsVolume, "from_spotsinfo",
                            classmethod(lambda cls, *a, **k: dummy))

        def fake_overlap(model, obs, euler, pos, sigma_px, geom_ov=None):
            w = geom_ov.wedge if (geom_ov is not None
                                  and geom_ov.wedge is not None) \
                else torch.tensor(0.0, dtype=torch.float64)
            # 0 * euler keeps the Euler leaves in the graph (zero gradient).
            return _two_gauss(w, a_good, a_bad) + 0.0 * euler.sum()

        def fake_loss(model, obs, euler, pos, sigma_px, geom_ov=None):
            return 1.0 - fake_overlap(model, obs, euler, pos, sigma_px,
                                      geom_ov)

        monkeypatch.setattr(fm, "soft_overlap", fake_overlap)
        monkeypatch.setattr(fm, "soft_overlap_loss", fake_loss)
        return pf

    return make


def _soft(pf):
    return fm.fit_multipoint_run(str(pf), device="cpu", verbose=False)


def test_soft_seed_start_finds_worse_basin_multistart_finds_better(
        soft_analytic, capsys):
    pf = soft_analytic(a_good=0.30, a_bad=0.20)
    res = _soft(pf)
    out = capsys.readouterr().out
    trials = res["geometry_trials"]
    assert [t["label"] for t in trials][0] == "seed"
    assert len(trials) == 5
    seed_t = trials[0]
    assert abs(seed_t["end_geometry"]["wedge"] - 0.05) < 0.05     # worse
    assert seed_t["end_frac_overlap"] == pytest.approx(0.20, abs=0.01)
    assert abs(res["wedge"] - 0.6) < 0.05                          # better
    assert 1.0 - res["loss"] == pytest.approx(0.30, abs=0.01)
    assert res["multimodal"] is False          # 33% gap >> 2% margin
    assert "NOT UNIQUELY DETERMINED" not in out
    js = json.loads(Path(res["result_json"]).read_text())
    assert js["wedge"] == pytest.approx(res["wedge"])
    assert len(js["geometry_trials"]) == 5 and js["multimodal"] is False


def test_soft_multimodal_flagged(soft_analytic, capsys):
    """Two distinct wedge basins within 1% of each other -> flagged, loudly,
    in the json and in the refined paramfile header."""
    pf = soft_analytic(a_good=0.300, a_bad=0.297)
    res = _soft(pf)
    out = capsys.readouterr().out
    assert res["multimodal"] is True
    assert res["n_competitive_basins"] == 2
    assert res["ambiguous_params"] == ["wedge"]
    assert "NOT UNIQUELY DETERMINED" in out
    js = json.loads(Path(res["result_json"]).read_text())
    assert js["multimodal"] is True
    assert js["multimodal_criteria"]["rel_margin"] == 0.02
    assert js["multimodal_criteria"]["basin_frac"] == 0.25
    assert "NOT UNIQUELY DETERMINED" in Path(res["params_refined"]).read_text()


def test_soft_single_basin_not_flagged(soft_analytic):
    """NULL for the flag: one basin, every start converges to it."""
    pf = soft_analytic(a_good=0.30, a_bad=0.0)
    res = _soft(pf)
    assert res["multimodal"] is False
    ends = [t["end_geometry"]["wedge"] for t in res["geometry_trials"]
            if t["end_frac_overlap"] > 0.1]
    assert ends and all(abs(w - 0.6) < 0.05 for w in ends)


def test_margin_is_configurable(soft_analytic):
    """With a 0.5% margin the 1%-apart pair is no longer flagged."""
    pf = soft_analytic(a_good=0.300, a_bad=0.297,
                       extra="MultipointBasinMargin 0.005\n")
    assert _soft(pf)["multimodal"] is False


# ---------------------------------------------------------------------------
#  Pure helpers
# ---------------------------------------------------------------------------

def test_starts_are_deterministic_and_inside_the_box():
    names = ["tx", "ty", "tz", "Lsd[0]", "ybc[0]", "zbc[0]", "wedge"]
    seed = np.array([0, 0, 0, 5000, 100, 100, 0.0])
    hw = np.array([0.5, 0.5, 0.5, 200, 2, 2, 0.3])
    kw = dict(scan=["wedge"], n_scan=4, n_total=8, rng_seed=7)
    a = gm.build_geometry_starts(names, seed, hw, **kw)
    b = gm.build_geometry_starts(names, seed, hw, **kw)
    assert [s.label for s in a] == [s.label for s in b]
    assert all(np.array_equal(x.x, y.x) for x, y in zip(a, b))
    assert len(a) == 8
    assert sum(s.label.startswith("random") for s in a) == 3
    for s in a:
        assert np.all(np.abs(s.x - seed) <= gm.BOX_FILL * hw + 1e-12)
    # truncation keeps the outermost scan points
    c = gm.build_geometry_starts(names, seed, hw, scan=["wedge"], n_scan=4,
                                 n_total=3, rng_seed=7)
    assert sorted(s.x[-1] for s in c[1:]) == pytest.approx([-0.27, 0.27])


def test_default_n_starts():
    class P:
        refine_wedge = True
        multipoint_scan_tilts = False
        multipoint_geom_scan = 4
        multipoint_geom_starts = 0
        num_iterations = 1
    assert gm.default_n_starts(P) == 5
    P.refine_wedge = False
    assert gm.default_n_starts(P) == 1          # old single-start behaviour
    P.multipoint_scan_tilts = True
    assert gm.default_n_starts(P) == 13
    P.multipoint_geom_starts = 3
    assert gm.default_n_starts(P) == 3


def test_analyse_basins_rule():
    names = ["a", "wedge"]
    hw = np.array([1.0, 0.4])
    tr = [dict(end=[0.0, 0.05], end_frac_overlap=0.312),
          dict(end=[0.0, 0.07], end_frac_overlap=0.311),   # same basin (<0.1)
          dict(end=[0.0, 0.30], end_frac_overlap=0.318),   # distinct, best
          dict(end=[0.0, -0.3], end_frac_overlap=0.200)]   # distinct, far
    r = gm.analyse_basins(tr, names, hw, basin_frac=0.25, rel_margin=0.02)
    assert r["n_basins"] == 3
    assert r["n_competitive_basins"] == 2
    assert r["multimodal"] is True
    assert r["ambiguous_params"] == ["wedge"]
    r = gm.analyse_basins(tr, names, hw, basin_frac=0.25, rel_margin=0.01)
    assert r["multimodal"] is False


# ---------------------------------------------------------------------------
#  The start geometry must REACH the objective. Regression for a defect found
#  on AlON NF layer 31 (hard, RefineWedge 0, MultipointScanTilts 1): every
#  scan start scored EXACTLY the seed overlap and never moved. Cause: the
#  hard path's torch.compile'd forward ignored overrides(), so the objective
#  always saw the seed geometry wherever inductor works (Linux).
# ---------------------------------------------------------------------------

@pytest.fixture
def dynamo_aot_eager(monkeypatch):
    """Make torch.compile work on any host while keeping Dynamo's tracing
    semantics (where the defect lives). Inductor needs a C++ toolchain; on
    the Mac it fails, the driver falls back to eager, and the defect hides."""
    real = torch.compile

    def _compile(fn, **kw):
        kw["backend"] = "aot_eager"
        return real(fn, **kw)

    monkeypatch.setattr(torch, "compile", _compile)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# seed + each tilt at -0.9 and +0.9 deg (outermost points kept first)
# NON-ZERO seed tilts (also the truth: the obs is lit at the paramfile
# geometry). With zero seed tilts the `_has_tilts` flag flips inside
# overrides() and forces Dynamo to recompile, which masked the defect; the
# AlON run seeded tx=0.576.
TX0 = 0.3
_TILT_SCAN = (f"tx {TX0}\nMultipointScanTilts 1\nTiltsTol 1.0\n"
              "MultipointGeomStarts 7\n")


def _tilt_scan_pf(make, **kw):
    pf = make(_TILT_SCAN, **kw)
    pf.write_text(pf.read_text().replace("RefineWedge 1", "RefineWedge 0")
                  .replace("Wedge 0.0", f"Wedge {WEDGE_GOOD}"))
    return pf


def _tx_scans(res):
    trials = {t["label"]: t for t in res["geometry_trials"]}
    scans = [t for l, t in trials.items() if l.startswith("scan tx=")]
    assert len(scans) == 2
    return trials["seed"], scans


def test_hard_compiled_objective_sees_geometry(two_basin_hard,
                                               dynamo_aot_eager):
    """tx started 0.9 deg off the truth must score differently from the seed
    and must move back. Before the fix both scored EXACTLY the seed overlap
    and ended where they started."""
    pf = _tilt_scan_pf(two_basin_hard)
    res = _hard(pf, compile_model=True, max_iter=200)
    seed, scans = _tx_scans(res)
    for t in scans:
        assert abs(t["start_geometry"]["tx"] - TX0) == pytest.approx(0.9)
        assert abs(t["start_frac_overlap"]
                   - seed["start_frac_overlap"]) > 0.05, t
        assert abs(t["end_geometry"]["tx"] - TX0) < 0.45, t
        assert t["end_frac_overlap"] > t["start_frac_overlap"] + 0.05, t
    # and it was the compiled forward that did it (not the eager fallback)
    assert res["compiled_forward"] is True


def test_hard_compiled_matches_eager(two_basin_hard, dynamo_aot_eager):
    """Same run with the compiled and the eager forward: same objective at
    every start."""
    pf = _tilt_scan_pf(two_basin_hard)
    a = _hard(pf, compile_model=True, max_iter=100)
    b = _hard(pf, compile_model=False, max_iter=100)
    assert len(a["geometry_trials"]) == len(b["geometry_trials"])
    for ta, tb in zip(a["geometry_trials"], b["geometry_trials"]):
        assert ta["start_frac_overlap"] == pytest.approx(
            tb["start_frac_overlap"], abs=1e-12)
    assert a["compiled_forward"] is True and b["compiled_forward"] is False


def test_soft_objective_sees_geometry(two_basin_hard):
    """Soft path, REAL surrogate (no compile there; its starts go through
    TanhBox.set_x): a tx start 0.9 deg off scores differently from the
    seed."""
    pf = _tilt_scan_pf(two_basin_hard, dense=True, tol=0.01)
    res = fm.fit_multipoint_run(str(pf), device="cpu", verbose=False)
    seed, scans = _tx_scans(res)
    for t in scans:
        assert abs(t["start_geometry"]["tx"] - TX0) == pytest.approx(
            0.9, abs=1e-6)
        assert abs(t["start_frac_overlap"]
                   - seed["start_frac_overlap"]) > 1e-3, t
