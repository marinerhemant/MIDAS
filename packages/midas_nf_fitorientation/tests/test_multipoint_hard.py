"""``fit_multipoint_hard_run`` (``--objective hard``) on a small synthetic.

Covers three defects found on real 1-ID AlON/Au NF data:

1. The refined geometry was only printed, and only under ``--verbose``;
   nothing reached disk. It must now always write ``multipoint_result.json``
   and a ready-to-use ``params_refined.txt``.
2. The hard path silently ignored ``RefineWedge 1``: no wedge slot, no
   bound, nothing reported. It must refine the wedge inside ``WedgeTol``.
3. On real data the hard objective saturated at exactly 1.0 in round 1, so
   the returned geometry was an arbitrary point on a plateau. That must be
   detected, warned about, and flagged in the result json.

The synthetic: a few voxels of an fcc crystal, an observation volume lit
(with a small dilation) exactly where the forward model puts their spots at
a KNOWN geometry, and a paramfile seeded away from it. Only file IO is
monkeypatched; the paramfile parser, forward model, packed obs lookup and
the optimiser are the real ones.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from midas_nf_fitorientation import fit_multipoint as fm
from midas_nf_fitorientation.obs_volume import ObsVolume
from midas_nf_fitorientation.params import parse_paramfile
from midas_nf_fitorientation.soft_overlap import (
    build_forward_model, hkls_cart_thetas,
)

# Voxel (x, y) in um and Euler seeds in radians -- the SAME numbers the
# forward model is called with, so the synthetic is self-consistent.
VOXELS = [
    (-50.0, 20.0, (0.3, 0.8, 1.1)),
    (40.0, -30.0, (1.7, 0.4, 2.6)),
    (10.0, 60.0, (2.9, 1.3, 0.2)),
]
TRUE_WEDGE = 0.4          # degrees


def _fcc_hkls(max_h2: int = 11) -> np.ndarray:
    rows = []
    r = range(-3, 4)
    for h in r:
        for k in r:
            for l in r:
                s = h * h + k * k + l * l
                if s == 0 or s > max_h2:
                    continue
                if len({h % 2, k % 2, l % 2}) != 1:     # fcc: unmixed
                    continue
                rows.append((h, k, l))
    return np.asarray(rows, dtype=np.float64)


def _paramfile_text(*, wedge: float, refine_wedge: int, extra: str = "",
                    tol: float = 1.0) -> str:
    """``tol`` scales every non-wedge tolerance (0 pins them)."""
    lines = [
        "nDistances 1",
        "Lsd 100000",
        f"LsdTol {20 * tol}",
        "BC 128 128",
        f"BCTol {0.2 * tol} {0.2 * tol}",
        "px 200",
        "NrPixels 256",
        "OmegaStart -180",
        "OmegaStep 2",
        "StartNr 1",
        "EndNr 180",
        "Wavelength 0.172979",
        "LatticeParameter 4.08 4.08 4.08 90 90 90",
        "ExcludePoleAngle 6",
        "tx 0", "ty 0", "tz 0",
        f"TiltsTol {0.02 * tol}",
        f"Wedge {wedge}",
        "WedgeTol 1.0",
        f"RefineWedge {refine_wedge}",
        f"OrientTol {0.02 * tol}",
        "NumIterations 1",
    ]
    for i, (x, y, (e1, e2, e3)) in enumerate(VOXELS):
        lines.append(
            f"GridPoints {i} {i} 0 {x} {y} 5 1 {e1} {e2} {e3} 0.9 1")
    return "\n".join(lines) + "\n" + extra


def _synthetic_obs(p, hkls, *, wedge: float, all_lit: bool = False,
                   dilate: int = 2, drop_every: int = 0):
    """Packed obs lit around the spots predicted at ``wedge``.

    ``drop_every=n`` leaves every n-th spot dark, so even the true geometry
    scores below 1.0 (an UNsaturated objective)."""
    D, F, H, W = 1, p.n_frames_per_distance, p.n_pixels_y, p.n_pixels_z
    if all_lit:
        arr = np.ones((D, F, H, W), dtype=np.uint8)
        return ObsVolume.from_dense_array(arr, packed=True)
    arr = np.zeros((D, F, H, W), dtype=np.uint8)
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
    n_lit = 0
    for j, (f_, y_, z_) in enumerate(zip(fr[va], yp[va], zp[va])):
        if drop_every and j % drop_every == 0:
            continue
        fi, yi, zi = int(f_), int(y_), int(z_)
        if not (0 <= fi < F and 0 <= yi < H and 0 <= zi < W):
            continue
        arr[0, fi, max(0, yi - dilate):yi + dilate + 1,
            max(0, zi - dilate):zi + dilate + 1] = 1
        n_lit += 1
    assert n_lit >= 20, f"synthetic too sparse: {n_lit} spots on detector"
    return ObsVolume.from_dense_array(arr, packed=True)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """Return ``make(seed_wedge, refine_wedge, all_lit)`` -> paramfile path."""
    hkls = _fcc_hkls()

    def make(*, seed_wedge: float, refine_wedge: int, all_lit: bool = False,
             tol: float = 1.0, drop_every: int = 0):
        pf = tmp_path / "params.txt"
        pf.write_text(_paramfile_text(wedge=seed_wedge,
                                      refine_wedge=refine_wedge, tol=tol,
                                      extra=f"OutputDirectory {tmp_path}\n"))
        p = parse_paramfile(pf)
        obs = _synthetic_obs(p, hkls, wedge=TRUE_WEDGE, all_lit=all_lit,
                             drop_every=drop_every)
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


def _run(pf, **kw):
    kw.setdefault("max_iter", 400)
    kw.setdefault("global_iters", 4)
    return fm.fit_multipoint_hard_run(
        str(pf), device="cpu", verbose=False, compile_model=False, **kw)


# ---------------------------------------------------------------------------
#  (a) RefineWedge is honoured by the hard path
# ---------------------------------------------------------------------------

def test_hard_path_refines_wedge(synthetic, capsys):
    """Seeded at 0 deg with the truth at 0.4 deg inside WedgeTol 1.0, the
    hard path must move the wedge towards the truth, stay in its box, and
    report it. It used to have no wedge slot at all."""
    pf = synthetic(seed_wedge=0.0, refine_wedge=1)
    res = _run(pf)
    out = capsys.readouterr().out

    assert "wedge ON" in out
    assert "Wedge:" in out and "refined" in out
    assert res["wedge_refined"] is True
    assert -1.0 <= res["wedge"] <= 1.0                  # WedgeTol box
    assert abs(res["wedge"] - TRUE_WEDGE) < abs(0.0 - TRUE_WEDGE) / 2
    assert res["final_frac_overlap"] > res["seed_frac_overlap"]


def test_hard_path_wedge_off_keeps_c_layout(synthetic, capsys):
    """RefineWedge 0: wedge untouched (reported as fixed), not a parameter."""
    pf = synthetic(seed_wedge=0.0, refine_wedge=0)
    res = _run(pf, max_iter=100, global_iters=1)
    out = capsys.readouterr().out

    assert "wedge OFF" in out
    assert res["wedge_refined"] is False
    assert res["wedge"] == 0.0
    assert "wedge" not in res["flat_params"]
    assert len(res["eulers"]) == len(VOXELS)


# ---------------------------------------------------------------------------
#  (b) saturation is detected and flagged; (1) the result reaches disk
# ---------------------------------------------------------------------------

def test_saturated_objective_is_flagged(synthetic, capsys):
    """Every pixel lit: every predicted spot matches at ANY geometry, the
    objective is 1.0 from the seed on, and the geometry is undetermined."""
    pf = synthetic(seed_wedge=0.0, refine_wedge=1, all_lit=True)
    res = _run(pf, max_iter=100, global_iters=1)
    out = capsys.readouterr().out

    assert res["final_frac_overlap"] == pytest.approx(1.0, abs=1e-12)
    assert res["saturated"] is True
    assert res["saturated_at"] == "seed"
    assert res["under_determined"] is True
    # With nothing to see, every geometry coordinate is flat.
    assert set(res["flat_params"]) == {
        "tx", "ty", "tz", "Lsd[0]", "ybc[0]", "zbc[0]", "wedge"}
    assert "UNDER-DETERMINED" in out and "saturated" in out

    js = json.loads(open(res["result_json"]).read())
    assert js["saturated"] is True and js["under_determined"] is True
    assert "UNDER-DETERMINED" in open(res["params_refined"]).read()


def test_unsaturated_well_posed_fit_is_not_flagged(synthetic, capsys):
    """NULL for the flag: only the wedge free, a quarter of the spots dark so
    even the truth scores < 1.0. The objective is peaked in the wedge, so
    neither saturation nor a flat parameter may be reported."""
    pf = synthetic(seed_wedge=0.0, refine_wedge=1, tol=0.0, drop_every=4)
    res = _run(pf)
    out = capsys.readouterr().out

    assert res["final_frac_overlap"] < 1.0 - 1e-6
    assert res["saturated"] is False
    assert res["flat_params"] == []
    assert res["under_determined"] is False
    assert "UNDER-DETERMINED" not in out
    assert abs(res["wedge"] - TRUE_WEDGE) < 0.1


def test_result_written_without_verbose(synthetic, tmp_path, capsys):
    """The summary prints and both files are written with verbose=False."""
    pf = synthetic(seed_wedge=0.0, refine_wedge=1)
    res = _run(pf, max_iter=100, global_iters=1)
    out = capsys.readouterr().out

    for line in ("Original val", "Final value", "Layer 0: Lsd=",
                 "Tilts (shared)", "Wedge:", "multipoint_result.json",
                 "params_refined.txt"):
        assert line in out, line
    assert res["result_json"] == str(tmp_path / "multipoint_result.json")

    js = json.loads(open(res["result_json"]).read())
    for k in ("Lsd", "y_BC", "z_BC", "tilts", "wedge", "final_frac_overlap",
              "seed_frac_overlap", "saturated", "under_determined"):
        assert k in js, k

    # The refined paramfile round-trips through the real parser and carries
    # exactly the refined geometry; everything else is untouched.
    p2 = parse_paramfile(res["params_refined"])
    p0 = parse_paramfile(pf)
    assert p2.Lsd == pytest.approx(res["Lsd"], abs=1e-6)
    assert p2.ybc == pytest.approx(res["y_BC"], abs=1e-6)
    assert p2.zbc == pytest.approx(res["z_BC"], abs=1e-6)
    assert [p2.tx, p2.ty, p2.tz] == pytest.approx(res["tilts"], abs=1e-8)
    assert p2.wedge == pytest.approx(res["wedge"], abs=1e-8)
    assert p2.lsd_tol == p0.lsd_tol and p2.bc_tol_a == p0.bc_tol_a
    assert len(p2.grid_points) == len(p0.grid_points)


def test_refined_paramfile_text_replaces_only_geometry():
    text = ("# comment\nnDistances 2\nLsd 1000\nLsdTol 50\nLsd 2000\n"
            "BC 1 2\nBCTol 1 1\nBC 3 4\ntx 0\nty 0\nOther 7\n")
    out = fm.refined_paramfile_text(
        text, Lsd=[1001.5, 2002.5], y_BC=[1.5, 3.5], z_BC=[2.5, 4.5],
        tilts=[0.1, 0.2, 0.3], wedge=0.05)
    lines = out.splitlines()
    assert lines[0] == "# comment"
    assert "LsdTol 50" in lines and "BCTol 1 1" in lines
    assert "Other 7" in lines
    assert [l for l in lines if l.split()[0] == "Lsd"] == [
        "Lsd 1001.500000", "Lsd 2002.500000"]
    assert [l for l in lines if l.split()[0] == "BC"] == [
        "BC 1.500000 2.500000", "BC 3.500000 4.500000"]
    # present keys replaced in place, absent ones (tz, Wedge) appended
    assert "tx 0.10000000" in lines and "tz 0.30000000" in lines
    assert "Wedge 0.05000000" in lines
    # wedge=None leaves any Wedge line alone
    out2 = fm.refined_paramfile_text(
        "Wedge 0.7\n", Lsd=[], y_BC=[], z_BC=[], tilts=[0, 0, 0])
    assert "Wedge 0.7" in out2.splitlines()
