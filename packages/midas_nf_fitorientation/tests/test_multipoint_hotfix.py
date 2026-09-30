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
TRUE_WEDGE = 0.4          # degrees (obs lit here; fixed, NOT refined in the hotfix)


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
    kw.setdefault("max_iter", 300)
    kw.setdefault("global_iters", 2)
    return fm.fit_multipoint_hard_run(str(pf), device="cpu", verbose=False, **kw)


def test_hotfix_result_written_without_verbose(synthetic, tmp_path, capsys):
    pf = synthetic(seed_wedge=TRUE_WEDGE, refine_wedge=0)
    res = _run(pf, compile_model=False, max_iter=100, global_iters=1)
    out = capsys.readouterr().out
    for line in ("Original val", "Final value", "Layer 0: Lsd=", "Tilts (shared)",
                 "Wrote", "multipoint_result.json", "params_refined.txt"):
        assert line in out, line
    js = json.loads(open(res["result_json"]).read())
    for k in ("Lsd", "y_BC", "z_BC", "tilts", "final_frac_overlap", "seed_frac_overlap"):
        assert k in js, k
    p2 = parse_paramfile(res["params_refined"])
    assert p2.Lsd == pytest.approx(res["Lsd"], abs=1e-6)
    assert [p2.tx, p2.ty, p2.tz] == pytest.approx(res["tilts"], abs=1e-8)


@pytest.fixture
def dynamo_aot_eager(monkeypatch):
    """Compile with Dynamo's ``aot_eager`` backend so the test runs anywhere
    (inductor needs a C++ toolchain and is unavailable on the Mac). Set
    ``MIDAS_TEST_REAL_INDUCTOR=1`` on a Linux box to use the default backend,
    which is what production runs use."""
    import os
    if not os.environ.get("MIDAS_TEST_REAL_INDUCTOR"):
        real = torch.compile

        def _compile(fn, **kw):
            kw["backend"] = "aot_eager"
            return real(fn, **kw)

        monkeypatch.setattr(torch, "compile", _compile)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_hotfix_compiled_forward_sees_geometry(synthetic, dynamo_aot_eager):
    """obs is lit for tx = 3 deg; the seed is tx = 8 (TiltsTol 10). A compiled
    forward that ignores the geometry override cannot move tx (flat objective)."""
    pf = synthetic(seed_wedge=TRUE_WEDGE, refine_wedge=0)
    truth_txt = pf.read_text().replace("tx 0\n", "tx 3\n")
    pf.write_text(truth_txt)
    p = parse_paramfile(pf)
    obs = _synthetic_obs(p, _fcc_hkls(), wedge=TRUE_WEDGE)
    orig = ObsVolume.from_spotsinfo
    ObsVolume.from_spotsinfo = classmethod(lambda cls, *a, **k: obs)
    try:
        pf.write_text(truth_txt.replace("tx 3\n", "tx 8\n").replace("TiltsTol 0.02", "TiltsTol 10"))
        a = _run(pf, compile_model=True, max_iter=1500)
        b = _run(pf, compile_model=False, max_iter=1500)
    finally:
        ObsVolume.from_spotsinfo = orig
    assert a["compiled_forward"] is True and b["compiled_forward"] is False
    assert b["final_frac_overlap"] > b["seed_frac_overlap"] + 0.02, "eager must improve"
    assert a["final_frac_overlap"] > a["seed_frac_overlap"] + 0.02
    assert a["tilts"][0] < 6.0
    assert a["final_frac_overlap"] == pytest.approx(b["final_frac_overlap"], abs=1e-6)
