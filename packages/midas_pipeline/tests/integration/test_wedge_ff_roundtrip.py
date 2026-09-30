"""Slow E2E: FF round trip under a rotation-axis Wedge (issues #17, #18).

Simulate an FF dataset with midas_diffract at Parameters ``Wedge = W``
(W in {0, +0.5, -2} deg; 10 grains, seed 42), run the full midas-pipeline FF
chain (c-omp indexer + c-omp refiner) with the SAME ``Wedge``, and compare
Grains.csv with the simulation truth IN THE SAME FRAME -- no rotation is
applied anywhere, because the one MIDAS convention (``midas_diffract.forward``
module doc) makes the Parameters value mean the same thing to simulator,
indexer and refiner. Also:

* the Python (torch) refiner kernel ``refine_block`` on the same seeds and
  ExtraInfo, with the model built by ``driver._build_model`` (wedge =
  paramstest Wedge) and the raw pre-wedge observation columns;
* the sign test: data simulated at W = -2 and run with +2 must not index;
* NF (acceptance b): for grains of the W = -2 Grains.csv, NF data of that
  orientation at the same Wedge, fitted by the production NF hard-overlap
  polish read from an NF parameter file with the same Wedge, returns the
  Grains.csv matrix to the NF pixel limit (and cannot with the other sign).

Before #18 was fixed the W = -2 run put grain positions ~440 um off (mostly
z) with correct orientations; before #17, midas_diffract needed Wedge -W and
a rotated frame to agree with the C chain.

Measured on the Mac (2026-09-22, this code, C chain): median misorientation
/ position error W = 0 0.0155 deg / 0.44 um, W = +0.5 0.0100 / 0.21,
W = -2 0.0087 / 0.35 (before the simulator frame / tGap fixes the W = 0
control itself was 0.187 deg / 3.7 um).

Also grain-tx (``midas_joint_ff_calibrate``), which is how Parameters-file
Wedge values are produced: at the true Wedge it must return it, and from 0
it must move toward it.

``slow``: seven pipeline runs, ~9 min on 8 CPU threads. Run with
``python -m pytest -m slow tests/integration/test_wedge_ff_roundtrip.py``.
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.slow

torch = pytest.importorskip("torch")
ms = pytest.importorskip("midas_stress.orientation")
fg_c = pytest.importorskip("midas_fit_grain.backend_c")
ix_c = pytest.importorskip("midas_index.backend_c")
pytest.importorskip("midas_diffract.simulate_panel_zarrs")

_MIDAS = Path(__file__).resolve().parents[4]
_TEMPLATE = _MIDAS / "FF_HEDM" / "Example" / "Parameters.txt"
N_GRAINS = 10
SEED = 42
WEDGES = (0.0, 0.5, -2.0)

if not _TEMPLATE.exists():
    pytest.skip(f"example parameter template absent: {_TEMPLATE}", allow_module_level=True)
if not (fg_c.available() and ix_c.available()):
    pytest.skip("c-omp indexer / refiner binaries not built", allow_module_level=True)


def _run(root: Path, w_sim: float, w_run: float) -> Path:
    from midas_pipeline.testing import generate_synthetic_dataset
    sim = root / "sim"
    sim.mkdir(parents=True)
    tmpl = root / "Parameters.txt"
    tmpl.write_text(re.sub(r"^Wedge .*$", f"Wedge {w_sim}", _TEMPLATE.read_text(),
                           flags=re.M))
    zarr = generate_synthetic_dataset(out_dir=sim, params_template=tmpl,
                                      n_grains=N_GRAINS, seed=SEED, n_cpus=8)
    pf = sim / tmpl.name
    pf.write_text(re.sub(r"^Wedge .*$", f"Wedge {w_run}", pf.read_text(), flags=re.M))
    cmd = [sys.executable, "-m", "midas_pipeline", "run", "--scan-mode", "ff",
           "--params", str(pf), "--result", str(root / "run"), "--zarr", str(zarr),
           "--n-cpus", "8", "--device", "cpu", "--dtype", "float64"]
    with open(root / "pipeline.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return root


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    for w in WEDGES:
        out[w] = _run(tmp_path_factory.mktemp(f"wedge_{w:+.1f}"), w, w)
    return out


def _truth(root):
    t = np.loadtxt(root / "sim" / "GrainsSim.csv", comments="%", ndmin=2)
    return t[:, 1:10].reshape(-1, 3, 3), t[:, 10:13]


def _miso(OM, OMs):
    return np.degrees(np.asarray(ms.misorientation_om_batch(
        np.repeat(OM[None], len(OMs), 0), OMs, 225), float))


def _compare(OMr, Pr, OMt, Pt):
    """Per truth grain: median over the recovered entries matched to it."""
    per = {}
    for om, p in zip(OMr, Pr):
        m = _miso(om, OMt)
        k = int(np.argmin(m))
        if m[k] < 1.0:
            per.setdefault(k, []).append((m[k], float(np.linalg.norm(p - Pt[k]))))
    mis = np.array([np.median([x[0] for x in v]) for v in per.values()])
    pos = np.array([np.median([x[1] for x in v]) for v in per.values()])
    return len(per), mis, pos


def _c_chain(root):
    g = np.loadtxt(root / "run" / "LayerNr_1" / "Grains.csv", comments="%", ndmin=2)
    return _compare(g[:, 1:10].reshape(-1, 3, 3), g[:, 10:13], *_truth(root))


def _python_refiner(root):
    from midas_fit_grain import driver as D
    from midas_fit_grain.config import FitConfig
    from midas_fit_grain.io_binary import read_extra_info
    from midas_fit_grain.observations import ObservedSpots
    from midas_fit_grain.refine_block import refine_block
    L = root / "run" / "LayerNr_1"
    cfg = FitConfig.from_param_file(L / "paramstest.txt")
    dev, dt = torch.device("cpu"), torch.float64
    ids = D._read_spots_to_index(L / "SpotsToIndex.csv")
    ib, ibf = D._read_consolidated_as_ff(L / "Output", len(ids))
    extra = read_extra_info(L / "ExtraInfo.bin", mmap=False)
    mtt = 2.0 * math.degrees(math.atan(cfg.RhoD / cfg.Lsd)) if cfg.RhoD > 0 else 180.0
    hk, th, rn = D._read_hkls_csv(L / "hkls.csv", cfg.RingNumbers, max_two_theta_deg=mtt)
    model, slot = D._build_model(cfg, device=dev, dtype=dt, hkls_int=hk,
                                 thetas_deg=th, ring_nr=rn)
    assert float(model.wedge) == pytest.approx(cfg.Wedge)
    obs, P0, E0, L0 = [], [], [], []
    for r in range(len(ids)):
        n = int(ib[r, 14])
        if n == 0:
            continue
        obs.append(ObservedSpots.from_extra_info(
            extra, ibf[r, :n, 0].astype(np.int64), device=dev, dtype=dt,
            wedge_deg=cfg.Wedge))
        P0.append(ib[r, 10:13])
        E0.append(D._orientmat_to_euler_zxz(ib[r, 1:10].reshape(3, 3)))
        L0.append(np.array(cfg.LatticeConstant, float))
    res = refine_block(cfg, model=model, grains_obs=obs,
                       init_positions=torch.tensor(np.array(P0), dtype=dt),
                       init_eulers=torch.tensor(np.array(E0), dtype=dt),
                       init_lattices=torch.tensor(np.array(L0), dtype=dt),
                       pred_ring_slot=slot)
    OMr = np.stack([model.euler2mat(g.euler.view(1, 3).double()).numpy()[0]
                    for g in res.grains])
    Pr = np.stack([g.position.detach().numpy().reshape(3) for g in res.grains])
    return _compare(OMr, Pr, *_truth(root))


@pytest.mark.parametrize("w", WEDGES)
def test_c_chain_roundtrip(runs, w):
    n0, mis0, _ = _c_chain(runs[0.0])
    n, mis, pos = _c_chain(runs[w])
    print(f"C chain W={w:+.1f}: {n} grains, miso median {np.median(mis):.4f} max "
          f"{mis.max():.4f} deg, pos median {np.median(pos):.2f} max {pos.max():.2f} um "
          f"(W=0 control miso median {np.median(mis0):.4f})")
    assert n >= 6, n
    assert np.median(mis) <= np.median(mis0) + 0.05, (np.median(mis), np.median(mis0))
    assert mis.max() < 0.15, mis
    assert np.median(pos) < 3.0 and pos.max() < 10.0, pos      # um


@pytest.mark.parametrize("w", WEDGES)
def test_python_refiner_roundtrip(runs, w):
    n0, mis0, _ = _python_refiner(runs[0.0]) if w != 0.0 else (None, None, None)
    n, mis, pos = _python_refiner(runs[w])
    print(f"torch refiner W={w:+.1f}: {n} grains, miso median {np.median(mis):.4f} max "
          f"{mis.max():.4f} deg, pos median {np.median(pos):.2f} max {pos.max():.2f} um")
    assert n >= 6, n
    if mis0 is not None:
        assert np.median(mis) <= np.median(mis0) + 0.05, (np.median(mis), np.median(mis0))
    assert mis.max() < 0.15, mis
    assert np.median(pos) < 3.0, pos


def test_wrong_sign_does_not_index(tmp_path):
    """Sign test at the chain level: W = -2 data with Parameters Wedge +2."""
    root = _run(tmp_path, -2.0, 2.0)
    g = root / "run" / "LayerNr_1" / "Grains.csv"
    if g.exists():
        OMr = np.loadtxt(g, comments="%", ndmin=2)
        n, mis, pos = _compare(OMr[:, 1:10].reshape(-1, 3, 3), OMr[:, 10:13], *_truth(root))
        assert n == 0 or np.median(pos) > 50.0, (n, pos)


def test_nf_fit_returns_the_ff_grains_matrix(runs, tmp_path):
    """Acceptance (b): NF data of the Grains.csv crystal at the same Wedge,
    fitted by the production NF hard-overlap polish, returns that matrix."""
    nfp = pytest.importorskip("midas_nf_fitorientation.params")
    so = pytest.importorskip("midas_nf_fitorientation.soft_overlap")
    ov = pytest.importorskip("midas_nf_fitorientation.obs_volume")
    hp = pytest.importorskip("midas_nf_fitorientation.hard_polish")
    from tests.integration.test_wedge_convention import reference_spots, _hkls, WL, A_AU
    from midas_diffract.forward import HEDMForwardModel
    W = -2.0
    g = np.loadtxt(runs[W] / "run" / "LayerNr_1" / "Grains.csv", comments="%", ndmin=2)
    LSD, PX, NP, BC, STEP = [5000.0, 7000.0], 3.0, 512, 256.0, 2.0

    def params(w):
        pf = tmp_path / f"nf_{w:+.1f}.txt"
        pf.write_text("\n".join([
            "nDistances 2", *(f"Lsd {L}" for L in LSD), f"BC {BC} {BC}", f"BC {BC} {BC}",
            f"px {PX}", f"NrPixels {NP}", "OmegaStart -180", f"OmegaStep {STEP}",
            "StartNr 1", f"EndNr {int(360 / STEP)}", f"Wavelength {WL}",
            f"LatticeParameter {A_AU} {A_AU} {A_AU} 90 90 90", "ExcludePoleAngle 6",
            "tx 0", "ty 0", "tz 0", f"Wedge {w}"]) + "\n")
        return nfp.parse_paramfile(pf)

    ints, _ = _hkls()
    pos = np.array([60.0, -40.0, 0.0])               # an NF voxel in the z = 0 layer
    n_checked = 0
    for row in g[:3]:
        O = row[1:10].reshape(3, 3)
        ref = reference_spots(W, O, pos, Lsds=LSD, flip_y=False, bc=BC, px=PX)
        p = params(W)
        F = p.n_frames_per_distance
        arr = np.zeros((2, F, NP, NP), np.uint8)
        for s in ref:
            f = int(math.floor((s["omega"] + 180.0) / STEP))
            if 0 <= f < F and all(0 <= d[2] < NP and 0 <= d[3] < NP for d in s["det"]):
                for d, (_, _, ypx, zpx) in enumerate(s["det"]):
                    arr[d, f, int(ypx), int(zpx)] = 1
        obs = ov.ObsVolume.from_dense_array(arr, packed=True)
        e0 = torch.tensor(_euler(O) + np.array([0.0015, -0.001, 0.0012]), dtype=torch.float64)
        out = {}
        for w in (W, -W):
            model = so.build_forward_model(params(w), ints, device="cpu", dtype=torch.float64)
            res = hp.polish_hard_frac(model, obs, e0, torch.tensor(pos),
                                      tol_rad=math.radians(1.0), max_iter=2000)
            Of = HEDMForwardModel.euler2mat(res.eul[None].double()).numpy()[0]
            out[w] = (float(_miso(Of, O[None])[0]), res.hard_frac)
        # The hard objective is flat (frac = 1) over a pixel-sized plateau:
        # 1 px = 3 um at 5 mm is 0.034 deg, and NM stops anywhere on it
        # (measured 0.011, 0.014, 0.025 deg on these grains; wrong sign
        # 0.45-0.58 deg at frac < 0.1). One pixel bounds the plateau.
        print(f"NF grain {n_checked}: hard polish {out}")
        assert out[W][1] > 0.99 and out[W][0] < 0.035, out
        assert out[-W][1] < 0.5, out                           # the other sign
        # The same NF forward, fitted to the continuous spot centres instead
        # of a pixelised overlap, returns the Grains.csv matrix exactly.
        from tests.integration.test_wedge_convention import _nf_lsq_fit_general
        e_ls = _nf_lsq_fit_general(
            so.build_forward_model(params(W), ints, device="cpu",
                                   dtype=torch.float64).double(),
            ref, e0.numpy(), pos, len(LSD))
        O_ls = HEDMForwardModel.euler2mat(torch.tensor(e_ls[None])).numpy()[0]
        mls = float(_miso(O_ls, O[None])[0])
        print(f"NF grain {n_checked}: continuous-centre fit miso {mls:.2e} deg")
        assert mls < 1e-4       # the forward solver floor; far below 1e-3
        n_checked += 1
    assert n_checked >= 1


def _grain_tx_wedge(root: Path, param_file: Path) -> float:
    """Run grain-tx (refine Wedge) on a run's layer; return the written Wedge."""
    gr = pytest.importorskip("midas_joint_ff_calibrate.grain_refine")
    out = root / "graintx_Wedge.txt"
    gr.refine_geometry_from_grains(param_file, root / "run" / "LayerNr_1",
                                   refine_params=("Wedge",), out_paramstest=out)
    return [float(l.split()[1]) for l in out.read_text().splitlines()
            if l.split()[:1] == ["Wedge"]][0]


def test_grain_tx_wedge_roundtrip(tmp_path):
    """grain-tx speaks the C convention (the owner's workflow: pipeline ->
    grain-tx Wedge -> Parameters -> rerun). Synthetic at C-Wedge W = -0.5:

    * run at the true W: grain-tx must return W (the fixed point is exact);
    * run at 0: pass 1 must move TOWARD W (right sign, smaller error). The
      pose is held at the C refiner's values, which absorbed part of the
      wedge, so one pass does not reach W; measured pass sequence from 0 on
      this synthetic: -0.207, -0.372, -0.462, -0.489.

    Before 2026-09 grain-tx fitted a relative, wedge-free-frame correction
    with the opposite-sign midas_diffract: +0.012 from 0 on a -0.3 synthetic,
    and a -0.41 fixed point on this one."""
    W = -0.5
    at_true = _run(tmp_path / "true", W, W)
    w_fix = _grain_tx_wedge(at_true, at_true / "sim" / "Parameters.txt")
    print(f"grain-tx at the true Wedge {W}: returned {w_fix:+.5f}")
    assert abs(w_fix - W) < 0.01, w_fix
    from_zero = _run(tmp_path / "zero", W, 0.0)
    w1 = _grain_tx_wedge(from_zero, from_zero / "sim" / "Parameters.txt")
    print(f"grain-tx from Wedge 0: returned {w1:+.5f}")
    assert np.sign(w1) == np.sign(W) and abs(w1 - W) < 0.8 * abs(W), w1


def _euler(O):
    from midas_fit_grain.driver import _orientmat_to_euler_zxz
    return np.asarray(_orientmat_to_euler_zxz(O), float)
