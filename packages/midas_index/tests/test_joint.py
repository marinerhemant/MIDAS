"""Joint multi-mounting indexing (midas_index.joint).

Synthetic, noise-free: one Au grain measured in two mountings (identity and a 90-degree remount about
the beam axis), each with only four narrow omega windows. Observed spots are the forward adapter's own
predictions (same column layout the matcher reads), binned with build_bin_index. The joint indexer must
recover the grain's orientation and position from seeds of either mounting.
"""

import math

import numpy as np
import pytest
import torch

from midas_index import IndexerParams
from midas_index.io import build_bin_index
from midas_index.io.csv import read_hkls_csv
from midas_index.joint import JointIndexer, Mounting, load_mount_matrix
from midas_index.pipeline import IndexerContext

A = 4.08
WL = 0.172979
LSD = 1_000_000.0
WINDOWS = [(-15.0, 15.0), (75.0, 105.0), (165.0, 180.0), (-180.0, -165.0), (-105.0, -75.0)]


def _rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _write_hkls(path):
    """Au rings 1-3 ((111), (200), (220)) in the MIDAS hkls.csv layout."""
    fams = {1: (1, 1, 1), 2: (2, 0, 0), 3: (2, 2, 0)}
    rows = []
    for ring, fam in fams.items():
        seen = set()
        import itertools
        for perm in set(itertools.permutations(fam)):
            for sg in itertools.product((1, -1), repeat=3):
                h = tuple(p * s for p, s in zip(perm, sg))
                if h in seen:
                    continue
                seen.add(h)
                d = A / math.sqrt(sum(x * x for x in h))
                th = math.degrees(math.asin(WL / (2 * d)))
                rad = LSD * math.tan(math.radians(2 * th))
                g = [x / A for x in h]
                rows.append((*h, d, ring, *g, th, 2 * th, rad))
    with open(path, "w") as f:
        f.write("h k l D-spacing RingNr g1 g2 g3 Theta 2Theta Radius\n")
        for r in rows:
            f.write(" ".join(str(v) for v in r) + "\n")
    return {ring: LSD * math.tan(2 * math.asin(WL * math.sqrt(sum(x * x for x in fam)) / (2 * A)))
            for ring, fam in fams.items()}


def _params(radii):
    p = IndexerParams()
    p.Distance = LSD
    p.Wavelength = WL
    p.Rsample = 60.0
    p.Hbeam = 60.0
    p.px = 200.0
    p.SpaceGroup = 225
    p.LatticeConstant = (A, A, A, 90.0, 90.0, 90.0)
    p.StepsizePos = 20.0
    p.StepsizeOrient = 1.0
    p.MarginOme = 0.5
    p.MarginRad = 500.0
    p.MarginRadial = 500.0
    p.MarginEta = 500.0
    p.EtaBinSize = 1.0
    p.OmeBinSize = 1.0
    p.ExcludePoleAngle = 6.0
    p.MinMatchesToAcceptFrac = 0.5
    p.RingNumbers = [1, 2, 3]
    p.RingRadii = dict(radii)
    p.OmegaRanges = list(WINDOWS)
    p.BoxSizes = [(-1.5e6, 1.5e6, -1.5e6, 1.5e6)] * len(WINDOWS)
    p.UseFriedelPairs = 0
    p.OutputFolder = "."
    return p


def _synthetic(tmp_path):
    radii = _write_hkls(tmp_path / "hkls.csv")
    hkls_real, hkls_int = read_hkls_csv(tmp_path / "hkls.csv", ring_numbers=[1, 2, 3])
    rng = np.random.default_rng(3)
    from scipy.spatial.transform import Rotation
    R_true = Rotation.random(random_state=11).as_matrix()
    p_true = np.array([30.0, -20.0, 10.0])
    mounts = [np.eye(3), _rx(90.0)]
    ctxs, seeds, sid = [], [], 1
    for k, Mk in enumerate(mounts):
        p = _params(radii)
        dummy = IndexerContext(params=p, hkls_real=hkls_real, hkls_int=hkls_int,
                               obs=np.zeros((1, 9)), bin_data=np.zeros(1, np.int32),
                               bin_ndata=np.zeros(2, np.int32), device=torch.device("cpu"), dtype=torch.float64)
        th, valid = dummy.adapter.simulate(torch.tensor(Mk @ R_true)[None], torch.tensor(Mk @ p_true)[None])
        th = th[0][valid[0]].numpy()
        obs = np.zeros((len(th), 9))
        obs[:, 0], obs[:, 1], obs[:, 2] = th[:, 10], th[:, 11], th[:, 6]
        obs[:, 3] = 0.0
        obs[:, 4] = np.arange(sid, sid + len(th)); sid += len(th)
        obs[:, 5], obs[:, 6], obs[:, 7], obs[:, 8] = th[:, 9], th[:, 12], np.degrees(2 * th[:, 8]), th[:, 13]
        data, ndata = build_bin_index(obs, eta_bin_size=p.EtaBinSize, ome_bin_size=p.OmeBinSize, n_rings=3)
        ctxs.append(IndexerContext(params=p, hkls_real=hkls_real, hkls_int=hkls_int, obs=obs, bin_data=data,
                                   bin_ndata=ndata, device=torch.device("cpu"), dtype=torch.float64))
        seeds += [(k, int(i)) for i, r in zip(obs[:, 4], obs[:, 5]) if int(r) == 1][:2]
    return R_true, p_true, mounts, ctxs, seeds


def test_frames_roundtrip():
    from scipy.spatial.transform import Rotation
    M = Rotation.random(random_state=5).as_matrix()
    ji = JointIndexer.__new__(JointIndexer)
    ji.device, ji.dtype = torch.device("cpu"), torch.float64
    ji.M = [torch.tensor(M)]
    ji.S = [torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)]
    R = torch.tensor(Rotation.random(random_state=6).as_matrix())[None]
    p = torch.tensor([[10.0, 20.0, -5.0]], dtype=torch.float64)
    Rk, pk = ji.to_mount(R, p, 0)
    Rs, ps = ji.to_sample(Rk, pk, 0)
    assert torch.allclose(Rs, R, atol=1e-12) and torch.allclose(ps, p, atol=1e-9)


def test_load_mount_matrix(tmp_path):
    M = _rx(90.0)
    np.savetxt(tmp_path / "m.txt", M)
    np.save(tmp_path / "m.npy", M)
    np.savez(tmp_path / "m.npz", mount=M)
    for f in ("m.txt", "m.npy", "m.npz"):
        assert np.allclose(load_mount_matrix(tmp_path / f), M)


def test_joint_recovers_grain_from_two_windowed_mountings(tmp_path):
    pytest.importorskip("scipy")
    R_true, p_true, mounts, ctxs, seeds = _synthetic(tmp_path)
    assert all(len(c.obs) >= 6 for c in ctxs), "each mounting must see a handful of spots"
    ji = JointIndexer([Mounting(layer_dir=".", rotation=M) for M in mounts], device="cpu",
                      contexts=ctxs, seeds=seeds)
    res = ji.run()
    assert res.grains, "no grain found"
    from midas_stress.orientation import misorientation_om_batch
    mis = [math.degrees(float(misorientation_om_batch(g.orient_mat.reshape(1, 9), R_true.reshape(1, 9), 225)[0]))
           for g in res.grains]
    i = int(np.argmin(mis))
    g = res.grains[i]
    assert mis[i] < 0.05, f"orientation error {mis[i]:.3f} deg"
    assert np.linalg.norm(g.position - p_true) < 2.0, f"position error {np.linalg.norm(g.position - p_true):.2f} um"
    assert g.completeness > 0.9
    assert (g.matches_per_mounting > 0).all(), "both mountings must contribute matches"
    assert len(res.grains) == 1, f"{len(res.grains)} grains for a one-grain sample"
