"""ONE rotation-axis Wedge convention across FF, NF and PF (issue #17).

The convention (``midas_diffract.forward`` module doc, "Wedge convention"),
which is what the FF C refiner (midas_fit_grain ``FitUnified.c``) has always
assumed: with W the Parameters-file ``Wedge``, a crystal with stage-frame
orientation O at stage-frame position p is seen at rotation angle omega as

    G_lab = R_y(-W) R_z(omega) O g,      pos_lab = R_y(-W) R_z(omega) p.

Every branch below is held to ONE independent reference -- a Rodrigues ray
trace written here from that statement alone (axis n = (-sin W, 0, cos W),
stage->lab tilt S = R_y(-W), both by Rodrigues, omega solved in closed form
from the elastic condition) -- and each check is run twice: with +W it must
agree, with -W it must NOT (the sign test, acceptance item d). Branches:

  FF   midas_diffract forward (flip_y, one Lsd)
  FF   C refiner port c_port: DisplacementInTheSpot + CorrectForOme must map
       the raw spot of (O, p) back onto the wedge-free spot of O   [#18]
  FF   midas_transforms fit-setup wedge correction (feeds the indexer)
  FF   midas_index numba forward kernel (its callers pass 0; the dormant
       wedge math must still be THIS convention)
  NF   midas_nf_fitorientation forward from a parsed NF parameter file
       (multi-distance, flip_y = False) and a least-squares orientation fit
       through it that must return O to 1e-6 deg          (acceptance b)
  NF   midas_nf_preprocess diffr_spots (reads Wedge from the parameter file)
  PF   midas_pf_odf geometry from a paramstest, pushed through the FF C
       refiner port back onto the same wedge-free spots      (acceptance c)

All synthetic, fixed numbers, noiseless.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

md = pytest.importorskip("midas_diffract.forward")

DEG2RAD = math.pi / 180.0
WL = 0.172979                       # A
A_AU = 4.08                         # A
W_TEST = 1.5                        # deg; every check also runs at -W_TEST
EULER = (1.9, 1.1, 4.2)             # rad, Bunge ZXZ
POS = np.array([420.0, -310.0, 95.0])   # um, stage frame (FF grains are 3-D)


# ---------------------------------------------------------------------------
#  Independent reference
# ---------------------------------------------------------------------------

def _rod(n, a):
    n = np.asarray(n, float)
    K = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)


def _om_bunge(e):
    """Bunge ZXZ passive-to-active orientation matrix, written out directly."""
    p1, P, p2 = e
    c1, s1, c, s, c2, s2 = (math.cos(p1), math.sin(p1), math.cos(P),
                            math.sin(P), math.cos(p2), math.sin(p2))
    return np.array([
        [c1 * c2 - s1 * s2 * c, -c1 * s2 - s1 * c2 * c, s1 * s],
        [s1 * c2 + c1 * s2 * c, -s1 * s2 + c1 * c2 * c, -c1 * s],
        [s2 * s, c2 * s, c]])


def _hkls(max_h2=12):
    ints = [(h, k, l) for h in range(-4, 5) for k in range(-4, 5) for l in range(-4, 5)
            if 0 < h * h + k * k + l * l <= max_h2
            and len({h % 2, k % 2, l % 2}) == 1]
    ints = np.array(ints, float)
    return ints, ints / A_AU


def reference_spots(W_deg, O, p, *, Lsds, flip_y, bc, px):
    """Raw spots of (O, p) at Wedge W: list of dicts with omega (deg), eta
    (deg), two_theta (deg) and per-distance (y_um, z_um, y_px, z_px)."""
    W = W_deg * DEG2RAD
    n = np.array([-math.sin(W), 0.0, math.cos(W)])
    S = _rod([0.0, 1.0, 0.0], -W)
    k_in = np.array([1.0 / WL, 0.0, 0.0])
    _, cart = _hkls()
    out = []
    for m, h in enumerate(cart):
        G = O @ h
        g2 = G @ G
        # R_y(-W) R_z(w) G has x = cosW (cos w Gx - sin w Gy) - sinW Gz, and
        # |k_in + G_lab| = |k_in|  <=>  G_lab_x = -WL |G|^2 / 2.
        A = math.cos(W) * G[0]
        B = -math.cos(W) * G[1]
        C = -WL * g2 / 2.0 + math.sin(W) * G[2]
        r = math.hypot(A, B)
        if r < 1e-12 or abs(C) > r:
            continue
        base = math.atan2(B, A)
        d = math.acos(C / r)
        for w in (base + d, base - d):
            w = math.atan2(math.sin(w), math.cos(w))
            R = _rod(n, w) @ S
            Gl = R @ G
            ko = k_in + Gl
            assert abs(np.linalg.norm(ko) * WL - 1.0) < 1e-12
            pl = R @ p
            eta = math.degrees(math.atan2(-ko[1], ko[2]))
            if abs(eta) < 6.0 or 180.0 - abs(eta) < 6.0:
                continue
            tth = math.degrees(math.acos(ko[0] / np.linalg.norm(ko)))
            det = []
            for L in Lsds:
                t = (L - pl[0]) / ko[0]
                y = pl[1] + t * ko[1]
                z = pl[2] + t * ko[2]
                ypx = bc - y / px if flip_y else bc + y / px
                det.append((y, z, ypx, bc + z / px))
            out.append(dict(m=m, omega=math.degrees(w), eta=eta, tth=tth, det=det))
    return out


def _match(ref, omega_deg, m, tol=1e-6):
    for s in ref:
        if s["m"] == m and abs(((s["omega"] - omega_deg + 180) % 360) - 180) < tol:
            return s
    return None


def _O():
    return _om_bunge(EULER)


@pytest.fixture(scope="module")
def O_model_agrees():
    """The reference Bunge matrix must be the forward's euler2mat (else every
    comparison below would test the Euler convention, not the wedge)."""
    O = _O()
    Om = md.HEDMForwardModel.euler2mat(torch.tensor([EULER], dtype=torch.float64))
    assert np.abs(Om.numpy()[0] - O).max() < 1e-14
    return O


# ---------------------------------------------------------------------------
#  FF: midas_diffract forward
# ---------------------------------------------------------------------------

def _ff_model(W):
    ints, cart = _hkls()
    th = np.arcsin(WL * np.linalg.norm(cart, axis=1) / 2.0)
    geom = md.HEDMGeometry(Lsd=1.0e6, y_BC=1024.0, z_BC=1024.0, px=200.0,
                           omega_start=-180.0, omega_step=0.25, n_frames=1440,
                           n_pixels_y=2048, n_pixels_z=2048, min_eta=6.0,
                           wavelength=WL, flip_y=True, wedge=W)
    # .double(): the model keeps hkls in fp32 by default, which alone is
    # ~4e-5 px here; the convention checks below want fp64 all through.
    return md.HEDMForwardModel(torch.tensor(cart), torch.tensor(th), geom).double()


def _model_vs_ref(model, W_ref, O, p, *, Lsds, flip_y, bc, px):
    ref = reference_spots(W_ref, O, p, Lsds=Lsds, flip_y=flip_y, bc=bc, px=px)
    with torch.no_grad():
        sp = model(torch.tensor(np.array([EULER]), dtype=torch.float64),
                   torch.tensor(np.asarray(p)[None], dtype=torch.float64))
    om = sp.omega.numpy().reshape(2, -1) / DEG2RAD
    yp = sp.y_pixel.numpy().reshape(len(Lsds), 2, -1)
    zp = sp.z_pixel.numpy().reshape(len(Lsds), 2, -1)
    ok = sp.valid.numpy().reshape(2, -1) > 0.5
    worst, n_hit, n = 0.0, 0, 0
    for k, m in zip(*np.nonzero(ok)):
        n += 1
        s = _match(ref, om[k, m], m, tol=1e-4)
        if s is None:
            continue
        n_hit += 1
        for d in range(len(Lsds)):
            worst = max(worst, abs(yp[d, k, m] - s["det"][d][2]),
                        abs(zp[d, k, m] - s["det"][d][3]))
    return n, n_hit, worst


@pytest.mark.parametrize("sgn", [1.0, -1.0])
def test_ff_forward_is_the_convention(O_model_agrees, sgn):
    W = W_TEST
    n, n_hit, worst = _model_vs_ref(_ff_model(sgn * W), W, O_model_agrees, POS,
                                    Lsds=[1.0e6], flip_y=True, bc=1024.0, px=200.0)
    assert n > 60
    if sgn > 0:
        # 1e-4 px: the forward's omega solver (EPS-regularised quadratic)
        # sits 3.6e-5 px from this closed form at W = 0 as well.
        assert n_hit == n and worst < 1e-4, (n, n_hit, worst)
    else:                       # the other sign: omegas move, pixels miss
        assert n_hit < n // 2 or worst > 1.0, (n, n_hit, worst)


# ---------------------------------------------------------------------------
#  FF: C refiner port (#18) and fit-setup wedge correction
# ---------------------------------------------------------------------------

def _wedge_free(O):
    """W = 0, p = 0 reference spots: what a wedge correction must return."""
    return reference_spots(0.0, O, np.zeros(3), Lsds=[1.0e6], flip_y=True,
                           bc=1024.0, px=200.0)


def _nearest_free(free, s, omega_c):
    cands = [f for f in free if f["m"] == s["m"]]
    return min(cands, key=lambda f: abs(((f["omega"] - omega_c + 180) % 360) - 180))


@pytest.mark.parametrize("sgn", [1.0, -1.0])
def test_c_refiner_port_maps_raw_spots_to_wedge_free(O_model_agrees, sgn):
    c_port = pytest.importorskip("midas_fit_grain.c_port")
    O, W, LSD = O_model_agrees, W_TEST, 1.0e6
    raw = reference_spots(W, O, POS, Lsds=[LSD], flip_y=True, bc=1024.0, px=200.0)
    free = _wedge_free(O)
    worst = 0.0
    for s in raw:
        y, z = s["det"][0][0], s["det"][0][1]
        dY, dZ = c_port._displacement_in_the_spot(*POS, LSD, y, z, s["omega"],
                                                  sgn * W, 0.0)
        ys, zs, oc, *_ = c_port._correct_for_ome(y - dY, z - dZ, LSD, s["omega"],
                                                 WL, sgn * W)
        f = _nearest_free(free, s, oc)
        worst = max(worst, abs(ys - f["det"][0][0]), abs(zs - f["det"][0][1]),
                    1e3 * abs(((oc - f["omega"] + 180) % 360) - 180))
    assert len(raw) > 60
    if sgn > 0:
        assert worst < 1e-3, worst          # um (and mdeg)
    else:
        assert worst > 100.0, worst


@pytest.mark.parametrize("sgn", [1.0, -1.0])
def test_fit_setup_wedge_correction(O_model_agrees, sgn):
    tr = pytest.importorskip("midas_transforms.fit_setup.transform")
    O, W, LSD = O_model_agrees, W_TEST, 1.0e6
    raw = reference_spots(W, O, np.zeros(3), Lsds=[LSD], flip_y=True, bc=1024.0, px=200.0)
    free = _wedge_free(O)
    t = lambda a: torch.tensor(a, dtype=torch.float64)
    y = t([s["det"][0][0] for s in raw]); z = t([s["det"][0][1] for s in raw])
    om = t([s["omega"] for s in raw])
    yw, zw, ow, ew, tw = tr.correct_wedge_full(y, z, t(LSD), om, t(WL), t(sgn * W))
    worst = 0.0
    for i, s in enumerate(raw):
        f = _nearest_free(free, s, float(ow[i]))
        worst = max(worst, abs(float(yw[i]) - f["det"][0][0]),
                    abs(float(zw[i]) - f["det"][0][1]),
                    1e3 * abs(((float(ow[i]) - f["omega"] + 180) % 360) - 180),
                    1e3 * abs(float(ew[i]) - f["eta"]))
    if sgn > 0:
        assert worst < 1e-3, worst
    else:
        assert worst > 100.0, worst


@pytest.mark.parametrize("sgn", [1.0, -1.0])
def test_index_numba_kernel_wedge_math(O_model_agrees, sgn):
    # The numba kernel runs in a SUBPROCESS: on macOS numba's omp threading
    # layer segfaults next to torch's libomp once another test in the same
    # process has initialised it (midas_index.compute.matching pins the
    # workqueue layer, but only if imported before numba's first launch).
    import json
    import os
    import subprocess
    import sys
    pytest.importorskip("numba")
    pytest.importorskip("midas_index.compute.forward_numba")
    O, W = O_model_agrees, W_TEST
    ints, cart = _hkls()
    code = f"""
import json, numpy as np
import midas_index.compute.matching
from midas_index.compute import forward_numba as fn
O = np.array({O.tolist()}); cart = np.array({cart.tolist()})
th = np.arcsin({WL} * np.linalg.norm(cart, axis=1) / 2.0)
theor, valid = fn._simulate_numba_inner(
    O[None].copy(), np.zeros((1, 3)), np.ascontiguousarray(cart), th,
    np.linalg.norm(cart, axis=1), np.ones(len(cart), np.int64), np.zeros(20), 19,
    {sgn * W * DEG2RAD!r}, 1.0e6, {6.0 * DEG2RAD!r},
    np.zeros((0, 2)), np.zeros((0, 4)), False, 1e-12)
k = np.nonzero(valid[0])[0]
print(json.dumps(theor[0][k][:, [2, 6, 7]].tolist()))
"""
    env = dict(os.environ, NUMBA_THREADING_LAYER="workqueue")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, check=True)
    rows = json.loads(r.stdout.strip().splitlines()[-1])
    ref = reference_spots(W, O, np.zeros(3), Lsds=[1.0e6], flip_y=True, bc=1024.0, px=200.0)
    n, n_hit, worst = 0, 0, 0.0
    for m, om, eta in rows:
        m = int(m)
        n += 1
        s = _match(ref, om, m, tol=1e-6)
        if s is not None:
            n_hit += 1
            worst = max(worst, abs(eta - s["eta"]))
    assert n > 60
    if sgn > 0:
        assert n_hit == n and worst < 1e-8, (n, n_hit, worst)
    else:
        assert n_hit < n // 2, (n, n_hit)


# ---------------------------------------------------------------------------
#  NF: fit-orientation forward + orientation fit, and diffr_spots
# ---------------------------------------------------------------------------

NF_LSD = [5000.0, 7000.0]
NF_PX = 1.5
NF_BC = 1024.0
NF_POS = np.array([150.0, -220.0, 0.0])      # NF voxels live in the z = 0 plane


def _nf_paramfile(tmp_path, W):
    pf = tmp_path / f"nf_params_{W:+.2f}.txt"
    pf.write_text("\n".join([
        "nDistances 2", *(f"Lsd {L}" for L in NF_LSD),
        f"BC {NF_BC} {NF_BC}", f"BC {NF_BC} {NF_BC}",
        f"px {NF_PX}", "NrPixels 2048",
        "OmegaStart -180", "OmegaStep 0.25", "StartNr 1", "EndNr 1440",
        f"Wavelength {WL}", f"LatticeParameter {A_AU} {A_AU} {A_AU} 90 90 90",
        "ExcludePoleAngle 6", "tx 0", "ty 0", "tz 0", f"Wedge {W}",
    ]) + "\n")
    return pf


def _nf_model(tmp_path, W):
    nfp = pytest.importorskip("midas_nf_fitorientation.params")
    so = pytest.importorskip("midas_nf_fitorientation.soft_overlap")
    p = nfp.parse_paramfile(_nf_paramfile(tmp_path, W))
    assert p.wedge == pytest.approx(W)
    ints, _ = _hkls()
    return so.build_forward_model(p, ints, device="cpu", dtype=torch.float64).double()


@pytest.mark.parametrize("sgn", [1.0, -1.0])
def test_nf_forward_is_the_convention(tmp_path, O_model_agrees, sgn):
    W = W_TEST
    n, n_hit, worst = _model_vs_ref(_nf_model(tmp_path, sgn * W), W, O_model_agrees,
                                    NF_POS, Lsds=NF_LSD, flip_y=False, bc=NF_BC, px=NF_PX)
    assert n > 40
    if sgn > 0:
        assert n_hit == n and worst < 1e-4, (n, n_hit, worst)   # see FF test
    else:
        assert n_hit < n // 2 or worst > 1.0, (n, n_hit, worst)


def _nf_lsq_fit(model, ref, seed_euler):
    e, n = _nf_lsq_fit_general(model, ref, seed_euler, NF_POS, len(NF_LSD),
                               return_n=True)
    return e, n


def _nf_lsq_fit_general(model, ref, seed_euler, pos_um, n_dist, *, return_n=False):
    """Least-squares Euler fit of the NF forward to reference spot centres at
    every distance; the correspondence is fixed from the (truth-free) seed."""
    e = torch.tensor(np.asarray(seed_euler, float), dtype=torch.float64,
                     requires_grad=True)
    pos = torch.tensor(np.asarray(pos_um, float)[None], dtype=torch.float64)
    with torch.no_grad():
        sp = model(e.detach()[None], pos)
    om = sp.omega.numpy().reshape(2, -1) / DEG2RAD
    ok = sp.valid.numpy().reshape(2, -1) > 0.5
    pairs = []
    for k, m in zip(*np.nonzero(ok)):
        s = _match(ref, om[k, m], m, tol=2.0)
        if s is not None:
            pairs.append((k, m, s))
    idx = torch.tensor([k * ok.shape[1] + m for k, m, _ in pairs])
    ty = torch.tensor(np.array([[s["det"][d][2] for _, _, s in pairs]
                                for d in range(n_dist)]))
    tz = torch.tensor(np.array([[s["det"][d][3] for _, _, s in pairs]
                                for d in range(n_dist)]))
    opt = torch.optim.LBFGS([e], lr=1.0, max_iter=200, tolerance_grad=1e-16,
                            tolerance_change=1e-18, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        s = model(e[None], pos)
        yp = s.y_pixel.reshape(n_dist, -1)[:, idx]
        zp = s.z_pixel.reshape(n_dist, -1)[:, idx]
        loss = ((yp - ty) ** 2 + (zp - tz) ** 2).sum()
        loss.backward()
        return loss

    for _ in range(3):
        opt.step(closure)
    if return_n:
        return e.detach().numpy(), len(pairs)
    return e.detach().numpy()


def _miso_deg(A, B):
    ms = pytest.importorskip("midas_stress.orientation")
    return float(np.degrees(np.asarray(ms.misorientation_om_batch(
        A[None], B[None], 225), float))[0])


@pytest.mark.parametrize("sgn", [1.0, -1.0])
def test_nf_orientation_fit_returns_the_same_matrix(tmp_path, O_model_agrees, sgn):
    """Acceptance (b), convention level: NF data of the crystal O at Wedge W,
    fitted by the NF forward read from an NF parameter file with Wedge W,
    returns O itself (the FF Grains.csv matrix of the same crystal) to far
    below 1e-3 deg; with -W it cannot."""
    O, W = O_model_agrees, W_TEST
    ref = reference_spots(W, O, NF_POS, Lsds=NF_LSD, flip_y=False, bc=NF_BC, px=NF_PX)
    model = _nf_model(tmp_path, sgn * W)
    seed = np.array(EULER) + np.array([0.004, -0.003, 0.005])      # ~0.3 deg off
    e_fit, n_pairs = _nf_lsq_fit(model, ref, seed)
    assert n_pairs > 40
    Ofit = md.HEDMForwardModel.euler2mat(torch.tensor(e_fit[None])).numpy()[0]
    miso = _miso_deg(Ofit, O)
    if sgn > 0:
        assert miso < 1e-6, miso
    else:
        assert miso > 0.05, miso


@pytest.mark.parametrize("sgn", [1.0, -1.0])
def test_nf_diffr_spots_reads_and_applies_wedge(tmp_path, O_model_agrees, sgn):
    dsp = pytest.importorskip("midas_nf_preprocess.diffr_spots.params")
    dpl = pytest.importorskip("midas_nf_preprocess.diffr_spots.pipeline")
    ori = pytest.importorskip("midas_nf_preprocess.diffr_spots.orientations")
    O, W = O_model_agrees, W_TEST
    params = dsp.DiffrSpotsParams.from_paramfile(_nf_paramfile(tmp_path, sgn * W))
    assert params.wedge == pytest.approx(sgn * W)
    ints, cart = _hkls()
    th = np.degrees(np.arcsin(WL * np.linalg.norm(cart, axis=1) / 2.0))
    q = ori.orient_matrix_to_quat(torch.tensor(O[None]))
    res = dpl.predict_spots(q, torch.tensor(cart), torch.tensor(th),
                            distance=params.primary_distance,
                            exclude_pole_angle=6.0, wedge_deg=params.wedge)
    ref = reference_spots(W, O, np.zeros(3), Lsds=NF_LSD, flip_y=False, bc=NF_BC, px=NF_PX)
    om = res.omegas.numpy()[0]; eta = res.etas.numpy()[0]; v = res.valid.numpy()[0]
    n, n_hit, worst = 0, 0, 0.0
    for m, j in zip(*np.nonzero(v)):
        n += 1
        s = _match(ref, om[m, j], m, tol=1e-6)
        if s is not None:
            n_hit += 1
            worst = max(worst, abs(eta[m, j] - s["eta"]))
    assert n > 40
    if sgn > 0:
        assert n_hit == n and worst < 1e-8, (n, n_hit, worst)
    else:
        assert n_hit < n // 2, (n, n_hit)


# ---------------------------------------------------------------------------
#  PF: pf_odf geometry from a paramstest == FF C refiner model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sgn", [1.0, -1.0])
def test_pf_odf_forward_matches_ff_refiner(tmp_path, O_model_agrees, sgn):
    """Acceptance (c). pf_odf builds its model from the paramstest ``Wedge``;
    its raw spots for (O, p), taken back through the FF C refiner port
    (DisplacementInTheSpot + CorrectForOme, Wedge from the same file) land on
    the wedge-free spots of O: PF and FF read one crystal the same way."""
    pio = pytest.importorskip("midas_pf_odf.io")
    c_port = pytest.importorskip("midas_fit_grain.c_port")
    O, W, LSD = O_model_agrees, W_TEST, 1.0e6
    pt = tmp_path / "paramstest.txt"
    pt.write_text("\n".join([
        f"Distance {LSD};", "YBCFit 1024;", "ZBCFit 1024;", "px 200;",
        f"Wavelength {WL};", "OmegaStart -180;", "OmegaStep 0.25;",
        "OmegaRange -180 180;", "MinEta 6;", "tyFit 0;", "tzFit 0;",
        f"Wedge {sgn * W};"]) + "\n")
    params = pio.parse_paramstest(pt)
    geom = pio.geometry_from_paramstest(params, n_pixels_y=2048, n_pixels_z=2048,
                                        n_frames=1440)
    assert geom.wedge == pytest.approx(sgn * W)
    ints, cart = _hkls()
    th = np.arcsin(WL * np.linalg.norm(cart, axis=1) / 2.0)
    model = md.HEDMForwardModel(torch.tensor(cart), torch.tensor(th), geom).double()
    with torch.no_grad():
        sp = model(torch.tensor(np.array([EULER]), dtype=torch.float64),
                   torch.tensor(POS[None], dtype=torch.float64))
    om = sp.omega.numpy().reshape(-1) / DEG2RAD
    y = ((1024.0 - sp.y_pixel.numpy().reshape(-1)) * 200.0)
    z = ((sp.z_pixel.numpy().reshape(-1) - 1024.0) * 200.0)
    ok = sp.valid.numpy().reshape(-1) > 0.5
    M = len(cart)
    free = _wedge_free(O)
    worst, n = 0.0, 0
    for i in np.nonzero(ok)[0]:
        dY, dZ = c_port._displacement_in_the_spot(*POS, LSD, y[i], z[i], om[i], W, 0.0)
        ys, zs, oc, *_ = c_port._correct_for_ome(y[i] - dY, z[i] - dZ, LSD, om[i], WL, W)
        f = _nearest_free(free, {"m": int(i % M)}, oc)
        worst = max(worst, abs(ys - f["det"][0][0]), abs(zs - f["det"][0][1]))
        n += 1
    assert n > 60
    if sgn > 0:
        assert worst < 0.02, worst      # um; the forward's 3.6e-5 px solver floor
    else:
        assert worst > 100.0, worst
