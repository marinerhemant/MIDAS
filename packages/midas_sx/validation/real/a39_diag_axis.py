"""A39 diagnostic: do the detector tilts (tx, ty, tz) and the omega-axis wedge explain the row drift of the later frames? Fit them (with U, centre, distance) to the reflections assigned in
frames < 450 by robust least squares on pixel positions predicted AT THE MEASURED omega (no Ewald-crossing solve: a diagnostic, ~0.3 px approximation), then report the residual per
50-frame window for ALL assigned blobs."""
import json, math, os, sys
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"), os.path.dirname(os.path.abspath(__file__))]
_root = os.path.dirname(os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"))
sys.path[:0] = [os.path.join(_root, p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(_root, p))]
import numpy as np, pandas as pd, torch, gemmi
from dataclasses import replace
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from midas_hkls import Lattice
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.assign import assign_hkl, reciprocal_basis
from midas_sx.conventions import Convention, transform_rc
from midas_sx.geometry import spot_directions
from midas_sx.orient import _newton_pixels
from midas_sx.setting import match_setting
from a39_geom import swing_geometry
out, cif = sys.argv[1], sys.argv[2]
sp = pd.read_csv(os.path.join(out, "spots.csv"))
H = json.load(open(os.path.join(out, "info.json")))["header"]; nr, nc = H["shape"]
conv = Convention(0, -1)
ss = gemmi.read_small_structure(cif); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90)
Bm = reciprocal_basis(lat)
r, c = transform_rc(sp.row.to_numpy(), sp.col.to_numpy(), nr, nc, conv.dihedral)
sp = sp.assign(row=r, col=c)
g0, scan = swing_geometry(H, conv, "cols", -1, -1)
d = sp[sp.frame < 450].sort_values("integrated", ascending=False).head(600)
q = pixel_to_qlab(d.row.to_numpy(), d.col.to_numpy(), g0, device="cpu", dtype=torch.float64)
om = torch.as_tensor(np.radians(scan.omega_of(d.frame.to_numpy())), dtype=torch.float64)
res = index_ab_initio(qlab_to_qsample(q, om, 0.0).numpy(), two_pi=True, tol=0.15, min_reflections=30)
U0, P, err = match_setting(np.asarray(res.UB), lat)
om_all = scan.omega_of(sp.frame.to_numpy())
s, tth, _ = spot_directions(sp.row.to_numpy(), sp.col.to_numpy(), g0)
hkl, hf, resid = assign_hkl(s, om_all, lat, U0, g0.wavelength_A)
use = (resid < 0.25) & (tth > 2) & np.any(hkl != 0, axis=1)
print("initially assigned:", use.sum(), "of", len(sp), flush=True)
fit = use & (sp.frame.to_numpy() < 450)
H_ = hkl[fit]; rr = sp.row.to_numpy()[fit]; cc = sp.col.to_numpy()[fit]; oo = np.radians(om_all[fit])


def geom_of(p):
    return replace(g0, bcz_px=g0.bcz_px + p[3], bcy_px=g0.bcy_px + p[4], lsd_um=g0.lsd_um * (1 + p[5]), tx_deg=g0.tx_deg + p[6], ty_deg=g0.ty_deg + p[7], tz_deg=g0.tz_deg + p[8])


def predict(p, hk, om_):
    U = Rot.from_rotvec(p[:3]).as_matrix() @ U0
    qs = 2 * np.pi * (U @ Bm @ hk.T).T
    W = math.radians(p[9])
    c_, s_ = np.cos(om_), np.sin(om_)
    ql = np.stack([c_ * qs[:, 0] - s_ * qs[:, 1], s_ * qs[:, 0] + c_ * qs[:, 1], qs[:, 2]], 1)
    # wedge: q_lab = R_y(-W) R_z(omega) q_sample
    cw, sw = math.cos(W), math.sin(W)
    ql = np.stack([cw * ql[:, 0] - sw * ql[:, 2], ql[:, 1], sw * ql[:, 0] + cw * ql[:, 2]], 1)
    return _newton_pixels(ql, geom_of(p))


def resid_fn(p, free):
    pf = np.zeros(10); pf[free] = p
    pr, pc = predict(pf, H_, oo)
    rres = np.where(np.isfinite(pr), (pr - rr), 1e3); cres = np.where(np.isfinite(pc), (pc - cc), 1e3)
    return np.concatenate([rres, cres])


for label, free in (("U + centre + distance", [0, 1, 2, 3, 4, 5]), ("+ detector tilts tx ty tz", [0, 1, 2, 3, 4, 5, 6, 7, 8]), ("+ wedge", [0, 1, 2, 3, 4, 5, 9]), ("+ tilts + wedge", list(range(10)))):
    sol = least_squares(resid_fn, np.zeros(len(free)), args=(free,), loss="soft_l1", f_scale=2.0, x_scale=[0.01] * 3 + [10, 10, 0.01] + [0.5] * 3 + [0.5] if len(free) == 10 else 1.0)
    rms = float(1.4826 * np.median(np.abs(sol.fun)))
    pf = np.zeros(10); pf[free] = sol.x
    print(f"{label:28s}: robust rms {rms:.3f} px; params " + ", ".join(f"{n}={v:+.3f}" for n, v in zip(["r1", "r2", "r3", "dbcz", "dbcy", "dlsd", "tx", "ty", "tz", "wedge"], pf)), flush=True)
    if label == "+ tilts + wedge":
        pr, pc = predict(pf, hkl[use], np.radians(om_all[use]))
        u = sp[use].assign(dr=sp.row.to_numpy()[use] - pr, dc=sp.col.to_numpy()[use] - pc)
        u = u[np.isfinite(u.dr)]
        u["win"] = (u.frame // 100) * 100
        mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
        print(u.groupby("win").agg(n=("dr", "size"), dr=("dr", "median"), dc=("dc", "median"), sr=("dr", mad), sc=("dc", mad)).round(2).to_string())
