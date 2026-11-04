"""A39 diagnostic 2: pool the reflections assigned by the SEQUENTIAL-WINDOW tracking (clean, ~1 px rms) of a39_diag_creep and fit ONE rigid model to all of them: U, centre, distance, detector tilts
and the omega-axis tilt about Y (MIDAS wedge) and about X. Does a single orientation + an axis/detector misalignment of ~0.3 deg explain the drift of the per-window orientation?"""
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
from midas_sx.orient import refine_orientation, _newton_pixels
from midas_sx.setting import match_setting
from a39_geom import swing_geometry
out, cif = sys.argv[1], sys.argv[2]
sp = pd.read_csv(os.path.join(out, "spots.csv"))
H = json.load(open(os.path.join(out, "info.json")))["header"]; nr, nc = H["shape"]
conv = Convention(0, -1)
ss = gemmi.read_small_structure(cif); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90); Bm = reciprocal_basis(lat)
r, c = transform_rc(sp.row.to_numpy(), sp.col.to_numpy(), nr, nc, conv.dihedral)
sp = sp.assign(row=r, col=c)
g0, scan = swing_geometry(H, conv, "cols", -1, -1)
d = sp[sp.frame < 100].sort_values("integrated", ascending=False).head(600)
q = pixel_to_qlab(d.row.to_numpy(), d.col.to_numpy(), g0, device="cpu", dtype=torch.float64)
om = torch.as_tensor(np.radians(scan.omega_of(d.frame.to_numpy())), dtype=torch.float64)
res = index_ab_initio(qlab_to_qsample(q, om, 0.0).numpy(), two_pi=True, tol=0.15, min_reflections=30)
U, P, err = match_setting(np.asarray(res.UB), lat)
U0 = U.copy(); geom = g0; pool = []
for a in range(0, 850, 100):
    w = sp[(sp.frame >= a) & (sp.frame < a + 100)].sort_values("integrated", ascending=False).head(1500)
    for tol in (0.30, 0.25, 0.2, 0.15):
        s, tth, _ = spot_directions(w.row.to_numpy(), w.col.to_numpy(), geom)
        hkl, hf, resid = assign_hkl(s, scan.omega_of(w.frame.to_numpy()), lat, U, geom.wavelength_A)
        use = (resid < tol) & (tth > 2) & np.any(hkl != 0, axis=1)
        U, geom_new, info = refine_orientation(U, geom, lat, scan, hkl[use], w.frame.to_numpy()[use], w.row.to_numpy()[use], w.col.to_numpy()[use], refine_geometry=(a == 0 and tol == 0.30))
        if a == 0 and tol == 0.30:
            geom = geom_new
    pool.append(w[use].assign(h=hkl[use, 0], k=hkl[use, 1], l=hkl[use, 2]))
P_ = pd.concat(pool, ignore_index=True)
print("pooled clean assignments:", len(P_), flush=True)
H_ = P_[["h", "k", "l"]].to_numpy(float); rr = P_.row.to_numpy(); cc = P_.col.to_numpy(); oo = np.radians(scan.omega_of(P_.frame.to_numpy()))
base = geom
NAMES = ["r1", "r2", "r3", "dbcz", "dbcy", "dlsd", "tx", "ty", "tz", "wedgeY", "wedgeX"]
XS = np.array([0.01, 0.01, 0.01, 5.0, 5.0, 0.002, 0.3, 0.3, 0.3, 0.3, 0.3])


def predict(p):
    Ur = Rot.from_rotvec(p[:3]).as_matrix() @ U0
    qs = 2 * np.pi * (Ur @ Bm @ H_.T).T
    c_, s_ = np.cos(oo), np.sin(oo)
    ql = np.stack([c_ * qs[:, 0] - s_ * qs[:, 1], s_ * qs[:, 0] + c_ * qs[:, 1], qs[:, 2]], 1)
    Wy, Wx = math.radians(p[9]), math.radians(p[10])
    cy, sy, cx, sx = math.cos(Wy), math.sin(Wy), math.cos(Wx), math.sin(Wx)
    ql = np.stack([cy * ql[:, 0] - sy * ql[:, 2], ql[:, 1], sy * ql[:, 0] + cy * ql[:, 2]], 1)          # R_y(-W): the MIDAS wedge
    ql = np.stack([ql[:, 0], cx * ql[:, 1] - sx * ql[:, 2], sx * ql[:, 1] + cx * ql[:, 2]], 1)          # R_x: sideways tilt of the axis
    g = replace(base, bcz_px=base.bcz_px + p[3], bcy_px=base.bcy_px + p[4], lsd_um=base.lsd_um * (1 + p[5]), tx_deg=base.tx_deg + p[6], ty_deg=base.ty_deg + p[7], tz_deg=base.tz_deg + p[8])
    return _newton_pixels(ql, g)


def fun(x, free):
    p = np.zeros(11); p[free] = x
    pr, pc = predict(p)
    return np.concatenate([np.where(np.isfinite(pr), pr - rr, 1e3), np.where(np.isfinite(pc), pc - cc, 1e3)])


for label, free in (("U + centre + distance", [0, 1, 2, 3, 4, 5]), ("+ wedge Y (MIDAS)", [0, 1, 2, 3, 4, 5, 9]), ("+ wedge X (sideways)", [0, 1, 2, 3, 4, 5, 10]), ("+ both axis tilts", [0, 1, 2, 3, 4, 5, 9, 10]),
                    ("+ detector tilts", [0, 1, 2, 3, 4, 5, 6, 7, 8]), ("everything", list(range(11)))):
    sol = least_squares(fun, np.zeros(len(free)), args=(free,), loss="soft_l1", f_scale=2.0, x_scale=XS[free])
    p = np.zeros(11); p[free] = sol.x
    print(f"{label:24s}: robust rms {1.4826*np.median(np.abs(sol.fun)):.3f} px; " + ", ".join(f"{NAMES[i]}={p[i]:+.3f}" for i in free), flush=True)
