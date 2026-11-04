"""PREREGISTER A38, step 2c (DIAGNOSTIC): free fit of the detector geometry (beam centre, distance, tx, ty, tz) to the vendor's REF1K reflections of run 01 (hkl known), for each of the 16
conventions and several starts; Kabsch inside the residual. The convention that reaches a small rms (1/d) with plausible parameters is the file's geometry."""
import json, os, sys
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")]
import numpy as np, pandas as pd, torch
from dataclasses import replace
from scipy.optimize import least_squares
from midas_hkls import Lattice
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.assign import reciprocal_basis
from midas_sx.conventions import candidate_conventions, transform_rc
from midas_sx.geometry import Scan, make_geometry

p4p, out = sys.argv[1], sys.argv[2]
H = json.load(open(os.path.join(out, "info.json")))["header"]
nr, nc = H["shape"]; ex = H["extra"]
foot = (ex["centre_xy_px"][1], ex["centre_xy_px"][0])
ref = pd.DataFrame([[float(x) for x in l.split()[2:11]] for l in open(p4p) if l.startswith("REF1K")], columns=["h", "k", "l", "tth", "omega", "phi", "chi", "x", "y"])
sel = ref[(ref.phi.sub(ex["phi_deg"]).abs() < 0.01) & (ref.chi.sub(ex["chi_deg"] - 360).abs() < 0.01)].reset_index(drop=True)
Bm = reciprocal_basis(Lattice(7.4962, 8.5211, 5.4695, 90, 90, 90))
qB = (Bm @ sel[["h", "k", "l"]].to_numpy(float).T).T


def q_of(conv, p):
    r, c = transform_rc(sel.y.to_numpy(), sel.x.to_numpy(), nr, nc, conv.dihedral)
    r0, c0 = (float(v[0]) for v in transform_rc(np.array([foot[0]]), np.array([foot[1]]), nr, nc, conv.dihedral))
    shape = (nr, nc) if conv.dihedral < 4 else (nc, nr)
    scan = Scan(conv.omega_sign * H["angle_start"], conv.omega_sign * H["angle_step"], H["n_frames"])
    g = make_geometry(wavelength_A=H["wavelength_A"], dist_mm=H["distance_mm"] * np.exp(p[2]), pixel_mm=H["pixel_mm"], centre_rc=(r0 + p[0], c0 + p[1]), shape=shape, scan=scan)
    g = replace(g, tx_deg=float(p[3]), ty_deg=float(p[4]), tz_deg=float(p[5]))
    qlab = pixel_to_qlab(r, c, g, device="cpu", dtype=torch.float64)
    om = torch.as_tensor(np.radians(conv.omega_sign * sel.omega.to_numpy()), dtype=torch.float64)
    return qlab_to_qsample(qlab, om, 0.0).numpy() / (2 * np.pi)


def resid(p, conv):
    q = q_of(conv, p)
    u, s, vt = np.linalg.svd(qB.T @ q)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    U = vt.T @ np.diag([1, 1, d]) @ u.T
    return (q - (U @ qB.T).T).ravel()


print(f"{len(sel)} reference reflections of run 01", flush=True)
rows = []
for conv in candidate_conventions():
    best = None
    for s0 in (0.0, np.log(3.8001 / 2.7961)):
        for tz0 in (-9.684, 0.0, 9.684):
            for ty0 in (0.0,):
                for dc in (-35.0, 0.0, 35.0):
                    p0 = np.array([0.0, dc, s0, 0.0, ty0, tz0])
                    try:
                        sol = least_squares(resid, p0, args=(conv,), x_scale=[10, 10, 0.1, 2, 2, 2], max_nfev=60)
                    except Exception:
                        continue
                    rms = float(np.sqrt((sol.fun ** 2).reshape(-1, 3).sum(1).mean()))
                    if best is None or rms < best[0]:
                        best = (rms, sol.x)
    rows.append(dict(convention=conv.name, rms=best[0], dr=best[1][0], dc=best[1][1], Dscale=float(np.exp(best[1][2])), D_mm=H["distance_mm"] * float(np.exp(best[1][2])),
                     tx=best[1][3], ty=best[1][4], tz=best[1][5]))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows).sort_values("rms")
print(df.round(4).to_string(index=False))
df.to_csv(os.path.join(out, "refpoints_fit.csv"), index=False)
