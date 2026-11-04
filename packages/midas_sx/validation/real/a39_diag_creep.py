"""A39 diagnostic: does the CRYSTAL ORIENTATION drift during the scan? Sequential 100-frame windows, each refined (U only, geometry from the first window) starting from the previous window's U;
report the misorientation of each window against the first, the robust rms, and the number of assigned strong blobs."""
import json, os, sys
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"), os.path.dirname(os.path.abspath(__file__))]
_root = os.path.dirname(os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"))
sys.path[:0] = [os.path.join(_root, p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(_root, p))]
import numpy as np, pandas as pd, torch, gemmi
from scipy.spatial.transform import Rotation as Rot
from midas_hkls import Lattice
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.assign import assign_hkl
from midas_sx.conventions import Convention, transform_rc
from midas_sx.geometry import spot_directions
from midas_sx.orient import refine_orientation
from midas_sx.setting import match_setting
from a39_geom import swing_geometry
out, cif = sys.argv[1], sys.argv[2]
sp = pd.read_csv(os.path.join(out, "spots.csv"))
H = json.load(open(os.path.join(out, "info.json")))["header"]; nr, nc = H["shape"]
conv = Convention(0, -1)
ss = gemmi.read_small_structure(cif); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90)
r, c = transform_rc(sp.row.to_numpy(), sp.col.to_numpy(), nr, nc, conv.dihedral)
sp = sp.assign(row=r, col=c)
g0, scan = swing_geometry(H, conv, "cols", -1, -1)
d = sp[sp.frame < 100].sort_values("integrated", ascending=False).head(600)
q = pixel_to_qlab(d.row.to_numpy(), d.col.to_numpy(), g0, device="cpu", dtype=torch.float64)
om = torch.as_tensor(np.radians(scan.omega_of(d.frame.to_numpy())), dtype=torch.float64)
res = index_ab_initio(qlab_to_qsample(q, om, 0.0).numpy(), two_pi=True, tol=0.15, min_reflections=30)
U, P, err = match_setting(np.asarray(res.UB), lat)
U_first, geom = None, g0
rows = []
for a in range(0, 850, 100):
    w = sp[(sp.frame >= a) & (sp.frame < a + 100)]
    w = w.sort_values("integrated", ascending=False).head(1500)
    ok_fit = False
    for tol in (0.30, 0.25, 0.2, 0.15):
        s, tth, _ = spot_directions(w.row.to_numpy(), w.col.to_numpy(), geom)
        hkl, hf, resid = assign_hkl(s, scan.omega_of(w.frame.to_numpy()), lat, U, geom.wavelength_A)
        use = (resid < tol) & (tth > 2) & np.any(hkl != 0, axis=1)
        if use.sum() < 40:
            break
        try:
            U, geom_new, info = refine_orientation(U, geom, lat, scan, hkl[use], w.frame.to_numpy()[use], w.row.to_numpy()[use], w.col.to_numpy()[use], refine_geometry=(a == 0 and tol == 0.30))
        except Exception as e:
            print(a, "refine failed:", repr(e)[:100]); break
        if a == 0 and tol == 0.30:
            geom = geom_new
        ok_fit = True
    if U_first is None:
        U_first = U.copy()
    rv = np.degrees(Rot.from_matrix(U @ U_first.T).as_rotvec()); mis = np.linalg.norm(rv)
    rows.append(dict(window=a, n_blobs=len(w), n_assigned=int(use.sum()) if ok_fit else 0, rms_px=round(info["rms_px"], 2) if ok_fit else None, rms_frame=round(info["rms_frame"], 2) if ok_fit else None,
                     misorientation_deg=round(float(mis), 3), rotvec_sample_xyz_deg=np.round(rv, 3).tolist()))
    print(rows[-1], flush=True)
