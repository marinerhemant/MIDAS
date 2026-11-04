"""A39 diagnostic: lattice from frames < 450 (blind), orientation refined on those, then the residuals (observed - predicted position) of ALL assigned blobs per 50-frame window, to see
where and how the later frames depart from a rigid single-crystal model."""
import json, os, sys
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"), os.path.dirname(os.path.abspath(__file__))]
_root = os.path.dirname(os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"))
sys.path[:0] = [os.path.join(_root, p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(_root, p))]
import numpy as np, pandas as pd, torch, gemmi
from midas_hkls import Lattice
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.assign import assign_hkl
from midas_sx.conventions import Convention, transform_rc
from midas_sx.geometry import spot_directions
from midas_sx.orient import index_and_refine, predict_positions
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
d = sp[sp.frame < 450].sort_values("integrated", ascending=False).head(600)
q = pixel_to_qlab(d.row.to_numpy(), d.col.to_numpy(), g0, device="cpu", dtype=torch.float64)
om = torch.as_tensor(np.radians(scan.omega_of(d.frame.to_numpy())), dtype=torch.float64)
res = index_ab_initio(qlab_to_qsample(q, om, 0.0).numpy(), two_pi=True, tol=0.15, min_reflections=30)
U0, P, err = match_setting(np.asarray(res.UB), lat)
sub = sp[sp.frame < 450]
o = index_and_refine(sub, lat, U0, g0, scan)
print(f"refined on frames<450: converged {o.converged} ({o.reason}) n_used {o.n_used}/{o.n_spots} rms {o.rms_px:.3f} px {o.rms_frame:.3f} fr; centre {o.geom.bcz_px:.2f},{o.geom.bcy_px:.2f} Lsd {o.geom.lsd_um/1000:.3f}", flush=True)
s, tth, _ = spot_directions(sp.row.to_numpy(), sp.col.to_numpy(), o.geom)
hkl, hf, resid = assign_hkl(s, scan.omega_of(sp.frame.to_numpy()), lat, o.U, o.geom.wavelength_A)
sp = sp.assign(h=hkl[:, 0], k=hkl[:, 1], l=hkl[:, 2], resid=resid, tth=tth)
use = (resid < 0.25) & (tth > 2) & np.any(hkl != 0, axis=1)
fr, ro, co, ok = predict_positions(o.U, lat, hkl[use], scan.omega_of(sp.frame.to_numpy()[use]), o.geom, scan)
u = sp[use].assign(pf=fr, pr=ro, pc=co)
u = u[np.isfinite(u.pf)]
u["dr"], u["dc"], u["df"] = u.row - u.pr, u.col - u.pc, u.frame - u.pf
u["win"] = (u.frame // 50) * 50
t = u.groupby("win").agg(n=("dr", "size"), dr=("dr", "median"), dc=("dc", "median"), df=("df", "median"), sr=("dr", lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))), sc=("dc", lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))), sf=("df", lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))))
print("assigned within 0.25 per window (all blobs):", sp.assign(win=(sp.frame // 50) * 50, ok=use).groupby("win").ok.mean().round(2).to_dict())
print(t.round(2).to_string())
