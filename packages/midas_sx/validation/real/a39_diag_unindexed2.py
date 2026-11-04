"""A39 diagnostic 2: strongest N blobs of the FIRST NF frames (the blind-index set), matched to predictions made with the refined geometry and U; matched vs unmatched blobs compared on intensity and shape.
Reads diag_unindexed_top2000.csv-style columns from the spots file itself.   python -I a39_diag_unindexed2.py <out dir> <step1 dir> <cif> <rNN> [NF=100]"""
import json, os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx, os.path.dirname(os.path.abspath(__file__))] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd, gemmi
from dataclasses import replace
from scipy.spatial import cKDTree
from midas_hkls import Lattice
from midas_sx.conventions import Convention
from midas_sx.geometry import spot_directions
from midas_sx.predict import enumerate_hkl, predicted_crossings
from a39_geom import swing_geometry
d, step1, cif, run = sys.argv[1:5]; NF = int(sys.argv[5]) if len(sys.argv) > 5 else 100
H = json.load(open(os.path.join(step1, "info.json")))["header"]; red = json.load(open(os.path.join(d, run, "reduction.json")))
U = np.load(os.path.join(d, run, "U.npy")); sp = pd.read_csv(os.path.join(d, run, "spots_blind.csv"))
ss = gemmi.read_small_structure(cif); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90)
g0, scan = swing_geometry(H, Convention(0, -1), "cols", -1, -1); tx, ty, tz = red["tilts_deg"]; rc = red["centre_rc"]
geom = replace(g0, tx_deg=tx, ty_deg=ty, tz_deg=tz, wedge_deg=red["wedge_deg"], lsd_um=red["lsd_mm"] * 1000.0, bcz_px=rc[0], bcy_px=rc[1])
pred = predicted_crossings(U, lat, enumerate_hkl(lat, 0.60), geom, scan, margin_px=3.0)
tree = cKDTree(np.c_[pred.frame.to_numpy() * 4.0, pred.row.to_numpy(), pred.col.to_numpy()])
b = sp[sp.frame < NF].sort_values("integrated", ascending=False).reset_index(drop=True)
b["dist"] = tree.query(np.c_[b.frame.to_numpy() * 4.0, b.row.to_numpy(), b.col.to_numpy()])[0]
_, tth, _ = spot_directions(b.row.to_numpy(), b.col.to_numpy(), geom); b["tth"] = tth
print(f"{run}: {len(b)} blobs in the first {NF} frames; predictions in this range: {int((pred.frame < NF).sum())}")
for N in (50, 100, 200, 300, 400, 600, 1000, len(b)):
    x = b.head(N); print(f"strongest {N:5d}: matched <4 {np.mean(x.dist < 4):.3f}   <8 {np.mean(x.dist < 8):.3f}   (integrated >= {x.integrated.min():.0f})")
t = b.head(600); m, u = t[t.dist < 4], t[t.dist >= 4]
print(f"\nstrongest 600 of the first {NF} frames: {len(m)} matched, {len(u)} unmatched")
cols = ["integrated", "peak_counts", "volume_vox", "n_frames", "length_px", "width_px", "aspect", "omega_width_frames", "tth"]
print(pd.DataFrame({"matched_med": m[cols].median(), "unmatched_med": u[cols].median(), "matched_p10": m[cols].quantile(.1), "unmatched_p10": u[cols].quantile(.1), "matched_p90": m[cols].quantile(.9), "unmatched_p90": u[cols].quantile(.9)}).round(2).to_string())
print("\nunmatched: fraction with aspect > 3 :", round((u.aspect > 3).mean(), 3), " matched:", round((m.aspect > 3).mean(), 3))
print("unmatched: fraction with n_frames <= 3 :", round((u.n_frames <= 3).mean(), 3), " matched:", round((m.n_frames <= 3).mean(), 3))
print("2theta (deg) histogram unmatched vs matched, 1 deg bins up to 45:"); bins = np.arange(0, 46, 3)
print(pd.DataFrame(dict(unmatched=np.histogram(u.tth, bins)[0], matched=np.histogram(m.tth, bins)[0]), index=bins[:-1]).T.to_string())
# location of unmatched
print("\nunmatched row/col percentiles:", np.round(np.percentile(u.row, [5, 50, 95])), np.round(np.percentile(u.col, [5, 50, 95])), "| matched:", np.round(np.percentile(m.row, [5, 50, 95])), np.round(np.percentile(m.col, [5, 50, 95])))
# unmatched spots: do they index with a SECOND orientation (another grain/twin)? try blind ab_initio on unmatched only
import torch
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
q = pixel_to_qlab(u.row.to_numpy(), u.col.to_numpy(), geom, device="cpu", dtype=torch.float64)
om = torch.as_tensor(np.radians(scan.omega_of(u.frame.to_numpy())), dtype=torch.float64)
res = index_ab_initio(qlab_to_qsample(q, om, float(np.radians(geom.wedge_deg))).numpy(), two_pi=True, tol=0.15, min_reflections=20)
print(f"\nblind ab-initio on the {len(u)} UNMATCHED strong blobs only: success {res.success}; indexed {int(res.n_indexed) if res.success else 0}; cell {np.round(res.cell[:3],3) if res.success else None}")
