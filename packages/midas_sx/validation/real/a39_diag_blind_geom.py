"""A39 diagnostic: blind ab-initio indexing (strongest 600 blobs of the first NF frames, tol 0.15, no cell supplied) as a function of the geometry used, to see which calibration term limits the indexed fraction.
Variants: header (swing_geometry start values), refined (all), and refined with one group put back to its header value.

    python -I a39_diag_blind_geom.py <out dir> <step1 dir> <rNN> [NF=100]
"""
import json, os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx, os.path.dirname(os.path.abspath(__file__))] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd, torch
from dataclasses import replace
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.conventions import Convention
from a39_geom import swing_geometry

d, step1, run = sys.argv[1:4]; NF = int(sys.argv[4]) if len(sys.argv) > 4 else 100
H = json.load(open(os.path.join(step1, "info.json")))["header"]; red = json.load(open(os.path.join(d, run, "reduction.json")))
sp = pd.read_csv(os.path.join(d, run, "spots_blind.csv")); sp = sp[sp.frame < NF].sort_values("integrated", ascending=False).head(600)
g0, scan = swing_geometry(H, Convention(0, -1), "cols", -1, -1)
tx, ty, tz = red["tilts_deg"]; rc = red["centre_rc"]
gR = replace(g0, tx_deg=tx, ty_deg=ty, tz_deg=tz, wedge_deg=red["wedge_deg"], lsd_um=red["lsd_mm"] * 1000.0, bcz_px=rc[0], bcy_px=rc[1])
V = {"header": g0, "refined (all)": gR,
     "refined, wedge->0": replace(gR, wedge_deg=0.0),
     "refined, tilts->header": replace(gR, tx_deg=g0.tx_deg, ty_deg=g0.ty_deg, tz_deg=g0.tz_deg),
     "refined, centre->header": replace(gR, bcz_px=g0.bcz_px, bcy_px=g0.bcy_px),
     "refined, Lsd->header": replace(gR, lsd_um=g0.lsd_um),
     "refined, ty->0": replace(gR, ty_deg=0.0),
     "header + refined wedge only": replace(g0, wedge_deg=gR.wedge_deg),
     "header + refined centre/Lsd": replace(g0, bcz_px=gR.bcz_px, bcy_px=gR.bcy_px, lsd_um=gR.lsd_um),
     "header + refined tilts": replace(g0, tx_deg=tx, ty_deg=ty, tz_deg=tz)}
print(f"{run}: header tilts {g0.tx_deg:+.3f} {g0.ty_deg:+.3f} {g0.tz_deg:+.3f} wedge {g0.wedge_deg:+.3f} centre ({g0.bcz_px:.2f}, {g0.bcy_px:.2f}) Lsd {g0.lsd_um/1000:.3f}; refined tilts {tx:+.3f} {ty:+.3f} {tz:+.3f} wedge {gR.wedge_deg:+.3f} centre ({rc[0]:.2f}, {rc[1]:.2f}) Lsd {gR.lsd_um/1000:.3f}")
for name, g in V.items():
    q = pixel_to_qlab(sp.row.to_numpy(), sp.col.to_numpy(), g, device="cpu", dtype=torch.float64)
    om = torch.as_tensor(np.radians(scan.omega_of(sp.frame.to_numpy())), dtype=torch.float64)
    # indexing ignores the wedge in qlab_to_qsample's second argument unless it is passed: pass it
    res = index_ab_initio(qlab_to_qsample(q, om, float(np.radians(g.wedge_deg))).numpy(), two_pi=True, tol=0.15, min_reflections=30)
    cell = np.round(res.cell[:3], 3) if res.success else None
    print(f"{name:32s} indexed {int(res.n_indexed) if res.success else 0:4d}/{len(sp)}  cell {cell}", flush=True)
