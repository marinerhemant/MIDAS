"""A39: Friedel-pair coverage. (1) per run, predicted crossings (refined geometry + U) whose Friedel mate (-h,-k,-l) is also predicted in the SAME run; (2) observed: reflections in obs.pkl whose mate is observed in the same run / in a different run.
python -I a39_diag_friedel.py <out dir> <step1 dir> <cif> <runs>"""
import json, os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx, os.path.dirname(os.path.abspath(__file__))] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd, gemmi
from dataclasses import replace
from midas_hkls import Lattice
from midas_sx.conventions import Convention
from midas_sx.predict import enumerate_hkl, predicted_crossings
from a39_geom import swing_geometry
d, step1, cif, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
ss = gemmi.read_small_structure(cif); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90)
H = json.load(open(os.path.join(step1, "run02", "info.json")))["header"]
g0, scan = swing_geometry(H, Convention(0, -1), "cols", -1, -1)
hk = enumerate_hkl(lat, 0.75)
obs = {}
for r in runs:
    red = json.load(open(os.path.join(d, r, "reduction.json"))); U = np.load(os.path.join(d, r, "U.npy")); tx, ty, tz = red["tilts_deg"]; rc = red["centre_rc"]
    g = replace(g0, tx_deg=tx, ty_deg=ty, tz_deg=tz, wedge_deg=red["wedge_deg"], lsd_um=red["lsd_mm"] * 1000.0, bcz_px=rc[0], bcy_px=rc[1])
    P = predicted_crossings(U, lat, hk, g, scan); S = set(map(tuple, P[["h", "k", "l"]].to_numpy(int)))
    mate = sum(1 for t in S if (-t[0], -t[1], -t[2]) in S)
    print(f"{r}: {len(P)} predicted crossings, {len(S)} distinct hkl; Friedel mate also predicted in the same run: {mate}")
    o = pd.read_pickle(os.path.join(d, r, "obs.pkl")); obs[r] = set(map(tuple, o[["h", "k", "l"]].to_numpy(int)))
    print(f"   observed {len(obs[r])} distinct hkl; mate observed in the same run: {sum(1 for t in obs[r] if (-t[0], -t[1], -t[2]) in obs[r])}")
allo = set().union(*obs.values())
both = sum(1 for t in allo if (-t[0], -t[1], -t[2]) in allo)
cross = sum(1 for t in allo if any(t in obs[a] and (-t[0], -t[1], -t[2]) in obs[b] for a in runs for b in runs if a != b))
print(f"all runs: {len(allo)} distinct signed hkl observed; {both} have the mate observed in some run; {cross} have it in a DIFFERENT run")
# omega separation of predicted mates: what scan length would be needed
