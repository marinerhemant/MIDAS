"""A39-FLOOR diagnostic (POST HOC, reference-free): is the L-dependent floor caused by evaluating the Lorentz factor at the CALCULATED crossing? Recompute L at the MEASURED spot direction (row, col of the blob) and compare the floor by L bin.
python -I a39_diag_leval.py <out dir> <step1 dir> <cif> <runs>"""
import json, os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx, os.path.dirname(os.path.abspath(__file__))] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd, gemmi
from dataclasses import replace
from midas_hkls import Lattice, SpaceGroup, corrections as C
from midas_sx.conventions import Convention
from midas_sx.geometry import spot_directions
from midas_sx.merge import scale_groups
from a39_geom import swing_geometry
out, step1, cif, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
sg = SpaceGroup.from_hm("P 21 21 21"); ss = gemmi.read_small_structure(cif); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90)
H = json.load(open(os.path.join(step1, "run02", "info.json")))["header"]; g0, scan = swing_geometry(H, Convention(0, -1), "cols", -1, -1)
parts = []
for r in runs:
    red = json.load(open(os.path.join(out, r, "reduction.json"))); tx, ty, tz = red["tilts_deg"]; rc = red["centre_rc"]
    g = replace(g0, tx_deg=tx, ty_deg=ty, tz_deg=tz, wedge_deg=red["wedge_deg"], lsd_um=red["lsd_mm"] * 1000.0, bcz_px=rc[0], bcy_px=rc[1])
    o = pd.read_pickle(os.path.join(out, r, "obs.pkl")).assign(run=r)
    s, _, _ = spot_directions(o.row.to_numpy(), o.col.to_numpy(), g); o["Lm"] = C.rotation_lorentz(s, min_abs=1e-3); parts.append(o)
O = pd.concat(parts, ignore_index=True); O = O[np.isfinite(O.Lm) & (O.Lm > 0)]
print("L (calculated crossing) vs L (measured spot): median ratio", round(float((O.Lm / O.L).median()), 3), " robust sd of log ratio", round(float(1.4826 * np.median(np.abs(np.log(O.Lm / O.L) - np.median(np.log(O.Lm / O.L))))), 3))
def floor_table(O, Lcol, tag):
    d = O.copy(); d["F2"] = d.I / (d[Lcol] * d.PT); d["sigF2"] = d.sigI / (d[Lcol] * d.PT)
    d, _ = scale_groups(d, sg); d["u"] = [tuple(sg.asu_representative(int(a), int(b), int(c))) for a, b, c in d[["h", "k", "l"]].to_numpy()]
    g = d.groupby("u"); d["n"] = g.F2.transform("size"); d = d[(d.n >= 2) & (d.F2 > 0)].copy(); s = g.F2.transform("sum").loc[d.index]; d["loo"] = (s - d.F2) / (d.n - 1); d = d[d.loo > 0].copy()
    d["dev"] = np.log(d.F2 / d.loo); d["ps"] = (d.sigF2 / d.F2) ** 2 * (1 + 1 / (d.n - 1))
    def fl(x): v = (1.4826 * np.median(np.abs(x.dev - np.median(x.dev)))) ** 2; return pd.Series(dict(n=len(x), f=np.sqrt(max(0, v - np.median(x.ps)) / 2)))
    t = d.groupby(pd.cut(d[Lcol], [0, 1.5, 2, 3, 5, 1e3]), observed=True).apply(fl)
    print(f"\n-- floor f by L bin, L = {tag}: overall f={fl(d).f:.3f} (n {len(d)})"); print(t.round(3).T.to_string())
floor_table(O, "L", "calculated crossing (as run)"); floor_table(O, "Lm", "measured spot direction")
