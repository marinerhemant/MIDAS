"""A39-FLOOR step 1: the worst-fitting merged reflections (|Fo2 - k Fc2| / sig_ext-inflated) and their individual observations.  python -I a39_diag_rejected.py <out dir> <cif> <runs>"""
import os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd, gemmi, torch
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.merge import scale_groups
from midas_sx.refine import EIGHT_PI2, RefineModel
out, cif, runs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
sg = SpaceGroup.from_hm("P 21 21 21"); ss = gemmi.read_small_structure(cif); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90)
cr = Crystal(lat, sg, [Atom(s.element.name, (s.fract.x, s.fract.y, s.fract.z), s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in ss.sites], "a")
m = pd.read_csv(os.path.join(out, "merged.csv")); hkl = m[["h", "k", "l"]].to_numpy(int)
mm = RefineModel(cr); m["Fc2"] = mm.make_f2(hkl)(torch.tensor(mm.x0())).detach().numpy()
k = (m.F2 * m.Fc2 / m.sigF2 ** 2).sum() / (m.Fc2 ** 2 / m.sigF2 ** 2).sum(); m["ratio"] = m.F2 / (k * m.Fc2); m["lr"] = np.log(m.ratio.clip(1e-4))
bad = m[(m.lr.abs() > 1.0) & (m.Fc2 > 50)].sort_values("lr")
print(f"scale k={k:.2f}; {len(m)} merged; |log ratio| > 1 and Fc2 > 50: {len(bad)}")
print(bad[["h", "k", "l", "n", "F2", "sigF2", "Fc2", "ratio"]].round(2).to_string(index=False))
o = pd.concat([pd.read_pickle(os.path.join(out, r, "obs.pkl")).assign(run=r) for r in runs], ignore_index=True)
o, _ = scale_groups(o, sg); o["u"] = [tuple(sg.asu_representative(int(a), int(b), int(c))) for a, b, c in o[["h", "k", "l"]].to_numpy()]
cols = ["run", "h", "k", "l", "frame", "row", "col", "tth", "L", "PT", "I", "sigI", "bg", "n_box", "blob_sum", "hkl_residual", "F2"]
for _, r in bad.head(12).iterrows():
    u = tuple(sg.asu_representative(int(r.h), int(r.k), int(r.l))); s = o[o.u == u]
    print(f"\n== {u}  F2o {r.F2:.1f} F2c*k {k * r.Fc2:.1f} ratio {r.ratio:.3g}"); print(s[cols].round(2).to_string(index=False))
