import os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd, gemmi, torch
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.refine import EIGHT_PI2, RefineModel
out, cif = sys.argv[1], sys.argv[2]
ss = gemmi.read_small_structure(cif); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90)
cr = Crystal(lat, SpaceGroup.from_hm("P 21 21 21"), [Atom(s.element.name, (s.fract.x, s.fract.y, s.fract.z), s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in ss.sites], "a")
m = pd.read_csv(os.path.join(out, "merged.csv"))
hkl = m[["h", "k", "l"]].to_numpy(int)
mm = RefineModel(cr); Fc2 = mm.make_f2(hkl)(torch.tensor(mm.x0())).detach().numpy()
m["Fc2"] = Fc2; m["s2"] = [1 / (2 * lat.d_spacing(*map(int, h))) ** 2 for h in hkl]; m["d"] = [lat.d_spacing(*map(int, h)) for h in hkl]
k = (m.F2 * m.Fc2).sum() / (m.Fc2 ** 2).sum(); m["ratio"] = m.F2 / (k * m.Fc2)
print("resolution shell: n, median Fo2/(k Fc2), R_F2")
m["bin"] = pd.cut(m.d, [0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0, 20])
print(m.groupby("bin", observed=True).apply(lambda g: pd.Series(dict(n=len(g), med_ratio=g.ratio.median(), R=np.abs(g.F2 - k * g.Fc2).sum() / g.F2.abs().sum(), n_ge2=(g.n >= 2).sum()))).round(3).to_string())
t = m.sort_values("Fc2", ascending=False).head(15)
print(t[["h", "k", "l", "n", "F2", "sigF2", "Fc2", "ratio", "d"]].round(2).to_string(index=False))
print("fraction of merged reflections with F2 < 0:", (m.F2 < 0).mean().round(3), "; with F2 < 3 sigma:", (m.F2 < 3 * m.sigF2).mean().round(3))
