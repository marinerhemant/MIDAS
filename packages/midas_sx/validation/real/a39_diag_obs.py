"""A39 diagnostic: per-observation F2/(k_run Fc2) against omega, detector position, tth, intensity, hkl_residual (plane given by the out dir)."""
import os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd, gemmi, torch
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.refine import EIGHT_PI2, RefineModel
out, cif, runs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
ss = gemmi.read_small_structure(cif); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90)
cr = Crystal(lat, SpaceGroup.from_hm("P 21 21 21"), [Atom(s.element.name, (s.fract.x, s.fract.y, s.fract.z), s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in ss.sites], "a")
o = pd.concat([pd.read_pickle(os.path.join(out, r, "obs.pkl")).assign(run=r) for r in runs], ignore_index=True)
mm = RefineModel(cr); o["Fc2"] = mm.make_f2(o[["h", "k", "l"]].to_numpy(int))(torch.tensor(mm.x0())).detach().numpy()
o = o[o.Fc2 > 0]
o["om"] = ((o.omega + 180) % 360) - 180
for r, g in o.groupby("run"):
    k = (g.F2 * g.Fc2).sum() / (g.Fc2 ** 2).sum(); o.loc[g.index, "ratio"] = g.F2 / (k * g.Fc2)
o["lr"] = np.log(o.ratio.clip(1e-3))
def tab(col, bins, sub=o):
    b = pd.cut(sub[col], bins)
    print(f"-- {col}"); print(sub.groupby(b, observed=True).lr.agg(n="size", med="median", mad=lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))).round(3).to_string())
print("runs:", o.groupby("run").lr.agg(n="size", med="median", mad=lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))).round(3).to_string())
tab("om", np.arange(-160, 20, 20)); tab("tth", [0, 10, 15, 20, 25, 30, 35, 45]); tab("eta", np.arange(-180, 181, 45))
tab("row", np.linspace(0, 1679, 8)); tab("col", np.linspace(0, 1475, 8))
tab("I", [-1, 50, 200, 1000, 5000, 2e4, 1e5, 1e9]); tab("hkl_residual", [0, 0.003, 0.006, 0.01, 0.02, 1]); tab("n_box", [0, 3000, 8000, 12000, 16000, 1e9])
tab("L", [0, 1, 2, 4, 8, 1e3]); tab("PT", [0, .5, .7, .8, .9, 1.01]); tab("bg", [-1, .1, .3, .6, 1, 3, 1e3])
