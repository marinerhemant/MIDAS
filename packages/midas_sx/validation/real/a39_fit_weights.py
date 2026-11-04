"""A39 B2 sensitivity to the fit weights (POST HOC; registration says only 'scale and overall B fitted'): sigma-weighted (as run in a39_merge.py), unweighted, and relative (1/F2) weights, on a merged.csv."""
import os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd, gemmi, torch
from scipy.optimize import least_squares
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.refine import EIGHT_PI2, RefineModel
out, cif = sys.argv[1], sys.argv[2]
ss = gemmi.read_small_structure(cif); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90)
cr = Crystal(lat, SpaceGroup.from_hm("P 21 21 21"), [Atom(s.element.name, (s.fract.x, s.fract.y, s.fract.z), s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in ss.sites], "a")
m = pd.read_csv(os.path.join(out, "merged.csv")); hkl = m[["h", "k", "l"]].to_numpy(int)
mm = RefineModel(cr); Fc2 = mm.make_f2(hkl)(torch.tensor(mm.x0())).detach().numpy()
Fo2, sig, n = m.F2.to_numpy(), m.sigF2.to_numpy(), m.n.to_numpy()
s2 = np.array([1 / (2 * lat.d_spacing(*map(int, h))) ** 2 for h in hkl]); sel = n >= 2
top = np.zeros(len(hkl), bool); top[np.argsort(-Fc2)[:100]] = True
for name, w in (("sigma-weighted", 1 / sig), ("unweighted", np.ones_like(sig)), ("relative 1/Fo2", 1 / np.maximum(Fo2, 1.0))):
    f = lambda p: w[sel] * (Fo2[sel] - p[0] * np.exp(-2 * p[1] * s2[sel]) * Fc2[sel])
    p = least_squares(f, [Fo2[sel].sum() / Fc2[sel].sum(), 0.0]).x; cal = p[0] * np.exp(-2 * p[1] * s2) * Fc2
    R = lambda s: np.abs(Fo2[s] - cal[s]).sum() / np.abs(Fo2[s]).sum()
    print(f"{name:16s} B={p[1]:+.3f} k={p[0]:.2f} R_F2 all n>=2={R(sel):.3f} top100(n>=2)={R(top & sel):.3f} n_top={int((top & sel).sum())}")
