"""A39-FLOOR reference readout (model-free): R_F2 of a merged F2 table against the reflections embedded in the COD CIF (_shelx_hkl_file, an independent measurement at 0.2489 A).
Registered definition: reference merged to the unique set of P 21 21 21 / mmm with Friedel mates merged (mean of equivalents, plain average, F2 = the CIF's intensity column), common reflections with n >= 2 in OUR table,
one scale only, UNWEIGHTED: R = sum|Fo2 - k Fref2| / sum Fo2 with k the scale that MINIMISES R (L1-optimal; Amendment 2 of PREREG_a39_floor.md, replacing the least-squares scale, which the strong reflections pull up). Also reported by resolution shell and for the 100 strongest reference reflections.
python -I a39_ref_hkl.py <merged.csv> <cif>"""
import os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd, gemmi
from midas_hkls import Lattice, SpaceGroup
csv, cif = sys.argv[1], sys.argv[2]
sg = SpaceGroup.from_hm("P 21 21 21"); ss = gemmi.read_small_structure(cif); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90)
txt = open(cif).read(); i = txt.index("_shelx_hkl_file"); a = txt.index(";", i) + 1; b = txt.index(";", a)
rows = []
for ln in txt[a:b].splitlines():
    p = ln.split()
    if len(p) >= 5:
        try: rows.append((int(p[0]), int(p[1]), int(p[2]), float(p[3]), float(p[4])))
        except ValueError: pass
R = pd.DataFrame(rows, columns=["h", "k", "l", "Fref2", "sig"]); R = R[(R.h != 0) | (R.k != 0) | (R.l != 0)]
R["u"] = [tuple(sg.asu_representative(int(x), int(y), int(z))) for x, y, z in R[["h", "k", "l"]].to_numpy()]
Ru = R.groupby("u").agg(Fref2=("Fref2", "mean"), nref=("Fref2", "size")).reset_index()
M = pd.read_csv(csv); M["u"] = [tuple(sg.asu_representative(int(x), int(y), int(z))) for x, y, z in M[["h", "k", "l"]].to_numpy()]
J = M.merge(Ru, on="u", how="inner"); J = J[J.n >= 2]
print(f"reference: {len(R)} lines -> {len(Ru)} unique; ours {len(M)} unique ({int((M.n >= 2).sum())} with n >= 2); common with n >= 2: {len(J)}")
from scipy.optimize import minimize_scalar
def rf(d):
    k0 = d.F2.sum() / d.Fref2.sum(); k = minimize_scalar(lambda kk: float((d.F2 - kk * d.Fref2).abs().sum()), bounds=(0.01 * k0, 10 * k0), method="bounded").x
    return float((d.F2 - k * d.Fref2).abs().sum() / d.F2.abs().sum()), float(k)
r, k = rf(J); print(f"R_F2 (all common n>=2, scale only, unweighted) = {r:.4f}  scale {k:.4g}")
J["d"] = [lat.d_spacing(*map(int, h)) for h in J[["h", "k", "l"]].to_numpy()]
top = J.sort_values("Fref2", ascending=False).head(100); print(f"R_F2 (100 strongest reference reflections) = {rf(top)[0]:.4f}")
print(J.groupby(pd.cut(J.d, [0.6, 0.75, 0.9, 1.1, 1.5, 2.5, 20]), observed=True).apply(lambda g: pd.Series(dict(n=len(g), R=rf(g)[0], med_ratio=float((g.F2 / (k * g.Fref2)).median())))).round(3).to_string())
# scale conventions (reported for the BASELINE to document which one reproduces earlier numbers; the registered one is the least-squares scale above)
from scipy.optimize import minimize_scalar
kl2 = (J.F2 * J.Fref2).sum() / (J.Fref2 ** 2).sum(); ksum = J.F2.sum() / J.Fref2.sum()
kl1 = minimize_scalar(lambda kk: float((J.F2 - kk * J.Fref2).abs().sum()), bounds=(0.01 * ksum, 10 * ksum), method="bounded").x
kw = ((J.F2 * J.Fref2 / J.F2.clip(lower=1)) .sum() / (J.Fref2 ** 2 / J.F2.clip(lower=1)).sum())
for name, kk in (("LSQ (not registered)", kl2), ("ratio of sums", ksum), ("L1-optimal (registered)", kl1), ("1/Fo2-weighted LSQ", kw)):
    print(f"scale convention {name:20s} k={kk:.4g}  R_F2 = {float((J.F2 - kk * J.Fref2).abs().sum() / J.F2.abs().sum()):.4f}")
