"""PREREGISTER A38 tier C: refine the non-H atoms of thiourea (H fixed at the CIF) against the merged F² of all runs, 10 starts displaced 0.05 A.

    python -I a38_refine.py <out dir> <cif> [merged csv name]
"""
import json, os, sys
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")]
import numpy as np, pandas as pd, gemmi
from dataclasses import replace
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine

out, cif = sys.argv[1], sys.argv[2]
mname = sys.argv[3] if len(sys.argv) > 3 else "merged.csv"
sg = SpaceGroup.from_hm("P 21 m a")
ss = gemmi.read_small_structure(cif)
truth = Crystal(Lattice(ss.cell.a, ss.cell.b, ss.cell.c, ss.cell.alpha, ss.cell.beta, ss.cell.gamma), sg,
                [Atom(s.element.name, (s.fract.x, s.fract.y, s.fract.z), s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in ss.sites], "thiourea130K")
fixed = tuple(a.label for a in truth.atoms if a.element == "H")
heavy = [a.label for a in truth.atoms if a.element != "H"]
m = pd.read_csv(os.path.join(out, mname))
m = m[m.sigF2 > 0]
hkl = m[["h", "k", "l"]].to_numpy(int)
F2, sig = m.F2.to_numpy(), m.sigF2.to_numpy()
print(f"{len(m)} unique reflections ({(m.n >= 2).sum()} with n >= 2); heavy atoms {heavy}; fixed {fixed}", flush=True)
A = truth.lattice.cartesian_vectors()
rows = []
for seed in range(10):
    start = perturbed_crystal(truth, np.random.default_rng(10_000 + seed), shift_A=0.05)
    start = Crystal(start.lattice, start.space_group, [replace(a, fract=t.fract, B_iso=t.B_iso) if a.label in fixed else a for a, t in zip(start.atoms, truth.atoms)], start.name)
    r = refine(RefineModel(start, fixed_atoms=fixed), hkl, F2, sig, outlier_sigma=5.0)
    ab = r.absolute
    d = {}
    for t in truth.atoms:
        if t.label in heavy:
            row = ab[ab.atom == t.label].iloc[0]
            dv = (np.array([row.x, row.y, row.z]) - np.array(t.fract)) @ A
            d[t.label] = float(np.linalg.norm(dv))
    strong = F2 > 2 * sig
    Fo, Fc = np.sqrt(np.maximum(F2, 0)), np.sqrt(np.maximum(r.f2_calc, 0))
    r1 = float(np.abs(Fo[strong] - Fc[strong]).sum() / Fo[strong].sum())
    rows.append(dict(seed=seed, converged=bool(r.converged), reason=r.reason, goof=r.goof, R1=r1, wR2=r.wR2, n_par=r.n_par, n_obs=r.n_obs, n_rejected=len(r.rejected),
                     at_bound=",".join(r.at_bound), harmless=",".join(r.at_bound_harmless), median_dpos=float(np.median(list(d.values()))), max_dpos=float(max(d.values())), **{f"d_{k}": v for k, v in d.items()}))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items() if not k.startswith("d_")}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, "refine_tierC.csv"), index=False)
print(df.drop(columns=["reason"]).round(4).to_string())
print(f"converged {int(df.converged.sum())}/10; GooF median {df.goof.median():.2f}; R1 median {df.R1.median():.4f}; median |dpos| {df.median_dpos.median():.4f} A; max |dpos| {df.max_dpos.max():.4f} A")
x = ab[ab.atom.isin(heavy)][["atom", "x", "y", "z", "U_iso"]]
print(x.round(4).to_string(index=False))
