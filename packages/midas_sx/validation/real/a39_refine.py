"""PREREGISTER A39 tier C: refine the non-H atoms of L-alanine (H fixed at COD 1574527) against the merged F², 10 starts displaced 0.05 A, two arms: isotropic U and anisotropic U for C, N, O.

    python -I a39_refine.py <out dir> <cif> [merged csv name]
"""
import json, os, sys
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")]
_root = os.path.dirname(os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"))
sys.path[:0] = [os.path.join(_root, p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(_root, p))]
import numpy as np, pandas as pd, gemmi
from dataclasses import replace
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine

out, cif = sys.argv[1], sys.argv[2]
mname = sys.argv[3] if len(sys.argv) > 3 else "merged.csv"
sg = SpaceGroup.from_hm("P 21 21 21")
ss = gemmi.read_small_structure(cif)
truth = Crystal(Lattice(ss.cell.a, ss.cell.b, ss.cell.c, ss.cell.alpha, ss.cell.beta, ss.cell.gamma), sg,
                [Atom(s.element.name, (s.fract.x, s.fract.y, s.fract.z), s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in ss.sites], "alanine100K")
fixed = tuple(a.label for a in truth.atoms if a.element == "H")
heavy = [a.label for a in truth.atoms if a.element != "H"]
m = pd.read_csv(os.path.join(out, mname))
m = m[m.sigF2 > 0]
hkl = m[["h", "k", "l"]].to_numpy(int)
F2, sig = m.F2.to_numpy(), m.sigF2.to_numpy()
print(f"{len(m)} unique reflections ({(m.n >= 2).sum()} with n >= 2); heavy {heavy}; fixed {fixed}", flush=True)
A = truth.lattice.cartesian_vectors()
allrows = []
for arm, aniso in (("iso", ()), ("aniso", tuple(heavy))):
    rows = []
    for seed in range(10):
        start = perturbed_crystal(truth, np.random.default_rng(10_000 + seed), shift_A=0.05)
        start = Crystal(start.lattice, start.space_group, [replace(a, fract=t.fract, B_iso=t.B_iso) if a.label in fixed else a for a, t in zip(start.atoms, truth.atoms)], start.name)
        model = RefineModel(start, fixed_atoms=fixed, aniso_sites=aniso)
        r = refine(model, hkl, F2, sig, outlier_sigma=5.0)
        ab = r.absolute
        d = {}
        for t in truth.atoms:
            if t.label in heavy:
                row = ab[ab.atom == t.label].iloc[0]
                d[t.label] = float(np.linalg.norm((np.array([row.x, row.y, row.z]) - np.array(t.fract)) @ A))
        strong = F2 > 2 * sig
        Fo, Fc = np.sqrt(np.maximum(F2, 0)), np.sqrt(np.maximum(r.f2_calc, 0))
        pulls = []
        for i, k in enumerate(r.kinds):
            if k == "coord":
                lab = model.names[i].split(".")[0]
                row = ab[ab.atom == lab].iloc[0]; t = next(t for t in truth.atoms if t.label == lab)
                b = model.basis_frac[i]
                dev = float(((np.array([row.x, row.y, row.z]) - np.array(t.fract)) @ b) / (b @ b))
                pulls.append((dev / r.sigma[i], dev / (r.sigma[i] * r.goof)))
        pl = np.array(pulls)
        rows.append(dict(arm=arm, seed=seed, converged=bool(r.converged), reason=r.reason, goof=r.goof, R1=float(np.abs(Fo[strong] - Fc[strong]).sum() / Fo[strong].sum()), wR2=r.wR2,
                         n_par=r.n_par, n_obs=r.n_obs, n_rejected=len(r.rejected), at_bound=",".join(r.at_bound), harmless=",".join(r.at_bound_harmless),
                         median_dpos=float(np.median(list(d.values()))), max_dpos=float(max(d.values())), pull_rms_formal=float(np.sqrt((pl[:, 0] ** 2).mean())),
                         pull_rms_rescaled=float(np.sqrt((pl[:, 1] ** 2).mean())), **{f"d_{k}": v for k, v in d.items()}))
    df = pd.DataFrame(rows)
    allrows.append(df)
    print(f"\n## arm {arm}")
    print(df.drop(columns=["reason"]).round(4).to_string())
    print(f"converged {int(df.converged.sum())}/10; GooF median {df.goof.median():.2f}; R1 median {df.R1.median():.4f}; median |dpos| {df.median_dpos.median():.4f} A; max |dpos| {df.max_dpos.max():.4f} A; "
          f"pull rms formal {df.pull_rms_formal.median():.2f}, GooF-rescaled {df.pull_rms_rescaled.median():.2f}", flush=True)
pd.concat(allrows).to_csv(os.path.join(out, "refine_tierC.csv"), index=False)
