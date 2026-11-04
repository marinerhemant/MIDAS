"""Exploratory control for A19's unmet paired-rms criterion: refine the v3 (exact U) and v4 (refined U) merged tables on their COMMON unique reflections (dev)."""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "..")
import numpy as np, pandas as pd
import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine
st = sim.structure_s2_bilayer(); sg = SpaceGroup.from_hm(st.spacegroup)
truth = Crystal(Lattice(*st.cell), sg, [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)
det = ["La2.z", "Ni.z", "O1.z", "O2.z", "La1.U", "La2.U", "Ni.U", "O1.U"]
def solve(hkl, F2, sig, seed):
    m = RefineModel(perturbed_crystal(truth, np.random.default_rng(10_000 * seed + 2), shift_A=0.10))
    r = refine(m, hkl, F2, sig)
    return m, r
def vec(m, r):
    out = {}
    for j, nm in enumerate(r.names):
        if nm not in det: continue
        ia = [a.label for a in truth.atoms].index(nm.split(".")[0])
        if nm.endswith(".U"): out[nm] = (r.x[j], r.sigma[j])
        else:
            b = m.basis_frac[j]; d = r.absolute.iloc[ia][["x", "y", "z"]].to_numpy(float) - np.array(truth.atoms[ia].fract)
            out[nm] = (float(d @ b) / float(b @ b), r.sigma[j])
    return out
rows, drops = [], []
for seed in range(50):
    a = np.load(f"results/c2v3_merged/{st.name}_seed{seed}.npz"); b = np.load(f"results/c2v4_merged/{st.name}_seed{seed}.npz")
    ka = {tuple(h): i for i, h in enumerate(a["hkl"])}; kb = {tuple(h): i for i, h in enumerate(b["hkl"])}
    common = sorted(set(ka) & set(kb)); drops.append((len(ka) - len(common), len(kb) - len(common)))
    ia = [ka[c] for c in common]; ib = [kb[c] for c in common]
    hkl = np.array(common)
    ma, ra = solve(hkl, a["F2"][ia], a["sig"][ia], seed); mb, rb = solve(hkl, b["F2"][ib], b["sig"][ib], seed)
    if not (ra.converged and rb.converged): continue
    va, vb = vec(ma, ra), vec(mb, rb)
    # compare the two ESTIMATES (not errors): their difference in units of the exact-U formal sigma
    rows.append({p: (vb[p][0] - va[p][0]) / va[p][1] for p in det} | {"seed": seed, "n_common": len(common), "dropped_v3": drops[-1][0], "dropped_v4": drops[-1][1]})
d = pd.DataFrame(rows).set_index("seed")
lines = [f"paired shift (v4 refined-U estimate - v3 exact-U estimate)/sigma, refined on the COMMON unique reflections of the two runs, {len(d)} seeds",
         f"unique reflections removed to make the sets equal: v3 median {np.median([x[0] for x in drops])}, v4 median {np.median([x[1] for x in drops])}; seeds where the sets already agreed: {sum(1 for x in drops if x == (0, 0))}",
         "param   mean    rms     max|.|   frac>1sigma", ]
for p in det:
    lines.append(f"{p:6s} {d[p].mean():+.3f}  {np.sqrt((d[p]**2).mean()):.3f}  {d[p].abs().max():.3f}  {float((d[p].abs()>1).mean()):.2f}")
ok = all(np.sqrt((d[p]**2).mean()) <= 0.5 for p in det)
lines.append(f"paired rms <= 0.5 sigma for all 8 on the common reflections: {'yes' if ok else 'no'}   (EXPLORATORY control, not the registered criterion)")
print("\n".join(lines)); open("results/c2v4_control_common_reflections.txt", "w").write("\n".join(lines) + "\n")
