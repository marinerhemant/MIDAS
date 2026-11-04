"""PREREGISTER A22: refine merged F2 with arm I (isotropic U everywhere) and arm A (anisotropic U on La1, La2, Ni).

    python run_aniso_refine.py <S2A|T0> <seed_lo> <seed_hi>     # appends results/aniso_<data>_<lo>.csv

S2A merged tables: results/c2v3_merged/S2A_bilayer_like_aniso_seed<N>.npz (run_c2.py S2A ... v3); T0 (isotropic truth) = the existing
S2 tables. Both start from the SAME displaced isotropic structure (0.10 A along the allowed shifts, seed 10000*seed+2, as c2v3 N3p).
Errors are against the truth: coordinates along the allowed shift (A); anisotropic components against the truth Cartesian tensor entries
(A^2); U_eq (trace/3 of the Cartesian tensor) for the cations in both arms.
"""
import sys, time
sys.path.insert(0, "."); sys.path.insert(0, "..")
from pathlib import Path
import numpy as np, pandas as pd
import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine, u_cart_of_cif

data, lo_s, hi_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ANISO = ("La1", "La2", "Ni")
st_true = sim.structure_s2_bilayer_aniso() if data == "S2A" else sim.structure_s2_bilayer()
st_iso = sim.structure_s2_bilayer()
lat = Lattice(*st_iso.cell)
start_crystal = Crystal(lat, SpaceGroup.from_hm(st_iso.spacegroup),
                        [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st_iso.sites], st_iso.name)
import os
SFX = f"_o{os.environ['ORI']}" if os.environ.get("ORI") else ""            # A30: another orientation (tables from run_c2.py with the same ORI)
tab = Path(f"results/c2v3{SFX}_merged") / (f"{st_true.name}_seed" + "{}.npz")
csv = Path("results") / f"aniso_{data}{SFX}_{lo_s}.csv"
# truth tensors (Cartesian, A^2) per site
truth_cart = {}
for s in st_true.sites:
    u6 = s.u_aniso if s.u_aniso is not None else (s.u_iso, s.u_iso, s.u_iso, 0.0, 0.0, 0.0)
    truth_cart[s.label] = u_cart_of_cif(lat, u6)
ent = {"Uxx": (0, 0), "Uyy": (1, 1), "Uzz": (2, 2), "Uxy": (0, 1), "Uxz": (0, 2), "Uyz": (1, 2)}
truth_ueq = {k: float(np.trace(v) / 3.0) for k, v in truth_cart.items()}
labels = [a.label for a in start_crystal.atoms]


def errors(r, model):
    out = {}
    ab = r.absolute
    for j, (nm, kind) in enumerate(zip(r.names, r.kinds)):
        if kind in ("scale", "ext"):
            continue
        label, sub = nm.split(".")
        ia = labels.index(label)
        if kind == "coord":
            b = model.basis_frac[j]
            d = ab.iloc[ia][["x", "y", "z"]].to_numpy(float) - np.array(start_crystal.atoms[ia].fract)
            out[nm] = float(d @ b) / float(b @ b)
        elif kind == "U":
            out[nm] = r.x[j] - truth_ueq[label]
        else:
            out[nm] = r.x[j] - truth_cart[label][ent[sub]]
    for lab in ANISO:
        out[f"{lab}.Ueq"] = float(ab.iloc[labels.index(lab)]["U_iso"]) - truth_ueq[lab] if lab in ab.atom.values else np.nan
    return out


for seed in range(lo_s, hi_s):
    f = Path(str(tab).format(seed))
    if not f.exists():
        print(f"seed {seed}: no merged table {f}"); continue
    z = np.load(f)
    hkl, F2, sig = z["hkl"], z["F2"], z["sig"]
    start = perturbed_crystal(start_crystal, np.random.default_rng(10_000 * seed + 2), shift_A=0.10)
    rows = []
    for arm in ("I", "A"):
        t0 = time.time()
        model = RefineModel(start, aniso_sites=ANISO if arm == "A" else ())
        r = refine(model, hkl, F2, sig)
        err = errors(r, model)
        row = dict(data=data, seed=seed, arm=arm, converged=r.converged, poor_fit=r.poor_fit, reason=r.reason, goof=r.goof, chi2=r.chi2,
                   n_unique=len(hkl), n_par=r.n_par, nfev=r.nfev, sv_ratio=r.sv_ratio, seconds=time.time() - t0)
        for j, nm in enumerate(r.names):
            row[f"x:{nm}"] = r.x[j]; row[f"s:{nm}"] = r.sigma[j]
            if nm in err:
                row[f"e:{nm}"] = err[nm]
        for k_, v_ in err.items():
            if k_.endswith(".Ueq"):
                row[f"e:{k_}"] = v_
        if arm == "A":
            C = r.correlation(); ix = {n: i for i, n in enumerate(r.names)}
            for lab in ANISO:
                a_, c_ = ix[f"{lab}.Uxx"], ix[f"{lab}.Uzz"]
                row[f"c:{lab}"] = C[a_, c_]
                var = r.cov[a_, a_] + r.cov[c_, c_] - 2 * r.cov[a_, c_]
                row[f"d:{lab}"] = r.x[c_] - r.x[a_]                                        # Uzz - Uxx
                row[f"ds:{lab}"] = float(np.sqrt(max(var, 0.0)))
            row["min_eig"] = float(r.absolute["U_min_eig"].min())
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(f"seed {seed} {data}: " + "; ".join(f"{r_['arm']} conv={r_['converged']} GooF {r_['goof']:.2f}" for r_ in rows), flush=True)
