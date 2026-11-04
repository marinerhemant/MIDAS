"""PREREGISTER A33: refine merged F2 of the JOINT dataset (aniso cations + 90 deg twin m = 0.30 + per-domain extinction t_b = 5 um; S2OA) with five arms.

    python run_joint_refine.py <seed_lo> <seed_hi>      # appends results/joint_<lo>.csv

Tables: results/c2v3_J30_merged/S2OA_immm_aniso_seed<N>.npz (JOINT=30 python run_c2.py S2OA 0 30 v3). Arms (all from the S2O structure displaced 0.10 A along the allowed shifts,
seed 10000*seed+2, isotropic U at the S2O values, t_b start 10 um, m start 0.15):  F = aniso(La1, La2, Ni) + twin + extinction; FmE = no extinction; FmA = isotropic U;
FmT = no twin; K = kinematic, isotropic, untwinned. gate_free = the oxygen U (undetermined, as A26).
"""
import sys, time
sys.path.insert(0, "."); sys.path.insert(0, "..")
from pathlib import Path
import numpy as np, pandas as pd
import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine, u_cart_of_cif

lo_s, hi_s = int(sys.argv[1]), int(sys.argv[2])
LAM = 12.3984 / 30.0
M_TRUE, TB_TRUE = 0.30, 5.0
ANISO = ("La1", "La2", "Ni")
GATE_FREE = ("O1a.U", "O1b.U", "O2.U", "O3.U")
st_true, st_start = sim.structure_s2oa(), sim.structure_s2o()
lat = Lattice(*st_true.cell)
start_crystal = Crystal(lat, SpaceGroup.from_hm(st_start.spacegroup), [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st_start.sites], st_start.name)
labels = [a.label for a in start_crystal.atoms]
tab = Path("results/c2v3_J30_merged") / (f"{st_true.name}_seed" + "{}.npz")
csv = Path("results") / f"joint_{lo_s}.csv"
truth_cart, truth_ueq = {}, {}
for s in st_true.sites:
    u6 = s.u_aniso if s.u_aniso is not None else (s.u_iso, s.u_iso, s.u_iso, 0.0, 0.0, 0.0)
    truth_cart[s.label] = u_cart_of_cif(lat, u6)
    truth_ueq[s.label] = float(np.trace(truth_cart[s.label]) / 3.0)
ent = {"Uxx": (0, 0), "Uyy": (1, 1), "Uzz": (2, 2), "Uxy": (0, 1), "Uxz": (0, 2), "Uyz": (1, 2)}
ARMS = {"F": dict(aniso_sites=ANISO, refine_twin=True, twin_start=0.15, refine_extinction=True, wavelength_A=LAM),
        "FmE": dict(aniso_sites=ANISO, refine_twin=True, twin_start=0.15),
        "FmA": dict(refine_twin=True, twin_start=0.15, refine_extinction=True, wavelength_A=LAM),
        "FmT": dict(aniso_sites=ANISO, refine_extinction=True, wavelength_A=LAM),
        "K": dict()}


def errors(r, model):
    out = {}
    ab = r.absolute
    for j, (nm, kind) in enumerate(zip(r.names, r.kinds)):
        if kind == "scale":
            continue
        if kind == "twin":
            out[nm] = r.x[j] - M_TRUE; continue
        if kind == "ext":
            out["t_b"] = r.t_b_um - TB_TRUE; continue
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
    return out


for seed in range(lo_s, hi_s):
    f = Path(str(tab).format(seed))
    if not f.exists():
        print(f"seed {seed}: no merged table {f}"); continue
    z = np.load(f)
    hkl, F2, sig = z["hkl"], z["F2"], z["sig"]
    start = perturbed_crystal(start_crystal, np.random.default_rng(10_000 * seed + 2), shift_A=0.10)
    rows = []
    for arm, kw in ARMS.items():
        t0 = time.time()
        model = RefineModel(start, **kw)
        r = refine(model, hkl, F2, sig, gate_free=GATE_FREE)
        err = errors(r, model)
        row = dict(seed=seed, arm=arm, converged=r.converged, poor_fit=r.poor_fit, reason=r.reason, goof=r.goof, chi2=r.chi2, n_unique=len(hkl), n_par=r.n_par,
                   nfev=r.nfev, sv_ratio=r.sv_ratio, seconds=time.time() - t0, at_bound=";".join(r.at_bound), null_params=";".join(r.null_params))
        for j, nm in enumerate(r.names):
            row[f"x:{nm}"] = r.x[j]; row[f"s:{nm}"] = r.sigma[j]
            if nm in err:
                row[f"e:{nm}"] = err[nm]
        if "t_b" in err:
            row["e:t_b"] = err["t_b"]; row["s:t_b"] = r.sigma_t_b_um; row["x:t_b"] = r.t_b_um
        C = r.correlation(); ix = {n: i for i, n in enumerate(r.names)}
        pairs = [("twin.m", "ext.t2"), ("twin.m", "scale"), ("ext.t2", "scale")] + [(f"{lab}.Uzz", q) for lab in ANISO for q in ("twin.m", "ext.t2", "scale")]
        for a_, b_ in pairs:
            if a_ in ix and b_ in ix:
                row[f"c:{a_}|{b_}"] = C[ix[a_], ix[b_]]
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(f"seed {seed}: " + "; ".join(f"{r_['arm']} conv={r_['converged']} GooF {r_['goof']:.2f}" for r_ in rows), flush=True)
