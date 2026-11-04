"""PREREGISTER A21: refine the merged F2 of the extinction variants with arms K / E / O (dev).

    python run_ext_refine.py <T0|T1_5|T1_10|T2> <seed_lo> <seed_hi>     # appends results/ext_<variant>.csv

Merged tables come from run_c2.py (profile v3, exact U and DAC model) with EXT=<variant> (T0 = the existing c2v3 tables).
K = kinematic refinement; E = extinction thickness free (start 2x truth for T1, 6 um for T2 and T0); O = oracle: the data divided by the
TRUE multiplier y_true(hkl) (from the truth structure) and refined kinematically (A21b: a stronger oracle than 'fix t_b at truth').
All three start from the SAME displaced structure (0.10 A along the allowed shifts, seed 10000*seed+2, as c2v3 N3p).
"""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "..")
from pathlib import Path
import numpy as np, pandas as pd
import simulate as sim
import extinction_truth as et
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine

var, lo_s, hi_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
LAM = 12.3984 / 30.0
VARIANTS = {"T0": (0.0, 0.0, 6.0), "T1_5": (5.0, 0.0, 10.0), "T1_10": (10.0, 0.0, 20.0), "T2": (7.0, 0.5, 6.0)}   # truth t, sigma_ln, E start
t_true, s_ln, t_start = VARIANTS[var]
st = sim.structure_s2_bilayer()
truth = Crystal(Lattice(*st.cell), SpaceGroup.from_hm(st.spacegroup),
                [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)
mdir = Path("results") / ("c2v3_merged" if var == "T0" else f"c2v3_{var}_merged")
csv = Path("results") / f"ext_{var}_{lo_s}.csv"
factor = et.make_factor(st, LAM, t_true, s_ln)


def errors(r, model):
    out = {}
    for j, (nm, kind) in enumerate(zip(r.names, r.kinds)):
        if kind in ("scale", "ext"):
            continue
        ia = [a.label for a in truth.atoms].index(nm.split(".")[0])
        if kind == "U":
            out[nm] = r.x[j] - truth.atoms[ia].B_iso / EIGHT_PI2
        else:
            b = model.basis_frac[j]
            d = r.absolute.iloc[ia][["x", "y", "z"]].to_numpy(float) - np.array(truth.atoms[ia].fract)
            out[nm] = float(d @ b) / float(b @ b)
    return out


for seed in range(lo_s, hi_s):
    f = mdir / f"{st.name}_seed{seed}.npz"
    if not f.exists():
        print(f"seed {seed}: no merged table"); continue
    z = np.load(f)
    hkl, F2, sig = z["hkl"], z["F2"], z["sig"]
    y_true = factor(hkl, sim.f2_truth(st, hkl))
    start = perturbed_crystal(truth, np.random.default_rng(10_000 * seed + 2), shift_A=0.10)
    rows = []
    for arm in ("K", "E", "O"):
        t0 = time.time()
        if arm == "E":
            model = RefineModel(start, refine_extinction=True, wavelength_A=LAM, ext_t_start_um=t_start)
            r = refine(model, hkl, F2, sig)
        elif arm == "O":
            model = RefineModel(start)
            r = refine(model, hkl, F2 / y_true, sig / y_true)
        else:
            model = RefineModel(start)
            r = refine(model, hkl, F2, sig)
        err = errors(r, model)
        row = dict(variant=var, seed=seed, arm=arm, converged=r.converged, poor_fit=r.poor_fit, reason=r.reason, goof=r.goof, chi2=r.chi2,
                   n_unique=len(hkl), n_par=r.n_par, nfev=r.nfev, sv_ratio=r.sv_ratio, seconds=time.time() - t0,
                   t_b=r.t_b_um, sigma_t_b=r.sigma_t_b_um, t_true=t_true)
        for j, nm in enumerate(r.names):
            row[f"x:{nm}"] = r.x[j]; row[f"s:{nm}"] = r.sigma[j]
            if nm in err:
                row[f"e:{nm}"] = err[nm]
        if arm == "E":
            C = r.correlation(); ie = r.names.index("ext.t2")
            for j, nm in enumerate(r.names):
                if j != ie:
                    row[f"c:{nm}"] = C[ie, j]
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(f"seed {seed} {var}: " + "; ".join(f"{r_['arm']} conv={r_['converged']} GooF {r_['goof']:.2f}" for r_ in rows)
          + f"; t_b {rows[1]['t_b']:.2f}", flush=True)
