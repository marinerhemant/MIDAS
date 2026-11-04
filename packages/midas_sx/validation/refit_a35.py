"""PREREGISTER A35: refit the P11 observation tables reduced with the ESTIMATED shell background and with the ORACLE background (0 on signal-only frames).

    python refit_a35.py <seed_lo> <seed_hi>      # reads results/a35_obs/P11_seed<N>.pkl (diag_bg_oracle.py, SAVE_OBS=1), appends results/a35_fit_<lo>.csv
Arm Tr of A27: component flags and partner indices, outlier_sigma 5, gate_free = the oxygen U; start = truth displaced 0.10 A (seed 10000*seed+2).
"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "..")
from pathlib import Path
import numpy as np, pandas as pd
import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine

lo_s, hi_s = int(sys.argv[1]), int(sys.argv[2])
GATE_FREE = ("O1a.U", "O1b.U", "O2.U", "O3.U")
st = sim.structure_s2p(1.011)
truth = Crystal(Lattice(*st.cell), SpaceGroup.from_hm(st.spacegroup), [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)
labels = [a.label for a in truth.atoms]
csv = Path("results") / f"a35_fit_{lo_s}.csv"


def errors(r, model):
    o = {}
    for j, (nm, kind) in enumerate(zip(r.names, r.kinds)):
        if kind in ("scale", "ext"):
            continue
        if kind == "twin":
            o[nm] = r.x[j] - 0.30; continue
        ia = labels.index(nm.split(".")[0])
        if kind == "U":
            o[nm] = r.x[j] - truth.atoms[ia].B_iso / EIGHT_PI2
        else:
            b = model.basis_frac[j]
            d = r.absolute.iloc[ia][["x", "y", "z"]].to_numpy(float) - np.array(truth.atoms[ia].fract)
            o[nm] = float(d @ b) / float(b @ b)
    return o


for seed in range(lo_s, hi_s):
    f = Path("results/a35_obs") / f"P11_seed{seed}.pkl"
    if not f.exists():
        print(f"seed {seed}: no table"); continue
    allobs = pd.read_pickle(f)
    start = perturbed_crystal(truth, np.random.default_rng(10_000 * seed + 2), shift_A=0.10)
    rows = []
    for variant in ("est_bg", "oracle_bg0"):
        O = allobs[allobs.variant == variant].reset_index(drop=True)
        H = O[["h", "k", "l"]].to_numpy(int); P = O[["p0", "p1", "p2"]].to_numpy(int)
        c2 = O.kind.to_numpy() == "c2"
        P_eff = P.copy(); P_eff[c2] = H[c2]
        model = RefineModel(start, refine_twin=True, twin_start=0.15)
        r = refine(model, H, O.F2.to_numpy(), O.sigF2.to_numpy(), partner=P_eff, flags=O[["f1", "f2"]].to_numpy(float), outlier_sigma=5.0, gate_free=GATE_FREE)
        err = errors(r, model)
        row = dict(seed=seed, variant=variant, converged=r.converged, goof=r.goof, n_obs=len(O), n_rej=len(r.rejected), m=r.twin_fraction, sigma_m=r.sigma_twin_fraction)
        for j, nm in enumerate(r.names):
            row[f"x:{nm}"] = r.x[j]; row[f"s:{nm}"] = r.sigma[j]
            if nm in err:
                row[f"e:{nm}"] = err[nm]
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(f"seed {seed}: " + "; ".join(f"{r_['variant']} conv={r_['converged']} GooF {r_['goof']:.2f} m {r_['m']:.3f}" for r_ in rows), flush=True)
