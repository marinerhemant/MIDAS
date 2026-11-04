"""Bootstrap (over hkl groups) sigma of the Friedel-pair estimator against the simulation truth: sd/sigma and 68 % coverage, ~100 pairs.   python -I friedel_sim_boot.py <n_rep> <n_boot> <workers> <out.csv>"""
import os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import numpy as np, pandas as pd
import friedel_sim as F
from multiprocessing import Pool
from midas_sx.friedel_geometry import fit_pairs, labelled_pairs, with_params
from midas_sx.geometry import Scan, make_geometry
from midas_stress.orientation_sampling import uniform_rotations

SC = {"S1": dict(tx=-0.10, ty=-0.19, tz=-20.07, wedge=0.41, bcz=861.9, bcy=401.3, lsd=171.14),
      "S2": dict(tx=0.30, ty=-0.19, tz=-20.07, wedge=-0.30, bcz=861.9, bcy=401.3, lsd=171.14)}


def one(args):
    name, rep, n_boot = args
    import torch; torch.set_num_threads(1)
    T = SC[name]; scan = Scan(155.0, -0.2, 850)
    g0 = with_params(make_geometry(wavelength_A=F.LAM, dist_mm=170.0, pixel_mm=F.PX, centre_rc=(862.0, 401.0), shape=(F.NR, F.NC), scan=scan), F.HEADER)
    rng = np.random.default_rng(5000 + 100 * list(SC).index(name) + rep)
    Us = uniform_rotations(400)
    obs = F.noisy(F.forward(T, Us[rng.integers(len(Us))]), rng, 0.4, 0.3)
    pr = F.subsample_pairs(labelled_pairs(obs), 100, rng)
    fit = fit_pairs(pr, g0, n_boot=n_boot, seed=rep)
    rec = dict(scenario=name, rep=rep, n_pairs=fit.n_pairs, n_groups=fit.n_groups)
    for n_, v, sf, sb in zip(fit.names, fit.values, fit.sigma_formal, fit.sigma_boot):
        rec[n_] = v; rec["sig_" + n_] = sf; rec["sigb_" + n_] = sb; rec["true_" + n_] = T[n_]
    return rec


if __name__ == "__main__":
    n_rep, n_boot, w, outp = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    with Pool(w) as pool:
        rows = pool.map(one, [(nm, r, n_boot) for nm in SC for r in range(n_rep)])
    d = pd.DataFrame(rows); d.to_csv(outp, index=False)
    for nm, g in d.groupby("scenario"):
        print(f"== {nm}  n={len(g)}  pairs median {g.n_pairs.median():.0f}")
        for p in ("tx", "ty", "tz", "wedge", "bcz", "bcy", "lsd"):
            e = g[p] - g["true_" + p]
            print(f"  {p:6s} bias {e.mean():+.4f}  emp_sd {e.std(ddof=1):.4f}  sigma_formal {g['sig_' + p].mean():.4f} (sd/sigma {e.std(ddof=1) / g['sig_' + p].mean():.2f}, cov {(e.abs() < g['sig_' + p]).mean():.2f})  "
                  f"sigma_boot {g['sigb_' + p].mean():.4f} (sd/sigma {e.std(ddof=1) / g['sigb_' + p].mean():.2f}, cov {(e.abs() < g['sigb_' + p]).mean():.2f})")
