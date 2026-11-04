"""Positive control for the CROSS-RUN Friedel estimator (midas_sx.friedel_geometry.fit_multirun) with the INDEPENDENT forward model of friedel_sim.py.
Three runs of one crystal: U_k = P_k U_0 with P_k random rotations (midas_stress.orientation_sampling.uniform_rotations); spots subsampled to the real per-run counts
(~800 per run before grouping), 3 % wrong labels, 0.4 px / 0.3 frame noise. Truth = S1 of friedel_sim.   python -I friedel_sim_cross.py <n_rep> <n_boot> <workers> <out.csv>"""
import os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import numpy as np, pandas as pd
import friedel_sim as F
from multiprocessing import Pool
from midas_sx.friedel_geometry import fit_multirun, with_params
from midas_sx.geometry import Scan, make_geometry
from midas_stress.orientation_sampling import uniform_rotations
T = F.SCEN_S1 if hasattr(F, "SCEN_S1") else dict(tx=-0.10, ty=-0.19, tz=-20.07, wedge=0.41, bcz=861.9, bcy=401.3, lsd=171.14)


def one(a):
    rep, n_boot, label_err = a
    import torch; torch.set_num_threads(1)
    g0 = with_params(make_geometry(wavelength_A=F.LAM, dist_mm=170.0, pixel_mm=F.PX, centre_rc=(862.0, 401.0), shape=(F.NR, F.NC), scan=Scan(155.0, -0.2, 850)), F.HEADER)
    rng = np.random.default_rng(7000 + rep); Us = uniform_rotations(400); U0 = Us[rng.integers(len(Us))]
    P = [np.eye(3)] + [Us[rng.integers(len(Us))] for _ in range(2)]
    runs = []
    for Pk in P:
        obs = F.noisy(F.forward(T, Pk @ U0), rng, 0.4, 0.3, label_err=label_err)
        keep = rng.random(len(obs)) < 800.0 / len(obs)
        runs.append(obs[keep].reset_index(drop=True))
    fit = fit_multirun(runs, g0, n_boot=n_boot, seed=rep)
    rec = dict(rep=rep, label_err=label_err, n_groups=fit.n_groups, n_cross=fit.n_cross_run_groups, rms=fit.rms_final)
    for n_, v, sf, sb in zip(fit.names, fit.values, fit.sigma_formal, fit.sigma_boot if fit.sigma_boot is not None else [np.nan] * 7):
        rec[n_] = v; rec["sig_" + n_] = sf; rec["sigb_" + n_] = sb; rec["true_" + n_] = T[n_]
    return rec


if __name__ == "__main__":
    n, nb, w, outp = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    with Pool(w) as p:
        d = pd.DataFrame(p.map(one, [(r, nb, 0.03) for r in range(n)]))
    d.to_csv(outp, index=False)
    print(f"n={len(d)} reps; groups median {d.n_groups.median():.0f}, cross-run groups median {d.n_cross.median():.0f}")
    for p_ in ("tx", "ty", "tz", "wedge", "bcz", "bcy", "lsd"):
        e = d[p_] - d["true_" + p_]
        print(f"  {p_:6s} bias {e.mean():+.4f} emp_sd {e.std(ddof=1):.4f} | formal {d['sig_' + p_].mean():.4f} (sd/sigma {e.std(ddof=1) / d['sig_' + p_].mean():.2f}, cov {(e.abs() < d['sig_' + p_]).mean():.2f}) | boot {d['sigb_' + p_].mean():.4f} (sd/sigma {e.std(ddof=1) / d['sigb_' + p_].mean():.2f}, cov {(e.abs() < d['sigb_' + p_]).mean():.2f})")
