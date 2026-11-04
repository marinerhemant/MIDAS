"""Robustness of the Friedel-pair estimator to wrong hkl labels (a fraction of labels replaced by a random other label), S1 truth, ~100 pairs before corruption.  python -I friedel_sim_labels.py <n_rep> <workers> <out.csv>"""
import os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import numpy as np, pandas as pd
import friedel_sim as F
from multiprocessing import Pool
from midas_sx.friedel_geometry import fit_pairs, labelled_pairs, with_params
from midas_sx.geometry import Scan, make_geometry
from midas_stress.orientation_sampling import uniform_rotations
T = dict(tx=-0.10, ty=-0.19, tz=-20.07, wedge=0.41, bcz=861.9, bcy=401.3, lsd=171.14)


def one(a):
    err, rep = a
    import torch; torch.set_num_threads(1)
    g0 = with_params(make_geometry(wavelength_A=F.LAM, dist_mm=170.0, pixel_mm=F.PX, centre_rc=(862.0, 401.0), shape=(F.NR, F.NC), scan=Scan(155.0, -0.2, 850)), F.HEADER)
    rng = np.random.default_rng(9000 + rep); Us = uniform_rotations(400)
    obs = F.noisy(F.forward(T, Us[rng.integers(len(Us))]), rng, 0.4, 0.3, label_err=err)
    pr = F.subsample_pairs(labelled_pairs(obs), 100, rng)
    fit = fit_pairs(pr, g0)
    return dict(label_err=err, rep=rep, n_pairs=fit.n_pairs, tx=fit.values[0], wedge=fit.values[3], sig_tx=fit.sigma_formal[0], sig_w=fit.sigma_formal[3], rms=fit.rms_final)


if __name__ == "__main__":
    n, w, outp = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    with Pool(w) as p:
        d = pd.DataFrame(p.map(one, [(e, r) for e in (0.0, 0.05, 0.15) for r in range(n)]))
    d.to_csv(outp, index=False)
    for e, g in d.groupby("label_err"):
        print(f"label error {e:.2f}: n={len(g)} pairs median {g.n_pairs.median():.0f} | tx bias {(g.tx - T['tx']).mean():+.4f} sd {g.tx.std():.4f} rms-err {np.sqrt(((g.tx - T['tx'])**2).mean()):.4f} | wedge bias {(g.wedge - T['wedge']).mean():+.4f} sd {g.wedge.std():.4f} rms-err {np.sqrt(((g.wedge - T['wedge'])**2).mean()):.4f}")
