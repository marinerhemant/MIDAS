"""In-run Friedel estimator on a 360 deg scan (every reflection has its mate in the same run), independent forward model; A39 detector geometry otherwise. python -I friedel_sim_360.py <n_rep> <workers> <out.csv>"""
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
    rep, = a
    import torch; torch.set_num_threads(1)
    F.OM_LO, F.OM_HI = -180.0, 180.0
    g0 = with_params(make_geometry(wavelength_A=F.LAM, dist_mm=170.0, pixel_mm=F.PX, centre_rc=(862.0, 401.0), shape=(F.NR, F.NC), scan=Scan(155.0, -0.2, 850)), F.HEADER)
    rng = np.random.default_rng(13000 + rep); Us = uniform_rotations(400)
    obs = F.noisy(F.forward(T, Us[rng.integers(len(Us))], d_min=1.0), rng, 0.4, 0.3)
    pairs = labelled_pairs(obs)
    pr = F.subsample_pairs(pairs, 100, rng)
    out = []
    for kind, p in (("100 pairs", pr), ("all pairs", pairs)):
        fit = fit_pairs(p, g0)
        out.append(dict(rep=rep, kind=kind, n_pairs=fit.n_pairs, tx=fit.values[0], wedge=fit.values[3], sig_tx=fit.sigma_formal[0], sig_w=fit.sigma_formal[3], n_avail=len(pairs)))
    return out


if __name__ == "__main__":
    n, w, outp = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    with Pool(w) as p:
        d = pd.DataFrame([r for rr in p.map(one, [(i,) for i in range(n)]) for r in rr])
    d.to_csv(outp, index=False)
    for k, g in d.groupby("kind"):
        print(f"360 deg, {k}: n={len(g)}, pairs median {g.n_pairs.median():.0f} (available {g.n_avail.median():.0f}) | tx bias {(g.tx - T['tx']).mean():+.4f} emp_sd {g.tx.std():.4f} formal {g.sig_tx.mean():.4f} | wedge bias {(g.wedge - T['wedge']).mean():+.4f} emp_sd {g.wedge.std():.4f} formal {g.sig_w.mean():.4f}")
