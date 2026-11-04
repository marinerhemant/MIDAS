"""Control for the per-run deviation arm of fit_multirun (per_run=("tx","wedge")), independent forward model of friedel_sim.py.
Mode null: all three runs share one geometry (S1) -> the deviations must scatter around 0 with their sigma (false-positive rate).
Mode inject: run 1 tx +0.12 / wedge -0.11, run 2 tx +0.03 / wedge -0.19 (what the real data gave) -> recovered.   python -I friedel_sim_dev.py <n_rep> <n_boot> <workers> <out.csv>"""
import os, sys
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__))]
import numpy as np, pandas as pd
import friedel_sim as F
from multiprocessing import Pool
from midas_sx.friedel_geometry import fit_multirun, with_params
from midas_sx.geometry import Scan, make_geometry
from midas_stress.orientation_sampling import uniform_rotations
T0 = dict(tx=-0.09, ty=-0.18, tz=-20.15, wedge=0.446, bcz=861.9, bcy=401.7, lsd=170.85)
DEV = {"null": [(0, 0), (0, 0)], "inject": [(0.12, -0.11), (0.03, -0.19)]}


def one(a):
    mode, rep, nb = a
    import torch; torch.set_num_threads(1)
    g0 = with_params(make_geometry(wavelength_A=F.LAM, dist_mm=170.0, pixel_mm=F.PX, centre_rc=(862.0, 401.0), shape=(F.NR, F.NC), scan=Scan(155.0, -0.2, 850)), F.HEADER)
    rng = np.random.default_rng(11000 + rep + (500 if mode == "inject" else 0)); Us = uniform_rotations(400); U0 = Us[rng.integers(len(Us))]
    P = [np.eye(3)] + [Us[rng.integers(len(Us))] for _ in range(2)]
    Ts = [T0] + [dict(T0, tx=T0["tx"] + dx, wedge=T0["wedge"] + dw) for dx, dw in DEV[mode]]
    runs = []
    for Pk, Tk in zip(P, Ts):
        obs = F.noisy(F.forward(Tk, Pk @ U0), rng, 0.4, 0.3, label_err=0.03)
        runs.append(obs[rng.random(len(obs)) < 1100.0 / len(obs)].reset_index(drop=True))
    fit = fit_multirun(runs, g0, n_boot=nb, seed=rep, per_run=("tx", "wedge"))
    rec = dict(mode=mode, rep=rep, n_groups=fit.n_groups, rms=fit.rms_final)
    for k in (1, 2):
        for j, n_ in enumerate(("tx", "wedge")):
            rec[f"dev{k}_{n_}"] = fit.dev[k - 1, j]; rec[f"sig{k}_{n_}"] = fit.sigma_dev[k - 1, j]
            rec[f"true{k}_{n_}"] = DEV[mode][k - 1][j]
    shared = fit_multirun(runs, g0)
    rec["rms_shared"] = shared.rms_final
    return rec


if __name__ == "__main__":
    n, nb, w, outp = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    with Pool(w) as p:
        d = pd.DataFrame(p.map(one, [(m, r, nb) for m in ("null", "inject") for r in range(n)]))
    d.to_csv(outp, index=False)
    for m, g in d.groupby("mode"):
        print(f"== {m}: n={len(g)}, groups median {g.n_groups.median():.0f}, rms per-run arm {g.rms.median():.5f} vs shared arm {g.rms_shared.median():.5f}")
        for k in (1, 2):
            for n_ in ("tx", "wedge"):
                e = g[f"dev{k}_{n_}"] - g[f"true{k}_{n_}"]; sg = g[f"sig{k}_{n_}"]
                print(f"   run{k} {n_:5s}: true {g[f'true{k}_{n_}'].iloc[0]:+.2f}  mean est {g[f'dev{k}_{n_}'].mean():+.4f}  bias {e.mean():+.4f}  emp_sd {e.std(ddof=1):.4f}  mean sigma_boot {sg.mean():.4f}  |est|/sigma > 3 in {(g[f'dev{k}_{n_}'].abs() / sg > 3).mean():.2f} of reps")
