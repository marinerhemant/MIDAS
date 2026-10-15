"""v6 design exploration (NOT a result): a cross-fitted amplitude calibration curve lambda(delta) learned on the training planes only,
tried on the ROIs excluded from scoring. delta = (obpitch position of the plane) - (fitted centre), in motor steps."""
import json, sys, time
import numpy as np
import midas_dfxm.forward_maps as fm
import midas_dfxm.forward_profile as fp

DATASETS = {"smoke": (150, 450, 700, 1000), "notebook": (300, 412, 1100, 1212)}
FOLDS = (1, 2, 3, 4, 5, 6); PRIMARY = (3, 4, 5)
EDGES = np.arange(-3.5, 3.51, 0.5)


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


def main(h5, out):
    R = {}
    for name, roi in DATASETS.items():
        d = fm.load_planes_cube(h5, roi=roi, require_consistent_order=False, log=lambda *a: None)
        cube, mu_g, chi_g, ob_g, rc = d["cube"], d["mu_g"], d["chi_g"], d["ob_g"], d["rc"]
        n_ob = len(ob_g)
        masks = {k: d["fold_masks"][k][rc[:, 0], rc[:, 1]] for k in range(n_ob)}
        v3 = fp.leave_one_plane_out_profile(cube, mu_g, chi_g, ob_g, list(FOLDS), masks, nested_sub=40_000, keep=True)
        pos = fp._positions(ob_g, range(n_ob))
        res = {}
        for k in FOLDS:
            r = v3[k]; p = r["_params"]; fit = r["_fit"]; m = masks[k]
            train = [j for j in range(n_ob) if j != k]
            S = r["_S"]; sig = r["_sig"]                      # (7, P) training totals above baseline
            # cross-fit: predict each training plane from the other six
            num = np.zeros(len(EDGES) - 1); den = np.zeros(len(EDGES) - 1)
            for t, j in enumerate(train):
                rest = [i for i in range(len(train)) if i != t]
                c, A = fp.fit_centres(sig[rest], S, r["_pos_tr"][rest], fit.w, fit.s)
                pred = A * fp.box_profile(r["_pos_tr"][t] - c, fit.w, fit.s) * S       # predicted total of plane j
                delta = r["_pos_tr"][t] - c
                b_ = np.clip(np.digitize(delta, EDGES) - 1, 0, len(EDGES) - 2)
                ok = m & (pred > 0)
                np.add.at(num, b_[ok], (sig[t] * pred)[ok]); np.add.at(den, b_[ok], (pred ** 2)[ok])
            lam = np.where(den > 0, num / np.maximum(den, 1e-12), 1.0)
            # apply to the held-out prediction: a = A P(pos_k - c) * lam(delta_k)
            delta_k = r["_pos_k"] - fit.c
            bk = np.clip(np.digitize(delta_k, EDGES) - 1, 0, len(EDGES) - 2)
            obs = cube[:, :, k, :].astype(np.float64); bb = r["_b"]; y = obs - bb[None, None]
            G = fm._gauss(mu_g[:, None], p.mu0[None], p.smu[None])[:, None] * fm._gauss(chi_g[:, None], p.chi0[None], p.schi[None])[None]
            u = (y * G).sum((0, 1)); v = (G ** 2).sum((0, 1)); yy = (y ** 2).sum((0, 1))
            sst = ((obs - obs.mean((0, 1), keepdims=True)) ** 2).sum((0, 1))
            def pooled(a): return float(1 - (yy - 2 * a * u + a ** 2 * v)[m].sum() / sst[m].sum())
            a_mod = fit.A * fp.box_profile(delta_k, fit.w, fit.s)
            res[k] = dict(v3=pooled(a_mod), calibrated=pooled(a_mod * lam[bk]), lam=lam.round(3).tolist())
            log(name, "fold", k, "v3", round(res[k]["v3"], 3), "calibrated", round(res[k]["calibrated"], 3))
        R[name] = {int(k): v for k, v in res.items()}
        del cube
    json.dump(R, open(out, "w"), indent=1)
    for name in R:
        print(name, "primary mean v3", np.mean([R[name][k]["v3"] for k in PRIMARY]), "calibrated", np.mean([R[name][k]["calibrated"] for k in PRIMARY]),
              "flank v3", [R[name][k]["v3"] for k in (1, 6)], "calibrated", [R[name][k]["calibrated"] for k in (1, 6)])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
