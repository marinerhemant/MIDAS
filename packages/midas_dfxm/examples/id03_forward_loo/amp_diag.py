"""v6 part 2: decompose the oracle-amplitude gain on the ROIs excluded from scoring (PREREGISTER_v6.md).

    python amp_diag.py RAW_H5 OUT.json

The v3 empirical model is separable, signal = a_p * G_p with G_p = g(mu) g(chi) fixed, so every oracle variant is a choice of the scalar a_p per pixel and
the held-out SSE_p = sum (obs-b)^2 - 2 a_p u_p + a_p^2 v_p, u_p = sum (obs-b) G, v_p = sum G^2: no render is needed.
"""
import json
import sys
import time

import numpy as np
from scipy import ndimage

import midas_dfxm.forward_maps as fm
import midas_dfxm.forward_profile as fp

DATASETS = {"smoke": (150, 450, 700, 1000), "notebook": (300, 412, 1100, 1212)}
FOLDS = (1, 2, 3, 4, 5, 6)
PRIMARY, FLANK = (3, 4, 5), (1, 6)
LS = (8, 16, 32, 64)
WS_GRID = [(w, s) for w in fp.W_GRID for s in fp.S_GRID]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def box(img, L):
    return ndimage.uniform_filter(img, size=L, mode="constant", cval=0.0)


def main(h5, out):
    R = {"datasets": {}}
    for name, roi in DATASETS.items():
        log("dataset", name)
        d = fm.load_planes_cube(h5, roi=roi, require_consistent_order=False, log=lambda *a: None)
        cube, mu_g, chi_g, ob_g, rc = d["cube"], d["mu_g"], d["chi_g"], d["ob_g"], d["rc"]
        n_ob, P = len(ob_g), cube.shape[-1]
        H, W = d["union"].shape
        masks = {k: d["fold_masks"][k][rc[:, 0], rc[:, 1]] for k in range(n_ob)}
        v3 = fp.leave_one_plane_out_profile(cube, mu_g, chi_g, ob_g, list(FOLDS), masks, nested_sub=40_000, keep=True)
        pos = fp._positions(ob_g, range(n_ob))
        cand = np.arange(fp.CAND_LO, fp.CAND_HI + 1e-9, fp.CAND_STEP)
        res = {}
        for k in FOLDS:
            r = v3[k]; p = r["_params"]; fit = r["_fit"]; m = masks[k]
            obs = cube[:, :, k, :].astype(np.float64)
            b = r["_b"]
            y = obs - b[None, None]
            G = fm._gauss(mu_g[:, None], p.mu0[None], p.smu[None])[:, None] * fm._gauss(chi_g[:, None], p.chi0[None], p.schi[None])[None]
            u = (y * G).sum((0, 1)); v = (G ** 2).sum((0, 1))
            yy = (y ** 2).sum((0, 1)); sst = ((obs - obs.mean((0, 1), keepdims=True)) ** 2).sum((0, 1))
            def pooled(a):
                sse = yy - 2 * a * u + a ** 2 * v
                return float(1 - sse[m].sum() / sst[m].sum())
            a_mod = fit.A * fp.box_profile(pos[k] - fit.c, fit.w, fit.s)
            row = {"v3": pooled(a_mod)}
            row["Ob"] = pooled(np.maximum(u / np.maximum(v, 1e-12), 0))
            lam = (u * a_mod)[m].sum() / max((v * a_mod ** 2)[m].sum(), 1e-12)
            row["Oa"] = pooled(lam * a_mod)
            def img(x):
                im = np.zeros((H, W)); im[rc[:, 0], rc[:, 1]] = np.where(m, x, 0.0); return im
            for L in LS:
                num = box(img(u * a_mod), L)[rc[:, 0], rc[:, 1]]; den = box(img(v * a_mod ** 2), L)[rc[:, 0], rc[:, 1]]
                row[f"Ob-s{L}"] = pooled(np.where(den > 0, num / np.maximum(den, 1e-12), 1.0) * a_mod)
            # centre oracle (amplitude from training): choose c' per pixel on the candidate grid
            best = None
            for c in cand:
                a_c = fit.A * fp.box_profile(pos[k] - c, fit.w, fit.s)
                sse = -2 * a_c * u + a_c ** 2 * v
                if best is None: best = (sse.copy(), np.full(P, c))
                else:
                    better = sse < best[0]; best[0][better] = sse[better]; best[1][better] = c
            c_or = best[1]
            row["Oc"] = pooled(fit.A * fp.box_profile(pos[k] - c_or, fit.w, fit.s))
            for L in LS:
                w_ = img(np.where(a_mod > 0, 1.0, 0.0) + 0.0)
                cs = box(img(c_or * (fit.A > 0)), L)[rc[:, 0], rc[:, 1]] / np.maximum(box(img((fit.A > 0).astype(float)), L)[rc[:, 0], rc[:, 1]], 1e-12)
                row[f"Oc-s{L}"] = pooled(fit.A * fp.box_profile(pos[k] - np.where(np.isfinite(cs) & (cs != 0), cs, fit.c), fit.w, fit.s))
            # profile oracle (global w, s), centres and amplitudes from training
            bestws = max(((w_, s_) for w_, s_ in WS_GRID), key=lambda ws: pooled(fit.A * fp.box_profile(pos[k] - fit.c, ws[0], ws[1])))
            row["Od"] = pooled(fit.A * fp.box_profile(pos[k] - fit.c, bestws[0], bestws[1])); row["Od_ws"] = list(bestws)
            res[k] = row
            log(name, "fold", k, {t: (round(x, 3) if not isinstance(x, list) else x) for t, x in row.items()})
        R["datasets"][name] = {int(k): v for k, v in res.items()}
        # pooled over primary folds needs per-fold sums; re-derive from the per-fold R2 is not exact, so keep the primary pooled via mean of folds weighted equally AND store for the rule below
        del cube
    # decision: use the mean over primary folds of the per-fold pooled R2 (SST-weighting differs a little; stated)
    def gain(ds, tag):
        return float(np.mean([R["datasets"][ds][k][tag] - R["datasets"][ds][k]["v3"] for k in PRIMARY]))
    tags = ["Oa", "Ob", "Oc", "Od"] + [f"Ob-s{L}" for L in LS] + [f"Oc-s{L}" for L in LS]
    G = {ds: {t: gain(ds, t) for t in tags} for ds in DATASETS}
    Gm = {t: float(np.mean([G[ds][t] for ds in DATASETS])) for t in tags}
    dec = {"G_per_dataset": G, "G_mean": Gm}
    Gb = Gm["Ob"]
    dec["normalisation_dominates"] = bool(Gb > 0 and Gm["Oa"] >= 0.5 * Gb)
    bsL = {ds: max(G[ds][f"Ob-s{L}"] for L in LS) / G[ds]["Ob"] if G[ds]["Ob"] > 0 else float("nan") for ds in DATASETS}
    dec["Ob_smooth_recovery_by_dataset"] = bsL
    dec["L1_carried"] = bool(not dec["normalisation_dominates"] and all(v >= 0.6 for v in bsL.values()))
    csL = {ds: max(G[ds][f"Oc-s{L}"] for L in LS) / G[ds]["Oc"] if G[ds]["Oc"] > 0 else float("nan") for ds in DATASETS}
    dec["Oc_smooth_recovery_by_dataset"] = csL
    dec["centre_dominates"] = bool(Gm["Oc"] >= 0.5 * Gb and Gm["Oc"] > Gm["Od"])
    dec["L3_carried"] = bool(dec["centre_dominates"] and all(v >= 0.6 for v in csL.values()) and not dec["L1_carried"] or (dec["centre_dominates"] and all(v >= 0.6 for v in csL.values())))
    R["decision"] = dec
    log("DECISION", json.dumps(dec, default=float))
    json.dump(R, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
