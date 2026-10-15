"""v6: v5 + coupling alpha = 0.153 + local-pooled amplitude, whole grain, as registered in PREREGISTER_v6.md.

    python run_physical_v6.py RAW_H5 OUTDIR [--workers 4] [--threads 2] [--n-perm 100] [--nsub-null 5000] [--roi r0 r1 c0 c1]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "v4"))   # fallback only: the sibling copy wins
import run_physical_full as rp                      # shared helpers, constants and the block null task
import midas_dfxm.forward_maps as fm
import midas_dfxm.forward_profile as fp
import midas_dfxm.forward_thin_slab as ts

ALPHA = 0.153
LGRID = (1, 8, 16, 32, 64)
V5_P = 0.699; V5_FLANK = {1: -0.339, 6: -0.197}
BAR, FLANK_DROP = 0.02, 0.15
G = rp.G


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _box(img, L):
    return img if L == 1 else ndimage.uniform_filter(img, size=L, mode="constant", cval=0.0)


def smoothed_amplitude(Wt_tr, S_tr, grain_img, rc, L):
    """local-pooled least squares amplitude: box_L(sum_j S_j Wt_j) / box_L(sum_j Wt_j^2) on the fold's grain image."""
    num = (S_tr * Wt_tr).sum(0); den = (Wt_tr ** 2).sum(0)
    H, W = grain_img.shape
    def im(x):
        out = np.zeros((H, W)); out[rc[:, 0], rc[:, 1]] = x; return out * grain_img
    nb = _box(im(num), L)[rc[:, 0], rc[:, 1]]; db = _box(im(den), L)[rc[:, 0], rc[:, 1]]
    return np.maximum(nb / np.maximum(db, 1e-30), 0.0)


def v6_task(args):
    import torch
    torch.set_num_threads(G["threads"])
    k, variant = args                                   # variant: v6, alpha, L1, gauss
    alpha = 0.0 if variant == "L1" else ALPHA
    use_L1 = variant in ("v6", "L1", "gauss")
    flat = variant != "gauss"
    cube, ob_g, mu_g, chi_g, step, lam, rc = (G[x] for x in ("cube", "ob_g", "mu_g", "chi_g", "step", "lam", "rc"))
    r = G["v3"][k]; p = r["_params"]; fit = r["_fit"]; m = G["masks"][k]
    ft = (fit.w * step, fit.s * step) if flat else None
    refs = ts.physical_references(p.mu0, p.chi0, fit.c, m, ob_g, step, lam, mu_g, chi_g, rp._targets(p, m), flat_top_ob_deg=ft, couple=alpha)
    pw = rp._pw(p, m, slice(None))
    Lbest, errs = 1, None
    amp = None
    if use_L1:
        n_ob = len(ob_g); train = [j for j in range(n_ob) if j != k]
        Wt = ts.physical_totals(p.mu0, p.chi0, fit.c, ob_g, mu_g, chi_g, refs, step, lam, flat_top_ob_deg=ft, couple=alpha, per_pixel_widths=pw)
        S_tr = r["_sig"]; Wt_tr = Wt[train]
        grain = G["fold_masks"][k][rc[:, 0], rc[:, 1]].astype(float)
        gimg = np.zeros(G["scored_img"].shape); gimg[rc[:, 0], rc[:, 1]] = grain
        errs = {}
        for L in LGRID:
            e = 0.0
            for t in range(len(train)):
                rest = [i for i in range(len(train)) if i != t]
                A_ = smoothed_amplitude(Wt_tr[rest], S_tr[rest], gimg, rc, L)
                e += float(((S_tr[t] - A_ * Wt_tr[t]) ** 2)[m].sum())
            errs[L] = e
        Lbest = min(errs, key=errs.get)
        amp = smoothed_amplitude(Wt_tr, S_tr, gimg, rc, Lbest)
    sse, sst = ts.physical_score(cube[:, :, k, :], p.mu0, p.chi0, fit.c, p.mob_train, p.base_ob, ob_g, k, mu_g, chi_g, refs, step, lam,
                                 flat_top_ob_deg=ft, couple=alpha, per_pixel_widths=pw, amplitude=amp)
    print(time.strftime("%H:%M:%S"), f"fold {k} {variant}: R2 {fm.pooled_r2(sse, sst, m):.3f} L {Lbest}", flush=True)
    return k, variant, sse[m], sst[m], Lbest, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5"); ap.add_argument("outdir")
    ap.add_argument("--n-perm", type=int, default=100); ap.add_argument("--nsub-null", type=int, default=5000)
    ap.add_argument("--nsub", type=int, default=40_000); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roi", type=int, nargs=4, default=None, help="smoke tests only")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    import h5py, hdf5plugin  # noqa: F401
    with h5py.File(a.h5, "r") as f:
        lam = float(f.attrs["wavelength_A"])
    R = {"h5": a.h5, "alpha": ALPHA, "seed": a.seed}
    log("loading")
    d = fm.load_planes_cube(a.h5, roi=tuple(a.roi) if a.roi else None, require_consistent_order=a.roi is None, log=lambda *x: None)
    cube, mu_g, chi_g, ob_g, rc = d["cube"], d["mu_g"], d["chi_g"], d["ob_g"], d["rc"]
    n_ob, P = len(ob_g), cube.shape[-1]
    step = float(np.median(np.abs(np.diff(np.sort(ob_g)))))
    keep = np.ones(len(rc), bool)
    if a.roi is None:
        for r0, r1, c0, c1 in rp.EXCLUDE:
            keep &= ~((rc[:, 0] >= r0) & (rc[:, 0] < r1) & (rc[:, 1] >= c0) & (rc[:, 1] < c1))
    masks = {k: d["fold_masks"][k][rc[:, 0], rc[:, 1]] & keep for k in range(n_ob)}
    scored_img = np.zeros(d["union"].shape, bool); scored_img[rc[:, 0], rc[:, 1]] = keep
    totals_union = [float(d["totals"][k][d["union"]].sum()) for k in range(n_ob)]
    primary = [int(k) for k in np.argsort(totals_union)[::-1][:3]]
    R.update(planes=d["planes"], primary=primary, excluded_px=int((~keep).sum()))
    log("v3 leave-one-plane-out (centres and (w, s))")
    v3 = fp.leave_one_plane_out_profile(cube, mu_g, chi_g, ob_g, list(range(n_ob)), masks, nested_sub=a.nsub, seed=a.seed, keep=True, log=log)
    R["v3_r2_rederived"] = {int(k): v3[k]["r2"] for k in v3}
    R["P_v3"] = float(1 - sum(v3[k]["sse"][masks[k]].sum() for k in primary) / sum(v3[k]["sst"][masks[k]].sum() for k in primary))
    G.update(cube=cube, ob_g=ob_g, mu_g=mu_g, chi_g=chi_g, step=step, lam=lam, v3=v3, masks=masks, rc=rc, fold_masks=d["fold_masks"],
             scored_img=scored_img, seed=a.seed, n_perm=a.n_perm, nsub_null=a.nsub_null, threads=a.threads, couple=ALPHA, pp=True)

    flank_folds = (1, 6); decomp = (1, 3, 4, 5, 6)
    tasks = [(k, "v6") for k in range(n_ob)] + [(k, "alpha") for k in decomp] + [(k, "L1") for k in decomp] + [(k, "gauss") for k in flank_folds]
    log("v6 tasks:", len(tasks), "workers", a.workers)
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers) as pool:
        out = pool.map(v6_task, tasks, chunksize=1)
        log("block null (primary folds, alpha = 0.153, unsmoothed amplitude)")
        nulls = dict(pool.map(rp.null_task, primary, chunksize=1))
    res = {}; sums = {}; Ls = {}
    for k, v, sse, sst, Lb, errs in out:
        res.setdefault(v, {})[k] = float(1 - sse.sum() / sst.sum())
        if k in primary:
            s_ = sums.setdefault(v, [0.0, 0.0]); s_[0] += sse.sum(); s_[1] += sst.sum()
        if errs is not None:
            Ls.setdefault(v, {})[k] = dict(L=Lb, errors={str(L): e for L, e in errs.items()})
    R["r2"] = res; R["chosen_L"] = Ls; R["null"] = nulls
    R["P"] = {v: float(1 - s[0] / s[1]) for v, s in sums.items()}
    log("R2 v6:", {k: round(x, 3) for k, x in sorted(res["v6"].items())}, "primary", round(R["P"]["v6"], 3))
    for v in ("alpha", "L1", "gauss"):
        if v in res: log("R2", v, {k: round(x, 3) for k, x in sorted(res[v].items())}, "primary", round(R["P"].get(v, float("nan")), 3))

    # ---- matched synthetic (alpha = 0.153, unsmoothed amplitude): same chain as run_physical_full with the coupling strength
    log("matched synthetic control")
    rng = np.random.default_rng(a.seed + 7)
    sub_c = np.sort(rng.choice(P, size=min(a.nsub_null, P), replace=False))
    cube_c = cube[:, :, :, sub_c]
    p_t = fm.fit_separable_params(cube_c, mu_g, chi_g, ob_g, list(range(n_ob)))
    S_t = fm._gauss(p_t.mu_g[:, None], p_t.mu0[None], p_t.smu[None]).sum(0) * fm._gauss(p_t.chi_g[:, None], p_t.chi0[None], p_t.schi[None]).sum(0)
    sig_t = p_t.mob_train - p_t.base_ob[None]
    pos_all = fp._positions(ob_g, range(n_ob))
    w_t, s_t, _ = fp.choose_profile(sig_t, S_t, pos_all)
    c_t, _A = fp.fit_centres(sig_t, S_t, pos_all, w_t, s_t)
    all_m = np.ones(len(sub_c), bool); ft_t = (w_t * step, s_t * step)
    tg_t = rp._targets(p_t, all_m)
    refs_t = ts.physical_references(p_t.mu0, p_t.chi0, c_t, all_m, ob_g, step, lam, mu_g, chi_g, tg_t, flat_top_ob_deg=ft_t, couple=ALPHA)
    th_t = np.deg2rad(refs_t["ob_ref"]) / 2
    slab_t = ts.thin_slab_from_maps(p_t.mu0 - refs_t["mu_ref"], p_t.chi0 - refs_t["chi_ref"],
                                    -(np.deg2rad(ob_g[0] + c_t * step - refs_t["ob_ref"]) / 2) / np.tan(th_t), two_theta0_deg=refs_t["ob_ref"],
                                    wavelength_A=lam, mu_ref_deg=refs_t["mu_ref"], chi_ref_deg=refs_t["chi_ref"])
    W_t = ts.render_weights(slab_t, mu_g, chi_g, ob_g, sigma_mu_deg=p_t.smu * refs_t["sigma_mu"] / tg_t[0], sigma_chi_deg=p_t.schi * refs_t["sigma_chi"] / tg_t[1],
                            flat_top_ob_deg=ft_t, couple=ALPHA)
    A_t = ts.fit_amplitudes(W_t, p_t.mob_train, p_t.base_ob, list(range(n_ob)))
    b_t = p_t.base_ob / (len(mu_g) * len(chi_g))
    far = int(np.argmax(np.abs(ob_g - np.median(p_t.ob0))))
    nv0 = fm.read_noise_floor(cube[:, :, far, :][:, :, sub_c])
    sg = A_t[None, None, None] * W_t
    syn = (b_t[None, None, None] + sg + np.sqrt(nv0 + rp.GAIN * np.maximum(sg, 0)) * rng.standard_normal(sg.shape)).astype(np.float32)
    vs = fp.leave_one_plane_out_profile(syn, mu_g, chi_g, ob_g, primary, {k: all_m for k in range(n_ob)}, nested_sub=len(sub_c), seed=a.seed, keep=True)
    sse_p = sst_p = sse_o = sst_o = 0.0
    for k in primary:
        r_ = vs[k]; p_ = r_["_params"]; fit_ = r_["_fit"]; tg_p = rp._targets(p_, all_m)
        refs = ts.physical_references(p_.mu0, p_.chi0, fit_.c, all_m, ob_g, step, lam, mu_g, chi_g, tg_p, flat_top_ob_deg=(fit_.w * step, fit_.s * step), couple=ALPHA)
        s_, t_ = ts.physical_score(syn[:, :, k, :], p_.mu0, p_.chi0, fit_.c, p_.mob_train, p_.base_ob, ob_g, k, mu_g, chi_g, refs, step, lam,
                                   flat_top_ob_deg=(fit_.w * step, fit_.s * step), couple=ALPHA, per_pixel_widths=(p_.smu, p_.schi, tg_p[0], tg_p[1]))
        sse_p += s_.sum(); sst_p += t_.sum()
        y = syn[:, :, k, :].astype(np.float64); po = b_t[None, None] + A_t[None, None] * W_t[:, :, k, :]
        sse_o += ((y - po) ** 2).sum(); sst_o += ((y - y.mean((0, 1), keepdims=True)) ** 2).sum()
    R["control_synthetic"] = dict(pipeline_primary=float(1 - sse_p / sst_p), oracle_primary=float(1 - sse_o / sst_o))
    log("synthetic:", R["control_synthetic"])

    # ---- verdict (registered rules only)
    Pv = R["P"]["v6"]; r6 = res["v6"]
    crit = dict(
        primary_gain=bool(Pv >= V5_P + BAR),
        flank_not_lower=all(r6[k] >= V5_FLANK[k] - FLANK_DROP for k in flank_folds),
        far_and_flank=all(r6[k] >= rp.FAR_MIN for k in rp.FAR) and all(r6[k] >= rp.FLANK_MIN and r6[k] >= rp.V2_FLANK[k] + rp.FLANK_GAIN for k in flank_folds),
        block_null=all(nulls[k]["p"] < rp.P_CONFIRM for k in primary),
        gaussian_control=any(res["gauss"][k] <= r6[k] - rp.GAUSS_GAP for k in flank_folds),
        synthetic=bool(abs(R["control_synthetic"]["pipeline_primary"] - R["control_synthetic"]["oracle_primary"]) <= rp.SYN_TOL))
    refuted = []
    if Pv < V5_P - 0.05: refuted.append(f"primary loss {Pv:.3f}")
    for k in flank_folds:
        if r6[k] < rp.V2_FLANK[k] + rp.FLANK_GAIN: refuted.append(f"flank fold {k} not improved by 2 over v2: {r6[k]:.3f}")
    if sum(nulls[k]["p"] >= rp.P_REFUTE for k in primary) >= 2: refuted.append("block null not rejected on >= 2 primary planes")
    verdict = "REFUTED" if refuted else ("CONFIRMED" if all(crit.values()) else "INCONCLUSIVE")
    R["verdict"] = dict(verdict=verdict, criteria=crit, refuted_reasons=refuted, P_v6=Pv, P_v5_reference=V5_P, gain_over_v5=Pv - V5_P,
                        no_improvement=bool(abs(Pv - V5_P) < BAR))
    log("VERDICT:", R["verdict"])
    with open(os.path.join(a.outdir, "results_v6.json"), "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
