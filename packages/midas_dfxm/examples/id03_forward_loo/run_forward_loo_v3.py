"""v3 leave-one-plane-out forward test with the flat-top obpitch profile, as registered in PREREGISTER_v3.md.

    python run_forward_loo_v3.py RAW_H5 OUTDIR [--n-perm 100] [--nsub 40000] [--seed 0]

Every threshold below is copied from PREREGISTER_v3.md. The v2 Gaussian pipeline is rescored on the same pixels.
"""
import argparse
import json
import os
import time

import numpy as np

import midas_dfxm.forward_maps as fm
import midas_dfxm.forward_profile as fp

# registered constants (PREREGISTER_v3.md)
PRIMARY_FLOOR = 0.548
FAR = (0, 1, 7)
NEAR = (2, 6)
FAR_MIN = -0.05
NEAR_MIN = 0.0
P_CONFIRM, P_REFUTE = 0.02, 0.1
LOSS_REFUTE = 0.05
NEAR_GAIN_REFUTE = 2.0
C1_PRIMARY_TOL, C1_FAR_TOL = 0.10, 0.05
GAIN = 2.23
BLOCK = 64
EXCLUDE = [(150, 450, 700, 1000), (300, 412, 1100, 1212)]     # v2 smoke ROI, v3 diagnosis ROI (crop coordinates)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5"); ap.add_argument("outdir")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--nsub", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roi", type=int, nargs=4, default=None, help="smoke tests only")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    R = {"h5": a.h5, "seed": a.seed, "n_perm": a.n_perm, "nsub": a.nsub}

    log("loading planes")
    d = fm.load_planes_cube(a.h5, roi=tuple(a.roi) if a.roi else None,
                           require_consistent_order=a.roi is None, log=log)
    cube, mu_g, chi_g, ob_g, rc = d["cube"], d["mu_g"], d["chi_g"], d["ob_g"], d["rc"]
    n_ob, P = len(ob_g), cube.shape[-1]
    keep = np.ones(len(rc), bool)
    if a.roi is None:
        for r0, r1, c0, c1 in EXCLUDE:
            keep &= ~((rc[:, 0] >= r0) & (rc[:, 0] < r1) & (rc[:, 1] >= c0) & (rc[:, 1] < c1))
    R["excluded_px"] = int((~keep).sum())
    masks = {k: d["fold_masks"][k][rc[:, 0], rc[:, 1]] & keep for k in range(n_ob)}
    scored_img = np.zeros(d["union"].shape, bool); scored_img[rc[:, 0], rc[:, 1]] = keep
    totals_union = [float(d["totals"][k][d["union"]].sum()) for k in range(n_ob)]
    primary = [int(k) for k in np.argsort(totals_union)[::-1][:3]]
    R.update(planes=d["planes"], ob_g=ob_g.tolist(), n_pixels_union=int(P), primary=primary,
             fold_mask_px={k: int(masks[k].sum()) for k in range(n_ob)})
    log("primary (0-based fold idx):", primary, "| excluded px:", R["excluded_px"])

    # ---- v2 (Gaussian) rescored on the same pixels
    log("v2 rescoring")
    v2 = fm.leave_one_plane_out(cube, mu_g, chi_g, ob_g, list(range(n_ob)), masks)
    R["v2_r2"] = {int(k): v2[k]["full"]["r2"] for k in v2}
    pool = lambda res, v: 1 - sum(res[k][v]["sse"][masks[k]].sum() for k in primary) / sum(res[k][v]["sst"][masks[k]].sum() for k in primary)
    R["P_v2"] = float(pool(v2, "full"))
    log("v2 R2:", {k: round(v, 3) for k, v in R["v2_r2"].items()}, "primary pooled", round(R["P_v2"], 3))

    # ---- v3
    log("v3 leave-one-plane-out")
    v3 = fp.leave_one_plane_out_profile(cube, mu_g, chi_g, ob_g, list(range(n_ob)), masks, nested_sub=a.nsub,
                                        seed=a.seed, keep=True, log=log)
    R["v3_r2"] = {int(k): v3[k]["r2"] for k in v3}
    R["v3_profile"] = {int(k): dict(w=v3[k]["w"], s=v3[k]["s"], at_edge=v3[k]["at_edge"]) for k in v3}
    R["P_v3"] = float(1 - sum(v3[k]["sse"][masks[k]].sum() for k in primary) / sum(v3[k]["sst"][masks[k]].sum() for k in primary))
    log("v3 R2:", {k: round(v, 3) for k, v in R["v3_r2"].items()}, "primary pooled", round(R["P_v3"], 3))

    # ---- v3 ablations on the primary folds (reported)
    abl = {"abl_widths": lambda p, fit, m: (fm.ablate_params(p, m, ["smu", "schi"]), fit),
           "abl_mu_chi_centres": lambda p, fit, m: (fm.ablate_params(p, m, ["mu0", "chi0"]), fit)}
    R["v3_ablation"] = {}
    for name, f in abl.items():
        sse_t = sst_t = 0.0
        for k in primary:
            r = v3[k]; q, fit = f(r["_params"], r["_fit"], masks[k])
            S = fm._gauss(q.mu_g[:, None], q.mu0[None], q.smu[None]).sum(0) * fm._gauss(q.chi_g[:, None], q.chi0[None], q.schi[None]).sum(0)
            A = fp.amplitudes_for_centres(r["_sig"], S, r["_pos_tr"], fit.c, fit.w, fit.s)
            fit2 = fp.ProfileFit(c=fit.c, A=A, w=fit.w, s=fit.s, at_edge=0.0)
            sse, sst = fp.score_plane_profile(cube[:, :, k, :], q, fit2, r["_pos_k"], r["_b"])
            sse_t += sse[masks[k]].sum(); sst_t += sst[masks[k]].sum()
        R["v3_ablation"][name] = float(1 - sse_t / sst_t)
    log("v3 ablations:", R["v3_ablation"])

    # ---- block null on the primary folds
    log("block null")
    R["null"] = {}
    for k in primary:
        r = v3[k]; p = r["_params"]; fit = r["_fit"]
        blocks = fm.complete_blocks(d["fold_masks"][k] & scored_img, BLOCK)
        in_blk = np.isin(rc[:, 0] // BLOCK * 100000 + rc[:, 1] // BLOCK,
                         [r0 // BLOCK * 100000 + c0 // BLOCK for r0, c0 in blocks]) & masks[k]
        pool_idx = np.flatnonzero(in_blk)
        rc_pool = rc[pool_idx]
        tab = fm._pool_index(rc_pool, blocks, BLOCK)
        sub_in_pool = np.sort(rng.choice(len(pool_idx), size=min(a.nsub, len(pool_idx)), replace=False))
        sub = pool_idx[sub_in_pool]
        own = p.take(sub); pool_p = p.take(pool_idx)
        obs = cube[:, :, k, :][:, :, sub]
        b_sub = r["_b"][sub]; sig_sub = r["_sig"][:, sub]
        fit_sub = fp.ProfileFit(c=fit.c[sub], A=fit.A[sub], w=fit.w, s=fit.s, at_edge=0.0)
        s0, t0 = fp.score_plane_profile(obs, own, fit_sub, r["_pos_k"], b_sub)
        observed = fm.pooled_r2(s0, t0)
        nul = np.empty(a.n_perm)
        for i in range(a.n_perm):
            donor = fm.block_donors(rc_pool, blocks, BLOCK, rng, idx_table=tab)[sub_in_pool]
            q = fm.swap_bundle(own, pool_p, donor)
            c_q = fit.c[pool_idx[donor]]
            S_q = fm._gauss(q.mu_g[:, None], q.mu0[None], q.smu[None]).sum(0) * fm._gauss(q.chi_g[:, None], q.chi0[None], q.schi[None]).sum(0)
            A_q = fp.amplitudes_for_centres(sig_sub, S_q, r["_pos_tr"], c_q, fit.w, fit.s)
            fq = fp.ProfileFit(c=c_q, A=A_q, w=fit.w, s=fit.s, at_edge=0.0)
            s1, t1 = fp.score_plane_profile(obs, q, fq, r["_pos_k"], b_sub)
            nul[i] = fm.pooled_r2(s1, t1)
        R["null"][int(k)] = dict(observed_on_subsample=observed, n_scored=int(len(sub)), n_blocks=len(blocks),
                                 mean=float(nul.mean()), sd=float(nul.std()), max=float(nul.max()),
                                 p=float((1 + (nul >= observed).sum()) / (1 + len(nul))))
        log(f"fold {k}: observed {observed:.3f} null {nul.mean():.3f}+-{nul.std():.3f} p {R['null'][int(k)]['p']:.4f}")

    # ---- controls: synthetic from the 8-plane v3 fit of a random pixel subsample
    log("controls")
    sub_c = np.sort(rng.choice(P, size=min(a.nsub, P), replace=False))
    cube_c = cube[:, :, :, sub_c]
    p_true = fm.fit_separable_params(cube_c, mu_g, chi_g, ob_g, list(range(n_ob)))
    S_t = fm._gauss(p_true.mu_g[:, None], p_true.mu0[None], p_true.smu[None]).sum(0) * fm._gauss(p_true.chi_g[:, None], p_true.chi0[None], p_true.schi[None]).sum(0)
    sig_t = p_true.mob_train - p_true.base_ob[None]
    pos_all = fp._positions(ob_g, range(n_ob))
    w_t, s_t, _ = fp.choose_profile(sig_t, S_t, pos_all)
    c_t, A_t = fp.fit_centres(sig_t, S_t, pos_all, w_t, s_t)
    fit_true = fp.ProfileFit(c=c_t, A=A_t, w=w_t, s=s_t, at_edge=0.0)
    b_t = p_true.base_ob / (len(mu_g) * len(chi_g))
    far_plane = int(np.argmax(np.abs(ob_g - np.median(p_true.ob0))))
    noise_var0 = fm.read_noise_floor(cube[:, :, far_plane, :][:, :, sub_c])
    R["controls"] = dict(truth_w=w_t, truth_s=s_t, noise_var0=noise_var0, far_plane=far_plane, gain=GAIN)
    all_true = {k: np.ones(len(sub_c), bool) for k in range(n_ob)}
    for name, gfwhm in (("C1_matched", None), ("C2_gaussian_truth", float(w_t))):
        syn = fp.synthetic_cube_profile(p_true, fit_true, ob_g, b_t, noise_var0=noise_var0, gain=GAIN,
                                        rng=np.random.default_rng(a.seed + 21), gaussian_fwhm=gfwhm)
        pipe = fp.leave_one_plane_out_profile(syn, mu_g, chi_g, ob_g, list(range(n_ob)), all_true,
                                              nested_sub=len(sub_c), seed=a.seed)
        v2c = fm.leave_one_plane_out(syn, mu_g, chi_g, ob_g, list(range(n_ob)), all_true)
        orc = {}
        for k in range(n_ob):
            sse, sst = fp.score_plane_profile(syn[:, :, k, :], p_true, fit_true, pos_all[k], b_t)
            orc[k] = (sse.sum(), sst.sum())
        R["controls"][name] = dict(
            pipeline_r2={k: pipe[k]["r2"] for k in range(n_ob)},
            oracle_r2={k: float(1 - orc[k][0] / orc[k][1]) for k in range(n_ob)},
            v2_pipeline_r2={k: v2c[k]["full"]["r2"] for k in range(n_ob)},
            pipeline_primary=float(1 - sum(pipe[k]["sse"].sum() for k in primary) / sum(pipe[k]["sst"].sum() for k in primary)),
            oracle_primary=float(1 - sum(orc[k][0] for k in primary) / sum(orc[k][1] for k in primary)))
        log(name, "primary pipeline/oracle", round(R["controls"][name]["pipeline_primary"], 3), round(R["controls"][name]["oracle_primary"], 3))
        del syn

    # ---- verdict (registered rules only)
    v3r, v2r = R["v3_r2"], R["v2_r2"]
    c1 = R["controls"]["C1_matched"]
    crit = dict(
        primary_no_loss=bool(R["P_v3"] >= R["P_v2"] and R["P_v3"] >= PRIMARY_FLOOR),
        far_planes=all(v3r[k] >= FAR_MIN for k in FAR),
        near_flank=all(v3r[k] >= NEAR_MIN for k in NEAR),
        block_null=all(R["null"][k]["p"] < P_CONFIRM for k in primary),
        c1=bool(abs(c1["pipeline_primary"] - c1["oracle_primary"]) <= C1_PRIMARY_TOL and
                all(abs(c1["pipeline_r2"][k] - c1["oracle_r2"][k]) <= C1_FAR_TOL for k in FAR)))
    refuted = []
    if R["P_v3"] < R["P_v2"] - LOSS_REFUTE: refuted.append(f"primary loss: {R['P_v3']:.3f} < v2 {R['P_v2']:.3f} - {LOSS_REFUTE}")
    for k in NEAR:
        if v3r[k] < v2r[k] + NEAR_GAIN_REFUTE: refuted.append(f"near flank {k}: {v3r[k]:.3f} not >= v2 {v2r[k]:.3f} + {NEAR_GAIN_REFUTE}")
    if sum(R["null"][k]["p"] >= P_REFUTE for k in primary) >= 2: refuted.append("block null not rejected on >= 2 primary planes")
    verdict = "REFUTED" if refuted else ("CONFIRMED" if all(crit.values()) else "INCONCLUSIVE")
    R["verdict"] = dict(verdict=verdict, criteria=crit, refuted_reasons=refuted, P_v3=R["P_v3"], P_v2=R["P_v2"])
    log("VERDICT:", R["verdict"])
    with open(os.path.join(a.outdir, "results_v3.json"), "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
