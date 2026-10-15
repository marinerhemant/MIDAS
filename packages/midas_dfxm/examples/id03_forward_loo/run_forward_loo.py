"""Leave-one-plane-out forward test of the inverted ID03 maps, as registered in PREREGISTER.md.

    python run_forward_loo.py RAW_H5 OUTDIR [--n-perm 200] [--nsub 40000] [--seed 0]

Reads every plane of the 8-plane ID03 file, runs the 8 folds on all grain pixels, the block and
pixel permutation nulls and the synthetic controls on the three primary planes, and writes
``results.json`` (every number, plus the verdict computed from the registered rules) and the
residual maps. Nothing here picks a threshold: they are the constants below, copied from
PREREGISTER.md section 5 and the v2 amendment (section 10).
"""
import argparse
import json
import os
import time

import numpy as np

import midas_dfxm.forward_maps as fm

R2_ABS = 0.50          # section 5: R^2_full >= 0.5
GAIN_OVER_N0 = 0.15    # section 5: R^2_full - R^2_N0 >= 0.15
REFUTE_GAIN = 0.05     # section 5: refuted if the gain is below this
P_CONFIRM, P_REFUTE = 0.01, 0.05
CONTROL_DROP = 0.10    # section 5: misspecified control must fall this far below the ceiling
ORACLE_DROP = 0.20     # section 10: the oracle must see the twin control or the clause is void
GAINS = {"g0": 0.0, "g2.23": 2.23, "g22.3": 22.3}   # section 10 item 3
JUDGED_GAIN = "g2.23"
TWIN = (3.0, 1.0, "chi")  # section 11: second component at +3 sigma_chi, equal height
EXCLUDE_ROI = (150, 450, 700, 1000)   # section 10 item 4: the smoke ROI, crop coordinates
BLOCK = 64


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def per_pixel_r(obs, p, ob_k, chunk=40_000):
    """Pearson r between prediction and observation over each pixel's 650 frames."""
    A, b = fm.amplitude_and_baseline(p)
    P = obs.shape[-1]
    out = np.empty(P)
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        pr = fm.predict_plane(p.take(slice(s, e)), A[s:e], b[s:e], ob_k).reshape(-1, e - s)
        y = obs[:, :, s:e].reshape(-1, e - s).astype(np.float64)
        pr = pr - pr.mean(0); y = y - y.mean(0)
        den = np.sqrt((pr ** 2).sum(0) * (y ** 2).sum(0))
        out[s:e] = np.where(den > 0, (pr * y).sum(0) / np.maximum(den, 1e-30), np.nan)
    return out


def pooled(res, folds, variant):
    sse = sum(res[k][variant]["sse"][MASKS[k]].sum() for k in folds)
    sst = sum(res[k][variant]["sst"][MASKS[k]].sum() for k in folds)
    return float(1 - sse / sst)


def main():
    global MASKS
    ap = argparse.ArgumentParser()
    ap.add_argument("h5"); ap.add_argument("outdir")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--nsub", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roi", type=int, nargs=4, default=None, help="smoke tests only")
    ap.add_argument("--no-exclude", action="store_true", help="smoke tests only: score the smoke ROI too")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    R = {"h5": a.h5, "seed": a.seed, "n_perm": a.n_perm, "nsub": a.nsub}

    log("loading planes")
    d = fm.load_planes_cube(a.h5, roi=tuple(a.roi) if a.roi else None, log=log)
    cube, mu_g, chi_g, ob_g, rc = d["cube"], d["mu_g"], d["chi_g"], d["ob_g"], d["rc"]
    n_ob, P = len(ob_g), cube.shape[-1]
    keep = np.ones(len(rc), bool)
    if a.roi is None and not a.no_exclude:
        r0, r1, c0, c1 = EXCLUDE_ROI
        keep = ~((rc[:, 0] >= r0) & (rc[:, 0] < r1) & (rc[:, 1] >= c0) & (rc[:, 1] < c1))
        log(f"smoke ROI excluded from scoring: {int((~keep).sum())} px")
    R["excluded_px"] = int((~keep).sum())
    MASKS = {k: d["fold_masks"][k][rc[:, 0], rc[:, 1]] & keep for k in range(n_ob)}
    scored_img = np.zeros(d["union"].shape, bool); scored_img[rc[:, 0], rc[:, 1]] = keep
    totals_union = [float(d["totals"][k][d["union"]].sum()) for k in range(n_ob)]
    primary = [int(k) for k in np.argsort(totals_union)[::-1][:3]]
    R.update(planes=d["planes"], ob_g=ob_g.tolist(), mu_g=mu_g.tolist(), chi_g=chi_g.tolist(),
             n_pixels_union=int(P), totals_union=totals_union, primary=primary,
             order_checks=[c.message.splitlines()[0] for c in d["order_checks"]],
             fold_mask_px={k: int(MASKS[k].sum()) for k in range(n_ob)})
    log("primary planes (by total intensity, blind to the model):", primary)

    variants = {
        "moment_widths": fm.moment_width_params,
        "uniform": lambda p, m: fm.uniform_params(p, m),
        "abl_smu": lambda p, m: fm.ablate_params(p, m, ["smu"]),
        "abl_schi": lambda p, m: fm.ablate_params(p, m, ["schi"]),
        "abl_sob": lambda p, m: fm.ablate_params(p, m, ["sob"]),
        "abl_widths": lambda p, m: fm.ablate_params(p, m, ["smu", "schi", "sob"]),
        "abl_centres": lambda p, m: fm.ablate_params(p, m, ["mu0", "chi0", "ob0"]),
    }
    log("leave-one-plane-out, all folds, all grain pixels")
    res = fm.leave_one_plane_out(cube, mu_g, chi_g, ob_g, list(range(n_ob)), MASKS, variants,
                                 keep_params=True)
    R["per_fold_r2"] = {int(k): {v: res[k][v]["r2"] for v in ["full", *variants]} for k in res}
    R["floored_fraction"] = {int(k): res[k]["_params"].floored for k in res}
    R["primary_pooled_r2"] = {v: pooled(res, primary, v) for v in ["full", *variants]}
    log("per-fold R2 (full):", {k: round(res[k]["full"]["r2"], 3) for k in res})
    log("primary pooled:", R["primary_pooled_r2"])

    # ---- per-pixel r, residual maps and where the misfit sits
    from scipy import ndimage
    H, W = d["union"].shape
    R["per_pixel"] = {}
    for k in primary:
        p = res[k]["_params"]
        m = MASKS[k]
        r = per_pixel_r(cube[:, :, k, :], p, ob_g[k])
        r2pix = 1 - res[k]["full"]["sse"] / np.maximum(res[k]["full"]["sst"], 1e-12)
        rmap = np.full((H, W), np.nan, np.float32); rmap[rc[:, 0], rc[:, 1]] = np.where(m, r, np.nan)
        r2map = np.full((H, W), np.nan, np.float32); r2map[rc[:, 0], rc[:, 1]] = np.where(m, r2pix, np.nan)
        np.savez_compressed(os.path.join(a.outdir, f"maps_fold{k}.npz"), r=rmap, r2=r2map,
                            mu0=p.mu0, chi0=p.chi0, ob0=p.ob0, smu=p.smu, schi=p.schi, sob=p.sob,
                            rc=rc, mask=m)
        edge = ndimage.distance_transform_edt(d["fold_masks"][k])[rc[:, 0], rc[:, 1]]
        c = np.clip(r2pix[m], -1, 1)
        R["per_pixel"][int(k)] = dict(
            median_r=float(np.nanmedian(r[m])), p05_r=float(np.nanpercentile(r[m], 5)),
            frac_r2_negative=float(np.mean(r2pix[m] < 0)),
            frac_neg_within_10px_of_edge=float(np.mean(edge[m][r2pix[m] < 0] <= 10)) if (r2pix[m] < 0).any() else None,
            frac_all_within_10px_of_edge=float(np.mean(edge[m] <= 10)),
            corr_r2_with_row=float(np.corrcoef(c, rc[m, 0])[0, 1]),
            corr_r2_with_col=float(np.corrcoef(c, rc[m, 1])[0, 1]))
        log(f"fold {k}: median r {R['per_pixel'][int(k)]['median_r']:.3f}, "
            f"frac R2<0 {R['per_pixel'][int(k)]['frac_r2_negative']:.3f}")

    # ---- permutation nulls (block and pixel) on a subsample of complete blocks
    log("permutation nulls")
    R["null"] = {}
    for k in primary:
        p = res[k]["_params"]
        blocks = fm.complete_blocks(d["fold_masks"][k] & scored_img, BLOCK)
        pool_idx = np.flatnonzero(
            np.isin(rc[:, 0] // BLOCK * 100000 + rc[:, 1] // BLOCK,
                    [r0 // BLOCK * 100000 + c0 // BLOCK for r0, c0 in blocks]) & MASKS[k])
        rc_pool = rc[pool_idx]
        # complete blocks only, in the order complete_blocks lists them (row-major)
        order = np.lexsort((rc_pool[:, 1] % BLOCK, rc_pool[:, 0] % BLOCK,
                            rc_pool[:, 1] // BLOCK, rc_pool[:, 0] // BLOCK))
        pool_idx, rc_pool = pool_idx[order], rc_pool[order]
        tab = fm._pool_index(rc_pool, blocks, BLOCK)
        sub_in_pool = np.sort(rng.choice(len(pool_idx), size=min(a.nsub, len(pool_idx)), replace=False))
        sub_idx = pool_idx[sub_in_pool]
        own, pool = p.take(sub_idx), p.take(pool_idx)
        obs = cube[:, :, k, :][:, :, sub_idx]
        sse, sst = fm.score_plane(obs, own, ob_g[k])
        observed = fm.pooled_r2(sse, sst)
        out = {"observed_on_subsample": observed, "n_blocks": len(blocks), "n_pool": int(len(pool_idx)),
               "n_scored": int(len(sub_idx))}
        for name, fn in (("block", lambda r: fm.block_donors(rc_pool, blocks, BLOCK, r, idx_table=tab)),
                         ("pixel", lambda r: fm.pixel_donors(len(pool_idx), r))):
            nul = fm.permutation_null(obs, own, pool, ob_g[k], fn, a.n_perm, rng,
                                      score_idx_in_pool=sub_in_pool)["r2"]
            out[name] = dict(mean=float(nul.mean()), sd=float(nul.std()), max=float(nul.max()),
                             p=float((1 + (nul >= observed).sum()) / (1 + len(nul))))
        # the uniform model on the same pixels, for the like-for-like gain
        un = fm.uniform_params(p, MASKS[k])
        A, b = fm.amplitude_and_baseline(un.take(sub_idx))
        s2, t2 = fm.score_plane(obs, un.take(sub_idx), ob_g[k], A=A, b=b)
        out["uniform_on_subsample"] = fm.pooled_r2(s2, t2)
        R["null"][int(k)] = out
        log(f"fold {k}: observed {observed:.3f}, block null {out['block']['mean']:.3f}+-{out['block']['sd']:.3f} "
            f"(p {out['block']['p']:.4f}), pixel null {out['pixel']['mean']:.3f}")

    # ---- noise for the controls, read from the data
    log("noise and synthetic controls")
    ob0_all = np.median(fm.fit_separable_params(cube[:, :, :, rng.choice(P, 20000, replace=False)],
                                                mu_g, chi_g, ob_g, list(range(n_ob))).ob0)
    far = int(np.argmax(np.abs(ob_g - ob0_all)))
    sub_c = np.sort(rng.choice(P, size=min(a.nsub, P), replace=False))
    noise_var0 = fm.read_noise_floor(cube[:, :, far, :][:, :, sub_c])
    preds, resids, bases = [], [], []
    for k in primary:
        p = res[k]["_params"].take(sub_c)
        A, b = fm.amplitude_and_baseline(p)
        pr = fm.predict_plane(p, A, b, ob_g[k])
        preds.append(pr.ravel()); resids.append((cube[:, :, k, :][:, :, sub_c] - pr).ravel())
        bases.append(np.broadcast_to(b[None, None], pr.shape).ravel())
    pr_all, rs_all, b_all = map(np.concatenate, (preds, resids, bases))
    gain_up, intercept_up = fm.gain_from_residuals(pr_all.reshape(-1, 1, 1), b_all.reshape(-1, 1, 1),
                                                    rs_all.reshape(-1, 1, 1))
    R["noise"] = dict(far_plane=far, ob0_median=float(ob0_all), sigma0_sq=noise_var0,
                      gain_residual_diagnostic_only=gain_up, intercept_residual_diagnostic_only=intercept_up,
                      gains_swept=GAINS)
    log("noise:", R["noise"])

    cube_c = cube[:, :, :, sub_c]
    p_true = fm.fit_separable_params(cube_c, mu_g, chi_g, ob_g, list(range(n_ob)))
    A_t, b_t = fm.amplitude_and_baseline(p_true)
    all_true = np.ones(len(sub_c), bool)
    R["controls"] = {}
    for tag, gain in GAINS.items():
        for kind, sp in (("matched", None), ("twin", TWIN)):
            syn = fm.synthetic_cube(p_true, ob_g, noise_var0=noise_var0, gain=gain,
                                    rng=np.random.default_rng(a.seed + 11), second_peak=sp)
            pipe = fm.leave_one_plane_out(syn, mu_g, chi_g, ob_g, primary, {k: all_true for k in primary})
            sse = sum(pipe[k]["full"]["sse"].sum() for k in primary)
            sst = sum(pipe[k]["full"]["sst"].sum() for k in primary)
            sse_o = sst_o = 0.0
            for k in primary:        # oracle: the true parameters scored directly
                s_, t_ = fm.score_plane(syn[:, :, k, :], p_true, ob_g[k], A=A_t, b=b_t)
                sse_o += s_.sum(); sst_o += t_.sum()
            R["controls"][f"{kind}_{tag}"] = dict(pipeline_r2=float(1 - sse / sst), oracle_r2=float(1 - sse_o / sst_o))
            log(f"control {kind}/{tag}: pipeline {1 - sse / sst:.3f}  oracle {1 - sse_o / sst_o:.3f}")
        del syn
    ceiling = R["controls"]["matched_g0"]["pipeline_r2"]

    # ---- verdict, from the registered rules only (PREREGISTER.md sections 5 and 10)
    Rp = R["primary_pooled_r2"]; gain = Rp["full"] - Rp["uniform"]
    ctl = R["controls"]
    j_ = JUDGED_GAIN
    drop_pipe = ctl[f"matched_{j_}"]["pipeline_r2"] - ctl[f"twin_{j_}"]["pipeline_r2"]
    drop_oracle = ctl[f"matched_{j_}"]["oracle_r2"] - ctl[f"twin_{j_}"]["oracle_r2"]
    control_valid = drop_oracle >= ORACLE_DROP
    pb = [R["null"][int(k)]["block"]["p"] for k in primary]
    refuted_reasons = []
    if gain < REFUTE_GAIN: refuted_reasons.append(f"gain over N0 {gain:.3f} < {REFUTE_GAIN}")
    if sum(p_ >= P_REFUTE for p_ in pb) >= 2: refuted_reasons.append(f"block null not rejected on >=2 planes (p={pb})")
    ok = dict(abs=Rp["full"] >= R2_ABS, gain=gain >= GAIN_OVER_N0, block=all(p_ < P_CONFIRM for p_ in pb),
              control_valid=control_valid, control_sees_twin=drop_pipe >= CONTROL_DROP)
    if refuted_reasons:
        verdict = "REFUTED"
    elif all(ok.values()):
        verdict = "CONFIRMED"
    else:
        verdict = "INCONCLUSIVE"
    R["verdict"] = dict(verdict=verdict, gain_over_N0=gain, R2_full=Rp["full"], R2_N0=Rp["uniform"],
                        R2_moment_widths_v1=Rp["moment_widths"], R2_fraction_of_noiseless_pipeline_ceiling=Rp["full"] / ceiling,
                        noiseless_pipeline_ceiling=ceiling, block_p=pb, drop_pipeline=drop_pipe,
                        drop_oracle=drop_oracle, criteria=ok, refuted_reasons=refuted_reasons)
    log("VERDICT:", R["verdict"])
    with open(os.path.join(a.outdir, "results.json"), "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
