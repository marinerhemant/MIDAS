"""v4: the physical flat-top forward on the whole grain, as registered in PREREGISTER_v4.md.

    python run_physical_full.py RAW_H5 OUTDIR [--n-perm 100] [--nsub-null 5000] [--workers 4] [--seed 0] [--roi r0 r1 c0 c1]
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import midas_dfxm.forward_maps as fm
import midas_dfxm.forward_profile as fp
import midas_dfxm.forward_thin_slab as ts

# registered constants (PREREGISTER_v4.md)
V2_FLOOR, V3_P, V3_TOL = 0.547, 0.695, 0.05
LOSS_REFUTE = 0.05
FLANK, FAR = (1, 6), (0, 7)
V2_FLANK = {1: -6.617, 6: -11.171}
FLANK_MIN, FLANK_GAIN, FAR_MIN = -1.0, 2.0, -0.05
P_CONFIRM, P_REFUTE = 0.02, 0.1
GAUSS_GAP, SYN_TOL, GAIN = 0.5, 0.10, 2.23
BLOCK = 64
EXCLUDE = [(150, 450, 700, 1000), (300, 412, 1100, 1212)]
G = {}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _targets(p, m):
    return [float(np.median(p.smu[m])), float(np.median(p.schi[m])), float(np.median(p.sob[m]))]


def _pw(p, m, idx):
    """per_pixel_widths argument for physical_score (None unless --per-pixel-widths)."""
    if not G["pp"]:
        return None
    t = _targets(p, m)
    return (p.smu[idx], p.schi[idx], t[0], t[1])


def fold_task(args):
    """One (fold, variant): per-pixel SSE and SST of the physical forward on the full fold."""
    import torch
    torch.set_num_threads(G["threads"])
    k, variant = args
    cube, ob_g, mu_g, chi_g, step, lam = G["cube"], G["ob_g"], G["mu_g"], G["chi_g"], G["step"], G["lam"]
    r = G["v3"][k]; p = r["_params"]; fit = r["_fit"]; m = G["masks"][k]
    ft = (fit.w * step, fit.s * step) if variant == "flat" else None
    refs = ts.physical_references(p.mu0, p.chi0, fit.c, m, ob_g, step, lam, mu_g, chi_g, _targets(p, m), flat_top_ob_deg=ft,
                                  couple=G["couple"])
    sse, sst = ts.physical_score(cube[:, :, k, :], p.mu0, p.chi0, fit.c, p.mob_train, p.base_ob, ob_g, k, mu_g, chi_g, refs,
                                 step, lam, flat_top_ob_deg=ft, couple=G["couple"], per_pixel_widths=_pw(p, m, slice(None)))
    t = time.strftime("%H:%M:%S")
    print(f"{t} fold {k} {variant}: R2 {fm.pooled_r2(sse, sst, m):.3f}", flush=True)
    return k, variant, sse[m], sst[m], refs


def null_task(k):
    """Block null for one primary fold on a small random subsample of complete blocks."""
    import torch
    torch.set_num_threads(G["threads"])
    rng = np.random.default_rng(G["seed"] + 100 + k)
    cube, ob_g, mu_g, chi_g, step, lam, rc = G["cube"], G["ob_g"], G["mu_g"], G["chi_g"], G["step"], G["lam"], G["rc"]
    r = G["v3"][k]; p = r["_params"]; fit = r["_fit"]; m = G["masks"][k]
    ft = (fit.w * step, fit.s * step)
    refs = ts.physical_references(p.mu0, p.chi0, fit.c, m, ob_g, step, lam, mu_g, chi_g, _targets(p, m), flat_top_ob_deg=ft,
                                  couple=G["couple"])
    blocks = fm.complete_blocks(G["fold_masks"][k] & G["scored_img"], BLOCK)
    in_blk = np.isin(rc[:, 0] // BLOCK * 100000 + rc[:, 1] // BLOCK,
                     [r0 // BLOCK * 100000 + c0 // BLOCK for r0, c0 in blocks]) & m
    pool_idx = np.flatnonzero(in_blk); rc_pool = rc[pool_idx]
    tab = fm._pool_index(rc_pool, blocks, BLOCK)
    sub_in_pool = np.sort(rng.choice(len(pool_idx), size=min(G["nsub_null"], len(pool_idx)), replace=False))
    sub = pool_idx[sub_in_pool]
    obs = cube[:, :, k, :][:, :, sub]; mob = p.mob_train[:, sub]; base = p.base_ob[sub]
    def score(mu0, chi0, c, idx):
        s_, t_ = ts.physical_score(obs, mu0, chi0, c, mob, base, ob_g, k, mu_g, chi_g, refs, step, lam, flat_top_ob_deg=ft,
                                   couple=G["couple"], per_pixel_widths=_pw(p, m, idx))
        return fm.pooled_r2(s_, t_)
    observed = score(p.mu0[sub], p.chi0[sub], fit.c[sub], sub)
    nul = np.empty(G["n_perm"])
    for i in range(G["n_perm"]):
        d = pool_idx[fm.block_donors(rc_pool, blocks, BLOCK, rng, idx_table=tab)[sub_in_pool]]
        nul[i] = score(p.mu0[d], p.chi0[d], fit.c[d], d)
    out = dict(observed_on_subsample=observed, n_scored=int(len(sub)), n_blocks=len(blocks), mean=float(nul.mean()),
               sd=float(nul.std()), max=float(nul.max()), p=float((1 + (nul >= observed).sum()) / (1 + len(nul))))
    print(time.strftime("%H:%M:%S"), f"null fold {k}: observed {observed:.3f} null {nul.mean():.3f}+-{nul.std():.3f} p {out['p']:.4f}", flush=True)
    return k, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5"); ap.add_argument("outdir")
    ap.add_argument("--n-perm", type=int, default=100); ap.add_argument("--nsub-null", type=int, default=5000)
    ap.add_argument("--nsub", type=int, default=40_000); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roi", type=int, nargs=4, default=None, help="smoke tests only")
    ap.add_argument("--no-couple", action="store_true", help="v5: switch off the obpitch-to-rock coupling")
    ap.add_argument("--per-pixel-widths", action="store_true", help="v5: per-voxel rock and roll widths")
    ap.add_argument("--version", default="v4", choices=["v4", "v5"])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    import h5py, hdf5plugin  # noqa: F401
    with h5py.File(a.h5, "r") as f:
        lam = float(f.attrs["wavelength_A"])
    R = {"h5": a.h5, "seed": a.seed, "n_perm": a.n_perm, "nsub_null": a.nsub_null, "version": a.version,
         "couple": not a.no_couple, "per_pixel_widths": a.per_pixel_widths}
    log("loading planes")
    d = fm.load_planes_cube(a.h5, roi=tuple(a.roi) if a.roi else None, require_consistent_order=a.roi is None, log=log)
    cube, mu_g, chi_g, ob_g, rc = d["cube"], d["mu_g"], d["chi_g"], d["ob_g"], d["rc"]
    n_ob, P = len(ob_g), cube.shape[-1]
    step = float(np.median(np.abs(np.diff(np.sort(ob_g)))))
    keep = np.ones(len(rc), bool)
    if a.roi is None:
        for r0, r1, c0, c1 in EXCLUDE:
            keep &= ~((rc[:, 0] >= r0) & (rc[:, 0] < r1) & (rc[:, 1] >= c0) & (rc[:, 1] < c1))
    masks = {k: d["fold_masks"][k][rc[:, 0], rc[:, 1]] & keep for k in range(n_ob)}
    scored_img = np.zeros(d["union"].shape, bool); scored_img[rc[:, 0], rc[:, 1]] = keep
    totals_union = [float(d["totals"][k][d["union"]].sum()) for k in range(n_ob)]
    primary = [int(k) for k in np.argsort(totals_union)[::-1][:3]]
    R.update(planes=d["planes"], primary=primary, excluded_px=int((~keep).sum()), n_pixels_union=int(P))
    log("primary:", primary, "excluded px:", R["excluded_px"])

    log("v3 leave-one-plane-out (re-derives centres and (w, s))")
    v3 = fp.leave_one_plane_out_profile(cube, mu_g, chi_g, ob_g, list(range(n_ob)), masks, nested_sub=a.nsub, seed=a.seed, keep=True, log=log)
    R["v3_r2_rederived"] = {int(k): v3[k]["r2"] for k in v3}
    R["v3_profile"] = {int(k): [v3[k]["w"], v3[k]["s"]] for k in v3}
    v2 = fm.leave_one_plane_out(cube, mu_g, chi_g, ob_g, list(range(n_ob)), masks)
    R["v2_r2_rederived"] = {int(k): v2[k]["full"]["r2"] for k in v2}
    R["P_v2"] = float(1 - sum(v2[k]["full"]["sse"][masks[k]].sum() for k in primary) / sum(v2[k]["full"]["sst"][masks[k]].sum() for k in primary))
    R["P_v3"] = float(1 - sum(v3[k]["sse"][masks[k]].sum() for k in primary) / sum(v3[k]["sst"][masks[k]].sum() for k in primary))
    del v2
    G.update(cube=cube, ob_g=ob_g, mu_g=mu_g, chi_g=chi_g, step=step, lam=lam, v3=v3, masks=masks, rc=rc,
             fold_masks=d["fold_masks"], scored_img=scored_img, seed=a.seed, n_perm=a.n_perm, nsub_null=a.nsub_null, threads=a.threads,
             couple=not a.no_couple, pp=a.per_pixel_widths)

    log("physical forward, flat-top and Gaussian acceptance, all folds, workers", a.workers)
    ctx = mp.get_context("fork")
    tasks = [(k, v) for k in range(n_ob) for v in ("flat", "gauss")]
    with ctx.Pool(a.workers) as pool:
        out = pool.map(fold_task, tasks, chunksize=1)
        log("block null on the primary folds")
        nulls = dict(pool.map(null_task, primary, chunksize=1))
    res = {"flat": {}, "gauss": {}}; sums = {"flat": [0.0, 0.0], "gauss": [0.0, 0.0]}; refs_out = {}
    for k, v, sse, sst, refs in out:
        res[v][k] = float(1 - sse.sum() / sst.sum())
        if k in primary:
            sums[v][0] += sse.sum(); sums[v][1] += sst.sum()
        if v == "flat":
            refs_out[k] = refs
    R["phys_flat_r2"] = res["flat"]; R["phys_gauss_r2"] = res["gauss"]
    R["P_phys_flat"] = float(1 - sums["flat"][0] / sums["flat"][1]); R["P_phys_gauss"] = float(1 - sums["gauss"][0] / sums["gauss"][1])
    R["refs_flat"] = refs_out; R["null"] = nulls
    log("physical flat R2:", {k: round(v, 3) for k, v in res["flat"].items()}, "primary", round(R["P_phys_flat"], 3))
    log("physical gauss R2:", {k: round(v, 3) for k, v in res["gauss"].items()}, "primary", round(R["P_phys_gauss"], 3))

    # ---- per-pixel agreement with v3 on a subsample of each primary fold (reported)
    R["corr_with_v3"] = {}
    rng = np.random.default_rng(a.seed + 7)
    for k in primary:
        r = v3[k]; p = r["_params"]; fit = r["_fit"]; m = masks[k]
        sub = np.sort(rng.choice(np.flatnonzero(m), size=min(20_000, int(m.sum())), replace=False))
        ft = (fit.w * step, fit.s * step)
        refs = ts.physical_references(p.mu0, p.chi0, fit.c, m, ob_g, step, lam, mu_g, chi_g, _targets(p, m), flat_top_ob_deg=ft,
                                      couple=G["couple"])
        _, _, pp = ts.physical_score(cube[:, :, k, :][:, :, sub], p.mu0[sub], p.chi0[sub], fit.c[sub], p.mob_train[:, sub], p.base_ob[sub],
                                     ob_g, k, mu_g, chi_g, refs, step, lam, flat_top_ob_deg=ft, return_pred=True,
                                     couple=G["couple"], per_pixel_widths=_pw(p, m, sub))
        ref = fp.predict_plane_profile(p.take(sub), fp.ProfileFit(c=fit.c[sub], A=fit.A[sub], w=fit.w, s=fit.s, at_edge=0.0), r["_pos_k"], r["_b"][sub])
        x = (pp - pp.mean((0, 1))).ravel(); y = (ref - ref.mean((0, 1))).ravel()
        R["corr_with_v3"][int(k)] = float(np.corrcoef(x, y)[0, 1])
    log("corr with v3:", R["corr_with_v3"])

    # ---- matched synthetic control
    log("matched synthetic control")
    sub_c = np.sort(rng.choice(P, size=min(a.nsub_null, P), replace=False))
    cube_c = cube[:, :, :, sub_c]
    p_t = fm.fit_separable_params(cube_c, mu_g, chi_g, ob_g, list(range(n_ob)))
    S_t = fm._gauss(p_t.mu_g[:, None], p_t.mu0[None], p_t.smu[None]).sum(0) * fm._gauss(p_t.chi_g[:, None], p_t.chi0[None], p_t.schi[None]).sum(0)
    sig_t = p_t.mob_train - p_t.base_ob[None]
    pos_all = fp._positions(ob_g, range(n_ob))
    w_t, s_t, _ = fp.choose_profile(sig_t, S_t, pos_all)
    c_t, _A = fp.fit_centres(sig_t, S_t, pos_all, w_t, s_t)
    all_m = np.ones(len(sub_c), bool)
    ft_t = (w_t * step, s_t * step)
    cp, ppw = (not a.no_couple), a.per_pixel_widths
    tg_t = _targets(p_t, all_m)
    refs_t = ts.physical_references(p_t.mu0, p_t.chi0, c_t, all_m, ob_g, step, lam, mu_g, chi_g, tg_t, flat_top_ob_deg=ft_t, couple=cp)
    th_t = np.deg2rad(refs_t["ob_ref"]) / 2
    slab_t = ts.thin_slab_from_maps(p_t.mu0 - refs_t["mu_ref"], p_t.chi0 - refs_t["chi_ref"],
                                    -(np.deg2rad(ob_g[0] + c_t * step - refs_t["ob_ref"]) / 2) / np.tan(th_t),
                                    two_theta0_deg=refs_t["ob_ref"], wavelength_A=lam, mu_ref_deg=refs_t["mu_ref"], chi_ref_deg=refs_t["chi_ref"])
    smu_t = p_t.smu * refs_t["sigma_mu"] / tg_t[0] if ppw else refs_t["sigma_mu"]
    schi_t = p_t.schi * refs_t["sigma_chi"] / tg_t[1] if ppw else refs_t["sigma_chi"]
    W_t = ts.render_weights(slab_t, mu_g, chi_g, ob_g, sigma_mu_deg=smu_t, sigma_chi_deg=schi_t, flat_top_ob_deg=ft_t, couple=cp)
    A_t = ts.fit_amplitudes(W_t, p_t.mob_train, p_t.base_ob, list(range(n_ob)))
    b_t = p_t.base_ob / (len(mu_g) * len(chi_g))
    far_plane = int(np.argmax(np.abs(ob_g - np.median(p_t.ob0))))
    noise_var0 = fm.read_noise_floor(cube[:, :, far_plane, :][:, :, sub_c])
    gen = np.random.default_rng(a.seed + 31)
    sig_sig = A_t[None, None, None] * W_t
    syn = (b_t[None, None, None] + sig_sig + np.sqrt(noise_var0 + GAIN * np.maximum(sig_sig, 0)) * gen.standard_normal(sig_sig.shape)).astype(np.float32)
    vs = fp.leave_one_plane_out_profile(syn, mu_g, chi_g, ob_g, primary, {k: all_m for k in range(n_ob)}, nested_sub=len(sub_c), seed=a.seed, keep=True)
    sse_p = sst_p = sse_o = sst_o = 0.0
    for k in primary:
        r = vs[k]; p = r["_params"]; fit = r["_fit"]
        refs = ts.physical_references(p.mu0, p.chi0, fit.c, all_m, ob_g, step, lam, mu_g, chi_g, _targets(p, all_m),
                                      flat_top_ob_deg=(fit.w * step, fit.s * step), couple=cp)
        tg_p = _targets(p, all_m)
        s_, t_ = ts.physical_score(syn[:, :, k, :], p.mu0, p.chi0, fit.c, p.mob_train, p.base_ob, ob_g, k, mu_g, chi_g, refs, step, lam,
                                   flat_top_ob_deg=(fit.w * step, fit.s * step), couple=cp,
                                   per_pixel_widths=(p.smu, p.schi, tg_p[0], tg_p[1]) if ppw else None)
        sse_p += s_.sum(); sst_p += t_.sum()
        y = syn[:, :, k, :].astype(np.float64); pred_o = b_t[None, None] + A_t[None, None] * W_t[:, :, k, :]
        sse_o += ((y - pred_o) ** 2).sum(); sst_o += ((y - y.mean((0, 1), keepdims=True)) ** 2).sum()
    R["control_synthetic"] = dict(truth_w=w_t, truth_s=s_t, noise_var0=noise_var0, pipeline_primary=float(1 - sse_p / sst_p),
                                  oracle_primary=float(1 - sse_o / sst_o))
    log("synthetic control:", R["control_synthetic"])

    # ---- verdict (registered rules only)
    Pp = R["P_phys_flat"]; r1 = res["flat"]
    V4_P, V4_FLANK = 0.624, {1: -0.658, 6: -0.336}
    v5 = a.version == "v5"
    crit = dict(
        no_loss_vs_v2=bool(Pp >= V2_FLOOR and (not v5 or Pp >= V4_P - 0.01)),
        agrees_with_v3=bool(abs(Pp - V3_P) <= V3_TOL),
        flank=all(r1[k] >= FLANK_MIN and r1[k] >= V2_FLANK[k] + FLANK_GAIN and (not v5 or r1[k] >= V4_FLANK[k] - 0.15) for k in FLANK),
        far=all(r1[k] >= FAR_MIN for k in FAR),
        block_null=all(nulls[k]["p"] < P_CONFIRM for k in primary),
        gaussian_control=any(res["gauss"][k] <= r1[k] - GAUSS_GAP for k in FLANK),
        synthetic=bool(abs(R["control_synthetic"]["pipeline_primary"] - R["control_synthetic"]["oracle_primary"]) <= SYN_TOL))
    refuted = []
    if Pp < V2_FLOOR - LOSS_REFUTE: refuted.append(f"primary loss {Pp:.3f}")
    if v5 and Pp < V4_P - LOSS_REFUTE: refuted.append(f"worse than v4 by more than {LOSS_REFUTE}: {Pp:.3f}")
    for k in FLANK:
        if r1[k] < V2_FLANK[k] + FLANK_GAIN: refuted.append(f"flank fold {k} not improved by {FLANK_GAIN}: {r1[k]:.3f}")
    if sum(nulls[k]["p"] >= P_REFUTE for k in primary) >= 2: refuted.append("block null not rejected on >= 2 primary planes")
    verdict = "REFUTED" if refuted else ("CONFIRMED" if all(crit.values()) else "INCONCLUSIVE")
    R["verdict"] = dict(verdict=verdict, criteria=crit, refuted_reasons=refuted, P_phys_flat=Pp)
    log("VERDICT:", R["verdict"])
    with open(os.path.join(a.outdir, f"results_{a.version}.json"), "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main()
