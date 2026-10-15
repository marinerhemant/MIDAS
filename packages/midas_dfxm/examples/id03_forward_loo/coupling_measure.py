"""v6 part 1: the obpitch-rock coupling strength alpha, measured without a model (PREREGISTER_v6.md).

    python coupling_measure.py RAW_H5 OUT.json [--nsub-ctrl 30000] [--seed 0] [--roi r0 r1 c0 c1]
"""
import argparse
import json
import time

import numpy as np

import midas_dfxm.forward_maps as fm
import midas_dfxm.forward_profile as fp
import midas_dfxm.forward_thin_slab as ts

S_MIN, N_LIT_MIN, EDGE_BINS, BASE_FRAC, BLOCK, N_BOOT = 2000.0, 3, 2, 0.30, 64, 2000
ROI_A, ROI_B = (300, 412, 1100, 1212), (20, 132, 400, 512)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def plane_stats(plane, mu_g):
    """plane (n_mu, n_chi, P) -> total S above baseline, mu centroid, 'centre not at the window edge' flag."""
    M = plane.sum(1, dtype=np.float64)                                   # (n_mu, P)
    k = max(1, int(round(BASE_FRAC * len(mu_g))))
    base = np.sort(M, 0)[:k].mean(0)
    Mc = np.maximum(M - base, 0)
    S = Mc.sum(0)
    mu = (Mc * mu_g[:, None]).sum(0) / np.maximum(S, 1e-12)
    ok = (mu >= mu_g[EDGE_BINS]) & (mu <= mu_g[-1 - EDGE_BINS])
    return S, mu, ok


def per_pixel_sums(S, mu, ok, ob):
    """per-pixel numerator and denominator of the pooled weighted slope; used flag. S, mu, ok: (n_ob, P); ob: (n_ob,) or (n_ob, P)."""
    lit = (S >= S_MIN) & ok
    w = np.where(lit, S, 0.0)
    ob2 = np.broadcast_to(ob if ob.ndim == 2 else ob[:, None], S.shape)
    sw = np.maximum(w.sum(0), 1e-12)
    obb = (w * ob2).sum(0) / sw; mub = (w * mu).sum(0) / sw
    num = (w * (ob2 - obb) * (mu - mub)).sum(0)
    den = (w * (ob2 - obb) ** 2).sum(0)
    used = lit.sum(0) >= N_LIT_MIN
    return np.where(used, num, 0.0), np.where(used, den, 0.0), used


def boot_slope(num, den, bid, rng, n_boot=N_BOOT):
    """block sums and bootstrap of the ratio of sums over blocks. bid: (P,) integer block id per pixel."""
    ub, inv = np.unique(bid, return_inverse=True)
    nb = np.bincount(inv, weights=num, minlength=len(ub)); db = np.bincount(inv, weights=den, minlength=len(ub))
    point = nb.sum() / max(db.sum(), 1e-12)
    keep = db > 0
    nb, db = nb[keep], db[keep]
    idx = rng.integers(0, len(nb), size=(n_boot, len(nb)))
    bs = nb[idx].sum(1) / np.maximum(db[idx].sum(1), 1e-12)
    return float(point), bs, int(keep.sum())


def summarize(num, den, used, bid, rng):
    point, bs, nblk = boot_slope(num, den, bid, rng)
    return dict(slope=point, ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], sd=float(bs.std()),
                n_pixels=int(used.sum()), n_blocks=nblk), bs


def estimate(cube, mu_g, ob_g, rc, rng, ob_override=None):
    S = np.empty((cube.shape[2], cube.shape[3])); mu = np.empty_like(S); ok = np.empty(S.shape, bool)
    for j in range(cube.shape[2]):
        S[j], mu[j], ok[j] = plane_stats(cube[:, :, j, :], mu_g)
    ob = ob_g if ob_override is None else ob_override
    num, den, used = per_pixel_sums(S, mu, ok, ob)
    bid = (rc[:, 0] // BLOCK) * 100000 + rc[:, 1] // BLOCK
    return num, den, used, bid, S, mu, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5"); ap.add_argument("out")
    ap.add_argument("--nsub-ctrl", type=int, default=30_000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roi", type=int, nargs=4, default=None, help="smoke tests only")
    a = ap.parse_args()
    import h5py, hdf5plugin  # noqa: F401
    with h5py.File(a.h5, "r") as f:
        lam = float(f.attrs["wavelength_A"])
    rng = np.random.default_rng(a.seed)
    log("loading")
    d = fm.load_planes_cube(a.h5, roi=tuple(a.roi) if a.roi else None, require_consistent_order=a.roi is None, log=lambda *x: None)
    cube, mu_g, chi_g, ob_g, rc = d["cube"], d["mu_g"], d["chi_g"], d["ob_g"], d["rc"]
    n_ob, P = len(ob_g), cube.shape[-1]
    step = float(np.median(np.abs(np.diff(np.sort(ob_g)))))
    R = {"h5": a.h5, "n_pixels_union": int(P), "S_MIN": S_MIN, "N_LIT_MIN": N_LIT_MIN}

    log("measured slope, whole grain")
    num, den, used, bid, S, mu, ok = estimate(cube, mu_g, ob_g, rc, rng)
    R["measured"], bs_meas = summarize(num, den, used, bid, rng)
    R["measured"]["lit_planes_hist"] = np.bincount(((S >= S_MIN) & ok).sum(0), minlength=n_ob + 1).tolist()
    log("measured", R["measured"])
    # strata
    strata = {"left": rc[:, 1] < np.median(rc[used, 1]), "right": rc[:, 1] >= np.median(rc[used, 1]),
              "top": rc[:, 0] < np.median(rc[used, 0]), "bottom": rc[:, 0] >= np.median(rc[used, 0])}
    if a.roi is None:
        for nm, (r0, r1, c0, c1) in (("ROI_A", ROI_A), ("ROI_B", ROI_B)):
            strata[nm] = (rc[:, 0] >= r0) & (rc[:, 0] < r1) & (rc[:, 1] >= c0) & (rc[:, 1] < c1)
    R["strata"] = {}
    for nm, m in strata.items():
        if (used & m).sum() < 200:
            R["strata"][nm] = {"n_pixels": int((used & m).sum()), "note": "too few pixels"}; continue
        R["strata"][nm], _ = summarize(np.where(m, num, 0), np.where(m, den, 0), used & m, bid, rng)
        log("stratum", nm, R["strata"][nm]["slope"], R["strata"][nm]["ci95"])
    # C-b shuffle of the obpitch labels within pixel
    perm = np.argsort(rng.random((n_ob, P)), 0)
    ob_shuf = np.take_along_axis(np.broadcast_to(ob_g[:, None], (n_ob, P)), perm, 0)
    num_s, den_s, used_s, _, _, _, _ = estimate(cube, mu_g, ob_g, rc, rng, ob_override=ob_shuf)
    R["control_shuffle"], _ = summarize(num_s, den_s, used_s, bid, rng)
    log("shuffle control", R["control_shuffle"])

    # C-a synthetic alpha controls
    log("synthetic controls")
    sub = np.sort(rng.choice(P, size=min(a.nsub_ctrl, P), replace=False))
    cube_c = cube[:, :, :, sub]; rc_c = rc[sub]
    p_t = fm.fit_separable_params(cube_c, mu_g, chi_g, ob_g, list(range(n_ob)))
    S_t = fm._gauss(p_t.mu_g[:, None], p_t.mu0[None], p_t.smu[None]).sum(0) * fm._gauss(p_t.chi_g[:, None], p_t.chi0[None], p_t.schi[None]).sum(0)
    sig_t = p_t.mob_train - p_t.base_ob[None]
    pos_all = fp._positions(ob_g, range(n_ob))
    w_t, s_t, _ = fp.choose_profile(sig_t, S_t, pos_all)
    c_t, _A = fp.fit_centres(sig_t, S_t, pos_all, w_t, s_t)
    allm = np.ones(len(sub), bool); ft = (w_t * step, s_t * step)
    tg = [float(np.median(p_t.smu)), float(np.median(p_t.schi)), float(np.median(p_t.sob))]
    far = int(np.argmax(np.abs(ob_g - np.median(p_t.ob0))))
    noise_var0 = fm.read_noise_floor(cube[:, :, far, :][:, :, sub])
    b_t = p_t.base_ob / (len(mu_g) * len(chi_g))
    R["controls_alpha"] = {"truth_w": w_t, "truth_s": s_t, "noise_var0": noise_var0}
    gen = np.random.default_rng(a.seed + 5)
    ctrl_bs = {}
    for alpha in (0.0, 0.5, 1.0):
        refs = ts.physical_references(p_t.mu0, p_t.chi0, c_t, allm, ob_g, step, lam, mu_g, chi_g, tg, flat_top_ob_deg=ft, couple=alpha)
        th = np.deg2rad(refs["ob_ref"]) / 2
        slab = ts.thin_slab_from_maps(p_t.mu0 - refs["mu_ref"], p_t.chi0 - refs["chi_ref"],
                                      -(np.deg2rad(ob_g[0] + c_t * step - refs["ob_ref"]) / 2) / np.tan(th), two_theta0_deg=refs["ob_ref"],
                                      wavelength_A=lam, mu_ref_deg=refs["mu_ref"], chi_ref_deg=refs["chi_ref"])
        W = ts.render_weights(slab, mu_g, chi_g, ob_g, sigma_mu_deg=p_t.smu * refs["sigma_mu"] / tg[0], sigma_chi_deg=p_t.schi * refs["sigma_chi"] / tg[1],
                              flat_top_ob_deg=ft, couple=alpha)
        A = ts.fit_amplitudes(W, p_t.mob_train, p_t.base_ob, list(range(n_ob)))
        sg = A[None, None, None] * W
        syn = (b_t[None, None, None] + sg + np.sqrt(noise_var0 + 2.23 * np.maximum(sg, 0)) * gen.standard_normal(sg.shape)).astype(np.float32)
        n_, d_, u_, bid_, _, _, _ = estimate(syn, mu_g, ob_g, rc_c, rng)
        R["controls_alpha"][f"alpha_{alpha}"], bsc = summarize(n_, d_, u_, bid_, rng)
        ctrl_bs[alpha] = bsc
        log("control alpha", alpha, R["controls_alpha"][f"alpha_{alpha}"])
        del W, syn
    ref_slope = R["controls_alpha"]["alpha_1.0"]["slope"]
    R["controls_alpha"]["ratios_to_alpha1"] = {str(k): R["controls_alpha"][f"alpha_{k}"]["slope"] / ref_slope for k in (0.0, 0.5, 1.0)}

    # alpha_hat: independent bootstrap draws of measured and alpha=1 control slope
    n = min(len(bs_meas), len(ctrl_bs[1.0]))
    ah = bs_meas[:n] / ctrl_bs[1.0][rng.permutation(len(ctrl_bs[1.0]))[:n]]
    alpha_hat = R["measured"]["slope"] / ref_slope
    R["alpha_hat"] = dict(value=float(alpha_hat), ci95=[float(np.percentile(ah, 2.5)), float(np.percentile(ah, 97.5))], reference_slope_alpha1=ref_slope)

    def classify(lo, hi):
        if -0.15 <= lo and hi <= 0.15: return "alpha=0 supported"
        if 0.85 <= lo and hi <= 1.15: return "alpha=1 supported"
        return "intermediate"
    R["classification_whole_grain"] = classify(*R["alpha_hat"]["ci95"])
    R["strata_alpha"] = {}
    for nm, v in R["strata"].items():
        if "slope" in v:
            lo, hi = v["ci95"][0] / ref_slope, v["ci95"][1] / ref_slope
            R["strata_alpha"][nm] = dict(value=v["slope"] / ref_slope, ci95=[lo, hi], classification=classify(lo, hi))
    ctrl_ok = (abs(R["controls_alpha"]["ratios_to_alpha1"]["0.0"]) <= 0.1 and abs(R["controls_alpha"]["ratios_to_alpha1"]["0.5"] - 0.5) <= 0.1)
    shuf = R["control_shuffle"]; shuf_ok = abs(shuf["slope"]) <= 2 * shuf["sd"]
    cls = {v["classification"] for v in R["strata_alpha"].values()}
    strata_agree = all(v["classification"] == R["classification_whole_grain"] for v in R["strata_alpha"].values())
    R["checks"] = dict(synthetic_alpha_controls_ok=bool(ctrl_ok), shuffle_ok=bool(shuf_ok), strata_agree=bool(strata_agree))
    R["carried"] = bool(ctrl_ok and shuf_ok and strata_agree and R["classification_whole_grain"] == "intermediate")
    log("ALPHA", json.dumps({"alpha_hat": R["alpha_hat"], "class": R["classification_whole_grain"], "checks": R["checks"], "carried": R["carried"]}))
    json.dump(R, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
