"""Section 14 of PREREGISTER_v3.md: is the physical forward with a flat-top acceptance the same model as v3?

    python equiv_check.py RAW_H5 OUT.json

Two 112 x 112 px ROIs, 8 folds each. Criteria are the registered ones (constants below).
"""
import json
import sys
import time

import numpy as np

import midas_dfxm.forward_maps as fm
import midas_dfxm.forward_profile as fp
import midas_dfxm.forward_thin_slab as ts

ROIS = {"A": (300, 412, 1100, 1212), "B": (20, 132, 400, 512)}
PRIMARY_TOL, FOLD_TOL, CORR_MIN, GAUSS_GAP = 0.02, 0.10, 0.98, 0.5
FLANK = (1, 6)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def physical_cube(p, fit, m, mu_g, chi_g, ob_g, step, lam, flat_top):
    """Weights (n_mu, n_chi, n_ob, P) from the thin slab built on v3's fitted centres."""
    ob_c = ob_g[0] + fit.c * step
    ob_ref = float(np.median(ob_c[m])); th = np.deg2rad(ob_ref) / 2
    mu_ref, chi_ref = float(np.median(p.mu0[m])), float(np.median(p.chi0[m]))
    strain = -(np.deg2rad(ob_c - ob_ref) / 2) / np.tan(th)
    slab = ts.thin_slab_from_maps(p.mu0 - mu_ref, p.chi0 - chi_ref, strain, two_theta0_deg=ob_ref,
                                  wavelength_A=lam, mu_ref_deg=mu_ref, chi_ref_deg=chi_ref)
    tgt = [float(np.median(p.smu[m])), float(np.median(p.schi[m])), float(np.median(p.sob[m]))]
    if flat_top:
        ft = (fit.w * step, fit.s * step)
        sig, _ = ts.match_marginal_widths(slab, mu_g, chi_g, ob_g, tgt, flat_top_ob_deg=ft)
        return ts.render_weights(slab, mu_g, chi_g, ob_g, sigma_mu_deg=sig[0], sigma_chi_deg=sig[1], flat_top_ob_deg=ft)
    sig, _ = ts.match_marginal_widths(slab, mu_g, chi_g, ob_g, tgt)
    return ts.render_weights(slab, mu_g, chi_g, ob_g, sigma_mu_deg=sig[0], sigma_chi_deg=sig[1], sigma_ob_deg=sig[2])


def main(h5, out):
    import h5py, hdf5plugin  # noqa: F401
    with h5py.File(h5, "r") as f:
        lam = float(f.attrs["wavelength_A"])
    R = {"h5": h5, "criteria": dict(primary_tol=PRIMARY_TOL, fold_tol=FOLD_TOL, corr_min=CORR_MIN, gauss_gap=GAUSS_GAP)}
    for name, roi in ROIS.items():
        log("ROI", name, roi)
        d = fm.load_planes_cube(h5, roi=roi, require_consistent_order=False, log=lambda *a: None)
        cube, mu_g, chi_g, ob_g, rc = d["cube"], d["mu_g"], d["chi_g"], d["ob_g"], d["rc"]
        n_ob, P = len(ob_g), cube.shape[-1]
        N = len(mu_g) * len(chi_g)
        step = float(np.median(np.abs(np.diff(np.sort(ob_g)))))
        masks = {k: d["fold_masks"][k][rc[:, 0], rc[:, 1]] for k in range(n_ob)}
        totals = [float(d["totals"][k][d["union"]].sum()) for k in range(n_ob)]
        primary = [int(k) for k in np.argsort(totals)[::-1][:3]]
        v3 = fp.leave_one_plane_out_profile(cube, mu_g, chi_g, ob_g, list(range(n_ob)), masks, nested_sub=P, keep=True)
        res = {"primary": primary, "v3": {k: v3[k]["r2"] for k in v3}, "profile": {k: [v3[k]["w"], v3[k]["s"]] for k in v3},
               "phys_flat": {}, "phys_gauss": {}, "corr": {}}
        sse_p = sst_p = sse_3 = sst_3 = 0.0
        for k in range(n_ob):
            r = v3[k]; p = r["_params"]; fit = r["_fit"]; m = masks[k]
            train = [j for j in range(n_ob) if j != k]
            obs = cube[:, :, k, :].astype(np.float64)
            sst = ((obs - obs.mean((0, 1), keepdims=True)) ** 2).sum((0, 1))
            for tag, flat in (("phys_flat", True), ("phys_gauss", False)):
                W = physical_cube(p, fit, m, mu_g, chi_g, ob_g, step, lam, flat)
                A = ts.fit_amplitudes(W, p.mob_train, p.base_ob, train)
                pred = (p.base_ob / N)[None, None] + A[None, None] * W[:, :, k, :]
                sse = ((obs - pred) ** 2).sum((0, 1))
                res[tag][k] = float(1 - sse[m].sum() / sst[m].sum())
                if tag == "phys_flat":
                    ref = fp.predict_plane_profile(p, fit, r["_pos_k"], r["_b"])
                    a_ = (pred - pred.mean((0, 1)))[:, :, m].ravel(); b_ = (ref - ref.mean((0, 1)))[:, :, m].ravel()
                    res["corr"][k] = float(np.corrcoef(a_, b_)[0, 1])
                    if k in primary:
                        sse_p += sse[m].sum(); sst_p += sst[m].sum()
                        sse_3 += r["sse"][m].sum(); sst_3 += r["sst"][m].sum()
                del W
            log(f"ROI {name} fold {k}: v3 {res['v3'][k]:.3f} phys-flat {res['phys_flat'][k]:.3f} phys-gauss {res['phys_gauss'][k]:.3f} corr {res['corr'][k]:.4f}")
        res["P_v3"] = float(1 - sse_3 / sst_3); res["P_phys_flat"] = float(1 - sse_p / sst_p)
        c1 = abs(res["P_phys_flat"] - res["P_v3"]) <= PRIMARY_TOL
        c2 = all(abs(res["phys_flat"][k] - res["v3"][k]) <= FOLD_TOL for k in range(n_ob))
        c3 = all(res["corr"][k] >= CORR_MIN for k in primary)
        c4 = any(res["phys_gauss"][k] <= res["v3"][k] - GAUSS_GAP for k in FLANK)
        res["criteria"] = dict(primary_pooled=bool(c1), every_fold=bool(c2), per_pixel_corr=bool(c3), gaussian_control_worse=bool(c4))
        R[name] = res
        log("ROI", name, "P_v3", round(res["P_v3"], 3), "P_phys_flat", round(res["P_phys_flat"], 3), res["criteria"])
    R["unit_test_criterion_5"] = "tests/test_forward_thin_slab.py::test_flat_top_acceptance_has_the_requested_obpitch_width_and_edges (passed)"
    R["verdict"] = "CONFIRMED" if all(all(R[n]["criteria"].values()) for n in ROIS) else "NOT CONFIRMED (see per-ROI criteria)"
    log("VERDICT:", R["verdict"])
    json.dump(R, open(out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
