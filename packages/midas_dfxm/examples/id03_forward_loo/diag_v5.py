"""v5 part A: test the three candidates on pixels already excluded from scoring (PREREGISTER_v5.md).

    python diag_v5.py RAW_H5 OUT.json
"""
import json
import sys
import time

import numpy as np

import midas_dfxm.forward_maps as fm
import midas_dfxm.forward_profile as fp
import midas_dfxm.forward_thin_slab as ts

DATASETS = {"smoke": ((150, 450, 700, 1000), 25_000), "notebook": ((300, 412, 1100, 1212), None)}
FOLDS = (1, 2, 3, 4, 5, 6)
PRIMARY, FLANK = (3, 4, 5), (1, 6)
VARIANTS = {"P0": dict(couple=True, pp=False), "Pa": dict(couple=False, pp=False),
            "Pb": dict(couple=True, pp=True), "Pab": dict(couple=False, pp=True)}
CARRY_FRAC, FLANK_DROP, AMP_FRAC, GAP_MIN = 0.30, 0.15, 0.70, 0.02


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def r2(obs, pred, b=None, oracle=False):
    y = obs.astype(np.float64); p_ = pred.astype(np.float64)
    if oracle:
        bb = b[None, None]
        num = ((y - bb) * (p_ - bb)).sum((0, 1)); den = np.maximum(((p_ - bb) ** 2).sum((0, 1)), 1e-12)
        p_ = bb + np.maximum(num / den, 0)[None, None] * (p_ - bb)
    sse = ((y - p_) ** 2).sum((0, 1)); sst = ((y - y.mean((0, 1), keepdims=True)) ** 2).sum((0, 1))
    return sse, sst


def main(h5, out):
    import h5py, hdf5plugin  # noqa: F401
    with h5py.File(h5, "r") as f:
        lam = float(f.attrs["wavelength_A"])
    R = {"datasets": {}}
    for name, (roi, nsub) in DATASETS.items():
        log("dataset", name, roi)
        d = fm.load_planes_cube(h5, roi=roi, require_consistent_order=False, log=lambda *a: None)
        cube, mu_g, chi_g, ob_g, rc = d["cube"], d["mu_g"], d["chi_g"], d["ob_g"], d["rc"]
        n_ob, P = len(ob_g), cube.shape[-1]
        step = float(np.median(np.abs(np.diff(np.sort(ob_g)))))
        masks = {k: d["fold_masks"][k][rc[:, 0], rc[:, 1]] for k in range(n_ob)}
        v3 = fp.leave_one_plane_out_profile(cube, mu_g, chi_g, ob_g, list(FOLDS), masks, nested_sub=40_000, keep=True)
        rng = np.random.default_rng(0)
        res = {}
        for k in FOLDS:
            r = v3[k]; p = r["_params"]; fit = r["_fit"]; m = masks[k]
            idx = np.flatnonzero(m)
            if nsub is not None and len(idx) > nsub:
                idx = np.sort(rng.choice(idx, size=nsub, replace=False))
            obs = cube[:, :, k, :][:, :, idx]
            b = (p.base_ob / (len(mu_g) * len(chi_g)))[idx]
            ft = (fit.w * step, fit.s * step)
            tg = [float(np.median(p.smu[m])), float(np.median(p.schi[m])), float(np.median(p.sob[m]))]
            row = {}
            ref = fp.predict_plane_profile(p.take(idx), fp.ProfileFit(c=fit.c[idx], A=fit.A[idx], w=fit.w, s=fit.s, at_edge=0.0), r["_pos_k"], r["_b"][idx])
            for tag, orc in (("v3", False), ("v3_oracleA", True)):
                s_, t_ = r2(obs, ref, b, oracle=orc); row[tag] = float(1 - s_.sum() / t_.sum())
            for vname, opt in VARIANTS.items():
                refs = ts.physical_references(p.mu0, p.chi0, fit.c, m, ob_g, step, lam, mu_g, chi_g, tg, flat_top_ob_deg=ft, couple=opt["couple"])
                pw = (p.smu[idx], p.schi[idx], tg[0], tg[1]) if opt["pp"] else None
                _, _, pred = ts.physical_score(obs, p.mu0[idx], p.chi0[idx], fit.c[idx], p.mob_train[:, idx], p.base_ob[idx], ob_g, k, mu_g, chi_g,
                                               refs, step, lam, flat_top_ob_deg=ft, return_pred=True, couple=opt["couple"], per_pixel_widths=pw)
                for tag, orc in ((vname, False), (vname + "_oracleA", True)):
                    s_, t_ = r2(obs, pred, b, oracle=orc)
                    row[tag] = (float(s_.sum()), float(t_.sum()))
            row["v3"] = row["v3"]
            # keep sums for pooling; v3 R2 only for the per-fold table
            sv, tv = r2(obs, ref, b); sov, tov = r2(obs, ref, b, oracle=True)
            row["v3"] = (float(sv.sum()), float(tv.sum())); row["v3_oracleA"] = (float(sov.sum()), float(tov.sum()))
            res[k] = row
            log(name, "fold", k, {t: round(1 - v[0] / v[1], 3) for t, v in row.items() if not t.endswith("oracleA")})
        R["datasets"][name] = {int(k): {t: [v[0], v[1]] for t, v in row.items()} for k, row in res.items()}
        del cube
    # ---- pooled scores, decision rule
    def pooled(name, tag, folds):
        ds = R["datasets"][name]
        return 1 - sum(ds[k][tag][0] for k in folds) / sum(ds[k][tag][1] for k in folds)
    tags = ["v3", "v3_oracleA"] + [t for v in VARIANTS for t in (v, v + "_oracleA")]
    summ = {}
    for tag in tags:
        summ[tag] = dict(primary={n: pooled(n, tag, PRIMARY) for n in DATASETS},
                         flank={k: float(np.mean([1 - R["datasets"][n][k][tag][0] / R["datasets"][n][k][tag][1] for n in DATASETS])) for k in FLANK})
        summ[tag]["primary_mean"] = float(np.mean(list(summ[tag]["primary"].values())))
    R["summary"] = summ
    gap0 = summ["v3"]["primary_mean"] - summ["P0"]["primary_mean"]
    dec = {"gap0": gap0, "candidates": {}}
    for cand, vname in (("a_coupling", "Pa"), ("b_per_pixel_widths", "Pb"), ("a+b", "Pab")):
        closed = (summ[vname]["primary_mean"] - summ["P0"]["primary_mean"]) / gap0 if gap0 > 0 else float("nan")
        worst_flank_drop = max(summ["P0"]["flank"][k] - summ[vname]["flank"][k] for k in FLANK)
        dec["candidates"][cand] = dict(variant=vname, gap_closed_fraction=closed, worst_flank_drop=worst_flank_drop,
                                       carried=bool(gap0 > GAP_MIN and closed >= CARRY_FRAC and worst_flank_drop <= FLANK_DROP))
    gap_or = summ["v3_oracleA"]["primary_mean"] - summ["P0_oracleA"]["primary_mean"]
    dec["amplitude"] = dict(gap_under_oracle_amplitude=gap_or, fraction_surviving=gap_or / gap0 if gap0 > 0 else float("nan"),
                            not_the_cause=bool(gap0 > 0 and gap_or / gap0 >= AMP_FRAC))
    dec["gap_exists"] = bool(gap0 > GAP_MIN)
    dec["v5_to_run"] = bool(dec["gap_exists"] and any(c["carried"] for c in dec["candidates"].values() if c["variant"] in ("Pa", "Pb")))
    R["decision"] = dec
    log("SUMMARY", json.dumps({t: {"primary": {k: round(v, 3) for k, v in s["primary"].items()}, "mean": round(s["primary_mean"], 3),
                                   "flank": {k: round(v, 3) for k, v in s["flank"].items()}} for t, s in summ.items()}))
    log("DECISION", json.dumps(dec, default=float))
    json.dump(R, open(out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
