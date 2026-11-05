#!/usr/bin/env python3
"""Phase hunt for ONE Varex sweep: are there diffraction lines that the dominant ("primary") phase does not explain, and do they fit a cubic cell?

Usage:  phase_hunt.py <data.vrx.h5> <out_prefix> <energy_keV> <lsd_mm>
            [--primary SG:a,b,c,alpha,beta,gamma]   dominant phase, its lines are masked (e.g. 166:2.876,2.876,14.2,90,90,120)
            [--candidates candidates.json]          [{"name": "...", "sg": 216, "a": 9.85}, ...] literature cubic cells to test
            [--px 150] [--ndraw 10000] [--seed 1]
            [--plant a,family,per_line,level --aref <amplitude>]    positive control, see below

1. M = per-pixel temporal median of <=120 frames (frame 0 skipped); X = per-pixel MAX over frames 1.. of (frame - M).
   A median keeps what is present in most frames (continuous powder rings); a max keeps single crystallite spots.
2. Beam centre from the data (ring_survey.find_bc on M). Ring-level lines from the radial profile of M; spot-level lines from a
   radial histogram of spot clusters in X (noise from the histogram's own running MAD: spot clusters are clumped, Poisson is wrong).
3. Lines within the primary-phase mask (+-(3 px + 0.5 % of R)) are removed; the rest is L.
4. Cubic indexing of L (P / I / F families, a 3 to 25 A), score = number of lines matched within 0.4 % in d; null = random radii in the
   same window. A free cell has little power with few lines (any 6 lines index): read the candidate table, not only this.
5. Candidate table: for each literature cell, lines in the window, matches, and the chance rate from random radii (cells are NOT refined).
--plant injects synthetic cubic spots (radius-faithful, not kinematic) for a positive control: per_line spots per ring at level x --aref,
where --aref is the 90th percentile of X within 3 px of the lowest-angle primary line (printed as A_ref by an unplanted run).
An unplanted run says nothing about what the method CANNOT see: run the control before reading any "nothing found".
Status: PROVISIONAL (see the kit README); on the reference beamtime the planted control recovered 7 of 20 rings at 1000 spots/ring.
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from ring_survey import find_bc, radial  # copied next to this script

PX = 150.0              # overridden by --px
RMIN, RMAX = 150, 1440
TOL = 0.004
HC = 12.398419843


def read_stack(fn, nmed=120, chunk=16, plant=None, rng=None):
    with h5py.File(fn, "r") as f:
        d = f["exchange/data"]
        n = d.shape[0]
        idx = np.unique(np.linspace(1, n - 1, min(nmed, n - 1)).astype(int))
        fr = np.stack([np.asarray(d[i], dtype=np.float32) for i in idx])
        M = np.median(fr, 0).astype(np.float32)
        del fr
        X = np.full(M.shape, -1e9, np.float32)
        for s in range(1, n, chunk):
            blk = np.asarray(d[s:min(s + chunk, n)], dtype=np.float32) - M
            if plant is not None:
                plant(blk, s)
            X = np.maximum(X, blk.max(0))
    return M, X, n


def cluster_radii(X, bc, mask_hot):
    yy, xx = np.indices(X.shape)
    r = np.hypot(yy - bc[0], xx - bc[1])
    rb = np.minimum(r.astype(int), RMAX + 1).ravel()
    flat = X.ravel()
    order = np.argsort(rb, kind="stable")
    cnt = np.bincount(rb, minlength=RMAX + 2)
    edges = np.concatenate([[0], np.cumsum(cnt)])
    med = np.zeros(RMAX + 2)
    sig = np.ones(RMAX + 2)
    for k in range(RMIN - 5, RMAX + 2):
        v = flat[order[edges[k]:edges[k + 1]]]
        if v.size > 50:
            m = np.median(v)
            med[k] = m
            sig[k] = max(1.4826 * np.median(np.abs(v - m)), 1.0)
    ex = (X - med[rb].reshape(X.shape)) / sig[rb].reshape(X.shape)
    ex[~mask_hot] = 0
    ex[(r < RMIN) | (r > RMAX)] = 0
    lab, nl = ndi.label(ex >= 5.0, structure=np.ones((3, 3)))
    if nl == 0:
        return np.zeros(0), np.zeros(0), 0
    sizes = ndi.sum(np.ones_like(ex), lab, index=np.arange(1, nl + 1))
    keep = np.where(sizes >= 2)[0] + 1
    com = ndi.center_of_mass(np.clip(ex, 0, None), lab, keep)
    com = np.array(com)
    pk = ndi.maximum(ex, lab, keep)
    rad = np.hypot(com[:, 0] - bc[0], com[:, 1] - bc[1])
    return rad, np.asarray(pk), int(nl)


def hist_lines(rad, lo=RMIN, hi=RMAX, bin_px=0.5, zmin=5.0, nmin=5):
    """Amendment 1: overdispersion-aware. Histogram smoothed with a 1 px Gaussian; sigma = running MAD of the residual (+-100 bins)."""
    edges = np.arange(lo, hi + bin_px, bin_px)
    h, _ = np.histogram(rad, bins=edges)
    hs = ndi.gaussian_filter1d(h.astype(float), 2.0)
    base = ndi.median_filter(hs, size=81, mode="nearest")
    base = ndi.uniform_filter1d(base, 9)
    res = hs - base
    sig = np.empty_like(res)
    for i in range(len(res)):
        w = res[max(i - 100, 0):i + 101]
        sig[i] = max(1.4826 * np.median(np.abs(w - np.median(w))), np.sqrt(max(base[i], 1.0)) * 0.3)
    z = res / sig
    cen = 0.5 * (edges[1:] + edges[:-1])
    peaks = []
    for i in range(1, len(h) - 1):
        if z[i] >= zmin and h[i] >= nmin and hs[i] >= hs[i - 1] and hs[i] >= hs[i + 1]:
            sl = slice(max(i - 2, 0), i + 3)
            w = np.clip(hs[sl] - base[sl], 0, None)
            rc = float((cen[sl] * w).sum() / w.sum()) if w.sum() > 0 else float(cen[i])
            peaks.append((rc, float(z[i]), int(h[i]), float(base[i])))
    out = []
    for p in sorted(peaks):                        # merge peaks closer than 1 px
        if out and p[0] - out[-1][0] < 1.0:
            if p[1] > out[-1][1]:
                out[-1] = p
        else:
            out.append(p)
    return out, h, base, cen


def ring_level_lines(M, bc):
    prof = radial(M, bc, RMAX + 1)
    base = ndi.median_filter(prof, size=61, mode="nearest")
    res = prof - base
    seg = res[RMIN:RMAX]
    sig = 1.4826 * np.median(np.abs(seg - np.median(seg)))
    out = []
    for i in range(RMIN + 1, RMAX - 1):
        if res[i] >= 5 * sig and res[i] >= res[i - 1] and res[i] >= res[i + 1]:
            w = np.clip(res[i - 3:i + 4], 0, None)
            rc = float((np.arange(i - 3, i + 4) * w).sum() / w.sum()) + 0.5
            out.append((rc, float(res[i] / sig), float(res[i])))
    return out, prof, base


def cubic_N(family, nmax=2000):
    s = set()
    m = int(np.sqrt(nmax)) + 1
    for h in range(0, m):
        for k in range(0, m):
            for l in range(0, m):
                n = h * h + k * k + l * l
                if n == 0 or n > nmax:
                    continue
                if family == "I" and (h + k + l) % 2:
                    continue
                if family == "F" and not (h % 2 == k % 2 == l % 2):
                    continue
                s.add(n)
    return np.array(sorted(s), float)


NSETS = {f: cubic_N(f) for f in "PIF"}


def best_cubic(d, family, tol=TOL, amin=3.0, amax=25.0):
    """max over a of the number of lines i with some allowed N with |a - d_i sqrt(N)|/a <= tol; candidates a_iN = d_i sqrt(N)."""
    N = NSETS[family]
    cands = []
    for i, di in enumerate(d):
        a = di * np.sqrt(N)
        a = a[(a >= amin) & (a <= amax)]
        cands.append(np.sort(np.log(a)))
    allc = np.sort(np.concatenate(cands)) if cands else np.zeros(0)
    if allc.size == 0:
        return 0, None
    cnt = np.zeros(allc.size, int)
    for c in cands:
        if c.size == 0:
            continue
        j = np.searchsorted(c, allc)
        ok = j < c.size
        cnt += ok & (c[np.minimum(j, c.size - 1)] <= allc + 2 * tol)
    k = int(cnt.argmax())
    return int(cnt[k]), float(np.exp(allc[k] + tol))


def fit_all(d, tol=TOL):
    out = {}
    for fam in "PIF":
        n, a = best_cubic(d, fam, tol)
        out[fam] = (n, a)
    return out


def primary_lines(lam, lsd, sg, lat):
    """d and radius (px) of every line of the primary phase in the detector window."""
    import midas_hkls as mh
    refl = mh.generate_hkls(mh.SpaceGroup.from_number(sg), mh.Lattice(*lat), d_min=lam / (2 * np.sin(np.arctan(RMAX * PX / (lsd * 1000)) / 2)) * 0.97, d_max=lam / (2 * np.sin(np.arctan(RMIN * PX / (lsd * 1000)) / 2)) * 1.03)
    ds = sorted({round(r.d_spacing, 5) for r in refl})
    th = np.arcsin(lam / (2 * np.array(ds)))
    return np.array(ds), lsd * 1000 / PX * np.tan(2 * th)


def main():
    a = sys.argv[1:]
    if len(a) < 4 or a[0] in ("-h", "--help"):
        sys.exit(__doc__)
    fn, pre, E, lsd = a[0], a[1], float(a[2]), float(a[3])
    opt = {a[i]: a[i + 1] for i in range(4, len(a) - 1, 2) if a[i].startswith("--")}
    lam = HC / E
    global PX
    PX = float(opt.get("--px", PX))
    prim = None
    if "--primary" in opt:
        _sg, _lat = opt["--primary"].split(":")
        prim = (int(_sg), tuple(float(x) for x in _lat.split(",")))
    ndraw = int(opt.get("--ndraw", 10000))
    rng = np.random.default_rng(int(opt.get("--seed", 1)))
    Path(pre).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report = {"file": fn, "energy_keV": E, "lambda_A": lam, "lsd_mm": lsd}
    # BC from the data on M (cheap pass first: M only)
    with h5py.File(fn, "r") as f:
        n = f["exchange/data"].shape[0]
        idx = np.unique(np.linspace(1, n - 1, min(120, n - 1)).astype(int))
        M0 = np.median(np.stack([np.asarray(f["exchange/data"][i], dtype=np.float32) for i in idx]), 0)
    prof0 = radial(M0, (1440.0, 1440.0), 1500)
    ring0 = int(np.argmax(prof0[200:600])) + 200
    bc, order = find_bc(M0.astype(np.float64), (1440.0, 1440.0), ring0)
    report["bc_row_col"] = [float(bc[0]), float(bc[1])]
    ds_prim, R_prim = primary_lines(lam, lsd, *prim) if prim else (np.zeros(0), np.zeros(0))
    A003 = None
    plant = None
    if "--plant" in opt:
        pa, fam, per_line, level = opt["--plant"].split(",")
        pa, per_line, level = float(pa), int(per_line), float(level)
        A003 = float(opt["--aref"])
        Nn = NSETS[fam]
        dd = pa / np.sqrt(Nn)
        R = lsd * 1000 / PX * np.tan(2 * np.arcsin(lam / (2 * dd)))
        sel = (R > RMIN + 5) & (R < RMAX - 5)
        Rpl = R[sel]
        report["planted"] = {"a": pa, "family": fam, "level_of_A_ref": level, "A_ref": A003, "per_line": per_line, "n_lines": int(sel.sum()), "R_px": Rpl.tolist()}
        spots = []
        for Rr in Rpl:
            got = 0
            while got < per_line:                      # only spots whose centre (+-4 px) lies on the detector
                eta = rng.uniform(0, 2 * np.pi)
                yy_, xx_ = bc[0] + Rr * np.sin(eta), bc[1] + Rr * np.cos(eta)
                if 5 <= yy_ < 2875 and 5 <= xx_ < 2875:
                    spots.append((Rr, eta, int(rng.integers(1, n - 3))))
                    got += 1
        spots = np.array(spots)
        sy = bc[0] + spots[:, 0] * np.sin(spots[:, 1])
        sx = bc[1] + spots[:, 0] * np.cos(spots[:, 1])
        sf = spots[:, 2].astype(int)
        report["planted"]["n_spots_on_detector"] = int(len(spots))

        def plant(blk, s0):
            lo, hi = s0 - 1, s0 + blk.shape[0]
            for k in np.where((sf >= lo) & (sf <= hi))[0]:
                for df, w in ((-1, 0.5), (0, 1.0), (1, 0.5)):
                    j = sf[k] + df - s0
                    if 0 <= j < blk.shape[0]:
                        y0, x0 = int(round(sy[k])), int(round(sx[k]))
                        ys, xs = slice(y0 - 4, y0 + 5), slice(x0 - 4, x0 + 5)
                        gy, gx = np.mgrid[ys, xs]
                        blk[j][ys, xs] += (level * A003 * w * np.exp(-((gy - sy[k]) ** 2 + (gx - sx[k]) ** 2) / 2.0)).astype(np.float32)
    M, X, nfr = read_stack(fn, plant=plant)
    report["n_frames"] = int(nfr)
    report["t_read_s"] = round(time.time() - t0, 1)
    # hot pixels: the brightest 0.01 % of the median image
    yy, xx = np.indices(M.shape)
    mask_ok = np.ones(M.shape, bool)
    thr = np.percentile(M, 99.99)
    mask_ok[M > thr] = False
    # A_ref: 90th percentile of X within 3 px of the lowest-angle primary line (reference spot amplitude for the planted control)
    r_img = np.hypot(yy - bc[0], xx - bc[1])
    if A003 is None and len(R_prim):
        R_ref = float(R_prim[np.argmax(ds_prim)])
        report["R_ref_px"] = R_ref
        sel = np.abs(r_img - R_ref) < 3
        report["A_ref"] = float(np.percentile(X[sel & mask_ok], 90))
    rad, pk, nl = cluster_radii(X, bc, mask_ok)
    report["n_clusters"] = int(len(rad))
    spot_lines, h, base, cen = hist_lines(rad)
    ring_lines, prof, pbase = ring_level_lines(M, bc)
    # primary-phase mask
    half = np.maximum(3.0 + 0.005 * R_prim, 4.0)

    def in_primary(rr):
        return bool(np.any(np.abs(rr - R_prim) <= half))
    cov = np.zeros(RMAX + 1, bool)
    for Rr, hh in zip(R_prim, half):
        cov[int(max(Rr - hh, 0)):int(min(Rr + hh, RMAX)) + 1] = True
    report["primary_mask_fraction_of_radii"] = float(cov[RMIN:RMAX].mean())
    report["n_spot_lines_all"] = len(spot_lines)
    report["n_ring_lines_all"] = len(ring_lines)
    def mk(lst, kind):
        o = [{"kind": kind, "R_px": t[0], "z": t[1]} for t in lst if not in_primary(t[0])]
        for t in o:
            t["d_A"] = lam / (2 * np.sin(np.arctan(t["R_px"] * PX / (lsd * 1000)) / 2))
        return sorted(o, key=lambda x: x["R_px"])
    Lring, Lspot = mk(ring_lines, "ring"), mk(spot_lines, "spot")
    uni = sorted(Lring + Lspot, key=lambda x: x["R_px"])
    Lunion = []
    for t in uni:                                  # merge ring/spot lines within 1.5 px
        if Lunion and t["R_px"] - Lunion[-1]["R_px"] < 1.5:
            Lunion[-1]["kind"] += "+" + t["kind"]
            Lunion[-1]["z"] = max(Lunion[-1]["z"], t["z"])
        else:
            Lunion.append(dict(t))
    L = Lunion
    report["sets"] = {}
    for name, Ls, nd in (("ring", Lring, ndraw), ("spot", Lspot, min(ndraw, 2000)), ("union", Lunion, min(ndraw, 2000))):
        ent = {"n_lines": len(Ls), "lines": Ls}
        d = np.array([t["d_A"] for t in Ls])
        if len(d) >= 5:
            fit = fit_all(d)
            ent["fit"] = {k: {"n_matched": v[0], "a_A": v[1]} for k, v in fit.items()}
            Rlo, Rhi = min(t["R_px"] for t in Ls), max(t["R_px"] for t in Ls)
            nb = {f: [] for f in "PIF"}
            for _ in range(nd):
                Rr = rng.uniform(Rlo, Rhi, len(Ls))
                dr = lam / (2 * np.sin(np.arctan(Rr * PX / (lsd * 1000)) / 2))
                for fam in "PIF":
                    nb[fam].append(best_cubic(dr, fam)[0])
            ent["null"] = {f: {"p": float((np.array(nb[f]) >= fit[f][0]).mean()), "null_mean": float(np.mean(nb[f])), "null_max": int(np.max(nb[f])), "ndraw": nd} for f in "PIF"}
        else:
            ent["fit"] = None
        report["sets"][name] = ent
    # candidate table (PREREGISTER step 6): literature cells, NOT refined here; equal provenance
    import midas_hkls as mh
    if "--candidates" in opt:
        CAND = [(c["name"], int(c["sg"]), float(c["a"])) for c in json.load(open(opt["--candidates"]))]
    else:      # EXAMPLES only (literature cells, equal provenance): replace with the phases your sample can contain
        CAND = [("Li6PS5Cl F-43m", 216, 9.85), ("LiCl Fm-3m", 225, 5.14), ("Li2S Fm-3m", 225, 5.71), ("Fe Im-3m", 229, 2.866),
                ("Al Fm-3m", 225, 4.05), ("Cu Fm-3m", 225, 3.615), ("Pb Fm-3m", 225, 4.95), ("CeO2 Fm-3m", 225, 5.411)]
    report["candidates"] = {}
    for sname, Ls in (("ring", Lring), ("union", Lunion)):
        if len(Ls) < 1:
            continue
        dobs = np.array([t["d_A"] for t in Ls])
        Rlo, Rhi = min(t["R_px"] for t in Ls) - 1, max(t["R_px"] for t in Ls) + 1
        dlo = lam / (2 * np.sin(np.arctan(Rhi * PX / (lsd * 1000)) / 2)); dhi = lam / (2 * np.sin(np.arctan(Rlo * PX / (lsd * 1000)) / 2))
        for cname, sg, ac in CAND:
            refl = mh.generate_hkls(mh.SpaceGroup.from_number(sg), mh.Lattice(ac, ac, ac, 90, 90, 90), d_min=dlo * 0.99, d_max=dhi * 1.01)
            dl = np.array(sorted({r.d_spacing for r in refl}))
            def nmatch(dv):
                return int(sum(np.abs(dl - x).min() / x <= TOL for x in dv)) if len(dl) else 0
            m = nmatch(dobs)
            rr = rng.uniform(Rlo, Rhi, (3000, len(Ls)))
            dr = lam / (2 * np.sin(np.arctan(rr * PX / (lsd * 1000)) / 2))
            nm = np.array([nmatch(row) for row in dr])
            report["candidates"].setdefault(sname, {})[cname] = {"a_A": ac, "lines_in_window": int(len(dl)), "n_obs_lines": len(Ls), "matched": m,
                                                              "chance_matched_mean": float(nm.mean()), "p_chance_ge": float((nm >= m).mean()), "ndraw": 3000,
                                                              "matched_lines_d": [round(float(x), 3) for x in dobs if len(dl) and np.abs(dl - x).min() / x <= TOL]}
    report["n_L_ring"], report["n_L_spot"], report["n_L_union"] = len(Lring), len(Lspot), len(Lunion)
    # figure + saved arrays
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    ax[0].plot(np.arange(len(prof)), prof - pbase, lw=0.7)
    ax[0].set_ylabel("M profile - baseline")
    ax[1].plot(cen, h, lw=0.7)
    ax[1].plot(cen, base, lw=0.7)
    ax[1].set_yscale("symlog")
    ax[1].set_ylabel("spot clusters per 0.5 px")
    for a_ in ax:
        for Rr in R_prim:
            a_.axvline(Rr, color="0.8", lw=0.5, zorder=0)
        for t in L:
            a_.axvline(t["R_px"], color="r", lw=0.8)
    ax[1].set_xlabel("radius (px); grey = primary-phase lines, red = lines it does not explain (L)")
    ax[0].set_title(Path(fn).name)
    fig.tight_layout()
    fig.savefig(pre + "_hunt.png", dpi=80)
    np.save(pre + "_clusters_rad.npy", rad)
    report["t_total_s"] = round(time.time() - t0, 1)
    json.dump(report, open(pre + "_hunt.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("sets", "planted")}, indent=1))
    for sn, cd in report.get("candidates", {}).items():
        for cn, v in cd.items():
            print("CAND", sn, cn, "lines_in_window", v["lines_in_window"], "obs", v["n_obs_lines"], "matched", v["matched"], "chance_mean %.2f" % v["chance_matched_mean"], "p", v["p_chance_ge"])
    for nm, e in report["sets"].items():
        print(nm, e["n_lines"], "lines; fit", e.get("fit"), "null", {f: (v["p"], v["null_mean"]) for f, v in (e.get("null") or {}).items()})
        print("   d_A:", [round(t["d_A"], 3) for t in e["lines"]][:30])


if __name__ == "__main__":
    main()
