"""PREREG_friedel_geometry.md positive control: Friedel-pair geometry estimator against an INDEPENDENT forward model.

Forward (written here, nothing from midas_sx / ewald_crossings / qlab_to_pixel): reflections q_s = U diag(1/a,1/b,1/c) h; Ewald crossings by sign changes of
g(w) = 2 k x(w) + |q|^2 on a 0.05 deg grid polished with brentq, x(w) = first component of R_y(-W) R_z(w) q_s; diffracted ray k s0 + q_lab; detector = plane through
(Lsd,0,0) with normal Rt e_x, Rt = Rx(tx) Ry(ty) Rz(tz) (midas_calibrate.geometry_torch.build_tilt_matrix_torch order), pixel (row, col) at (Lsd,0,0) + Rt (0, -(col-bcy) px, (row-bcz) px) (the MIDAS C convention, re-derived from CorrectTiltSpatialDistortion;
checked below against the package's pixel_to_qlab AS A CONVENTION CHECK ONLY), closed-form ray-plane intersection.
Estimator: midas_sx.friedel_geometry (spot_directions + qlab_to_qsample).

    python -I friedel_sim.py [n_rep=50] [out.csv]
"""
import os, sys, time
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress", "midas_transforms", "midas_calibrate") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd
from scipy.optimize import brentq
from dataclasses import replace
from midas_sx.friedel_geometry import fit_pairs, labelled_pairs, with_params
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_stress.orientation_sampling import uniform_rotations

LAM, PX = 0.6889, 0.172
A, B, C = 5.8033, 5.95953, 12.2978
NR, NC = 1679, 1475
OM_LO, OM_HI = -15.0, 155.0                                     # the A39 scan, omega sign -1
K = 1.0 / LAM
HEADER = dict(tx=0.0, ty=0.0, tz=-20.0, wedge=0.0, bcz=862.41, bcy=401.77, lsd=170.268)


def Rx(a): c, s = np.cos(np.radians(a)), np.sin(np.radians(a)); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def Ry(a): c, s = np.cos(np.radians(a)), np.sin(np.radians(a)); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rz(a): c, s = np.cos(np.radians(a)), np.sin(np.radians(a)); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def plane_dir(rows, cols, T):
    Rt = Rx(T["tx"]) @ Ry(T["ty"]) @ Rz(T["tz"])
    lsd = T["lsd"] * 1000.0
    loc = np.stack([np.zeros_like(rows, float), -(np.asarray(cols, float) - T["bcy"]) * PX * 1000.0, (np.asarray(rows, float) - T["bcz"]) * PX * 1000.0], 1)
    p = np.array([lsd, 0, 0]) + loc @ Rt.T
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def convention_check(T, geom):
    rng = np.random.default_rng(0); r = rng.uniform(100, 1500, 200); c = rng.uniform(50, 1400, 200)
    s_pkg, _, _ = spot_directions(r, c, geom)
    return float(np.abs(plane_dir(r, c, T) - s_pkg).max())


def forward(T, U, d_min=0.75):
    """list of dict(h,k,l,row,col,omega) for every crossing in the scan that lands on the detector."""
    Rt = Rx(T["tx"]) @ Ry(T["ty"]) @ Rz(T["tz"]); n = Rt @ np.array([1.0, 0, 0]); O = np.array([T["lsd"] * 1000.0, 0, 0])
    W = np.radians(T["wedge"]); cw, sw = np.cos(W), np.sin(W)
    H = int(A / d_min) + 1; Kk = int(B / d_min) + 1; L = int(C / d_min) + 1
    grid = np.arange(OM_LO - 0.1, OM_HI + 0.1, 0.05); gr = np.radians(grid)
    out = []
    for h in range(-H, H + 1):
        for k in range(-Kk, Kk + 1):
            for l in range(-L, L + 1):
                if h == k == l == 0:
                    continue
                q0 = np.array([h / A, k / B, l / C])
                if np.linalg.norm(q0) > 1 / d_min:
                    continue
                q = U @ q0
                def x_of(w):
                    xr = np.cos(w) * q[0] - np.sin(w) * q[1]
                    return cw * xr - sw * q[2]
                g = 2 * K * x_of(gr) + q @ q
                for i in np.flatnonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0):
                    w = brentq(lambda t: 2 * K * x_of(t) + q @ q, gr[i], gr[i + 1], xtol=1e-14)
                    xr, yr = np.cos(w) * q[0] - np.sin(w) * q[1], np.sin(w) * q[0] + np.cos(w) * q[1]
                    ql = np.array([cw * xr - sw * q[2], yr, sw * xr + cw * q[2]])
                    v = K * np.array([1.0, 0, 0]) + ql; d = v / np.linalg.norm(v)
                    den = n @ d
                    if den <= 1e-6:
                        continue
                    P = (n @ O) / den * d; loc = Rt.T @ (P - O)
                    col, row = T["bcy"] - loc[1] / (PX * 1000.0), T["bcz"] + loc[2] / (PX * 1000.0)
                    if 3 < row < NR - 3 and 3 < col < NC - 3:
                        out.append((h, k, l, row, col, np.degrees(w)))
    return pd.DataFrame(out, columns=["h", "k", "l", "row", "col", "omega"])


def noisy(df, rng, sig_px, sig_frame, label_err=0.0):
    d = df.copy()
    d["row"] += rng.normal(0, sig_px, len(d)); d["col"] += rng.normal(0, sig_px, len(d))
    d["omega_meas"] = d["omega"] + rng.normal(0, sig_frame * 0.2, len(d))
    if label_err > 0:                                          # a fraction of labels replaced by a random other label
        m = rng.random(len(d)) < label_err
        d.loc[m, ["h", "k", "l"]] = d[["h", "k", "l"]].to_numpy()[rng.permutation(len(d))[: m.sum()]]
    d["hkl_residual"] = 0.0
    return d


def subsample_pairs(pairs, n, rng):
    if n is None or len(pairs) <= n:
        return pairs
    g = pairs.group.unique(); rng.shuffle(g); keep, tot = [], 0
    cnt = pairs.group.value_counts()
    for x in g:
        keep.append(x); tot += cnt[x]
        if tot >= n:
            break
    return pairs[pairs.group.isin(keep)].reset_index(drop=True)


if __name__ == "__main__":
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    outp = sys.argv[2] if len(sys.argv) > 2 else "friedel_sim_results.csv"
    scan = Scan(155.0, -0.2, 850)
    g0 = with_params(make_geometry(wavelength_A=LAM, dist_mm=170.0, pixel_mm=PX, centre_rc=(862.0, 401.0), shape=(NR, NC), scan=scan), HEADER)
    SCEN = {"S1 run02-like (tx -0.10, W +0.41)": dict(tx=-0.10, ty=-0.19, tz=-20.07, wedge=0.41, bcz=861.9, bcy=401.3, lsd=171.14),
            "S2 (tx +0.30, W -0.30)": dict(tx=0.30, ty=-0.19, tz=-20.07, wedge=-0.30, bcz=861.9, bcy=401.3, lsd=171.14),
            "S3 zero (tx 0, W 0)": dict(tx=0.0, ty=-0.19, tz=-20.07, wedge=0.0, bcz=861.9, bcy=401.3, lsd=171.14)}
    Us = uniform_rotations(400); rows = []
    t0 = time.time()
    for name, T in SCEN.items():
        print(name, "convention check (max |dir_indep - dir_package|):", convention_check(T, with_params(g0, T)), flush=True)
        for rep in range(n_rep):
            rng = np.random.default_rng(1000 * (list(SCEN).index(name) + 1) + rep)
            fw = forward(T, Us[rng.integers(len(Us))])
            obs = noisy(fw, rng, 0.4, 0.3)
            pairs_all = labelled_pairs(obs)
            for kind in ("pairs~100", "scrambled~100", "all"):
                pr = subsample_pairs(pairs_all, 100 if kind != "all" else None, rng)
                if kind == "scrambled~100":
                    pr = pr.copy()
                    for c in ("row2", "col2", "om2"):
                        pr[c] = rng.permutation(pr[c].to_numpy())
                if kind == "all" and rep >= max(1, n_rep // 5):
                    continue
                fit = fit_pairs(pr, g0)
                rec = dict(scenario=name, rep=rep, kind=kind, n_pairs=fit.n_pairs, n_groups=fit.n_groups, rms=fit.rms_final, scale=fit.scale, rank_ratio=float(fit.singular_values[-1] / fit.singular_values[0]))
                for n_, v, s in zip(fit.names, fit.values, fit.sigma_formal):
                    rec[n_] = v; rec["sig_" + n_] = s; rec["true_" + n_] = T[n_]
                rows.append(rec)
            print(f"  {name} rep {rep} done ({len(pairs_all)} pairs available), {time.time() - t0:.0f} s", flush=True)
            pd.DataFrame(rows).to_csv(outp, index=False)
