"""Fork A POST HOC (not registered): refit the assigned reflections of one run with the package parameter set (U, centre, Lsd, tilts, wedge) and with extras: wavelength scale, rotation-speed (omega step) scale,
sideways rotation-axis tilt V (axis = R_y(-W) R_x(V) e_z). Own predictor, checked against midas_sx.orient.predict_positions at V = 0.   python -I a39_diag_extfit.py <out_p0 dir> <step1 dir> <cif> <rNN> [<rNN>...]"""
import json, os, sys
_sx = os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")
sys.path[:0] = [_sx, os.path.dirname(os.path.abspath(__file__))] + [os.path.join(os.path.dirname(_sx), p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(os.path.dirname(_sx), p))]
import numpy as np, pandas as pd, gemmi
from dataclasses import replace
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from midas_hkls import Lattice
from midas_sx.assign import reciprocal_basis
from midas_sx.conventions import Convention
from midas_sx.geometry import Scan
from midas_sx.orient import predict_positions, _safe_pixels, _unpack
from a39_geom import swing_geometry
pd.set_option("display.width", 250)
d, step1, cif, runs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
ss = gemmi.read_small_structure(cif); lat = Lattice(ss.cell.a, ss.cell.b, ss.cell.c, 90, 90, 90)
H = json.load(open(os.path.join(step1, "run02", "info.json")))["header"]
g0, scan0 = swing_geometry(H, Convention(0, -1), "cols", -1, -1)
B = reciprocal_basis(lat)
XS = np.array([0.01, 0.01, 0.01, 1.0, 1.0, 0.003, 0.3, 0.3, 0.3, 0.3, 1e-3, 1e-3, 0.3, 1e-3, 1e-3, 1e-3])      # 10 package params, lambda rel, step rel, V deg


def predict(p, U0, g00, scan, hkl, hint_deg, extras):
    pf = np.zeros(16); pf[:len(p)] = p
    U, g = _unpack(pf[:10], U0, g00)
    lam = g.wavelength_A * (1.0 + pf[10] * extras["lam"]); g = replace(g, wavelength_A=lam)
    stepe = scan.omega_step * (1.0 + pf[11] * extras["step"]); V = np.radians(pf[12] * extras["V"])
    W = np.radians(g.wedge_deg)
    Ry = np.array([[np.cos(-W), 0, np.sin(-W)], [0, 1, 0], [-np.sin(-W), 0, np.cos(-W)]]); Rx = np.array([[1, 0, 0], [0, np.cos(V), -np.sin(V)], [0, np.sin(V), np.cos(V)]])
    T = Ry @ Rx
    Bc_ = B @ np.diag(1.0 / (1.0 + pf[13:16] * extras.get('cell', 0)))
    q = 2 * np.pi * (U @ Bc_ @ np.asarray(hkl, float).T).T; k = 2 * np.pi / lam
    A = T[0, 0] * q[:, 0] + T[0, 1] * q[:, 1]; Bc = T[0, 1] * q[:, 0] - T[0, 0] * q[:, 1]; C = -(q * q).sum(1) / (2 * k) - T[0, 2] * q[:, 2]
    R = np.hypot(A, Bc); ok = (R > 0) & (np.abs(C) <= R)
    phi = np.arctan2(Bc, A); dl = np.arccos(np.clip(C / np.where(R > 0, R, 1), -1, 1))
    hint = np.radians(hint_deg); sols = np.stack([phi + dl, phi - dl])
    dd = np.abs((sols - hint + np.pi) % (2 * np.pi) - np.pi); w = np.where(dd[0] <= dd[1], sols[0], sols[1])
    om = np.degrees(hint) + np.degrees((w - hint + np.pi) % (2 * np.pi) - np.pi)
    c, s = np.cos(w), np.sin(w); rz = np.stack([c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1], q[:, 2]], 1)
    ql = rz @ T.T
    ro = np.full(len(q), np.nan); co = np.full(len(q), np.nan)
    if ok.any():
        ro[ok], co[ok] = _safe_pixels(ql[ok], g)
    fr = (om - scan.omega_start) / stepe - 0.5
    return fr, ro, co, ok & np.isfinite(ro)


res_all = []
for run in runs:
    red = json.load(open(os.path.join(d, run, "reduction.json"))); U0 = np.load(os.path.join(d, run, "U.npy")); o = pd.read_pickle(os.path.join(d, run, "obs.pkl"))
    o = o[(o.tth > 5)].reset_index(drop=True)
    tx, ty, tz = red["tilts_deg"]; rc = red["centre_rc"]
    g00 = replace(g0, tx_deg=tx, ty_deg=ty, tz_deg=tz, wedge_deg=red["wedge_deg"], lsd_um=red["lsd_mm"] * 1000.0, bcz_px=rc[0], bcy_px=rc[1])   # refined geometry = zero point
    hkl = o[["h", "k", "l"]].to_numpy(int); fo, ro_, co_ = o.frame.to_numpy(float), o.row.to_numpy(float), o.col.to_numpy(float); hint = scan0.omega_of(fo)
    # check: own predictor at p = 0 vs the package
    fr_p, r_p, c_p, ok_p = predict_positions(U0, lat, hkl, hint, g00, scan0)
    fr_m, r_m, c_m, ok_m = predict(np.zeros(10), U0, g00, scan0, hkl, hint, dict(lam=0, step=0, V=0, cell=0))
    okk = ok_p & ok_m
    print(f"== {run}: {len(o)} obs; own predictor vs package at p=0: max |dframe| {np.nanmax(np.abs(fr_p[okk]-fr_m[okk])):.2e}, max |drow| {np.nanmax(np.abs(r_p[okk]-r_m[okk])):.2e}, max |dcol| {np.nanmax(np.abs(c_p[okk]-c_m[okk])):.2e}")
    eta = o.eta.to_numpy(); tth = o.tth.to_numpy(); colv = co_
    models = {"M0 package set": dict(lam=0, step=0, V=0), "M1 +lambda": dict(lam=1, step=0, V=0), "M2 +omega-step scale": dict(lam=0, step=1, V=0), "M3 +sideways axis tilt V": dict(lam=0, step=0, V=1), "M4 all extras": dict(lam=1, step=1, V=1), "M5 cell a,b,c scales": dict(lam=0, step=0, V=0, cell=1)}
    for name, ex in models.items():
        idx = list(range(10)) + [10] * ex["lam"] + [11] * ex["step"] + [12] * ex["V"] + ([13, 14, 15] if ex.get("cell") else [])
        def resid(p, full=False):
            pf = np.zeros(16); pf[idx] = p
            fr, ro, co, ok = predict(pf, U0, g00, scan0, hkl, hint, ex)
            r = np.concatenate([(ro - ro_) / 0.25, (co - co_) / 0.25, (fr - fo) / 0.3]); r[~np.tile(ok, 3)] = 30.0
            return (r, fr, ro, co, ok) if full else r
        sol = least_squares(resid, np.zeros(len(idx)), loss="soft_l1", f_scale=3.0, x_scale=XS[idx], diff_step=1e-5, max_nfev=200)
        r, fr, ro, co, ok = resid(sol.x, True); n = len(o)
        dfr = fr - fo; drow = ro - ro_; dcol = co - co_
        rob = lambda x: 1.4826 * np.median(np.abs(x[np.isfinite(x)]))
        pf = np.zeros(16); pf[idx] = sol.x
        side = lambda m: np.nanmedian(dfr[m])
        print(f"{name:26s} cost {sol.cost:9.1f}  rob rms px {rob(np.r_[drow, dcol]):.3f} frame {rob(dfr):.3f} | model-minus-blob frame: col<487 {side(colv < 487):+.3f} col>=487 {side(colv >= 487):+.3f} tth<20 {side(tth < 20):+.3f} tth>40 {side(tth > 40):+.3f} | extras: lam {pf[10]:+.2e} step {pf[11]:+.2e} V {pf[12]:+.3f} deg; cell scales {np.round(pf[13:16], 5)} -> a b c {np.round(np.array([5.8033, 5.95953, 12.2978]) * (1 + pf[13:16]), 4)}; wedge {red['wedge_deg']+pf[9]:+.3f}", flush=True)
        res_all.append(dict(run=run, model=name, cost=sol.cost, rob_frame=rob(dfr), rob_px=rob(np.r_[drow, dcol]), col_lo=side(colv < 487), col_hi=side(colv >= 487), lam=pf[10], step=pf[11], V=pf[12]))
pd.DataFrame(res_all).to_csv(os.path.join(d, "extfit_summary.csv"), index=False)
