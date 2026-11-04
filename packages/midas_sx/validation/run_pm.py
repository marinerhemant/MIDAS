"""PREREGISTER A26: pseudo-merohedral twin (a != b). Predict-and-integrate both twin components, refine the twin fraction with component flags.

    python run_pm.py <P11|P30|P00> <seed_lo> <seed_hi> [explore]    # appends results/pm_<variant>_<lo>.csv  (explore: results/pm_explore_*.csv, unscored seeds)

P11: b/a = 1.011, m = 0.30; P30: b/a = 1.03, m = 0.30; P00: b/a = 1.011, domain 1 only (m = 0, the null). Frames = Poisson(sig1 + r sig2 + background), r = m/0.3 * 0.4286
relative to the saved 0.3 caches. The chain never sees the truth except to start U1 0.3 deg off (A19) and to score.
"""
import sys, time, json, resource
sys.path.insert(0, "."); sys.path.insert(0, "..")
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot

import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup, corrections as C
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.integrate import build_profile
from midas_sx.pipeline import find_spots
from midas_sx.reduce import DacModel, PolarizationModel, reduce_spots
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine
from midas_sx.orient import index_and_refine
from midas_sx.predict import TWIN_90_C, enumerate_hkl, twin_targets

var, lo_s, hi_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
EXPLORE = len(sys.argv) > 4 and sys.argv[4] == "explore"
B_OVER_A = {"P11": 1.011, "P30": 1.03, "P00": 1.011}[var]
M_TRUE = 0.0 if var == "P00" else 0.30
CACHE = Path("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache")
st = sim.structure_s2p(B_OVER_A)
f1, f2 = CACHE / f"{st.name}_dom1.npz", CACHE / f"{st.name}_dom2.npz"
z1 = np.load(f1)
exp = sim.Experiment(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(str(f1) + ".exp.json")).items()})
U1 = z1["U"]
sig = z1["signal"]
if M_TRUE > 0:
    sig += np.load(f2)["signal"]
res = sim.SimResult(exp=exp, structure=st, U=U1, signal=sig, truth=pd.read_pickle(str(f1) + ".truth.pkl"), scale=float(z1["scale"]))
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
lattice = Lattice(*st.cell)
sg = SpaceGroup.from_hm(st.spacegroup)
truth = Crystal(lattice, sg, [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)
labels = [a.label for a in truth.atoms]
dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm, mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
pol = PolarizationModel(fraction=exp.pol_fraction)
d_min = max(exp.d_min_A, exp.wavelength_A / (2.0 * np.sin(np.radians(exp.tth_max_deg() / 2.0))))
HKL = enumerate_hkl(lattice, d_min, sg)
out = Path("results"); csv = out / (f"pm_explore_{var}_{lo_s}.csv" if EXPLORE else f"pm_{var}_{lo_s}.csv")
RZ = Rot.from_euler("z", 90.0, degrees=True).as_matrix()
mis = lambda A_, B_: float(np.degrees(np.linalg.norm(Rot.from_matrix(A_ @ B_.T).as_rotvec())))


def errors(r, model):
    o = {}
    for j, (nm, kind) in enumerate(zip(r.names, r.kinds)):
        if kind in ("scale", "ext"):
            continue
        if kind == "twin":
            o[nm] = r.x[j] - M_TRUE; continue
        ia = labels.index(nm.split(".")[0])
        if kind == "U":
            o[nm] = r.x[j] - truth.atoms[ia].B_iso / EIGHT_PI2
        else:
            b = model.basis_frac[j]
            d = r.absolute.iloc[ia][["x", "y", "z"]].to_numpy(float) - np.array(truth.atoms[ia].fract)
            o[nm] = float(d @ b) / float(b @ b)
    return o


def reduce_targets(tg, raw, geomU, U, profile, mask, neigh):
    if not len(tg):
        return pd.DataFrame(), {}
    obs, rej, cnt = reduce_spots(tg[["frame", "row", "col", "extra_px", "extra_frames"]].reset_index(drop=True), raw, geomU, scan, lattice, U, profile,
                                 dac=dac, pol=pol, mask=mask, crowd_px=6.0, neighbours=neigh)
    return obs, cnt


for seed in range(lo_s + (200 if EXPLORE else 0), hi_s + (200 if EXPLORE else 0)):
    t0 = time.time()
    raw = res.frames(seed=1000 + seed)
    rng_a = np.random.default_rng(777 + seed)
    ax = rng_a.normal(size=3); ax /= np.linalg.norm(ax)
    U1s = Rot.from_rotvec(np.radians(0.3) * ax).as_matrix() @ U1
    phi = rng_a.uniform(0, 2 * np.pi)
    g0 = replace(geom, bcz_px=geom.bcz_px + 0.5 * np.cos(phi), bcy_px=geom.bcy_px + 0.5 * np.sin(phi), lsd_um=geom.lsd_um * (1 + 0.01 * rng_a.choice([-1.0, 1.0])))
    spots, mask, info = find_spots(raw, g0)
    o1 = index_and_refine(spots, lattice, U1s, g0, scan)
    U2 = o1.U @ RZ                                           # the twin law is known: second component = first rotated +90 deg about c
    gu = o1.geom
    s_all, tth_all, _ = spot_directions(spots.row.to_numpy(float), spots.col.to_numpy(float), gu)
    L_all = C.rotation_lorentz(s_all, min_abs=1e-3)
    okm = (tth_all > 2.0) & np.isfinite(L_all)
    try:
        profile, _ = build_profile(raw, spots[["frame", "row", "col"]].to_numpy(float)[okm], L_all[okm], mask=mask)
    except RuntimeError as e:
        pd.DataFrame([dict(seed=seed, arm="-", note=f"profile model failed: {e}")]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        continue
    tg = twin_targets(o1.U, U2, lattice, gu, scan, HKL, d_res_px=2.0 * profile.half_widths(10.0)[1], d_res_frames=2.0 * profile.half_widths(10.0)[0])
    neigh = tg[["frame", "row", "col"]]
    key = lambda d: list(zip(d.frame.round(6), d.row.round(6), d.col.round(6)))
    rows_obs = []
    for kinds, Uu in ((("blend", "c1"), o1.U), (("c2",), U2)):
        sub = tg[tg.kind.isin(kinds)].reset_index(drop=True)
        obs, cnt = reduce_targets(sub, raw, gu, Uu, profile, mask, neigh)
        if len(obs):
            kk = {k_: i_ for i_, k_ in enumerate(key(sub))}
            idx = [kk[k_] for k_ in key(obs)]
            obs = obs.assign(kind=sub.kind.to_numpy()[idx], f1=sub.f1.to_numpy()[idx], f2=sub.f2.to_numpy()[idx],
                             p0=sub.p0.to_numpy()[idx], p1=sub.p1.to_numpy()[idx], p2=sub.p2.to_numpy()[idx], sep_px=sub.sep_px.to_numpy()[idx])
            rows_obs.append(obs)
    O = pd.concat(rows_obs, ignore_index=True) if rows_obs else pd.DataFrame()
    if len(O) < 20:
        pd.DataFrame([dict(seed=seed, arm="-", note=f"too few observations: {len(O)}")]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        continue
    np.savez(out / "pm_merged" / f"{var}_seed{seed}.npz", **{c_: O[c_].to_numpy() for c_ in ("h", "k", "l", "F2", "sigF2", "kind", "f1", "f2", "p0", "p1", "p2", "sep_px")}) if (out / "pm_merged").exists() else None
    H = O[["h", "k", "l"]].to_numpy(int)
    P = O[["p0", "p1", "p2"]].to_numpy(int)
    F2o, sg_ = O.F2.to_numpy(), O.sigF2.to_numpy()
    fl = O[["f1", "f2"]].to_numpy(float)
    c2m = O.kind.to_numpy() == "c2"
    H_eff = H.copy(); P_eff = P.copy()
    H_eff[c2m] = H[c2m]                                      # a c2 row: assigned under U2, so (h,k,l) is already the domain-2 index of the component
    P_eff[c2m] = H[c2m]
    start = perturbed_crystal(truth, np.random.default_rng(10_000 * seed + 2), shift_A=0.10)
    base = dict(seed=seed, variant=var, m_true=M_TRUE, b_over_a=B_OVER_A, u1_err_deg=mis(o1.U, U1), o1_converged=o1.converged, n_blobs=len(spots),
                n_targets=len(tg), n_obs=len(O), n_blend=int((O.kind == "blend").sum()), n_c1=int((O.kind == "c1").sum()), n_c2=int(c2m.sum()),
                sep_med=float(np.nanmedian(O.sep_px)) if O.sep_px.notna().any() else np.nan, bc_row_err=gu.bcz_px - geom.bcz_px, bc_col_err=gu.bcy_px - geom.bcy_px,
                dlsd_rel_err=gu.lsd_um / geom.lsd_um - 1.0)
    rows = []
    for arm in ("U", "T", "X"):
        t1 = time.time()
        if arm == "U":                                       # today's practice: the domain-1 components only, no twin
            sel = ~c2m & (fl[:, 0] == 1)
            model = RefineModel(start)
            r = refine(model, H[sel], F2o[sel], sg_[sel])
        elif arm == "T":                                     # the model of this stage: component flags and partner indices
            model = RefineModel(start, refine_twin=True, twin_start=0.15)
            r = refine(model, H_eff, F2o, sg_, partner=P_eff, flags=fl)
        else:                                                # the stage-2 (exact overlap) model applied to every row, split ignored
            model = RefineModel(start, refine_twin=True, twin_start=0.15)
            sel = ~c2m
            r = refine(model, H[sel], F2o[sel], sg_[sel])
        err = errors(r, model)
        row = dict(base, arm=arm, converged=r.converged, poor_fit=r.poor_fit, reason=r.reason, goof=r.goof, chi2=r.chi2, n_used=int(sel.sum()) if arm != "T" else len(O),
                   n_par=r.n_par, nfev=r.nfev, sv_ratio=r.sv_ratio, seconds=time.time() - t1, m=r.twin_fraction, sigma_m=r.sigma_twin_fraction)
        for j, nm in enumerate(r.names):
            row[f"x:{nm}"] = r.x[j]; row[f"s:{nm}"] = r.sigma[j]
            if nm in err:
                row[f"e:{nm}"] = err[nm]
        if arm == "T":
            Cm = r.correlation(); im = r.names.index("twin.m")
            for j, nm in enumerate(r.names):
                if j != im:
                    row[f"c:{nm}"] = Cm[im, j]
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(f"seed {seed} {var}: U1 err {base['u1_err_deg']:.4f} deg; obs {len(O)} (blend {base['n_blend']}, c1 {base['n_c1']}, c2 {base['n_c2']}); "
          + "; ".join(f"{r_['arm']} conv={r_['converged']} GooF {r_['goof']:.2f}" for r_ in rows) + f"; m {rows[1]['m']:.3f} +- {rows[1]['sigma_m']:.3f}; {time.time()-t0:.0f} s "
          f"RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MB", flush=True)
