"""PREREGISTER A23: two crystal domains. Frames = Poisson(signal_1 + signal_2 + background); arms S1 (single-domain chain) and M (two-domain chain).

    python run_twin.py <seed_lo> <seed_hi>      # appends results/twin_<lo>.csv; merged tables in results/twin_merged/

Needs the caches of run_c2.py (domain 1: orientation seed 1; domain 2: `DOM2=1 python run_c2.py S2 make-cache`, orientation seed 3, intensity
0.3/0.7 of domain 1 at the same absolute scale). Nothing here sees the true U except to start 0.3 deg off and to score.
"""
import sys, time, json, resource
sys.path.insert(0, "."); sys.path.insert(0, "..")
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup, corrections as C
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.integrate import build_profile
from midas_sx.merge import merge_observations
from midas_sx.pipeline import find_spots
from midas_sx.reduce import DacModel, PolarizationModel, reduce_spots
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine
from midas_sx.orient import index_and_refine
from midas_sx.assign import assign_hkl

lo_s, hi_s = int(sys.argv[1]), int(sys.argv[2])
CACHE = Path("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache")
st = sim.structure_s2_bilayer()
f1, f2 = CACHE / "S2_bilayer_like_n100000_o1.npz", CACHE / "S2_bilayer_like_n100000_o3.npz"
z1, z2 = np.load(f1), np.load(f2)
exp = sim.Experiment(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(str(f1) + ".exp.json")).items()})
truth1, truth2 = pd.read_pickle(str(f1) + ".truth.pkl"), pd.read_pickle(str(f2) + ".truth.pkl")
U1, U2 = z1["U"], z2["U"]
sig_sum = z1["signal"]; sig_sum += z2["signal"]                           # in place: two 335 MB arrays at the peak, not three (Mac RAM rule)
res = sim.SimResult(exp=exp, structure=st, U=U1, signal=sig_sum, truth=truth1, scale=float(z1["scale"]))
true_ratio = float(z2["scale"]) / float(z1["scale"])
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc,
                     shape=(exp.det_n, exp.det_n), scan=scan)
truth = Crystal(Lattice(*st.cell), SpaceGroup.from_hm(st.spacegroup),
                [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)
sg = SpaceGroup.from_hm(st.spacegroup)
lattice = Lattice(*st.cell)
dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm,
               mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
pol = PolarizationModel(fraction=exp.pol_fraction)
out = Path("results"); (out / "twin_merged").mkdir(parents=True, exist_ok=True)
csv = out / f"twin_{lo_s}.csv"
mis = lambda A, B: float(np.degrees(np.linalg.norm(Rot.from_matrix(A @ B.T).as_rotvec())))


def errors(r, model):
    o = {}
    for j, (nm, kind) in enumerate(zip(r.names, r.kinds)):
        if kind in ("scale", "ext"):
            continue
        ia = [a.label for a in truth.atoms].index(nm.split(".")[0])
        if kind == "U":
            o[nm] = r.x[j] - truth.atoms[ia].B_iso / EIGHT_PI2
        else:
            b = model.basis_frac[j]
            d = r.absolute.iloc[ia][["x", "y", "z"]].to_numpy(float) - np.array(truth.atoms[ia].fract)
            o[nm] = float(d @ b) / float(b @ b)
    return o


def rot_start(U, rng, deg=0.3):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    return Rot.from_rotvec(np.radians(deg) * ax).as_matrix() @ U


for seed in range(lo_s, hi_s):
    t0 = time.time()
    raw = res.frames(seed=1000 + seed)
    rng_a, rng_b = np.random.default_rng(777 + seed), np.random.default_rng(888 + seed)
    U1s = rot_start(U1, rng_a); phi = rng_a.uniform(0, 2 * np.pi)
    g_start = replace(geom, bcz_px=geom.bcz_px + 0.5 * np.cos(phi), bcy_px=geom.bcy_px + 0.5 * np.sin(phi),
                      lsd_um=geom.lsd_um * (1 + 0.01 * rng_a.choice([-1.0, 1.0])))
    U2s = rot_start(U2, rng_b)
    spots, mask, info = find_spots(raw, g_start)
    o1 = index_and_refine(spots, lattice, U1s, g_start, scan)
    fr_, ro_, co_ = (spots[c].to_numpy(float) for c in ("frame", "row", "col"))
    s_, tth_, _ = spot_directions(ro_, co_, o1.geom)
    om_ = scan.omega_of(fr_)
    _, _, r1 = assign_hkl(s_, om_, lattice, o1.U, o1.geom.wavelength_A)
    cand = spots[r1 > 0.25].reset_index(drop=True)
    try:
        o2 = index_and_refine(cand, lattice, U2s, o1.geom, scan, refine_geometry=False)
    except Exception as e:                                                  # noqa: BLE001
        pd.DataFrame([dict(seed=seed, arm="-", note=f"domain-2 orientation failed: {e}")]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        continue
    odiag = dict(u1_err_deg=mis(o1.U, U1), u2_err_deg=mis(o2.U, U2), u1_start_deg=mis(U1s, U1), u2_start_deg=mis(U2s, U2),
                 o1_converged=o1.converged, o2_converged=o2.converged, o1_n_used=o1.n_used, o2_n_used=o2.n_used, n_blobs=len(spots), n_cand=len(cand),
                 bc_row_err=o1.geom.bcz_px - geom.bcz_px, bc_col_err=o1.geom.bcy_px - geom.bcy_px, dlsd_rel_err=o1.geom.lsd_um / geom.lsd_um - 1.0)
    geom_use = o1.geom
    s_all, tth_all, _ = spot_directions(ro_, co_, geom_use)
    L_all = C.rotation_lorentz(s_all, min_abs=1e-3)
    okm = (tth_all > 2.0) & np.isfinite(L_all)
    try:
        model, _ = build_profile(raw, spots[["frame", "row", "col"]].to_numpy(float)[okm], L_all[okm], mask=mask)
    except RuntimeError as e:
        pd.DataFrame([dict(seed=seed, arm="-", note=f"profile model failed: {e}")]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
        continue
    # S1: single-domain chain: every found spot goes to reduce_spots with domain 1's U
    obs1, rej1, cnt1 = reduce_spots(spots, raw, geom_use, scan, lattice, o1.U, model, dac=dac, pol=pol, mask=mask)
    mS1 = merge_observations(obs1, sg)
    # M: label each spot by which domain explains it; both -> ambiguous (dropped from both), neither -> unexplained
    sA, _, _ = spot_directions(ro_, co_, geom_use)
    _, _, r2 = assign_hkl(sA, om_, lattice, o2.U, geom_use.wavelength_A)
    _, _, r1b = assign_hkl(sA, om_, lattice, o1.U, geom_use.wavelength_A)
    lab = np.zeros(len(spots), int)
    lab[(r1b <= 0.15) & (r2 > 0.25)] = 1
    lab[(r2 <= 0.15) & (r1b > 0.25)] = 2
    obsA, _, cntA = reduce_spots(spots[lab == 1].reset_index(drop=True), raw, geom_use, scan, lattice, o1.U, model, dac=dac, pol=pol, mask=mask,
                                 neighbours=spots)
    obsB, _, cntB = reduce_spots(spots[lab == 2].reset_index(drop=True), raw, geom_use, scan, lattice, o2.U, model, dac=dac, pol=pol, mask=mask,
                                 neighbours=spots)
    mA, mB = merge_observations(obsA, sg), merge_observations(obsB, sg)
    # M2 (PREREGISTER A23a, exploratory): ambiguous only when BOTH domains explain the spot inside the reduction gate (0.15)
    lab2 = np.zeros(len(spots), int)
    lab2[(r1b <= 0.15) & (r2 > 0.15)] = 1
    lab2[(r2 <= 0.15) & (r1b > 0.15)] = 2
    obsA2, _, cntA2 = reduce_spots(spots[lab2 == 1].reset_index(drop=True), raw, geom_use, scan, lattice, o1.U, model, dac=dac, pol=pol, mask=mask,
                                   neighbours=spots)
    obsB2, _, cntB2 = reduce_spots(spots[lab2 == 2].reset_index(drop=True), raw, geom_use, scan, lattice, o2.U, model, dac=dac, pol=pol, mask=mask,
                                   neighbours=spots)
    mA2, mB2 = merge_observations(obsA2, sg), merge_observations(obsB2, sg)
    np.savez(out / "twin_merged" / f"S2_twin_seed{seed}.npz", s1_hkl=mS1[["h", "k", "l"]].to_numpy(int), s1_F2=mS1.F2.to_numpy(), s1_sig=mS1.sigF2.to_numpy(),
             a_hkl=mA[["h", "k", "l"]].to_numpy(int), a_F2=mA.F2.to_numpy(), a_sig=mA.sigF2.to_numpy(),
             b_hkl=mB[["h", "k", "l"]].to_numpy(int), b_F2=mB.F2.to_numpy(), b_sig=mB.sigF2.to_numpy(),
             a2_hkl=mA2[["h", "k", "l"]].to_numpy(int), a2_F2=mA2.F2.to_numpy(), a2_sig=mA2.sigF2.to_numpy(),
             b2_hkl=mB2[["h", "k", "l"]].to_numpy(int), b2_F2=mB2.F2.to_numpy(), b2_sig=mB2.sigF2.to_numpy())
    # contamination (truth): kept domain-1 reflections whose support box contains a true domain-2 spot (any intensity)
    cont = np.nan
    try:
        t1 = obs1[["frame", "row", "col"]].to_numpy(float) if {"frame", "row", "col"} <= set(obs1.columns) else None
        if t1 is not None and len(t1):
            om2 = (truth2.omega.to_numpy() - exp.omega_start) / exp.omega_step - 0.5
            d = np.hypot(t1[:, 1][:, None] - truth2.row.to_numpy()[None, :], t1[:, 2][:, None] - truth2.col.to_numpy()[None, :])
            df = np.abs(t1[:, 0][:, None] - om2[None, :])
            cont = float(np.mean(np.any((d < 8.0) & (df < 6.0), axis=1)))
    except Exception:                                                       # noqa: BLE001
        pass
    # refinements
    rows = []
    start = perturbed_crystal(truth, np.random.default_rng(10_000 * seed + 2), shift_A=0.10)
    for arm in ("S1", "M", "M2"):
        if arm == "S1":
            model_r = RefineModel(start)
            H, F2o, sg_ = mS1[["h", "k", "l"]].to_numpy(int), mS1.F2.to_numpy(), mS1.sigF2.to_numpy()
            r = refine(model_r, H, F2o, sg_)
            nu, cnt = len(H), cnt1
        else:
            mX, mY, cX, cY, oX, oY = (mA, mB, cntA, cntB, obsA, obsB) if arm == "M" else (mA2, mB2, cntA2, cntB2, obsA2, obsB2)
            model_r = RefineModel(start, n_domains=2)
            H = np.vstack([mX[["h", "k", "l"]].to_numpy(int), mY[["h", "k", "l"]].to_numpy(int)])
            F2o = np.concatenate([mX.F2.to_numpy(), mY.F2.to_numpy()]); sg_ = np.concatenate([mX.sigF2.to_numpy(), mY.sigF2.to_numpy()])
            dom = np.concatenate([np.zeros(len(mX), int), np.ones(len(mY), int)])
            r = refine(model_r, H, F2o, sg_, domain=dom)
            nu, cnt = len(H), {k: cX.get(k, 0) + cY.get(k, 0) for k in cX}
        err = errors(r, model_r)
        row = dict(seed=seed, arm=arm, converged=r.converged, poor_fit=r.poor_fit, reason=r.reason, goof=r.goof, chi2=r.chi2, n_unique=nu,
                   n_obs=int(len(obs1) if arm == "S1" else (len(obsA) + len(obsB) if arm == "M" else len(obsA2) + len(obsB2))), n_par=r.n_par,
                   n_unique_A2=len(mA2), n_unique_B2=len(mB2), nfev=r.nfev, sv_ratio=r.sv_ratio,
                   n_unique_A=len(mA), n_unique_B=len(mB), n_unique_S1=len(mS1), contamination=cont, true_ratio=true_ratio,
                   k_ratio=(r.x[-1] / r.x[0]) if arm in ("M", "M2") else np.nan, rejected=json.dumps({k: v for k, v in cnt.items() if v and k not in ("input", "kept")}),
                   seconds=time.time() - t0, **odiag)
        for j, nm in enumerate(r.names):
            row[f"x:{nm}"] = r.x[j]; row[f"s:{nm}"] = r.sigma[j]
            if nm in err:
                row[f"e:{nm}"] = err[nm]
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(f"seed {seed}: U1 err {odiag['u1_err_deg']:.4f} U2 err {odiag['u2_err_deg']:.4f}; S1 {len(mS1)} unique conv={rows[0]['converged']} GooF {rows[0]['goof']:.2f}; "
          f"M {len(mA)}+{len(mB)} unique conv={rows[1]['converged']} GooF {rows[1]['goof']:.2f} k2/k1 {rows[1]['k_ratio']:.3f} (true {true_ratio:.3f}); "
          f"M2 {len(mA2)}+{len(mB2)} unique conv={rows[2]['converged']} GooF {rows[2]['goof']:.2f}; "
          f"{time.time()-t0:.0f} s RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MB", flush=True)
