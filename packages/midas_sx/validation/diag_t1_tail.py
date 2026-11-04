"""DIAGNOSTIC for the A36 T1 tail (not a registered test): where do the rows with a large quadratic-estimator deficit come from? Reduced on BACKGROUND-ONLY frames (expected intensity 0: any
nonzero value is background structure the 7-term fit does not capture) and on SIGNAL-ONLY frames (expected = exp: any deficit is signal in the shell).

PREREGISTER A36 T1: the A34 harness with the quadratic shell background. Clean frames (signal + background), TRUE U and geometry, boxes from the noisy-frame profile (mean-background build, as the chain made them),
reduced with background="mean" and background="quadratic".

    python run_t1_a36.py <P11|P30|P00> <seed_lo> <seed_hi>     # appends results/a36t1tail_rows_<var>_<lo>.csv

Same targets, reduced twice: (a) on CLEAN frames (signal + background) with the estimated background; (b) on SIGNAL-ONLY frames with the background fixed to its true value 0
(`midas_sx.integrate._robust_mean` patched to return 0 for this reduction only). TRUE U and geometry (the orientation refinement is not the question); the profile is built from the
NOISY frames' blobs exactly as in the chain, so the boxes have the size the chain gives them.
"""
import os, sys, json, time
from dataclasses import replace
sys.path.insert(0, "."); sys.path.insert(0, "..")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
import simulate as sim
from midas_hkls import Lattice, SpaceGroup, corrections as C
import midas_sx.integrate as integ
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.integrate import build_profile
from midas_sx.pipeline import find_spots
from midas_sx.reduce import DacModel, PolarizationModel, reduce_spots
from midas_sx.predict import enumerate_hkl, twin_targets

var, lo_s, hi_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
B_OVER_A = {"P11": 1.011, "P30": 1.03, "P00": 1.011}[var]
M_TRUE = 0.0 if var == "P00" else 0.30
CACHE = Path("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache")
st = sim.structure_s2p(B_OVER_A)
f1, f2 = CACHE / f"{st.name}_dom1.npz", CACHE / f"{st.name}_dom2.npz"
z1 = np.load(f1)
exp = sim.Experiment(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(str(f1) + ".exp.json")).items()})
U1 = z1["U"]; sig = z1["signal"]
if M_TRUE > 0:
    sig += np.load(f2)["signal"]
t1 = pd.read_pickle(str(f1) + ".truth.pkl"); t2 = pd.read_pickle(str(f2) + ".truth.pkl")
res = sim.SimResult(exp=exp, structure=st, U=U1, signal=sig, truth=t1, scale=float(z1["scale"]))
clean = (sig + exp.background_map()[None]).astype(np.float32)
bgonly = np.broadcast_to(exp.background_map()[None].astype(np.float32), sig.shape)
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
lattice = Lattice(*st.cell); sg = SpaceGroup.from_hm(st.spacegroup)
dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm, mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
pol = PolarizationModel(fraction=exp.pol_fraction)
d_min = max(exp.d_min_A, exp.wavelength_A / (2.0 * np.sin(np.radians(exp.tth_max_deg() / 2.0))))
HKL = enumerate_hkl(lattice, d_min, sg)
RZ = Rot.from_euler("z", 90.0, degrees=True).as_matrix()
U2 = U1 @ RZ
csv = Path("results") / f"a36t1tail_rows_{var}_{lo_s}.csv"
_orig = integ._robust_mean


def exact(tab, h, k, l, omega, tol=1.5):
    m = tab[(tab.h == h) & (tab.k == k) & (tab.l == l)]
    if m.empty:
        return 0.0
    j = (m.omega - omega).abs().idxmin()
    return float(m.expected_counts[j]) if abs(m.omega[j] - omega) <= tol else 0.0


key = lambda d: list(zip(d.frame.round(6), d.row.round(6), d.col.round(6)))
for seed in range(lo_s, hi_s):
    t0 = time.time()
    raw = res.frames(seed=1000 + seed)
    spots, mask, info = find_spots(raw, geom)
    s_all, tth_all, _ = spot_directions(spots.row.to_numpy(float), spots.col.to_numpy(float), geom)
    L_all = C.rotation_lorentz(s_all, min_abs=1e-3)
    okm = (tth_all > 2.0) & np.isfinite(L_all)
    profile, _ = build_profile(raw, spots[["frame", "row", "col"]].to_numpy(float)[okm], L_all[okm], mask=mask)
    del raw
    hw_f, hw_p = profile.half_widths(10.0)
    tg = twin_targets(U1, U2, lattice, geom, scan, HKL, d_res_px=2.0 * hw_p, d_res_frames=2.0 * hw_f)
    neigh = tg[["frame", "row", "col"]]
    out_rows = []
    obs_keep = []
    for variant, frames, patch in (("quad_bgonly", bgonly, "quadratic"), ("mean_bgonly", bgonly, "mean"), ("quad_sigonly", sig.astype(np.float32), "quadratic"), ("mean_sigonly", sig.astype(np.float32), "mean")):
        profile_v = replace(profile, background=patch)
        try:
            for kinds, Uu in ((("blend", "c1"), U1), (("c2",), U2)):
                sub = tg[tg.kind.isin(kinds)].reset_index(drop=True)
                if not len(sub):
                    continue
                obs, _, _ = reduce_spots(sub[["frame", "row", "col", "extra_px", "extra_frames"]], frames, geom, scan, lattice, Uu, profile_v, dac=dac, pol=pol, mask=mask,
                                         crowd_px=6.0, neighbours=neigh)
                if not len(obs):
                    continue
                kk = {k_: i_ for i_, k_ in enumerate(key(sub))}
                idx = [kk[k_] for k_ in key(obs)]
                obs = obs.assign(kind=sub.kind.to_numpy()[idx], f1=sub.f1.to_numpy()[idx], f2=sub.f2.to_numpy()[idx], p0=sub.p0.to_numpy()[idx], p1=sub.p1.to_numpy()[idx], p2=sub.p2.to_numpy()[idx], sep_px=sub.sep_px.to_numpy()[idx])
                if False:
                    obs_keep.append(obs.assign(variant=variant))
                for r in obs.itertuples():
                    if r.kind == "c2":
                        e1, e2 = 0.0, (exact(t2, r.h, r.k, r.l, r.omega) if M_TRUE > 0 else 0.0)
                    elif r.kind == "c1":
                        e1, e2 = exact(t1, r.h, r.k, r.l, r.omega), 0.0
                    else:
                        e1 = exact(t1, r.h, r.k, r.l, r.omega)
                        e2 = exact(t2, r.p0, r.p1, r.p2, r.omega, tol=3.0) if M_TRUE > 0 else 0.0
                    out_rows.append(dict(seed=seed, var=var, variant=variant, kind=r.kind, h=r.h, k=r.k, l=r.l, I=r.I, sigI=r.sigI, bg=r.bg, n_box=r.n_box, sep_px=r.sep_px,
                                         exp1=e1, exp2=e2, exp_tot=e1 + e2))
        finally:
            pass
    pd.DataFrame(out_rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    if obs_keep:
        Path("results/a35_obs").mkdir(exist_ok=True)
        pd.concat(obs_keep, ignore_index=True).to_pickle(f"results/a35_obs/{var}_seed{seed}.pkl")
    print(f"seed {seed} {var}: rows {len(out_rows)}; {time.time()-t0:.0f} s", flush=True)
