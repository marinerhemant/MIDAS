"""PREREGISTER A36 T2(a): the A29 harness (noisy-frame chain with the refined orientation, rows scored against the simulator's exact expected counts) with the background estimator
as the only variable: every target is reduced on the NOISY frames with background="mean" and with "quadratic" from the same refined U and geometry. No clean-frame arm, no D2.

    python run_t2a_a36.py <P11|P30|P00> <seed_lo> <seed_hi>     # appends results/a36t2a_rows_<var>_<lo>.csv

The A26 chain is run unchanged (run_pm.py; same wrong-start geometry, same index_and_refine, same twin_targets) on NEW noise seeds (50 and up, never scored in A26/A27).
Spot finding and the orientation refinement see the NOISY frames, exactly as before. The measurement targets are then reduced TWICE at the same positions: on the noisy
frames and on the matching NOISE-FREE frames (signal + background, no Poisson). Each reduced row is compared with the simulator's exact expected counts of the
component(s) it claims to measure:  c1 -> dom-1 (h,k,l);  c2 -> dom-2 (h,k,l);  blend -> dom-1 (h,k,l) + dom-2 (p0,p1,p2).
Also D2: the orientation refinement is repeated on the noise-free frames (same blobs positions are NOT available there, so the noisy blob list is reused).
"""
import sys, time, json, resource
sys.path.insert(0, "."); sys.path.insert(0, "..")
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot

import simulate as sim
from midas_hkls import Lattice, SpaceGroup, corrections as C
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.integrate import build_profile
from midas_sx.pipeline import find_spots
from midas_sx.reduce import DacModel, PolarizationModel, reduce_spots
from midas_sx.orient import index_and_refine
from midas_sx.predict import enumerate_hkl, twin_targets

var, lo_s, hi_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
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
t1 = pd.read_pickle(str(f1) + ".truth.pkl")
t2 = pd.read_pickle(str(f2) + ".truth.pkl")
res = sim.SimResult(exp=exp, structure=st, U=U1, signal=sig, truth=t1, scale=float(z1["scale"]))
bgmap = exp.background_map()
clean = (sig + bgmap[None]).astype(np.float32)                    # NOISE-FREE frames
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
lattice = Lattice(*st.cell)
sg = SpaceGroup.from_hm(st.spacegroup)
dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm, mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
pol = PolarizationModel(fraction=exp.pol_fraction)
d_min = max(exp.d_min_A, exp.wavelength_A / (2.0 * np.sin(np.radians(exp.tth_max_deg() / 2.0))))
HKL = enumerate_hkl(lattice, d_min, sg)
RZ = Rot.from_euler("z", 90.0, degrees=True).as_matrix()
mis = lambda A_, B_: float(np.degrees(np.linalg.norm(Rot.from_matrix(A_ @ B_.T).as_rotvec())))
csv = Path("results") / f"a36t2a_rows_{var}_{lo_s}.csv"
csv_o = Path("results") / f"a29_orient_{var}_{lo_s}.csv"


def exact(tab, h, k, l, omega, tol=1.5):
    m = tab[(tab.h == h) & (tab.k == k) & (tab.l == l)]
    if m.empty:
        return 0.0
    j = (m.omega - omega).abs().idxmin()
    return float(m.expected_counts[j]) if abs(m.omega[j] - omega) <= tol else 0.0


def reduce_targets(tg, raw, geomU, U, profile, mask, neigh):
    obs, rej, cnt = reduce_spots(tg[["frame", "row", "col", "extra_px", "extra_frames"]].reset_index(drop=True), raw, geomU, scan, lattice, U, profile,
                                 dac=dac, pol=pol, mask=mask, crowd_px=6.0, neighbours=neigh)
    return obs


key = lambda d: list(zip(d.frame.round(6), d.row.round(6), d.col.round(6)))
for seed in range(lo_s, hi_s):
    t0 = time.time()
    raw = res.frames(seed=1000 + seed)
    rng_a = np.random.default_rng(777 + seed)
    ax = rng_a.normal(size=3); ax /= np.linalg.norm(ax)
    U1s = Rot.from_rotvec(np.radians(0.3) * ax).as_matrix() @ U1
    phi = rng_a.uniform(0, 2 * np.pi)
    g0 = replace(geom, bcz_px=geom.bcz_px + 0.5 * np.cos(phi), bcy_px=geom.bcy_px + 0.5 * np.sin(phi), lsd_um=geom.lsd_um * (1 + 0.01 * rng_a.choice([-1.0, 1.0])))
    spots, mask, info = find_spots(raw, g0)
    o1 = index_and_refine(spots, lattice, U1s, g0, scan)
    U2 = o1.U @ RZ
    gu = o1.geom
    s_all, tth_all, _ = spot_directions(spots.row.to_numpy(float), spots.col.to_numpy(float), gu)
    L_all = C.rotation_lorentz(s_all, min_abs=1e-3)
    okm = (tth_all > 2.0) & np.isfinite(L_all)
    profile, _ = build_profile(raw, spots[["frame", "row", "col"]].to_numpy(float)[okm], L_all[okm], mask=mask)
    tg = twin_targets(o1.U, U2, lattice, gu, scan, HKL, d_res_px=2.0 * profile.half_widths(10.0)[1], d_res_frames=2.0 * profile.half_widths(10.0)[0])
    neigh = tg[["frame", "row", "col"]]
    out_rows = []
    for frames_kind, fr in (("mean", raw), ("quadratic", raw)):
        profile_v = replace(profile, background=frames_kind)
        for kinds, Uu in ((("blend", "c1"), o1.U), (("c2",), U2)):
            sub = tg[tg.kind.isin(kinds)].reset_index(drop=True)
            if not len(sub):
                continue
            obs = reduce_targets(sub, fr, gu, Uu, profile_v, mask, neigh)
            if not len(obs):
                continue
            kk = {k_: i_ for i_, k_ in enumerate(key(sub))}
            idx = [kk[k_] for k_ in key(obs)]
            obs = obs.assign(kind=sub.kind.to_numpy()[idx], p0=sub.p0.to_numpy()[idx], p1=sub.p1.to_numpy()[idx], p2=sub.p2.to_numpy()[idx], sep_px=sub.sep_px.to_numpy()[idx])
            for r in obs.itertuples():
                if r.kind == "c2":
                    e1, e2 = 0.0, (exact(t2, r.h, r.k, r.l, r.omega) if M_TRUE > 0 else 0.0)
                elif r.kind == "c1":
                    e1, e2 = exact(t1, r.h, r.k, r.l, r.omega), 0.0
                else:
                    e1 = exact(t1, r.h, r.k, r.l, r.omega)
                    e2 = exact(t2, r.p0, r.p1, r.p2, r.omega, tol=3.0) if M_TRUE > 0 else 0.0
                out_rows.append(dict(seed=seed, var=var, frames=frames_kind, kind=r.kind, h=r.h, k=r.k, l=r.l, I=r.I, sigI=r.sigI, bg=r.bg, n_box=r.n_box,
                                     sep_px=r.sep_px, tth=r.tth, L=r.L, exp1=e1, exp2=e2, exp_tot=e1 + e2))
    pd.DataFrame(out_rows).to_csv(csv, mode="a", header=not csv.exists(), index=False)
    print(f"seed {seed} {var}: rows {len(out_rows)}; U1 err noisy {mis(o1.U, U1):.4f} deg; {time.time()-t0:.0f} s "
          f"RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MB", flush=True)
