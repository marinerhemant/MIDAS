"""PREREGISTER A32: does an orientation error at the precision reached (0.01 / 0.02 / 0.05 deg) move the refined structure parameters?

    python run_u_precision.py <seed_lo> <seed_hi>      # appends results/uprec_<lo>.csv (one row per seed x level)

S2 orientation-seed-1 cache; noisy frames 1000+seed; exact geometry and DAC model; reduction at U_k = Rot(theta_k, per-seed random axis) U_true;
refinement from the C2 N3p start; all levels fitted on the spots shared by ALL levels (the pairing defect of the original C3 is avoided).
"""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "..")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup, corrections as C
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.integrate import build_profile
from midas_sx.merge import merge_observations
from midas_sx.pipeline import find_spots
from midas_sx.reduce import DacModel, PolarizationModel, reduce_spots
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine

lo_s, hi_s = int(sys.argv[1]), int(sys.argv[2])
LEVELS = (0.0, 0.01, 0.02, 0.05)
st = sim.structure_s2_bilayer()
CACHE = Path("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache")
cache = CACHE / f"{st.name}_n100000_o1.npz"
z = np.load(cache)
exp = sim.Experiment(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(str(cache) + ".exp.json")).items()})
res = sim.SimResult(exp=exp, structure=st, U=z["U"], signal=z["signal"], truth=pd.read_pickle(str(cache) + ".truth.pkl"), scale=float(z["scale"]))
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
sg = SpaceGroup.from_hm(st.spacegroup); lattice = Lattice(*st.cell)
truth = Crystal(lattice, sg, [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)
dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm, mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
pol = PolarizationModel(exp.pol_fraction)
out = Path("results") / f"uprec_{lo_s}.csv"
key = lambda o: list(zip(o["frame"].round(3), o["row"].round(3), o["col"].round(3)))

for seed in range(lo_s, hi_s):
    t0 = time.time()
    raw = res.frames(seed=1000 + seed)
    spots, mask, info = find_spots(raw, geom)
    s_all, tth_all, _ = spot_directions(spots.row.to_numpy(), spots.col.to_numpy(), geom)
    L_all = C.rotation_lorentz(s_all, min_abs=1e-3)
    ok = (tth_all > 2.0) & np.isfinite(L_all)
    profile, _ = build_profile(raw, spots[["frame", "row", "col"]].to_numpy(float)[ok], L_all[ok], mask=mask)
    rng = np.random.default_rng(31_000 + seed)
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    obs_by = {}
    for th in LEVELS:
        U = Rot.from_rotvec(np.radians(th) * ax).as_matrix() @ res.U
        obs, rej, cnt = reduce_spots(spots, raw, geom, scan, lattice, U, profile, dac=dac, pol=pol, mask=mask)
        obs_by[th] = obs
    del raw
    shared = set.intersection(*[set(key(o)) for o in obs_by.values()])
    start = perturbed_crystal(truth, np.random.default_rng(10_000 * seed + 2), shift_A=0.10)
    for th in LEVELS:
        o = obs_by[th]
        o = o[[k in shared for k in key(o)]].reset_index(drop=True)
        mm = merge_observations(o, sg)
        m = RefineModel(start)
        r = refine(m, mm[["h", "k", "l"]].to_numpy(int), mm.F2.to_numpy(), mm.sigF2.to_numpy())
        row = dict(seed=seed, theta=th, converged=r.converged, goof=r.goof, n_shared=len(shared), n_unique=len(mm), n_obs=len(o))
        for j, nm in enumerate(r.names):
            if nm == "scale":
                continue
            ia = [a.label for a in truth.atoms].index(nm.split(".")[0])
            if nm.endswith(".U"):
                v = r.x[j]
            else:
                b = m.basis_frac[j]
                dfrac = r.absolute.iloc[ia][["x", "y", "z"]].to_numpy(float) - np.array(start.atoms[ia].fract)
                v = float(dfrac @ b) / float(b @ b)             # displacement from the START along the free direction (a paired difference only needs a common origin)
            row[f"x:{nm}"] = v
            row[f"s:{nm}"] = r.sigma[j]
        with open(out, "a") as fh:
            pd.DataFrame([row]).to_csv(fh, header=fh.tell() == 0, index=False)
    print(f"seed {seed}: shared {len(shared)} of {[len(o) for o in obs_by.values()]}; {time.time()-t0:.0f} s", flush=True)
