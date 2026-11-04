"""EXPLORATORY (not preregistered): can U be refined from the MEASURED spot positions well enough, and what does the
refined U do to the structure refinement?   python run_u_refine.py
Spots: the C2 seed-0 noisy S2 frames through the real front end. hkl assignment uses the true U (indexing is not tested here).
U refinement: midas_defect.seed_index.refine_U_from_centroids (cosine loss on sample-frame q directions).
"""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "..")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as R
import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup, corrections as C
from midas_sx.assign import assign_hkl, q_sample_of_spots
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.integrate import build_profile
from midas_sx.merge import merge_observations
from midas_sx.pipeline import find_spots
from midas_sx.reduce import DacModel, PolarizationModel, reduce_spots
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine
from midas_defect.seed_index import refine_U_from_centroids

st = sim.structure_s2_bilayer()
CACHE = Path("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache")
cache = CACHE / f"{st.name}_n100000_o1.npz"
z = np.load(cache)
exp = sim.Experiment(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(str(cache) + ".exp.json")).items()})
res = sim.SimResult(exp=exp, structure=st, U=z["U"], signal=z["signal"], truth=pd.read_pickle(str(cache) + ".truth.pkl"), scale=float(z["scale"]))
U0 = res.U
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
sg = SpaceGroup.from_hm(st.spacegroup); lattice = Lattice(*st.cell)
truth = Crystal(lattice, sg, [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)
raw = res.frames(seed=1000)
spots, mask, info = find_spots(raw, geom)
s, tth, _ = spot_directions(spots.row.to_numpy(), spots.col.to_numpy(), geom)
om = scan.omega_of(spots.frame.to_numpy())
hkl, hf, resid = assign_hkl(s, om, lattice, U0, geom.wavelength_A)
ok = (tth > 2.0) & (resid < 0.15) & np.any(hkl != 0, axis=1)
print(f"spots {len(spots)}, assigned with true U: {int(ok.sum())}")
q = q_sample_of_spots(s[ok], om[ok], geom.wavelength_A) * 2 * np.pi
pairs = [(tuple(map(int, h)), qq) for h, qq in zip(hkl[ok], q)]
a, c = st.cell[0], st.cell[2]


def mis_deg(U1, U2):
    return float(np.degrees(np.linalg.norm(R.from_matrix(U1 @ U2.T).as_rotvec())))


def start(deg, axis, seed=None):
    ax = np.asarray(axis, float) if axis is not None else np.random.default_rng(seed).normal(size=3)
    return R.from_rotvec(np.radians(deg) * ax / np.linalg.norm(ax)).as_matrix() @ U0


lines = [f"EXPLORATORY U refinement from MEASURED positions: {int(ok.sum())} assigned spots, pixel = {exp.pixel_mm / exp.dist_mm * 180 / np.pi:.3f} deg",
         "start offset -> refined misorientation from truth [deg] (n_steps 400 / 2000 / 6000, Adam lr 5e-3)"]
for deg, axis, name in ((0.1, (1, 1, 1), "0.1 deg (1,1,1)"), (0.3, (1, 1, 1), "0.3 deg (1,1,1)"), (0.6, (1, 1, 1), "0.6 deg (1,1,1)"),
                        (0.3, (1, 0, 0), "0.3 deg about beam"), (0.3, None, "0.3 deg random axis")):
    Us = start(deg, axis, seed=3)
    out = []
    for n in (400, 2000, 6000):
        Ur, _, _, loss = refine_U_from_centroids(Us, pairs, a=a, c=c, n_steps=n, lr=5e-3, device="cpu", dtype="float64")
        out.append(mis_deg(Ur, U0))
    lines.append(f"  {name:22s} start {mis_deg(Us, U0):.3f} -> " + " / ".join(f"{x:.4f}" for x in out))

# bootstrap on spots: scatter of the refined U
rng = np.random.default_rng(0)
Us = start(0.3, (1, 1, 1))
ref_all, _, _, _ = refine_U_from_centroids(Us, pairs, a=a, c=c, n_steps=6000, lr=5e-3, device="cpu", dtype="float64")
boots = []
for _ in range(20):
    idx = rng.choice(len(pairs), size=int(0.7 * len(pairs)), replace=False)
    Ub, _, _, _ = refine_U_from_centroids(Us, [pairs[i] for i in idx], a=a, c=c, n_steps=6000, lr=5e-3, device="cpu", dtype="float64")
    boots.append(Ub)
dev = [mis_deg(Ub, ref_all) for Ub in boots]
lines.append(f"  bootstrap (70 % of spots, 20 draws): rms spread of refined U about the full-data refined U {np.sqrt(np.mean(np.square(dev))):.4f} deg; full-data refined U is {mis_deg(ref_all, U0):.4f} deg from truth")

# propagate: structure refinement with the refined U versus the exact U (same spots, same noise)
profile, _ = build_profile(raw, spots[["frame", "row", "col"]].to_numpy(float)[(tth > 2.0) & np.isfinite(C.rotation_lorentz(s, min_abs=1e-3))], C.rotation_lorentz(s, min_abs=1e-3)[(tth > 2.0) & np.isfinite(C.rotation_lorentz(s, min_abs=1e-3))], mask=mask)
dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm, mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
pol = PolarizationModel(exp.pol_fraction)
start_crystal = perturbed_crystal(truth, np.random.default_rng(7), shift_A=0.10)


def structure_errors(U_used):
    obs, rej, counts = reduce_spots(spots, raw, geom, scan, lattice, U_used, profile, dac=dac, pol=pol, mask=mask)
    mm = merge_observations(obs, sg)
    m = RefineModel(start_crystal)
    r = refine(m, mm[["h", "k", "l"]].to_numpy(int), mm.F2.to_numpy(), mm.sigF2.to_numpy())
    err = {}
    for j, (nm, kind) in enumerate(zip(r.names, r.kinds)):
        if kind == "scale":
            continue
        ia = [x.label for x in truth.atoms].index(nm.split(".")[0])
        if kind == "U":
            err[nm] = (r.x[j] - truth.atoms[ia].B_iso / EIGHT_PI2) / r.sigma[j]
        else:
            b = m.basis_frac[j]
            d = r.absolute.iloc[ia][["x", "y", "z"]].to_numpy(float) - np.array(truth.atoms[ia].fract)
            err[nm] = float(d @ b) / float(b @ b) / r.sigma[j]
    return err, r.goof, counts["kept"], r.converged

rows = {"exact U": structure_errors(U0)}
rows[f"U refined from measured spots, start 0.3 deg ({mis_deg(ref_all, U0):.4f} deg off)"] = structure_errors(ref_all)
rows["U 0.3 deg off, NOT refined"] = structure_errors(start(0.3, (1, 1, 1)))
rows["U 0.05 deg off, NOT refined"] = structure_errors(start(0.05, (1, 1, 1)))
names = list(rows["exact U"][0])
lines += ["", "structure-parameter ERROR in units of that run's formal sigma (estimate - truth)/sigma, same spots and noise:",
          "  " + f"{'':62s}" + " ".join(f"{n:>7s}" for n in names) + "   GooF kept conv"]
for k, (err, gf, kept, conv) in rows.items():
    lines.append("  " + f"{k:62s}" + " ".join(f"{err[n]:+7.2f}" for n in names) + f"  {gf:5.2f} {kept:4d} {conv}")
print("\n".join(lines))
Path("results/u_refine_exploratory.txt").write_text("\n".join(lines) + "\n")
