"""C3 (PREREGISTER A14): how far does a wrong assumption move the refined parameters? Paired, same spots (dev).  python run_c3.py

2026-10-21 (/verify of claim 5017f82f9b6f): the original pairing restricted the VARIANT to the baseline's spots but never the baseline to the variant's, so a perturbation
that loses spots (beam centre +1.0 px loses 21 of 137) compared two fits on different data. Now the baseline is refitted on exactly the spots the variant keeps
(`obs_shared`, `n_base`, `n_var` written), and a non-converged row is marked in the MATTER list. Output: results/c3_misspecification_map_paired.txt (the original
results/c3_misspecification_map.txt is the file the claim was verified against and is not overwritten). Geometry and omega rows are null BY CONSTRUCTION here
(U is held at truth, L, P, T are evaluated at the calculated crossing); the open-loop effect of a wrong start geometry is measured by the A19 runs (run_c2.py v4).
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "..")
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup, corrections as C
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.integrate import build_profile
from midas_sx.merge import merge_observations
from midas_sx.pipeline import find_spots
from midas_sx.reduce import DacModel, PolarizationModel, Switches, reduce_spots
from midas_sx.refine import EIGHT_PI2, RefineModel, perturbed_crystal, refine

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
raw = res.frames(seed=1000)                                         # = C2 seed 0
spots, mask, info = find_spots(raw, geom)
s_all, tth_all, _ = spot_directions(spots.row.to_numpy(), spots.col.to_numpy(), geom)
L_all = C.rotation_lorentz(s_all, min_abs=1e-3)
ok = (tth_all > 2.0) & np.isfinite(L_all)
profile, _ = build_profile(raw, spots[["frame", "row", "col"]].to_numpy(float)[ok], L_all[ok], mask=mask)
dac0 = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm, mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
pol0 = PolarizationModel(exp.pol_fraction)
start = perturbed_crystal(truth, np.random.default_rng(7), shift_A=0.10)


def run(label, *, scan_=scan, geom_=geom, dac=dac0, pol=pol0, sw=Switches(), keep=None, U_=None):
    obs, rej, counts = reduce_spots(spots, raw, geom_, scan_, lattice, res.U if U_ is None else U_, profile, dac=dac, pol=pol, switches=sw, mask=mask)
    key = lambda o: list(zip(o["frame"].round(3), o["row"].round(3), o["col"].round(3)))
    if keep is not None:
        obs = obs[[k in keep for k in key(obs)]].reset_index(drop=True)
    mm = merge_observations(obs, sg)
    hkl = mm[["h", "k", "l"]].to_numpy(int)
    m = RefineModel(start)
    r = refine(m, hkl, mm.F2.to_numpy(), mm.sigF2.to_numpy())
    return r, m, set(key(obs)), counts


base, mbase, keys, cbase = run("baseline")
print(f"baseline: converged={base.converged} GooF={base.goof:.3f} n_unique={len(base.f2_calc)}")
tol = {nm: (0.00127 if nm.endswith(".U") else 0.02) for nm in base.names if nm != "scale"}
pert = lambda **kw: dict(**kw)
from scipy.spatial.transform import Rotation as _R
def _rotU(deg, axis):
    return _R.from_rotvec(np.radians(deg) * np.asarray(axis, float) / np.linalg.norm(axis)).as_matrix() @ res.U
variants = [
    ("anvil thickness +20 %", dict(dac=replace(dac0, t_plus_mm=dac0.t_plus_mm * 1.2, t_minus_mm=dac0.t_minus_mm * 1.2))),
    ("anvil thickness -20 %", dict(dac=replace(dac0, t_plus_mm=dac0.t_plus_mm * 0.8, t_minus_mm=dac0.t_minus_mm * 0.8))),
    ("mu x1.25", dict(dac=replace(dac0, mu_per_cm=dac0.mu_per_cm * 1.25))),
    ("mu x0.80", dict(dac=replace(dac0, mu_per_cm=dac0.mu_per_cm * 0.80))),
    ("DAC axis tilt absent (truth 2,1 deg)", dict(dac=replace(dac0, axis0=(1.0, 0.0, 0.0)))),
    ("DAC axis tilt +2 deg wrong", dict(dac=replace(dac0, axis0=tuple(__import__("scipy.spatial.transform", fromlist=["Rotation"]).Rotation.from_euler("yz", [4.0, 1.0], degrees=True).apply([1, 0, 0]))))),
    ("polarization fraction 0.90 (true 0.99)", dict(pol=PolarizationModel(0.90))),
    ("polarization fraction 0.50", dict(pol=PolarizationModel(0.50))),
    ("polarization plane 10 deg off", dict(pol=PolarizationModel(0.99, 80.0))),
    ("omega zero-point +0.1 deg", dict(scan_=Scan(scan.omega_start + 0.1, scan.omega_step, scan.n_frames))),
    ("omega zero-point +0.3 deg", dict(scan_=Scan(scan.omega_start + 0.3, scan.omega_step, scan.n_frames))),
    ("beam centre +0.5 px (col)", dict(geom_=replace(geom, bcy_px=geom.bcy_px + 0.5))),
    ("beam centre +1.0 px (col)", dict(geom_=replace(geom, bcy_px=geom.bcy_px + 1.0))),
    ("detector distance +0.5 %", dict(geom_=replace(geom, lsd_um=geom.lsd_um * 1.005))),
    ("U misorientation 0.1 deg about (1,1,1)", dict(U_=_rotU(0.1, (1, 1, 1)))),
    ("U misorientation 0.3 deg about (1,1,1)", dict(U_=_rotU(0.3, (1, 1, 1)))),
    ("U misorientation 0.6 deg about (1,1,1)", dict(U_=_rotU(0.6, (1, 1, 1)))),
    ("U misorientation 0.3 deg about the beam (x)", dict(U_=_rotU(0.3, (1, 0, 0)))),
    ("U misorientation 0.3 deg about the rotation axis (z)", dict(U_=_rotU(0.3, (0, 0, 1)))),
    ("NO absorption correction", dict(sw=Switches(True, True, False))),
    ("NO polarization correction", dict(sw=Switches(True, False, True))),
    ("NO Lorentz correction", dict(sw=Switches(False, True, True))),
]
rows = []
base_cache = {}
def base_on(shared):
    k = frozenset(shared)
    if k not in base_cache:
        base_cache[k] = run("baseline", keep=shared)
    return base_cache[k]
for label, kw in variants:
    _, _, kv, _ = run(label, **kw)                                      # the variant on ITS OWN kept spots
    shared = keys & kv
    r, m, k_, c = run(label, keep=shared, **kw)                         # variant and baseline on the SAME spots
    rb, mb, _, _ = base_on(shared)
    row = dict(variant=label, converged=r.converged, GooF=r.goof, n_unique=len(r.f2_calc), n_base=len(keys), n_var=len(kv), obs_shared=len(shared))
    for j, nm in enumerate(base.names):
        if nm == "scale":
            continue
        label_ = nm.split(".")[0]
        ia = [a.label for a in truth.atoms].index(label_)
        if nm.endswith(".U"):
            d = r.x[j] - rb.x[j]
        else:
            b = m.basis_frac[j]
            dfrac = r.absolute.iloc[ia][["x", "y", "z"]].to_numpy(float) - rb.absolute.iloc[ia][["x", "y", "z"]].to_numpy(float)
            d = float(dfrac @ b) / float(b @ b)
        row[nm] = d
        row["m:" + nm] = bool(abs(d) > min(base.sigma[j], tol[nm]))
    rows.append(row)
t = pd.DataFrame(rows)
pcols = [nm for nm in base.names if nm != "scale"]
lines = [f"C3 paired shifts refined(wrong) - refined(correct) on the same spots; baseline formal sigma and tolerance per parameter:",
         "  " + "  ".join(f"{nm}: sig {base.sigma[base.names.index(nm)]:.2g} tol {tol[nm]:.2g}" for nm in pcols), ""]
disp = t[["variant", "converged", "GooF", "n_base", "n_var", "obs_shared"] + pcols].copy()
lines.append(disp.to_string(index=False, float_format=lambda v: f"{v:+.2e}" if abs(v) < 0.1 and v != 0 else f"{v:.3f}"))
lines += ["", "Parameters that MATTER (shift > min(formal sigma, tolerance)):"]
for r in rows:
    mat = [nm for nm in pcols if r["m:" + nm]]
    lines.append(f"  {r['variant']:42s} {', '.join(mat) if mat else '-'}" + ('' if r['converged'] else '   (NOT CONVERGED)'))
print("\n".join(lines))
Path("results/c3_misspecification_map_paired.txt").write_text("\n".join(lines) + "\n")
