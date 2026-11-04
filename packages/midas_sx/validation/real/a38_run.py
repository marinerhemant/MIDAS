"""PREREGISTER A38: one Bruker run -> blind lattice (no cell supplied) -> setting of the CIF cell -> `workflow.reduce_scan`.

    python -I a38_run.py <frames dir> <run prefix> <active-mask frame> <cif> <outdir> [--background mean|quadratic]

Convention class found by step 2 (thiourea, this instrument): array as the file (identity), omega = +phi/omega as logged, the detector swung by ANGLES[0] about the vertical axis
with the beam D tan(alpha) off the perpendicular foot along +x (cols) and the detector tilted by tz = +alpha; Lsd = D / cos(alpha), D = 38.001 mm (second DISTANC value).
"""
import json, os, sys, time, argparse
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")]
import numpy as np, pandas as pd, torch, gemmi
from dataclasses import replace
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.geometry import spot_directions
from midas_sx.io_frames import read_sfrm_mask, read_sfrm_scan
from midas_sx.pipeline import find_spots
from midas_sx.refine import EIGHT_PI2
from midas_sx.reduce import DacModel, PolarizationModel
from midas_sx.setting import match_setting
from midas_sx.workflow import ScanSetup, reduce_scan

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("prefix"); ap.add_argument("maskf"); ap.add_argument("cif"); ap.add_argument("out")
ap.add_argument("--background", default="mean"); ap.add_argument("--pol", type=float, default=0.5)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

ss = gemmi.read_small_structure(a.cif)
sg = SpaceGroup.from_hm("P 21 m a")
ops_cif = {o.triplet() for o in ss.spacegroup.operations()}
ops_mid = {gemmi.Op(o.triplet()).triplet() if hasattr(o, "triplet") else str(o) for o in sg.operations} if False else None
atoms = [Atom(s.element.name, (s.fract.x, s.fract.y, s.fract.z), s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in ss.sites]
cell = (ss.cell.a, ss.cell.b, ss.cell.c, ss.cell.alpha, ss.cell.beta, ss.cell.gamma)
crystal = Crystal(Lattice(*cell), sg, atoms, "thiourea130K")
json.dump(dict(cif=a.cif, cell=cell, n_atoms=len(atoms), space_group=str(sg)), open(os.path.join(a.out, "crystal.json"), "w"))

t0 = time.time()
stack, h = read_sfrm_scan(a.src, prefix=a.prefix)
dead = read_sfrm_mask(a.maskf)
raw = stack.astype(np.float32); raw[:, dead] = 0.0
del stack
alpha = h.extra["detector_2theta_deg"]
D = h.distance_mm
foot = (h.extra["centre_xy_px"][1], h.extra["centre_xy_px"][0])
shift = D * np.tan(np.radians(alpha)) / h.pixel_mm
setup = ScanSetup(h.wavelength_A, D / np.cos(np.radians(alpha)), h.pixel_mm, (foot[0], foot[1] + shift), h.angle_start, h.angle_step,
                  dac=DacModel(), pol=PolarizationModel(fraction=a.pol, plane_eta_deg=90.0), tilts_deg=(0.0, 0.0, float(alpha)))
print(f"{a.prefix}: {raw.shape}, omega {h.angle_start}+{h.angle_step}n, swing {alpha}, D {D} mm, centre {setup.centre_rc}, Lsd {setup.dist_mm:.3f}; read {time.time()-t0:.0f} s", flush=True)

# blind lattice from this run's own spots
geom0 = setup.geometry(h.shape, h.n_frames)
scan = setup.scan(h.n_frames)
spots, mask, info = find_spots(raw, geom0)
st = spots.sort_values("integrated", ascending=False).head(700)
qlab = pixel_to_qlab(st.row.to_numpy(), st.col.to_numpy(), geom0, device="cpu", dtype=torch.float64)
om = torch.as_tensor(np.radians(scan.omega_of(st.frame.to_numpy())), dtype=torch.float64)
g = qlab_to_qsample(qlab, om, 0.0).numpy()
res = index_ab_initio(g, two_pi=True, tol=0.15, min_reflections=30)
print(f"blind lattice: {res}", flush=True)
if not res.success:
    raise SystemExit("no blind lattice")
U0, P, err = match_setting(np.asarray(res.UB), crystal.lattice)
spots.to_csv(os.path.join(a.out, "spots_blind.csv"), index=False); np.save(os.path.join(a.out, "U0.npy"), U0)
from midas_sx.assign import assign_hkl
_s, _tth, _ = spot_directions(spots.row.to_numpy(), spots.col.to_numpy(), geom0)
_h, _hf, _res = assign_hkl(_s, scan.omega_of(spots.frame.to_numpy()), crystal.lattice, U0, geom0.wavelength_A)
print(f"debug: det(UB) {np.linalg.det(np.asarray(res.UB)):+.3f}, det(P) {np.linalg.det(P):+.0f}, assigned within 0.25 / 0.15 before refinement: "
      f"{(_res < 0.25).mean():.0%} / {(_res < 0.15).mean():.0%} of {len(spots)} blobs", flush=True)
print(f"setting match: orthogonality error {err:.4f}; found cell {np.round(res.cell,3)} vs CIF {np.round(cell,3)}", flush=True)
json.dump(dict(found_cell=[float(x) for x in res.cell], n_indexed=int(res.n_indexed), n=int(res.n_reflections), match_err=err, n_blobs=len(spots)), open(os.path.join(a.out, "blind.json"), "w"))

red = reduce_scan(raw, crystal, U0, setup, allow_unconverged=True, background=a.background)
o = red.orientation
print(f"orientation: converged {o.converged} ({o.reason}); n_used {o.n_used}/{o.n_spots}; rms {o.rms_px:.3f} px, {o.rms_frame:.3f} frames; centre {o.geom.bcz_px:.2f},{o.geom.bcy_px:.2f}; "
      f"Lsd {o.geom.lsd_um/1000:.3f} mm; rotation sigma {np.round(o.sigma_rot_deg,4)} deg", flush=True)
print(f"counts: {red.counts}; merged within run: {len(red.merged)} unique, R_int {red.r_int:.4f}", flush=True)
red.obs.assign(run=a.prefix).to_pickle(os.path.join(a.out, "obs.pkl"))
np.save(os.path.join(a.out, "U.npy"), red.U)
json.dump(dict(converged=bool(o.converged), reason=o.reason, n_used=int(o.n_used), n_spots=int(o.n_spots), rms_px=float(o.rms_px), rms_frame=float(o.rms_frame),
               centre_rc=[float(o.geom.bcz_px), float(o.geom.bcy_px)], lsd_mm=float(o.geom.lsd_um / 1000), counts=red.counts, r_int_run=float(red.r_int),
               n_unique_run=int(len(red.merged))), open(os.path.join(a.out, "reduction.json"), "w"), indent=1, default=str)
print(f"done in {time.time()-t0:.0f} s", flush=True)
