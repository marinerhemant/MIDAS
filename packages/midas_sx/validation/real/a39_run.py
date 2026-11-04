"""PREREGISTER A39: one Pilatus run -> blind lattice -> setting of the COD cell -> `workflow.reduce_scan`.

    python -I a39_run.py <extracted CBF dir> <run prefix> <cif> <outdir> --conv <dihedral> <omega_sign> --swing <axis> <tz_sign> <off_sign> [--pol-plane 90] [--background mean]

Convention and swing come from step 2 (the winning class). K = 0.99 from the header; the polarization PLANE (angle of the polarization vector from the omega axis in the MIDAS frame, 0 or 90) is
a registered two-way choice made on the merged R_int.
"""
import argparse, json, os, sys, time
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"), os.path.dirname(os.path.abspath(__file__))]
_root = os.path.dirname(os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"))
sys.path[:0] = [os.path.join(_root, p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(_root, p))]
import numpy as np, pandas as pd, torch, gemmi
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.conventions import Convention, transform_frame
from midas_sx.io_frames import read_pilatus_cbf_scan
from midas_sx.pipeline import find_spots
from midas_sx.refine import EIGHT_PI2
from midas_sx.reduce import DacModel, PolarizationModel
from midas_sx.setting import match_setting
from midas_sx.workflow import ScanSetup, reduce_scan
from a39_geom import swing_geometry

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("prefix"); ap.add_argument("cif"); ap.add_argument("out")
ap.add_argument("--conv", nargs=2, type=int, required=True); ap.add_argument("--swing", nargs=3, required=True)
ap.add_argument("--pol-plane", type=float, default=90.0); ap.add_argument("--background", default="mean"); ap.add_argument("--margin", type=int, default=None); ap.add_argument("--predicted", action="store_true"); ap.add_argument("--lambda-scale", type=float, default=1.0); ap.add_argument("--refine-wavelength", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
conv = Convention(a.conv[0], a.conv[1]); axis, tzs, offs = a.swing[0], int(a.swing[1]), int(a.swing[2])

ss = gemmi.read_small_structure(a.cif)
sg = SpaceGroup.from_hm("P 21 21 21")
atoms = [Atom(s.element.name, (s.fract.x, s.fract.y, s.fract.z), s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in ss.sites]
cell = (ss.cell.a, ss.cell.b, ss.cell.c, ss.cell.alpha, ss.cell.beta, ss.cell.gamma)
crystal = Crystal(Lattice(*cell), sg, atoms, "L-alanine100K")

t0 = time.time()
stack, h = read_pilatus_cbf_scan(a.src, prefix=a.prefix, keep_invalid=True)
raw = transform_frame(stack, conv.dihedral).astype(np.float32)
del stack
H = {k: v for k, v in h.__dict__.items() if k not in ("files", "extra")}
H["extra"] = {k: v for k, v in h.extra.items() if k != "dead_mask"}
geom0, scan0 = swing_geometry(H, conv, axis, tzs, offs)
setup = ScanSetup(h.wavelength_A * a.lambda_scale, geom0.lsd_um / 1000.0, h.pixel_mm, (geom0.bcz_px, geom0.bcy_px), scan0.omega_start, scan0.omega_step, dac=DacModel(),
                  pol=PolarizationModel(fraction=h.extra["polarization"], plane_eta_deg=a.pol_plane), tilts_deg=(geom0.tx_deg, geom0.ty_deg, geom0.tz_deg))
print(f"{a.prefix}: {raw.shape} omega {scan0.omega_start}+{scan0.omega_step}n, convention {conv.name}, swing {axis} tz {tzs} off {offs}; Lsd {setup.dist_mm:.3f}; "
      f"centre {np.round(setup.centre_rc, 2)}; pol {h.extra['polarization']} plane {a.pol_plane}; read {time.time()-t0:.0f} s", flush=True)
FK = dict(threshold=3.0, low_count_threshold=0.0)
BLIND_FRAMES = int(os.environ.get("BLIND_FRAMES", 100))
geom = setup.geometry(raw.shape[1:], h.n_frames); scan = setup.scan(h.n_frames)
cache_s, cache_m = os.path.join(a.out, "spots_blind.csv"), os.path.join(a.out, "mask.npy")
if os.path.exists(cache_s) and os.path.exists(cache_m):                    # restart: the 20-minute spot finding is not repeated
    spots, mask, info = pd.read_csv(cache_s), np.load(cache_m), {"reused": True}
    print(f"spots reused from {cache_s}: {len(spots)} blobs", flush=True)
else:
    spots, mask, info = find_spots(raw, geom, **FK)
    spots.to_csv(cache_s, index=False); np.save(cache_m, mask)
st = spots[spots.frame < BLIND_FRAMES].sort_values("integrated", ascending=False).head(600)         # blind lattice from the first 20 deg of the run (see A39 amendment 1)
qlab = pixel_to_qlab(st.row.to_numpy(), st.col.to_numpy(), geom, device="cpu", dtype=torch.float64)
om = torch.as_tensor(np.radians(scan.omega_of(st.frame.to_numpy())), dtype=torch.float64)
g = qlab_to_qsample(qlab, om, 0.0).numpy()
res = index_ab_initio(g, two_pi=True, tol=0.15, min_reflections=30)
print(f"blind lattice: {res}  ({len(spots)} blobs)", flush=True)
if not res.success:
    raise SystemExit("no blind lattice")
U0, P, err = match_setting(np.asarray(res.UB), crystal.lattice)
print(f"setting match: orthogonality error {err:.4f}; found cell {np.round(res.cell, 3)} vs COD {np.round(cell, 3)}", flush=True)
json.dump(dict(found_cell=[float(x) for x in res.cell], n_indexed=int(res.n_indexed), n=int(res.n_reflections), match_err=err, n_blobs=len(spots)), open(os.path.join(a.out, "blind.json"), "w"))
if not info.get("reused"):                                                 # a reused CSV is not rewritten: a read-write round trip changes the last digit of some floats
    spots.to_csv(os.path.join(a.out, "spots_blind.csv"), index=False)
np.save(os.path.join(a.out, "U0.npy"), U0)

red = reduce_scan(raw, crystal, U0, setup, allow_unconverged=True, background=a.background, find_kwargs=FK, refine_tilts=True, refine_wedge=True, found=(spots, mask, info), margin=a.margin, integrate_predicted=a.predicted, refine_wavelength=a.refine_wavelength)
o = red.orientation
print(f"geometry after refinement: tilts (tx,ty,tz) {o.geom.tx_deg:+.3f} {o.geom.ty_deg:+.3f} {o.geom.tz_deg:+.3f} deg (start {geom.tx_deg:+.3f} {geom.ty_deg:+.3f} {geom.tz_deg:+.3f}), wedge {o.geom.wedge_deg:+.3f} deg", flush=True)
print(f"orientation: converged {o.converged} ({o.reason}); n_used {o.n_used}/{o.n_spots}; rms {o.rms_px:.3f} px, {o.rms_frame:.3f} frames; centre {o.geom.bcz_px:.2f},{o.geom.bcy_px:.2f}; "
      f"Lsd {o.geom.lsd_um/1000:.3f} mm; rotation sigma {np.round(o.sigma_rot_deg,4)} deg; wavelength {o.geom.wavelength_A:.5f} A (header {h.wavelength_A:.5f}), "
      f"joint lambda/cell scale eps {o.lambda_cell_scale:+.3e} +- {o.sigma_lambda_cell_scale:.2e} (refined: {a.refine_wavelength})", flush=True)
print(f"counts: {red.counts}; merged within run: {len(red.merged)} unique, R_int {red.r_int:.4f}; profile {red.profile}", flush=True)
red.obs.assign(run=a.prefix).to_pickle(os.path.join(a.out, "obs.pkl")); (red.targets.to_pickle(os.path.join(a.out, "targets.pkl")) if red.targets is not None else None); red.rejected.to_pickle(os.path.join(a.out, "rejected.pkl")); print("predicted:", red.info.get("predicted"), flush=True); np.save(os.path.join(a.out, "U.npy"), red.U)
json.dump(dict(tilts_deg=[o.geom.tx_deg, o.geom.ty_deg, o.geom.tz_deg], wedge_deg=o.geom.wedge_deg, converged=bool(o.converged), reason=o.reason, n_used=int(o.n_used), n_spots=int(o.n_spots), rms_px=float(o.rms_px), rms_frame=float(o.rms_frame), centre_rc=[float(o.geom.bcz_px), float(o.geom.bcy_px)],
               lsd_mm=float(o.geom.lsd_um / 1000), wavelength_A=float(o.geom.wavelength_A), header_wavelength_A=float(h.wavelength_A), lambda_cell_scale=float(o.lambda_cell_scale), sigma_lambda_cell_scale=float(o.sigma_lambda_cell_scale), refine_wavelength=bool(a.refine_wavelength), counts=red.counts, r_int_run=float(red.r_int), n_unique_run=int(len(red.merged)), pol_plane=a.pol_plane, profile=str(red.profile)),
          open(os.path.join(a.out, "reduction.json"), "w"), indent=1, default=str)
print(f"done in {time.time()-t0:.0f} s", flush=True)
