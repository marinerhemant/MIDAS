"""PREREGISTER A37 step 2d (tier A path): refine orientation + beam centre + distance on the ab-initio lattice of step 2b with midas_sx.orient.index_and_refine, report
the indexed fraction of the strong spots before and after, and look for a SECOND lattice in the spots the first one does not own."""
import json, os, sys
CODE = os.environ["CODE"]
sys.path[:0] = [os.path.join(CODE, p) for p in ("midas_sx", "midas_hkls", "midas_defect", "midas_stress")]
import numpy as np, pandas as pd, torch
from dataclasses import replace
from midas_hkls import Lattice
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.assign import reciprocal_basis, assign_hkl
from midas_sx.conventions import Convention, transform_rc
from midas_sx.geometry import Scan, make_geometry, spot_directions
from midas_sx.orient import index_and_refine

out = sys.argv[1]
spots = pd.read_csv(os.path.join(out, "spots.csv"))
H = json.load(open(os.path.join(out, "info.json")))["header"]
nr, nc = H["shape"]
GEO = json.load(open(os.path.join(out, 'geometry_scan.json')))['state']       # step 2e offsets
conv = Convention(7, +1)                                                      # the header-derived convention (see step 2e)
r, c = transform_rc(spots.row.to_numpy(), spots.col.to_numpy(), nr, nc, conv.dihedral)
cr, cc = transform_rc(np.array([1021.1]), np.array([1025.4]), nr, nc, conv.dihedral)
shape = (nr, nc) if conv.dihedral < 4 else (nc, nr)
spots = spots.assign(row=r, col=c)
scan = Scan(conv.omega_sign * H["angle_start"], conv.omega_sign * H["angle_step"], H["n_frames"])
geom0 = make_geometry(wavelength_A=H["wavelength_A"], dist_mm=H["distance_mm"] * GEO['dscale'], pixel_mm=H["pixel_mm"], centre_rc=(float(cr[0]) + GEO['drow'], float(cc[0]) + GEO['dcol']), shape=shape, scan=scan)


def gvec(sp, geom):
    qlab = pixel_to_qlab(sp.row.to_numpy(), sp.col.to_numpy(), geom, device="cpu", dtype=torch.float64)
    om = torch.as_tensor(np.radians(scan.omega_start + (sp.frame.to_numpy() + 0.5) * scan.omega_step), dtype=torch.float64)
    return qlab_to_qsample(qlab, om, 0.0).numpy()


def owned(lattice, U, geom, sp, tol):
    s, tth, _ = spot_directions(sp.row.to_numpy(), sp.col.to_numpy(), geom)
    hkl, hf, resid = assign_hkl(s, scan.omega_of(sp.frame.to_numpy()), lattice, U, geom.wavelength_A)
    return (resid < tol) & (tth > 2.0) & np.any(hkl != 0, axis=1), resid


strong = spots.sort_values("integrated", ascending=False).head(2500).reset_index(drop=True)
res = index_ab_initio(gvec(strong, geom0), two_pi=True, tol=0.10, min_reflections=40)
print(res, flush=True)
lat = Lattice(*res.cell)
B = reciprocal_basis(lat)
for name, M in (("UB", np.asarray(res.UB)), ("UB.T", np.asarray(res.UB).T)):
    U0 = M @ np.linalg.inv(B)
    print(name, "U0 orthogonality error", np.abs(U0 @ U0.T - np.eye(3)).max(), "det", np.linalg.det(U0), flush=True)
U0 = np.asarray(res.UB) @ np.linalg.inv(B)
if np.abs(U0 @ U0.T - np.eye(3)).max() > 1e-2:
    U0 = np.asarray(res.UB).T @ np.linalg.inv(B)
u, _, vt = np.linalg.svd(U0); U0 = u @ vt
m0, _ = owned(lat, U0, geom0, strong, 0.15)
print(f"before refinement: {m0.sum()}/{len(strong)} strong spots within 0.15 of the lattice", flush=True)
o = index_and_refine(spots, lat, U0, geom0, scan)
print(f"index_and_refine: converged {o.converged} ({o.reason}); n_used {o.n_used}/{o.n_spots}; rms {o.rms_px:.3f} px, {o.rms_frame:.3f} frames; "
      f"centre (row,col) {o.geom.bcz_px:.2f},{o.geom.bcy_px:.2f} (start {geom0.bcz_px:.2f},{geom0.bcy_px:.2f}); distance {o.geom.lsd_um/1000:.3f} mm (header {geom0.lsd_um/1000:.3f})", flush=True)
for tol in (0.10, 0.15, 0.25):
    m, _ = owned(lat, o.U, o.geom, strong, tol)
    ma, _ = owned(lat, o.U, o.geom, spots, tol)
    print(f"after refinement, tol {tol}: strong {m.sum()}/{len(strong)} ({m.mean():.0%}); all blobs {ma.sum()}/{len(spots)} ({ma.mean():.0%})", flush=True)
ma, _ = owned(lat, o.U, o.geom, spots, 0.15)
rest = spots[~ma].sort_values("integrated", ascending=False).head(2500).reset_index(drop=True)
r2 = index_ab_initio(gvec(rest, o.geom), two_pi=True, tol=0.15, min_reflections=40)
print("\nsecond lattice among the spots the first does not own:", r2, flush=True)
json.dump(dict(cell=[float(x) for x in res.cell], U=o.U.tolist(), converged=bool(o.converged), reason=o.reason, rms_px=float(o.rms_px), rms_frame=float(o.rms_frame),
               lsd_mm=float(o.geom.lsd_um / 1000), centre_rc=[float(o.geom.bcz_px), float(o.geom.bcy_px)], n_used=int(o.n_used), n_total=int(o.n_spots)),
          open(os.path.join(out, "orientation_found_geo.json"), "w"), indent=1)
