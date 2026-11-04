"""PREREGISTER A38, step 2b (DIAGNOSTIC of the step-2 failure; the vendor's reflection list used as a geometry TEST, not as input to the reduction): the SAINT p4p lists
`REF1K` observed reflections with hkl, omega, phi, chi and pixel (x, y). For run 01 (phi 299.978, chi -40.319) fit every candidate geometry (8 dihedral x omega sign x swing
hypothesis x distance scale) by Kabsch: q_sample from my geometry for the observed pixel and omega, against B h of the CIF cell; the right geometry gives a small rms
in 1/d. Prints the best candidates. Conventions found here are then checked by BLIND indexing (step 2, ab initio)."""
import json, os, re, sys
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")]
import numpy as np, pandas as pd, torch
from dataclasses import replace
from midas_hkls import Lattice
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.assign import reciprocal_basis
from midas_sx.conventions import candidate_conventions, transform_rc
from midas_sx.geometry import Scan, make_geometry

p4p, out = sys.argv[1], sys.argv[2]
H = json.load(open(os.path.join(out, "info.json")))["header"]
nr, nc = H["shape"]; ex = H["extra"]
alpha = ex["detector_2theta_deg"]
foot = (ex["centre_xy_px"][1], ex["centre_xy_px"][0])
rows = []
for line in open(p4p):
    if line.startswith("REF1K"):
        t = line.split()
        # REF1K <elem> h k l 2th omega phi chi x y I? ...   (the element column can be glued to nothing: parse numerically from the right of the name)
        nums = [float(x) for x in t[2:]]
        rows.append(nums)
ref = pd.DataFrame(rows).iloc[:, :9]
ref.columns = ["h", "k", "l", "tth", "omega", "phi", "chi", "x", "y"]
sel = ref[(ref.phi.sub(ex["phi_deg"]).abs() < 0.01) & (ref.chi.sub(ex["chi_deg"] - 360).abs() < 0.01)].reset_index(drop=True)
print(f"{len(ref)} REF1K lines, {len(sel)} belong to run 01 (phi {ex['phi_deg']}, chi {ex['chi_deg']-360:.3f}); omega range {sel.omega.min():.1f}-{sel.omega.max():.1f}", flush=True)
lat = Lattice(7.4962, 8.5211, 5.4695, 90, 90, 90)
Bm = reciprocal_basis(lat)                                    # columns: reciprocal vectors in 1/d (no 2 pi)
hkl = sel[["h", "k", "l"]].to_numpy(float)
qB = (Bm @ hkl.T).T


def evaluate(conv, tz_sign, off_sign, scale):
    r, c = transform_rc(sel.y.to_numpy(), sel.x.to_numpy(), nr, nc, conv.dihedral)          # Bruker (x, y) = (col, row) of the file array
    r0, c0 = (float(v[0]) for v in transform_rc(np.array([foot[0]]), np.array([foot[1]]), nr, nc, conv.dihedral))
    shape = (nr, nc) if conv.dihedral < 4 else (nc, nr)
    scan = Scan(conv.omega_sign * H["angle_start"], conv.omega_sign * H["angle_step"], H["n_frames"])
    D = H["distance_mm"] * scale
    a = np.radians(alpha) if (tz_sign or off_sign) else 0.0
    g = make_geometry(wavelength_A=H["wavelength_A"], dist_mm=D / np.cos(a), pixel_mm=H["pixel_mm"], centre_rc=(r0, c0 + off_sign * D * np.tan(a) / H["pixel_mm"]), shape=shape, scan=scan)
    g = replace(g, tz_deg=float(tz_sign * alpha))
    qlab = pixel_to_qlab(r, c, g, device="cpu", dtype=torch.float64)
    om = torch.as_tensor(np.radians(conv.omega_sign * sel.omega.to_numpy()), dtype=torch.float64)       # REF1K omega is the reflection's centre angle
    q = qlab_to_qsample(qlab, om, 0.0).numpy() / (2 * np.pi)
    # Kabsch: U q_B ~ q   (proper rotation)
    Hm = qB.T @ q
    u, s, vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    U = vt.T @ np.diag([1, 1, d]) @ u.T
    res = q - (U @ qB.T).T
    return float(np.sqrt((res ** 2).sum(1).mean())), float(d)


HYP = [(0, 0), (+1, +1), (-1, -1), (+1, -1), (-1, +1)]
out_rows = []
for conv in candidate_conventions():
    for tzs, offs in HYP:
        for sc in (0.6, 0.8, 1.0, 1.2, 1.36, 1.5, 2.0):
            rms, det = evaluate(conv, tzs, offs, sc)
            out_rows.append(dict(convention=conv.name, tz=tzs, off=offs, scale=sc, rms_1_over_d=rms, handed=det))
df = pd.DataFrame(out_rows).sort_values("rms_1_over_d")
pd.set_option("display.width", 200)
print(df.head(20).to_string(index=False), flush=True)
df.to_csv(os.path.join(out, "refpoints_scan.csv"), index=False)
