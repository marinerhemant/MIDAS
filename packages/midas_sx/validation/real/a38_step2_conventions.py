"""PREREGISTER A38, step 2: resolve the image orientation, the omega sense and the detector swing of ONE Bruker run by indexing with NO cell supplied, then compare the
cell with the CIF.

    python -I a38_step2_conventions.py <step1 outdir>

Candidates: 8 dihedral transforms x omega sign (2) x swing hypotheses. The swing (detector 2theta = ANGLES[0] about the vertical axis, here the omega axis = lab Z after
the transform) is a rotation of the detector plane about the beam centre by tz = +- alpha, with the beam hitting the plane D tan(alpha) off the perpendicular foot along cols
(foot = header CENTER) and Lsd = D / cos(alpha); the four sign combinations and the no-swing case are scanned. Score = spots indexed by `midas_hkls.ab_initio`.
"""
import json, os, sys, time
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")]
import numpy as np, pandas as pd, torch
from dataclasses import replace
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.conventions import candidate_conventions, rank_conventions, transform_rc
from midas_sx.geometry import Scan, make_geometry

out = sys.argv[1]
N = int(os.environ.get("N_STRONG", 700))
spots = pd.read_csv(os.path.join(out, "spots.csv")).sort_values("integrated", ascending=False).head(N).reset_index(drop=True)
H = json.load(open(os.path.join(out, "info.json")))["header"]
nr, nc = H["shape"]
ex = H["extra"]
alpha = ex["detector_2theta_deg"]
foot_rc = (ex["centre_xy_px"][1], ex["centre_xy_px"][0])                   # (row, col) of the perpendicular foot, array coordinates
CIF = (7.4962, 8.5211, 5.4695)


def geometry(conv, tz_sign, off_sign, scale=1.0):
    """Swing of the detector (2theta = alpha) along the file's x axis (columns of the 768 x 1024 array): the beam hits the plane D tan(alpha) from the perpendicular foot along +-x
    (off_sign); the plane is tilted by alpha about the lab axis that x maps to after the dihedral transform (cols' -> Z: tz, rows' -> Y: ty). Lsd = D / cos(alpha)."""
    D = H["distance_mm"] * scale
    a = np.radians(alpha) if (tz_sign or off_sign) else 0.0
    shift_px = D * np.tan(a) / H["pixel_mm"]
    beam_raw = (foot_rc[0], foot_rc[1] + off_sign * shift_px)                    # (row, col) in the file array: the swing moves the beam along x = col
    r0, c0 = (float(x[0]) for x in transform_rc(np.array([beam_raw[0]]), np.array([beam_raw[1]]), nr, nc, conv.dihedral))
    shape = (nr, nc) if conv.dihedral < 4 else (nc, nr)
    scan = Scan(conv.omega_sign * H["angle_start"], conv.omega_sign * H["angle_step"], H["n_frames"])
    g = make_geometry(wavelength_A=H["wavelength_A"], dist_mm=D / np.cos(a), pixel_mm=H["pixel_mm"], centre_rc=(r0, c0), shape=shape, scan=scan)
    # which lab-frame axis does the file's x (cols) become?  dihedral 0-3: cols' (x stays the horizontal array axis -> tilt about Z); 4-7: rows' (-> tilt about Y)
    kw = dict(tz_deg=float(tz_sign * alpha)) if conv.dihedral < 4 else dict(ty_deg=float(tz_sign * alpha))
    return replace(g, **kw), scan


def score(conv, tz_sign, off_sign, scale=1.0):
    r, c = transform_rc(spots.row.to_numpy(), spots.col.to_numpy(), nr, nc, conv.dihedral)
    geom, scan = geometry(conv, tz_sign, off_sign, scale)
    qlab = pixel_to_qlab(r, c, geom, device="cpu", dtype=torch.float64)
    om = torch.as_tensor(np.radians(scan.omega_start + (spots.frame.to_numpy() + 0.5) * scan.omega_step), dtype=torch.float64)
    g = qlab_to_qsample(qlab, om, 0.0).numpy()
    res = index_ab_initio(g, two_pi=True, tol=0.15, min_reflections=30)
    return res


def cell_match(cell):
    a = sorted(cell[:3]); b = sorted(CIF)
    return max(abs(x - y) / y for x, y in zip(a, b))


print(f"{len(spots)} strongest of the run's blobs; swing alpha {alpha} deg; foot (row, col) {foot_rc}; D {H['distance_mm']} mm; px {H['pixel_mm']:.4f} mm", flush=True)
HYP = [(0, 0), (+1, +1), (-1, -1), (+1, -1), (-1, +1)]
rows = []
for conv in candidate_conventions():
    for tzs, offs in HYP:
        t0 = time.time()
        res = score(conv, tzs, offs)
        row = dict(convention=conv.name, tz=tzs, off=offs, n_indexed=int(res.n_indexed) if res.success else 0, n=len(spots), seconds=round(time.time() - t0, 1))
        if res.success:
            row.update(dict(zip("abc", (round(float(v), 3) for v in res.cell[:3]))), alpha=round(float(res.cell[3]), 2), beta=round(float(res.cell[4]), 2),
                       gamma=round(float(res.cell[5]), 2), cell_vs_CIF=round(cell_match(res.cell[:3]), 4))
        rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
print(df.sort_values("n_indexed", ascending=False).head(25).to_string(), flush=True)
df.to_csv(os.path.join(out, "conventions.csv"), index=False)
