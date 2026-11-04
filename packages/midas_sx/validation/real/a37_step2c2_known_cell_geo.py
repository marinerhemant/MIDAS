"""(Variant 2c-bis with the geometry offsets of step 2e and the header-derived convention only.)
PREREGISTER A37 step 2c (DIAGNOSTIC, not a registered tier): is there ANY orientation of the published Rochelle-salt lattice that indexes the spots?

Forced test of the hypothesis the ab-initio cell (step 2b) contradicts. For every convention (16) and for the 100 K and the 308 K COD cells: an orientation grid
(orthorhombic Laue class, 2 deg covering) is scored by the fraction of the strongest spots whose fractional indices are all within `tol` of an integer; the
null is the same score for the winner with the frame index permuted. Reports the best score per case against the null's best.
"""
import json, os, sys, time
CODE = os.environ["CODE"]
sys.path[:0] = [os.path.join(CODE, p) for p in ("midas_sx", "midas_hkls", "midas_defect", "midas_stress")]
import numpy as np, pandas as pd, torch
from midas_stress.orientation_sampling import orientation_grid
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.conventions import candidate_conventions, transform_rc
from midas_sx.geometry import Scan, make_geometry

out = sys.argv[1]
STEP = float(os.environ.get("STEP", 2.0)); TOL = float(os.environ.get("TOL", 0.15)); N = int(os.environ.get("N_STRONG", 1200))
spots = pd.read_csv(os.path.join(out, "spots.csv")).sort_values("integrated", ascending=False).head(N).reset_index(drop=True)
H = json.load(open(os.path.join(out, "info.json")))["header"]
nr, nc = H["shape"]
CENTRE_RAW = (1021.1, 1025.4)
GEO = json.load(open(os.path.join(out, 'geometry_scan.json')))['state']
CELLS = {"100K": (11.7721, 14.2054, 6.1836), "308K": (11.9247, 14.3066, 6.2444)}


def g_of(conv, frames=None):
    r, c = transform_rc(spots.row.to_numpy(), spots.col.to_numpy(), nr, nc, conv.dihedral)
    cr, cc = transform_rc(np.array([CENTRE_RAW[0]]), np.array([CENTRE_RAW[1]]), nr, nc, conv.dihedral)
    shape = (nr, nc) if conv.dihedral < 4 else (nc, nr)
    scan = Scan(conv.omega_sign * H["angle_start"], conv.omega_sign * H["angle_step"], H["n_frames"])
    geom = make_geometry(wavelength_A=H["wavelength_A"], dist_mm=H["distance_mm"] * GEO['dscale'], pixel_mm=H["pixel_mm"], centre_rc=(float(cr[0]) + GEO['drow'], float(cc[0]) + GEO['dcol']), shape=shape, scan=scan)
    qlab = pixel_to_qlab(r, c, geom, device="cpu", dtype=torch.float64)
    fr = spots.frame.to_numpy() if frames is None else frames
    om = torch.as_tensor(np.radians(scan.omega_start + (fr + 0.5) * scan.omega_step), dtype=torch.float64)
    return qlab_to_qsample(qlab, om, 0.0).numpy() / (2 * np.pi)          # 1/d, A^-1


grid = orientation_grid(STEP, 18)
print(f"orientation grid: {len(grid)} rotations at {STEP} deg (Laue mmm)", flush=True)


def best_score(g, cell):
    a, b, c = cell
    Binv = np.diag([a, b, c])                                  # h = diag(a,b,c) U^T g for an orthorhombic cell in the standard setting
    best = (0.0, None)
    for i0 in range(0, len(grid), 20000):
        U = grid[i0:i0 + 20000]                                # (m,3,3): q_sample = U B h  ->  h = B^-1 U^T g
        h = np.einsum("ij,mkj,nk->mni", Binv, U, g)            # (m, n, 3)
        d = np.abs(h - np.rint(h)).max(axis=2)
        frac = (d < TOL).mean(axis=1)
        k = int(frac.argmax())
        if frac[k] > best[0]:
            best = (float(frac[k]), i0 + k)
    return best


rows = []
for conv in [c_ for c_ in candidate_conventions() if c_.dihedral == 7 and c_.omega_sign == 1]:
    g = g_of(conv)
    for name, cell in CELLS.items():
        t0 = time.time()
        s, k = best_score(g, cell)
        rows.append(dict(convention=conv.name, cell=name, best_fraction=round(s, 4), seconds=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
top = df.sort_values("best_fraction", ascending=False).iloc[0]
print("\nbest:", dict(top))
rng = np.random.default_rng(37)
from midas_sx.conventions import Convention
cv = [c for c in candidate_conventions() if c.name == top.convention][0]
nulls = [best_score(g_of(cv, rng.permutation(spots.frame.to_numpy())), CELLS[top.cell])[0] for _ in range(3)]
print("omega-scrambled null best fractions:", [round(x, 4) for x in nulls])
df.to_csv(os.path.join(out, "known_cell_scan_geo.csv"), index=False)
