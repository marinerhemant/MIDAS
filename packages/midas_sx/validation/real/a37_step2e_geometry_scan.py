"""PREREGISTER A37 step 2e (DIAGNOSTIC): is the poor indexing (47 % at tol 0.15, 4.5 px rms after refinement) a geometry error? Coordinate scan of the beam centre and the
distance, scoring each by the number of spots the ab-initio lattice (no cell supplied) owns at tol 0.10.

Convention fixed by the HEADER, derived before looking at any scan result: in the CBF's own imgCIF block the beam travels along -Z, the phi axis is +X, the
fast array axis points along -X and the slow axis along +Y, so in the MIDAS frame (beam +X, rotation axis +Z, rows -> +Z, cols -> -Y) the array is the ANTI-TRANSPOSE
of the file and omega = +phi; step 2 found that convention (in a four-way tie with its mirror images) to index best. Header centre (row 1021.1, col 1025.4) in the file.
"""
import json, os, sys, time
CODE = os.environ["CODE"]
sys.path[:0] = [os.path.join(CODE, p) for p in ("midas_sx", "midas_hkls", "midas_defect", "midas_stress")]
import numpy as np, pandas as pd, torch
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.conventions import Convention, transform_rc
from midas_sx.geometry import Scan, make_geometry

out = sys.argv[1]
N = int(os.environ.get("N_STRONG", 1500))
spots = pd.read_csv(os.path.join(out, "spots.csv")).sort_values("integrated", ascending=False).head(N).reset_index(drop=True)
H = json.load(open(os.path.join(out, "info.json")))["header"]
nr, nc = H["shape"]
conv = Convention(7, +1)
r, c = transform_rc(spots.row.to_numpy(), spots.col.to_numpy(), nr, nc, conv.dihedral)
cr0, cc0 = (float(x[0]) for x in transform_rc(np.array([1021.1]), np.array([1025.4]), nr, nc, conv.dihedral))
scan = Scan(H["angle_start"], H["angle_step"], H["n_frames"])


def score(drow, dcol, dscale):
    geom = make_geometry(wavelength_A=H["wavelength_A"], dist_mm=H["distance_mm"] * dscale, pixel_mm=H["pixel_mm"], centre_rc=(cr0 + drow, cc0 + dcol), shape=(nc, nr), scan=scan)
    qlab = pixel_to_qlab(r, c, geom, device="cpu", dtype=torch.float64)
    om = torch.as_tensor(np.radians(scan.omega_start + (spots.frame.to_numpy() + 0.5) * scan.omega_step), dtype=torch.float64)
    g = qlab_to_qsample(qlab, om, 0.0).numpy()
    res = index_ab_initio(g, two_pi=True, tol=0.10, min_reflections=40)
    return (int(res.n_indexed) if res.success else 0), res


state = dict(drow=0.0, dcol=0.0, dscale=1.0)
best, _ = score(**state)
print(f"start (header geometry): n_indexed {best}/{len(spots)}", flush=True)
steps = dict(drow=(np.arange(-12, 12.1, 3.0)), dcol=(np.arange(-12, 12.1, 3.0)), dscale=(1 + np.arange(-0.03, 0.031, 0.01)))
for sweep in range(2):
    for key in ("drow", "dcol", "dscale"):
        scale = 1.0 if sweep == 0 else (0.35 if key != "dscale" else 0.35)
        cand = state[key] + (steps[key] - (0 if key != "dscale" else 1.0)) * scale if key != "dscale" else state[key] * (1 + (steps[key] - 1.0) * scale)
        vals = []
        for v in cand:
            s = dict(state); s[key] = float(v)
            n, _ = score(**s)
            vals.append((n, float(v)))
        n_best, v_best = max(vals)
        print(f"sweep {sweep} {key}: " + " ".join(f"{v:+.3f}:{n}" for n, v in vals) + f"  -> {v_best:+.3f} ({n_best})", flush=True)
        if n_best >= best:
            best, state[key] = n_best, v_best
n, res = score(**state)
print(f"\nbest geometry offsets {state}  n_indexed {n}/{len(spots)}\n{res}", flush=True)
json.dump(dict(state=state, n_indexed=n, n=len(spots), cell=[float(x) for x in res.cell] if res.success else None), open(os.path.join(out, "geometry_scan.json"), "w"), indent=1)
