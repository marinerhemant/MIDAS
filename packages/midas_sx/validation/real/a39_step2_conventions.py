"""PREREGISTER A39, step 2: resolve the image orientation, omega sense and swing of ONE Pilatus run by indexing with NO cell supplied, then compare with COD 1574527.

    python -I a39_step2_conventions.py <step1 outdir>

Candidates: 8 dihedral x omega sign (2) x swing hypotheses (none, or axis cols / rows x tilt sign x beam-offset sign = 8). Score = spots indexed by `midas_hkls.ab_initio` (tol 0.15). Header geometry only.
"""
import json, os, sys, time
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"), os.path.dirname(os.path.abspath(__file__))]
_root = os.path.dirname(os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx"))
sys.path[:0] = [os.path.join(_root, p) for p in ("midas_hkls", "midas_defect", "midas_stress") if os.path.isdir(os.path.join(_root, p))]
import numpy as np, pandas as pd, torch
from midas_hkls.ab_initio import index_ab_initio
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample
from midas_sx.conventions import candidate_conventions, transform_rc
from a39_geom import swing_geometry

out = sys.argv[1]
N = int(os.environ.get("N_STRONG", 600))
spots = pd.read_csv(os.path.join(out, "spots.csv")).sort_values("integrated", ascending=False).head(N).reset_index(drop=True)
H = json.load(open(os.path.join(out, "info.json")))["header"]
nr, nc = H["shape"]
CELL = sorted((5.8033, 5.95953, 12.2978))


def cell_match(cell):
    a = sorted(cell[:3])
    return max(abs(x - y) / y for x, y in zip(a, CELL))


def score(conv, axis, tzs, offs):
    r, c = transform_rc(spots.row.to_numpy(), spots.col.to_numpy(), nr, nc, conv.dihedral)
    geom, scan = swing_geometry(H, conv, axis, tzs, offs)
    qlab = pixel_to_qlab(r, c, geom, device="cpu", dtype=torch.float64)
    om = torch.as_tensor(np.radians(scan.omega_of(spots.frame.to_numpy())), dtype=torch.float64)
    return index_ab_initio(qlab_to_qsample(qlab, om, 0.0).numpy(), two_pi=True, tol=0.15, min_reflections=30)


print(f"{len(spots)} strongest blobs; swing {H['extra']['detector_2theta_deg']} deg; foot (col,row) {H['extra']['beam_xy_px']}; D {H['distance_mm']} mm; px {H['pixel_mm']} mm", flush=True)
HYP = [("cols", 0, 0)] + [(ax, t, o) for ax in ("cols", "rows") for t in (1, -1) for o in (1, -1)]
rows = []
for conv in candidate_conventions():
    for ax, tzs, offs in HYP:
        t0 = time.time()
        res = score(conv, ax, tzs, offs)
        row = dict(convention=conv.name, axis=ax, tz=tzs, off=offs, n_indexed=int(res.n_indexed) if res.success else 0, n=len(spots), seconds=round(time.time() - t0, 1))
        if res.success:
            row.update(dict(zip("abc", (round(float(v), 3) for v in res.cell[:3]))), alpha=round(float(res.cell[3]), 2), beta=round(float(res.cell[4]), 2),
                       gamma=round(float(res.cell[5]), 2), cell_vs_COD=round(cell_match(res.cell[:3]), 4))
        rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
print(df.sort_values("n_indexed", ascending=False).head(25).to_string(), flush=True)
df.to_csv(os.path.join(out, "conventions.csv"), index=False)
