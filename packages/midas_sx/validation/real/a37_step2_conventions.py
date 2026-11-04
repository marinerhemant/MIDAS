"""PREREGISTER A37, step 2: resolve the image orientation and the sense of rotation by indexing, with the cell NOT supplied, plus the omega-scramble null.

    CODE=<code dir> python -I a37_step2_conventions.py <step1 outdir> <header json from step1 is read from it>

For each of the 16 candidate conventions: raw (row, col) -> MIDAS array coordinates, geometry with the header distance and the header beam-centre candidate
transformed with the same map, g-vectors in the sample frame, ``midas_hkls.ab_initio.index_ab_initio`` on the strongest spots. Score = number of spots
indexed. The pick must beat the runner-up by >= 20 % (registered); otherwise the result is "conventions not resolved". Null: the frame index of the winner
is randomly permuted (omega scrambled) and indexed the same way. Writes conventions.csv and prints the table.
"""
import json
import os
import sys
import time

CODE = os.environ["CODE"]
sys.path[:0] = [os.path.join(CODE, p) for p in ("midas_sx", "midas_hkls", "midas_defect", "midas_stress")]
import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
import torch                                                     # noqa: E402

from midas_hkls.ab_initio import index_ab_initio                 # noqa: E402
from midas_defect.geometry import pixel_to_qlab, qlab_to_qsample  # noqa: E402
from midas_sx.conventions import candidate_conventions, rank_conventions, transform_rc   # noqa: E402
from midas_sx.geometry import Scan, make_geometry               # noqa: E402

out = sys.argv[1]
N_STRONG = int(os.environ.get("N_STRONG", 2500))
spots = pd.read_csv(os.path.join(out, "spots.csv"))
H = json.load(open(os.path.join(out, "info.json")))["header"]
nr, nc = H["shape"]
strong = spots.sort_values("integrated", ascending=False).head(N_STRONG).reset_index(drop=True)
print(f"{len(spots)} blobs; using the {len(strong)} strongest", flush=True)
CENTRE_RAW = (1021.1, 1025.4)                                      # header imgCIF offset (105.0, -104.562) mm / 0.1024 mm; sign / axis assignment unverified


def g_vectors(conv, frames=None):
    r, c = transform_rc(strong.row.to_numpy(), strong.col.to_numpy(), nr, nc, conv.dihedral)
    cr, cc = transform_rc(np.array([CENTRE_RAW[0]]), np.array([CENTRE_RAW[1]]), nr, nc, conv.dihedral)
    shape = (nr, nc) if conv.dihedral < 4 else (nc, nr)
    scan = Scan(conv.omega_sign * H["angle_start"], conv.omega_sign * H["angle_step"], H["n_frames"])
    geom = make_geometry(wavelength_A=H["wavelength_A"], dist_mm=H["distance_mm"], pixel_mm=H["pixel_mm"], centre_rc=(float(cr[0]), float(cc[0])), shape=shape, scan=scan)
    fr = strong.frame.to_numpy() if frames is None else frames
    qlab = pixel_to_qlab(r, c, geom, device="cpu", dtype=torch.float64)
    om = torch.as_tensor(np.radians(scan.omega_start + (fr + 0.5) * scan.omega_step), dtype=torch.float64)
    return qlab_to_qsample(qlab, om, 0.0).numpy()


def run(conv, frames=None, tag=""):
    t0 = time.time()
    g = g_vectors(conv, frames)
    res = index_ab_initio(g, two_pi=True, tol=0.15, min_reflections=40)
    row = dict(convention=conv.name, tag=tag, success=bool(res.success), n_indexed=int(res.n_indexed), n_spots=len(g),
               fraction=float(res.indexed_fraction), seconds=round(time.time() - t0, 1))
    if res.success:
        row.update(dict(zip(("a", "b", "c", "alpha", "beta", "gamma"), (round(float(v), 4) for v in res.cell))))
        row["volume"] = round(float(np.linalg.det(np.linalg.inv(res.UBI)) if res.UBI is not None else np.nan), 2)
    row["notes"] = " | ".join(res.notes)[:300]
    return row


rows = [run(c) for c in candidate_conventions()]
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(df.drop(columns=["notes"]).to_string(), flush=True)
cs = candidate_conventions()
best, s1, resolved = rank_conventions([(c, r["n_indexed"] if r["success"] else 0.0) for c, r in zip(cs, rows)])
print(f"\nbest convention: {best.name}; runner-up n_indexed {s1}; resolved (>= 20 % margin): {resolved}", flush=True)
rng = np.random.default_rng(37)
null_rows = [run(best, frames=rng.permutation(strong.frame.to_numpy()), tag=f"scramble{i}") for i in range(3)]
print("omega-scrambled null (winner convention):")
print(pd.DataFrame(null_rows).drop(columns=["notes"]).to_string(), flush=True)
pd.concat([df, pd.DataFrame(null_rows)], ignore_index=True).to_csv(os.path.join(out, "conventions.csv"), index=False)
json.dump(dict(best=best.name, dihedral=best.dihedral, omega_sign=best.omega_sign, runner_up_n=s1, resolved=resolved), open(os.path.join(out, "convention_pick.json"), "w"))
