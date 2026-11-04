"""PREREGISTER A31: find_domains with a WRONG declared space group, single pass, on the cached blobs of the A25 frames (seeds 25..49 = held out from A28).

    python run_fd_sg.py <seed_lo> <seed_hi>      # appends results/fdsg_<lo>.jsonl (one line per seed x declared space group)
"""
import sys, time, json
sys.path.insert(0, "."); sys.path.insert(0, "..")
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot

import simulate as sim
from midas_sx.geometry import Scan, make_geometry
from midas_defect.multiplicity import spots_to_qsample
from midas_defect.domains import find_domains

lo_s, hi_s = int(sys.argv[1]), int(sys.argv[2])
CACHE = Path("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache")
st = sim.structure_s2_bilayer()
f2 = CACHE / "S2_bilayer_like_n100000_o1.npz"                  # only its .exp.json is read (the o3 cache is gone; the config is identical)
U1, U2 = sim.random_orientation(1), sim.random_orientation(3)   # domain 2 = orientation seed 3; the SG139 control would fail at once if this were not the A25 orientation
exp = sim.Experiment(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(str(f2) + ".exp.json")).items()})
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
A, C = st.cell[0], st.cell[2]
SIGMA = tuple(json.load(open("results/fd_sigma_rtn.json")))
D4 = [Rot.from_rotvec(np.radians(90.0 * k) * np.array([0, 0, 1.0])).as_matrix() for k in range(4)]
D4 += [Rot.from_rotvec(np.radians(180.0) * v / np.linalg.norm(v)).as_matrix() for v in map(np.array, ([1.0, 0, 0], [0, 1.0, 0], [1.0, 1.0, 0], [1.0, -1.0, 0]))]
fold = lambda Uf, Ut: min(float(np.degrees(np.linalg.norm(Rot.from_matrix(Uf @ S @ Ut.T).as_rotvec()))) for S in D4)
SGS = {"SG139": 139, "SG123": 123, "SG129": 129, "SG1": 1}
if len(sys.argv) > 3 and sys.argv[3] == "ext":                   # A31b: I-centred groups other than 139 (+ the control)
    SGS = {"SG139": 139, "SG87": 87, "SG107": 107, "SG119": 119, "SG140": 140, "SG71": 71}


def start_geometry(seed):
    rng = np.random.default_rng(777 + seed)
    rng.normal(size=3); phi = rng.uniform(0, 2 * np.pi)
    return replace(geom, bcz_px=geom.bcz_px + 0.5 * np.cos(phi), bcy_px=geom.bcy_px + 0.5 * np.sin(phi), lsd_um=geom.lsd_um * (1 + 0.01 * rng.choice([-1.0, 1.0])))


def score(ds):
    out = []
    for d in ds.domains:
        ang = [fold(d.U, U1), fold(d.U, U2), fold(d.U.T, U1), fold(d.U.T, U2)]
        i = int(np.argmin(ang))
        out.append(dict(n=int(d.n), truth=1 + (i % 2), fold=float(ang[i]), a=float(d.lat.a), c=float(d.lat.c)))
    out.sort(key=lambda r: -r["n"])
    top = out[:2]
    ok = len(top) == 2 and top[0]["truth"] != top[1]["truth"] and all(t["fold"] <= 1.0 for t in top)
    return dict(n_found=len(out), ok=bool(ok), domains=out[:4], explained=int(ds.explained.sum()))


out = Path("results") / (f"fdsg_ext_{lo_s}.jsonl" if len(sys.argv) > 3 and sys.argv[3] == "ext" else f"fdsg_{lo_s}.jsonl")
for seed in range(lo_s, hi_s):
    spots = pd.read_pickle(Path("results/fd_spots") / f"seed{seed}.pkl")
    q = spots_to_qsample(spots, start_geometry(seed))
    inten = spots["integrated"].to_numpy(float)
    for name, sgn in SGS.items():
        t0 = time.time()
        row = dict(seed=seed, cond=name, sg=sgn)
        try:
            ds = find_domains(q, inten, spots["row"].to_numpy(float), spots["col"].to_numpy(float), spots["frame"].to_numpy(float), a=A, c=C, space_group_number=sgn, sigma_rtn=SIGMA)
            row.update(score(ds))
        except Exception as e:                                              # noqa: BLE001
            row["error"] = str(e)[:200]
        row["seconds"] = time.time() - t0
        with open(out, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
    print(f"seed {seed} done", flush=True)
