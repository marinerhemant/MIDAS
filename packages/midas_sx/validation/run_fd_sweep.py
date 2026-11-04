"""PREREGISTER A28: find_domains with a WRONG declared cell, single pass (P1) and with a self-calibrating second pass (P2), on the cached blobs of the A25 frames.

    python run_fd_sweep.py <seed_lo> <seed_hi>      # appends results/fdsweep_<lo>.jsonl (one line per seed x condition)

P1: find_domains with the declared (a, c) = truth x (1 + da, 1 + dc), everything else as A25 (sigma_rtn from results/fd_sigma_rtn.json).
P2: P1; if it returned a domain, rerun with the cell AND nominal_c of the LARGEST domain it found; keep the rerun when it explains more spots.
Scored with A25's rule: the two largest found domains each within 1 deg (D4-folded) of a different truth orientation.
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
f1, f2 = CACHE / "S2_bilayer_like_n100000_o1.npz", CACHE / "S2_bilayer_like_n100000_o3.npz"
U1, U2 = sim.random_orientation(1), np.load(f2)["U"]                # domain 1 = orientation seed 1 (its signal cache is not needed here)
import json as _j
exp = sim.Experiment(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in _j.load(open(str(f2) + ".exp.json")).items()})
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
A, C = st.cell[0], st.cell[2]
SIGMA = tuple(json.load(open("results/fd_sigma_rtn.json")))
D4 = [Rot.from_rotvec(np.radians(90.0 * k) * np.array([0, 0, 1.0])).as_matrix() for k in range(4)]
D4 += [Rot.from_rotvec(np.radians(180.0) * v / np.linalg.norm(v)).as_matrix() for v in map(np.array, ([1.0, 0, 0], [0, 1.0, 0], [1.0, 1.0, 0], [1.0, -1.0, 0]))]
fold = lambda Uf, Ut: min(float(np.degrees(np.linalg.norm(Rot.from_matrix(Uf @ S @ Ut.T).as_rotvec()))) for S in D4)
CONDITIONS = {"iso-5": (-.05, -.05), "iso-3": (-.03, -.03), "iso-1.5": (-.015, -.015), "iso0": (0, 0), "iso+1.5": (.015, .015), "iso+3": (.03, .03), "iso+5": (.05, .05),
              "c-3": (0, -.03), "c+3": (0, .03), "a-3": (-.03, 0), "a+3": (.03, 0)}
CONDITIONS.update({"iso-5": (-.05, -.05)})


def start_geometry(seed):
    rng = np.random.default_rng(777 + seed)
    rng.normal(size=3); phi = rng.uniform(0, 2 * np.pi)
    return replace(geom, bcz_px=geom.bcz_px + 0.5 * np.cos(phi), bcy_px=geom.bcy_px + 0.5 * np.sin(phi), lsd_um=geom.lsd_um * (1 + 0.01 * rng.choice([-1.0, 1.0])))


def run(q, inten, spots, a, c, nominal=None):
    kw = dict(nominal_c=nominal) if nominal else {}
    return find_domains(q, inten, spots["row"].to_numpy(float), spots["col"].to_numpy(float), spots["frame"].to_numpy(float), a=a, c=c, space_group_number=139,
                        sigma_rtn=SIGMA, **kw)


def score(ds):
    out = []
    for d in ds.domains:
        ang = [fold(d.U, U1), fold(d.U, U2), fold(d.U.T, U1), fold(d.U.T, U2)]
        i = int(np.argmin(ang))
        out.append(dict(n=int(d.n), truth=1 + (i % 2), fold=float(ang[i]), a=float(d.lat.a), c=float(d.lat.c), b=float(d.lat.b)))
    out.sort(key=lambda r: -r["n"])
    top = out[:2]
    ok = len(top) == 2 and top[0]["truth"] != top[1]["truth"] and all(t["fold"] <= 1.0 for t in top)
    return dict(n_found=len(out), ok=bool(ok), domains=out[:4], explained=int(ds.explained.sum()))


out = Path("results") / f"fdsweep_{lo_s}.jsonl"
for seed in range(lo_s, hi_s):
    spots = pd.read_pickle(Path("results/fd_spots") / f"seed{seed}.pkl")
    q = spots_to_qsample(spots, start_geometry(seed))
    inten = spots["integrated"].to_numpy(float)
    for name, (da, dc) in CONDITIONS.items():
        t0 = time.time()
        row = dict(seed=seed, cond=name, da=da, dc=dc)
        try:
            ds1 = run(q, inten, spots, A * (1 + da), C * (1 + dc))
            row["P1"] = score(ds1)
            best, cell2 = ds1, None
            if ds1.domains:
                big = max(ds1.domains, key=lambda d: d.n)
                cell2 = (float(big.lat.a), float(big.lat.c))
                ds2 = run(q, inten, spots, cell2[0], cell2[1], nominal=cell2[1])
                if ds2.explained.sum() > ds1.explained.sum():
                    best = ds2
            row["P2"] = score(best); row["P2_recal_cell"] = cell2; row["P2_used_rerun"] = best is not ds1
        except Exception as e:                                              # noqa: BLE001
            row["error"] = str(e)[:200]
        row["seconds"] = time.time() - t0
        with open(out, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
    print(f"seed {seed} done", flush=True)
