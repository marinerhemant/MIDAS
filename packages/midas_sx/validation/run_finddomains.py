"""PREREGISTER A25: domain finding FROM SCRATCH with midas_defect.domains.find_domains on the two-domain frames of A23.

    python run_finddomains.py <seed_lo> <seed_hi> [calib]     # appends results/fd_<lo>.csv (or fd_calib_<lo>.csv)

Frames: Poisson(signal_1 + signal_2 + background), seeds 1000+seed (calib: the same recipe on seeds 100..109, which are never scored).
No orientation is given to the search. The geometry is the A19/A23 wrong-start geometry (0.5 px / 1 %), the cell is the declared one
(a = 3.83, c = 20.5, space group 139); `sigma_rtn` is calibrated on the calibration seeds from the TRUE orientations (see `--calib`).
Scoring only: the truth orientations, folded by the tetragonal proper point group (D4).
"""
import sys, time, json, resource, os
sys.path.insert(0, "."); sys.path.insert(0, "..")
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot

import simulate as sim
from midas_hkls import Lattice
from midas_sx.geometry import Scan, make_geometry
from midas_sx.pipeline import find_spots
from midas_sx.orient import index_and_refine
from midas_sx.assign import assign_hkl, q_sample_of_spots
from midas_sx.geometry import spot_directions
from midas_defect.multiplicity import spots_to_qsample
from midas_defect.domains import find_domains

lo_s, hi_s = int(sys.argv[1]), int(sys.argv[2])
CALIB = len(sys.argv) > 3 and sys.argv[3] == "calib"
CACHE = Path("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache")
st = sim.structure_s2_bilayer()
f1, f2 = CACHE / "S2_bilayer_like_n100000_o1.npz", CACHE / "S2_bilayer_like_n100000_o3.npz"
z1, z2 = np.load(f1), np.load(f2)
exp = sim.Experiment(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(str(f1) + ".exp.json")).items()})
U1, U2 = z1["U"], z2["U"]
SINGLE = os.environ.get("SINGLE", "") == "1"                 # null arm: domain 1 only (find_domains must not invent a second domain)
sig = z1["signal"]
if not SINGLE:
    sig += z2["signal"]
res = sim.SimResult(exp=exp, structure=st, U=U1, signal=sig, truth=pd.read_pickle(str(f1) + ".truth.pkl"), scale=float(z1["scale"]))
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
lattice = Lattice(*st.cell)
A, C = st.cell[0], st.cell[2]
out = Path("results"); csv = out / (f"fd_calib_{lo_s}.jsonl" if CALIB else (f"fd_single_{lo_s}.jsonl" if SINGLE else f"fd_{lo_s}.jsonl"))


def write(row):
    with open(csv, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
SIGMA_RTN = tuple(json.load(open(out / "fd_sigma_rtn.json"))) if (not CALIB and (out / "fd_sigma_rtn.json").exists()) else (0.0071, 0.0145, 0.0094)

# proper rotations of the tetragonal point group 4/mmm (D4)
D4 = [Rot.from_rotvec(np.radians(90.0 * k) * np.array([0, 0, 1.0])).as_matrix() for k in range(4)]
D4 += [Rot.from_rotvec(np.radians(180.0) * v / np.linalg.norm(v)).as_matrix() for v in map(np.array, ([1.0, 0, 0], [0, 1.0, 0], [1.0, 1.0, 0], [1.0, -1.0, 0]))]
fold = lambda Uf, Ut: min(float(np.degrees(np.linalg.norm(Rot.from_matrix(Uf @ S @ Ut.T).as_rotvec()))) for S in D4)
mis = lambda A_, B_: float(np.degrees(np.linalg.norm(Rot.from_matrix(A_ @ B_.T).as_rotvec())))


def start_geometry(seed):
    rng = np.random.default_rng(777 + seed)
    ax = rng.normal(size=3); phi = rng.uniform(0, 2 * np.pi)
    return replace(geom, bcz_px=geom.bcz_px + 0.5 * np.cos(phi), bcy_px=geom.bcy_px + 0.5 * np.sin(phi),
                   lsd_um=geom.lsd_um * (1 + 0.01 * rng.choice([-1.0, 1.0])))


rows = []
for seed in range(lo_s, hi_s):
    t0 = time.time()
    raw = res.frames(seed=1000 + seed)
    g0 = start_geometry(seed)
    spots, mask, info = find_spots(raw, g0)
    q = spots_to_qsample(spots, g0)
    inten = spots["integrated"].to_numpy(float) if "integrated" in spots else np.ones(len(spots))
    row = dict(seed=seed, n_blobs=len(spots), sigma_rtn=json.dumps(SIGMA_RTN))
    if CALIB:
        # residuals of the TRUE orientations in the sample frame: (radial, transverse, normal) in q [1/A, with 2 pi]
        B = np.asarray(lattice.reciprocal_cartesian_vectors(), float).T * 2 * np.pi
        rr = []
        for Ut in (U1, U2):
            h = np.linalg.solve(Ut @ B, q.T).T
            hi = np.rint(h)
            ok = np.abs(h - hi).max(axis=1) < 0.15
            ok &= np.any(hi != 0, axis=1)
            dq = (q[ok] - (Ut @ B @ hi[ok].T).T)
            qn = (Ut @ B @ hi[ok].T).T
            rhat = qn / np.linalg.norm(qn, axis=1, keepdims=True)
            rad = (dq * rhat).sum(1)
            zhat = np.array([0, 0, 1.0])
            tr = np.cross(zhat, rhat); tr /= np.maximum(np.linalg.norm(tr, axis=1, keepdims=True), 1e-9)
            nrm = np.cross(rhat, tr)
            rr.append(np.c_[rad, (dq * tr).sum(1), (dq * nrm).sum(1)])
        rr = np.vstack(rr)
        row.update(n_resid=len(rr), rad=float(rr[:, 0].std()), tr=float(rr[:, 1].std()), nrm=float(rr[:, 2].std()))
        rows.append(row); write(row)
        print(f"calib seed {seed}: n={len(rr)} sigma_rtn ~ ({row['rad']:.4f}, {row['tr']:.4f}, {row['nrm']:.4f})", flush=True)
        continue
    try:
        ds = find_domains(q, inten, spots["row"].to_numpy(float), spots["col"].to_numpy(float), spots["frame"].to_numpy(float), a=A, c=C,
                          space_group_number=139, sigma_rtn=SIGMA_RTN)
    except Exception as e:                                                  # noqa: BLE001
        row.update(error=str(e)[:200], n_found=-1)
        write(row)
        print(f"seed {seed}: find_domains failed: {e}", flush=True); continue
    row.update(n_found=len(ds.domains), seconds_search=time.time() - t0, explained=int(ds.explained.sum()), null_threshold=ds.null_threshold)
    # match found domains to the two truth orientations (both conventions U / U^T are scored; the better one is the one whose fold angle is small)
    for k, dom in enumerate(ds.domains[:4]):
        a_d, a_t = fold(dom.U, U1), fold(dom.U.T, U1)
        b_d, b_t = fold(dom.U, U2), fold(dom.U.T, U2)
        row[f"d{k}_n"] = dom.n; row[f"d{k}_branch"] = dom.branch; row[f"d{k}_a"] = dom.lat.a; row[f"d{k}_b"] = dom.lat.b; row[f"d{k}_c"] = dom.lat.c
        row[f"d{k}_fold1"] = a_d; row[f"d{k}_fold2"] = b_d; row[f"d{k}_fold1_T"] = a_t; row[f"d{k}_fold2_T"] = b_t
    # downstream: refine each found domain with index_and_refine from the found U (geometry from the start geometry, refined on the first)
    try:
        if ds.domains:
            d0 = ds.domains[0]
            U_a = d0.U if min(fold(d0.U, U1), fold(d0.U, U2)) <= min(fold(d0.U.T, U1), fold(d0.U.T, U2)) else d0.U.T
            o1 = index_and_refine(spots, lattice, U_a, g0, scan)
            row.update(o1_err_vs_1=fold(o1.U, U1), o1_err_vs_2=fold(o1.U, U2), o1_converged=o1.converged, o1_n_used=o1.n_used)
            if len(ds.domains) > 1:
                d1 = ds.domains[1]
                U_b = d1.U if min(fold(d1.U, U1), fold(d1.U, U2)) <= min(fold(d1.U.T, U1), fold(d1.U.T, U2)) else d1.U.T
                fr_, ro_, co_ = (spots[c_].to_numpy(float) for c_ in ("frame", "row", "col"))
                s_, _, _ = spot_directions(ro_, co_, o1.geom)
                _, _, r1 = assign_hkl(s_, scan.omega_of(fr_), lattice, o1.U, o1.geom.wavelength_A)
                o2 = index_and_refine(spots[r1 > 0.25].reset_index(drop=True), lattice, U_b, o1.geom, scan, refine_geometry=False)
                row.update(o2_err_vs_1=fold(o2.U, U1), o2_err_vs_2=fold(o2.U, U2), o2_converged=o2.converged, o2_n_used=o2.n_used)
    except Exception as e:                                                  # noqa: BLE001
        row["refine_error"] = str(e)[:200]
    row["seconds"] = time.time() - t0
    write(row)
    print(f"seed {seed}: {len(ds.domains)} domain(s) [{ds}]; fold vs truth: " + "; ".join(
        f"d{k}: {min(row.get(f'd{k}_fold1', 99), row.get(f'd{k}_fold2', 99)):.2f} deg n={row.get(f'd{k}_n')}" for k in range(min(len(ds.domains), 4))) +
        f"; {row['seconds']:.0f} s RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MB", flush=True)
