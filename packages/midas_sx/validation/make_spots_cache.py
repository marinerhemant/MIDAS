"""Cache the blobs of the A25 two-domain frames (seeds lo..hi-1, under the A19 start geometry) so that find_domains sweeps (A28) do not repeat the 60 s spot finding.

    python make_spots_cache.py <lo> <hi>      # writes results/fd_spots/seed<N>.pkl (DataFrame frame,row,col,integrated)
"""
import sys, json
sys.path.insert(0, "."); sys.path.insert(0, "..")
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
import simulate as sim
from midas_sx.geometry import Scan, make_geometry
from midas_sx.pipeline import find_spots

lo_s, hi_s = int(sys.argv[1]), int(sys.argv[2])
CACHE = Path("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache")
st = sim.structure_s2_bilayer()
f1, f2 = CACHE / "S2_bilayer_like_n100000_o1.npz", CACHE / "S2_bilayer_like_n100000_o3.npz"
z1, z2 = np.load(f1), np.load(f2)
exp = sim.Experiment(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in json.load(open(str(f1) + ".exp.json")).items()})
sig = z1["signal"]; sig += z2["signal"]
res = sim.SimResult(exp=exp, structure=st, U=z1["U"], signal=sig, truth=pd.read_pickle(str(f1) + ".truth.pkl"), scale=float(z1["scale"]))
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm, centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
out = Path("results/fd_spots"); out.mkdir(parents=True, exist_ok=True)
for seed in range(lo_s, hi_s):
    if (out / f"seed{seed}.pkl").exists():
        continue
    raw = res.frames(seed=1000 + seed)
    rng = np.random.default_rng(777 + seed)
    ax = rng.normal(size=3); phi = rng.uniform(0, 2 * np.pi)                    # same draw order as run_finddomains.start_geometry
    g0 = replace(geom, bcz_px=geom.bcz_px + 0.5 * np.cos(phi), bcy_px=geom.bcy_px + 0.5 * np.sin(phi), lsd_um=geom.lsd_um * (1 + 0.01 * rng.choice([-1.0, 1.0])))
    spots, mask, info = find_spots(raw, g0)
    spots.to_pickle(out / f"seed{seed}.pkl")
    print(f"seed {seed}: {len(spots)} blobs", flush=True)
