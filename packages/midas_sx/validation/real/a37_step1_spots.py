"""PREREGISTER A37, step 1: read the Rochelle-salt CBF frames and find 3-D spots in RAW array coordinates (no convention applied yet).

    CODE=<dir holding midas_sx/ midas_hkls/ midas_defect/ midas_stress/> python -I a37_step1_spots.py <frames dir or glob> <outdir>

Frames are converted to electrons (counts x the header gain) and treated as Poisson. The beam centre used for the POLAR BACKGROUND only is the header's
first candidate (a few px of error moves nothing there); the convention scan (step 2) decides the real geometry. Writes spots.csv, mask.npy, info.json.
"""
import json
import os
import sys
import time

CODE = os.environ["CODE"]
sys.path[:0] = [os.path.join(CODE, p) for p in ("midas_sx", "midas_hkls", "midas_defect", "midas_stress")]
import numpy as np                                               # noqa: E402

from midas_sx.geometry import Scan, make_geometry               # noqa: E402
from midas_sx.io_frames import read_cbf_scan                    # noqa: E402
from midas_sx.pipeline import find_spots                        # noqa: E402

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
t0 = time.time()
stack, h = read_cbf_scan(src)
print(f"read {stack.shape} {stack.dtype} in {time.time() - t0:.0f} s; header: wl {h.wavelength_A} A, D {h.distance_mm} mm, px {h.pixel_mm} mm, gain {h.gain}, "
      f"phi {h.angle_start} + {h.angle_step} x n, regular {h.angle_steps_regular}", flush=True)
sat = int((stack >= (h.overload or 65535)).sum())
print(f"voxels at/above the overload {h.overload}: {sat}; stack max {int(stack.max())}; median {float(np.median(stack[::20, ::8, ::8])):.1f}", flush=True)
raw = stack.astype(np.float32) * np.float32(h.gain)
del stack
nr, nc = h.shape
scan = Scan(h.angle_start, h.angle_step, h.n_frames)
geom = make_geometry(wavelength_A=h.wavelength_A, dist_mm=h.distance_mm, pixel_mm=h.pixel_mm, centre_rc=(1021.1, 1025.4), shape=(nr, nc), scan=scan)
t1 = time.time()
spots, mask, info = find_spots(raw, geom)
print(f"find_spots: {len(spots)} blobs in {time.time() - t1:.0f} s; info {info}", flush=True)
spots.to_csv(os.path.join(out, "spots.csv"), index=False)
np.save(os.path.join(out, "mask.npy"), mask)
json.dump(dict(header={k: v for k, v in h.__dict__.items() if k != "files"}, overload_voxels=sat, n_spots=len(spots),
               info={k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in info.items()}),
          open(os.path.join(out, "info.json"), "w"), indent=1, default=str)
print("done", flush=True)
