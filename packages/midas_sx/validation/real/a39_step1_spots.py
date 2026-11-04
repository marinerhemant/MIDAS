"""PREREGISTER A39, step 1: one Pilatus run (Diamond I19-1) -> 3-D spots in RAW array coordinates.

    python -I a39_step1_spots.py <dir of extracted CBF> <run prefix, e.g. pgw240050_02_> <outdir>

The -1 / -2 pixel flags stay in the stack (build_mask reads them); the background of this detector is ~0 counts, so the 'low count' mask is off (threshold 0) and the blob threshold is
an explicit 3 counts above the polar-median background (the MAD-based default collapses to 0 on a zero background; P(X >= 3 | 0.05 counts/voxel) = 2e-5 per voxel and a blob needs >= 10
connected voxels). The beam centre for the polar background is the perpendicular foot of the file (Beam_xy); the real geometry is resolved in step 2.
"""
import json, os, sys, time
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")]
import numpy as np
from midas_sx.geometry import Scan, make_geometry
from midas_sx.io_frames import read_pilatus_cbf_scan
from midas_sx.pipeline import find_spots

src, prefix, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
t0 = time.time()
stack, h = read_pilatus_cbf_scan(src, prefix=prefix, keep_invalid=True)
print(f"read {stack.shape} {stack.dtype} in {time.time()-t0:.0f} s; axis {h.scan_axis} start {h.angle_start} step {h.angle_step} regular {h.angle_steps_regular}; "
      f"dead pixels {h.extra['dead_mask'].mean():.2%}; max {int(stack.max())}; phi {h.extra['phi_deg']} chi {h.extra['chi_deg']} swing {h.extra['detector_2theta_deg']}", flush=True)
raw = stack.astype(np.float32)
del stack
nr, nc = h.shape
scan = Scan(h.angle_start, h.angle_step, h.n_frames)
bx, by = h.extra["beam_xy_px"]
geom = make_geometry(wavelength_A=h.wavelength_A, dist_mm=h.distance_mm, pixel_mm=h.pixel_mm, centre_rc=(by, bx), shape=(nr, nc), scan=scan)
t1 = time.time()
spots, mask, info = find_spots(raw, geom, threshold=3.0, low_count_threshold=0.0)
print(f"find_spots: {len(spots)} blobs in {time.time()-t1:.0f} s; info {info}", flush=True)
spots.to_csv(os.path.join(out, "spots.csv"), index=False)
np.save(os.path.join(out, "mask.npy"), mask)
hd = {k: v for k, v in h.__dict__.items() if k not in ("files", "extra")}
hd["extra"] = {k: (v.tolist() if hasattr(v, "tolist") and k not in ("dead_mask",) else v) for k, v in h.extra.items() if k != "dead_mask"}
json.dump(dict(header=hd, n_spots=len(spots), dead_fraction=float(h.extra["dead_mask"].mean()), info={k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in info.items()}),
          open(os.path.join(out, "info.json"), "w"), indent=1, default=str)
print("done", flush=True)
