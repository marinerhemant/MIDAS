"""PREREGISTER A38, step 1: one Bruker run -> 3-D spots in RAW array coordinates (no convention applied).

    python -I a38_step1_spots.py <frames dir> <run prefix, e.g. mo_18MPC001_rampT160K_01_> <active-mask frame> <outdir>

Dead pixels (active mask == 0) are set to 0 in every frame so that `find_spots` masks them. The beam centre used for the polar background is the header CENTER
(a few tens of pixels of error from the detector swing change the background profile, not the spots); the convention / geometry scan (step 2) decides the real geometry.
"""
import json, os, sys, time
sys.path[:0] = [os.environ.get("MIDAS_SX_PATH", "/Volumes/T5 EVO/travel/work/opt/MIDAS/packages/midas_sx")]
import numpy as np
from midas_sx.geometry import Scan, make_geometry
from midas_sx.io_frames import read_sfrm_mask, read_sfrm_scan
from midas_sx.pipeline import find_spots

src, prefix, maskf, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
t0 = time.time()
stack, h = read_sfrm_scan(src, prefix=prefix)
dead = read_sfrm_mask(maskf)
print(f"read {stack.shape} {stack.dtype} in {time.time()-t0:.0f} s; axis {h.scan_axis} start {h.angle_start} step {h.angle_step} regular {h.angle_steps_regular}; {dead.mean():.3%} dead pixels; "
      f"stack max {int(stack.max())}, median {float(np.median(stack[::10, ::4, ::4])):.0f}; extra {h.extra['detector_2theta_deg']}, phi {h.extra['phi_deg']}, chi {h.extra['chi_deg']}", flush=True)
raw = stack.astype(np.float32)
raw[:, dead] = 0.0
del stack
nr, nc = h.shape
scan = Scan(h.angle_start, h.angle_step, h.n_frames)
geom = make_geometry(wavelength_A=h.wavelength_A, dist_mm=h.distance_mm, pixel_mm=h.pixel_mm, centre_rc=(h.extra["centre_xy_px"][1], h.extra["centre_xy_px"][0]), shape=(nr, nc), scan=scan)
t1 = time.time()
spots, mask, info = find_spots(raw, geom)
print(f"find_spots: {len(spots)} blobs in {time.time()-t1:.0f} s; info {info}", flush=True)
spots.to_csv(os.path.join(out, "spots.csv"), index=False)
np.save(os.path.join(out, "mask.npy"), mask)
hd = {k: v for k, v in h.__dict__.items() if k not in ("files", "extra")}
hd["extra"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in h.extra.items()}
json.dump(dict(header=hd, n_spots=len(spots), info={k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in info.items()}), open(os.path.join(out, "info.json"), "w"), indent=1, default=str)
print("done", flush=True)
