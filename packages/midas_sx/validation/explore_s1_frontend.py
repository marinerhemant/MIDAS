"""Look at what the real front end finds in simulated S1 frames, vs the simulator's truth table (dev)."""
import sys, time, resource
sys.path.insert(0, "."); sys.path.insert(0, "..")   # simulate.py here, midas_sx one level up (not pip-installed)
import numpy as np, pandas as pd
import simulate as sim
from midas_sx.geometry import Scan, make_geometry
from midas_sx.pipeline import find_spots
pd.set_option("display.width", 250)

exp = sim.make_experiment(dac_tilt_deg=(2.0, 1.0), t_plus_mm=2.5, t_minus_mm=3.0)
r = sim.simulate(sim.structure_cu(), exp, target_hits=100_000, n_int_max=1e5)
scan = Scan(exp.omega_start, exp.omega_step, exp.n_frames)
geom = make_geometry(wavelength_A=exp.wavelength_A, dist_mm=exp.dist_mm, pixel_mm=exp.pixel_mm,
                     centre_rc=exp.centre_rc, shape=(exp.det_n, exp.det_n), scan=scan)
raw = r.frames(seed=11)
print("raw", raw.shape, raw.dtype, f"{raw.nbytes/2**20:.0f} MB; RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MB")
t0 = time.time()
spots, mask, info = find_spots(raw, geom)
print(f"find_spots {time.time()-t0:.1f} s  noise={info['noise']:.2f} threshold={info['threshold']:.1f} masked={info['n_masked']} "
      f"counts={info['blob_counts']}  peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MB")
print("found", len(spots), "spots; truth has", len(r.truth))
# match found to truth by position (row, col, frame), simulator truth omega -> frame index
t = r.truth.copy()
t["frame_idx"] = (t.omega - exp.omega_start) / exp.omega_step - 0.5
d = np.sqrt((spots.row.to_numpy()[:, None] - t.row.to_numpy()[None, :]) ** 2 + (spots.col.to_numpy()[:, None] - t.col.to_numpy()[None, :]) ** 2)
j = d.argmin(1); dmin = d.min(1)
print("nearest-truth distance (px): median %.2f  max %.2f ; matched within 2 px: %d/%d" % (np.median(dmin), dmin.max(), (dmin < 2).sum(), len(spots)))
dfr = spots.frame.to_numpy() - t.frame_idx.to_numpy()[j]
print("frame centroid - truth frame (frames): median %.3f  MAD*1.48 %.3f  max|.| %.2f" % (np.median(dfr[dmin < 2]), 1.4826*np.median(np.abs(dfr[dmin<2]-np.median(dfr[dmin<2]))), np.abs(dfr[dmin<2]).max()))
missing = sorted(set(range(len(t))) - set(j[dmin < 2]))
print("truth spots not found:", len(missing), " their expected_counts:", np.round(np.sort(t.expected_counts.to_numpy()[missing]), 0)[:12])
np.save("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/s1_spots.npy", spots.to_records())
