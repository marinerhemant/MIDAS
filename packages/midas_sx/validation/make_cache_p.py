"""PREREGISTER A26: signal caches for the pseudo-merohedral twin S2P (a != b): domain 1 (orientation seed 1) and domain 2 (= domain 1 rotated +90 deg about c).

    python make_cache_p.py <b_over_a>        # writes S2P_immm_b<b>_dom1.npz and ..._dom2.npz (+ .exp.json, .truth.pkl) in the scratch cache

Domain 2's signal is built with scale_override = scale_1 * 3/7 (m = 0.30 as the intensity ratio at the same absolute scale; any other m is a rescaling of the
saved arrays, done by the run script).
"""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "..")
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import simulate as sim

b = float(sys.argv[1])
CACHE = Path("/private/tmp/claude-503/-Volumes-T5-EVO-travel-work-opt-MIDAS/0372d3ae-3aef-4d1a-ba24-2ca52591f359/scratchpad/c2_cache")
st = sim.structure_s2p(b)
exp = sim.make_experiment(dac_tilt_deg=(2.0, 1.0), t_plus_mm=2.5, t_minus_mm=3.0, bandwidth_rel=1e-3, wavelength_A=12.3984 / 30.0)
U1 = sim.random_orientation(1)
U2 = U1 @ R.from_euler("z", 90.0, degrees=True).as_matrix()
t0 = time.time()
r1 = sim.simulate(st, exp, orientation_seed=1, mc_seed=2, n_int_max=1e5, target_hits=100_000, max_pairs=20_000_000)
np.savez(CACHE / f"{st.name}_dom1.npz", signal=r1.signal, U=r1.U, scale=r1.scale)
r1.truth.to_pickle(str(CACHE / f"{st.name}_dom1.npz") + ".truth.pkl")
json.dump(asdict(exp), open(str(CACHE / f"{st.name}_dom1.npz") + ".exp.json", "w"))
print(f"domain 1 written ({time.time()-t0:.0f} s), scale {r1.scale:.6g}", flush=True)
r2 = sim.simulate(st, exp, mc_seed=4, n_int_max=1e5, target_hits=100_000, max_pairs=20_000_000, U_override=U2, scale_override=r1.scale * 3.0 / 7.0)
np.savez(CACHE / f"{st.name}_dom2.npz", signal=r2.signal, U=r2.U, scale=r2.scale)
r2.truth.to_pickle(str(CACHE / f"{st.name}_dom2.npz") + ".truth.pkl")
json.dump(asdict(exp), open(str(CACHE / f"{st.name}_dom2.npz") + ".exp.json", "w"))
print(f"domain 2 written ({time.time()-t0:.0f} s), scale {r2.scale:.6g}", flush=True)
