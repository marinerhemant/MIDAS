import sys, warnings
from dataclasses import replace
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]
import numpy as np, pandas as pd
import simulate as sim
from midas_hkls import Atom, Crystal, Lattice, SpaceGroup
from midas_sx.reduce import DacModel, PolarizationModel
from midas_sx.refine import EIGHT_PI2
from midas_sx.workflow import ScanSetup, reduce_scan
name = sys.argv[1] if len(sys.argv) > 1 else "cu"
exp = sim.make_experiment(det_n=512, centre_rc=(255.5, 255.5), omega_start=-38.0, n_frames=304, dac_tilt_deg=(2.0, 1.0), t_plus_mm=2.5, t_minus_mm=3.0, bg0=40.0, bg1=0.0)
st = sim.structure_cu() if name == "cu" else sim.structure_s2_bilayer()
res = sim.simulate(st, exp, orientation_seed=3, n_int_max=2e5, target_hits=40_000, max_pairs=8_000_000)
print(name, "rendered crossings", len(res.truth), "F2 range", res.truth.F2.min(), res.truth.F2.max(), "expected_counts range", res.truth.expected_counts.min(), res.truth.expected_counts.max())
cr = Crystal(Lattice(*st.cell), SpaceGroup.from_hm(st.spacegroup), [Atom(s.element, s.fract, s.occ, EIGHT_PI2 * s.u_iso, label=s.label) for s in st.sites], st.name)
dac = DacModel(axis0=tuple(exp.dac_axis0()), t_plus_mm=exp.t_plus_mm, t_minus_mm=exp.t_minus_mm, mu_per_cm=exp.mu_diamond_per_mm * 10.0, half_opening_deg=exp.half_opening_deg)
setup = ScanSetup(exp.wavelength_A, exp.dist_mm, exp.pixel_mm, exp.centre_rc, exp.omega_start, exp.omega_step, dac=dac, pol=PolarizationModel(exp.pol_fraction))
raw = res.frames(seed=5)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rb = reduce_scan(raw, cr, res.U, setup, refine_orientation=False); rp = reduce_scan(raw, cr, res.U, setup, refine_orientation=False, integrate_predicted=True)
print("blob-driven counts", rb.counts); print("predicted counts", rp.counts, rp.info.get("predicted"))
print("predicted rejected reasons:", rp.rejected.reason.value_counts().to_dict() if len(rp.rejected) else None)
k = lambda o: set(zip(o.h, o.k, o.l))
miss = [t for t in k(rb.obs) if t not in k(rp.obs)]
print("hkl in blob-driven obs but not in predicted obs:", miss)
for t in miss[:6]:
    print(t, "blob obs:", rb.obs[(rb.obs.h == t[0]) & (rb.obs.k == t[1]) & (rb.obs.l == t[2])][["frame", "row", "col", "I", "F2"]].round(1).values.tolist(),
          "| predicted rows with this hkl (all targets):", rp.targets[(rp.targets.h_pred == t[0]) & (rp.targets.k_pred == t[1]) & (rp.targets.l_pred == t[2])][["frame", "row", "col", "blob_found"]].round(1).values.tolist(),
          "| rejected:", rp.rejected[(rp.rejected.h == t[0]) & (rp.rejected.k == t[1]) & (rp.rejected.l == t[2])][["frame", "row", "col", "reason"]].round(1).values.tolist() if len(rp.rejected) else None)
