import sys, warnings
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[1])]
import numpy as np
import simulate as sim
from midas_sx.assign import calculated_crossing, qsample_of_hkl, assign_hkl, q_sample_of_spots
from midas_sx.geometry import spot_directions
from midas_sx.predict import predicted_crossings, enumerate_hkl
from midas_hkls import Lattice, SpaceGroup
exp = sim.make_experiment(det_n=512, centre_rc=(255.5, 255.5), omega_start=-38.0, n_frames=304, dac_tilt_deg=(2.0, 1.0), t_plus_mm=2.5, t_minus_mm=3.0, bg0=40.0, bg1=0.0)
st = sim.structure_cu(); res = sim.simulate(st, exp, orientation_seed=3, n_int_max=2e5, target_hits=40_000, max_pairs=8_000_000)
from midas_sx.workflow import ScanSetup
setup = ScanSetup(exp.wavelength_A, exp.dist_mm, exp.pixel_mm, exp.centre_rc, exp.omega_start, exp.omega_step)
geom = setup.geometry((512, 512), 304); scan = setup.scan(304); lat = Lattice(*st.cell); sg = SpaceGroup.from_hm(st.spacegroup)
pc = predicted_crossings(res.U, lat, enumerate_hkl(lat, 0.8, sg), geom, scan)
bad = 0
for r in pc.itertuples():
    s, tth, eta = spot_directions(np.array([r.row]), np.array([r.col]), geom); om = scan.omega_of(np.array([r.frame]))
    hk, hf, resid = assign_hkl(s, om, lat, res.U, geom.wavelength_A, geom.wedge_deg)
    qs = qsample_of_hkl(hk, lat, res.U)[0]
    c = calculated_crossing(qs, float(om[0]), geom.wavelength_A, wedge_deg=geom.wedge_deg)
    if c is None:
        bad += 1
        if bad <= 4:
            print("None for", (r.h, r.k, r.l), "assigned", hk[0], "resid", round(float(resid[0]), 4), "predicted omega", r.omega, "omega_of(frame)", float(om[0]), "diff deg", float(om[0] - r.omega))
print("None:", bad, "of", len(pc))
