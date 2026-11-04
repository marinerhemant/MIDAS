import sys; sys.path.insert(0, ".")
import numpy as np, simulate as sim, pandas as pd
from midas_hkls import corrections as C
pd.set_option("display.width", 250)
TILT = (2.0, 1.0) if len(sys.argv) > 1 and sys.argv[1] == 'tilted' else (0.0, 0.0)
TM = 3.0 if TILT != (0.0, 0.0) else 2.5
BW = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-3
exp = sim.make_experiment(dac_tilt_deg=TILT, t_plus_mm=2.5, t_minus_mm=TM, bandwidth_rel=BW)
r = sim.simulate(sim.structure_cu(), exp, target_hits=50_000, max_pairs=20_000_000)
t = r.truth.copy(); s = t[["sx", "sy", "sz"]].to_numpy()
t["PT_sim"] = t.w_tot_all.to_numpy() * exp.k * np.pi / 180.0 / t.lorentz_emergent.to_numpy()
c = C.rotate_about_axis(exp.dac_axis0(), [0, 0, 1], t.omega.to_numpy())
tth = t.tth.to_numpy(); eta = np.degrees(np.arctan2(-t.sy.to_numpy(), t.sz.to_numpy()))
t["eta"] = eta
t["P"] = C.polarization_factor(tth, eta, pol_fraction=exp.pol_fraction)
t["T"] = C.diamond_anvil_transmission(s, c, mu_per_cm=exp.mu_diamond_per_mm*10, t_up_mm=TM, t_down_mm=2.5, half_opening_deg=35.0)
with np.errstate(all="ignore"):
    t["ratio"] = t.PT_sim.to_numpy() / (t["P"].to_numpy() * t["T"].to_numpy())
t["ang_in"] = np.degrees(np.arccos(np.abs(c[:, 0]))); t["ang_out"] = np.degrees(np.arccos(np.abs(np.sum(c*s, axis=1))))
bad = t[~(np.abs(t.ratio - 1) < 1e-4)]
print(bad[["h","k","l","omega","tth","eta","ang_in","ang_out","P","T","PT_sim","ratio","n_hits"]].to_string())

sel = t[(t.h==-1)&(t.k==-1)&(t.l==-1)]
print("bandwidth_rel", BW, " (-1,-1,-1): ratio =", sel.ratio.to_numpy(), " hits:", sel.n_hits.to_numpy())
