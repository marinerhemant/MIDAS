import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, simulate as sim
from midas_hkls import corrections as C
pd.set_option("display.width", 250)
exp = sim.make_experiment(dac_tilt_deg=(2.0, 1.0), t_plus_mm=2.5, t_minus_mm=3.0)
r = sim.simulate(sim.structure_s2_bilayer(), exp, target_hits=50_000, max_pairs=20_000_000)
t = r.truth.copy(); s = t[["sx", "sy", "sz"]].to_numpy()
c = C.rotate_about_axis(exp.dac_axis0(), [0, 0, 1], t.omega.to_numpy())
sgn = np.sign(c[:, 0]); tu = np.where(sgn > 0, exp.t_minus_mm, exp.t_plus_mm); td = np.where(sgn > 0, exp.t_plus_mm, exp.t_minus_mm)
eta = np.degrees(np.arctan2(-t.sy.to_numpy(), t.sz.to_numpy()))
P = C.polarization_factor(t.tth.to_numpy(), eta, pol_fraction=exp.pol_fraction)
T = C.diamond_anvil_transmission(s, c, mu_per_cm=exp.mu_diamond_per_mm*10, t_up_mm=tu, t_down_mm=td, half_opening_deg=35.0)
t["PT_sim"] = t.w_tot_all.to_numpy() * exp.k * np.pi / 180.0 / t.lorentz_emergent.to_numpy()
with np.errstate(all="ignore"):
    t["ratio"] = t.PT_sim.to_numpy() / (P * T)
t["eta"] = eta
t["ang_in"] = np.degrees(np.arccos(np.abs(c[:, 0]))); t["ang_out"] = np.degrees(np.arccos(np.abs(np.sum(c * s, axis=1))))
t["margin"] = 35.0 - np.maximum(t.ang_in, t.ang_out)
t["L"] = C.rotation_lorentz(s, min_abs=0.0)
# omega width of a reflection ~ (shell + mosaic) / slope ~ L * (w_q + |q| 2 sigma_m) in degrees
k = exp.k; w_q = exp.bandwidth_rel * k
t["slope_deg_per_unit_L"] = 1.0 / (t.L * k * np.pi / 180.0)               # deg per (1/A) of Ewald-function change
sel = t[(t.margin.abs() > 1.5) & ~(np.abs(t.ratio - 1) < 1e-3)]
print(sel[["h","k","l","omega","tth","eta","ang_in","ang_out","margin","L","ratio","n_hits"]].to_string())
print("\ntypical spot: median L =", float(t.L.median()), " | for this spot L =", sel.L.to_numpy())
print("approx full omega width of the shell crossing (deg) = w_q * L * 180/(pi k):", (w_q * sel.L.to_numpy() * 180/(np.pi*k)), " | median spot:", float(w_q * t.L.median() * 180/(np.pi*k)))
