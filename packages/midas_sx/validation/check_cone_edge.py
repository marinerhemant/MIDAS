"""Classify every spot where simulator P*T and midas_hkls.corrections P*T differ (dev).
Run: python check_cone_edge.py [S1|S2] [untilted|tilted]"""
import sys; sys.path.insert(0, ".")
import numpy as np, simulate as sim
from midas_hkls import corrections as C

which = sys.argv[1] if len(sys.argv) > 1 else "S1"
tilt, tm = ((2.0, 1.0), 3.0) if (len(sys.argv) > 2 and sys.argv[2] == "tilted") else ((0.0, 0.0), 2.5)
st = sim.structure_cu() if which == "S1" else sim.structure_s2_bilayer()
exp = sim.make_experiment(dac_tilt_deg=tilt, t_plus_mm=2.5, t_minus_mm=tm)
r = sim.simulate(st, exp, target_hits=50_000, max_pairs=20_000_000)
t = r.truth; s = t[["sx", "sy", "sz"]].to_numpy()
PT_sim = t.w_tot_all.to_numpy() * exp.k * np.pi / 180.0 / t.lorentz_emergent.to_numpy()
c = C.rotate_about_axis(exp.dac_axis0(), [0, 0, 1], t.omega.to_numpy())
sgn = np.sign(c[:, 0]); t_up = np.where(sgn > 0, exp.t_minus_mm, exp.t_plus_mm); t_dn = np.where(sgn > 0, exp.t_plus_mm, exp.t_minus_mm)
tth = t.tth.to_numpy(); eta = np.degrees(np.arctan2(-t.sy.to_numpy(), t.sz.to_numpy()))
P = C.polarization_factor(tth, eta, pol_fraction=exp.pol_fraction)
T = C.diamond_anvil_transmission(s, c, mu_per_cm=exp.mu_diamond_per_mm * 10, t_up_mm=t_up, t_down_mm=t_dn, half_opening_deg=exp.half_opening_deg)
with np.errstate(all="ignore"):
    ratio = PT_sim / (P * T)
ang_in = np.degrees(np.arccos(np.abs(c[:, 0])))
ang_out = np.degrees(np.arccos(np.abs(np.sum(c * s, axis=1))))
margin = exp.half_opening_deg - np.maximum(ang_in, ang_out)       # >0 inside the cone
print(f"{st.name} {sys.argv[2] if len(sys.argv) > 2 else 'untilted'}  bandwidth_rel={exp.bandwidth_rel}  spots={len(t)}")
for tol in (1e-4, 1e-3, 1e-2):
    bad = ~(np.abs(ratio - 1) < tol)
    deep = bad & (np.abs(margin) > 1.5)
    print(f"  |P*T ratio - 1| >= {tol:g} (or non-finite): {int(bad.sum()):4d} spots; of these {int(deep.sum())} lie more than 1.5 deg from the cone edge"
          + (f"  (worst deep: {np.nanmax(np.abs(ratio[deep] - 1)):.2e})" if deep.any() else ""))
good = np.abs(ratio - 1) < 1e-3
print(f"  spots with margin > 1.5 deg: {int((margin > 1.5).sum())}; their max |ratio-1| = {np.nanmax(np.abs(ratio[margin > 1.5] - 1)):.2e}")
