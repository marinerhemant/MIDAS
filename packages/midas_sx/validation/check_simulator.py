"""Simulator self-test (dev). Run: python check_simulator.py [S1|S2] [target_hits]"""
import sys, time, resource
import numpy as np
sys.path.insert(0, ".")
import simulate as sim
from midas_hkls import corrections as C

which = sys.argv[1] if len(sys.argv) > 1 else "S1"
hits = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
st = sim.structure_cu() if which == "S1" else sim.structure_s2_bilayer()
print(f"{st.name}: min interatomic distance = {sim.min_interatomic_distance(st):.3f} A")
tilt = (2.0, 1.0) if (len(sys.argv) > 3 and sys.argv[3] == 'tilted') else (0.0, 0.0)
exp = sim.make_experiment(dac_tilt_deg=tilt, t_plus_mm=2.5, t_minus_mm=3.0 if tilt != (0.0, 0.0) else 2.5)
print(f"mu_diamond = {exp.mu_diamond_per_mm*10:.4f} cm^-1  tth_max = {exp.tth_max_deg():.2f} deg  c0 = {exp.dac_axis0()}")
t0 = time.time()
r = sim.simulate(st, exp, target_hits=hits, verbose=False)
print(f"simulate: {time.time()-t0:.1f} s, spots={len(r.truth)}, complete={int(r.truth.complete.sum())}, "
      f"peak RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.0f} MB")
t = r.truth
s = t[["sx", "sy", "sz"]].to_numpy()
Lcf = C.rotation_lorentz(s, min_abs=0.0)
ratio = t.lorentz_emergent.to_numpy() / Lcf
print("emergent/closed-form Lorentz: n=%d  median=%.5f  robust spread (MAD*1.4826)=%.5f  min=%.4f  max=%.4f" % (
    len(ratio), np.median(ratio), 1.4826*np.median(np.abs(ratio-np.median(ratio))), ratio.min(), ratio.max()))
print("hits per spot: min=%d median=%d" % (t.n_hits.min(), t.n_hits.median()))
print(t.head(8).to_string())

# --- polarization x anvil transmission: simulator (Thomson + ray tracing) vs midas_hkls.corrections
k = exp.k
PT_sim = t.w_tot_all.to_numpy() * k * np.pi / 180.0 / t.lorentz_emergent.to_numpy()
tth = t.tth.to_numpy(); eta = np.degrees(np.arctan2(-t.sy.to_numpy(), t.sz.to_numpy()))
P = C.polarization_factor(tth, eta, pol_fraction=exp.pol_fraction, pol_plane_eta_deg=90.0)
c = C.rotate_about_axis(exp.dac_axis0(), [0, 0, 1], t.omega.to_numpy())
ci_sign = np.sign(c @ np.array([1.0, 0.0, 0.0]))
t_up = np.where(ci_sign > 0, exp.t_minus_mm, exp.t_plus_mm)       # upstream anvil = the one on the -sign(c.s0) side
t_dn = np.where(ci_sign > 0, exp.t_plus_mm, exp.t_minus_mm)
T = C.diamond_anvil_transmission(s, c, mu_per_cm=exp.mu_diamond_per_mm * 10, t_up_mm=t_up, t_down_mm=t_dn,
                                 half_opening_deg=exp.half_opening_deg)
ref = P * T
r2 = PT_sim / ref
print("P*T  simulator / corrections: n=%d median=%.5f  robust spread=%.5f  min=%.4f max=%.4f" % (
    len(r2), np.median(r2), 1.4826*np.median(np.abs(r2-np.median(r2))), r2.min(), r2.max()))
print("P alone range over spots: %.4f..%.4f   T range: %.4f..%.4f" % (P.min(), P.max(), T.min(), T.max()))
# what the same comparison says if the omega sign convention were wrong (positive control for detectability)
c_bad = C.rotate_about_axis(exp.dac_axis0(), [0, 0, 1], -t.omega.to_numpy())
T_bad = C.diamond_anvil_transmission(s, c_bad, mu_per_cm=exp.mu_diamond_per_mm * 10, t_up_mm=t_up, t_down_mm=t_dn,
                                     half_opening_deg=exp.half_opening_deg)
rb = PT_sim / (P * T_bad)
print("wrong omega sign: P*T ratio median=%.4f spread=%.4f min=%.4f max=%.4f  (zeros from cone: %d)" % (
    np.nanmedian(rb[np.isfinite(rb)]), np.nanstd(rb[np.isfinite(rb)]), np.nanmin(rb[np.isfinite(rb)]), np.nanmax(rb[np.isfinite(rb)]), int((~np.isfinite(rb)).sum())))
