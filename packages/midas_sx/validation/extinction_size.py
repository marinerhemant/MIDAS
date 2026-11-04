"""How big is primary extinction for the S2 (bilayer-like) structure at 30 keV? (dev, uses midas_hkls.extinction)  python extinction_size.py"""
import sys; sys.path.insert(0, "."); sys.path.insert(0, "..")
import numpy as np
import simulate as sim
from midas_hkls import Lattice, SpaceGroup
from midas_hkls.extinction import extinction_length_um, primary_extinction_factor
st = sim.structure_s2_bilayer(); lat = Lattice(*st.cell); sg = SpaceGroup.from_hm(st.spacegroup)
lam = 12.3984 / 30.0
hkl_all = sim.enumerate_hkl(st, 0.8)
reps = sorted({sg.asu_representative(*map(int, h)) for h in hkl_all})
hkl = np.array(reps); F2 = sim.f2_truth(st, hkl); keep = F2 > 1e-6 * F2.max(); hkl, F2 = hkl[keep], F2[keep]
d = np.array([lat.d_spacing(*map(int, h)) for h in hkl]); theta = np.degrees(np.arcsin(lam / (2 * d)))
V = lat.volume()
Lam = np.array([extinction_length_um(np.sqrt(f), unit_cell_volume_A3=V, wavelength_A=lam, theta_deg=t) for f, t in zip(F2, theta)])
order = np.argsort(F2)[::-1]
lines = [f"S2 bilayer-like, 30 keV (lambda {lam:.4f} A), cell volume {V:.1f} A^3, {len(hkl)} unique allowed reflections to d = 0.8 A",
         "extinction length Lambda (um), strongest 8 reflections:  " + "; ".join(f"{tuple(hkl[i])}: {Lam[i]:.1f}" for i in order[:8]),
         f"Lambda over all reflections: min {Lam.min():.1f}, median {np.median(Lam):.1f}, max {Lam.max():.1f} um", "",
         "crystal path t (um) | primary-extinction factor y=tanh(x)/x, x=t/Lambda: strongest reflection y, fraction of reflections with y<0.99, with y<0.95, intensity-weighted mean (1-y)"]
for t in (2.0, 5.0, 10.0, 20.0):
    y = np.array([primary_extinction_factor(t, L) for L in Lam])
    w = F2 / F2.sum()
    lines.append(f"  t = {t:5.1f} um | y(strongest) = {y[order[0]]:.4f} | {np.mean(y < 0.99):5.1%} below 0.99 | {np.mean(y < 0.95):5.1%} below 0.95 | weighted mean(1-y) = {np.sum(w * (1 - y)):.4f}")
print("\n".join(lines))
open("results/extinction_size_S2.txt", "w").write("\n".join(lines) + "\n")
