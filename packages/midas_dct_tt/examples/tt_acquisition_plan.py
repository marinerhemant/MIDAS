"""FF grain map -> a TT acquisition plan a beamline can act on.

Fang & Ludwig's fwd-DCT is deployed at a beamline; ours is not, and the gap is not
the physics but the last mile. This turns a scan plan into the artifact an
experimenter actually needs: the sample orientation to set, the scan to run, the
relative exposure per reflection, and a predicted first frame to compare against.

    python examples/tt_acquisition_plan.py Grains.csv [energy_keV] [grain_id]

DELIBERATELY NOT EMITTED: goniometer motor angles. ESRF solves the alignment as
(samrx, samry, diffrz); we return a rotation matrix. Mapping one to the other needs
sign conventions this package has not pinned down, and a wrong sign there mirrors
the reconstruction with no other symptom. The rotation matrix is stated instead so
the local convention can be applied by someone who knows it.
"""
import sys

import torch

from midas_dct_tt import (PlaneDetector, kinematical_validity, psi_scan,
                          read_grains_csv, select_tt_candidates, tt_scan_plan)
from midas_dct_tt.acceptance import ObjectiveFreeAcceptance, orientation_resolution_deg
from midas_dfxm.io import fcc_reference_crystal

path = sys.argv[1] if len(sys.argv) > 1 else "Grains.csv"
energy = float(sys.argv[2]) if len(sys.argv) > 2 else 71.7
lam = 12.398 / energy

grains, meta = read_grains_csv(path)
cands = select_tt_candidates(grains, min_confidence=0.5, max_offcenter_um=300.0,
                             top_n=5)
if len(sys.argv) > 3:
    want = int(sys.argv[3])
    cands = [g for g in grains if g.grain_id == want] or cands
g = cands[0]
# Pass the crystal or the planner ranks systematic absences as optimal:
# {100} is forbidden in fcc, and being the lowest-angle set it wins a
# missing-cone ranking outright.
xtal = fcc_reference_crystal()
plan = tt_scan_plan(g, lam, crystal=xtal)

print("=" * 74)
print(f"TT ACQUISITION PLAN   grain {g.grain_id}   {energy} keV (lambda {lam:.5f} A)")
print(f"source: {path}   ({len(grains)} grains)")
print("=" * 74)

print("\n1. GRAIN")
print(f"   position (sample frame)  {[round(float(x),1) for x in g.position_um]} um")
print(f"   offset from rotation axis {g.offcenter_um():.1f} um  "
      f"(the grain orbits this distance; keep it in the beam)")
print(f"   refined cell  a={float(g.lattice[0]):.5f} A  "
      f"alpha={float(g.lattice[3]):.4f} deg")
print("   NOTE the alignment below uses this REFINED cell, not a nominal one --")
print("   aligning on a reference lattice leaves the scan off the Bragg condition.")

print("\n2. VALIDITY CHECK BEFORE YOU SCAN")
size = 2.0 * float(g.radius_um) if g.radius_um == g.radius_um else float("nan")
if size == size:
    kv = kinematical_validity(plan.report.hkls[0], crystal=xtal, wavelength_A=lam,
                              thickness_um=size)
    print(f"   path length ~{size:.1f} um -> t/Lambda = {kv['ratio']:.3f}  "
          f"[{kv['regime']}]")
    if kv["regime"] != "kinematical":
        print(f"   *** {kv['relative_error']*100:.0f}% kinematical error. Intensities from a")
        print("   *** kinematical model are NOT quantitative for this grain.")
else:
    print("   GrainRadius unavailable or untrustworthy in this file -- check it.")

print("\n3. REFLECTIONS  (sorted by missing cone; see the CRLB/leakage trade)")
print(f"   {plan.report.summary()}")
tot = 0.0
for hkl, al in plan.alignments:
    r = orientation_resolution_deg(al)
    acc = ObjectiveFreeAcceptance(k_in=al.k_in, q_nom=al.G_lab)
    print(f"   {str(tuple(hkl)):12s} theta {float(al.theta_deg):6.3f} deg   "
          f"missing cone {float(al.missing_cone_deg()):5.2f} deg   "
          f"rock res {r['rock']:.4f} deg")
    tot += 1.0
print(f"   relative exposure: equal across the {int(tot)} reflections "
      "(CRLB is |Q|-driven; see paper Sec. 4)")

print("\n4. SAMPLE ORIENTATION TO SET  (rotation matrix, sample -> lab)")
for hkl, al in plan.alignments[:1]:
    R = al.sample_to_lab
    for row in R:
        print("      [" + "  ".join(f"{float(v):+.6f}" for v in row) + "]")
    print(f"   tomographic axis (lab) "
          f"{[round(float(v),6) for v in al.rotation_axis]}")
    print(f"   Bragg residual at this setting {float(al.bragg_residual()):+.3e} "
          "(must be ~0)")

print("\n5. SCAN")
psi = psi_scan(180)
print(f"   psi: {len(psi)} steps over 360 deg, step {float(psi[1]-psi[0]):.2f} deg")
print("   rotate about the tomographic axis above; G is invariant, so the Bragg")
print("   condition holds for the whole sweep.")
print("   REQUIREMENT: per-frame registration to ~0.1 px. At 0.3 px the")
print("   intragranular signal is lost (paper Sec. 6.4).")
print("\n" + "=" * 74)
