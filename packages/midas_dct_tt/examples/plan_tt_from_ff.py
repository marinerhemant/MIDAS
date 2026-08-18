"""End-to-end: a MIDAS FF grain map -> a topotomography scan plan.

Run against any ``ProcessGrains`` output::

    python examples/plan_tt_from_ff.py /path/to/Grains.csv [energy_keV]

Everything the plan needs already exists in the FF result, so no extra
measurement is required to decide what to scan.
"""
import sys

from midas_dfxm.io import fcc_reference_crystal

from midas_dct_tt import read_grains_csv, select_tt_candidates, tt_scan_plan

path = sys.argv[1] if len(sys.argv) > 1 else "Grains.csv"
energy_keV = float(sys.argv[2]) if len(sys.argv) > 2 else 71.7
lam = 12.398 / energy_keV

grains, meta = read_grains_csv(path)
print(f"{path}: {len(grains)} grains  (NumGrains header: {meta.get('NumGrains')})")
print(f"energy {energy_keV} keV -> lambda {lam:.5f} A\n")

cands = select_tt_candidates(grains, min_confidence=0.5, max_offcenter_um=300.0,
                             top_n=5)
print(f"{len(cands)} TT candidates (confidence>=0.5, within 300 um of centre):")
for g in cands:
    print(f"  grain {g.grain_id:>7d}  R {g.radius_um:6.1f} um  "
          f"conf {g.confidence:.3f}  offcentre {g.offcenter_um():6.1f} um")

print()
for g in cands[:2]:
    # Without `crystal` the planner ranks systematic absences as optimal.
    print(tt_scan_plan(g, lam, crystal=fcc_reference_crystal()).summary())
    print()
