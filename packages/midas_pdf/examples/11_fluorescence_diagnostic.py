"""11 — Fluorescence diagnostic: will my sample fluoresce at this energy?

Fluorescence is the dominant smooth-background contaminant. Given a composition
and incident energy, list the elements that fluoresce and on which lines so you
know whether a smooth-background term is needed (or to change energy).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from midas_pdf import Composition, expected_fluorescence

# Fe2O3 at 20 keV: Fe is above its K-edge (7.11 keV) -> fluoresces strongly
print("Fe2O3 @ 20 keV:")
for d in expected_fluorescence(["Fe", "O"], incident_energy_keV=20.0):
    y = f"{d['yield']:.2f}" if d["yield"] else "  - "
    line = f"{d['line_keV']:.2f} keV" if d["line_keV"] else "   -    "
    print(f"   {d['element']:>2} {d['shell']:>2}  edge {d['edge_keV']:6.3f} keV  "
          f"line {line}  yield {y}")

# SiO2 at a high-energy PDF beamline (74.5 keV): the edges of Si/O are far
# below the incident energy, so the diagnostic *does* flag them — but the lines
# are soft (~0.5-1.7 keV), heavily self-absorbed, and practically negligible at
# a hard-X-ray PDF beamline. The diagnostic reports; the practitioner judges.
print("\nSiO2 @ 74.5 keV (soft lines -> practically negligible at high energy):")
for d in expected_fluorescence(["Si", "O"], wavelength_A=0.1665):
    line = f"{d['line_keV']:.2f} keV" if d["line_keV"] else "   -    "
    print(f"   {d['element']:>2} {d['shell']:>2}  line {line}")
