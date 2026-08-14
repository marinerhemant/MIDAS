"""Index the observed U3O8 rings against candidate phases.

The rings come from dev/inspect_u3o8_lineout.py (translation 27, 24 frames
averaged, 60-560 px at 1 px bins). Assignment is what turns a d-spacing map
into a lattice measurement, so it is done with midas_hkls' reflection lists
rather than by eye.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from midas_dt.index_rings import ALPHA_U3O8, CEO2, PhaseCandidate, index_rings  # noqa: E402

# R (px), d (A) -- measured by dev/inspect_u3o8_lineout.py with a ROLLING
# baseline. The earlier 6-ring list came from a global median, which is far
# above the background at low R and below it at high R; it found the strong
# inner rings and lost the rest.
OBSERVED = [
    (91.06, 9.3691), (95.07, 8.9742), (98.08, 8.6992), (115.11, 7.4121),
    (205.29, 4.1573), (231.34, 3.6895), (235.35, 3.6268), (238.36, 3.5811),
    (244.37, 3.4931), (248.38, 3.4368), (253.39, 3.3689), (264.41, 3.2286),
    (270.42, 3.1570), (323.53, 2.6396), (326.53, 2.6153), (332.55, 2.5681),
    (336.55, 2.5376), (411.70, 2.0755), (420.72, 2.0312), (425.73, 2.0074),
    (428.74, 1.9933), (437.76, 1.9524), (475.83, 1.7968), (483.85, 1.7672),
    (498.88, 1.7142), (508.90, 1.6806), (539.96, 1.5844), (548.98, 1.5585),
]

# gamma-UO3, orthorhombic. The 600A/700A/800A series is an oxidation study, so
# the sample is not guaranteed to be single-phase U3O8.
GAMMA_UO3 = PhaseCandidate(
    name="gamma-UO3", space_group=62, a=9.813, b=19.93, c=9.711,
    reference="orthorhombic Pbnm, a=9.813 b=19.93 c=9.711 A",
)


def main() -> int:
    radii = [r for r, _ in OBSERVED]
    dobs = [d for _, d in OBSERVED]

    print("observed rings (from the raw lineout):")
    for r, d in OBSERVED:
        print(f"  R = {r:7.2f} px   d = {d:6.3f} A")
    print()

    # UO2, fluorite -- the reduced end member of the oxidation series.
    UO2 = PhaseCandidate(
        name="UO2", space_group=225, a=5.4704, b=5.4704, c=5.4704,
        reference="fluorite Fm-3m, a=5.4704 A",
    )
    # U4O9, cubic superstructure of UO2.
    U4O9 = PhaseCandidate(
        name="U4O9", space_group=197, a=21.77, b=21.77, c=21.77,
        reference="cubic I23, a=21.77 A",
    )

    for phase in (ALPHA_U3O8, GAMMA_UO3, UO2, U4O9, CEO2):
        for tol in (20_000.0, 5_000.0):
            res = index_rings(dobs, phase, radii_px=radii, tolerance_ppm=tol)
            print(f"[tolerance {tol/1e4:.1f}%]  {phase.name}: "
                  f"{res.n_matched}/{len(res.matches)} matched, "
                  f"rms {res.rms_residual_ppm:.0f} ppm")
        print()

    print("Reading this: with 28 observed rings a WRONG cell can still match")
    print("many of them by chance, especially a large one like U4O9 with a")
    print("dense reflection list. What separates a real assignment is the RMS")
    print("residual at a TIGHT tolerance -- a correct cell should sit at a few")
    print("hundred ppm, not a few thousand.")
    return 0


def coverage_check():
    """Do two phases TOGETHER account for rings neither covers alone?

    The 600A/700A/800A series is an oxidation study, so a mixture is the
    obvious explanation for no single cell indexing everything. If the two
    leading candidates are complementary -- each claiming rings the other
    misses -- that supports a mixture. If they claim the SAME rings and leave
    the same ones unexplained, it does not, and a third phase (or wrong cells)
    is indicated.
    """
    from midas_dt.index_rings import index_rings

    GAMMA_UO3 = PhaseCandidate(
        name="gamma-UO3", space_group=62, a=9.813, b=19.93, c=9.711)
    radii = [r for r, _ in OBSERVED]
    dobs = [d for _, d in OBSERVED]
    tol = 5_000.0

    a = index_rings(dobs, ALPHA_U3O8, radii_px=radii, tolerance_ppm=tol)
    g = index_rings(dobs, GAMMA_UO3, radii_px=radii, tolerance_ppm=tol)
    ia = {i for i, m in enumerate(a.matches) if m.matched}
    ig = {i for i, m in enumerate(g.matches) if m.matched}

    print(f"\nat {tol/1e4:.1f}% tolerance, over {len(OBSERVED)} rings:")
    print(f"  alpha-U3O8 only : {len(ia - ig):2d}")
    print(f"  gamma-UO3 only  : {len(ig - ia):2d}")
    print(f"  both            : {len(ia & ig):2d}")
    print(f"  NEITHER         : {len(set(range(len(OBSERVED))) - ia - ig):2d}")
    left = sorted(set(range(len(OBSERVED))) - ia - ig)
    if left:
        print("  unexplained rings (R px, d A):")
        for i in left:
            print(f"    {OBSERVED[i][0]:7.2f}  {OBSERVED[i][1]:7.4f}")


if __name__ == "__main__":
    rc = main()
    coverage_check()
    raise SystemExit(rc)
