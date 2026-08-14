"""D0 gate: read the real U3O8 600 A scan and check every convention.

Run on haydn (reach: ssh chiltepin -> ssh haydn, both s1iduser):

    PYTHONPATH=. python dev/verify_u3o8_600a.py

Checks, in order of how badly each would hurt if wrong:

1. the raw layout divides exactly for data AND dark
2. a frame reads back with a sane mean, and the flip is applied
3. omega covers 360 degrees and is negated
4. the snake is DETECTED, not assumed -- the scan sets BadRotation 1
5. the geometry file parses and gives 90.5 keV, not the commented 55.618
6. recon_size(55) == 128, matching what the 2023 runs used
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from midas_dt import (                                        # noqa: E402
    DTScan, PILATUS_1475x1679, detect_snake, geometry_from_legacy_params,
    recon_size,
)

RAW = Path("/scratch/s1iduser/mpe_nov22_midas2/mpe_nov22")
PARAMS = Path("/scratch/s1iduser/DTnewversion/"
              "ps_dt_u3o8_600A_fileNr_161_215_rad_105_525.txt")
STEM, START, END = "dm_dt_pf_U3O8_600A", 161, 215
DARK = RAW / "dark_before_000159.raw"

ok = True


def check(label, condition, detail=""):
    global ok
    mark = "PASS" if condition else "FAIL"
    if not condition:
        ok = False
    print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    fmt = PILATUS_1475x1679

    print("1. raw layout")
    f0 = RAW / f"{STEM}_{START:06d}.raw"
    n = fmt.n_frames(f0)                       # raises if it does not divide
    check("data file divides exactly", n == 1441, f"{n} frames, {f0.stat().st_size} B")
    nd = fmt.n_frames(DARK)
    check("dark divides exactly", nd == 10, f"{nd} frames")

    print("2. scan + a real frame")
    scan = DTScan.from_stem(
        RAW, STEM, START, END, fmt=fmt,
        start_omega=180.25, omega_step=-0.25, dark_file=DARK,
    )
    check("55 translations", scan.n_translations == 55, str(scan.n_translations))
    check("1440 usable frames (1441 minus the throwaway)",
          scan.n_frames == 1440, str(scan.n_frames))
    frame = scan.frame(0, 0)
    check("frame shape is (1679, 1475)", frame.shape == (1679, 1475), str(frame.shape))
    check("frame has signal", frame.mean() > 0,
          f"mean {frame.mean():.2f}, max {frame.max()}")
    unflipped = np.asarray(scan.translation(0)[0])
    check("vertical flip applied", not np.array_equal(frame, unflipped))
    dark = scan.dark()
    check("dark averages to a frame", dark.shape == (1679, 1475),
          f"mean {dark.mean():.2f}")

    print("3. omega")
    span = abs(scan.omega_deg[-1] - scan.omega_deg[0]) + 0.25
    check("covers 360 degrees", abs(span - 360.0) < 0.3, f"{span:.2f} deg")
    check("negated (1-ID aerotech)", scan.omega_deg[0] < 0,
          f"first omega {scan.omega_deg[0]:.2f}")

    print("4. snake detection (scan declares BadRotation 1)")
    # Cheap per-(translation, frame) summary: mean of a detector stripe.
    prof = np.empty((scan.n_translations, 60))
    idx = np.linspace(0, scan.n_frames - 1, 60).astype(int)
    for t in range(scan.n_translations):
        mm = scan.translation(t)
        prof[t] = [float(np.asarray(mm[i, 800:840, :]).mean()) for i in idx]
    is_snake, gain = detect_snake(prof)
    check("snake DETECTED from the data", is_snake, f"gain {gain:.3f}")

    print("5. geometry")
    geo = geometry_from_legacy_params(PARAMS)
    check("Lsd", abs(geo.lsd_um - 1071098.336) < 1e-3, f"{geo.lsd_um}")
    check("energy is 90.5 keV, not the commented 55.618",
          abs(geo.energy_kev - 90.5) < 0.1, f"{geo.energy_kev:.2f} keV")
    check("detector size", (geo.n_pixels_y, geo.n_pixels_z) == (1475, 1679))

    print("6. reconstruction size")
    rs = recon_size(scan.n_translations, extra_pad=True)
    check("recon_size(55) == 128 (matches the 2023 runs)", rs == 128, str(rs))

    print()
    print("D0 GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
