"""Look at the raw azimuthally-integrated lineout before trusting any fit.

Motivated by a result that looked fine and was not: a run over the 105-125 px
window returned RMEAN ~= 115.87 with a spread of under 2 px. 115 is the centre
of that window. A fitted peak centre pinned to the middle of its own window,
with no spread, is what fitting a FLAT window looks like -- the moment seed
lands on the centroid of noise, which is the centre.

So: integrate one frame over a wide radius range and print where the peaks
actually are, before deciding which windows are worth reconstructing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from midas_dt import Channel, DTScan, geometry_from_legacy_params, parse_legacy_params  # noqa: E402
from midas_dt.maps import radius_to_d_spacing                                            # noqa: E402
from midas_dt.reduce import FrameReducer                                                 # noqa: E402

RAW = Path("/scratch/s1iduser/mpe_nov22_midas2/mpe_nov22")
PARAMS = Path("/scratch/s1iduser/DTnewversion/"
              "ps_dt_u3o8_600A_fileNr_161_215_rad_105_525.txt")
STEM, START, END = "dm_dt_pf_U3O8_600A", 161, 215
DARK = RAW / "dark_before_000159.raw"


def main() -> int:
    geo = geometry_from_legacy_params(PARAMS)
    params = parse_legacy_params(PARAMS)
    scan = DTScan.from_stem(RAW, STEM, START, END,
                            start_omega=params["startOme"],
                            omega_step=params["omeStep"], dark_file=DARK)

    # Wide window at 1 px bins, covering everything the 2023 runs ever fitted.
    wide = Channel(60.0, 560.0, eta_min=-180.0, eta_max=180.0,
                   r_bin=1.0, eta_bin=360.0, label="survey")
    reducer = FrameReducer(geo, wide, dark=scan.dark())

    # Average several frames from a middle translation: one frame at one
    # rotation may sit between rings for a textured or coarse sample.
    mid = scan.n_translations // 2
    frames = list(np.linspace(0, scan.n_frames - 1, 24).astype(int))
    acc = None
    for f in frames:
        r = reducer.reduce_frame(scan.frame(mid, f))
        acc = r.lineout if acc is None else acc + r.lineout
    prof = acc / len(frames)

    radii = np.linspace(wide.r_min, wide.r_max, wide.n_r)
    d = radius_to_d_spacing(radii, geo)

    print(f"translation {mid}, {len(frames)} frames averaged")
    print(f"lineout: {prof.size} bins over {wide.r_min:g}-{wide.r_max:g} px")
    print(f"  min {np.nanmin(prof):.4g}  median {np.nanmedian(prof):.4g}  "
          f"max {np.nanmax(prof):.4g}")

    # Rolling baseline, not a global median: the background falls steeply
    # with radius, so one global threshold finds the inner rings and loses the
    # outer ones. The first version of this script did exactly that and
    # reported 6 rings where the image plainly shows 15-20.
    from midas_dt.rings import find_rings

    rings = find_rings(radii, prof, geometry=geo, min_snr=3.0,
                       min_separation_px=3.0)
    print(f"\nrings above 3 sigma of a rolling baseline: {len(rings)}")
    for ring in rings:
        print("  " + ring.describe())

    print("\nas an indexing input (R px, d A):")
    print("OBSERVED = [")
    for ring in rings:
        print(f"    ({ring.radius_px:.2f}, {ring.d_spacing_a:.4f}),")
    print("]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
