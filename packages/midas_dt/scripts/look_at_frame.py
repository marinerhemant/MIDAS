"""Look at a raw detector frame, with the ring radii overlaid.

Everything so far has been mediated: integrate, fit, index. This looks at the
image. Two questions the integrated lineout cannot answer:

1. **Are the rings concentric about the stated beam centre?** If they are not,
   the geometry is wrong and every d-spacing is wrong with it -- which is a
   candidate explanation for the failed indexing.
2. **Are the rings CONTINUOUS?** midas-dt's whole scope boundary is that
   XRD-CT assumes powder-like continuous rings. If they break into discrete
   spots the sample is coarse-grained and this is the wrong technique --
   scanning-3DXRD is. That has never been checked for this dataset.

Writes PNGs next to the output directory, not to /tmp.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from midas_dt import DTScan, geometry_from_legacy_params, parse_legacy_params  # noqa: E402

RAW = Path("/scratch/s1iduser/mpe_nov22_midas2/mpe_nov22")
PARAMS = Path("/scratch/s1iduser/DTnewversion/"
              "ps_dt_u3o8_600A_fileNr_161_215_rad_105_525.txt")
STEM, START, END = "dm_dt_pf_U3O8_600A", 161, 215
DARK = RAW / "dark_before_000159.raw"

#: Observed ring radii, px (dev/inspect_u3o8_lineout.py).
RINGS = [115.11, 205.29, 248.38, 253.39, 323.53, 483.85]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("./frame_look"))
    ap.add_argument("--translation", type=int, default=27)
    ap.add_argument("--nframes", type=int, default=40,
                    help="frames to sum; a single frame of a coarse sample "
                         "can look spotty purely from counting statistics")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    geo = geometry_from_legacy_params(PARAMS)
    params = parse_legacy_params(PARAMS)
    scan = DTScan.from_stem(RAW, STEM, START, END,
                            start_omega=params["startOme"],
                            omega_step=params["omeStep"], dark_file=DARK)
    dark = scan.dark()

    idx = np.linspace(0, scan.n_frames - 1, args.nframes).astype(int)
    acc = np.zeros(scan.fmt.frame_shape, dtype=np.float64)
    for f in idx:
        acc += scan.frame(args.translation, f).astype(np.float64)
    img = acc / len(idx) - dark

    # BC as the parameter file states it, used AS-IS.
    #
    # scan.frame() already applies ImTransOpt 2 (flip top-bottom), so what it
    # returns is in the MIDAS orientation that BC_z is defined in. Flipping
    # BC_z as well double-applies the transform: the first version of this
    # script did that, put the marker 51 px off the beamstop, and made the ring
    # radii look wrong when they were not.
    bc_y = geo.bc_y_px
    bc_z = geo.bc_z_px

    print(f"frame: translation {args.translation}, {len(idx)} frames averaged")
    print(f"  shape {img.shape}, min {img.min():.1f}, "
          f"median {np.median(img):.1f}, max {img.max():.1f}")
    print(f"  beam centre (file) y={geo.bc_y_px:.2f} z={geo.bc_z_px:.2f}")
    print(f"  overlay centre (frame is already flipped) y={bc_y:.2f} z={bc_z:.2f}")

    disp = np.log10(np.clip(img, 1.0, None))
    vmax = float(np.percentile(disp, 99.9))

    # ---- full frame with the ring radii overlaid
    fig, ax = plt.subplots(figsize=(11, 12))
    ax.imshow(disp, cmap="viridis", vmin=0, vmax=vmax, origin="upper")
    for r in RINGS:
        ax.add_patch(Circle((bc_y, bc_z), r, fill=False, color="red",
                            lw=0.8, alpha=0.9))
        ax.annotate(f"{r:.0f}", (bc_y + r * 0.707, bc_z - r * 0.707),
                    color="red", fontsize=7)
    ax.plot(bc_y, bc_z, "r+", ms=14)
    ax.set_title(f"{STEM} t={args.translation}, {len(idx)} frames, log10\n"
                 f"red = radii from the integrated lineout, about the file's BC")
    fig.savefig(args.out / "frame_rings.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- zoom on the innermost ring: is it continuous or spotty?
    r0 = RINGS[0]
    pad = 90
    y0, y1 = int(bc_y - r0 - pad), int(bc_y + r0 + pad)
    z0, z1 = int(bc_z - r0 - pad), int(bc_z + r0 + pad)
    y0, z0 = max(y0, 0), max(z0, 0)
    y1, z1 = min(y1, img.shape[1]), min(z1, img.shape[0])
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(disp[z0:z1, y0:y1], cmap="viridis", vmin=0, vmax=vmax)
    ax.add_patch(Circle((bc_y - y0, bc_z - z0), r0, fill=False,
                        color="red", lw=1.0))
    ax.plot(bc_y - y0, bc_z - z0, "r+", ms=14)
    ax.set_title(f"zoom on R = {r0:.1f} px -- continuous ring, or spots?")
    fig.savefig(args.out / "frame_zoom_inner.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- azimuthal profile ON the ring: the quantitative version of "spotty?"
    yy, zz = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    rr = np.hypot(yy - bc_y, zz - bc_z)
    # Pilatus module gaps read as dead pixels. Counting them as azimuthal
    # structure is what made the first run call every ring SPOTTY.
    live = img > -1e3
    print(f"  live-pixel fraction {live.mean():.3f} "
          f"(the rest is module gaps / the beamstop)")
    for r in (RINGS[0], RINGS[1]):
        band = (rr > r - 2) & (rr < r + 2) & live
        eta = np.degrees(np.arctan2(zz[band] - bc_z, yy[band] - bc_y))
        val = img[band]
        nb = 180
        prof = np.array([val[(eta >= -180 + i * 2) & (eta < -178 + i * 2)].mean()
                         if np.any((eta >= -180 + i * 2) & (eta < -178 + i * 2))
                         else np.nan for i in range(nb)])
        good = np.isfinite(prof)
        cv = float(np.nanstd(prof) / max(abs(np.nanmean(prof)), 1e-9))
        print(f"  R={r:6.1f}: azimuthal bins {good.sum()}/{nb}, "
              f"mean {np.nanmean(prof):8.2f}, CV {cv:5.2f}"
              f"   {'SPOTTY' if cv > 1.0 else 'continuous-ish'}")

    print(f"\nwrote {args.out}/frame_rings.png and frame_zoom_inner.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
