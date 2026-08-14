"""Write a small synthetic XRD-CT scan to disk, in a real detector layout.

The notebooks use this so they run on any machine with no beamline data. It
deliberately writes **actual raw files** -- 8192-byte header, contiguous int32
frames -- rather than handing back arrays, so the notebook exercises the same
``DTScan`` / ``FrameReducer`` path a real scan takes. A demo that bypasses the
reader proves nothing about the reader.

The sample is two phases in different places, each with its own ring radius.
That is the minimum needed for the workflow to mean anything: a single uniform
phase would reconstruct correctly even if the channel selection, the snake
handling and the rotation-axis shift were all wrong.

Physics, such as it is: each ring's integrated intensity along a ray is the
line integral of that phase's density, i.e. its Radon transform. So the frame
for one (translation, rotation) is painted with each ring at an amplitude given
by the projection of that phase map. That is the correct relationship, which is
what makes the reconstruction recover the phase maps rather than something that
merely looks plausible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["DemoScan", "make_scan", "make_calibrant_frame",
           "EXAMPLE_INSTRUMENT"]

HEADER_BYTES = 8192


@dataclass
class DemoScan:
    """Everything the notebook needs to point at what was written."""

    directory: Path
    stem: str
    start_nr: int
    end_nr: int
    dark_file: Path
    n_pixels: int
    beam_centre: tuple[float, float]
    ring_radii_px: tuple[float, float]
    start_omega: float
    omega_step: float
    n_rotations: int
    n_translations: int
    truth: dict          # phase name -> (n_trans, n_trans) density map

    def describe(self) -> str:
        return (f"{self.n_translations} translations x {self.n_rotations} "
                f"rotations of {self.n_pixels}x{self.n_pixels} px, rings at "
                f"{self.ring_radii_px[0]:.0f} and {self.ring_radii_px[1]:.0f} px")


def _phase_maps(n: int) -> dict:
    """Two overlapping regions: a big disc, and a smaller off-centre one."""
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - c, yy - c)
    outer = (rr <= 0.34 * n).astype(np.float64)
    inner = (np.hypot(xx - c - 0.16 * n, yy - c) <= 0.12 * n).astype(np.float64)
    # Phase A everywhere in the sample, phase B only in the inclusion, so the
    # two maps are neither identical nor disjoint -- the interesting case.
    return {"A": outer, "B": inner * 1.5}


def _radon(img: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """(n_angles, n_det) line integrals, matching midas_dt's convention.

    ``t = x sin(theta) + y cos(theta)``. Linear splat into neighbouring bins.
    Written out here rather than imported so the demo data does not depend on
    the code under test.
    """
    n = img.shape[0]
    c = (n - 1) / 2.0
    out = np.zeros((len(angles_deg), n))
    yy, xx = np.mgrid[0:n, 0:n]
    x = (xx - c).ravel()
    y = (yy - c).ravel()
    v = img.ravel()
    keep = v != 0
    x, y, v = x[keep], y[keep], v[keep]
    for a, th in enumerate(np.deg2rad(angles_deg)):
        t = x * np.sin(th) + y * np.cos(th) + c
        lo = np.floor(t).astype(int)
        frac = t - lo
        for shift, w in ((0, 1.0 - frac), (1, frac)):
            d = lo + shift
            ok = (d >= 0) & (d < n)
            np.add.at(out[a], d[ok], v[ok] * w[ok])
    return out


def make_scan(
    outdir,
    *,
    n_pixels: int = 192,
    n_translations: int = 16,
    n_rotations: int = 24,
    ring_radii_px: tuple[float, float] = (55.0, 80.0),
    ring_width_px: float = 2.2,
    peak_counts: float = 900.0,
    background: float = 12.0,
    dark_level: float = 40.0,
    start_omega: float = 0.0,
    omega_step: float = 180.0 / 24,
    negate_omega: bool = False,
    drop_first_frame: bool = True,
    seed: int = 0,
) -> DemoScan:
    """Write ``n_translations`` raw files plus a dark, and return the pointers.

    ``drop_first_frame`` writes one extra throwaway frame per file, so the
    default ``DTScan.from_stem(drop_first_frame=True)`` is correct for this
    data. Set it False to emit files with no throwaway, which is what most
    beamlines other than 1-ID produce -- the notebook uses it to show that the
    reader's convention has to match the writer's.

    ``negate_omega`` only affects the *reported* truth angles; the frames are
    written in acquisition order either way.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    n = n_pixels
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - c, yy - c)

    truth = _phase_maps(n_translations)
    omega = start_omega + omega_step * np.arange(n_rotations)
    # Projections of each phase map: (n_rotations, n_translations)
    proj = {k: _radon(v, -omega if negate_omega else omega)
            for k, v in truth.items()}
    scale = max(max(p.max() for p in proj.values()), 1e-9)

    # Ring shape, precomputed once: a Gaussian annulus per radius.
    shells = [np.exp(-0.5 * ((rr - R) / ring_width_px) ** 2)
              for R in ring_radii_px]

    n_extra = 1 if drop_first_frame else 0
    for t in range(n_translations):
        frames = np.empty((n_rotations + n_extra, n, n), dtype=np.float64)
        if drop_first_frame:
            # A throwaway: 1-ID writes one, and it is NOT a real exposure.
            # Making it obviously different (near-zero signal) means a notebook
            # that forgets to drop it produces a visibly wrong first projection
            # instead of a subtly wrong one.
            frames[0] = dark_level + rng.normal(0, 1.0, (n, n))
        for i in range(n_rotations):
            amp_a = peak_counts * proj["A"][i, t] / scale
            amp_b = peak_counts * proj["B"][i, t] / scale
            img = background + amp_a * shells[0] + amp_b * shells[1]
            frames[i + n_extra] = rng.poisson(img) + dark_level
        _write_raw(outdir / f"demo_scan_{t + 1:06d}.raw", frames)

    dark = dark_level + rng.normal(0, 1.0, (4, n, n))
    _write_raw(outdir / "demo_dark_000000.raw", dark)

    return DemoScan(
        directory=outdir, stem="demo_scan", start_nr=1, end_nr=n_translations,
        dark_file=outdir / "demo_dark_000000.raw", n_pixels=n,
        beam_centre=(c, c), ring_radii_px=ring_radii_px,
        start_omega=start_omega, omega_step=omega_step,
        n_rotations=n_rotations, n_translations=n_translations, truth=truth,
    )


def _write_raw(path: Path, frames: np.ndarray) -> None:
    """Header then contiguous int32 frames -- the layout ``RawFormat`` reads."""
    with open(path, "wb") as fh:
        fh.write(b"\0" * HEADER_BYTES)
        np.clip(frames, 0, None).astype("<i4").tofile(fh)


# ----------------------------------------------------------- calibrant frame
#: Instrument settings of the example beamline. Generic detector/optics
#: parameters -- nothing here identifies a sample, a proposal or a user.
#: Matched to the archived single-detector calibration the notebook quotes as
#: its reference standard, so synthetic and real numbers are comparable.
EXAMPLE_INSTRUMENT = dict(
    wavelength_a=0.189714,      # 65.4 keV
    px_um=200.0,
    n_pixels_y=2048,
    n_pixels_z=2048,
    lsd_um=940_086.0,
)


def make_calibrant_frame(
    *,
    lsd_um: float | None = None,
    px_um: float | None = None,
    n_pixels_y: int | None = None,
    n_pixels_z: int | None = None,
    wavelength_a: float | None = None,
    bc_offset_px: tuple[float, float] = (14.0, -9.0),
    tilt_deg: float = 0.0,
    lattice_a: float = 5.4116,          # CeO2
    peak_counts: float = 4.0e3,
    background: float = 30.0,
    ring_width_px: float = 2.4,
    seed: int = 0,
):
    """A synthetic CeO2 powder pattern with a KNOWN, deliberately-off geometry.

    Returns ``(frame, truth)``. ``truth`` carries the beam centre, distance and
    tilt actually used, so a calibration can be scored against them rather than
    against "it looks about right".

    The beam centre is offset from the detector centre and one tilt is applied,
    because a calibration demo where the guess is already correct teaches
    nothing -- the point is to see rings that are visibly not concentric with
    the guess and then watch them come right.

    **``tilt_deg`` defaults to 0, deliberately.** The stretch it applies,
    ``r -> r (1 + tilt cos eta)``, is to first order *exactly a beam-centre
    shift* -- degenerate with BC by construction -- and it is not the
    projective deformation ``midas_calibrate_v2`` actually refines against. So a
    non-zero value plants something the refiner cannot null: measured, it left
    174.8 ue (a FAIL) and pushed BC_y 1.9 px off, neither of which says anything
    about the calibration. With tilt 0 the planted geometry is one the pipeline
    can represent exactly, so the demo tests recovery rather than model error.

    Real detectors do tilt. That is what the archived single-detector example
    in the notebook is for -- it carries genuine tilts and genuine distortion.
    """
    inst = EXAMPLE_INSTRUMENT
    lsd = float(lsd_um if lsd_um is not None else inst["lsd_um"])
    px = float(px_um if px_um is not None else inst["px_um"])
    ny = int(n_pixels_y if n_pixels_y is not None else inst["n_pixels_y"])
    nz = int(n_pixels_z if n_pixels_z is not None else inst["n_pixels_z"])
    lam = float(wavelength_a if wavelength_a is not None else inst["wavelength_a"])

    rng = np.random.default_rng(seed)
    bc_y = (ny - 1) / 2.0 + bc_offset_px[0]
    bc_z = (nz - 1) / 2.0 + bc_offset_px[1]

    # CeO2 is fcc (sg 225): h,k,l all even or all odd.
    hkls, seen = [], set()
    for h in range(0, 7):
        for k in range(0, 7):
            for l in range(0, 7):
                if h == k == l == 0:
                    continue
                if not (h % 2 == k % 2 == l % 2):
                    continue
                s = h * h + k * k + l * l
                if s in seen:
                    continue
                seen.add(s)
                hkls.append((h, k, l, s))
    hkls.sort(key=lambda t: t[3])

    yy, zz = np.meshgrid(np.arange(ny), np.arange(nz))
    rr = np.hypot(yy - bc_y, zz - bc_z)
    eta = np.arctan2(zz - bc_z, yy - bc_y)

    frame = np.full((nz, ny), background, dtype=np.float64)
    rings = []
    for (h, k, l, s) in hkls:
        d = lattice_a / np.sqrt(s)
        arg = lam / (2.0 * d)
        if arg >= 1.0:
            continue
        two_theta = 2.0 * np.arcsin(arg)
        radius = lsd * np.tan(two_theta) / px
        # keep rings that actually land on the detector, with a margin
        if radius < 40 or radius > 0.98 * min(bc_y, bc_z, ny - bc_y, nz - bc_z) * 1.35:
            continue
        # a tilt makes the ring radius vary with azimuth, to first order
        r_eff = radius * (1.0 + np.deg2rad(tilt_deg) * np.cos(eta))
        amp = peak_counts / (1.0 + 0.05 * s)        # weaker at high Q
        frame += amp * np.exp(-0.5 * ((rr - r_eff) / ring_width_px) ** 2)
        rings.append({"hkl": (h, k, l), "d_a": d, "radius_px": float(radius),
                      "two_theta_deg": float(np.degrees(two_theta))})

    frame = rng.poisson(np.clip(frame, 0, None)).astype(np.float64)
    truth = {"lsd_um": lsd, "bc_y_px": bc_y, "bc_z_px": bc_z, "px_um": px,
             "wavelength_a": lam, "n_pixels_y": ny, "n_pixels_z": nz,
             "tilt_deg": tilt_deg,
             "lattice_a": lattice_a, "rings": rings}
    return frame, truth
