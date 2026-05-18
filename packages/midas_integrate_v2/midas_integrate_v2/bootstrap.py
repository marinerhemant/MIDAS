"""Initial-geometry estimators for users who don't have a paramstest.

If you arrive at the beamline with a calibrant frame and only know the
detector basics (pixel size, sample-to-detector distance from the
stage encoder, wavelength from the beamline mono), you can use these
to bootstrap a starting :class:`IntegrationSpec` good enough for
calibrate-v2 / integrate-v2 refinement to take over.

Two estimators:

- :func:`estimate_BC_from_image` — find the centre of the brightest
  ring by 2D centroid in a wide-window region around the image
  centre. Robust to module gaps if a mask is provided.
- :func:`estimate_initial_spec` — wraps the above plus reasonable
  defaults so you get a ready-to-use ``IntegrationSpec`` from a few
  scalars + an image.

These are meant for *bootstrapping* — the BC estimate is typically
within a few px of the true value but not sub-pixel; finish with
joint refinement (notebook 03 / 06 / 07).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from midas_integrate.params import IntegrationParams

from .compat import spec_from_v1_params
from .spec import IntegrationSpec


def _ring_centroid(
    image: np.ndarray,
    *,
    initial_BC: Tuple[float, float],
    inner_radius_px: float,
    outer_radius_px: float,
    mask: Optional[np.ndarray] = None,
    intensity_threshold_pct: float = 90.0,
) -> Tuple[float, float]:
    """Intensity-weighted centroid of pixels in a thin annulus.

    Uses pixels whose intensity is above the
    ``intensity_threshold_pct``th percentile within the annulus. This
    ignores noise and faint background, picking out the bright Bragg
    rings in a calibrant.
    """
    NZ, NY = image.shape
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    Yc = yy - initial_BC[0]
    Zc = zz - initial_BC[1]
    R = np.sqrt(Yc * Yc + Zc * Zc)
    in_annulus = (R >= inner_radius_px) & (R <= outer_radius_px)
    if mask is not None:
        in_annulus = in_annulus & ~mask.astype(bool)
    if not in_annulus.any():
        raise ValueError(
            f"no pixels in annulus [{inner_radius_px}, {outer_radius_px}] "
            f"around BC ({initial_BC[0]}, {initial_BC[1]})"
        )
    # Threshold to keep only the brightest fraction (the ring itself, not
    # background)
    intens = image[in_annulus]
    threshold = np.percentile(intens, intensity_threshold_pct)
    bright = in_annulus & (image >= threshold)
    if bright.sum() < 100:
        # too few points for a stable centroid — fall back to all in-annulus
        bright = in_annulus
    w = image[bright].astype(np.float64)
    w_sum = w.sum() + 1e-30
    BC_y = float((yy[bright] * w).sum() / w_sum)
    BC_z = float((zz[bright] * w).sum() / w_sum)
    return BC_y, BC_z


def estimate_BC_from_image(
    image: np.ndarray,
    *,
    initial_BC: Optional[Tuple[float, float]] = None,
    ring_radius_px: float = None,
    annulus_half_width_px: float = 30.0,
    mask: Optional[np.ndarray] = None,
    n_iterations: int = 3,
    intensity_threshold_pct: float = 90.0,
) -> Tuple[float, float]:
    """Estimate beam centre by iterated centroid of the brightest ring.

    Algorithm:
      1. Start with ``initial_BC`` (defaults to image centre).
      2. Find the radial position of the strongest peak in the image's
         radial profile around that BC.
      3. Compute the intensity-weighted centroid of the bright pixels
         in an annulus around that radius.
      4. Use that as the next BC estimate; repeat ``n_iterations``
         times. Each iteration sharpens the estimate.

    ``ring_radius_px`` may be supplied to skip step 2 (faster +
    deterministic when you already know the first ring's approximate R).
    """
    NZ, NY = image.shape
    if initial_BC is None:
        initial_BC = (NY / 2.0, NZ / 2.0)

    BC_y, BC_z = float(initial_BC[0]), float(initial_BC[1])
    for _ in range(n_iterations):
        if ring_radius_px is None:
            R_pred = _find_brightest_ring_R(image, (BC_y, BC_z), mask)
        else:
            R_pred = float(ring_radius_px)
        BC_y, BC_z = _ring_centroid(
            image,
            initial_BC=(BC_y, BC_z),
            inner_radius_px=max(1.0, R_pred - annulus_half_width_px),
            outer_radius_px=R_pred + annulus_half_width_px,
            mask=mask,
            intensity_threshold_pct=intensity_threshold_pct,
        )
    return BC_y, BC_z


def _find_brightest_ring_R(
    image: np.ndarray,
    BC: Tuple[float, float],
    mask: Optional[np.ndarray],
) -> float:
    """Return the R (px) of the strongest peak in a quick radial profile.

    Uses 1-px R bins, no η weighting — just argmax in the radial
    intensity histogram.
    """
    NZ, NY = image.shape
    yy, zz = np.meshgrid(np.arange(NY), np.arange(NZ), indexing="xy")
    R = np.sqrt((yy - BC[0]) ** 2 + (zz - BC[1]) ** 2)
    R_int = np.floor(R).astype(np.int64)
    R_max = int(R.max()) + 1
    flat_R = R_int.ravel()
    flat_I = image.ravel().astype(np.float64)
    if mask is not None:
        m = mask.astype(bool).ravel()
        flat_R = flat_R[~m]
        flat_I = flat_I[~m]
    sums = np.bincount(flat_R, weights=flat_I, minlength=R_max)
    counts = np.bincount(flat_R, minlength=R_max)
    safe_counts = np.where(counts > 0, counts, 1)
    profile = sums / safe_counts
    # Skip the very-low-R region (beam stop / centre artefacts)
    skip = max(20, R_max // 30)
    if skip >= R_max - 5:
        skip = max(0, R_max // 4)
    return float(np.argmax(profile[skip:]) + skip)


def estimate_initial_spec(
    image: np.ndarray,
    *,
    NrPixelsY: int,
    NrPixelsZ: int,
    pxY_um: float,
    Lsd_um: float,
    Wavelength_A: float,
    pxZ_um: Optional[float] = None,
    initial_BC: Optional[Tuple[float, float]] = None,
    mask: Optional[np.ndarray] = None,
    RhoD_px: Optional[float] = None,
    RBinSize: float = 1.0,
    EtaBinSize: float = 5.0,
    requires_grad: bool = False,
    lattice: str = "cartesian",
    apothem_um: Optional[float] = None,
    lattice_orientation_deg: float = 0.0,
) -> IntegrationSpec:
    """Bootstrap a usable :class:`IntegrationSpec` from minimal inputs.

    Returns a spec with:
      - Beam centre estimated from the image (via
        :func:`estimate_BC_from_image`).
      - Lsd, wavelength, pixel size as supplied.
      - Tilts, distortion, parallax all zero (refine later).
      - R range automatically set from 5 px to (max R in image).
      - η range -180..+180.

    Use this output as the seed for joint refinement (notebook 03/06/07).

    For PIXIRAD-style hex detectors, pass ``lattice='hex_offset_y'`` and
    ``apothem_um`` (typically 30 μm for PIXIRAD-1).  ``pxY_um``/``pxZ_um``
    are then derived from the apothem (``pxY=2a, pxZ=a√3``) and any
    values supplied for those kwargs are ignored.
    """
    import math
    if lattice == "hex_offset_y":
        if apothem_um is None:
            raise ValueError("lattice='hex_offset_y' requires apothem_um (μm)")
        pxY_um = 2.0 * apothem_um
        pxZ_um = apothem_um * math.sqrt(3.0)
    else:
        pxZ_um = pxZ_um or pxY_um
    if image.shape != (NrPixelsZ, NrPixelsY):
        raise ValueError(
            f"image shape {image.shape} must be (NrPixelsZ, NrPixelsY) = "
            f"({NrPixelsZ}, {NrPixelsY})"
        )
    BC_y, BC_z = estimate_BC_from_image(
        image, initial_BC=initial_BC, mask=mask,
    )
    NY, NZ = NrPixelsY, NrPixelsZ
    R_max_px = float(np.sqrt(max(BC_y, NY - BC_y) ** 2 +
                              max(BC_z, NZ - BC_z) ** 2))
    if RhoD_px is None:
        RhoD_px = R_max_px

    p = IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ,
        pxY=pxY_um, pxZ=pxZ_um,
        Lsd=Lsd_um, BC_y=BC_y, BC_z=BC_z,
        RhoD=RhoD_px,
        RMin=5.0, RMax=R_max_px - 5.0, RBinSize=RBinSize,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=EtaBinSize,
        Wavelength=Wavelength_A,
    )
    spec = spec_from_v1_params(p, requires_grad=requires_grad)
    if lattice != "cartesian":
        from .spec import _t
        spec.lattice = lattice
        spec.Apothem = _t(apothem_um, dtype=spec.dtype())
        spec.LatticeOrientation = _t(lattice_orientation_deg,
                                      dtype=spec.dtype())
        if requires_grad:
            spec.Apothem = spec.Apothem.detach().clone().requires_grad_(True)
            spec.LatticeOrientation = (
                spec.LatticeOrientation.detach().clone().requires_grad_(True)
            )
    return spec


__all__ = [
    "estimate_BC_from_image",
    "estimate_initial_spec",
]
