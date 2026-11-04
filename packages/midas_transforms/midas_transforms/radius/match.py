"""Ring-matching helpers for ``calc_radius``.

``calc_radius`` assigns a merged spot to a ring when ``|R_obs - R_ring| < Width``.
``R_ring`` is the per-material ``hkls.csv`` radius, ``Lsd * tan(2theta)`` on the
plane at ``Lsd``. ``R_obs`` used to be the *flat* distance of the spot from the
beam centre, which is not that quantity once the detector is tilted: the flat
distance of a point on a ring is off by about ``|tilt| * R^2 / Lsd`` (tens to
thousands of um on an outer ring at ~1 deg), so real, correctly thresholded
spots fell outside ``Width`` and were dropped without a message.

The peak-search band (``midas_peakfit.geometry.compute_good_coords``), the eta
coverage table and ``global_powder`` all select pixels with the tilt-corrected
radius. This module gives ``calc_radius`` the same definition, by running the
spot pixel coordinates through the kernel ``fit_setup`` uses
(:func:`midas_transforms.fit_setup.transform.apply_tilt_distortion`).

It also resolves which ``Width`` applies, and builds the per-run report of how
many peaks matched no ring.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

LOG = logging.getLogger(__name__)

#: ``Width`` (um) when the archive carries neither ``Width`` nor ``WidthTthPx``.
#: Matches the MIDAS default (CalcRadiusAllZarr.c, the params registry,
#: ``global_powder``, ``eta_coverage``) and ``midas_peakfit``'s default.
DEFAULT_WIDTH_UM = 1500.0

#: A spot counts as a near miss when it lies within this many Widths of a ring
#: but outside the Width window.
NEAR_MISS_FACTOR = 2.0

#: A configured ring that received no spot, but has at least this many near
#: misses, is reported as a warning (the window is probably too narrow).
NEAR_MISS_WARN = 5


def resolve_width_um(zarr_params) -> Tuple[float, str]:
    """Return ``(Width_um, source)`` for the ring-assignment window.

    Order: ``WidthTthPx`` (the C code's override for this window; despite the
    name its unit is um), then ``Width``, then ``WidthOrig``, then
    :data:`DEFAULT_WIDTH_UM`. A non-positive value counts as absent. Before
    this existed an absent ``Width`` reached the kernel as -1 and matched no
    spot at all.
    """
    for name in ("WidthTthPx", "Width", "WidthOrig"):
        v = float(getattr(zarr_params, name, -1.0) or -1.0)
        if v > 0:
            return v, name
    LOG.warning(
        "calc_radius: neither Width nor WidthTthPx is set in the archive; "
        "using the default %.0f um for the ring-assignment window.",
        DEFAULT_WIDTH_UM,
    )
    return DEFAULT_WIDTH_UM, "default"


def _load_residual_corr_map(zarr_params, dev, dt) -> Optional[torch.Tensor]:
    """Same lookup table ``fit_setup`` applies (``ResidualCorrectionMap``).

    File layout: ``(NrPixelsZ, NrPixelsY)`` row-major float64.
    """
    path = getattr(zarr_params, "ResidualCorrectionMap", "") or ""
    if not path:
        return None
    ny, nz = zarr_params.NrPixelsY, zarr_params.NrPixelsZ
    try:
        md = np.fromfile(path, dtype=np.float64)
    except (OSError, ValueError) as e:
        LOG.warning("calc_radius: could not load residual map %s: %s", path, e)
        return None
    if ny > 0 and nz > 0 and md.size == ny * nz:
        return torch.from_numpy(md.reshape(nz, ny)).to(device=dev, dtype=dt)
    LOG.warning(
        "calc_radius: residual map %s has %d elements, expected %d "
        "(NrPixelsZ*NrPixelsY); ignoring.", path, md.size, ny * nz,
    )
    return None


def corrected_radius_um(
    zarr_params,
    Y_pix: torch.Tensor,
    Z_pix: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Tilt- and distortion-corrected radius (um, on the plane at ``Lsd``).

    ``Y_pix`` / ``Z_pix`` are the spot pixel coordinates (``Result_*.csv``
    columns 3 and 4), exactly what ``fit_setup`` feeds to
    ``apply_tilt_distortion``. Returns ``None`` when the archive carries no
    usable geometry (``Lsd`` or ``PixelSize`` not positive), in which case the
    caller keeps the flat radius.
    """
    if not (zarr_params.Lsd > 0 and zarr_params.PixelSize > 0):
        return None
    from midas_distortion import v1_to_v2_coeffs, v2_to_v1_coeffs

    from ..fit_setup.transform import apply_tilt_distortion

    dev, dt = Y_pix.device, Y_pix.dtype

    def _t(v):
        return torch.tensor(float(v), dtype=dt, device=dev)

    coeffs_v2 = zarr_params.dist_coeffs_v2
    if coeffs_v2 is None:
        coeffs_v2 = v1_to_v2_coeffs(np.array(
            [getattr(zarr_params, f"p{i}") for i in range(15)], dtype=np.float64))
    p_v1 = np.asarray(v2_to_v1_coeffs(np.asarray(coeffs_v2, dtype=np.float64)))
    rho_d = zarr_params.RhoD if zarr_params.RhoD > 0 else zarr_params.MaxRingRad

    Y_lab, Z_lab = apply_tilt_distortion(
        Y_pix, Z_pix,
        Lsd=_t(zarr_params.Lsd), BC_y=_t(zarr_params.YCen), BC_z=_t(zarr_params.ZCen),
        tx=_t(zarr_params.tx), ty=_t(zarr_params.ty), tz=_t(zarr_params.tz),
        p_coeffs=torch.tensor(p_v1, dtype=dt, device=dev),
        px=_t(zarr_params.PixelSize), rho_d=_t(rho_d),
        residual_corr_map=_load_residual_corr_map(zarr_params, dev, dt),
    )
    return torch.sqrt(Y_lab * Y_lab + Z_lab * Z_lab)


def build_match_report(
    R_match_um: torch.Tensor,
    R_flat_um: torch.Tensor,
    ring_numbers: Sequence[int],
    ring_radii_um: torch.Tensor,
    width_um: float,
    width_source: str,
    geometry_corrected: bool,
) -> Dict:
    """Count how many input peaks matched which ring, and how many matched none.

    ``R_match_um`` is the radius used for the assignment, ``R_flat_um`` the
    plain distance from the beam centre (what the assignment used before it was
    tilt aware). Their disagreement is reported so a run shows what the
    correction changed.
    """
    n_in = int(R_match_um.shape[0])
    rings = [int(r) for r in ring_numbers]
    stats: Dict = {
        "n_input": n_in,
        "width_um": float(width_um),
        "width_source": width_source,
        "tilt_aware": bool(geometry_corrected),
        "n_matched": 0,
        "n_unmatched": 0,
        "n_near_miss": 0,
        "n_recovered_vs_flat": 0,
        "n_rejected_vs_flat": 0,
        "per_ring": {r: {"matched": 0, "near_miss": 0} for r in rings},
    }
    if n_in == 0 or not rings:
        stats["summary"] = "calc_radius: no input peaks or no configured rings."
        return stats

    rad = ring_radii_um.to(device=R_match_um.device, dtype=R_match_um.dtype)
    d = (R_match_um.unsqueeze(1) - rad.unsqueeze(0)).abs()          # (N, R)
    d_flat = (R_flat_um.unsqueeze(1) - rad.unsqueeze(0)).abs()
    in_win = d < width_um
    in_win_flat = d_flat < width_um
    matched = in_win.any(dim=1)
    matched_flat = in_win_flat.any(dim=1)

    dmin, nearest = d.min(dim=1)
    near = (~matched) & (dmin < NEAR_MISS_FACTOR * width_um)

    stats["n_matched"] = int(matched.sum())
    stats["n_unmatched"] = n_in - stats["n_matched"]
    stats["n_near_miss"] = int(near.sum())
    stats["n_recovered_vs_flat"] = int((matched & ~matched_flat).sum())
    stats["n_rejected_vs_flat"] = int((~matched & matched_flat).sum())
    per_ring_matched = in_win.sum(dim=0)
    near_idx = nearest[near]
    for i, r in enumerate(rings):
        stats["per_ring"][r]["matched"] = int(per_ring_matched[i])
        stats["per_ring"][r]["near_miss"] = int((near_idx == i).sum())

    lines: List[str] = [
        "calc_radius: %d of %d peaks matched a ring (Width %.0f um from %s; "
        "%s radius)" % (
            stats["n_matched"], n_in, width_um, width_source,
            "tilt-corrected" if geometry_corrected else "flat, no geometry in archive"),
        "calc_radius: %d peaks matched no ring (%d of them within %.0f x Width "
        "of one)" % (stats["n_unmatched"], stats["n_near_miss"], NEAR_MISS_FACTOR),
    ]
    if geometry_corrected and (stats["n_recovered_vs_flat"] or stats["n_rejected_vs_flat"]):
        lines.append(
            "calc_radius: tilt correction recovered %d peaks a flat-radius match "
            "would have dropped and rejected %d it would have kept" % (
                stats["n_recovered_vs_flat"], stats["n_rejected_vs_flat"]))
    stats["summary"] = "\n".join(lines)
    return stats


def log_match_report(stats: Dict) -> None:
    """Emit the report: INFO summary, WARNING for rings that look starved."""
    LOG.info("%s", stats["summary"])
    if stats["n_input"] > 0 and stats["n_matched"] == 0:
        LOG.warning(
            "calc_radius: no peak matched any configured ring (Width %.0f um). "
            "Check Width, the ring list and the geometry in the archive.",
            stats["width_um"])
        return
    for ring, c in stats["per_ring"].items():
        if c["matched"] == 0 and c["near_miss"] >= NEAR_MISS_WARN:
            LOG.warning(
                "calc_radius: ring %d got no peaks but %d lie within %.0f x Width "
                "(%.0f um) of it. Width may be too narrow for this geometry.",
                ring, c["near_miss"], NEAR_MISS_FACTOR, stats["width_um"])
