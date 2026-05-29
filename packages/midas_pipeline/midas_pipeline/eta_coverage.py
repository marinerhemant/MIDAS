"""Per-detector η coverage via pixel enumeration.

For each (detector, ring) pair we enumerate every panel pixel whose
distortion-corrected radial coordinate falls inside ``[R_n - W, R_n + W]``,
compute its η angle in the panel's lab frame, then collapse the η values
into a sorted set of contiguous arcs.

Stored back into the per-detector ``paramstest.txt`` as one row per arc:

    EtaCoverage_DetN ring_nr eta_lo_deg eta_hi_deg

All angles are in degrees, in [-180°, 180°), with the convention
``eta_hi - eta_lo`` is the positive arc length and arcs do not wrap past
±180° (a wrap-around arc is split into two rows).

Downstream stages (calc_radius Pass 2, midas-index, midas-fit-grain)
read these rows to:
  * scale per-detector observed powder intensity by 360° / Σ arc lengths
  * route a predicted spot at η_pred to a panel whose arcs cover η_pred
  * mask predicted spots that fall in no panel's coverage out of the
    completeness denominator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# Pixel rounding tolerance used to merge tiny gaps inside a single arc:
# distortion + sub-pixel motion frequently leaves 1–2-pixel-wide gaps in
# the in-band mask near corners. Anything narrower than this is treated
# as a single arc.
DEFAULT_GAP_MERGE_DEG = 1.0


@dataclass
class CoverageArc:
    ring_nr: int
    eta_lo_deg: float
    eta_hi_deg: float

    @property
    def width_deg(self) -> float:
        return self.eta_hi_deg - self.eta_lo_deg


def _calc_eta_angle_deg(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """MIDAS convention: ``eta = -arccos(z/r)`` for y>0, ``+arccos(z/r)`` else.

    Returns degrees in (-180, 180].
    """
    r = np.sqrt(y * y + z * z)
    r_safe = np.where(r == 0, 1.0, r)
    alpha = RAD2DEG * np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    return np.where(y > 0, -alpha, alpha)


def _tilt_matrix(tx_deg: float, ty_deg: float, tz_deg: float) -> np.ndarray:
    """Returns ``Rx(tx) · Ry(ty) · Rz(tz)`` matching the MIDAS / peakfit
    convention (also used by the C ``CorrectTiltSpatialDistortion``).
    """
    txr, tyr, tzr = (math.radians(a) for a in (tx_deg, ty_deg, tz_deg))
    Rx = np.array(
        [[1, 0, 0],
         [0, math.cos(txr), -math.sin(txr)],
         [0, math.sin(txr),  math.cos(txr)]],
        dtype=np.float64,
    )
    Ry = np.array(
        [[math.cos(tyr), 0,  math.sin(tyr)],
         [0,             1,  0],
         [-math.sin(tyr), 0, math.cos(tyr)]],
        dtype=np.float64,
    )
    Rz = np.array(
        [[math.cos(tzr), -math.sin(tzr), 0],
         [math.sin(tzr),  math.cos(tzr), 0],
         [0,              0,             1]],
        dtype=np.float64,
    )
    return Rx @ (Ry @ Rz)


def _pixel_rt_eta(
    n_pixels: int,
    px_um: float,
    lsd_um: float,
    y_bc_px: float,
    z_bc_px: float,
    tx_deg: float, ty_deg: float, tz_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute distortion-free per-pixel ``(R_panel_px, η_deg)``.

    Returns
    -------
    R_panel_px : (NrPixels, NrPixels) float64
        Radial pixel distance in the lab frame after tilt rotation.
    eta_deg : (NrPixels, NrPixels) float64
        Azimuthal angle in degrees in (-180, 180].

    Distortion (p0..p10) is intentionally omitted: the user specified
    pixel-enumeration on the *nominal* geometry; downstream tools that
    want a distortion-aware coverage should plug in their own residual
    map.
    """
    a = np.arange(n_pixels, dtype=np.float64)
    A, B = np.meshgrid(a, a, indexing="ij")     # A = row index, B = col index
    Yc = (-A + y_bc_px) * px_um                  # FF convention (flip_y=True)
    Zc = (B - z_bc_px) * px_um

    TRs = _tilt_matrix(tx_deg, ty_deg, tz_deg)
    ABCPr0 = TRs[0, 1] * Yc + TRs[0, 2] * Zc
    ABCPr1 = TRs[1, 1] * Yc + TRs[1, 2] * Zc
    ABCPr2 = TRs[2, 1] * Yc + TRs[2, 2] * Zc

    X = lsd_um + ABCPr0
    # Beam-axis projection: keep only the (Y, Z) image at the Lsd plane.
    proj = lsd_um / np.where(X == 0, 1e-9, X)
    Yp = ABCPr1 * proj                            # μm
    Zp = ABCPr2 * proj
    Rt_um = np.sqrt(Yp * Yp + Zp * Zp)
    Rt_px = Rt_um / px_um
    eta_deg = _calc_eta_angle_deg(Yp, Zp)
    return Rt_px, eta_deg


def _arc_collapse(eta_values: np.ndarray, gap_merge_deg: float) -> List[Tuple[float, float]]:
    """Collapse a 1-D array of η values into ``(eta_lo, eta_hi)`` arcs.

    Uses a circular gap analysis: sort the η values, find the largest
    angular gaps, treat those as inter-arc separators if they exceed
    ``gap_merge_deg``.

    Returns a list of arcs in ascending order. For full ring coverage
    returns ``[(-180, 180)]``.
    """
    if eta_values.size == 0:
        return []
    eta_sorted = np.sort(eta_values)
    # Gaps between consecutive samples (sorted, plus wrap-around).
    diffs = np.diff(eta_sorted)
    wrap_gap = 360.0 - (eta_sorted[-1] - eta_sorted[0])
    # The full set of gaps between consecutive samples on the circle.
    all_gaps = np.concatenate([diffs, [wrap_gap]])
    # Index of separators: gaps wider than gap_merge_deg.
    sep_mask = all_gaps > gap_merge_deg
    if not sep_mask.any():
        # Effectively contiguous full coverage of [eta_min, eta_max] with
        # no significant gap; collapse to one arc.
        return [(float(eta_sorted[0]), float(eta_sorted[-1]))]
    # Each separator splits the sorted array into a contiguous arc.
    # The wrap-around at the end means arcs may also span -180 -> +180.
    sep_indices = np.where(sep_mask)[0]            # gap indices
    arcs: List[Tuple[float, float]] = []
    # Start: index 0; chunks of indices between separators.
    prev = 0
    for sep in sep_indices:
        if sep == len(eta_sorted) - 1:
            # The "wrap-around" gap is between last and first; current
            # chunk is sorted[prev .. last], no wrap.
            chunk = eta_sorted[prev:]
        else:
            chunk = eta_sorted[prev:sep + 1]
        if chunk.size > 0:
            arcs.append((float(chunk[0]), float(chunk[-1])))
        prev = sep + 1
    # Last chunk after the final non-wrap separator (if the wrap gap
    # was *not* a separator, chunks need to merge across the wrap).
    if not sep_mask[-1]:
        # Wrap-around closes the last arc with the first; merge.
        if arcs and prev < len(eta_sorted):
            tail = eta_sorted[prev:]
            tail_arc = (float(tail[0]), float(tail[-1]))
            head_arc = arcs[0]
            # Merge across ±180 boundary by representing as
            # (tail_lo, +180) + (-180, head_hi) — keep them separate
            # rows, but the consumer should treat them as one ring.
            arcs[0] = (-180.0, head_arc[1])
            arcs.insert(0, (tail_arc[0], 180.0))
            return arcs
        elif prev < len(eta_sorted):
            tail = eta_sorted[prev:]
            arcs.append((float(tail[0]), float(tail[-1])))
    return arcs


def compute_panel_eta_coverage(
    *,
    n_pixels: int,
    px_um: float,
    lsd_um: float,
    y_bc_px: float,
    z_bc_px: float,
    tx_deg: float,
    ty_deg: float,
    tz_deg: float,
    ring_radii_um: Sequence[Tuple[int, float]],     # [(ring_nr, R_um), ...]
    width_um: float = 1500.0,
    gap_merge_deg: float = DEFAULT_GAP_MERGE_DEG,
    gap_arc_min_deg: float = 0.5,
) -> List[CoverageArc]:
    """Per-panel η coverage across all configured rings.

    Parameters
    ----------
    n_pixels, px_um
        Detector pixel count + pixel size (μm). Square panels assumed.
    lsd_um, y_bc_px, z_bc_px, tx/ty/tz_deg
        Panel geometry (paramstest convention).
    ring_radii_um
        Per ring: ``(ring_nr, ring_radius_um)``.
    width_um
        Half-width of the ring band in μm (default 1500 μm matches MIDAS
        ``Width`` paramstest entry).
    gap_merge_deg
        Treat angular gaps narrower than this as part of a single arc.
    gap_arc_min_deg
        Drop arcs narrower than this (single-pixel artefacts at corners).

    Returns
    -------
    arcs : list of CoverageArc
        One row per (ring_nr, contiguous arc).
    """
    Rt_px, eta_deg = _pixel_rt_eta(
        n_pixels, px_um, lsd_um, y_bc_px, z_bc_px,
        tx_deg, ty_deg, tz_deg,
    )
    width_px = width_um / px_um
    flat_R = Rt_px.ravel()
    flat_eta = eta_deg.ravel()

    out: List[CoverageArc] = []
    for ring_nr, R_um in ring_radii_um:
        R_px = R_um / px_um
        in_band = (flat_R >= R_px - width_px) & (flat_R <= R_px + width_px)
        if not in_band.any():
            continue
        arcs = _arc_collapse(flat_eta[in_band], gap_merge_deg)
        for lo, hi in arcs:
            if hi - lo < gap_arc_min_deg:
                continue
            out.append(CoverageArc(ring_nr=int(ring_nr),
                                   eta_lo_deg=float(lo),
                                   eta_hi_deg=float(hi)))
    return out


def write_coverage_block(
    paramstest_path: Path,
    det_id: int,
    arcs: Iterable[CoverageArc],
) -> None:
    """Append ``EtaCoverage_DetN`` rows to a paramstest file.

    Idempotent: if rows for this det already exist they're rewritten.
    """
    text = paramstest_path.read_text() if paramstest_path.exists() else ""
    key = f"EtaCoverage_Det{det_id} "
    new_lines = [ln for ln in text.splitlines() if not ln.startswith(key)]
    for arc in arcs:
        new_lines.append(
            f"EtaCoverage_Det{det_id} {arc.ring_nr} "
            f"{arc.eta_lo_deg:.6f} {arc.eta_hi_deg:.6f}"
        )
    paramstest_path.write_text("\n".join(new_lines) + "\n")


def parse_coverage_blocks(paramstest_text: str) -> dict[int, list[CoverageArc]]:
    """Inverse of ``write_coverage_block``: read the per-det arc tables."""
    out: dict[int, list[CoverageArc]] = {}
    for raw in paramstest_text.splitlines():
        line = raw.strip().rstrip(";").rstrip()
        if not line.startswith("EtaCoverage_Det"):
            continue
        toks = line.split()
        try:
            det_id = int(toks[0][len("EtaCoverage_Det"):])
            ring = int(float(toks[1]))
            lo = float(toks[2])
            hi = float(toks[3])
        except (IndexError, ValueError):
            continue
        out.setdefault(det_id, []).append(
            CoverageArc(ring_nr=ring, eta_lo_deg=lo, eta_hi_deg=hi)
        )
    return out


def total_coverage_per_ring(
    arcs: Iterable[CoverageArc],
) -> dict[int, float]:
    """Aggregate arc widths per ring number (degrees)."""
    out: dict[int, float] = {}
    for arc in arcs:
        out[arc.ring_nr] = out.get(arc.ring_nr, 0.0) + max(0.0, arc.width_deg)
    return out


def panel_for_eta(
    eta_deg: float,
    ring_nr: int,
    arcs_by_det: dict[int, list[CoverageArc]],
) -> list[int]:
    """Return all detector IDs whose coverage contains (ring, η).

    Empty list if the predicted spot is outside every panel's coverage —
    callers use this to mask the spot from the completeness denominator.
    """
    matches: list[int] = []
    for det_id, arcs in arcs_by_det.items():
        for arc in arcs:
            if arc.ring_nr != ring_nr:
                continue
            if arc.eta_lo_deg <= eta_deg <= arc.eta_hi_deg:
                matches.append(det_id)
                break
    return matches
