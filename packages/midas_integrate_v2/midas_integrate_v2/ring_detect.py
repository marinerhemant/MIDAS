"""Auto-detect Debye-Scherrer rings + suggest a calibrant material.

A user with a calibrant frame and rough geometry can call
:func:`detect_rings` to find the strongest peaks in the radial
profile and get back their R, 2θ, d-spacing values. They can then
identify the calibrant by passing a candidate material's d-spacings to
:func:`suggest_material`, which scores how well the observed and
predicted ring patterns match.

Built-in calibrants:
- CeO₂ (cubic fluorite, a = 5.411 Å)
- LaB₆ (cubic, a = 4.156 Å)
- Si (diamond cubic, a = 5.4309 Å)
- Cr₂O₃ (corundum-type R-3c, a = 4.961 Å, c = 13.599 Å; JCPDS 38-1479)

Add your own by passing ``predicted_d_A=[...]`` to ``suggest_material``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# Built-in calibrant d-spacings (Å). Ordered from largest to smallest;
# include enough rings to identify reliably (typically the first 5-8).
CALIBRANTS: Dict[str, List[float]] = {
    "ceo2": [3.124, 2.706, 1.913, 1.633, 1.562, 1.353,
             1.241, 1.210, 1.105, 1.043],
    "lab6": [4.156, 2.939, 2.400, 2.078, 1.859, 1.697,
             1.470, 1.385, 1.314, 1.252],
    "si":   [3.135, 1.921, 1.638, 1.357, 1.246, 1.108,
             1.045, 0.960, 0.918, 0.859],
    # Cr2O3 — JCPDS 38-1479 (corundum structure). Indexed: (012),
    # (104), (110), (006), (113), (024), (116), (211)/(122), (018),
    # (214). Values to 0.001 Å.
    "cr2o3": [3.633, 2.666, 2.480, 2.265, 2.175, 1.815,
              1.672, 1.579, 1.466, 1.430],
}


@dataclass
class DetectedRing:
    """One detected Debye-Scherrer ring in the calibrant profile."""
    R_px: float                  # radial position
    intensity: float             # integrated profile value at the peak
    two_theta_deg: float         # diffraction angle (depends on Lsd, px)
    d_spacing_A: float           # Bragg d-spacing (depends on λ)


def detect_rings(
    r_axis_px: np.ndarray,
    profile: np.ndarray,
    *,
    Lsd_um: float,
    px_um: float,
    wavelength_A: float,
    min_relative_height: float = 0.05,
    min_separation_px: int = 5,
    max_rings: int = 12,
) -> List[DetectedRing]:
    """Find peaks in the radial profile and convert to (2θ, d).

    Parameters
    ----------
    r_axis_px :
        R bin centres in pixels.
    profile :
        Integrated 1-D intensity, same length as r_axis_px.
    Lsd_um, px_um, wavelength_A :
        Geometry needed to convert R → 2θ → d.
    min_relative_height :
        Peaks must rise above ``min_relative_height · profile.max()``.
    min_separation_px :
        Reject peaks closer than this many pixels (typically larger than
        the calibrant ring intrinsic width).
    max_rings :
        Truncate to the strongest ``max_rings`` peaks.

    Returns
    -------
    list of :class:`DetectedRing` sorted by R (smallest first).
    """
    if r_axis_px.shape != profile.shape:
        raise ValueError(
            f"r_axis shape {r_axis_px.shape} != profile shape "
            f"{profile.shape}"
        )
    try:
        from scipy.signal import find_peaks
    except ImportError as e:
        raise ImportError(
            "detect_rings requires scipy; pip install scipy"
        ) from e
    threshold = profile.max() * min_relative_height
    peaks, _ = find_peaks(profile, height=threshold,
                            distance=min_separation_px)
    if len(peaks) > max_rings:
        # Keep the strongest max_rings
        idx_strongest = np.argsort(profile[peaks])[-max_rings:]
        peaks = np.sort(peaks[idx_strongest])

    rings = []
    for p in peaks:
        R = float(r_axis_px[p])
        # Bragg: 2θ = atan(R · px / Lsd), d = λ / (2 sin θ)
        two_theta_rad = np.arctan(R * px_um / Lsd_um)
        sin_theta = np.sin(two_theta_rad / 2.0)
        d_A = float(wavelength_A / (2.0 * sin_theta))
        rings.append(DetectedRing(
            R_px=R,
            intensity=float(profile[p]),
            two_theta_deg=float(np.degrees(two_theta_rad)),
            d_spacing_A=d_A,
        ))
    return rings


@dataclass
class MaterialMatch:
    """How well a candidate material matches an observed ring pattern."""
    name: str
    matched_pairs: List[Tuple[DetectedRing, float]]  # (observed, predicted_d)
    rms_d_diff_A: float                              # smaller = better
    n_matched: int


def suggest_material(
    detected: List[DetectedRing],
    *,
    candidates: Optional[List[str]] = None,
    custom: Optional[Dict[str, List[float]]] = None,
    d_match_tol_A: float = 0.05,
) -> List[MaterialMatch]:
    """Score how well each candidate material's d-spacings match the
    detected rings.

    Parameters
    ----------
    detected :
        Output of :func:`detect_rings`.
    candidates :
        Built-in calibrant names to try; defaults to all
        (``CALIBRANTS.keys()``).
    custom :
        Additional ``{name: [d_A, …]}`` to test.
    d_match_tol_A :
        Maximum d-spacing mismatch to count a ring as "matched".
        0.05 Å is reasonable for an initial geometry guess.

    Returns
    -------
    list of :class:`MaterialMatch` sorted by ``n_matched`` desc, then by
    RMS d-diff asc. The first entry is the best guess.
    """
    if not detected:
        return []
    cand = dict(CALIBRANTS) if candidates is None else {
        n: CALIBRANTS[n] for n in candidates if n in CALIBRANTS
    }
    if custom:
        cand.update(custom)
    matches = []
    for name, predicted in cand.items():
        matched: List[Tuple[DetectedRing, float]] = []
        residuals = []
        for ring in detected:
            # Find closest predicted d
            diffs = [abs(ring.d_spacing_A - d) for d in predicted]
            min_d = min(diffs)
            if min_d <= d_match_tol_A:
                matched.append((ring, predicted[diffs.index(min_d)]))
                residuals.append(min_d)
        rms = float(np.sqrt(np.mean(np.asarray(residuals) ** 2))) \
              if residuals else float("inf")
        matches.append(MaterialMatch(
            name=name, matched_pairs=matched,
            rms_d_diff_A=rms, n_matched=len(matched),
        ))
    matches.sort(key=lambda m: (-m.n_matched, m.rms_d_diff_A))
    return matches


__all__ = [
    "CALIBRANTS",
    "DetectedRing",
    "MaterialMatch",
    "detect_rings",
    "suggest_material",
]
