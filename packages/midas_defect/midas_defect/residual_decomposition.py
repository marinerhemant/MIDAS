"""Split an indexing residual into RADIAL and ANGULAR parts, because they mean
opposite things.

A single scalar "rms |dG|" hides the one distinction that decides what to do
next. Predicted and observed reflections can disagree in two independent ways:

**Radially** — |G| is wrong. That is a *d-spacing* error, so it indicts the
**cell**, the sample-detector distance, or the wavelength.

**Angularly** — the direction of G is wrong. That indicts the **orientation**,
or the presence of more than one grain.

Getting a large angular residual with a small radial one is not "a slightly bad
fit". It means *the cell is right and the orientation is not* — the reflections
are present, they simply do not all belong to one grain. Adding precision will
not help; adding grains will. In the analysis this came from, that decomposition
turned a stalled refinement into a diagnosis: 0.24 % radial against 1.5°
angular, a factor of **10.6** at the relevant |G|, on a sample everyone had been
treating as a single crystal.

The comparison must be made in commensurate units
-------------------------------------------------
Percent and degrees cannot be compared directly. At scattering vector |G|, an
angular error of θ displaces the reflection by ``|G|·θ`` in reciprocal space,
while a relative radial error ε displaces it by ``|G|·ε``. So the honest ratio
is ``θ_radians / ε_relative`` — dimensionless, and independent of |G|.
:func:`decompose_residuals` reports it that way, and also reports the
displacement each contributes in 1/Å so the two can be read on one axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["ResidualDecomposition", "decompose_residuals"]


@dataclass
class ResidualDecomposition:
    """Radial and angular parts of an indexing residual, kept apart."""
    radial_rel: np.ndarray               # (n,) signed (|q_obs| - |q_pred|)/|q_pred|
    angular_deg: np.ndarray              # (n,) angle between q_obs and q_pred
    displacement_radial: np.ndarray      # (n,) 1/A
    displacement_angular: np.ndarray     # (n,) 1/A
    q_mag: np.ndarray

    @property
    def median_radial_pct(self) -> float:
        return float(np.median(np.abs(self.radial_rel)) * 100.0)

    @property
    def median_angular_deg(self) -> float:
        return float(np.median(self.angular_deg))

    @property
    def angular_over_radial(self) -> float:
        """Dimensionless: median angular (rad) / median relative radial.

        > 1 means the misfit is dominated by ORIENTATION, not by the cell.
        ``inf`` when the radial residual is identically zero.
        """
        rad = float(np.median(np.abs(self.radial_rel)))
        ang = float(np.median(np.radians(self.angular_deg)))
        return float("inf") if rad == 0.0 else ang / rad

    @property
    def verdict(self) -> str:
        r = self.angular_over_radial
        if not np.isfinite(r):
            return ("ANGULAR-dominated (radial residual is zero): the cell is "
                    "exact and the misfit is entirely orientation. Suspect "
                    "multiple grains; more precision will not help.")
        if r > 3.0:
            return (f"ANGULAR-dominated ({r:.1f}x): the cell is right and the "
                    "orientation is not. Suspect multiple grains; more "
                    "precision will not help.")
        if r < 0.33:
            return (f"RADIAL-dominated ({r:.2f}x): the directions are right and "
                    "the d-spacings are not. Suspect the cell, Lsd or lambda.")
        return (f"mixed ({r:.1f}x): neither part dominates; treat cell and "
                "orientation together.")

    def __str__(self) -> str:
        return (f"radial {self.median_radial_pct:.3f} % | angular "
                f"{self.median_angular_deg:.3f} deg | ratio "
                f"{self.angular_over_radial:.1f}x -> {self.verdict}")


def decompose_residuals(q_observed: np.ndarray,
                        q_predicted: np.ndarray) -> ResidualDecomposition:
    """Split reflection-by-reflection misfit into radial and angular parts.

    Parameters
    ----------
    q_observed, q_predicted : (n, 3) arrays
        Matched scattering vectors, in any consistent convention (1/d or 2π/d —
        both parts are ratios or angles, so the convention cancels).

    Notes
    -----
    Reflections with a zero-length predicted vector are rejected rather than
    silently producing a NaN that would be averaged away.
    """
    qo = np.asarray(q_observed, float)
    qp = np.asarray(q_predicted, float)
    if qo.shape != qp.shape or qo.ndim != 2 or qo.shape[1] != 3:
        raise ValueError(f"need matching (n, 3) arrays, got {qo.shape} and {qp.shape}")
    if len(qo) == 0:
        raise ValueError("no matched reflections")

    no = np.linalg.norm(qo, axis=1)
    np_ = np.linalg.norm(qp, axis=1)
    if np.any(np_ <= 0) or np.any(no <= 0):
        raise ValueError("zero-length scattering vector in the input; drop it "
                         "rather than letting it become a NaN")

    radial_rel = (no - np_) / np_
    cos = np.clip(np.einsum("ij,ij->i", qo, qp) / (no * np_), -1.0, 1.0)
    angular = np.degrees(np.arccos(cos))

    return ResidualDecomposition(
        radial_rel=radial_rel,
        angular_deg=angular,
        displacement_radial=np.abs(no - np_),
        displacement_angular=np_ * np.radians(angular),
        q_mag=np_)
