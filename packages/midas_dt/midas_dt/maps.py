"""Turning fit-parameter maps into physical ones.

A peak centre in detector pixels becomes a d-spacing through the detector
geometry, and a d-spacing becomes a strain against a reference. Both steps are
short; what matters is being explicit about what the result *is*, because the
numbers look like a strain map whether or not they mean one.

What a single-channel strain map is
-----------------------------------
The shift of one ring at one azimuth measures the strain component along that
scattering vector -- one number, not the tensor. With a single (R, eta)
channel you have one projection of a rank-2 tensor per voxel, and calling it
"the strain" is an over-claim. Several eta bins give several components and
the tensor can be fitted; :func:`strain_map` therefore reports which case it
is in and refuses to pretend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .branches import BranchResult
from .geometry import DTGeometry

__all__ = [
    "radius_to_d_spacing",
    "radius_to_two_theta",
    "d_spacing_map",
    "strain_map",
    "phase_fraction_map",
]

log = logging.getLogger(__name__)


def radius_to_two_theta(radius_px, geometry: DTGeometry) -> np.ndarray:
    """Scattering angle 2theta (degrees) for a detector radius in pixels."""
    r = np.asarray(radius_px, dtype=np.float64)
    r_um = r * geometry.px_um
    return np.degrees(np.arctan2(r_um, geometry.lsd_um))


def radius_to_d_spacing(radius_px, geometry: DTGeometry) -> np.ndarray:
    """d-spacing (angstrom) for a detector radius, via Bragg's law.

    ``tan(2theta) = R.px / Lsd`` then ``d = lambda / (2 sin theta)``.

    The small-angle approximation is NOT used: at 90.5 keV and Lsd ~ 1.07 m a
    radius of 500 px is 2theta ~ 4.6 deg, where ``sin theta ~ theta`` is good
    to a part in 10^3 -- comfortably larger than the strains being measured.
    """
    two_theta = np.radians(radius_to_two_theta(radius_px, geometry))
    sin_theta = np.sin(two_theta / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = geometry.wavelength_a / (2.0 * sin_theta)
    return np.where(sin_theta > 0, d, np.nan)


def d_spacing_map(result: BranchResult, geometry: DTGeometry, *,
                  output: str = "RMEAN") -> np.ndarray:
    """Per-voxel d-spacing from a fitted peak-centre map."""
    if output not in result.maps:
        raise KeyError(
            f"{output!r} is not in this result; available: "
            f"{', '.join(sorted(result.maps))}"
        )
    if result.linearity.get(output) == "approximate":
        log.warning(
            "%s was back-projected directly (weighting='none'), so it is not a "
            "physically meaningful per-voxel quantity. The d-spacing map "
            "inherits that.", output,
        )
    return radius_to_d_spacing(result.maps[output], geometry)


@dataclass
class StrainMap:
    """A strain map, with what it actually represents attached."""

    strain: np.ndarray
    d0_a: float
    n_eta_bins: int
    component: str          # 'scalar-projection' or 'multi-component'

    @property
    def is_tensor(self) -> bool:
        return False        # never, from this module

    def caveats(self) -> list[str]:
        out = [
            f"Strain is the component along the scattering vector, not the "
            f"tensor. This map is one projection per voxel "
            f"({self.n_eta_bins} eta bin(s) available)."
        ]
        if self.n_eta_bins == 1:
            out.append(
                "With a single eta bin the tensor is not recoverable at all -- "
                "use several azimuthal bins if you need more than one component."
            )
        return out


def strain_map(result: BranchResult, geometry: DTGeometry, *,
               d0_a: float | None = None, output: str = "RMEAN",
               d0_percentile: float = 50.0) -> StrainMap:
    """Per-voxel strain, ``(d - d0) / d0``.

    Parameters
    ----------
    d0_a : float, optional
        Unstrained reference d-spacing. If omitted, the *median* d over the
        map is used and the result becomes a **relative** strain map: its zero
        is the sample's own median, not a physical unstrained state. That is
        often what is wanted for contrast, and is never what should be quoted
        as an absolute strain, so the choice is recorded.
    d0_percentile : float
        Percentile used for the fallback reference. 50 (median) is robust to a
        minority phase; change it only with a reason.
    """
    d = d_spacing_map(result, geometry, output=output)
    finite = np.isfinite(d)
    if not finite.any():
        raise ValueError("no finite d-spacings: the peak-centre map is empty")

    if d0_a is None:
        d0_a = float(np.nanpercentile(d[finite], d0_percentile))
        log.warning(
            "no d0 given; using the map's own %g-th percentile (%.6f A). The "
            "result is a RELATIVE strain map -- its zero is this sample's "
            "median, not an unstrained reference.", d0_percentile, d0_a,
        )
    if d0_a <= 0:
        raise ValueError(f"d0 must be positive, got {d0_a}")

    strain = (d - d0_a) / d0_a
    n_eta = result.channel.n_eta
    return StrainMap(strain=strain, d0_a=float(d0_a), n_eta_bins=int(n_eta),
                     component="scalar-projection")


def phase_fraction_map(results: dict[str, BranchResult], *,
                       output: str = "TotalIntensityBackgroundCorr",
                       ) -> dict[str, np.ndarray]:
    """Relative phase fractions from per-phase integrated intensities.

    Each entry of *results* is one phase's channel (typically a reflection
    unique to it). The maps are normalised to sum to one per voxel.

    **This is not a quantitative phase analysis.** Diffracted intensity
    depends on structure factor, multiplicity, Lorentz-polarisation and
    absorption, none of which is corrected here -- so these are *relative*
    fractions, comparable between voxels of the same map and not between
    phases. A quantitative result needs those corrections plus a
    self-absorption correction; see the known-limits ledger.

    Only additive outputs are accepted, since the arithmetic assumes
    intensities add.
    """
    from .conventions import is_additive

    if not is_additive(output):
        raise ValueError(
            f"{output!r} does not add along a ray, so it cannot be used as an "
            f"intensity weight. Use one of TotalIntensity, "
            f"TotalIntensityBackgroundCorr or FitIntegratedIntensity."
        )
    if len(results) < 2:
        raise ValueError(
            f"phase fractions need at least 2 phases, got {len(results)}"
        )

    stacks, names = [], []
    for name, res in results.items():
        if output not in res.maps:
            raise KeyError(f"phase {name!r} has no {output!r} map")
        stacks.append(np.clip(res.maps[output], 0.0, None))
        names.append(name)

    arr = np.stack(stacks)
    total = np.nansum(arr, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(total > 0, arr / total, np.nan)
    log.info("relative phase fractions for %s (uncorrected for structure "
             "factor, LP and absorption)", ", ".join(names))
    return {n: frac[i] for i, n in enumerate(names)}
