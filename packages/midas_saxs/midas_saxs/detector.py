"""Render source terms onto a 2-D SAXS detector, and reduce frames.

This is the module that makes the point of the whole exercise visible. A
dislocation loop and a void of the same relaxation volume are **identical** in a
radial average and **different** on the 2-D frame: the void is isotropic, the
loop varies between ``kappa dV`` in its own plane and ``dV`` along its normal.
Azimuthally averaging a simulated frame throws away exactly the information the
simulation was built to produce, so :func:`radial_average` is offered for
comparison against measurement, never as the primary output.

Everything is torch-differentiable in the geometry and in the source-term
parameters, which is the deliverable: an ML inversion trains against this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .geometry import SAXSGeometry, pixel_to_q

__all__ = [
    "Frame",
    "azimuthal_profile",
    "radial_average",
    "simulate_frame",
]

#: Classical electron radius, angstroms. Turns electrons^2 into a cross-section.
R_E_A = 2.8179403262e-5


@dataclass
class Frame:
    """A simulated detector frame plus what went into it."""

    intensity: torch.Tensor                    # (Z, Y), arbitrary or absolute units
    geometry: SAXSGeometry
    q: torch.Tensor                            # (Z, Y, 3), 1/A
    mask: torch.Tensor                         # (Z, Y) bool, True = usable
    components: Dict[str, torch.Tensor] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def q_magnitude(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.q, dim=-1)

    def __repr__(self) -> str:                 # pragma: no cover - cosmetic
        return (f"Frame({tuple(self.intensity.shape)}, "
                f"components={list(self.components)}, "
                f"{int((~self.mask).sum())} masked px)")


def simulate_frame(
    geom: SAXSGeometry,
    *,
    network=None,
    stiffness=None,
    electron_density_e_per_A3: Optional[float] = None,
    particles: Sequence = (),
    sample_volume_A3: float = 1.0,
    incoherent_loops: bool = True,
    absolute_units: bool = False,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> Frame:
    """Simulate a transmission-SAXS frame from a dislocation network and/or particles.

    Parameters
    ----------
    geom
        Detector geometry. Its ``beamstop_radius_px`` masks the direct beam.
    network, stiffness, electron_density_e_per_A3
        The dislocation source term (:mod:`midas_saxs.strain_source`). All three
        are needed together; omit them for a particles-only frame.
    particles
        :class:`midas_saxs.particles.SpherePopulation` instances -- voids,
        bubbles, precipitates. Their intensities are per A^3 of sample and are
        scaled by ``sample_volume_A3``.
    incoherent_loops
        Sum loop intensities rather than amplitudes. Default True, which is what
        a conventional SAXS beam -- much larger than the inter-loop spacing --
        actually measures. Set False for the coherent (speckle) case.
    absolute_units
        Multiply by ``r_e^2`` to get a differential cross-section rather than
        electrons².

    Returns
    -------
    Frame
        ``.intensity`` is the total; ``.components`` holds each contribution
        separately, which is how you check whether the loops are visible above
        the voids at all.
    """
    rows, cols = geom.pixel_grid(dtype=dtype, device=device)
    q = pixel_to_q(rows, cols, geom, dtype=dtype, device=device)     # (Z, Y, 3)
    shape = q.shape[:2]
    mask = geom.beamstop_mask(dtype=dtype, device=device)

    # Evaluate source terms ONLY on unmasked pixels. Two reasons, both hard:
    # the beam centre has q = 0 exactly, where the acoustic tensor is singular
    # and the kernel (rightly) refuses; and a beamstop typically hides 1-5 % of
    # a frame, so skipping it is free speed.
    flat_mask = mask.reshape(-1)
    q_flat = q.reshape(-1, 3)[flat_mask]
    if bool((torch.linalg.vector_norm(q_flat, dim=-1) == 0).any()):
        raise ValueError(
            "an unmasked pixel sits at exactly q = 0 (the direct beam), where the "
            "scattering kernel is singular. Set `beamstop_radius_px` on the "
            "geometry -- a real SAXS measurement always has one, because the "
            "direct beam is orders of magnitude above the signal.")

    total = torch.zeros(q_flat.shape[0], dtype=dtype, device=device)
    components: Dict[str, torch.Tensor] = {}
    warnings: List[str] = []

    def _unflatten(v):
        """Scatter a masked-pixel vector back onto the full panel, zeros elsewhere."""
        full = torch.zeros(shape.numel() if hasattr(shape, "numel") else
                           shape[0] * shape[1], dtype=v.dtype, device=v.device)
        full[flat_mask] = v
        return full.reshape(shape)

    if network is not None:
        if stiffness is None or electron_density_e_per_A3 is None:
            raise ValueError(
                "a dislocation network needs both `stiffness` and "
                "`electron_density_e_per_A3`; without them there is no way to turn "
                "q.u~ into an amplitude")
        from .strain_source import loop_amplitude

        out, res = loop_amplitude(
            network, q_flat, stiffness,
            electron_density_e_per_A3=electron_density_e_per_A3,
            incoherent=incoherent_loops, return_result=True)
        I_loops = out if incoherent_loops else out.abs() ** 2
        components["loops"] = _unflatten(I_loops)
        total = total + I_loops
        warnings.extend(res.warnings)
        if res.n_loops == 0:
            warnings.append(
                "the network contains no closed loops, so the dislocation term is "
                "identically zero. Open lines have no relaxation volume; this is "
                "physics, not a failure, but the frame shows only the particles.")

    for k, pop in enumerate(particles):
        I_p = pop.intensity(q_flat) * sample_volume_A3
        name = pop.label or f"particles_{k}"
        components[name] = _unflatten(I_p)
        total = total + I_p

    if absolute_units:
        total = total * R_E_A ** 2
        components = {k: v * R_E_A ** 2 for k, v in components.items()}

    intensity = _unflatten(total)

    if not components:
        warnings.append("no source terms were given; the frame is identically zero.")

    return Frame(intensity=intensity, geometry=geom, q=q, mask=mask,
                 components=components, warnings=warnings)


def radial_average(
    frame: Frame,
    *,
    n_bins: int = 200,
    q_range: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Azimuthally average a frame to ``I(|q|)``.

    Returns ``(q_centres, I, n_pixels)``.

    Use this to compare against a measured 1-D curve -- but be aware it discards
    the loop-versus-void anisotropy, which is the one thing a 2-D simulation buys
    you. :func:`azimuthal_profile` is the complementary reduction that keeps it.
    """
    qm = frame.q_magnitude[frame.mask]
    I = frame.intensity[frame.mask]
    lo, hi = q_range if q_range is not None else (float(qm.min()), float(qm.max()))
    edges = torch.linspace(lo, hi, n_bins + 1, dtype=qm.dtype, device=qm.device)
    idx = torch.bucketize(qm, edges) - 1
    keep = (idx >= 0) & (idx < n_bins)
    idx, I, qm = idx[keep], I[keep], qm[keep]

    counts = torch.zeros(n_bins, dtype=qm.dtype, device=qm.device)
    counts.index_add_(0, idx, torch.ones_like(I))
    sums = torch.zeros(n_bins, dtype=qm.dtype, device=qm.device)
    sums.index_add_(0, idx, I)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, sums / counts.clamp(min=1.0), counts


def azimuthal_profile(
    frame: Frame,
    *,
    q_centre_inv_A: float,
    q_width_inv_A: float,
    n_bins: int = 72,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``I(azimuth)`` in a thin ``|q|`` ring. Returns ``(azimuth_deg, I, n_px)``.

    This is the reduction that *keeps* the anisotropy. A void population gives a
    flat profile; a population of aligned loops gives a modulated one, and the
    modulation depth is set by ``kappa = lambda/(lambda + 2 mu)``. Comparing the
    two is the loop-versus-void measurement.
    """
    qm = frame.q_magnitude
    sel = frame.mask & (torch.abs(qm - q_centre_inv_A) <= 0.5 * q_width_inv_A)
    if not bool(sel.any()):
        raise ValueError(
            f"no unmasked pixels within {q_width_inv_A:g} of |q| = "
            f"{q_centre_inv_A:g} 1/A. The panel spans "
            f"{frame.geometry.q_range_inv_A[0]:.3g} to "
            f"{frame.geometry.q_range_inv_A[1]:.3g} 1/A.")

    # Azimuth in the detector plane, measured from +y (horizontal) toward +z.
    qy = frame.q[..., 1][sel]
    qz = frame.q[..., 2][sel]
    az = torch.rad2deg(torch.atan2(qz, qy)) % 360.0
    I = frame.intensity[sel]

    edges = torch.linspace(0.0, 360.0, n_bins + 1, dtype=az.dtype, device=az.device)
    idx = torch.bucketize(az, edges) - 1
    idx = idx.clamp(0, n_bins - 1)
    counts = torch.zeros(n_bins, dtype=az.dtype, device=az.device)
    counts.index_add_(0, idx, torch.ones_like(I))
    sums = torch.zeros(n_bins, dtype=az.dtype, device=az.device)
    sums.index_add_(0, idx, I)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, sums / counts.clamp(min=1.0), counts
