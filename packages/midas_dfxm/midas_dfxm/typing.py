"""Defect typing: g.b visibility, edge/screw character, Burgers-vector recovery.

Phase 4 of ``implementation_plan.md`` — the emphasized goal.

DFXM discriminates dislocation types through two classical handles, both of which
fall directly out of the Stroh distortion field (:mod:`midas_dfxm.dislocation`):

* **g.b invisibility.** A dislocation's contrast in reflection ``g`` scales with the
  projection of its displacement field onto ``g``. For a screw (``u ∥ b``) this is
  ``g.b``; the dislocation goes (nearly) dark when ``g.b = 0``. Scanning a set of
  reflections and reading off which extinguish pins the Burgers-vector direction.
* **Edge vs screw dilatation.** A screw distortion is pure shear (zero dilatation
  ``tr(eps)``); an edge produces a tension/compression dipole (nonzero dilatation).
  The dilatation-to-shear ratio of the recovered field is a character discriminant.

``g.b`` is a frame-independent integer for Miller inputs (``g`` reciprocal ``(hkl)``,
``b`` direct ``[uvw]`` ⇒ ``g.b = h u + k v + l w``), so the visibility bookkeeping
needs no metric. Contrast itself is measured from the differentiable forward.

Everything is torch-differentiable and device-portable.
"""
from __future__ import annotations

from typing import Sequence

import torch

from .conventions import GoniometerSetting
from .dislocation import StrohDislocation, dislocation_deformation_field
from .field import DeformationField
from .forward import voxel_intensity
from .resolution import ResolutionFunction, aligned_resolution
from .scan import reference_q_nom


def g_dot_b(g_hkl: Sequence[int], burgers_uvw: Sequence[int]) -> float:
    """Frame-independent ``g.b = h u + k v + l w`` for the invisibility criterion.

    Zero (within rounding) ⇒ the screw component is invisible in reflection ``g``.
    """
    g = torch.as_tensor(g_hkl, dtype=torch.float64)
    b = torch.as_tensor(burgers_uvw, dtype=torch.float64)
    return float(g @ b)


def dilatation_ratio(dislocation: StrohDislocation, positions: torch.Tensor) -> torch.Tensor:
    """Dilatation-to-total distortion ratio of a dislocation's field.

    ``mean|tr(beta)| / mean||beta||``. Near 0 for a screw (pure shear), large for an
    edge (tension/compression dipole). Differentiable scalar.
    """
    beta = dislocation.displacement_gradient(positions)
    dil = (beta[:, 0, 0] + beta[:, 1, 1] + beta[:, 2, 2]).abs().mean()
    mag = beta.abs().mean()
    return dil / (mag + 1e-30)


def classify_character(
    dislocation: StrohDislocation,
    positions: torch.Tensor,
    *,
    screw_threshold: float = 0.05,
    edge_threshold: float = 0.3,
) -> str:
    """Label a dislocation ``'screw'`` / ``'edge'`` / ``'mixed'`` by dilatation ratio."""
    r = float(dilatation_ratio(dislocation, positions))
    if r < screw_threshold:
        return "screw"
    if r > edge_threshold:
        return "edge"
    return "mixed"


def reflection_signal(field: DeformationField, hkl) -> torch.Tensor:
    """Geometry-independent visibility signal: ``std(|Q(r) - G0|) / |G0|``.

    The DFXM observable is ``ΔQ = -grad(g.u)``, whose magnitude scales cleanly with
    ``g.b`` regardless of goniometer geometry — unlike a single-setting image
    contrast, which also depends on the angle between the reflection and the rocking
    axis. This is the right handle for the invisibility criterion and Burgers
    recovery. Differentiable scalar.
    """
    G0 = field.reference_G(hkl)
    dQ = field.local_Q(hkl) - G0
    return dQ.norm(dim=-1).std() / (G0.norm() + 1e-30)


def dfxm_contrast(
    field: DeformationField,
    hkl,
    *,
    resolution: ResolutionFunction | None = None,
    goniometer: GoniometerSetting | None = None,
    rocking_offset_deg: float = 0.0,
    sigma_par: float = 5e-3,
    sigma_perp: float = 5e-3,
) -> torch.Tensor:
    """Weak-beam contrast of a field in reflection ``hkl`` (coefficient of variation).

    A perfect crystal gives spatially uniform intensity (contrast ~0); a dislocation
    modulates the local scattering vector and lights up as ``std(I)/mean(I)`` over the
    voxels. ``rocking_offset_deg`` moves onto the resolution flank (true weak-beam,
    first-order sensitivity); 0 uses the aligned setting. Differentiable scalar.
    """
    center = goniometer or GoniometerSetting()
    if resolution is None:
        q_nom = reference_q_nom(field, hkl, center)
        resolution = aligned_resolution(q_nom, sigma_par=sigma_par, sigma_perp=sigma_perp)
    setting = GoniometerSetting(
        mu=center.mu, omega=center.omega,
        chi=center.chi + rocking_offset_deg, phi=center.phi,
    )
    inten = voxel_intensity(field, hkl, setting, resolution)
    mean = inten.mean()
    std = inten.std()
    return std / (mean + 1e-30)


def visibility_series(
    dislocation: StrohDislocation,
    positions: torch.Tensor,
    reflections: Sequence[Sequence[int]],
    *,
    mode: str = "intrinsic",
    rocking_offset_deg: float = 0.05,
    reference_orientation=None,
    lattice_params=(3.6356, 3.6356, 3.6356, 90.0, 90.0, 90.0),
) -> dict:
    """Visibility of one dislocation across a set of reflections.

    Returns ``{hkl_tuple: signal_float}``. ``mode='intrinsic'`` (default) uses the
    geometry-independent ``std(|ΔQ|)/|G0|`` (:func:`reflection_signal`) — the clean
    ``g.b`` handle for Burgers recovery. ``mode='image'`` uses the single-setting
    weak-beam image contrast (:func:`dfxm_contrast`), which also carries rocking
    geometry. Reflections that extinguish are the invisibility conditions.
    """
    field = dislocation_deformation_field(
        positions, dislocation,
        reference_orientation=reference_orientation, lattice_params=lattice_params,
    )
    out = {}
    for hkl in reflections:
        if mode == "intrinsic":
            c = reflection_signal(field, tuple(hkl))
        elif mode == "image":
            c = dfxm_contrast(field, tuple(hkl), rocking_offset_deg=rocking_offset_deg)
        else:
            raise ValueError("mode must be 'intrinsic' or 'image'")
        out[tuple(int(x) for x in hkl)] = float(c)
    return out


def recover_burgers(
    contrasts: dict,
    candidate_burgers: Sequence[Sequence[int]],
    *,
    visible_frac: float = 0.25,
) -> list:
    """Rank candidate Burgers vectors by consistency with a visibility pattern.

    A candidate ``b`` is consistent when every *invisible* reflection (contrast below
    ``visible_frac`` of the max) satisfies ``g.b = 0`` and visible reflections do not.
    Returns candidates sorted best-first as ``[(burgers_tuple, mismatch_count), ...]``.
    """
    cmax = max(contrasts.values()) if contrasts else 1.0
    thresh = visible_frac * cmax
    ranked = []
    for b in candidate_burgers:
        mismatch = 0
        for hkl, c in contrasts.items():
            invisible_obs = c < thresh
            invisible_pred = abs(g_dot_b(hkl, b)) < 1e-9
            if invisible_obs != invisible_pred:
                mismatch += 1
        ranked.append((tuple(int(x) for x in b), mismatch))
    ranked.sort(key=lambda t: t[1])
    return ranked
