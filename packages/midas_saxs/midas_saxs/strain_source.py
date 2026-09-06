"""Small-angle amplitude of a dislocation network's strain field.

The small-angle amplitude of a dislocation loop has TWO parts, and taking only
the first inverts the answer:

    A(q) ~ rho_e [ i q.u~(q)  +  b A~(q) ]        (distortion + LAUE)

Ehrhart, Trinkaus & Larson, *Diffuse scattering from dislocation loops*, Phys.
Rev. B **25** (1982) 834, Eq. (8a). The Laue term is the scattering from the
loop's own extra or missing atoms, and at small angle -- where the scattering
vector IS the deviation, ``K = q`` -- it is the same order as the distortion
term, not a correction to it.

Their Eq. (8b) recasts the sum so that it carries a factor ``q x A~(q)``, hence
**the total vanishes identically for q along the loop normal**, precisely where
the distortion term alone is at its maximum ``dV``. The small-q laws are

    distortion only :  dV [ kappa + (1 - kappa) cos^2(theta) ]   <- NOT the answer
    total           :  dV (1 - kappa) sin^2(theta)               <- the answer

which sum to ``dV`` identically. This module uses
:func:`midas_ddd.small_angle_amplitude` (the total). An earlier version used
``q_dot_u_tilde`` (distortion only) and therefore reported a loop as scattering
most strongly along its normal, when in fact it does not scatter there at all.

Near a Bragg peak the distortion term carries a factor ``G`` the Laue term does
not, so ``G/q >> 1`` and :mod:`midas_defect.huang` is correct as it stands --
this is a small-angle-specific correction.

What this does and does not include
-----------------------------------
**Included:** closed dislocation loops -- the irradiation population, with both
the distortion and the Laue term. Their amplitude at ``q -> 0`` is
``rho_e dV (1 - kappa) sin^2(theta)``: zero along the loop normal, maximal in the
loop plane.

**Not included:** open dislocation lines -- the deformation population. They have
no cut surface and no relaxation volume; their small-angle signature is a weak
transverse streak that the kernel does not model. :func:`loop_amplitude` reports
the ignored line length through ``FourierResult`` rather than pretending it
accounted for everything.

**Not included:** voids, bubbles and precipitates. Those are *density* contrast
rather than strain contrast and live in :mod:`midas_saxs.particles`. In an
irradiated material they typically dominate the small-angle image by two to
three orders of magnitude, so a strain-only image will not resemble a
measurement. Use :func:`midas_saxs.detector.simulate_frame` with both.

Units
-----
``q`` arrives in **inverse angstroms** (the SAXS convention, what
:mod:`midas_saxs.geometry` returns) and is converted to inverse micrometers for
the kernel, whose node positions are micrometers. ``q.u~`` comes back in µm^3
and is converted to A^3 so it multiplies an electron density in e/A^3. The
conversions are named calls, never bare literals.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import torch

from .geometry import inv_A_to_inv_um

__all__ = [
    "A3_PER_UM3",
    "electron_density_per_A3",
    "loop_amplitude",
    "loop_intensity",
]

#: 1 um^3 = 1e12 A^3.
A3_PER_UM3 = 1.0e12

_DDD_HINT = (
    "midas_ddd is required for the dislocation source term but is not installed. "
    "It is an optional extra so that plain particle SAXS does not pull a "
    "dislocation-dynamics stack:  pip install 'midas-saxs[dislocations]'"
)


def _require_ddd():
    try:
        import midas_ddd  # noqa: F401
        return midas_ddd
    except ImportError as exc:                    # pragma: no cover - env dependent
        raise ImportError(_DDD_HINT) from exc


def electron_density_per_A3(
    elements: Sequence[str],
    counts: Sequence[float],
    cell_volume_A3: float,
) -> float:
    """Average electron density, e/A^3, from the unit-cell contents.

    ``Z`` per species comes from :mod:`midas_hkls.form_factors` at ``s = 0``,
    where the atomic form factor equals the electron count. Never hard-code a
    Z table; the one in midas-hkls is the single source.

    Examples
    --------
    Copper, FCC, a = 3.615 A, 4 atoms per cell::

        electron_density_per_A3(["Cu"], [4], 3.615 ** 3)   # -> 2.455 e/A^3
    """
    from midas_hkls.form_factors import form_factor

    if len(elements) != len(counts):
        raise ValueError(f"{len(elements)} elements but {len(counts)} counts")
    if cell_volume_A3 <= 0:
        raise ValueError(f"cell_volume_A3 must be positive, got {cell_volume_A3}")
    total_Z = 0.0
    for el, n in zip(elements, counts):
        total_Z += float(n) * float(form_factor(0.0, el))
    return total_Z / float(cell_volume_A3)


def loop_amplitude(
    net,
    q_inv_A: torch.Tensor,
    C6: torch.Tensor,
    *,
    electron_density_e_per_A3: float,
    incoherent: bool = False,
    return_result: bool = False,
):
    """Small-angle scattering amplitude of the closed loops in ``net``.

    Parameters
    ----------
    net : midas_ddd.DislocationNetwork
    q_inv_A : (Q, 3) tensor
        Scattering vectors, **inverse angstroms**.
    C6 : (6, 6) tensor
        Voigt stiffness (only ratios matter).
    electron_density_e_per_A3
        Matrix electron density; see :func:`electron_density_per_A3`.
    incoherent
        ``False`` (default): sum loop amplitudes **coherently**, keeping the
        positional phases. This is what a specific configuration scatters, and
        it produces speckle. ``True``: sum ``|A|^2`` loop by loop, the
        ensemble-averaged result appropriate to a dilute random population under
        a beam much larger than the inter-loop spacing -- which is what a
        conventional SAXS measurement actually reports.

    Returns
    -------
    Complex ``(Q,)`` amplitude in electrons when ``incoherent`` is False; real
    ``(Q,)`` intensity in electrons² when it is True. Pass ``return_result=True``
    to also get the :class:`midas_ddd.FourierResult`, which carries the ignored
    open-line length and the quadrature diagnostic.
    """
    ddd = _require_ddd()
    from midas_ddd.fourier import small_angle_amplitude, u_tilde
    from midas_ddd.validate import find_loops

    q_inv_A = torch.as_tensor(q_inv_A, dtype=net.nodes_um.dtype,
                              device=net.nodes_um.device)
    if q_inv_A.ndim == 1:
        q_inv_A = q_inv_A.unsqueeze(0)
    q_um = inv_A_to_inv_um(q_inv_A)

    loops = find_loops(net)
    scale = electron_density_e_per_A3 * A3_PER_UM3      # e/A^3 * A^3/um^3 -> e/um^3

    # ONE pass, whichever mode: `per_loop=True` returns both the coherent total
    # and the per-loop stack the incoherent sum needs. Asking for them
    # separately evaluated every loop's surface form factor twice, which
    # profiling showed was the entire cost of a frame.
    # `res` is kept only for its diagnostics (ignored open-line length,
    # quadrature qh, loop count); the AMPLITUDE comes from the total, which
    # includes the Laue term.
    res = u_tilde(net, q_um, C6, loops=loops)
    per = small_angle_amplitude(net, q_um, C6, loops=loops, per_loop=True)  # (L,Q)

    if incoherent:
        total = ((scale * per).abs() ** 2).sum(dim=0)
        return (total, res) if return_result else total

    amp = -1j * scale * per.sum(dim=0)
    return (amp, res) if return_result else amp


def loop_intensity(
    net,
    q_inv_A: torch.Tensor,
    C6: torch.Tensor,
    *,
    electron_density_e_per_A3: float,
    incoherent: bool = False,
) -> torch.Tensor:
    """``|A(q)|^2`` for the loop population. ``(Q,)`` real, electrons²."""
    out = loop_amplitude(net, q_inv_A, C6,
                         electron_density_e_per_A3=electron_density_e_per_A3,
                         incoherent=incoherent)
    return out if incoherent else out.abs() ** 2
