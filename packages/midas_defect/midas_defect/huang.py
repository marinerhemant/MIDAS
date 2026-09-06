"""Near-Bragg diffuse (Huang) scattering from a dislocation network.

The same Fourier kernel that :mod:`midas_saxs` evaluates at ``q``, evaluated at
``G + q`` instead. Kinematically,

    A(G + q) ~ -i F_G rho_e (G + q) . u~(q),

so the whole near-Bragg diffuse field of a dislocation microstructure is again
the scalar :func:`midas_ddd.q_dot_u_tilde`, contracted against ``G + q`` rather
than ``q``. Nothing about the elasticity is re-derived here.

Why this is the stronger measurement, and by exactly how much
-------------------------------------------------------------
The small-angle amplitude carries ``q . u~``; the near-Bragg amplitude carries
``(G + q) . u~``. With ``|G| ~ 3 1/A`` against a small-angle ``|q| ~ 1e-3 to
1e-1 1/A``, the naive expectation is a factor ``(G/q)^2``, i.e. 1e3 to 1e7.

Measured over 400 random q directions for a single prismatic loop
(:func:`huang_vs_small_angle_ratio`), the picture is more specific than that:

===========  ==============  ==============  ==============
|q| (1/A)    max over dirs   median          naive (G/q)^2
===========  ==============  ==============  ==============
1e-3         9.4e6           5.2e5           9.4e6
1e-2         9.5e4           5.2e3           9.4e4
1e-1         1.0e3           5.7e1           9.4e2
===========  ==============  ==============  ==============

So ``(G/q)^2`` is an **upper bound**, attained almost exactly in the most
favourable direction; the typical gain is about 18x smaller. And in special
directions -- where ``G . u~`` nearly vanishes while ``q . u~`` does not -- the
ratio drops below 1 and small-angle is momentarily the better probe. Quote the
median, not the bound, when advising on beamtime.

The advice still holds overwhelmingly: for defect-type discrimination in a
single crystal, near-Bragg beats small-angle by two to five orders of magnitude
in a typical direction. Small-angle wins when you cannot reach a Bragg peak, or
when the sample is not a single crystal.

What this does NOT give you: interstitial/vacancy typing
--------------------------------------------------------
It is widely known that the asymmetry of diffuse scattering between the two
sides of a Bragg peak carries defect character. **This model does not reproduce
that**, and the reason is worth stating so nobody builds on it.

At first order in the displacement the intensity is ``|(G + q) . u~(q)|^2``.
Flipping a loop from interstitial to vacancy flips the sign of ``u~``, which
leaves ``|...|^2`` untouched. There *is* a strong q/-q asymmetry here -- it
comes from the geometric difference between ``G + q`` and ``G - q``, and reaches
tens of percent -- but it is identical for the two characters, verified in
``test_huang.py::test_asymmetry_does_not_distinguish_interstitial_from_vacancy``.

Recovering the character needs a term that is *linear* in the defect strength,
i.e. interference between the long-range strain field (odd in q) and the
localized core / Laue scattering from the extra or missing atoms (even in q).
The core term is not modelled here. Do not read defect character off this
forward model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import math

import torch

__all__ = [
    "HuangResult",
    "huang_amplitude",
    "huang_intensity",
    "huang_vs_small_angle_ratio",
    "qG_asymmetry",
]

_DDD_HINT = (
    "midas_ddd is required for the near-Bragg diffuse forward but is not "
    "installed:  pip install midas-ddd"
)


def _require_ddd():
    try:
        from midas_ddd.fourier import u_tilde  # noqa: F401
    except ImportError as exc:                     # pragma: no cover - env dependent
        raise ImportError(_DDD_HINT) from exc


@dataclass
class HuangResult:
    """Diffuse amplitude near one reflection, plus what the kernel could account for."""

    amplitude: torch.Tensor          # (Q,) complex
    q: torch.Tensor                  # (Q, 3) deviation from G, inverse µm
    G: torch.Tensor                  # (3,) reflection, inverse µm
    n_loops: int
    n_segments_ignored: int
    warnings: List[str] = field(default_factory=list)

    @property
    def intensity(self) -> torch.Tensor:
        return self.amplitude.abs() ** 2

    def __repr__(self) -> str:       # pragma: no cover - cosmetic
        return (f"HuangResult({self.q.shape[0]} q-points, {self.n_loops} loops, "
                f"|G|={float(torch.linalg.norm(self.G)):.4g} 1/um)")


def huang_amplitude(
    net,
    G_inv_um: torch.Tensor,
    q_inv_um: torch.Tensor,
    C6: torch.Tensor,
    *,
    electron_density_e_per_A3: float = 1.0,
    structure_factor: complex = 1.0,
) -> HuangResult:
    """Diffuse amplitude ``A(G + q)`` for the closed loops in ``net``.

    Parameters
    ----------
    net : midas_ddd.DislocationNetwork
    G_inv_um : (3,) tensor
        Reflection, **inverse micrometers** -- the same units the kernel uses for
        ``q``, because ``G`` and ``q`` are added. A Cu 111 at 3.07 1/A is
        ``3.07e4`` 1/um. Getting this wrong by 1e4 is the easy mistake here.
    q_inv_um : (Q, 3) tensor
        Deviation from ``G``. Must not be exactly zero (that is the Bragg peak,
        where the kernel is singular and the diffuse expansion does not apply).
    C6 : (6, 6) tensor
        Voigt stiffness.
    structure_factor
        ``F_G`` for the reflection. Left at 1 the result is relative.

    Notes
    -----
    Uses ``(G + q)``, not ``G``. The difference is O(q/G) in amplitude and is
    exactly what produces the q/-q asymmetry -- dropping it would make the
    diffuse field artificially symmetric.
    """
    _require_ddd()
    from midas_ddd.fourier import u_tilde

    dt = net.nodes_um.dtype
    G = torch.as_tensor(G_inv_um, dtype=dt, device=net.nodes_um.device).reshape(3)
    q = torch.as_tensor(q_inv_um, dtype=dt, device=net.nodes_um.device)
    if q.ndim == 1:
        q = q.unsqueeze(0)

    res = u_tilde(net, q, C6)
    K = (G.unsqueeze(0) + res.q).to(res.u_tilde.dtype)
    scale = electron_density_e_per_A3 * 1.0e12          # e/A^3 -> e/um^3
    amp = -1j * complex(structure_factor) * scale * torch.einsum(
        "qi,qi->q", K, res.u_tilde)

    warnings = list(res.warnings)
    qmax = float(torch.linalg.vector_norm(q, dim=-1).max())
    Gmag = float(torch.linalg.norm(G))
    if Gmag > 0 and qmax > 0.2 * Gmag:
        warnings.append(
            f"|q| reaches {qmax:.3g} 1/um against |G| = {Gmag:.3g} 1/um. The "
            f"near-Bragg expansion assumes |q| << |G|; beyond ~0.2 |G| this is "
            f"no longer 'near' the reflection and neighbouring reflections may "
            f"contribute.")
    return HuangResult(amplitude=amp, q=res.q, G=G, n_loops=res.n_loops,
                       n_segments_ignored=res.n_segments_ignored,
                       warnings=warnings)


def huang_intensity(net, G_inv_um, q_inv_um, C6, **kw) -> torch.Tensor:
    """``|A(G + q)|^2``. ``(Q,)`` real."""
    return huang_amplitude(net, G_inv_um, q_inv_um, C6, **kw).intensity


def qG_asymmetry(
    net,
    G_inv_um: torch.Tensor,
    q_inv_um: torch.Tensor,
    C6: torch.Tensor,
    **kw,
) -> torch.Tensor:
    """``I(G + q) / I(G - q)`` for each ``q``. ``(Q,)`` real.

    A genuine, often large effect -- but see the module docstring: it does
    **not** distinguish interstitial from vacancy loops in this model, because
    the intensity depends on the square of the defect strength. It measures the
    geometry of the strain field, not the sign of the relaxation volume.
    """
    q = torch.as_tensor(q_inv_um, dtype=net.nodes_um.dtype,
                        device=net.nodes_um.device)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    plus = huang_intensity(net, G_inv_um, q, C6, **kw)
    minus = huang_intensity(net, G_inv_um, -q, C6, **kw)
    return plus / minus.clamp(min=1e-300)


def huang_vs_small_angle_ratio(
    net,
    G_inv_um: torch.Tensor,
    q_inv_um: torch.Tensor,
    C6: torch.Tensor,
) -> torch.Tensor:
    """``I(G + q) / I(q)`` -- near-Bragg over small-angle, same defect, same q.

    The number that decides which experiment to do. Expected to scale as
    ``(G/q)^2``, so 1e4 to 1e7 across a normal small-angle range. Measured, not
    assumed: :func:`midas_defect.huang.huang_vs_small_angle_ratio` divides two
    evaluations of one kernel.
    """
    _require_ddd()
    from midas_ddd.fourier import u_tilde

    q = torch.as_tensor(q_inv_um, dtype=net.nodes_um.dtype,
                        device=net.nodes_um.device)
    if q.ndim == 1:
        q = q.unsqueeze(0)

    # Both sides from ONE evaluation of u~, contracted against (G+q) and against
    # q, with no electron density and no structure factor on either. Those are
    # common prefactors that cancel in a same-material comparison -- and applying
    # the density to only one side (which an earlier version did, via
    # huang_intensity) inflates the ratio by rho^2 * 1e24, i.e. by 24 orders of
    # magnitude. The answer looked like 7e28 instead of 7e4.
    res = u_tilde(net, q, C6)
    G = torch.as_tensor(G_inv_um, dtype=net.nodes_um.dtype,
                        device=net.nodes_um.device).reshape(3)
    cdt = res.u_tilde.dtype
    near = torch.einsum("qi,qi->q", (G.unsqueeze(0) + res.q).to(cdt),
                        res.u_tilde).abs() ** 2
    small = torch.einsum("qi,qi->q", res.q.to(cdt), res.u_tilde).abs() ** 2
    return near / small.clamp(min=1e-300)
