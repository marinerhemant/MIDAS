"""MD-coupled differentiable diffraction: scatter straight from atomic coordinates.

Two forward paths, both autograd-differentiable **w.r.t. the atomic coordinates
themselves** (this is the novel hook -- gradients flow from a measured pattern
back to per-atom positions / displacement statistics, and onward to an MD force
field):

* :func:`coherent_amplitude` -- oriented, complex
      A(Q) = sum_i  f_i(|Q|) exp(i Q . r_i),     I = |A|^2
  Sampling ``Q`` along c* reproduces the analytic shape-function fringes of
  Phase 1 (see the cross-check test) -- but here the fringes, the thermal
  Debye-Waller falloff, and any *anisotropic* disordering all emerge from the
  coordinates rather than from a phenomenological factor.

* :func:`debye_intensity` -- orientationally averaged (the colloidal-solution
  case for the Schaller/Flanders nanoplatelets)
      I(q) = sum_i sum_j  f_i(q) f_j(q) sinc(q r_ij)

:func:`ensemble_intensity` averages the incoherent intensity over MD frames;
the spread of atomic positions across the trajectory *is* the disorder, so the
anisotropic out-of-plane vs in-plane Debye-Waller falloff is reproduced with no
DWF assumption at all.

Form factors come from ``midas_hkls`` (Cromer-Mann), evaluated at the physical
``s = |Q| / (4 pi)``.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

from midas_hkls.form_factors import form_factor_batch

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "atomic_form_factors",
    "coherent_amplitude",
    "coherent_intensity",
    "debye_intensity",
    "ensemble_intensity",
]

_FOUR_PI = 4.0 * math.pi


def atomic_form_factors(elements, q_mag):
    """Per-atom Cromer-Mann form factors at scattering magnitude ``q_mag`` (1/A).

    ``q_mag`` may have any shape; returns shape ``(*q_mag.shape, M)`` with M the
    number of atoms in ``elements``.
    """
    import torch
    q_mag = torch.as_tensor(q_mag)
    s2 = (q_mag / _FOUR_PI) ** 2                       # s = |Q|/(4 pi)
    return form_factor_batch(s2, list(elements))       # (..., M)


def coherent_amplitude(coords, elements, q_vec):
    """Complex coherent amplitude ``A(Q) = sum_i f_i(|Q|) exp(i Q . r_i)``.

    Parameters
    ----------
    coords : tensor (M, 3)
        Cartesian atomic coordinates (Angstrom).  Differentiable.
    elements : sequence[str], length M
    q_vec : tensor (..., 3)
        Scattering vectors ``Q = 2 pi (h a* + k b* + l c*)`` in 1/A.

    Returns
    -------
    tensor (...) complex
    """
    import torch
    coords = torch.as_tensor(coords)
    q_vec = torch.as_tensor(q_vec, dtype=coords.dtype, device=coords.device)
    qmag = torch.linalg.vector_norm(q_vec, dim=-1)             # (...)
    f = atomic_form_factors(elements, qmag)                    # (..., M)
    phase = q_vec @ coords.T                                   # (..., M)
    re = (f * torch.cos(phase)).sum(dim=-1)
    im = (f * torch.sin(phase)).sum(dim=-1)
    return torch.complex(re, im)


def coherent_intensity(coords, elements, q_vec):
    """``|A(Q)|^2`` for the coherent (oriented) path."""
    A = coherent_amplitude(coords, elements, q_vec)
    return A.real * A.real + A.imag * A.imag


def debye_intensity(coords, elements, q_mag, *, tile=2048):
    """Orientationally-averaged intensity via the Debye scattering equation.

        I(q) = sum_i sum_j f_i(q) f_j(q) sinc(q r_ij),   sinc(x)=sin(x)/x

    Parameters
    ----------
    coords : tensor (M, 3)
        Cartesian coordinates (A).  Differentiable.
    elements : sequence[str], length M
    q_mag : tensor (Q,)
        Scattering-vector magnitudes |Q| (1/A).
    tile : int
        Row-tile size over atoms to bound memory (the pair matrix is M x M).

    Returns
    -------
    tensor (Q,)
    """
    import torch
    coords = torch.as_tensor(coords)
    q_mag = torch.as_tensor(q_mag, dtype=coords.dtype, device=coords.device)
    M = coords.shape[0]
    f = atomic_form_factors(elements, q_mag)                   # (Q, M)

    out = torch.zeros(q_mag.shape, dtype=coords.dtype, device=coords.device)
    for start in range(0, M, tile):
        stop = min(start + tile, M)
        ci = coords[start:stop]                                # (b, 3)
        # pairwise distances block (b, M)
        rij = torch.cdist(ci, coords)                          # (b, M)
        qr = q_mag[:, None, None] * rij[None, :, :]            # (Q, b, M)
        sinc = torch.sinc(qr / math.pi)                        # sin(qr)/(qr), =1 at 0
        fi = f[:, start:stop]                                  # (Q, b)
        # sum_j f_j sinc  -> (Q, b); then weight by f_i and sum over i
        contrib = (fi * torch.einsum('qm,qbm->qb', f, sinc)).sum(dim=-1)
        out = out + contrib
    return out


def ensemble_intensity(frames, elements, q, *, coherent=False, q_is_vector=None,
                       tile=2048):
    """Average intensity over a stack of MD frames (the disorder = the spread).

    Parameters
    ----------
    frames : tensor (F, M, 3)
        F snapshots of M atoms (Angstrom).
    elements : sequence[str], length M
    q : tensor
        ``(Q, 3)`` Q-vectors if ``coherent`` (or ``q_is_vector``), else ``(Q,)``
        magnitudes for the Debye path.
    coherent : bool
        If True, average ``|A(Q)|^2`` over frames (oriented).  Else average the
        Debye powder intensity.

    Returns
    -------
    tensor (Q,)
        Frame-averaged intensity.
    """
    import torch
    frames = torch.as_tensor(frames)
    if frames.dim() == 2:
        frames = frames.unsqueeze(0)
    use_vec = coherent if q_is_vector is None else q_is_vector

    acc = None
    for fr in frames:
        if use_vec:
            I = coherent_intensity(fr, elements, q)
        else:
            I = debye_intensity(fr, elements, q, tile=tile)
        acc = I if acc is None else acc + I
    return acc / frames.shape[0]
