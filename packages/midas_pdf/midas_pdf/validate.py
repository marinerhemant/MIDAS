"""Independent references for validating the pipeline end to end.

No external PDF tool (PDFgetX3/Gudrun) or beamline data is required: the Debye
scattering equation gives the *exact* powder-averaged coherent intensity of a
finite cluster of atoms,

    I(Q) = Σ_i Σ_j f_i(Q) f_j(Q) · sin(Q r_ij) / (Q r_ij)

(the i=j terms contribute Σ_i f_i², the self-scattering). Feeding this I(Q)
through ``midas-pdf`` must yield a G(r) whose peaks sit at the true interatomic
distances r_ij — a model-free, physics-level check of the normalization,
conventions, and Fourier transform together. It is also a convenient way to
synthesize realistic test/demo data with a known answer.

Differentiable in the atomic positions (gradients flow through r_ij), so the
same routine doubles as a forward model for structure refinement against G(r).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from midas_hkls import form_factor_batch

__all__ = [
    "debye_scattering_intensity",
    "interatomic_distances",
    "synthetic_powder_image",
]

_FOUR_PI = 4.0 * float(np.pi)


def interatomic_distances(positions: torch.Tensor) -> torch.Tensor:
    """Pairwise distance matrix (N, N) for positions (N, 3) in Å.

    Uses a masked "safe sqrt": ``d(√x)/dx → ∞`` at x=0 would put NaN gradients
    on the (zero) diagonal and contaminate ``positions.grad``. The double-
    ``where`` evaluates √ only on strictly-positive squared distances, so the
    routine stays differentiable in ``positions`` (it is a forward model for
    structure refinement against G(r), so the gradient must be clean).
    """
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)   # (N, N, 3)
    sq = (diff * diff).sum(dim=-1)                           # (N, N)
    positive = sq > 0
    sq_safe = torch.where(positive, sq, torch.ones_like(sq))
    return torch.where(positive, torch.sqrt(sq_safe), torch.zeros_like(sq))


def debye_scattering_intensity(
    q: torch.Tensor | np.ndarray,
    elements: Sequence[str],
    positions: torch.Tensor | np.ndarray,
    *,
    thermal_B: float = 0.0,
) -> torch.Tensor:
    """Powder-averaged coherent X-ray intensity of an atom cluster (Debye).

    Parameters
    ----------
    q :
        Q grid (Å⁻¹), 1-D.
    elements :
        Length-N element symbols, one per atom.
    positions :
        (N, 3) Cartesian coordinates in Å.
    thermal_B :
        Optional isotropic Debye-Waller B (Å²) applied as exp(-B s²) per atom,
        s² = (Q/4π)². Set >0 to broaden/damp the high-Q oscillations.

    Returns
    -------
    I(Q) : torch.Tensor, shape == q.shape. Differentiable in ``positions``.
    """
    q_t = torch.as_tensor(q, dtype=torch.float64)
    pos = torch.as_tensor(positions, dtype=torch.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must be (N, 3)")
    n = pos.shape[0]
    if len(elements) != n:
        raise ValueError("elements length must match positions")

    s2 = (q_t / _FOUR_PI) ** 2                          # (Nq,)
    f = form_factor_batch(s2, list(elements))           # (Nq, N)
    if thermal_B > 0.0:
        f = f * torch.exp(-thermal_B * s2).unsqueeze(-1)

    dist = interatomic_distances(pos)                   # (N, N)
    qr = q_t.view(-1, 1, 1) * dist.view(1, n, n)        # (Nq, N, N)
    sinc = torch.where(qr.abs() < 1e-12, torch.ones_like(qr),
                       torch.sin(qr) / qr)
    # I(Q) = Σ_ij f_i f_j sinc(Q r_ij)
    fij = f.unsqueeze(2) * f.unsqueeze(1)               # (Nq, N, N)
    return (fij * sinc).sum(dim=(1, 2))


def synthetic_powder_image(
    spec,
    q_profile: torch.Tensor | np.ndarray,
    I_profile: torch.Tensor | np.ndarray,
    *,
    counts: float = 5.0e4,
    seed: int = 0,
    flat_detector: bool = True,
) -> torch.Tensor:
    """Render a 2-D powder-diffraction image from a 1-D I(Q) profile.

    Maps each pixel to Q from the ``spec`` geometry, interpolates the profile,
    scales to a peak count level, and draws Poisson counting noise. Used to
    test/demo the pixels → I(Q) front-end with a known answer (no beamline data
    needed). ``flat_detector`` ignores tilts (tx, ty, tz) for the synthetic
    image; the integrator still applies the full calibrated geometry on the way
    back.
    """
    ny, nz = int(spec.NrPixelsY), int(spec.NrPixelsZ)
    bc_y, bc_z = float(spec.BC_y), float(spec.BC_z)
    px, lsd, lam = float(spec.pxY), float(spec.Lsd), float(spec.Wavelength)

    yy, zz = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
    R_px = np.sqrt((yy - bc_y) ** 2 + (zz - bc_z) ** 2)
    two_theta = np.arctan(R_px * px / lsd)
    Q_pix = (4.0 * np.pi / lam) * np.sin(0.5 * two_theta)

    qp = np.asarray(q_profile, dtype=np.float64)
    ip = np.asarray(I_profile, dtype=np.float64)
    order = np.argsort(qp)
    I_pix = np.interp(Q_pix.ravel(), qp[order], ip[order]).reshape(Q_pix.shape)
    I_pix = np.clip(I_pix, 0.0, None)

    scale = counts / max(float(I_pix.max()), 1e-30)
    rng = np.random.default_rng(seed)
    img = rng.poisson(I_pix * scale).astype(np.float64) / scale
    return torch.as_tensor(img, dtype=torch.float64)
