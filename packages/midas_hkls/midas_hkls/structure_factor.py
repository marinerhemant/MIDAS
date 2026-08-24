"""Differentiable structure factor F_hkl in PyTorch.

F_hkl = Σⱼ fⱼ(s) · occⱼ · DWFⱼ(s, hkl) · exp(2πi h·rⱼ)

Sum is over the *unit-cell* atom list (asymmetric unit expanded by symmetry +
centering — see ``crystal_torch.crystal_to_tensor``).  Anomalous f', f'' are
added to f when ``anomalous=True``.

Differentiable through:
- atomic fractional coordinates
- occupancies
- isotropic B-factors and anisotropic U-tensors
- the six lattice parameters
- wavelength (when anomalous=True)
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .crystal_torch import CrystalTensor
from .form_factors import form_factor_batch
from .lattice_torch import reciprocal_metric_tensor, s_squared

if TYPE_CHECKING:  # pragma: no cover
    import torch


__all__ = ["structure_factors", "structure_factor_intensity", "f2_normalised"]


_TWOPI = 2.0 * math.pi


def structure_factors(
    crystal_t: CrystalTensor,
    hkl,
    *,
    wavelength_A: "Optional[float | torch.Tensor]" = None,
    anomalous: bool = False,
) -> "torch.Tensor":
    """F_hkl, complex tensor of shape (M,).

    ``hkl`` may be a (M, 3) integer array/tensor, a list of triples, or a
    ``Sequence[Reflection]``.

    With ``anomalous=True`` you must pass ``wavelength_A``; the Cromer-Liberman
    f', f'' lookup is added to the real form factor.  Anomalous gradients are
    available w.r.t. ``wavelength_A`` when it's a tensor.
    """
    import torch

    hkl_t = _coerce_hkl(hkl, crystal_t)                  # (M, 3) long
    s2 = s_squared(hkl_t.to(crystal_t.lattice_params.dtype),
                   crystal_t.lattice_params)             # (M,)

    fract, occ, B_iso, U_aniso = crystal_t.unit_cell_view()
    f0 = form_factor_batch(s2, crystal_t.elements)        # (M, N)

    if anomalous:
        if wavelength_A is None:
            raise ValueError("anomalous=True requires wavelength_A")
        from .anomalous import anomalous_correction
        fp, fpp = anomalous_correction(
            crystal_t.elements, wavelength_A,
            dtype=f0.dtype, device=f0.device,
        )                                                  # each (N,)
        # Per-atom f' and f'' (broadcast over reflections)
        f_complex = torch.complex(f0 + fp.unsqueeze(0), fpp.unsqueeze(0).expand_as(f0))
    else:
        f_complex = torch.complex(f0, torch.zeros_like(f0))

    # Phase term exp(2πi h·r)
    phase = _TWOPI * (hkl_t.to(fract.dtype) @ fract.T)                       # (M, N)
    geom = torch.complex(torch.cos(phase), torch.sin(phase))                  # (M, N)

    # Debye-Waller
    DWF = _debye_waller(s2, hkl_t, crystal_t, B_iso, U_aniso)                 # (M, N)
    DWF = DWF.to(geom.dtype)

    occ_c = occ.to(geom.dtype)                                                # (N,)
    F = (f_complex * DWF * geom * occ_c.unsqueeze(0)).sum(dim=-1)             # (M,)
    return F


def structure_factor_intensity(F) -> "torch.Tensor":
    """|F|² for a complex F tensor."""
    import torch
    return F.real * F.real + F.imag * F.imag


def f2_normalised(lattice, space_group, atoms, hkl) -> "np.ndarray":
    """Per-reflection |F|², normalised to its maximum. ``(M,)`` float64 ndarray.

    The reflection weight used by every HEDM technique's completeness metric.
    Extracted from ``nf_hkls`` so NF and FF/PF share one implementation rather
    than two that can drift.

    **Why a weight is needed at all.** Space-group extinction rules cannot see
    *basis*-dependent extinctions. For a DHCP polytype (P6_3/mmc, sites 2a and
    2c) 126 of 736 reflections have |F|² = 0 and can never produce a spot — yet
    they sat in the confidence denominator, capping the achievable overlap
    fraction at 0.829, while fcc (one atom at the origin, nothing extra
    extinct) reached 1.000. A grain of the first phase and a grain of the
    second are then not comparable on a number both are reported against.

    **|F|², deliberately not intensity.** Lorentz-polarisation diverges at low
    2θ and would swamp the normalisation, and for the rotation method the
    Lorentz factor *cancels* in per-frame peak detection: Δω = w_rlp·L, so
    I_peak = I_int/Δω is L-free. Measured on ``nf_sampleB_htB_s2`` — spot peak
    height is flat in η where 1/|sin η| predicts 4.3×, and spot density along a
    ring rises as sin η, the complementary half of the same model. LP belongs
    in the *integrated*-intensity path (grain size, see
    ``midas_transforms.radius.theoretical``), not here.

    Normalising to the maximum is what makes the value a weight in [0, 1]
    rather than an absolute electron count, so a run's numbers do not move when
    the cell content changes scale.

    Parameters
    ----------
    lattice, space_group, atoms
        Passed straight to :class:`midas_hkls.crystal.Crystal`.
    hkl
        ``(M, 3)`` integer Miller indices.

    Returns
    -------
    ``(M,)`` float64 in [0, 1]. All-zero if every reflection is extinct (an
    all-zero return is a real answer, not a failure — but it means no
    reflection can be observed, so callers should treat it as a configuration
    error rather than divide by it).
    """
    import numpy as np
    import torch

    from .crystal import Crystal
    from .structure_factor import structure_factors

    hkl_arr = np.asarray(hkl)
    cry = Crystal(lattice, space_group, list(atoms))
    F = structure_factors(cry.to_torch(), torch.as_tensor(hkl_arr.astype(int)))
    f2 = np.abs(np.asarray(F.detach().cpu().numpy()).ravel()) ** 2
    mx = float(f2.max()) if f2.size else 0.0
    return (f2 / mx) if mx > 0 else f2


# ============================================================ private helpers

def _coerce_hkl(hkl, crystal_t: CrystalTensor):
    """Accept Reflection list, ndarray, tensor, or list-of-triples → (M, 3) long tensor."""
    import torch
    device = crystal_t.lattice_params.device

    if hasattr(hkl, "__len__") and len(hkl) > 0 and hasattr(hkl[0], "h"):
        rows = [(r.h, r.k, r.l) for r in hkl]
        return torch.tensor(rows, dtype=torch.long, device=device)
    if isinstance(hkl, torch.Tensor):
        return hkl.to(device=device, dtype=torch.long)
    arr = np.asarray(hkl, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
    return torch.tensor(arr, dtype=torch.long, device=device)


def _debye_waller(s2, hkl_t, crystal_t: CrystalTensor, B_iso, U_aniso):
    """Per-atom DWF: shape (M, N).  Anisotropic if U_aniso is set, else isotropic."""
    import torch
    if U_aniso is not None and U_aniso.abs().sum() > 0:
        # T = exp(-2π² hᵀ U* h) where U* is the reciprocal-basis ADP tensor.
        # CIF U is in *fractional* basis already; combine with hkl directly:
        #   2π² · Σᵢⱼ hᵢ Uᵢⱼ hⱼ                 (Trueblood et al. 1996 convention β-form)
        # Note: this matches the cctbx / gemmi "U_cif" convention.
        h = hkl_t.to(U_aniso.dtype)                          # (M, 3)
        U = _u6_to_mat3(U_aniso)                              # (N, 3, 3)
        # exponent = 2π² · h^T · U_cart · h  with h in reciprocal lattice units
        # CIF stores U_ij in fractional basis; the standard structure-factor form is:
        #   T = exp(-2π² Σᵢⱼ hᵢhⱼ Uᵢⱼ aᵢ* aⱼ*)
        # so we need the reciprocal-axis lengths.
        Gstar = reciprocal_metric_tensor(crystal_t.lattice_params)
        a_star = torch.sqrt(torch.diag(Gstar))                # (3,)
        h_scaled = h * a_star                                 # (M, 3) in Å⁻¹ axis units
        # exponent[m, n] = 2π² · h_scaled[m] · U[n] · h_scaled[m]
        exponent = 2.0 * (math.pi ** 2) * torch.einsum('mi,nij,mj->mn', h_scaled, U, h_scaled)
        return torch.exp(-exponent)

    # Isotropic: exp(-B s²)
    B = B_iso.to(s2.dtype)                                    # (N,)
    return torch.exp(-B.unsqueeze(0) * s2.unsqueeze(1))       # (M, N)


def _u6_to_mat3(u6):
    import torch
    u11, u22, u33, u12, u13, u23 = u6.unbind(dim=-1)
    row0 = torch.stack([u11, u12, u13], dim=-1)
    row1 = torch.stack([u12, u22, u23], dim=-1)
    row2 = torch.stack([u13, u23, u33], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)
