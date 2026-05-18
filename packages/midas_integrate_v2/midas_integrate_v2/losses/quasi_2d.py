"""Quasi-2D losses — for textured / single-crystal / oriented samples.

The standard `EtaUniformityLoss` assumes a Debye-Scherrer ring (perfect
powder, uniform along η). For:

- **Textured samples** (preferred orientation in a polycrystal),
- **Single-crystal-like / strongly oriented** samples,
- **Single-spot diffraction** (HEDM grain centres),

the η profile is not uniform — it has structure that's a feature, not a
bug. Forcing η-uniformity would push the optimiser away from the right
geometry. These losses operate on the 2D ``(n_eta, n_r)`` array directly
without assuming η-flatness.

Three loss types:

1. :class:`EtaSliceLoss` — pick K specific η slices, compare each
   slice's R-profile against a reference. Good for periodic-texture
   samples where each azimuthal slice has its own characteristic
   profile.
2. :class:`WedgeLoss` — average over a configurable η wedge, compare
   to a reference 1D profile. Good for samples with localised
   diffraction (e.g., single-grain HEDM where the spot lives in one η
   range).
3. :class:`RingMaskedLoss` — apply a 2D (η, R) "ring mask" picking
   out specific (η, R) regions to compare against a reference. Most
   general; user-defined regions of interest.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


class EtaSliceLoss(nn.Module):
    """Per-η-slice MSE between integrated 2D and a reference 2D.

    Parameters
    ----------
    eta_indices :
        Iterable of η-bin indices to include. Defaults to all bins.
    r_indices :
        Iterable of R-bin indices to include. Defaults to all bins.

    Forward signature:
        ``loss(int2d, spec, reference_2d) -> scalar``

    ``int2d`` and ``reference_2d`` must both be shape
    ``(n_eta_total, n_r_total)``. The loss is the mean of squared
    differences across the (eta_indices × r_indices) sub-block.
    """

    def __init__(
        self,
        *,
        eta_indices: Optional[Iterable[int]] = None,
        r_indices: Optional[Iterable[int]] = None,
    ):
        super().__init__()
        self._eta_idx = (None if eta_indices is None
                          else torch.tensor(list(eta_indices), dtype=torch.long))
        self._r_idx = (None if r_indices is None
                        else torch.tensor(list(r_indices), dtype=torch.long))

    def forward(
        self,
        int2d: torch.Tensor,
        spec,                # IntegrationSpec (unused; kept for the loss API)
        reference_2d: torch.Tensor,
    ) -> torch.Tensor:
        if int2d.shape != reference_2d.shape:
            raise ValueError(
                f"int2d shape {tuple(int2d.shape)} != reference shape "
                f"{tuple(reference_2d.shape)}"
            )
        a, r = int2d, reference_2d
        if self._eta_idx is not None:
            a = a.index_select(0, self._eta_idx.to(a.device))
            r = r.index_select(0, self._eta_idx.to(r.device))
        if self._r_idx is not None:
            a = a.index_select(1, self._r_idx.to(a.device))
            r = r.index_select(1, self._r_idx.to(r.device))
        return ((a - r) ** 2).mean()


class WedgeLoss(nn.Module):
    """Wedge-averaged MSE between integrated 2D and a reference 1D.

    Averages ``int2d`` over the η bins inside ``[eta_min_deg,
    eta_max_deg]`` (handles wraparound at ±180°), then compares against
    a 1D reference profile. Pixels outside the wedge are ignored.

    Use this when the sample produces diffraction *only* in a known η
    band — single grains in HEDM, spotty samples, sample-shadow
    asymmetry.

    Forward signature:
        ``loss(int2d, spec, reference_1d) -> scalar``
    """

    def __init__(
        self, *,
        eta_min_deg: float, eta_max_deg: float,
        r_indices: Optional[Iterable[int]] = None,
    ):
        super().__init__()
        if eta_max_deg <= eta_min_deg:
            raise ValueError(
                f"eta_max_deg ({eta_max_deg}) must be > eta_min_deg "
                f"({eta_min_deg})"
            )
        self.eta_min_deg = float(eta_min_deg)
        self.eta_max_deg = float(eta_max_deg)
        self._r_idx = (None if r_indices is None
                        else torch.tensor(list(r_indices), dtype=torch.long))

    def forward(
        self,
        int2d: torch.Tensor,
        spec,
        reference_1d: torch.Tensor,
    ) -> torch.Tensor:
        n_eta = int2d.shape[0]
        eta_centres = (
            spec.EtaMin + spec.EtaBinSize * (
                torch.arange(n_eta, dtype=int2d.dtype, device=int2d.device) + 0.5
            )
        )
        in_wedge = (eta_centres >= self.eta_min_deg) & (eta_centres <= self.eta_max_deg)
        if not in_wedge.any():
            raise ValueError(
                f"wedge [{self.eta_min_deg}, {self.eta_max_deg}] contains no "
                f"η bins (EtaMin={spec.EtaMin}, EtaMax={spec.EtaMax}, "
                f"EtaBinSize={spec.EtaBinSize})"
            )
        # Wedge-averaged 1D profile (mean over the η bins in the wedge)
        wedge_2d = int2d[in_wedge]
        wedge_1d = wedge_2d.mean(dim=0)
        if self._r_idx is not None:
            wedge_1d = wedge_1d.index_select(0, self._r_idx.to(wedge_1d.device))
            ref     = reference_1d.index_select(0, self._r_idx.to(reference_1d.device))
        else:
            ref = reference_1d
        if wedge_1d.shape != ref.shape:
            raise ValueError(
                f"wedge profile shape {tuple(wedge_1d.shape)} != reference "
                f"shape {tuple(ref.shape)}"
            )
        return ((wedge_1d - ref) ** 2).mean()


class RingMaskedLoss(nn.Module):
    """User-defined 2D mask in (η, R) space; MSE against a reference 2D.

    The mask is a boolean / float 2D array of shape ``(n_eta, n_r)``;
    bins where the mask is non-zero contribute to the loss.

    This is the most general of the three: define any region you care
    about (a single ring, a wedge in a ring, the union of several
    rings) and the loss is the MSE inside that region.

    Forward signature:
        ``loss(int2d, spec, reference_2d, mask_2d) -> scalar``
    """

    def forward(
        self,
        int2d: torch.Tensor,
        spec,
        reference_2d: torch.Tensor,
        mask_2d: torch.Tensor,
    ) -> torch.Tensor:
        if int2d.shape != reference_2d.shape:
            raise ValueError(
                f"int2d shape {tuple(int2d.shape)} != reference shape "
                f"{tuple(reference_2d.shape)}"
            )
        if int2d.shape != mask_2d.shape:
            raise ValueError(
                f"int2d shape {tuple(int2d.shape)} != mask shape "
                f"{tuple(mask_2d.shape)}"
            )
        m = mask_2d.to(int2d.dtype)
        wsum = m.sum() + 1e-30
        return (m * (int2d - reference_2d) ** 2).sum() / wsum


__all__ = ["EtaSliceLoss", "WedgeLoss", "RingMaskedLoss"]
