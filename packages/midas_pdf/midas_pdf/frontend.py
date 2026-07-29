"""Front of the chain: detector image → 1-D I(Q) with σ → G(r).

The pixels → caked-profile step is done entirely by ``midas-integrate-v2``
(polygon-exact binning with polarization / solid-angle / dark corrections and
per-bin variance). We only (a) collapse the η axis to a 1-D I(Q) with the
NaN-aware reduction used in ``integrate_to_Gr_with_variance`` and (b) map the
radial-pixel axis to Q via the calibrated geometry — then hand off to the
polyatomic ``i_of_q_to_Gr``. This completes the pixels → G(r) pipeline with σ
carried the whole way.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .composition import Composition
from .gr import R_px_to_Q
from .pipeline import i_of_q_to_Gr

__all__ = ["image_to_iq", "image_to_Gr"]


def image_to_iq(
    image: torch.Tensor | np.ndarray,
    spec,
    *,
    binning: str = "polygon",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detector image → 1-D ``(Q, I, sigma_I)`` using midas-integrate-v2.

    ``spec`` is a populated ``midas_integrate_v2.spec.IntegrationSpec`` (geometry,
    wavelength, RMin/RMax/RBinSize, η binning, corrections). Returns Q in Å⁻¹.
    """
    from midas_integrate_v2.binning import (
        HardBinGeometry,
        PolygonBinGeometry,
        integrate_hard_with_variance,
        integrate_polygon_with_variance,
    )

    img = torch.as_tensor(image, dtype=torch.float64)
    if binning == "polygon":
        geom = PolygonBinGeometry.from_spec(spec)
        mean2d, sigma2d = integrate_polygon_with_variance(img, geom)
    elif binning == "hard":
        geom = HardBinGeometry.from_spec(spec)
        mean2d, sigma2d = integrate_hard_with_variance(img, geom)
    else:
        raise ValueError(f"binning must be 'polygon' or 'hard', got {binning!r}")

    # Collapse η with NaN-aware reductions (off-detector bins come back NaN);
    # same scheme as midas_integrate_v2.pdf.integrate_to_Gr_with_variance.
    valid = torch.isfinite(mean2d)
    n_valid = valid.sum(dim=0).clamp(min=1)
    I = torch.where(valid, mean2d, torch.zeros_like(mean2d)).sum(dim=0) / n_valid
    sig2 = torch.where(valid, sigma2d * sigma2d, torch.zeros_like(sigma2d))
    sigma_I = torch.sqrt(sig2.sum(dim=0)) / n_valid

    R_axis = (
        spec.RMin
        + (torch.arange(I.shape[0], dtype=torch.float64) + 0.5) * spec.RBinSize
    )
    Q = R_px_to_Q(R_axis, Lsd_um=spec.Lsd, px_um=spec.pxY,
                  lambda_A=spec.Wavelength)
    return Q, I, sigma_I


def image_to_Gr(
    image: torch.Tensor | np.ndarray,
    spec,
    composition: Composition,
    r_grid: torch.Tensor | np.ndarray,
    *,
    scale: Union[float, torch.Tensor] = 1.0,
    compton: Union[bool, torch.Tensor, np.ndarray, None] = True,
    background: Optional[torch.Tensor | np.ndarray] = None,
    q_min: float = 0.7,
    q_max: Optional[float] = None,
    window: str = "lorch",
    binning: str = "polygon",
    fractions: Optional[torch.Tensor | Sequence[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detector image → ``(Q, G, sigma_G, S)``, end to end with σ propagation.

    A ``q_min`` floor drops the unreliable low-Q bins (beam stop / first ring
    region) before normalization. ``spec.Wavelength`` sets the Compton scale.
    ``background`` (after the q_min/q_max trim, same length as the kept Q) is the
    lumped Tier-1 smooth term; see :mod:`midas_pdf.multiple_scattering`.
    """
    Q, I, sigma_I = image_to_iq(image, spec, binning=binning)
    keep = (Q >= q_min) & torch.isfinite(I)
    if q_max is not None:
        keep = keep & (Q <= q_max)
    Q, I, sigma_I = Q[keep], I[keep], sigma_I[keep]
    if background is not None:
        background = torch.as_tensor(background, dtype=torch.float64)[keep]

    r_t = torch.as_tensor(r_grid, dtype=torch.float64)
    G, sigma_G, S = i_of_q_to_Gr(
        Q, I, composition, r_t,
        wavelength_A=float(spec.Wavelength), scale=scale, compton=compton,
        background=background, sigma_intensity=sigma_I, q_max=q_max,
        window=window, fractions=fractions,
    )
    return Q, G, sigma_G, S
