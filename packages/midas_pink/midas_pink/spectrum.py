"""Parameterised pink-beam spectrum for fitting against data.

The spectrum is represented as a fixed dense grid of N_E energies plus
N_E learnable logits normalised through softmax to non-negative weights
summing to 1. This is expressive enough for any single-peak boxcar,
Gaussian, asymmetric, or multi-modal S(E) the beamline would realistically
produce, and it keeps the gradient flow trivial: gradients into the
weights are standard PyTorch.

The energies themselves are fixed (uniform grid in E around E0). For
proto3 / proto4 we are NOT trying to recover the energy locations -- only
the spectral weight density on a fixed support. This avoids needing to
make ``HEDMForwardModel.wavelength`` a differentiable tensor (it stays a
Python float per energy sample, and we rebuild a model per energy at
bank construction).

Initialisation conventions:
    init_kind="boxcar"    weights uniform across the support
    init_kind="gaussian"  weights ~ exp(-(E-E0)^2 / (2*sigma^2))
    init_kind="delta"     weight 1 at the centre energy, 0 elsewhere
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H_C_KEV_A = 12.398419739  # lambda[A] = H_C_KEV_A / E[keV]


class ParameterisedSpectrum(nn.Module):
    """Discrete-grid spectrum with learnable softmax weights.

    Parameters
    ----------
    E0_keV : float
        Centre energy (sets the symmetric energy support).
    half_bw : float
        Half-width of the energy support, as a fractional bandwidth
        (e.g., 0.05 -> support spans E0 * [1 - 0.05, 1 + 0.05]).
    n_samples : int
        Number of energy samples on the grid (odd is preferred so the
        centre energy is sampled exactly).
    init_kind : str
        ``"boxcar"`` | ``"gaussian"`` | ``"delta"`` -- how to initialise
        the logits. The grid itself does not change once constructed.
    init_rel_bw : float, optional
        Bandwidth of the initialiser (FWHM for gaussian; full width for
        boxcar). Required for ``init_kind in {"boxcar", "gaussian"}``.
    fixed : bool
        If True, the logits are buffers (not parameters) -- use this for
        proto2 where S(E) is known exactly and we don't fit it.
    """
    def __init__(
        self,
        E0_keV: float,
        half_bw: float,
        n_samples: int = 51,
        init_kind: str = "boxcar",
        init_rel_bw: Optional[float] = None,
        fixed: bool = False,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        if half_bw <= 0:
            raise ValueError("half_bw must be > 0")
        if init_kind not in {"boxcar", "gaussian", "delta"}:
            raise ValueError(f"init_kind must be boxcar|gaussian|delta, got {init_kind!r}")
        self.E0_keV = float(E0_keV)
        self.half_bw = float(half_bw)
        self.n_samples = int(n_samples)
        self.fixed = bool(fixed)

        E_min = self.E0_keV * (1.0 - self.half_bw)
        E_max = self.E0_keV * (1.0 + self.half_bw)
        Es = np.linspace(E_min, E_max, self.n_samples, dtype=np.float64)
        lams = H_C_KEV_A / Es
        self.register_buffer("energies_keV", torch.tensor(Es, dtype=dtype))
        self.register_buffer("lambdas_A", torch.tensor(lams, dtype=dtype))

        # Build initial logits to match the requested init_kind
        with torch.no_grad():
            logits = self._initial_logits(init_kind, init_rel_bw, dtype)
        if fixed:
            self.register_buffer("logits", logits)
        else:
            self.logits = nn.Parameter(logits)

    def _initial_logits(
        self, init_kind: str, init_rel_bw: Optional[float], dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build logits ``l`` such that ``softmax(l)`` matches the target shape.

        For boxcar: uniform over a sub-range, otherwise -large.
        For gaussian: -((E-E0)/(sqrt(2)*sigma))^2.
        For delta: high logit at centre, low elsewhere.
        """
        Es = self.energies_keV.detach().cpu().numpy()
        n = Es.shape[0]
        E0 = self.E0_keV

        if init_kind == "delta":
            l = -1e3 * np.ones(n, dtype=np.float64)
            ic = n // 2
            l[ic] = 0.0
            return torch.tensor(l, dtype=dtype)

        if init_rel_bw is None:
            raise ValueError(f"init_kind={init_kind!r} requires init_rel_bw")

        if init_kind == "boxcar":
            half = init_rel_bw / 2.0
            in_box = (Es >= E0 * (1.0 - half)) & (Es <= E0 * (1.0 + half))
            l = np.where(in_box, 0.0, -1e3)
            return torch.tensor(l, dtype=dtype)

        # gaussian
        fwhm = init_rel_bw * E0
        sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        l = -((Es - E0) / (math.sqrt(2.0) * sigma)) ** 2
        return torch.tensor(l, dtype=dtype)

    def weights(self) -> torch.Tensor:
        """Softmax-normalised weights summing to 1, shape (n_samples,)."""
        return F.softmax(self.logits, dim=0)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (energies_keV, lambdas_A, weights). All tensors."""
        return self.energies_keV, self.lambdas_A, self.weights()

    # --------------------------------------------------------------
    #  Convenience setters / inspectors used by proto2 and friends
    # --------------------------------------------------------------

    def set_to_boxcar(self, rel_bw: float):
        """Re-initialise logits to a boxcar of the given bandwidth."""
        with torch.no_grad():
            l = self._initial_logits("boxcar", rel_bw, self.logits.dtype)
            self.logits.data.copy_(l)

    def set_to_gaussian(self, rel_bw: float):
        with torch.no_grad():
            l = self._initial_logits("gaussian", rel_bw, self.logits.dtype)
            self.logits.data.copy_(l)

    def set_to_delta(self):
        with torch.no_grad():
            l = self._initial_logits("delta", None, self.logits.dtype)
            self.logits.data.copy_(l)

    @property
    def support_E_keV(self) -> Tuple[float, float]:
        return float(self.energies_keV.min()), float(self.energies_keV.max())

    @property
    def support_lambda_A(self) -> Tuple[float, float]:
        return float(self.lambdas_A.min()), float(self.lambdas_A.max())

    def effective_bandwidth(self) -> float:
        """Weighted std of E, divided by E0 -- a scalar summary of S(E)."""
        w = self.weights().detach().cpu().numpy()
        E = self.energies_keV.detach().cpu().numpy()
        Em = float((w * E).sum())
        var = float((w * (E - Em) ** 2).sum())
        return math.sqrt(var) / self.E0_keV

    def effective_centre_keV(self) -> float:
        w = self.weights().detach().cpu().numpy()
        E = self.energies_keV.detach().cpu().numpy()
        return float((w * E).sum())
