"""Resample (η, R, E) data cubes onto (Q, E) for RIXS analysis.

For 27-ID / 30-ID RIXS spectrometers, the analyser produces an
energy-resolved cube ``(n_eta, n_R, n_E)``. The downstream physics
lives in ``(Q, E)`` so this helper averages over η (azimuthally
isotropic for an inelastic scattering signal off a powder) and remaps
``R → Q`` using the detector geometry.

Caveats:
- Single-crystal RIXS needs (qx, qy, qz, E); this stub only supports
  the powder-averaged (|Q|, E) form.
- Coherence / polarization-resolved RIXS deserves its own module;
  flagged as future work.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def regroup_eta_R_E_to_Q_E(
    cube: torch.Tensor,
    eta_axis_deg: torch.Tensor,
    R_axis_to_Q: torch.Tensor,
    E_axis_eV: torch.Tensor,
    *,
    Q_grid: torch.Tensor,
    interpolation: str = "linear",
    outside: str = "nan",
) -> torch.Tensor:
    """Remap a ``(n_eta, n_R, n_E)`` cube onto ``(Q_grid, E_axis_eV)``.

    Returns ``(n_Q, n_E)`` after η-averaging.

    Parameters
    ----------
    outside :
        What to emit for points of ``Q_grid`` outside the measured Q range.

        - ``"nan"`` (**default since 2026-08-29**) — NaN, so unmeasured Q is
          visibly unmeasured.
        - ``"clamp"`` — the nearest endpoint value, which is what
          :func:`numpy.interp` does by default and what this function did
          unconditionally before.

        **Why the default changed.** Clamping is silent fabrication: asking for
        a Q grid wider than the data returned a flat plateau of the first/last
        measured intensity, with no NaN and no warning, indistinguishable from
        real signal. Measured on a ramp from 10 to 50 over Q = 2–6 regrouped
        onto Q = 0–10: every point below 2 came back as exactly 10.0 and every
        point above 6 as exactly 50.0.
    """
    if interpolation != "linear":
        raise NotImplementedError(
            f"only linear interpolation supported; got {interpolation!r}"
        )
    cube_t = torch.as_tensor(cube, dtype=torch.float64)
    if cube_t.ndim != 3:
        raise ValueError(f"cube must be 3-D, got shape {cube_t.shape}")
    Q_axis = torch.as_tensor(R_axis_to_Q, dtype=torch.float64)
    E_axis = torch.as_tensor(E_axis_eV, dtype=torch.float64)
    Q_grid = torch.as_tensor(Q_grid, dtype=torch.float64)
    if cube_t.shape != (eta_axis_deg.shape[0], Q_axis.shape[0],
                          E_axis.shape[0]):
        raise ValueError(
            f"cube shape {cube_t.shape} does not match axes "
            f"({eta_axis_deg.shape[0]}, {Q_axis.shape[0]}, {E_axis.shape[0]})"
        )
    if outside not in ("nan", "clamp"):
        raise ValueError(f"outside must be 'nan' or 'clamp', got {outside!r}")
    eta_avg = cube_t.mean(dim=0).numpy()                # (n_R, n_E)
    Q_axis_np = Q_axis.numpy()
    sort_idx = np.argsort(Q_axis_np)
    # np.interp CLAMPS to the endpoint values by default. Pass left/right
    # explicitly so unmeasured Q is NaN rather than a plausible-looking
    # plateau of the first/last measured intensity.
    edge = None if outside == "clamp" else np.nan
    out = np.empty((Q_grid.shape[0], E_axis.shape[0]), dtype=np.float64)
    for j in range(E_axis.shape[0]):
        out[:, j] = np.interp(
            Q_grid.numpy(), Q_axis_np[sort_idx], eta_avg[sort_idx, j],
            left=edge, right=edge,
        )
    return torch.as_tensor(out, dtype=torch.float64)


__all__ = ["regroup_eta_R_E_to_Q_E"]
