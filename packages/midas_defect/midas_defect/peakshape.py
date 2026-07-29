"""Differentiable size / microstrain / mosaic from FF-HEDM spot widths.

Companion to :mod:`midas_defect.williamson_hall` (which operates on
``AsterismFit`` objects from the asterism-extraction pipeline); this module
takes the generic raw ``(q, width)`` interface, adds a **mosaic** channel from
the azimuthal widths, exposes a **differentiable** joint fit (so any one of
size/strain/mosaic can be a fittable knob in a larger objective), and recovers
a **crystallite-size distribution** by mixture deconvolution.

Physics signatures (same as :mod:`williamson_hall`):

* size D  -> radial width CONSTANT in q (Scherrer ``w_size = 2*pi*K / D``),
* microstrain ``<dd/d>``  -> radial width grows AS q  (``w_strain = eps * q``),
* mosaic (orientation spread)  -> azimuthal width q-independent in angle.

Quadrature combine:
    ``w_rad^2  = w_inst_rad^2 + w_size^2 + (eps * q)^2``
    ``w_az^2   = w_inst_az^2  + mosaic^2``

Built on ``midas_invert`` (shared fit + mixture-deconvolution primitives).
Originally prototyped as the standalone ``midas_peakshape`` package; folded
into ``midas_defect`` so HEDM size-strain-mosaic analysis has one home.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

__all__ = [
    "size_width_q",
    "crystallite_size_from_width",
    "radial_width",
    "azimuthal_width",
    "gaussian_profile",
    "size_broadened_profile",
    "williamson_hall_qw",
    "fit_size_strain_mosaic",
    "recover_size_distribution",
]

_SCHERRER_K = 0.9


# --------------------------------------------------------------- forward

def size_width_q(D, *, K=_SCHERRER_K):
    """q-space size broadening (FWHM-like) ``2*pi*K / D``."""
    import torch
    D = torch.as_tensor(D)
    return 2.0 * math.pi * float(K) / torch.clamp(D, min=1e-9)


def crystallite_size_from_width(w_size, *, K=_SCHERRER_K):
    """Invert :func:`size_width_q`: ``D = 2*pi*K / w_size``."""
    return 2.0 * math.pi * float(K) / float(w_size)


def radial_width(q, *, w_size, eps, w_inst=0.0):
    """Radial spot width vs |q|: ``sqrt(w_inst^2 + w_size^2 + (eps q)^2)``."""
    import torch
    q = torch.as_tensor(q)
    w_size = torch.as_tensor(w_size, dtype=q.dtype, device=q.device)
    eps = torch.as_tensor(eps, dtype=q.dtype, device=q.device)
    return torch.sqrt(float(w_inst) ** 2 + w_size ** 2 + (eps * q) ** 2)


def azimuthal_width(mosaic, *, w_inst_az=0.0, n=1):
    """Azimuthal (eta) spot width from mosaic spread (q-independent in angle)."""
    import torch
    mosaic = torch.as_tensor(mosaic)
    w = torch.sqrt(float(w_inst_az) ** 2 + mosaic ** 2)
    return w.expand(n) if n > 1 else w


def gaussian_profile(x, center, width):
    """Unit-area Gaussian (``width`` = standard deviation)."""
    import torch
    x = torch.as_tensor(x)
    center = torch.as_tensor(center, dtype=x.dtype, device=x.device)
    width = torch.as_tensor(width, dtype=x.dtype, device=x.device)
    return (1.0 / (width * math.sqrt(2.0 * math.pi))) * torch.exp(
        -0.5 * ((x - center) / width) ** 2)


def size_broadened_profile(dq, D, *, K=_SCHERRER_K, center=0.0):
    """Radial line profile of a single crystallite size ``D``."""
    w = size_width_q(D, K=K) / (2.0 * math.sqrt(2.0 * math.log(2.0)))   # FWHM->sigma
    return gaussian_profile(dq, center, w)


# --------------------------------------------------------------- inverse

def williamson_hall_qw(q, w_rad, *, w_inst=0.0, K=_SCHERRER_K):
    """Closed-form linear Williamson-Hall on **raw** ``(q, width)`` arrays.

    Distinct from :func:`midas_defect.williamson_hall.williamson_hall`, which
    operates on the per-hkl ``AsterismFit`` objects produced upstream; this is
    the generic raw-array entry point used as an initialiser for the
    differentiable joint fit below.

    Fits ``w_rad^2 - w_inst^2 = w_size^2 + eps^2 q^2``.  Returns dict with
    ``D``, ``eps``, ``w_size``.
    """
    import torch
    q = torch.as_tensor(q, dtype=torch.float64)
    w = torch.as_tensor(w_rad, dtype=torch.float64)
    y = (w ** 2 - float(w_inst) ** 2).reshape(-1, 1)
    A = torch.stack([torch.ones_like(q), q ** 2], dim=1)
    sol = torch.linalg.lstsq(A, y).solution.squeeze(1)
    w_size = math.sqrt(max(float(sol[0]), 0.0))
    eps = math.sqrt(max(float(sol[1]), 0.0))
    return {"D": (crystallite_size_from_width(w_size, K=K) if w_size > 0
                  else float("inf")),
            "eps": eps, "w_size": w_size}


def fit_size_strain_mosaic(q, w_rad_obs, w_az_obs=None, *, w_inst=0.0,
                           w_inst_az=0.0, init=None, steps=800, lr=0.05,
                           K=_SCHERRER_K):
    """Differentiable joint fit of (D, microstrain, mosaic) to measured spot
    widths.  ``w_az_obs`` optional; if given, mosaic is fit too.  Returns dict
    with ``D``, ``eps``, ``mosaic`` (or None), ``w_size``, ``loss``."""
    import torch
    from midas_invert.optimize import fit, relative_l2_loss

    q = torch.as_tensor(q, dtype=torch.float64)
    w_rad_obs = torch.as_tensor(w_rad_obs, dtype=torch.float64)
    if init is None:
        wh = williamson_hall_qw(q, w_rad_obs, w_inst=w_inst, K=K)
        init = {"w_size": max(wh["w_size"], 1e-3), "eps": max(wh["eps"], 1e-5),
                "mosaic": 1e-3}
    sp = torch.nn.functional.softplus
    raw_ws = torch.tensor(math.log(math.expm1(init["w_size"])), dtype=torch.float64, requires_grad=True)
    raw_eps = torch.tensor(math.log(math.expm1(init["eps"])), dtype=torch.float64, requires_grad=True)
    params = [raw_ws, raw_eps]
    raw_mos = None
    if w_az_obs is not None:
        w_az_obs = torch.as_tensor(w_az_obs, dtype=torch.float64)
        raw_mos = torch.tensor(math.log(math.expm1(init["mosaic"])), dtype=torch.float64, requires_grad=True)
        params.append(raw_mos)

    def loss_fn():
        pred_rad = radial_width(q, w_size=sp(raw_ws), eps=sp(raw_eps), w_inst=w_inst)
        loss = relative_l2_loss(pred_rad, w_rad_obs)
        if raw_mos is not None:
            pred_az = torch.sqrt(float(w_inst_az) ** 2 + sp(raw_mos) ** 2)
            loss = loss + relative_l2_loss(pred_az.expand_as(w_az_obs), w_az_obs)
        return loss

    out = fit(params, loss_fn, steps=steps, lr=lr)
    w_size = float(sp(raw_ws).detach())
    return {"D": (crystallite_size_from_width(w_size, K=K) if w_size > 0 else float("inf")),
            "eps": float(sp(raw_eps).detach()),
            "mosaic": (float(sp(raw_mos).detach()) if raw_mos is not None else None),
            "w_size": w_size, "loss": out["loss"]}


def recover_size_distribution(dq_grid, observed_profile, D_grid, *,
                              K=_SCHERRER_K, steps=1200, lr=0.05,
                              entropy_weight=0.0):
    """Recover a crystallite-size distribution from a broadened radial profile
    by mixture deconvolution over ``D_grid``.  Returns dict with ``D_grid`` and
    ``weights``.  Width-only deconvolution (profiles co-centred, differ only in
    width) is mildly ill-conditioned; expect ~10% mean-size tolerance."""
    import torch
    from midas_invert.mixture import mixture_deconvolution

    dq = torch.as_tensor(dq_grid, dtype=torch.float64)
    basis = torch.stack([size_broadened_profile(dq, float(D), K=K) for D in D_grid])
    w = mixture_deconvolution(observed_profile, basis, loss="rel_l2",
                              steps=steps, lr=lr, entropy_weight=entropy_weight)
    return {"D_grid": torch.as_tensor(D_grid, dtype=torch.float64), "weights": w}
