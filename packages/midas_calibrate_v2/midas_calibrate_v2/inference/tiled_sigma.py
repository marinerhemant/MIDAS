"""1-sigma on the GLOBAL geometry of a tiled detector, with the gauge stated.

``calibrate()`` on a tiled detector refines the global geometry first, then
freezes it and fits one radial offset per (panel, ring), ``panel_ring_delta_r``.
The Laplace at that point is conditional on the frozen globals, so it can say
nothing about ``Lsd``, ``BC``, the tilts or the distortion. This module asks
the joint question: free the globals AND the panel-ring offsets together.

The joint problem has a null space, not a small one. A uniform shift of one
ring's radius on every panel is exactly what a change of ``Lsd`` (or of the
radially symmetric distortion) does to that ring, so ``Lsd`` and ``iso_R*``
trade against the per-ring panel mean freely. Something has to say which panel
is the reference. That choice is the gauge, and the global sigma on the
gauge-coupled parameters is a statement about the gauge, not only about the
data. Two gauges are computed so the dependence is visible instead of hidden:

* ``mean``  : per ring, the mean of the panel offsets over the panels that
  carry fits is zero (the Wright, Giacobbe and Lawrence Bright 2022 gauge).
* ``best``  : per ring, the best-sampled panel is the exact reference.

A parameter whose sigma differs by more than ``GAUGE_RATIO`` between the two is
reported as gauge-dependent. Beam centre and tilts are not: they act on a ring
as a function of azimuth, which a per-(panel, ring) constant cannot mimic.

Sigmas are Gauss-Newton, white noise, statistical only (see the paper's
uncertainty section); they say nothing about wavelength or lattice-constant
error.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import torch

__all__ = ["tiled_global_sigma", "TiledSigma", "GAUGE_RATIO",
           "GAUGE_MEAN", "GAUGE_BEST"]

#: sigma(best)/sigma(mean) above this flags a parameter as set by the gauge.
GAUGE_RATIO = 10.0
GAUGE_MEAN = ("panel_ring_delta_r: per ring, sum over the panels carrying fits "
              "= 0 (mean-panel reference)")
GAUGE_BEST = ("panel_ring_delta_r: per ring, the best-sampled panel is the "
              "exact reference")


class TiledSigma:
    """Plain container; see :func:`tiled_global_sigma`."""
    __slots__ = ("sigma", "sigma_alt", "gauge", "gauge_alt",
                 "gauge_dependent", "n_params", "n_rings", "n_cells_observed")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def _observed(panel_idx: torch.Tensor, ring_idx: torch.Tensor,
              n_panels: int, n_rings: int) -> torch.Tensor:
    """Fit count per (panel, ring) cell, shape ``[n_panels, n_rings]``."""
    ok = panel_idx >= 0
    flat = panel_idx[ok].long() * n_rings + ring_idx[ok].long()
    cnt = torch.bincount(flat, minlength=n_panels * n_rings)
    return cnt.reshape(n_panels, n_rings)


def tiled_global_sigma(
    spec,
    unpacked: Dict[str, torch.Tensor],
    fits,
    *,
    panel_layout,
    free_globals: List[str],
    lambda_gauge: float = 1e3,
    ridge: float = 1e-9,
    dtype=torch.float64,
    device="cpu",
) -> Optional[TiledSigma]:
    """Joint 1-sigma on ``free_globals`` and ``panel_ring_delta_r``.

    ``spec`` is the frozen-globals panel-phase spec; ``unpacked`` its MAP;
    ``fits`` the final ``BatchedFits``-like object (``Y_pix``, ``Z_pix``,
    ``ring_two_theta_deg``, ``ring_idx``, ``panel_idx``, ``rho_d``,
    ``weights``). Returns ``None`` when the spec has no ``panel_ring_delta_r``.

    ``lambda_gauge`` weights the gauge rows. Data sensitivity to one panel-ring
    offset is about ``1/R_px`` per pixel (~1e-3), so ``1e3`` already dominates
    the gauge direction by orders of magnitude; the larger default used for
    panel-shift gauges only worsens the conditioning of the Fisher matrix.
    """
    from midas_peakfit.laplace import fisher_at_map
    from ..loss.pseudo_strain import pseudo_strain_residual

    if "panel_ring_delta_r" not in spec.parameters:
        return None
    prd0 = unpacked["panel_ring_delta_r"]
    n_panels, n_rings = int(prd0.shape[0]), int(prd0.shape[1])

    sp = copy.deepcopy(spec)
    for n in free_globals:
        if n in sp.parameters:
            sp.parameters[n].refined = True
    sp.parameters["panel_ring_delta_r"].refined = True

    unp = {k: (v.detach().clone() if torch.is_tensor(v) else v)
           for k, v in unpacked.items()}

    pidx = fits.panel_idx
    ridx = fits.ring_idx
    cnt = _observed(pidx, ridx, n_panels, n_rings).to(dtype=dtype)
    obs = cnt > 0                                        # [P, K]
    ring_has = obs.any(dim=0)                            # [K]
    best_p = cnt.argmax(dim=0)                           # [K]
    n_obs_cells = int(obs.sum())

    def data_res(p):
        return pseudo_strain_residual(
            fits.Y_pix, fits.Z_pix, fits.ring_two_theta_deg, p,
            rho_d=fits.rho_d, weights=fits.weights,
            panel_layout=panel_layout, panel_idx=pidx, ring_idx=ridx)

    sqrt_lam = float(lambda_gauge) ** 0.5
    obs_f = obs.to(dtype)
    sel_best = torch.zeros(n_panels, n_rings, dtype=dtype)
    sel_best[best_p[ring_has], torch.nonzero(ring_has).reshape(-1)] = 1.0

    def gauge_mean(p):
        prd = p["panel_ring_delta_r"]
        return sqrt_lam * (prd * obs_f).sum(dim=0)[ring_has]

    def gauge_best(p):
        prd = p["panel_ring_delta_r"]
        return sqrt_lam * (prd * sel_best).sum(dim=0)[ring_has]

    with torch.no_grad():
        r0 = data_res(unp)
    n_par = int(sum(unp[n].numel() for n, q in sp.parameters.items()
                    if q.refined and n in unp))
    # observed cells only: an unobserved cell has no data and is not a parameter
    n_par = n_par - int(prd0.numel()) + n_obs_cells
    dof = max(int(r0.numel()) - n_par, 1)
    sigma_r = float((r0 * r0).sum().detach()) / dof
    sigma_r = max(sigma_r, 1e-60) ** 0.5

    def run(gauge_fn):
        def res(p):
            return torch.cat([data_res(p), gauge_fn(p)])
        lap = fisher_at_map(sp, res, unp, sigma_r=sigma_r, ridge=ridge,
                            dtype=dtype, device=device)
        out: Dict[str, float] = {}
        sig = lap.sigma_per_dim.detach().cpu().numpy()
        off = 0
        for nm, sz in zip(lap.refined_names, lap.refined_sizes):
            if sz == 1:
                out[nm] = float(sig[off])
            off += sz
        return out

    s_mean = run(gauge_mean)
    s_best = run(gauge_best)

    dep: List[str] = []
    for nm, a in s_mean.items():
        b = s_best.get(nm)
        if b is None or not (a == a) or not (b == b):
            continue
        if a > 0.0 and b / a > GAUGE_RATIO:
            dep.append(nm)

    return TiledSigma(sigma=s_mean, sigma_alt=s_best, gauge=GAUGE_MEAN,
                      gauge_alt=GAUGE_BEST, gauge_dependent=dep,
                      n_params=n_par, n_rings=int(ring_has.sum()),
                      n_cells_observed=n_obs_cells)
