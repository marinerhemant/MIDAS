"""midas_propagate.joint_nll — per-grain Hessian blocks (H_gg, H_gc) at MAP.

For paper-1's Schur marginal, each grain needs the two Hessian blocks of
the joint NLL on (g, c) where:

    g = (euler_rad[3], latc[6], pos_um[3])      → 12 grain params
    c = (Lsd, BC_y, BC_z, ty, tz, ...)          → user-chosen calibration subset

Spot-to-prediction associations are frozen at MAP (same trick as
``midas_uq.laplace_covariance``) so the residual is a smooth function of
(g, c) and Hessian/Jacobian are well-defined.

**Why mixed autograd + FD.** ``HEDMForwardModel.__init__`` snapshots
its geometry fields through ``float(...)`` or ``torch.tensor(...)``
casts (e.g. ``self.ty = float(ty_list[0])`` at forward.py:337), which
silently *detaches* gradients of the forward outputs from the geometry
input tensors. That makes pure autograd on ``c`` produce zero Jacobian.
We get around this by computing the c-Jacobian with **central finite
differences** through fresh forward-model constructions — slow per-grain
but ~5 min for the full 200-grain population paper-1 scale. The g-Jacobian
stays on autograd via ``jacfwd`` (the model IS differentiable on its
``forward(euler, pos, lattice_params=latc)`` inputs).

Two paths for H_gg:

- **Fisher** (default): ``J_g^T·J_g`` where ``J_g`` is the autograd
  Jacobian of the per-spot residual on grain state. Gauss-Newton.
- **Hessian**: full ``torch.autograd.functional.hessian`` on ``0.5·||r||²``.
  Use to validate Fisher.

H_gc is always FD (autograd path is structurally blocked; see above).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

# Map paper-1 calibration parameter names (matching the joint Σ_cc output)
# to HEDMGeometry's field names.
_CALIB_NAME_TO_GEOM = {
    "BC_y": "y_BC",
    "BC_z": "z_BC",
    # Lsd, ty, tz, tx, wedge, wavelength: same name in both
}


@dataclass
class GrainObs:
    """Per-grain observed-spot bundle at MAP geometry.

    The observed quantity is ``(y_pixel, z_pixel, frame_nr)`` in **detector
    space** — these are the directly-measured properties whose values
    depend on calibration through the detector projection. Angular
    coordinates ``(2θ, η, ω)`` are crystal-frame quantities and don't
    couple to (Lsd, BC, ty, tz) by physics, so an angular-space residual
    has *zero* gradient on calibration; we'd see no inflation in the
    Schur marginal regardless of Σ_cc.

    Attributes
    ----------
    spot_id : int
        Seed spot ID (from indexing) — paper-bookkeeping only.
    euler_rad : (3,) float64
        MAP Bunge ZXZ Euler angles in radians.
    latc : (6,) float64
        MAP lattice constants ``(a, b, c, alpha, beta, gamma)``
        (lengths in Å, angles in degrees — same convention as
        :class:`midas_uq.GrainState`).
    pos_um : (3,) float64
        MAP grain position in sample frame, in micrometers.
    observed_detector : (N, 3) float64
        Per-spot ``(y_pixel, z_pixel, frame_nr)`` observed at the MAP
        calibration. y_pixel, z_pixel are fractional pixels; frame_nr is
        fractional frame index (matches what
        ``HEDMForwardModel.predict_spot_coords(space="detector")`` returns).
    """
    spot_id: int
    euler_rad: torch.Tensor
    latc: torch.Tensor
    pos_um: torch.Tensor
    observed_detector: torch.Tensor


@dataclass
class GrainHessianBlocks:
    """Output of :func:`per_grain_hessian_blocks`.

    ``H_cc_data`` is the grain's *contribution* to the joint calibration
    precision. Summing it across grains and adding the calibrant prior
    precision (Σ_cc⁻¹) gives the joint H_cc_total whose inverse is the
    effective calibration covariance the Schur formula needs::

        H_cc_total = Sigma_cc_prior^{-1} + sum_g H_cc_data_g
        Sigma_cc_effective = inv(H_cc_total)

    Without this, the Schur correction silently over-counts the
    calibration uncertainty whenever the grain data are informative
    on calibration (which they are in practice — see paper-1 §Methods).
    """
    H_gg: torch.Tensor              # (n_g, n_g)
    H_gc: torch.Tensor              # (n_g, n_c)
    H_cc_data: torch.Tensor         # (n_c, n_c) — this grain's J_c^T J_c
    n_spots_matched: int
    residual_at_map: torch.Tensor   # (n_matched * 3,) residual vector at MAP
    sigma_r_used: float             # noise scale used for Fisher denominator


def _geometry_with_calibration(
    base: "midas_diffract.HEDMGeometry",
    c_names: Sequence[str],
    c_flat: torch.Tensor,
) -> "midas_diffract.HEDMGeometry":
    """Return a fresh HEDMGeometry with overrides from ``c_flat`` applied.

    Names follow paper-1's convention (matches the joint-Σ_cc output);
    they're translated to HEDMGeometry field names via
    :data:`_CALIB_NAME_TO_GEOM` so callers don't have to know about
    midas_diffract's ``y_BC``/``z_BC`` casing.
    """
    overrides: Dict[str, torch.Tensor] = {}
    for name, val in zip(c_names, c_flat):
        gname = _CALIB_NAME_TO_GEOM.get(name, name)
        overrides[gname] = val
    return replace(base, **overrides)


def _build_model(
    hkls_cart: torch.Tensor,
    hkls_int: Optional[torch.Tensor],
    thetas: torch.Tensor,
    geometry: "midas_diffract.HEDMGeometry",
    scan_config: Optional["midas_diffract.ScanConfig"] = None,
):
    """Construct a fresh ``HEDMForwardModel`` for the current geometry."""
    from midas_diffract import HEDMForwardModel
    return HEDMForwardModel(
        hkls=hkls_cart,
        hkls_int=hkls_int,
        thetas=thetas,
        geometry=geometry,
        scan_config=scan_config,
    )


def _freeze_associations_at(
    model,
    grain_obs: GrainObs,
    *,
    max_match_dist: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(pred_idx, obs_idx)`` matched at MAP via nearest-neighbour.

    Mirrors :func:`midas_uq.laplace._freeze_associations` but accepts a
    ``GrainObs`` directly and exposes the obs-side index too (we need both
    sides since paper-1's residual is ``pred[pred_idx] - obs[obs_idx]``).
    """
    from midas_diffract import HEDMForwardModel
    with torch.no_grad():
        spots = model(
            grain_obs.euler_rad.unsqueeze(0),
            grain_obs.pos_um.unsqueeze(0),
            lattice_params=grain_obs.latc,
        )
        coords, valid = HEDMForwardModel.predict_spot_coords(
            spots, space="detector",
        )
        pred = coords.squeeze().reshape(-1, 3)
        vmask = valid.squeeze().reshape(-1) > 0.5
        valid_idx = torch.nonzero(vmask, as_tuple=False).squeeze(-1)
        pred_v = pred[vmask]
        dists = torch.cdist(grain_obs.observed_detector, pred_v)
        min_d, nn = dists.min(dim=1)
        keep = min_d < max_match_dist
        pred_idx = valid_idx[nn[keep]]
        obs_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    return pred_idx, obs_idx


def make_per_grain_residual(
    grain_obs: GrainObs,
    *,
    hkls_cart: torch.Tensor,
    hkls_int: Optional[torch.Tensor],
    thetas: torch.Tensor,
    base_geometry: "midas_diffract.HEDMGeometry",
    scan_config: Optional["midas_diffract.ScanConfig"],
    calibration_names: Sequence[str],
    calibration_map: torch.Tensor,
    sigma_obs_detector: torch.Tensor,
    max_match_dist: float = 5.0,
) -> Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], int]:
    """Return ``(r_fn, n_matched)`` for one grain.

    ``r_fn(g_flat, c_flat) -> (3·M,) tensor`` is the per-coordinate
    measurement residual (in units of σ_obs) for ``M`` spots whose MAP
    associations were frozen below ``max_match_dist`` (in pixel-equivalent
    units of the detector residual).

    The MAP geometry is reconstructed inside ``r_fn`` from ``c_flat`` so
    perturbations to calibration shift the projected detector coords.
    """
    from midas_diffract import HEDMForwardModel
    map_geom = _geometry_with_calibration(
        base_geometry, calibration_names, calibration_map,
    )
    map_model = _build_model(hkls_cart, hkls_int, thetas, map_geom, scan_config)
    pred_idx, obs_idx = _freeze_associations_at(
        map_model, grain_obs, max_match_dist=max_match_dist,
    )
    obs_match = grain_obs.observed_detector[obs_idx].detach().clone()

    sigma = sigma_obs_detector.detach().clone()
    # σ may be one of:
    #   (3,)        — per-coord scalar, broadcast across all spots
    #   (N_obs, 3)  — per-spot per-coord (heteroscedastic — paper-1 v1
    #                 with peak-fit FitRMSE per spot). Filter to the
    #                 matched-spot subset via obs_idx.
    if sigma.ndim == 2:
        if sigma.shape[0] != grain_obs.observed_detector.shape[0]:
            raise ValueError(
                f"per-spot sigma_obs_detector must have N_obs rows = "
                f"{grain_obs.observed_detector.shape[0]}; got {sigma.shape}"
            )
        sigma_match = sigma[obs_idx]   # (M, 3)
    else:
        sigma_match = sigma            # (3,) — broadcasts

    def r_fn(g_flat: torch.Tensor, c_flat: torch.Tensor) -> torch.Tensor:
        euler = g_flat[:3]
        latc = g_flat[3:9]
        pos = g_flat[9:12]
        geom = _geometry_with_calibration(base_geometry, calibration_names, c_flat)
        model = _build_model(hkls_cart, hkls_int, thetas, geom, scan_config)
        spots = model(euler.unsqueeze(0), pos.unsqueeze(0),
                       lattice_params=latc)
        coords, _ = HEDMForwardModel.predict_spot_coords(spots, space="detector")
        pred_flat = coords.squeeze().reshape(-1, 3)
        pred_match = pred_flat[pred_idx]
        if sigma_match.ndim == 1:
            resid = (pred_match - obs_match) / sigma_match.unsqueeze(0)
        else:
            resid = (pred_match - obs_match) / sigma_match
        return resid.flatten()

    return r_fn, int(obs_idx.numel())


def _fd_jacobian_on_c(
    r_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    g_map: torch.Tensor,
    c_map: torch.Tensor,
    *,
    rel_step: float = 1e-5,
    abs_floor: float = 1e-8,
) -> torch.Tensor:
    """Central-difference Jacobian ``dr/dc`` at ``(g_map, c_map)``.

    Step per parameter is ``max(rel_step * |c_j|, abs_floor)``. Used
    because ``HEDMForwardModel`` snapshots geometry through ``float(...)``
    casts that detach autograd on calibration inputs (see module
    docstring). Returns ``(M, n_c)``.
    """
    n_c = int(c_map.numel())
    with torch.no_grad():
        cols = []
        for j in range(n_c):
            step = max(rel_step * abs(float(c_map[j])), abs_floor)
            c_plus = c_map.clone()
            c_plus[j] = c_plus[j] + step
            c_minus = c_map.clone()
            c_minus[j] = c_minus[j] - step
            r_plus = r_fn(g_map, c_plus)
            r_minus = r_fn(g_map, c_minus)
            cols.append((r_plus - r_minus) / (2.0 * step))
        return torch.stack(cols, dim=-1)


def per_grain_hessian_blocks(
    grain_obs: GrainObs,
    *,
    hkls_cart: torch.Tensor,
    hkls_int: Optional[torch.Tensor],
    thetas: torch.Tensor,
    base_geometry: "midas_diffract.HEDMGeometry",
    scan_config: Optional["midas_diffract.ScanConfig"],
    calibration_names: Sequence[str],
    calibration_map: torch.Tensor,
    sigma_obs_detector: torch.Tensor,
    method: str = "fisher",
    max_match_dist: float = 0.5,
    fd_rel_step: float = 1e-5,
    fd_abs_floor: float = 1e-8,
) -> GrainHessianBlocks:
    """Compute ``(H_gg, H_gc)`` for one grain at MAP.

    Parameters
    ----------
    grain_obs : GrainObs
        MAP state + observed spots for one grain.
    calibration_names : sequence of str
        Calibration parameter names to free (paper-1 convention; e.g.
        ``["Lsd", "BC_y", "BC_z", "ty", "tz"]``). Must match the order
        used to build ``calibration_map`` AND the column order expected
        by ``midas_propagate.schur.per_grain_schur_marginal``'s ``H_gc``.
    calibration_map : (n_c,) tensor
        MAP values for the named calibration parameters.
    sigma_obs_angular : (3,) tensor
        Per-coordinate ``(σ_2θ, σ_η, σ_ω)`` measurement noise in radians.
    method : {'fisher', 'hessian'}
        Path for the Hessian blocks. See module docstring.
    max_match_dist : float
        Threshold (radians) for nearest-neighbour spot association at MAP.

    Returns
    -------
    GrainHessianBlocks
    """
    if method not in ("fisher", "hessian"):
        raise ValueError(f"method must be 'fisher' or 'hessian'; got {method!r}")

    r_fn, n_matched = make_per_grain_residual(
        grain_obs,
        hkls_cart=hkls_cart, hkls_int=hkls_int, thetas=thetas,
        base_geometry=base_geometry, scan_config=scan_config,
        calibration_names=calibration_names,
        calibration_map=calibration_map,
        sigma_obs_detector=sigma_obs_detector,
        max_match_dist=max_match_dist,
    )
    if n_matched == 0:
        raise RuntimeError(
            f"grain {grain_obs.spot_id}: no spots matched at MAP "
            f"(max_match_dist={max_match_dist}); cannot build Hessian blocks"
        )

    g_map = torch.cat([
        grain_obs.euler_rad.detach().reshape(-1),
        grain_obs.latc.detach().reshape(-1),
        grain_obs.pos_um.detach().reshape(-1),
    ]).clone()
    c_map = calibration_map.detach().clone()
    n_g = int(g_map.numel())
    n_c = int(c_map.numel())

    with torch.no_grad():
        r_map = r_fn(g_map, c_map)
    sigma_r_used = float(torch.sqrt((r_map ** 2).mean()).clamp(min=1e-30).item())

    # J_c via finite differences (HEDMForwardModel detaches autograd on
    # geometry inputs — see module docstring).
    J_c = _fd_jacobian_on_c(
        r_fn, g_map, c_map, rel_step=fd_rel_step, abs_floor=fd_abs_floor,
    )

    if method == "fisher":
        from torch.func import jacfwd
        # Forward-mode autograd on g is exact (model IS differentiable on
        # euler/pos/latc inputs to forward()). N_g << N_resid so jacfwd
        # beats jacrev.
        def r_of_g(g_flat):
            return r_fn(g_flat, c_map)
        J_g = jacfwd(r_of_g)(g_map)                   # (M*3, n_g)
        # Residual already normalised by sigma_obs in r_fn → Fisher is J^T J.
        H_gg = J_g.transpose(0, 1) @ J_g
        H_gc = J_g.transpose(0, 1) @ J_c
        H_cc_data = J_c.transpose(0, 1) @ J_c
    else:  # "hessian"
        def nll_g(g_flat):
            r = r_fn(g_flat, c_map)
            return 0.5 * (r * r).sum()
        H_gg = torch.autograd.functional.hessian(nll_g, g_map, vectorize=False)
        H_gg = 0.5 * (H_gg + H_gg.transpose(0, 1))
        # H_gc + H_cc_data via Gauss-Newton (J^T J) — autograd can't reach
        # the calibration leg in the dense Hessian, so we use the Fisher
        # form for both off-diagonal and the cc-data block.
        from torch.func import jacfwd
        def r_of_g(g_flat):
            return r_fn(g_flat, c_map)
        J_g = jacfwd(r_of_g)(g_map)
        H_gc = J_g.transpose(0, 1) @ J_c
        H_cc_data = J_c.transpose(0, 1) @ J_c

    return GrainHessianBlocks(
        H_gg=H_gg.detach(),
        H_gc=H_gc.detach(),
        H_cc_data=H_cc_data.detach(),
        n_spots_matched=n_matched,
        residual_at_map=r_map.detach(),
        sigma_r_used=sigma_r_used,
    )


__all__ = [
    "GrainObs",
    "GrainHessianBlocks",
    "make_per_grain_residual",
    "per_grain_hessian_blocks",
]
