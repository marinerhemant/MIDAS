"""Dict-functional HEDM residual closure for joint differentiable refinement.

Designed to be plugged into :func:`midas_peakfit.lm_minimise` as one of the
modalities in a joint loss. Operates on the canonical-named ``unpacked``
parameter dict produced by :func:`midas_peakfit.unpack_spec`:

    Lsd, BC_y, BC_z, tx, ty, tz, Wedge, Wavelength,
    grain_euler [N_g, 3], grain_pos [N_g, 3], grain_lattice [N_g, 6]

(plus optional per-detector replicas for multi-panel ``Lsd_per_det``,
``BC_y_per_det``, ``BC_z_per_det``, ``tilts_per_det``).

Autograd flows back through every refined entry via
:func:`torch.func.functional_call`, which substitutes the model's geometry
``nn.Parameter`` values at call-site without mutating the module state.
This is what makes joint optimisation work without a model rebuild per LM
step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from torch.func import functional_call

from midas_diffract import HEDMForwardModel  # type: ignore
from midas_diffract.forward import SpotDescriptors  # type: ignore

from .matching import MatchResult
from .observations import ObservedSpots
from .residuals import grain_residuals, _g_unit_from_pred  # noqa: F401


@dataclass
class HEDMResidualBundle:
    """Frozen per-grain inputs that don't change across LM iterations.

    Built once before the joint refinement; reused in every residual call.
    The mutable side (current geometry, current per-grain Euler/pos/strain)
    comes from the ``unpacked`` dict passed at residual-evaluation time.
    """
    model: HEDMForwardModel
    observations: List[ObservedSpots]
    matches: List[MatchResult]
    kind: str = "pixel"               # "pixel" | "angular" | "internal_angle"
    weights: Optional[List[torch.Tensor]] = None
    # Names in the unpacked dict that supply per-grain per-grain state.
    grain_euler_key: str = "grain_euler"     # (N_g, 3)
    grain_pos_key: str = "grain_pos"         # (N_g, 3)
    grain_lattice_key: str = "grain_lattice" # (N_g, 6)


def _build_param_overrides(
    unpacked: Dict[str, torch.Tensor],
    model: HEDMForwardModel,
) -> Dict[str, torch.Tensor]:
    """Construct the parameter override dict for ``functional_call``.

    Pulls Lsd, BC_y, BC_z, tilts, wedge from the canonical-named keys in
    ``unpacked`` (or falls back to the model's current values if a key is
    absent — which is the right behaviour for "geometry frozen, only grains
    refined" passes of an alternating driver).

    Multi-detector convention: scalar ``Lsd``, ``BC_y``, ``BC_z`` are
    broadcast to length-D vectors so they match the model's stored shape.
    For per-detector variation use the explicit ``*_per_det`` keys.
    """
    overrides: Dict[str, torch.Tensor] = {}
    D = int(model.n_distances)
    dev = model._Lsd.device
    dtype = model._Lsd.dtype

    def _as_vec(name: str, fallback: torch.Tensor) -> torch.Tensor:
        per_det = unpacked.get(f"{name}_per_det")
        scalar = unpacked.get(name)
        if per_det is not None:
            return per_det.reshape(D).to(dtype=dtype, device=dev)
        if scalar is not None:
            return scalar.reshape(()).to(dtype=dtype, device=dev).expand(D).contiguous()
        return fallback

    overrides["_Lsd"] = _as_vec("Lsd", model._Lsd.detach())
    overrides["_y_BC"] = _as_vec("BC_y", model._y_BC.detach())
    overrides["_z_BC"] = _as_vec("BC_z", model._z_BC.detach())

    # Tilts: per-detector (D, 3) tensor.  Three sources, in order of priority:
    #   1. tilts_per_det  [D, 3] — explicit multi-detector
    #   2. tx, ty, tz     scalars — broadcast across detectors
    #   3. model's current tilts  — fallback
    tilts_per_det = unpacked.get("tilts_per_det")
    if tilts_per_det is not None:
        overrides["tilts"] = tilts_per_det.reshape(D, 3).to(dtype=model.tilts.dtype, device=dev)
    elif "tx" in unpacked or "ty" in unpacked or "tz" in unpacked:
        tx = unpacked.get("tx", torch.tensor(float(model.tx), dtype=model.tilts.dtype, device=dev))
        ty = unpacked.get("ty", torch.tensor(float(model.ty), dtype=model.tilts.dtype, device=dev))
        tz = unpacked.get("tz", torch.tensor(float(model.tz), dtype=model.tilts.dtype, device=dev))
        row = torch.stack([
            tx.reshape(()).to(dtype=model.tilts.dtype, device=dev),
            ty.reshape(()).to(dtype=model.tilts.dtype, device=dev),
            tz.reshape(()).to(dtype=model.tilts.dtype, device=dev),
        ])
        overrides["tilts"] = row.unsqueeze(0).expand(D, 3).contiguous()

    if "Wedge" in unpacked:
        overrides["wedge"] = unpacked["Wedge"].reshape(()).to(
            dtype=model.wedge.dtype, device=dev)

    return overrides


def _vectorized_pixel_residual(
    unpacked: Dict[str, torch.Tensor],
    bundle: HEDMResidualBundle,
) -> torch.Tensor:
    """Fast path for ``kind == 'pixel'``: one batched forward over all grains.

    Replaces the per-grain Python loop in :func:`hedm_spot_residual` with a
    single ``functional_call`` that batches across grains, then a flat gather
    into the (N_g, K, M) prediction tensor. ~N_g× fewer kernel launches per LM
    iteration; eliminates the dispatch overhead that pinned the GPU at 0% util
    on real-data 20-grain joint refinement.
    """
    model = bundle.model
    overrides = _build_param_overrides(unpacked, model)

    grain_eulers = unpacked[bundle.grain_euler_key]
    grain_positions = unpacked[bundle.grain_pos_key]
    grain_lattices = unpacked[bundle.grain_lattice_key]
    n_grains = grain_eulers.shape[0]
    if n_grains != len(bundle.observations):
        raise ValueError(
            f"grain count mismatch: unpacked has {n_grains} but bundle has "
            f"{len(bundle.observations)} observation sets")

    px = float(model.px)
    y_BC0 = float(overrides.get("_y_BC", model._y_BC).detach()[0])
    z_BC0 = float(overrides.get("_z_BC", model._z_BC).detach()[0])

    # Concatenate per-grain index + observation arrays into (S_total,) views.
    g_list, k_list, m_list, mask_list, y_list, z_list = [], [], [], [], [], []
    w_list: Optional[List[torch.Tensor]] = [] if bundle.weights is not None else None
    for g in range(n_grains):
        obs_g = bundle.observations[g]
        match_g = bundle.matches[g]
        S = match_g.k_idx.shape[0]
        if S == 0:
            continue
        dev = match_g.k_idx.device
        g_list.append(torch.full((S,), g, dtype=torch.int64, device=dev))
        k_list.append(match_g.k_idx)
        m_list.append(match_g.m_idx)
        mask_list.append(match_g.mask)
        y_list.append(obs_g.y_lab)
        z_list.append(obs_g.z_lab)
        if w_list is not None:
            w_list.append(bundle.weights[g])

    if not g_list:
        return torch.zeros(0, dtype=grain_eulers.dtype, device=grain_eulers.device)

    g_idx = torch.cat(g_list)
    k_idx = torch.cat(k_list)
    m_idx = torch.cat(m_list)
    mask = torch.cat(mask_list)
    obs_y = torch.cat(y_list).to(dtype=grain_eulers.dtype)
    obs_z = torch.cat(z_list).to(dtype=grain_eulers.dtype)
    w_all = torch.cat(w_list).to(dtype=grain_eulers.dtype) if w_list else None

    # Single batched forward over all grains (B=N_g, N=1).
    eulers = grain_eulers.view(n_grains, 1, 3)
    positions = grain_positions.view(n_grains, 1, 3)
    lattices = grain_lattices.view(n_grains, 6)
    spots = functional_call(
        model, overrides,
        args=(eulers, positions),
        kwargs={"lattice_params": lattices},
    )
    yp = spots.y_pixel.squeeze(1)  # (N_g, K, M)
    zp = spots.z_pixel.squeeze(1)
    Ng, K, M = yp.shape

    flat_idx = g_idx * (K * M) + k_idx * M + m_idx
    pick_y = yp.reshape(-1).gather(0, flat_idx)
    pick_z = zp.reshape(-1).gather(0, flat_idx)

    obs_y_pix = y_BC0 - obs_y / px
    obs_z_pix = z_BC0 + obs_z / px

    r = torch.stack([pick_y - obs_y_pix, pick_z - obs_z_pix], dim=-1)
    mask_f = mask.to(r.dtype).unsqueeze(-1)
    r = r * mask_f
    if w_all is not None:
        r = r * w_all.view(-1, 1)
    return r.flatten()


def hedm_spot_residual(
    unpacked: Dict[str, torch.Tensor],
    bundle: HEDMResidualBundle,
) -> torch.Tensor:
    """Concatenated per-grain residual vector for joint LM.

    Returns a 1-D tensor; the joint pipeline weights and concatenates this
    with the powder pseudo-strain residual and any gauge/prior rows.

    Autograd flows back to every refined key in ``unpacked`` that the
    residual depends on:

        - geometry: Lsd, BC_y, BC_z, tx, ty, tz, Wedge (or *_per_det)
        - per-grain: grain_euler[g], grain_pos[g], grain_lattice[g]

    The model itself is *not* mutated; ``functional_call`` substitutes
    overrides at the forward call site.
    """
    if bundle.kind == "pixel":
        return _vectorized_pixel_residual(unpacked, bundle)

    model = bundle.model
    overrides = _build_param_overrides(unpacked, model)

    grain_eulers = unpacked[bundle.grain_euler_key]    # (N_g, 3)
    grain_positions = unpacked[bundle.grain_pos_key]   # (N_g, 3)
    grain_lattices = unpacked[bundle.grain_lattice_key]  # (N_g, 6)

    n_grains = grain_eulers.shape[0]
    if n_grains != len(bundle.observations):
        raise ValueError(
            f"grain count mismatch: unpacked has {n_grains} but bundle has "
            f"{len(bundle.observations)} observation sets")

    # Convenience aliases pulled out of the (possibly overridden) model.
    # Use ``.detach().cpu()`` so the float() conversion doesn't leak grads;
    # px / y_BC / z_BC are plumbed into ``grain_residuals`` as Python floats
    # for the obs-pixel-coord conversion but the actual residual values come
    # from the model's pixel projection (which DOES carry grad through Lsd
    # etc. via the ``functional_call`` overrides).
    px = float(model.px)
    y_BC0 = float(overrides.get("_y_BC", model._y_BC).detach()[0])
    z_BC0 = float(overrides.get("_z_BC", model._z_BC).detach()[0])

    pieces: List[torch.Tensor] = []
    for g in range(n_grains):
        euler_g = grain_eulers[g]
        pos_g = grain_positions[g]
        latc_g = grain_lattices[g]
        obs_g = bundle.observations[g]
        match_g = bundle.matches[g]
        w_g = bundle.weights[g] if bundle.weights is not None else None

        # functional_call substitutes model parameters at call-site.
        def _forward_call(eu, po, la):
            return functional_call(
                model, overrides,
                args=(eu.view(1, 1, 3), po.view(1, 1, 3)),
                kwargs={"lattice_params": la.view(1, 6)},
            )

        spots = _forward_call(euler_g, pos_g, latc_g)

        # Squeeze leading (B=1, N=1) dims to match grain_residuals'
        # expected (K, M) shape.
        def _drop_lead(t: torch.Tensor) -> torch.Tensor:
            while t.dim() > 2 and t.shape[0] == 1:
                t = t.squeeze(0)
            return t

        spots_sg = SpotDescriptors(
            omega=_drop_lead(spots.omega),
            eta=_drop_lead(spots.eta),
            two_theta=_drop_lead(spots.two_theta),
            y_pixel=_drop_lead(spots.y_pixel),
            z_pixel=_drop_lead(spots.z_pixel),
            frame_nr=_drop_lead(spots.frame_nr),
            valid=_drop_lead(spots.valid),
        )

        # Reuse the existing per-grain residual logic by hand-rolling the
        # gather + diff pipeline (we can't call ``grain_residuals`` directly
        # because it constructs the model output internally — we already
        # have the spots from functional_call).
        S = match_g.k_idx.shape[0]
        if S == 0:
            continue
        flat_idx = match_g.k_idx * spots_sg.omega.shape[-1] + match_g.m_idx

        def _pick(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(-1).gather(0, flat_idx)

        pick_y = _pick(spots_sg.y_pixel)
        pick_z = _pick(spots_sg.z_pixel)

        if bundle.kind == "pixel":
            obs_y_pixel = y_BC0 - obs_g.y_lab / px
            obs_z_pixel = z_BC0 + obs_g.z_lab / px
            r = torch.stack([pick_y - obs_y_pixel,
                              pick_z - obs_z_pixel], dim=-1)
        elif bundle.kind == "angular":
            from .residuals import _angular_diff
            r = torch.stack([
                _angular_diff(_pick(spots_sg.two_theta), obs_g.two_theta),
                _angular_diff(_pick(spots_sg.eta), obs_g.eta),
                _angular_diff(_pick(spots_sg.omega), obs_g.omega),
            ], dim=-1)
        elif bundle.kind == "internal_angle":
            pick_theta = _pick(spots_sg.two_theta)
            pick_eta = _pick(spots_sg.eta)
            pick_om = _pick(spots_sg.omega)
            g_obs = obs_g.g_unit_lab()
            g_pred = _g_unit_from_pred({
                "two_theta": pick_theta, "eta": pick_eta, "omega": pick_om,
            })
            cos_ang = (g_pred * g_obs).sum(dim=-1).abs().clamp(0.0, 1.0 - 1e-12)
            r = torch.acos(cos_ang).unsqueeze(-1)
        else:
            raise ValueError(f"unknown bundle.kind {bundle.kind!r}")

        mask = match_g.mask.to(r.dtype).unsqueeze(-1)
        r = r * mask
        if w_g is not None:
            r = r * w_g.view(-1, 1).to(r.dtype)
        pieces.append(r.flatten())

    if not pieces:
        return torch.zeros(0, dtype=grain_eulers.dtype, device=grain_eulers.device)
    return torch.cat(pieces)


__all__ = ["HEDMResidualBundle", "hedm_spot_residual"]
