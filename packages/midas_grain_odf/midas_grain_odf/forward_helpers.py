"""Adapters around midas_diffract.forward.HEDMForwardModel.

The differentiable HEDM forward model is provided by the ``midas-diffract``
package (Sharma et al., IUCrJ 2025). That model takes Euler angles; ODF
samples are most naturally represented as rotation matrices, so we expose
a thin wrapper that calls the geometry kernels directly with matrices,
skipping the euler2mat conversion.

torch.compile path
------------------
When the host ``HEDMForwardModel`` was built with ``compile=True`` (i.e.
``model._compile_mode`` is set to a non-None mode string), this module
transparently routes ``forward_orientations`` and ``forward_pairs`` through
a cached compiled version. The compile-wrap is per-(model_id, function)
to avoid retracing across distinct model instances, and the compile mode
mirrors what the caller asked for at model construction.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from midas_diffract.forward import HEDMForwardModel, SpotDescriptors


# Cache: (id(model), kind) -> compiled callable. We key by id(model) so each
# distinct HEDMForwardModel instance gets its own compiled specialisation;
# distinct devices/dtypes are handled inside dynamo's own caching.
_COMPILED_CACHE: dict = {}


def _maybe_compile(model: HEDMForwardModel, kind: str, fn):
    """Return a compiled wrapper around fn if model._compile_mode is set.

    Eager fallthrough is intentional when compile is off so callers pay no
    overhead. The compile is keyed by id(model) so two distinct
    HEDMForwardModel instances do not contend for the same cache slot;
    dynamo handles per-shape specialisations within each cache entry.
    """
    mode = getattr(model, "_compile_mode", None)
    if mode is None:
        return fn
    key = (id(model), kind)
    cached = _COMPILED_CACHE.get(key)
    if cached is not None:
        return cached
    compiled = torch.compile(fn, mode=mode, dynamic=False)
    _COMPILED_CACHE[key] = compiled
    return compiled


def _forward_orientations_eager(
    model: HEDMForwardModel,
    OM: torch.Tensor,
    pos: torch.Tensor,
    lattice_params: Optional[torch.Tensor],
    strain: Optional[torch.Tensor],
) -> SpotDescriptors:
    """Tensor-only inner. All shape validation done by the caller."""
    hkls_cart = None
    thetas = None
    if lattice_params is not None:
        hkls_cart, thetas = model.correct_hkls_latc(lattice_params, strain=strain)

    omega, eta, two_theta, valid = model.calc_bragg_geometry(
        OM, hkls_cart, thetas
    )
    spots = model.project_to_detector(omega, eta, two_theta, pos, valid)

    if model.scan_config is not None:
        spots = model.filter_by_scan(spots, pos)
    return spots


def forward_orientations(
    model: HEDMForwardModel,
    orientation_matrices: torch.Tensor,
    position: torch.Tensor,
    lattice_params: Optional[torch.Tensor] = None,
    strain: Optional[torch.Tensor] = None,
) -> SpotDescriptors:
    """Run the forward model from rotation matrices (skip euler2mat).

    Parameters
    ----------
    model : HEDMForwardModel
    orientation_matrices : Tensor (K, 3, 3) or (..., 3, 3)
        K orientations to evaluate at the same grain position.
    position : Tensor (3,)
        Single grain centroid in micrometers (sample frame).
    lattice_params : Tensor (6,) or None
        [a,b,c,alpha,beta,gamma]. Required if strain is given.
    strain : Tensor (6,) or None
        Crystal-frame Voigt strain.

    Returns
    -------
    SpotDescriptors with shape (..., 2, M)  -- K orientations × 2 omega
    solutions × M reflections at one position.
    """
    if orientation_matrices.dim() < 3:
        raise ValueError(
            "orientation_matrices must have shape (..., 3, 3); "
            f"got {tuple(orientation_matrices.shape)}"
        )
    if lattice_params is None and strain is not None:
        raise ValueError("strain requires lattice_params to be supplied.")

    # Promote orientation matrices to (..., N=1, 3, 3) so the geometry
    # kernel's "N voxels" dimension is preserved.
    OM = orientation_matrices.unsqueeze(-3)  # (..., 1, 3, 3)
    pos = position.reshape(1, 3).to(OM.dtype).to(OM.device)

    fn = _maybe_compile(model, "forward_orientations", _forward_orientations_eager)
    return fn(model, OM, pos, lattice_params, strain)


def grain_avg_predicted_spots(
    model: HEDMForwardModel,
    R_avg: torch.Tensor,
    position: torch.Tensor,
    lattice_params: Optional[torch.Tensor] = None,
) -> SpotDescriptors:
    """Forward-simulate the grain-averaged single-orientation prediction.

    Convenience wrapper for the upstream-fit path used to build the
    initial Delta table.
    """
    R = R_avg.reshape(1, 3, 3)
    return forward_orientations(model, R, position, lattice_params=lattice_params)


def forward_pairs(
    model: HEDMForwardModel,
    orientation_matrices: torch.Tensor,
    positions: torch.Tensor,
    lattice_params: Optional[torch.Tensor] = None,
) -> SpotDescriptors:
    """Multi-grain forward path: P (orientation, position) pairs in one call.

    Parameters
    ----------
    orientation_matrices : Tensor (P, 3, 3)
        One orientation per pair.
    positions : Tensor (P, 3)
        One centroid per pair (matches the orientation row-by-row).
    lattice_params : optional, (6,) or (P, 6)

    Returns
    -------
    SpotDescriptors with shapes (1, 2P, M) where M = number of HKLs.
    Reshape to ``(P, 2, M)`` to recover per-pair spot lists.

    The leading "1" comes from the model's batch convention (see
    ``midas_diffract.forward.HEDMForwardModel.calc_bragg_geometry`` /
    ``project_to_detector``); we expose this single API call instead of
    looping over P pairs and incurring P kernel launches.
    """
    if orientation_matrices.dim() != 3 or orientation_matrices.shape[1:] != (3, 3):
        raise ValueError(
            "orientation_matrices must have shape (P, 3, 3); got "
            f"{tuple(orientation_matrices.shape)}"
        )
    if positions.dim() != 2 or positions.shape[1] != 3:
        raise ValueError(
            "positions must have shape (P, 3); got "
            f"{tuple(positions.shape)}"
        )
    if positions.shape[0] != orientation_matrices.shape[0]:
        raise ValueError(
            "orientation_matrices and positions must agree on the leading "
            f"P dimension; got {orientation_matrices.shape[0]} vs "
            f"{positions.shape[0]}"
        )

    OM = orientation_matrices.unsqueeze(0)   # (1, P, 3, 3)
    pos = positions.to(OM.dtype).to(OM.device)

    fn = _maybe_compile(model, "forward_pairs", _forward_pairs_eager)
    return fn(model, OM, pos, lattice_params)


def _forward_pairs_eager(
    model: HEDMForwardModel,
    OM: torch.Tensor,
    pos: torch.Tensor,
    lattice_params: Optional[torch.Tensor],
) -> SpotDescriptors:
    """Tensor-only inner of forward_pairs (compile target)."""
    hkls_cart = None
    thetas = None
    if lattice_params is not None:
        hkls_cart, thetas = model.correct_hkls_latc(lattice_params, strain=None)

    omega, eta, two_theta, valid = model.calc_bragg_geometry(
        OM, hkls_cart, thetas
    )
    spots = model.project_to_detector(omega, eta, two_theta, pos, valid)

    if model.scan_config is not None:
        spots = model.filter_by_scan(spots, pos)
    return spots
