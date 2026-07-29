"""Physics-model coupling: infer mesoscale parameters through the DFXM forward.

Item #1 of the post-Phase-5 roadmap — the highest-novelty capability, and the one
hardest to scoop piecemeal: fit a *physical* model's parameters end-to-end through
the differentiable DFXM forward, rather than recovering a raw field.

The proof of concept here is the geometrically-necessary-dislocation (GND) content
of a single-slip lattice curvature. Nye/Ashby: a GND density ``rho`` on one slip
system bends the lattice at curvature ``kappa = rho * b`` (b the Burgers magnitude),
producing an orientation gradient the DFXM mosaicity scan sees. We parameterise the
deformation field by the single physical scalar ``rho`` and fit it by matching the
DFXM mosaicity stack — recovering a dislocation density, not just a rotation map.
This is the DFXM<->dislocation-dynamics interface (cf. Henningsson 2025) done
differentiably.

A discrete-dislocation analogue (``fit_wall_spacing``) recovers a low-angle tilt
boundary's dislocation spacing ``D`` (Frank's relation ``theta = b / D``) through the
anisotropic Stroh forward — a genuine DDD parameter fit.

Everything is torch-differentiable and device-portable.
"""
from __future__ import annotations

import torch

from midas_invert.optimize import fit

from .conventions import GoniometerSetting
from .field import DeformationField
from .forward import voxel_intensity
from .resolution import ResolutionFunction, aligned_resolution
from .scan import reference_q_nom

_ANGSTROM_PER_UM = 1e4
_DEG_PER_RAD = 57.29577951308232


def gnd_curvature_field(
    rho_gnd_per_um2,
    positions: torch.Tensor,
    *,
    burgers_length_A: float = 2.556,
    curvature_axis=(0.0, 0.0, 1.0),
    along: int = 0,
    orientation=None,
    lattice_params=(3.6356, 3.6356, 3.6356, 90.0, 90.0, 90.0),
    shape=None,
) -> DeformationField:
    """Lattice-curvature field from a single-slip GND density (Nye ``kappa = rho b``).

    ``rho_gnd_per_um2`` is the GND density (dislocations / um^2); ``kappa = rho * b``
    (rad/um) is the lattice curvature; the field rotates about ``curvature_axis``
    linearly along spatial direction ``along``. Differentiable in ``rho_gnd_per_um2``,
    so it can be *fit* through the DFXM forward (:func:`fit_gnd_density`).
    """
    device, dtype = positions.device, positions.dtype
    if not isinstance(rho_gnd_per_um2, torch.Tensor):
        rho_gnd_per_um2 = torch.as_tensor(rho_gnd_per_um2, device=device, dtype=dtype)
    b_um = burgers_length_A / _ANGSTROM_PER_UM
    kappa = rho_gnd_per_um2 * b_um                      # rad/um
    angle_rad = kappa * positions[:, along]             # (N,)
    axis = torch.as_tensor(curvature_axis, device=device, dtype=dtype)
    axis = axis / torch.linalg.vector_norm(axis)
    # Rodrigues rotation per voxel (kept local so grad flows through kappa).
    K = torch.zeros(3, 3, device=device, dtype=dtype)
    K[0, 1], K[0, 2] = -axis[2], axis[1]
    K[1, 0], K[1, 2] = axis[2], -axis[0]
    K[2, 0], K[2, 1] = -axis[1], axis[0]
    a = angle_rad[:, None, None]
    eye = torch.eye(3, device=device, dtype=dtype)
    R = eye + torch.sin(a) * K + (1 - torch.cos(a)) * (K @ K)   # (N,3,3)
    if orientation is None:
        orientation = torch.eye(3, device=device, dtype=dtype)
    else:
        orientation = torch.as_tensor(orientation, device=device, dtype=dtype)
    latc = torch.as_tensor(lattice_params, device=device, dtype=dtype)
    return DeformationField(
        positions=positions, F=R,
        reference_orientation=orientation, lattice_params=latc, shape=shape,
    )


def fit_gnd_density(
    observed_stack: torch.Tensor,
    settings,
    positions: torch.Tensor,
    hkl,
    *,
    rho_init: float = 1.0,
    burgers_length_A: float = 2.556,
    curvature_axis=(0.0, 0.0, 1.0),
    along: int = 0,
    sigma_par: float = 8e-3,
    sigma_perp: float = 4e-3,
    steps: int = 300,
    lr: float = 0.05,
) -> dict:
    """Recover the GND density from a DFXM mosaicity stack (differentiable fit).

    ``observed_stack`` is ``(S, N)`` per-voxel intensities over ``S`` goniometer
    settings. Fits the single physical scalar ``rho`` so the modelled mosaicity stack
    matches. Returns ``{'rho', 'curvature_rad_per_um', 'loss'}``. The recovered ``rho``
    is a dislocation density inferred end-to-end through the imaging forward.
    """
    device, dtype = positions.device, positions.dtype
    rho = torch.tensor(float(rho_init), device=device, dtype=dtype, requires_grad=True)
    center = GoniometerSetting()
    ref_field = gnd_curvature_field(rho.detach(), positions, burgers_length_A=burgers_length_A,
                                    curvature_axis=curvature_axis, along=along)
    q_nom = reference_q_nom(ref_field, hkl, center)
    res = aligned_resolution(q_nom, sigma_par=sigma_par, sigma_perp=sigma_perp)
    obs = observed_stack / (observed_stack.abs().max() + 1e-30)

    def loss_fn():
        field = gnd_curvature_field(rho, positions, burgers_length_A=burgers_length_A,
                                    curvature_axis=curvature_axis, along=along)
        pred = torch.stack([voxel_intensity(field, hkl, s, res) for s in settings], dim=0)
        pred = pred / (pred.abs().max() + 1e-30)
        return ((pred - obs) ** 2).mean()

    fit([rho], loss_fn, steps=steps, lr=lr)
    b_um = burgers_length_A / _ANGSTROM_PER_UM
    return {
        "rho": float(rho.detach()),
        "curvature_rad_per_um": float(rho.detach()) * b_um,
        "loss": float(loss_fn().detach()),
    }


def fit_wall_spacing(
    observed_stack: torch.Tensor,
    settings,
    positions: torch.Tensor,
    hkl,
    dislocations_builder,
    *,
    resolution: ResolutionFunction | None = None,
    spacing_bounds=(1.0, 8.0),
    n_coarse: int = 15,
    steps: int = 150,
    lr: float = 0.05,
) -> dict:
    """Recover a tilt-boundary dislocation spacing ``D`` from a DFXM mosaicity stack.

    ``observed_stack`` is the **per-voxel** stack ``(S, N)`` (spatially resolved — the
    integrated curve is too weakly sensitive to ``D``). ``dislocations_builder(D)``
    returns a list of :class:`StrohDislocation` at spacing ``D`` (differentiable via the
    core positions). Frank's relation ``theta = b / D`` is the physical check.

    The residual-vs-``D`` landscape is multimodal (spacings can alias to similar tilt
    patterns), so we coarse-scan ``D`` over ``spacing_bounds`` before gradient-refining —
    the same coarse->fine pattern used for core-position and thickness fits. Returns
    ``{'spacing', 'loss'}``.
    """
    from .dislocation import dislocation_deformation_field

    device, dtype = positions.device, positions.dtype
    center = GoniometerSetting()
    if resolution is None:
        f0 = dislocation_deformation_field(positions, dislocations_builder(
            torch.tensor(float(sum(spacing_bounds) / 2), device=device, dtype=dtype)))
        q_nom = reference_q_nom(f0, hkl, center)
        resolution = aligned_resolution(q_nom, sigma_par=8e-3, sigma_perp=6e-3)
    obs = observed_stack / (observed_stack.abs().max() + 1e-30)

    def predict(D):
        field = dislocation_deformation_field(positions, dislocations_builder(D))
        pred = torch.stack([voxel_intensity(field, hkl, s, resolution) for s in settings], dim=0)
        return pred / (pred.abs().max() + 1e-30)

    def residual(D):
        return ((predict(D) - obs) ** 2).mean()

    # Coarse scan for a robust init.
    grid = torch.linspace(spacing_bounds[0], spacing_bounds[1], n_coarse, dtype=dtype, device=device)
    with torch.no_grad():
        best = min(grid.tolist(), key=lambda d: float(residual(torch.tensor(d, dtype=dtype, device=device))))
    D = torch.tensor(float(best), device=device, dtype=dtype, requires_grad=True)
    fit([D], lambda: residual(D), steps=steps, lr=lr)
    with torch.no_grad():
        loss = float(residual(D))
    return {"spacing": float(D.detach()), "loss": loss}
