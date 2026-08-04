"""Joint per-grain peak-shape inversion driver.

Free parameters per grain:
    δ_V        (G, 3)   axis-angle perturbation around R_init,V (radians)
    ε_V        (G, 6)   per-voxel Voigt strain (crystal frame)
    lattice    (6,)     global per-grain reference lattice [a,b,c,α,β,γ]

Identifiability: the global lattice and per-voxel strain trade off — a
uniform shift in ε_V is geometrically the same as a shift in lattice.
Two operating modes (selected via ``IdentifiabilityMode``):

  PROJECT_EPS_MEAN_ZERO (default):
      ε_V is internally projected to mean-zero across voxels at every
      forward pass. The lattice absorbs the bulk strain. One global
      reference per grain — physical, well-defined.

  FREE:
      ε_V left free, lattice fixed at warm-start. The C-style
      over-parameterized convention. Useful for direct comparison
      against ``FitOrStrainsScanningOMP`` output.

Loss = data MSE + smoothness regularizer (default off).

Outputs are returned via :class:`GrainPeakFitResult`.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from midas_diffract.forward import HEDMForwardModel, ScanConfig
from midas_grain_odf.spot_extract import SpotPatchSpec

from midas_pf_odf.forward import joint_grain_forward, closed_form_per_spot_scale
from midas_pf_odf.simulate import GrainPatchData


def _aa_to_R(axis_angle: torch.Tensor) -> torch.Tensor:
    """Axis-angle → rotation matrix via Rodrigues. Smooth at zero.

    Unlike ``midas_grain_odf.odf.axis_angle_to_matrix`` which uses
    ``torch.where(near_zero, I, R)`` and **kills the gradient at
    zero**, this implementation evaluates the Taylor expansions of
    ``sin(θ)/θ`` and ``(1−cos(θ))/θ²`` directly. Both are analytic at
    θ = 0 and PyTorch autograd handles the limit correctly.
    """
    eps = 1e-12
    theta_sq = (axis_angle ** 2).sum(dim=-1, keepdim=True)
    theta = (theta_sq + eps).sqrt()                    # smooth, ≥ √eps
    # Avoid 0/0 by carrying θ through the sin/(1−cos)/θ ratios.
    sinc = torch.sin(theta) / theta                     # = sin(θ)/θ
    one_minus_cos_over_theta_sq = (1.0 - torch.cos(theta)) / (theta * theta)

    aa = axis_angle.unsqueeze(-1)                       # (..., 3, 1)
    # Skew-symmetric K from axis-angle (no normalization needed — sinc and
    # the (1−cos)/θ² factors absorb the magnitude).
    zero = torch.zeros_like(axis_angle[..., 0])
    K = torch.stack([
        torch.stack([zero,                  -axis_angle[..., 2],  axis_angle[..., 1]], dim=-1),
        torch.stack([ axis_angle[..., 2],   zero,                -axis_angle[..., 0]], dim=-1),
        torch.stack([-axis_angle[..., 1],   axis_angle[..., 0],  zero               ], dim=-1),
    ], dim=-2)                                          # (..., 3, 3)

    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    K2 = K @ K
    s = sinc.unsqueeze(-1)                              # (..., 1, 1) for matmul broadcast
    c = one_minus_cos_over_theta_sq.unsqueeze(-1)
    return eye + s * K + c * K2


class IdentifiabilityMode(str, enum.Enum):
    PROJECT_EPS_MEAN_ZERO = "project_eps_mean_zero"
    FREE = "free"


@dataclass
class GrainPeakFitResult:
    R_fit: torch.Tensor              # (G, 3, 3)
    eps_fit: torch.Tensor            # (G, 6)
    lattice_fit: torch.Tensor        # (6,)
    c_per_spot: torch.Tensor         # (S,)
    pos_fit: Optional[torch.Tensor] = None   # (G, 3) refined voxel positions (if refine_position)
    spread_fit: Optional[torch.Tensor] = None  # (G,) intra-voxel peak-width spread (if refine_spread)
    strain_spread_fit: Optional[torch.Tensor] = None  # (G,) intra-voxel radial (microstrain) spread
    spot_offset_fit: Optional[torch.Tensor] = None  # (S, 2) per-spot detector offset (if refine_spot_offset)
    losses: List[float] = field(default_factory=list)
    holdout_score: Optional[float] = None
    converged: bool = False
    optimizer_used: str = ""
    voxel_holdout_r2: Optional[torch.Tensor] = None     # (G,)
    metadata: dict = field(default_factory=dict)


def _resolve_optimizer_name(name: Optional[str]) -> str:
    if name is None:
        import os
        return os.environ.get("MIDAS_OPTIMIZER", "adam").lower()
    return str(name).lower()


def _build_optimizer(
    name: str,
    delta: torch.Tensor,
    eps_param: torch.Tensor,
    lattice_param: torch.Tensor,
    *,
    lr_aa: float,
    lr_eps: float,
    lr_lat: float,
    extra: "Optional[List[tuple]]" = None,
    lbfgs_max_iter: int = 20,
    lbfgs_history: int = 10,
):
    """Build the optimizer over the parameter groups.

    Any group with lr == 0 is treated as **locked**: ``requires_grad`` is
    cleared so the param doesn't enter autograd, and it is excluded from
    the optimizer's parameter list. This is the cleanest way to lock
    params in L-BFGS (which uses a single global lr and would otherwise
    update locked params). ``extra`` is an optional list of ``(param, lr)``
    pairs (e.g. a voxel-position perturbation).
    """
    pairs = [(delta, lr_aa), (eps_param, lr_eps), (lattice_param, lr_lat)]
    pairs += list(extra or [])
    for p, lr in pairs:
        if lr == 0.0:
            p.requires_grad_(False)

    active = [(p, lr) for (p, lr) in pairs if p.requires_grad]
    if not active:
        # All-locked = a warm-start sanity check (no movement). Return None;
        # the caller skips the optimizer loop.
        return None

    if name == "lbfgs":
        return torch.optim.LBFGS(
            [p for p, _ in active],
            lr=max(lr for _, lr in active),
            max_iter=lbfgs_max_iter,
            history_size=lbfgs_history,
            line_search_fn="strong_wolfe",
        )
    return torch.optim.Adam([{"params": [p], "lr": lr} for p, lr in active])


def _project_mean_zero(eps_param: torch.Tensor) -> torch.Tensor:
    """Return ε with column-mean subtracted (per Voigt component)."""
    return eps_param - eps_param.mean(dim=0, keepdim=True)


def _neighbor_smooth_loss(
    delta: torch.Tensor, eps: torch.Tensor, grid_shape: tuple,
) -> torch.Tensor:
    """4-conn neighbor difference penalty on (δ, ε) for a 2D voxel grid."""
    Gx, Gy = grid_shape
    if delta.shape[0] != Gx * Gy:
        raise ValueError(
            f"lambda_smooth>0 with grid_shape {tuple(grid_shape)} but "
            f"G={delta.shape[0]} voxels — the grain is a SPARSE subset of "
            "the scan grid (P2-8). Pass neighbor_edges="
            "neighbor_edges_from_grid_ij(dataset.grid_ij) instead of "
            "grid_shape."
        )
    dG = delta.reshape(Gx, Gy, 3)
    eG = eps.reshape(Gx, Gy, 6)
    dx = dG[1:, :] - dG[:-1, :]
    dy = dG[:, 1:] - dG[:, :-1]
    ex = eG[1:, :] - eG[:-1, :]
    ey = eG[:, 1:] - eG[:, :-1]
    return (dx ** 2).sum() + (dy ** 2).sum() + (ex ** 2).sum() + (ey ** 2).sum()


def _graph_smooth_loss(
    delta: torch.Tensor, eps: torch.Tensor, edges: torch.Tensor,
    *, eps_only: bool = False,
) -> torch.Tensor:
    """Neighbour-difference penalty over an explicit edge list.

    For an irregular grain (not a full rectangle), spatial neighbours are
    supplied as ``edges`` of shape ``(E, 2)`` (voxel index pairs). Penalises
    ``Σ_e ||ε_u − ε_v||²`` (and ``||δ_u − δ_v||²`` unless ``eps_only``). This
    is the workhorse regulariser for real grains where the boundary/low-
    coverage voxels are otherwise under-determined and over-fit.
    """
    u, v = edges[:, 0], edges[:, 1]
    de = eps[u] - eps[v]
    out = (de ** 2).sum()
    if not eps_only:
        dd = delta[u] - delta[v]
        out = out + (dd ** 2).sum()
    return out


def build_compat_stencils(grid_ij: "np.ndarray") -> dict:
    """Precompute finite-difference index sets for the 2D St-Venant
    compatibility penalty on an irregular grain.

    ``grid_ij`` is ``(G, 2)`` integer (row=i↔sample-x, col=j↔sample-y) grid
    positions of each voxel. For every voxel that has all of ``±i, ±j`` and
    the four diagonal neighbours present, we record the index octet needed for
    the in-plane compatibility residual (see :func:`_compatibility_loss`).
    Returns ``{} `` if no interior voxel qualifies.
    """
    import numpy as np
    gi = np.asarray(grid_ij).astype(int)
    pos = {(int(i), int(j)): k for k, (i, j) in enumerate(gi)}

    def g(i, j):
        return pos.get((i, j))

    c, xp, xm, yp, ym = [], [], [], [], []
    pp, pm, mp, mm = [], [], [], []
    for k, (i, j) in enumerate(gi):
        n = {d: g(i + di, j + dj) for d, (di, dj) in {
            "xp": (1, 0), "xm": (-1, 0), "yp": (0, 1), "ym": (0, -1),
            "pp": (1, 1), "pm": (1, -1), "mp": (-1, 1), "mm": (-1, -1)}.items()}
        if any(v is None for v in n.values()):
            continue
        c.append(k); xp.append(n["xp"]); xm.append(n["xm"])
        yp.append(n["yp"]); ym.append(n["ym"])
        pp.append(n["pp"]); pm.append(n["pm"]); mp.append(n["mp"]); mm.append(n["mm"])
    if not c:
        return {}
    t = lambda a: torch.tensor(a, dtype=torch.long)
    return {"c": t(c), "xp": t(xp), "xm": t(xm), "yp": t(yp), "ym": t(ym),
            "pp": t(pp), "pm": t(pm), "mp": t(mp), "mm": t(mm)}


def build_stiffness_voigt(c6: "np.ndarray") -> torch.Tensor:
    """Expand a 6x6 Voigt stiffness ``c6`` (GPa) into the rank-4 tensor
    ``C_ijkl`` (3,3,3,3) so that ``σ_ij = Σ_kl C_ijkl ε_kl`` for a *true*
    (tensor, not engineering) strain. The factor-of-2 for shear is absorbed by
    summing both off-diagonal index pairs, so no engineering-strain bookkeeping
    is needed downstream.

    Voigt index map (0-based): (0,0)->0 (1,1)->1 (2,2)->2 (1,2)->3 (0,2)->4
    (0,1)->5, symmetric in (i,j) and (k,l).
    """
    import numpy as np
    c6 = np.asarray(c6, dtype=float)
    vmap = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 1): 3,
            (0, 2): 4, (2, 0): 4, (0, 1): 5, (1, 0): 5}
    C4 = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C4[i, j, k, l] = c6[vmap[(i, j)], vmap[(k, l)]]
    return torch.as_tensor(C4)


def hcp_stiffness_voigt(c11: float, c12: float, c13: float,
                        c33: float, c44: float) -> torch.Tensor:
    """Rank-4 stiffness for an hcp (6mm, transversely isotropic) crystal with
    the c-axis along index 3. ``c66 = (c11-c12)/2`` is implied. Constants in GPa.
    """
    import numpy as np
    c66 = 0.5 * (c11 - c12)
    c6 = np.array([
        [c11, c12, c13, 0, 0, 0],
        [c12, c11, c13, 0, 0, 0],
        [c13, c13, c33, 0, 0, 0],
        [0, 0, 0, c44, 0, 0],
        [0, 0, 0, 0, c44, 0],
        [0, 0, 0, 0, 0, c66]])
    return build_stiffness_voigt(c6)


def _equilibrium_loss(eps: torch.Tensor, R: torch.Tensor, st: dict,
                      C4: torch.Tensor) -> torch.Tensor:
    """In-plane stress-equilibrium penalty ``Σ_voxel ‖(∇·σ)_in-plane‖²``.

    σ = C:ε is computed per voxel in the crystal frame and rotated into the
    sample frame (``σ_sam = R σ_crys Rᵀ``, R = crystal→sample). With a single
    scanned layer only the in-plane spatial gradients ∂/∂x (row i) and ∂/∂y
    (col j) are available, so we enforce the projected balance
    ``∂σ_aj/∂x_j ≈ 0`` dropping the ∂/∂z term, for a ∈ {x,y,z}. This is the
    *elastic-constant* prior — physically distinct from (and complementary to)
    the kinematic St-Venant :func:`_compatibility_loss`, with a different null
    space. Voigt order [xx,xy,xz,yy,yz,zz]; unit grid spacing.
    """
    G = eps.shape[0]
    et = eps.new_zeros(G, 3, 3)
    et[:, 0, 0] = eps[:, 0]
    et[:, 0, 1] = et[:, 1, 0] = eps[:, 1]
    et[:, 0, 2] = et[:, 2, 0] = eps[:, 2]
    et[:, 1, 1] = eps[:, 3]
    et[:, 1, 2] = et[:, 2, 1] = eps[:, 4]
    et[:, 2, 2] = eps[:, 5]
    C4 = C4.to(eps.dtype).to(eps.device)
    sig_c = torch.einsum("ijkl,gkl->gij", C4, et)            # crystal stress
    sig = torch.einsum("gip,gpq,gjq->gij", R, sig_c, R)      # sample stress
    c, xp, xm, yp, ym = st["c"], st["xp"], st["xm"], st["yp"], st["ym"]
    dsig_dx = 0.5 * (sig[xp] - sig[xm])                      # (Nc,3,3)
    dsig_dy = 0.5 * (sig[yp] - sig[ym])
    rx = dsig_dx[:, 0, 0] + dsig_dy[:, 0, 1]
    ry = dsig_dx[:, 1, 0] + dsig_dy[:, 1, 1]
    rz = dsig_dx[:, 2, 0] + dsig_dy[:, 2, 1]
    return (rx ** 2 + ry ** 2 + rz ** 2).sum()


def _compatibility_loss(eps: torch.Tensor, st: dict) -> torch.Tensor:
    """2D in-plane St-Venant compatibility penalty (unit grid spacing).

    Enforces ``∂²ε_xx/∂y² + ∂²ε_yy/∂x² − 2 ∂²ε_xy/∂x∂y = 0`` — the condition
    that a strain field derive from a continuous displacement field. Voigt
    order ``[xx, xy, xz, yy, yz, zz]`` (MIDAS): ε_xx=col0, ε_xy=col1, ε_yy=col3.
    Row i ↔ sample-x, col j ↔ sample-y. Penalises the squared residual summed
    over interior voxels — distinct from smoothness (it permits smooth strain
    *gradients* but forbids incompatible ones, the physically novel prior).
    """
    exx, exy, eyy = eps[:, 0], eps[:, 1], eps[:, 3]
    c = st["c"]
    d2exx_dy2 = exx[st["yp"]] - 2 * exx[c] + exx[st["ym"]]
    d2eyy_dx2 = eyy[st["xp"]] - 2 * eyy[c] + eyy[st["xm"]]
    d2exy_dxdy = 0.25 * (exy[st["pp"]] - exy[st["pm"]] - exy[st["mp"]] + exy[st["mm"]])
    resid = d2exx_dy2 + d2eyy_dx2 - 2.0 * d2exy_dxdy
    return (resid ** 2).sum()


def _pool_init_by_region(
    init: torch.Tensor,
    reg_map: "Optional[torch.Tensor]",
    n_regions: int,
    G: int,
    *,
    what: str,
) -> torch.Tensor:
    """P2-9: accept a warm-start that is EITHER per-region (n_regions,) or
    per-voxel (G,). A per-voxel init (e.g. ``GrainPeakFitResult.spread_fit``
    from a previous stage) is pooled to per-region means via the region
    map. The old code ``reshape(n_regions)``-ed blindly →
    ``reshape('[16]') invalid for size 298`` on any Stage-2→Stage-3
    hand-off."""
    init = init.reshape(-1)
    if init.numel() == n_regions:
        return init.clone()
    if init.numel() == G and reg_map is not None:
        pooled = torch.zeros(n_regions, dtype=init.dtype, device=init.device)
        counts = torch.zeros(n_regions, dtype=init.dtype, device=init.device)
        pooled.scatter_add_(0, reg_map, init)
        counts.scatter_add_(0, reg_map, torch.ones_like(init))
        return pooled / counts.clamp(min=1.0)
    raise ValueError(
        f"{what} has {init.numel()} entries; expected n_regions="
        f"{n_regions} (per-region) or G={G} with a region map "
        "(per-voxel, pooled by mean)."
    )


def neighbor_edges_from_grid_ij(grid_ij) -> torch.Tensor:
    """P2-8: build the (E, 2) neighbour-edge list for a SPARSE voxel set.

    ``grid_ij`` is the (G, 2) integer row/col of each voxel in the full
    scan grid (``PFGrainDataset.grid_ij``). Edges connect 4-neighbour
    pairs that BOTH belong to the grain. Use with
    ``fit_grain_peakshape(neighbor_edges=..., lambda_smooth=...)`` — the
    dense ``grid_shape`` reshape path requires ``G == prod(grid_shape)``
    and crashes on real (sparse) grains, which is why the datasetE driver
    had to set lambda_smooth = 0.
    """
    ij = torch.as_tensor(grid_ij, dtype=torch.long).reshape(-1, 2)
    index = {(int(r), int(c)): k for k, (r, c) in enumerate(ij.tolist())}
    edges = []
    for k, (r, c) in enumerate(ij.tolist()):
        for dr, dc in ((0, 1), (1, 0)):
            j = index.get((r + dr, c + dc))
            if j is not None:
                edges.append((k, j))
    if not edges:
        return torch.zeros((0, 2), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long)


def fit_grain_peakshape(
    data: GrainPatchData,
    model: HEDMForwardModel,
    *,
    voxel_pos: torch.Tensor,                # (G, 3)
    R_init: torch.Tensor,                   # (G, 3, 3)
    eps_init: torch.Tensor,                 # (G, 6)
    lattice_init: torch.Tensor,             # (6,)
    grid_shape: Optional[tuple] = None,
    neighbor_edges: Optional[torch.Tensor] = None,
    smooth_eps_only: bool = False,
    identifiability: IdentifiabilityMode = IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO,
    optimizer: Optional[str] = None,
    inner_steps: int = 100,
    dump_every: int = 0,                    # 0=off; N>0 dumps per-step state every N steps
    dump_dir: Optional[str] = None,         # required when dump_every>0
    lr_aa: float = 1e-3,
    lr_eps: float = 1e-4,
    lr_lat: float = 1e-4,
    lambda_smooth: float = 0.0,
    compat_stencils: Optional[dict] = None,
    lambda_compat: float = 0.0,
    lambda_equil: float = 0.0,
    stiffness_C4: Optional[torch.Tensor] = None,
    voxel_weight: Optional[torch.Tensor] = None,
    lambda_tikhonov: float = 0.0,
    refine_position: bool = False,
    lr_pos: float = 0.0,
    lambda_pos: float = 1.0,
    refine_spread: bool = False,
    spread_init: Optional[torch.Tensor] = None,
    spread_region_map: Optional[torch.Tensor] = None,
    lr_spread: float = 2e3,
    lambda_spread: float = 0.0,
    # Anisotropic (radial-microstrain σ_ε) DOF — broadens RADIAL direction only.
    # When refine_strain_spread=True (and refine_spread is on), σ_θ broadens the
    # eta+ω directions while σ_ε broadens radial — disentangles intra-voxel
    # orientation spread from intra-voxel microstrain. Shares the same
    # region-pooling map as `spread` unless an override map is supplied.
    refine_strain_spread: bool = False,
    strain_spread_init: Optional[torch.Tensor] = None,
    strain_spread_region_map: Optional[torch.Tensor] = None,
    lr_strain_spread: float = 2e3,
    lambda_strain_spread: float = 0.0,
    strain_spread_gain: float = 1.0,
    # Per-spot tan(θ)-based microstrain scaling.  When True and the
    # anisotropic-σ_ε regime is active, the splat's radial broadening
    # contributed by σ_ε is rigorous mWH:
    #     σ_radial_px[voxel, spot] = sqrt(σ_yz² + (gain·ε[voxel]·c_s)²)
    # with c_s = 2·Lsd·tan(θ_s)/px_um (per-spot, fixed from geometry).
    # σ_ε then carries dimensionless microstrain units instead of patch px,
    # removing the ~18% systematic bias of the ring-averaged ⟨tan θ⟩
    # conversion. Default False = legacy pixel-width regime.
    strain_spread_microstrain_units: bool = False,
    # PSF / sequential-refinement anchors. These break the per-voxel strain↔spread
    # identifiability (only ~1.3 reflections/voxel in pf-HEDM scanning data) by
    # anchoring spread to a measured instrument/PSF floor and/or anchoring strain
    # to a warm-start solution. Used by the staged-refinement protocol:
    #   Stage 1: no-spread LBFGS (well-posed strain inverse)
    #   Stage 2: freeze_eps=True, refine spread only, anchored to PSF floor
    #   Stage 3: unfreeze, anchor eps to stage-1 solution (±bound via L2)
    freeze_eps: bool = False,                    # if True, eps_param has requires_grad=False
    spread_l2_anchor: Optional[torch.Tensor] = None,   # scalar or (G,) — pulls spread toward this
    lambda_spread_anchor: float = 0.0,           # strength of spread→anchor L2 (data-MSE units)
    eps_l2_anchor: Optional[torch.Tensor] = None,      # (G, 6) — pulls eps toward warm-start
    lambda_eps_anchor: float = 0.0,              # strength of eps→anchor L2 (data-MSE units)
    spread_gain_yz: float = 1.0,
    spread_gain_f: float = 1.0,
    splat_radius_yz: int = 2,
    splat_radius_f: int = 1,
    gate_tau_um: float = 0.5,
    holdout_frac: float = 0.0,
    holdout_seed: int = 0,
    soft_voxel_mask: bool = True,
    chunk_size_g: Optional[int] = None,
    refine_spot_offset: bool = False,
    lr_spot_offset: float = 0.5,
    lambda_spot_offset: float = 0.0,
    verbose: bool = False,
) -> GrainPeakFitResult:
    """Joint per-grain peak-shape fit.

    Parameters
    ----------
    data : GrainPatchData
        Output of ``simulate_grain_patches`` (or real-data ingest).
    model : HEDMForwardModel
        Same model used by the simulator (or rebuilt with matching
        geometry/scan_config/hkls).
    voxel_pos : Tensor (G, 3)
    R_init, eps_init, lattice_init : warm-start parameter tensors
    grid_shape : (G_x, G_y) tuple
        Required when ``lambda_smooth > 0``.
    identifiability : IdentifiabilityMode
    optimizer : "adam" | "lbfgs" | None (uses MIDAS_OPTIMIZER or "adam")
    inner_steps : int
    lr_aa, lr_eps, lr_lat : learning rates
    lambda_smooth : neighbor-smoothness weight (default 0)
    gate_tau_um : soft beam-gate transition width
    holdout_frac : fraction of (s, σ) cells reserved for held-out scoring
        (informational only — they are still in the loss; this is a
        scoring channel, not a true CV split — that's a Phase-1B feature).
    soft_voxel_mask : bool
        When True (default), per-voxel held-out R² is used to soft-weight
        each voxel's contribution to the loss. Currently a no-op stub:
        weights = 1 (added as a hook so future work can plug in held-out CV).

    Returns
    -------
    GrainPeakFitResult
    """
    optimizer_name = _resolve_optimizer_name(optimizer)
    G = int(R_init.shape[0])
    dtype = R_init.dtype
    device = R_init.device
    S = int(data.measured_patches.shape[0])
    Sigma = int(data.scan_positions.numel())

    # Trainable parameters
    delta = torch.nn.Parameter(torch.zeros(G, 3, dtype=dtype, device=device))
    eps_param = torch.nn.Parameter(eps_init.detach().clone().to(dtype).to(device))
    lattice_param = torch.nn.Parameter(
        lattice_init.detach().clone().to(dtype).to(device)
    )
    # Sequential-refinement: optionally hold strain fixed (Stage 2) so the spread
    # DOF can ONLY absorb the residual peak width — strain can't bleed into it.
    if freeze_eps:
        eps_param.requires_grad_(False)
    # PSF / warm-start anchor buffers (used by the L2 prior terms below). Promote
    # scalar spread anchor to a (n_spread,) tensor so .sum() broadcasts cleanly.
    spread_anchor_buf = None
    if spread_l2_anchor is not None and lambda_spread_anchor > 0.0:
        if torch.is_tensor(spread_l2_anchor) and spread_l2_anchor.numel() > 1:
            spread_anchor_buf = spread_l2_anchor.detach().to(dtype).to(device).reshape(-1)
        else:
            v = float(spread_l2_anchor.item()) if torch.is_tensor(spread_l2_anchor) else float(spread_l2_anchor)
            spread_anchor_buf = torch.full((G,), v, dtype=dtype, device=device)
    eps_anchor_buf = None
    if eps_l2_anchor is not None and lambda_eps_anchor > 0.0:
        eps_anchor_buf = eps_l2_anchor.detach().to(dtype).to(device).reshape(G, 6)
    R_init_buf = R_init.detach().to(dtype).to(device)
    voxel_pos_buf = voxel_pos.detach().to(dtype).to(device)        # (G, 3)
    # Per-voxel position perturbation (the along-beam DOF that, when fixed,
    # leaks into strain — see the streak diagnosis). Refined when
    # refine_position; the Friedel ±s_V branches are already in the beam gate.
    dpos = torch.nn.Parameter(torch.zeros(G, 3, dtype=dtype, device=device))
    if not refine_position:
        dpos.requires_grad_(False)

    # Per-spot detector position offset (Δy, Δz), SHARED across all G voxels —
    # absorbs the systematic per-reflection position residual (distortion / tilt /
    # beam-center / lattice-reference), like midas-fit-grain's recorded residuals.
    # Because it is per-spot (not per-voxel) it soaks up the grain-MEAN residual
    # without competing with the per-voxel strain (the voxel-to-voxel deviation).
    S_spots = int(data.measured_patches.shape[0])
    spot_offset = torch.nn.Parameter(torch.zeros(S_spots, 2, dtype=dtype, device=device))
    if not refine_spot_offset:
        spot_offset.requires_grad_(False)

    # Per-voxel intra-voxel spread (extra peak width, in patch px/frame units) —
    # the second-moment DOF that captures the local strain/orientation
    # distribution a centroid cannot. Optimised directly and clamped ≥0. The
    # width enters the splat as a quadrature sqrt(σ²+spread²) whose gradient
    # vanishes at spread→0, so we initialise OFF the floor (default 0.5 px) to
    # keep the gradient healthy in both directions.
    # Optional region pooling: fit ONE spread per sub-grain region (n_regions
    # << G) instead of per voxel, which is far better conditioned in the
    # pencil-beam geometry (where ~Gx voxels mix into every patch). The
    # parameter is (n_regions,); it expands to per-voxel via the region map.
    reg_map = (spread_region_map.to(device).long().reshape(G)
               if spread_region_map is not None else None)
    n_spread = int(reg_map.max().item()) + 1 if reg_map is not None else G
    if refine_spread:
        if spread_init is not None:
            sp0 = _pool_init_by_region(
                spread_init.detach().to(dtype).to(device),
                reg_map, n_spread, G, what="spread_init",
            ).clamp(min=0.05)
        else:
            sp0 = torch.full((n_spread,), 0.5, dtype=dtype, device=device)
        raw_spread = torch.nn.Parameter(sp0)
    else:
        raw_spread = None

    # Microstrain (radial) spread parameter — same region pooling as `spread`
    # unless an override map is supplied. σ_ε broadens the splat along the
    # per-spot radial detector direction only; σ_θ keeps the eta+ω broadening.
    reg_map_eps = (strain_spread_region_map.to(device).long().reshape(G)
                   if strain_spread_region_map is not None else reg_map)
    n_strain_spread = (int(reg_map_eps.max().item()) + 1
                       if reg_map_eps is not None else G)
    if refine_strain_spread:
        if strain_spread_init is not None:
            # The 0.05 clamp prevents starting at the zero-gradient trap of
            # the σ² formulation. In pixel units 0.05 px is a reasonable
            # below-instrumental floor. In microstrain units 0.05 = 5%
            # strain is catastrophic; the chain-rule through c_s amplifies
            # the gradient so a much smaller floor is needed.
            _floor = 5e-7 if strain_spread_microstrain_units else 0.05
            ssp0 = _pool_init_by_region(
                strain_spread_init.detach().to(dtype).to(device),
                reg_map_eps, n_strain_spread, G,
                what="strain_spread_init",
            ).clamp(min=_floor)
        else:
            ssp0 = torch.full((n_strain_spread,), 0.5, dtype=dtype, device=device)
        raw_strain_spread = torch.nn.Parameter(ssp0)
    else:
        raw_strain_spread = None

    # Per-voxel coverage weight for the soft-voxel-mask Tikhonov: pulls
    # low-coverage (boundary) voxels' ε toward the grain mean. Higher weight =
    # less trusted. Defaults to uniform.
    if voxel_weight is not None:
        vox_w = voxel_weight.detach().to(dtype).to(device).reshape(G)
    else:
        vox_w = torch.ones(G, dtype=dtype, device=device)

    # Patch spec — anchors fixed at observed (y_obs, z_obs, f_obs) of each
    # spot, repeated across Σ scans.
    a_y = data.anchor_y.unsqueeze(-1).expand(S, Sigma).reshape(-1).to(device)
    a_z = data.anchor_z.unsqueeze(-1).expand(S, Sigma).reshape(-1).to(device)
    a_f = data.anchor_f.unsqueeze(-1).expand(S, Sigma).reshape(-1).to(device)
    spec = SpotPatchSpec(
        n_spots=S * Sigma,
        patch_F=data.patch_F, patch_P=data.patch_P,
        sigma_yz=data.sigma_yz, sigma_f=data.sigma_f,
        anchor_y=a_y, anchor_z=a_z, anchor_f=a_f,
    )

    # Radial unit vectors per spot for the anisotropic-σ splat (Path 2 σ_ε).
    # Beam-center comes from the forward model; we accept both list (multi-
    # detector) and scalar forms.
    strain_spread_per_spot_scale = None
    if refine_strain_spread:
        y_BC = float(model.y_BC if not isinstance(model.y_BC, (list, tuple))
                     else model.y_BC[0])
        z_BC = float(model.z_BC if not isinstance(model.z_BC, (list, tuple))
                     else model.z_BC[0])
        dyR = a_y.to(dtype) - y_BC
        dzR = a_z.to(dtype) - z_BC
        rR = torch.sqrt(dyR * dyR + dzR * dzR).clamp(min=1.0)
        spec.radial_y = (dyR / rR).detach()
        spec.radial_z = (dzR / rR).detach()

        # Per-spot Bragg-law scale c_s = 2·Lsd·tan(θ_s)/px_um so that a
        # dimensionless microstrain ε maps to a pixel radial width ε·c_s.
        # Computed once from geometry, fixed across iterations. Built from
        # model.thetas[m] for the M unique reflections, broadcast over the
        # 2M (Friedel) × Σ scan spots in the spec's flat S×Σ layout.
        if strain_spread_microstrain_units:
            Lsd_um = float(model.Lsd if not isinstance(model.Lsd, (list, tuple))
                           else model.Lsd[0])
            px_um = float(model.px if not isinstance(model.px, (list, tuple))
                          else model.px[0])
            thetas_rad = model.thetas.to(dtype).to(device)              # (M,)
            tan_per_hkl = torch.tan(thetas_rad)                         # (M,)
            # Spot layout in the model is (G, branch=2, hkl=M) → reshape
            # (G, 2M) with branch-major ordering: spots [0,M) are branch 0,
            # spots [M,2M) are branch 1. Both branches of an hkl share θ.
            tan_per_spot = tan_per_hkl.repeat(2)                        # (S=2M,)
            # Replicate across Σ scans to match anchors_y/z/f flat layout.
            c_s_flat = (2.0 * Lsd_um * tan_per_spot / px_um)
            strain_spread_per_spot_scale = c_s_flat.unsqueeze(-1).expand(
                S, Sigma).reshape(-1).detach()                          # (S × Σ,)

    meas = data.measured_patches.to(dtype).to(device)              # (S, Σ, F, P, P)

    # Spot keep mask: spots that any voxel emitted to.
    spot_keep = data.spot_observed.to(dtype).to(device)            # (S,)

    # P2-7: per-pixel saturation mask (1 = valid, 0 = clamped at the
    # detector ceiling). Saturated flat-tops carry no shape information;
    # fitting Gaussian splats against them floors the loss and the strain
    # runs away (SOH: ε 4237 → 5959 µε with more steps). getattr: cached
    # GrainPatchData pickles from before this field default to None.
    _sat = getattr(data, "saturation_mask", None)
    sat_keep = _sat.to(device=device, dtype=dtype) if _sat is not None else None

    # Optional held-out (s, σ) cells (not yet a true CV split — scored on
    # the same forward but excluded from the data MSE).
    if holdout_frac and holdout_frac > 0.0:
        n_total = S * Sigma
        n_hold = max(1, int(round(holdout_frac * n_total)))
        gen = torch.Generator(device="cpu").manual_seed(int(holdout_seed))
        perm = torch.randperm(n_total, generator=gen)
        hold_mask = torch.zeros(n_total, dtype=torch.bool)
        hold_mask[perm[:n_hold]] = True
        hold_mask = hold_mask.reshape(S, Sigma).to(device)
    else:
        hold_mask = torch.zeros(S, Sigma, dtype=torch.bool, device=device)

    extra = []
    if refine_position:
        extra.append((dpos, lr_pos))
    if refine_spot_offset:
        extra.append((spot_offset, lr_spot_offset))
    optimizer_obj = _build_optimizer(
        optimizer_name, delta, eps_param, lattice_param,
        lr_aa=lr_aa, lr_eps=lr_eps, lr_lat=lr_lat,
        extra=(extra or None),
    )
    # The per-voxel spread gets its OWN plain-SGD(+momentum) optimizer, NOT
    # Adam: the data MSE is a .mean() over ~1e6 cells, so ∂loss/∂spread is tiny
    # (~1e-4) with a wide per-voxel dynamic range. Adam's scale-invariant
    # normalisation amplifies that noisy tiny gradient and either stalls or
    # collapses spread to 0; plain scaled GD follows the (consistent) descent
    # direction and recovers the map. lr_spread is therefore a *large* raw step.
    # Region pooling sums the per-voxel gradient over each region, so the
    # per-region gradient is ~(G/n_spread)x larger; scale lr down by that factor
    # so a single lr_spread works across pooling levels (avoids overshoot/
    # collapse-to-0 at the clamp boundary).
    _lr_spread = lr_spread * (n_spread / float(G))
    spread_opt = (torch.optim.SGD([raw_spread], lr=_lr_spread, momentum=0.0)
                  if refine_spread else None)
    _lr_strain_spread = lr_strain_spread * (n_strain_spread / float(G))
    strain_spread_opt = (torch.optim.SGD([raw_strain_spread],
                                          lr=_lr_strain_spread, momentum=0.0)
                          if refine_strain_spread else None)

    losses: List[float] = []
    last_loss_box: List[Optional[float]] = [None]

    def _eps_used() -> torch.Tensor:
        if identifiability == IdentifiabilityMode.PROJECT_EPS_MEAN_ZERO:
            return _project_mean_zero(eps_param)
        return eps_param

    def _spread_used():
        if not refine_spread:
            return None
        sp = raw_spread.clamp(min=0.0)
        return sp[reg_map] if reg_map is not None else sp   # expand to (G,)

    def _strain_spread_used():
        if not refine_strain_spread:
            return None
        ssp = raw_strain_spread.clamp(min=0.0)
        return (ssp[reg_map_eps] if reg_map_eps is not None else ssp)   # → (G,)

    def _forward_and_loss():
        R_v = R_init_buf @ _aa_to_R(delta)                         # (G, 3, 3)
        pos_v = voxel_pos_buf + dpos                               # (G, 3)
        eps_u = _eps_used()
        sp_v = _spread_used()
        ssp_v = _strain_spread_used()
        out = joint_grain_forward(
            model, R_v, eps_u, lattice_param, pos_v,
            spec=spec, n_scans=Sigma, gate_tau_um=gate_tau_um,
            voxel_spread=sp_v, voxel_strain_spread=ssp_v,
            spread_gain_yz=spread_gain_yz,
            spread_gain_f=spread_gain_f,
            strain_spread_gain=strain_spread_gain,
            strain_spread_per_spot_scale=strain_spread_per_spot_scale,
            splat_radius_yz=splat_radius_yz,
            splat_radius_f=splat_radius_f, chunk_size_g=chunk_size_g,
            spot_offset=(spot_offset if refine_spot_offset else None),
        )
        pred = out.pred_patches                                    # (S, Σ, F, P, P)
        c = closed_form_per_spot_scale(pred, meas, spot_keep,
                                       pixel_weight=sat_keep)      # (S,)
        diff = c[:, None, None, None, None] * pred - meas
        # Per-pixel weight = saturation mask × held-out mask (both
        # optional). Normalise by the live-pixel count so masking does
        # not silently rescale the loss.
        w = sat_keep
        if hold_mask.any():
            hk = (~hold_mask).to(dtype)[..., None, None, None]
            w = hk if w is None else w * hk
        if w is not None:
            w_sum = w.expand_as(meas).sum()      # live-pixel count (handles
            data_mse = ((diff ** 2) * w).sum() / w_sum.clamp(min=1.0)  # broadcast hk)
        else:
            data_mse = (diff ** 2).mean()

        loss = data_mse
        if lambda_smooth > 0.0:
            if neighbor_edges is not None:
                smooth = _graph_smooth_loss(delta, eps_u, neighbor_edges,
                                            eps_only=smooth_eps_only)
            else:
                assert grid_shape is not None, \
                    "grid_shape or neighbor_edges required for lambda_smooth>0"
                smooth = _neighbor_smooth_loss(delta, eps_u, grid_shape)
            loss = loss + lambda_smooth * smooth
        if lambda_compat > 0.0 and compat_stencils:
            loss = loss + lambda_compat * _compatibility_loss(eps_u, compat_stencils)
        if lambda_equil > 0.0 and compat_stencils and stiffness_C4 is not None:
            loss = loss + lambda_equil * _equilibrium_loss(
                eps_u, R_v, compat_stencils, stiffness_C4)
        if lambda_tikhonov > 0.0:
            # coverage-weighted pull of ε toward the grain mean (0 deviatoric)
            loss = loss + lambda_tikhonov * (vox_w[:, None] * eps_u ** 2).sum()
        if refine_position and lambda_pos > 0.0:
            loss = loss + lambda_pos * (dpos ** 2).sum()
        if refine_spot_offset and lambda_spot_offset > 0.0:
            loss = loss + lambda_spot_offset * (spot_offset ** 2).sum()
        if refine_spread and lambda_spread > 0.0 and neighbor_edges is not None:
            u, v = neighbor_edges[:, 0], neighbor_edges[:, 1]
            loss = loss + lambda_spread * ((sp_v[u] - sp_v[v]) ** 2).sum()
        if (refine_strain_spread and lambda_strain_spread > 0.0
                and neighbor_edges is not None and ssp_v is not None):
            u, v = neighbor_edges[:, 0], neighbor_edges[:, 1]
            loss = loss + lambda_strain_spread * ((ssp_v[u] - ssp_v[v]) ** 2).sum()
        # PSF / warm-start L2 anchors (resolve strain↔spread identifiability).
        # spread anchor: pulls each voxel's spread toward the PSF floor measured
        # from 0N data. Penalty grows with deviation, so spread will only depart
        # from the floor when the data residual demands it strongly enough.
        if (refine_spread and lambda_spread_anchor > 0.0
                and spread_anchor_buf is not None and sp_v is not None):
            loss = loss + lambda_spread_anchor * ((sp_v - spread_anchor_buf) ** 2).sum()
        # eps anchor: pulls each voxel's strain toward the warm-start (e.g. the
        # no-spread Stage-1 solution). Effective ±bound for joint Stage 3.
        if lambda_eps_anchor > 0.0 and eps_anchor_buf is not None:
            loss = loss + lambda_eps_anchor * ((eps_u - eps_anchor_buf) ** 2).sum()
        return loss

    # Per-step state dump (annealing movie / debugging). Saves the CURRENT
    # eps_used + R_voxel + lattice + loss to <dump_dir>/step_NNN.npz every
    # `dump_every` steps. No-op when dump_every<=0. (Cheap: only the per-voxel
    # arrays, not the full optimizer state.)
    _dump = (dump_every and dump_every > 0 and dump_dir is not None)
    if _dump:
        import os as _os, numpy as _np
        _os.makedirs(dump_dir, exist_ok=True)
    def _do_dump(step_idx, loss_val):
        if not _dump or (step_idx % dump_every != 0 and step_idx != inner_steps - 1):
            return
        with torch.no_grad():
            _eps = _eps_used().detach().cpu().numpy()
            _R = (R_init_buf @ _aa_to_R(delta)).detach().cpu().numpy()
            _lat = lattice_param.detach().cpu().numpy()
        _np.savez(f"{dump_dir}/step_{step_idx:04d}.npz",
                  eps=_eps, R=_R, lattice=_lat, loss=float(loss_val), step=int(step_idx))

    if optimizer_obj is None and spread_opt is None and strain_spread_opt is None:
        # All params locked — no-op fit, used as a warm-start sanity check.
        with torch.no_grad():
            losses.append(float(_forward_and_loss().item()))
    elif optimizer_obj is not None and optimizer_name == "lbfgs":
        for step in range(inner_steps):
            def closure():
                optimizer_obj.zero_grad()
                if spread_opt is not None:
                    spread_opt.zero_grad()
                if strain_spread_opt is not None:
                    strain_spread_opt.zero_grad()
                loss = _forward_and_loss()
                loss.backward()
                last_loss_box[0] = float(loss.detach())
                return loss
            optimizer_obj.step(closure)
            if spread_opt is not None:
                spread_opt.step()       # uses grad from the last closure eval
            if strain_spread_opt is not None:
                strain_spread_opt.step()
            losses.append(last_loss_box[0])
            _do_dump(step, last_loss_box[0])
            if verbose and (step % max(1, inner_steps // 10) == 0
                            or step == inner_steps - 1):
                print(f"  step {step:4d}  loss={last_loss_box[0]:.6e}")
    else:
        # Adam (main params) and/or plain-SGD (spread): one backward, step both.
        for step in range(inner_steps):
            if optimizer_obj is not None:
                optimizer_obj.zero_grad()
            if spread_opt is not None:
                spread_opt.zero_grad()
            if strain_spread_opt is not None:
                strain_spread_opt.zero_grad()
            loss = _forward_and_loss()
            loss.backward()
            if optimizer_obj is not None:
                optimizer_obj.step()
            if spread_opt is not None:
                spread_opt.step()
            if strain_spread_opt is not None:
                strain_spread_opt.step()
            losses.append(float(loss.item()))
            _do_dump(step, losses[-1])
            if verbose and (step % max(1, inner_steps // 10) == 0
                            or step == inner_steps - 1):
                print(f"  step {step:4d}  loss={losses[-1]:.6e}")

    # Final fit summary
    with torch.no_grad():
        R_v_final = R_init_buf @ _aa_to_R(delta)
        eps_final = _eps_used()
        pos_final = voxel_pos_buf + dpos
        spread_final = _spread_used()
        strain_spread_final = _strain_spread_used()
        out_final = joint_grain_forward(
            model, R_v_final, eps_final, lattice_param, pos_final,
            spec=spec, n_scans=Sigma, gate_tau_um=gate_tau_um,
            voxel_spread=spread_final, voxel_strain_spread=strain_spread_final,
            spread_gain_yz=spread_gain_yz,
            spread_gain_f=spread_gain_f,
            strain_spread_gain=strain_spread_gain,
            strain_spread_per_spot_scale=strain_spread_per_spot_scale,
            splat_radius_yz=splat_radius_yz,
            splat_radius_f=splat_radius_f, chunk_size_g=chunk_size_g,
        )
        c_final = closed_form_per_spot_scale(out_final.pred_patches, meas,
                                             spot_keep)

        holdout_score = None
        if hold_mask.any():
            diff_h = c_final[:, None, None, None, None] * out_final.pred_patches - meas
            num = ((diff_h ** 2) * hold_mask[..., None, None, None].to(dtype)).sum()
            den = ((meas ** 2) * hold_mask[..., None, None, None].to(dtype)).sum().clamp(min=1e-12)
            holdout_score = float(1.0 - num / den)

    return GrainPeakFitResult(
        R_fit=R_v_final.detach(),
        eps_fit=eps_final.detach(),
        spread_fit=(spread_final.detach().abs() if refine_spread else None),
        strain_spread_fit=(strain_spread_final.detach().abs()
                             if refine_strain_spread else None),
        lattice_fit=lattice_param.detach().clone(),
        c_per_spot=c_final.detach(),
        pos_fit=(pos_final.detach() if refine_position else None),
        spot_offset_fit=(spot_offset.detach() if refine_spot_offset else None),
        losses=losses,
        holdout_score=holdout_score,
        converged=False,        # convergence detection is a Phase-1B refinement
        optimizer_used=optimizer_name,
        voxel_holdout_r2=None,
        metadata={
            "identifiability": str(identifiability),
            "lambda_smooth": float(lambda_smooth),
            "lambda_compat": float(lambda_compat),
            "lambda_equil": float(lambda_equil),
            "lambda_tikhonov": float(lambda_tikhonov),
            "refine_position": bool(refine_position),
            "gate_tau_um": float(gate_tau_um),
            "soft_voxel_mask": bool(soft_voxel_mask),
            "freeze_eps": bool(freeze_eps),
            "lambda_spread_anchor": float(lambda_spread_anchor),
            "lambda_eps_anchor": float(lambda_eps_anchor),
            "has_spread_anchor": bool(spread_anchor_buf is not None),
            "has_eps_anchor": bool(eps_anchor_buf is not None),
        },
    )
