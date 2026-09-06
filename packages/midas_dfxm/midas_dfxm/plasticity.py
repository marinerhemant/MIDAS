"""Crystal-plasticity <-> DFXM coupling: per-slip-system GND-density inversion.

Scaling item of the roadmap — the deepest physics coupling that runs locally.

DFXM measures a lattice-orientation field; its spatial gradient is the lattice
curvature, and (Nye 1953) the curvature fixes the geometrically-necessary
dislocation (GND) content:

    alpha = sum_s rho_s (b_hat_s (x) xi_hat_s)        (Nye tensor, 3x3)

summed over slip-system dislocation types s (edge line ``xi = n x b``, screw line
``xi = b``), each carrying a signed density ``rho_s``. This module builds that forward
(``nye_from_densities``) and inverts it (``recover_gnd_densities``) with a sparsity
prior, and — honestly — reports the **null space**: a 3x3 Nye tensor has 9 numbers
but FCC offers 12 edge + 12 screw density types, so the decomposition is massively
underdetermined. Only sparsity (few active systems, the physical case) makes it
identifiable; this is the classical Arsenlis-Parks GND-decomposition problem, here
driven by DFXM-measured curvature and differentiable end-to-end.

This is the interpretation layer between a crystal-plasticity field (e.g. a JAX-CPFEM
``F(r)`` via :func:`midas_dfxm.generators.field_from_deformation_gradient`) and the
DFXM measurement: CPFEM predicts slip -> GND content -> curvature -> DFXM, and this
inverts DFXM curvature back to per-system GND densities for comparison.

Reuses ``midas_defect`` slip-system catalogs; torch-differentiable, device-portable.
"""
from __future__ import annotations

import torch

from midas_invert.optimize import fit

from .dislocation import fcc_slip_systems

_ANGSTROM_PER_UM = 1e4


def slip_dislocation_types(systems=None, *, characters=("edge", "screw"), dtype=torch.float64, device=None):
    """Return ``(labels, b_hat, xi_hat)`` for the dislocation types of a slip catalog.

    ``labels`` is a list of ``(normal, burgers, character)``; ``b_hat`` and ``xi_hat``
    are ``(T, 3)`` unit Burgers and line directions (edge line ``n x b``, screw ``b``).
    """
    systems = systems if systems is not None else fcc_slip_systems()
    labels, bs, xis = [], [], []
    for normal, burgers in systems:
        n = torch.as_tensor(normal, dtype=dtype, device=device)
        b = torch.as_tensor(burgers, dtype=dtype, device=device)
        b_hat = b / torch.linalg.norm(b)
        for ch in characters:
            xi = b if ch == "screw" else torch.linalg.cross(n, b)
            xi_hat = xi / torch.linalg.norm(xi)
            labels.append((tuple(int(x) for x in normal), tuple(int(x) for x in burgers), ch))
            bs.append(b_hat)
            xis.append(xi_hat)
    return labels, torch.stack(bs), torch.stack(xis)


def nye_from_densities(rho, b_hat, xi_hat, *, burgers_length_A: float = 2.556) -> torch.Tensor:
    """Nye tensor ``alpha = sum_s rho_s b_um (b_hat_s (x) xi_hat_s)`` (3x3, units 1/um).

    ``rho`` is ``(T,)`` signed densities (1/um^2). Differentiable in ``rho``.
    """
    b_um = burgers_length_A / _ANGSTROM_PER_UM
    dyads = b_hat[:, :, None] * xi_hat[:, None, :]        # (T, 3, 3)
    return b_um * torch.einsum("t,tij->ij", rho, dyads)


def gnd_design_matrix(b_hat, xi_hat) -> torch.Tensor:
    """Geometric decomposition operator ``A`` (9, T): ``vec(alpha)/b_um = A rho``.

    Pure ``b_hat (x) xi_hat`` dyads (the physical ``b_um`` factor is carried separately
    so ``A`` and ``rho`` are O(1) — otherwise a sparsity prior on ``rho`` is swamped by
    the tiny ``alpha`` scale).
    """
    dyads = b_hat[:, :, None] * xi_hat[:, None, :]           # (T,3,3)
    return dyads.reshape(dyads.shape[0], 9).transpose(0, 1)  # (9, T)


def gnd_identifiability(b_hat, xi_hat) -> dict:
    """Rank / null-space dimension of the GND decomposition (honest identifiability).

    Returns ``{'n_types', 'rank', 'null_dim', 'singular_values'}``. ``null_dim > 0``
    means densities are only recoverable up to that many unobservable combinations —
    sparsity (few active systems) is what pins a unique solution.
    """
    A = gnd_design_matrix(b_hat, xi_hat)
    s = torch.linalg.svdvals(A)
    rank = int((s > s.max() * 1e-10).sum())
    return {"n_types": A.shape[1], "rank": rank, "null_dim": A.shape[1] - rank,
            "singular_values": s}


def recover_gnd_densities(
    alpha: torch.Tensor,
    b_hat: torch.Tensor,
    xi_hat: torch.Tensor,
    *,
    burgers_length_A: float = 2.556,
    lambda_sparse: float = 1e-3,
    nonneg: bool = False,
    steps: int = 800,
    lr: float = 0.05,
) -> torch.Tensor:
    """Recover per-type densities from a Nye tensor with an L1 (sparsity) prior.

    Minimises ``||A rho - vec(alpha)||^2 + lambda_sparse ||rho||_1``. With ``nonneg``,
    densities are constrained >= 0 (signed types then need both +/- entries in the
    catalog). Sparsity resolves the null space toward a few active systems — the
    physical single/double-slip case. Returns ``(T,)``. Differentiable.
    """
    b_um = burgers_length_A / _ANGSTROM_PER_UM
    A = gnd_design_matrix(b_hat, xi_hat)
    target = alpha.reshape(9) / b_um            # geometric units, O(rho)
    T = A.shape[1]
    raw = torch.zeros(T, dtype=alpha.dtype, device=alpha.device, requires_grad=True)

    def rho_of():
        return torch.nn.functional.softplus(raw) if nonneg else raw

    def loss_fn():
        rho = rho_of()
        resid = A @ rho - target
        return (resid ** 2).mean() + lambda_sparse * rho.abs().mean()

    fit([raw], loss_fn, steps=steps, lr=lr)
    return rho_of().detach()


def multislip_gnd_field(
    rho,
    positions: torch.Tensor,
    b_hat: torch.Tensor,
    xi_hat: torch.Tensor,
    *,
    burgers_length_A: float = 2.556,
    along: int = 0,
    orientation=None,
    lattice_params=(3.6356, 3.6356, 3.6356, 90.0, 90.0, 90.0),
    shape=None,
):
    """Build a DFXM-ready field whose lattice curvature encodes multi-slip GND content.

    The Nye tensor ``alpha`` from ``rho`` sets a lattice rotation gradient; we realise it
    as a rotation that grows linearly along axis ``along`` about the axial vector of
    ``alpha`` (the curvature axis), giving a :class:`DeformationField` the DFXM forward
    consumes. This closes the loop CPFEM/slip -> GND -> curvature -> DFXM. Differentiable
    in ``rho``.
    """
    from .field import DeformationField

    device, dtype = positions.device, positions.dtype
    if not isinstance(rho, torch.Tensor):
        rho = torch.as_tensor(rho, dtype=dtype, device=device)
    alpha = nye_from_densities(rho, b_hat, xi_hat, burgers_length_A=burgers_length_A)
    # Curvature axis-angle rate: the axial (skew) vector of alpha gives the lattice
    # rotation gradient; its magnitude is rad/um, direction the rotation axis.
    kvec = torch.stack([alpha[2, 1] - alpha[1, 2],
                        alpha[0, 2] - alpha[2, 0],
                        alpha[1, 0] - alpha[0, 1]]) * 0.5
    kmag = torch.linalg.vector_norm(kvec) + 1e-30
    axis = kvec / kmag
    angle = kmag * positions[:, along]                       # (N,) rad
    K = torch.zeros(3, 3, device=device, dtype=dtype)
    K[0, 1], K[0, 2], K[1, 2] = -axis[2], axis[1], -axis[0]
    K = K - K.T
    a = angle[:, None, None]
    eye = torch.eye(3, device=device, dtype=dtype)
    R = eye + torch.sin(a) * K + (1 - torch.cos(a)) * (K @ K)
    orientation = torch.eye(3, device=device, dtype=dtype) if orientation is None \
        else torch.as_tensor(orientation, device=device, dtype=dtype)
    latc = torch.as_tensor(lattice_params, device=device, dtype=dtype)
    return DeformationField(positions=positions, F=R, reference_orientation=orientation,
                            lattice_params=latc, shape=shape)


# ---------------------------------------------------------------------------
# The missing link: per-voxel orientation/distortion field -> Nye tensor
# ---------------------------------------------------------------------------
#
# `nye_from_densities` is the forward and `recover_gnd_densities` inverts the
# decomposition, but nothing turned a MEASURED per-voxel field into alpha.
# (`midas_defect.gnd.per_grain_nye_tensor` is inter-GRAIN, k-NN over an FF-HEDM
# grain map -- a different measurement.) DFXM delivers exactly a per-voxel
# field, so this is the step between the instrument and the GND content, and it
# is where the detection limit lives: alpha is a CURL, i.e. a spatial
# derivative, so orientation noise is amplified by 1/spacing.

def nye_tensor_from_field(
    F: torch.Tensor,
    shape,
    spacing_um,
    *,
    rotation_only: bool = False,
) -> torch.Tensor:
    """Nye tensor per voxel from a deformation-gradient field. ``(N, 3, 3)``, 1/µm.

    ``alpha_ij = eps_jkl d_k beta_il`` with ``beta = F - I`` the elastic
    distortion (Nye 1953; Arsenlis & Parks 1999). Central differences on the
    interior, one-sided at the faces.

    Das, Hofmann & Tarleton, *Int. J. Plasticity* (2018),
    `10.1016/j.ijplas.2018.05.001`, show the three curl conventions in common use
    agree when applied consistently, so the choice above is not a free parameter
    -- and they also find the volumetric elastic strain barely affects alpha,
    which is why ``rotation_only`` (use the skew part of beta alone, the
    HR-EBSD-style estimate) is offered and is usually close.

    Parameters
    ----------
    F : (N, 3, 3)
        Deformation gradient per voxel, voxels ordered C-style over ``shape``.
    shape : (nx, ny, nz)
    spacing_um : float or 3-sequence
        Voxel pitch. A scalar is broadcast.
    rotation_only :
        Differentiate only the skew (lattice-rotation) part of ``beta``.

    Notes
    -----
    Differentiable. A dimension of extent 1 contributes no derivative (its
    gradient is taken as zero), so a single slice returns the components a 2-D
    measurement can support and zeros for the rest -- it does NOT silently
    invent the out-of-plane terms.
    """
    nx, ny, nz = (int(s) for s in shape)
    if F.shape[0] != nx * ny * nz:
        raise ValueError(f"F has {F.shape[0]} voxels but shape {tuple(shape)} "
                         f"implies {nx * ny * nz}")
    sp = torch.as_tensor(spacing_um, dtype=F.dtype, device=F.device)
    if sp.ndim == 0:
        sp = sp.repeat(3)

    eye = torch.eye(3, dtype=F.dtype, device=F.device)
    beta = (F - eye).reshape(nx, ny, nz, 3, 3)
    if rotation_only:
        beta = 0.5 * (beta - beta.transpose(-1, -2))

    # d_k beta_il for k = 0,1,2
    grads = []
    for k, n in enumerate((nx, ny, nz)):
        if n < 2:
            grads.append(torch.zeros_like(beta))
            continue
        g = torch.gradient(beta, spacing=float(sp[k]), dim=k)[0]
        grads.append(g)
    dbeta = torch.stack(grads, dim=0)            # (3, nx, ny, nz, 3, 3) = d_k beta_il

    eps = torch.zeros(3, 3, 3, dtype=F.dtype, device=F.device)
    for i, j, k, v in ((0, 1, 2, 1.0), (1, 2, 0, 1.0), (2, 0, 1, 1.0),
                       (0, 2, 1, -1.0), (2, 1, 0, -1.0), (1, 0, 2, -1.0)):
        eps[i, j, k] = v
    # alpha_ij = eps_jkl d_k beta_il
    alpha = torch.einsum("jkl,kxyzil->xyzij", eps, dbeta)
    return alpha.reshape(nx * ny * nz, 3, 3)


def gnd_density_from_field(F, shape, spacing_um, *, burgers_length_A=2.556,
                           rotation_only=False) -> torch.Tensor:
    """Scalar GND density per voxel, m^-2, from ``|alpha| / b``. ``(N,)``.

    The Frobenius norm of the Nye tensor over the Burgers magnitude -- the
    standard scalar summary, and a LOWER BOUND on the true dislocation content:
    it counts only the geometrically necessary part, and statistically stored
    dipoles cancel out of alpha exactly. Kysar et al., *Int. J. Plasticity* 26
    (2010) 1097, derive the rigorous lower bound this norm approximates.
    """
    alpha = nye_tensor_from_field(F, shape, spacing_um,
                                  rotation_only=rotation_only)
    b_um = burgers_length_A / _ANGSTROM_PER_UM
    return torch.linalg.matrix_norm(alpha) / b_um * 1e12    # 1/um^2 -> 1/m^2


def gnd_noise_floor(sigma_orientation_rad: float, spacing_um: float, *,
                    burgers_length_A: float = 2.556, stencil: float = 2.0
                    ) -> float:
    """Analytic GND detection floor, m^-2. **A scaling estimate, not a measurement.**

    ``rho_min ~ sigma_omega / (b * dx)``: a curl differentiates, so per-voxel
    orientation noise of ``sigma_omega`` over a baseline ``dx`` produces a
    spurious curvature ``~sigma_omega/dx`` and hence an apparent GND density
    ``~sigma_omega/(b dx)``. ``stencil`` is the effective number of spacings in
    the derivative (2 for central differences).

    This assumes UNCORRELATED per-voxel noise and says nothing about the
    correlation the resolution function imposes, nor about regularisation. It is
    the number to test a measurement against, not to quote in place of one.
    """
    b_m = burgers_length_A * 1e-10
    dx_m = spacing_um * 1e-6
    return float(sigma_orientation_rad / (b_m * dx_m * stencil / 2.0))
