"""Unified full deformation-gradient-tensor inverse: recover F(x) from DFXM.

Reconstructs the **full nine-component** deformation-gradient tensor field ``F(x)``
(equivalently the elastic distortion, hence strain + lattice rotation) per voxel from
multi-reflection DFXM, matching what Henningsson 2025 (JMPS) and Kanesalingam 2025
(arXiv:2507.17929) do with closed-form / finite-difference solvers -- but as a
**differentiable, regularizable, uncertainty-quantified** inverse, and as the
physics-regularized *intermediate* that feeds anisotropic dislocation typing
(:mod:`midas_dfxm.typing`) and crystal-plasticity inference (:mod:`midas_dfxm.cpfem`).

Physics (exact, finite strain -- no small-deformation assumption in the recovery).
The deformed reciprocal vector is ``Q_s = F^{-T} Q0`` exactly (Poulsen 2021 Eq. 32;
Detlefs 2025 Eq. 5), so the measured reciprocal-space shift is

    dQ^(g) = Q_s^(g) - Q0^(g) = H Q0^(g),   with   H = F^{-T} - I   (exact).

Each reflection ``g`` contributes the 3-vector ``dQ^(g) = H Q0^(g)`` -- linear in the
nine components of ``H`` -- so one reflection constrains only the 3-D subspace
``H Q0^(g)`` (the "3 of 9" of Poulsen Eq. 38); >=3 non-coplanar reflections span all
nine and ``F = (I + H)^{-T}`` is recovered exactly. Recovery is thus exact for clean
data with a rank-9 reflection set; regularization, UQ and Fisher-optimal reflection
design are the value-adds the closed-form incumbents lack.

Row-major vectorization ``vec(H)[3 i + j] = H_ij`` is used throughout. Everything is
torch-differentiable and device/dtype-portable. Reuses ``midas_invert`` for the fit
loop and ``midas_dfxm.field`` for geometry.
"""
from __future__ import annotations

import torch

from midas_invert.optimize import fit

from .field import DeformationField, polar_decomposition, reciprocal_basis


def reference_Q(hkl, orientation, lattice_params) -> torch.Tensor:
    """Reference (undeformed) reciprocal vector ``Q0 = orientation @ B @ hkl`` (sample frame)."""
    latc = torch.as_tensor(lattice_params, dtype=orientation.dtype, device=orientation.device)
    hkl_t = torch.as_tensor(hkl, dtype=orientation.dtype, device=orientation.device)
    B = reciprocal_basis(latc)
    return orientation @ (B @ hkl_t)


def _orientation(orientation, dtype, device) -> torch.Tensor:
    if orientation is None:
        return torch.eye(3, dtype=dtype, device=device)
    return torch.as_tensor(orientation, dtype=dtype, device=device)


def deformation_design_matrix(
    reflections,
    *,
    orientation=None,
    lattice_params=(3.6356, 3.6356, 3.6356, 90.0, 90.0, 90.0),
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Design matrix ``A`` ``(3m, 9)`` mapping ``vec(H)`` to the stacked shifts ``dQ``.

    Row block ``g`` (3 rows) implements ``dQ^(g) = H Q0^(g)``: ``A[3g+i, 3i:3i+3] =
    Q0^(g)``. Differentiable in the lattice/orientation. Rank 9 iff the reflection set
    is sufficient to recover the full tensor (see :func:`deformation_identifiability`).
    """
    ori = _orientation(orientation, dtype, device)
    rows = []
    for hkl in reflections:
        Q0 = reference_Q(hkl, ori, lattice_params)
        block = torch.zeros(3, 9, dtype=dtype, device=device)
        for i in range(3):
            block[i, 3 * i:3 * i + 3] = Q0
        rows.append(block)
    return torch.cat(rows, dim=0)


def deformation_identifiability(reflections, **kw) -> dict:
    """Rank / conditioning of a reflection set for full-F recovery.

    Returns ``{'rank', 'cond', 'singular_values', 'recoverable'}`` where
    ``recoverable`` is ``True`` iff all nine components of ``H`` (hence ``F``) are
    constrained (rank 9). This is the differentiable generalization of the reflection-
    set conditioning of Detlefs 2025 (``det[B0 H (B0 H)^T]`` large) and the
    ``kappa(nu)`` ranking of Kanesalingam 2025 -- here the condition number of the
    exact design matrix rather than a finite-difference sensitivity matrix.
    """
    A = deformation_design_matrix(reflections, **kw)
    s = torch.linalg.svdvals(A)
    tol = s.max() * 1e-10
    rank = int((s > tol).sum())
    cond = float(s.max() / s[8]) if rank == 9 else float("inf")
    return {"rank": rank, "cond": cond, "singular_values": s, "recoverable": rank == 9}


def deformation_observable(field: DeformationField, reflections) -> torch.Tensor:
    """Forward: per-voxel stacked reciprocal-space shifts ``dQ`` ``(N, 3m)``.

    ``dQ^(g) = (F^{-T} - I) Q0^(g)`` per reflection (exact). Differentiable in the
    field; this is the forward operator the inverse inverts and the generator of
    synthetic multi-reflection data.
    """
    F = field.F
    ori = field.reference_orientation
    Finv_T = torch.linalg.inv(F).transpose(-1, -2)                 # (N,3,3)
    eye = torch.eye(3, dtype=F.dtype, device=F.device)
    cols = []
    for hkl in reflections:
        Q0 = reference_Q(hkl, ori, field.lattice_params)           # (3,)
        dQ = torch.einsum("nij,j->ni", Finv_T - eye, Q0)           # (N,3)
        cols.append(dQ)
    return torch.cat(cols, dim=-1)                                 # (N, 3m)


def _H_to_F(H: torch.Tensor) -> torch.Tensor:
    """``F = (I + H)^{-T}`` from ``H = F^{-T} - I`` (exact, batched ``(...,3,3)``)."""
    eye = torch.eye(3, dtype=H.dtype, device=H.device)
    return torch.linalg.inv(eye + H).transpose(-1, -2)


def _frame_kw(field, orientation, lattice_params, kw) -> dict:
    """Resolve the reference frame (orientation + lattice) for the design matrix.

    The inverse MUST build its design matrix from the SAME ``Q0 = orientation @ B(lattice)
    @ hkl`` the forward :func:`deformation_observable` used, or the recovered ``H`` is
    expressed in the wrong reference and a silent bias appears (identity vs a real grain
    orientation, or the FCC-Al default lattice vs the sample's -- the latter is a
    microstrain-scale trap). Passing ``field=`` threads the exact forward frame and makes
    this desync impossible; explicit ``orientation``/``lattice_params`` is for raw data
    with no field object. A bare call falls back to the design-matrix defaults (identity
    orientation, FCC-Al lattice) for backward compatibility -- correct only when the data
    were generated in that frame.
    """
    if field is not None:
        if orientation is not None or lattice_params is not None:
            raise ValueError("pass either field= or explicit orientation/lattice_params, not both")
        return {**kw, "orientation": field.reference_orientation,
                "lattice_params": field.lattice_params}
    if orientation is not None:
        kw = {**kw, "orientation": orientation}
    if lattice_params is not None:
        kw = {**kw, "lattice_params": lattice_params}
    return kw


def recover_deformation_direct(measured: torch.Tensor, reflections, *,
                               field=None, orientation=None, lattice_params=None,
                               **kw) -> torch.Tensor:
    """Per-voxel least-squares full-F recovery -- the closed-form incumbent analogue.

    ``measured`` is ``(N, 3m)`` stacked shifts. Solves ``A vec(H) = dQ`` per voxel,
    then ``F = (I+H)^{-T}``. Returns ``F`` ``(N, 3, 3)``. Exact for clean data with a
    rank-9 set; noise passes straight through (motivating the regularized variant).

    Reference frame: pass ``field=`` (a :class:`DeformationField`) so the design matrix
    uses the SAME orientation + lattice the forward did -- the desync-proof path. For raw
    data, give ``orientation`` and/or ``lattice_params`` explicitly. See :func:`_frame_kw`.
    """
    frame = _frame_kw(field, orientation, lattice_params, kw)
    A = deformation_design_matrix(reflections, dtype=measured.dtype, device=measured.device, **frame)
    vecH = torch.linalg.lstsq(A, measured.transpose(0, 1)).solution   # (9, N)
    H = vecH.transpose(0, 1).reshape(-1, 3, 3)                        # (N,3,3) row-major
    return _H_to_F(H)


def _curvature_penalty(vol9: torch.Tensor, shape) -> torch.Tensor:
    """Second-difference (curvature) smoothness on the 9-component H field over the grid."""
    nx, ny, nz = shape
    v = vol9.reshape(nx, ny, nz, 9)
    pen = torch.zeros((), device=vol9.device, dtype=vol9.dtype)
    if nx > 2:
        pen = pen + (v[2:] - 2 * v[1:-1] + v[:-2]).pow(2).mean()
    if ny > 2:
        pen = pen + (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]).pow(2).mean()
    if nz > 2:
        pen = pen + (v[:, :, 2:] - 2 * v[:, :, 1:-1] + v[:, :, :-2]).pow(2).mean()
    return pen


def recover_deformation_regularised(
    measured: torch.Tensor,
    reflections,
    shape,
    *,
    lambda_smooth: float = 1e-3,
    steps: int = 400,
    lr: float = 1e-2,
    init=None,
    field=None,
    orientation=None,
    lattice_params=None,
    **kw,
) -> torch.Tensor:
    """Spatially-regularized, differentiable full-F field recovery.

    Minimises ``||A vec(H) - dQ||^2 + lambda_smooth * curvature(H)`` over the whole
    field jointly (the cross-voxel coupling a per-voxel closed-form solve cannot
    express), then ``F = (I+H)^{-T}``. Under realistic noise / limited reflections this
    lowers the recovery error vs :func:`recover_deformation_direct`. Returns ``F``
    ``(N, 3, 3)``. Reuses ``midas_invert.optimize.fit``. Reference frame handled as in
    :func:`recover_deformation_direct` (pass ``field=`` to match the forward exactly).
    """
    frame = _frame_kw(field, orientation, lattice_params, kw)
    A = deformation_design_matrix(reflections, dtype=measured.dtype, device=measured.device, **frame)
    if init is None:
        F0 = recover_deformation_direct(measured, reflections, **frame).detach()
        eye = torch.eye(3, dtype=measured.dtype, device=measured.device)
        vecH0 = (torch.linalg.inv(F0).transpose(-1, -2) - eye).reshape(-1, 9)
    else:
        vecH0 = torch.as_tensor(init, dtype=measured.dtype, device=measured.device).reshape(-1, 9)
    vecH = vecH0.clone().requires_grad_(True)

    def loss_fn():
        pred = vecH @ A.transpose(0, 1)                          # (N, 3m)
        data = (pred - measured).pow(2).mean()
        return data + lambda_smooth * _curvature_penalty(vecH, shape)

    fit([vecH], loss_fn, steps=steps, lr=lr)
    H = vecH.detach().reshape(-1, 3, 3)
    return _H_to_F(H)


def deformation_covariance(reflections, *, noise_std: float = 1.0, **kw) -> torch.Tensor:
    """Per-voxel covariance ``(A^T A)^{-1} sigma^2`` of ``vec(H)`` ``(9, 9)``.

    Linear-Gaussian propagation of the shift-measurement noise into the recovered
    tensor. The differentiable, closed-form UQ the incumbents provide only for their
    own estimator; here it composes with the Fisher matrix below.
    """
    A = deformation_design_matrix(reflections, **kw)
    return torch.linalg.inv(A.transpose(0, 1) @ A) * (noise_std ** 2)


def fisher_information(reflections, *, noise_std: float = 1.0, **kw) -> torch.Tensor:
    """Fisher information ``A^T A / sigma^2`` ``(9, 9)`` for ``vec(H)``.

    The principled generalization of Kanesalingam 2025's ``kappa(nu)`` proxy: the CRLB
    is its inverse (``= deformation_covariance``), and ``-log det`` / ``trace`` of the
    CRLB are the D-/A-optimality objectives for **gradient-based reflection-set design**
    (differentiable in the reflection geometry).
    """
    A = deformation_design_matrix(reflections, **kw)
    return (A.transpose(0, 1) @ A) / (noise_std ** 2)


def angular_sensitivity_matrix(
    reflections,
    *,
    wavelength_A: float = 0.71,
    orientation=None,
    lattice_params=(3.6356, 3.6356, 3.6356, 90.0, 90.0, 90.0),
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Sensitivity matrix ``(3m, 9)`` in ANGULAR units -- the Kanesalingam-2025 nu analogue.

    Maps ``vec(H)`` to the three measured angular shifts ``(Delta theta, phi, chi)`` per
    reflection: ``Delta theta = tan(theta) (dQ . e_par)/|Q0|`` (Bragg), and the transverse
    rocking/rolling ``(dQ . e_rock)/|Q0|``, ``(dQ . e_roll)/(|Q0| sin theta)``, with
    ``dQ = H Q0``. Unlike :func:`deformation_design_matrix` (raw ``dQ``), this normalizes
    by ``|Q0|`` and weights by the Bragg geometry, so its condition number is directly
    comparable to Kanesalingam's ``kappa(nu)`` reflection-set ranking. Differentiable.
    """
    from .resolution import _orthonormal_frame

    ori = _orientation(orientation, dtype, device)
    two_pi_over_lambda = 2.0 * torch.pi / wavelength_A
    rows = []
    for hkl in reflections:
        Q0 = reference_Q(hkl, ori, lattice_params)                 # (3,)
        qn = torch.linalg.vector_norm(Q0)
        sin_th = (qn / (2.0 * two_pi_over_lambda)).clamp(max=0.999999)
        theta = torch.arcsin(sin_th)
        tan_th = torch.tan(theta)
        e_par, e_rock, e_roll = _orthonormal_frame(Q0)
        # W rows map dQ -> (Delta theta, phi, chi)
        W = torch.stack([tan_th * e_par / qn, e_rock / qn, e_roll / (qn * sin_th)], dim=0)  # (3,3)
        # nu block: psi = W (H Q0) -> d psi_a / d H_ij = W[a,i] Q0[j]
        block = torch.stack([torch.outer(W[a], Q0).reshape(9) for a in range(3)], dim=0)     # (3,9)
        rows.append(block)
    return torch.cat(rows, dim=0)


def angular_condition_number(reflections, **kw) -> float:
    """Condition number of the angular sensitivity matrix -- the Kanesalingam ``kappa(nu)``."""
    s = torch.linalg.svdvals(angular_sensitivity_matrix(reflections, **kw))
    return float(s.max() / s[s > s.max() * 1e-12][-1])


def crlb_trace(reflections, *, noise_std: float = 1.0, **kw) -> float:
    """Total Cramer-Rao lower bound ``tr[(A^T A)^-1] sigma^2`` on ``vec(H)`` for a set.

    The A-optimality objective for reflection-set design: the minimum achievable
    sum-of-variances of the recovered tensor under measurement noise ``noise_std``.
    ``inf`` if the set is rank-deficient (cannot recover all nine components).
    """
    info = deformation_identifiability(reflections, **kw)
    if not info["recoverable"]:
        return float("inf")
    return float(torch.diagonal(deformation_covariance(reflections, noise_std=noise_std, **kw)).sum())


def select_reflections_greedy(
    pool,
    n_select: int,
    *,
    noise_std: float = 1.0,
    ridge: float = 1e-9,
    **kw,
) -> list:
    """Greedy A-optimal reflection-set selection (minimise the CRLB trace).

    Generalises Kanesalingam 2025's *enumerate-families-and-rank-kappa* to a
    differentiable-Fisher-driven *selection* from an arbitrary candidate ``pool``:
    greedily add the reflection that most lowers ``tr[(A^T A + ridge I)^-1]`` (the ridge
    keeps the objective finite while the set is still rank-deficient). Returns the chosen
    reflections (a list of ``n_select`` hkl tuples). Because the objective is the exact
    Fisher information, the selected set has provably lower recovered-tensor variance than
    a naively-chosen set (verified empirically in the R3.2 study).
    """
    pool = [tuple(h) for h in pool]
    selected: list = []
    dtype = kw.get("dtype", torch.float64)

    def score(refl):
        A = deformation_design_matrix(refl, **kw)
        G = A.transpose(0, 1) @ A + ridge * torch.eye(9, dtype=A.dtype, device=A.device)
        return float(torch.diagonal(torch.linalg.inv(G)).sum())

    while len(selected) < n_select:
        best, best_s = None, None
        for r in pool:
            if r in selected:
                continue
            s = score(selected + [r])
            if best_s is None or s < best_s:
                best, best_s = r, s
        selected.append(best)
    # swap-based local search: greedy alone is suboptimal (can be beaten by chance), so
    # refine by swapping each selected reflection for each unselected one while it lowers
    # the CRLB, until no swap improves -> a local optimum that empirically matches the
    # exhaustive optimum on modest pools.
    improved = True
    cur = score(selected)
    while improved:
        improved = False
        for i in range(len(selected)):
            for r in pool:
                if r in selected:
                    continue
                trial = selected[:i] + [r] + selected[i + 1:]
                s = score(trial)
                if s < cur - 1e-18:
                    selected, cur, improved = trial, s, True
    return selected


def decompose_deformation(F: torch.Tensor) -> dict:
    """Split ``F`` into strain + lattice rotation (small-strain and finite-strain).

    Returns ``{'strain_small', 'strain_green', 'rotation', 'rotation_vector'}``:
    ``strain_small = sym(F) - I``; ``strain_green = U - I`` (Biot) from the polar
    decomposition ``F = R U`` (finite strain, matching Kanesalingam 2025); ``rotation``
    the proper-orthogonal ``R``; ``rotation_vector`` its axial vector (small-angle).
    Batched ``(..., 3, 3)``. Differentiable.
    """
    eye = torch.eye(3, dtype=F.dtype, device=F.device)
    R, U = polar_decomposition(F)
    strain_small = 0.5 * (F + F.transpose(-1, -2)) - eye
    strain_green = U - eye
    w = 0.5 * torch.stack([R[..., 2, 1] - R[..., 1, 2],
                           R[..., 0, 2] - R[..., 2, 0],
                           R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    return {"strain_small": strain_small, "strain_green": strain_green,
            "rotation": R, "rotation_vector": w}
