"""P2 — Per-hkl asterism fitting.

Given an average crystal orientation U, for every predicted (hkl) Bragg
position fit a 3-D anisotropic Gaussian (in sample-frame q-space) to the
voxels in a crop around the prediction. The fitted covariance Σ encodes
the orientation-spread (asterism) at that hkl; aggregating across hkls
gives a discrete ODF estimate.

Why sample-frame q-space?  Once we map a voxel from `(detY, detZ, ω)` to a
sample-frame q-vector, a perfect crystal's reflection sits at a single
fixed q — independent of ω.  An asterized reflection forms a 3-D cloud
around that q.  Fitting in q-space is the most direct measurement of
orientation spread.

Pipeline
--------
1. Predict the q_sample position of every allowed (hkl) up to qmax via
   `q_sample(hkl) = U @ g_cry(hkl, a, c)`.
2. For each predicted q0, gather voxels inside a crop box of half-extent
   `crop_halfwidth + crop_q_scale * |q0|` (asterism scales with |q|).
3. Fit a 3-D anisotropic Gaussian
       I(q) = A * exp( -1/2 * (q-q0)ᵀ Σ⁻¹ (q-q0) )  +  baseline
   with `Σ⁻¹ = L L.T` (Cholesky, 6 params) for positive-definite
   guarantee.  Adam optimizer on the weighted-least-squares loss.
4. Eigendecompose Σ to report principal half-widths and axis directions.

Differentiability
-----------------
* Every fitted parameter is a torch tensor with `requires_grad=True`.
* Cholesky parameterization keeps Σ ≻ 0 under autograd without projection.
* `predict_q_from_U` (from `seed_index`) gives the q0 prediction in a
  differentiable way w.r.t. (U, a, c), so the same fit can later be
  composed with a refinement of U or (a, c).

MIDAS reuses
------------
* `midas_stress.orientation` (when we eventually express axes in crystal frame)
* `midas_hkls.lattice_torch.d_spacing` (via `lattice.q_inv_of_hkl_torch`)
* `seed_index.predict_q_from_U` to keep the prediction path canonical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch

from midas_transforms.device import resolve_device, resolve_dtype

from .lattice import Shell, cual2_crystal, tetragonal_shells
from .seed_index import predict_q_from_U


__all__ = [
    "AsterismFit",
    "fit_asterism_patches",
    "predict_hkl_positions",
    "fit_single_patch",
    "build_bragg_residual_intensity",
    "strain_tensor_from_centroids",
]


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AsterismFit:
    """Single-hkl 3-D Gaussian fit result."""
    hkl: Tuple[int, int, int]
    q_pred: np.ndarray           # (3,) predicted q in sample frame
    q_fit: np.ndarray            # (3,) fitted centre
    amplitude: float             # peak height
    baseline: float              # constant offset
    sigma_eig: np.ndarray        # (3,) principal half-widths (sqrt eigenvalues of Σ)
    sigma_axes: np.ndarray       # (3, 3) principal axes (columns = eigenvectors of Σ)
    integrated_intensity: float  # Σ I over the crop
    n_voxels: int                # voxels in the crop
    final_loss: float
    converged: bool              # loss decreased monotonically

    def dominant_axis(self) -> np.ndarray:
        """Eigenvector of Σ with the largest eigenvalue (broadest asterism direction)."""
        return self.sigma_axes[:, np.argmax(self.sigma_eig)]

    def isotropy(self) -> float:
        """Ratio min(σ)/max(σ); 1.0 = perfectly spherical, near 0 = needle-like."""
        sg = np.sort(self.sigma_eig)
        return float(sg[0] / max(sg[-1], 1e-30))


# ---------------------------------------------------------------------------
# Forward prediction
# ---------------------------------------------------------------------------

def predict_hkl_positions(
    U: np.ndarray, a: float, c: float, *,
    q_max_inv_A: float = 10.0,
    crystal=None,
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Return predicted sample-frame q-vectors for every allowed (hkl).

    Returns (q_pred (N, 3), [hkl_i, ...]).
    """
    if crystal is None:
        crystal = cual2_crystal(a=a, c=c)
    shells = tetragonal_shells(crystal, q_max_inv_A=q_max_inv_A)
    hkls: List[Tuple[int, int, int]] = []
    g_cry = []
    twopi = 2.0 * math.pi
    for s in shells:
        for hkl in s.hkls:
            hkls.append(hkl)
            g_cry.append([twopi * hkl[0] / a,
                          twopi * hkl[1] / a,
                          twopi * hkl[2] / c])
            # And the centro-symmetric partner (Friedel pair) -- both light up
            hkls.append((-hkl[0], -hkl[1], -hkl[2]))
            g_cry.append([-twopi * hkl[0] / a,
                          -twopi * hkl[1] / a,
                          -twopi * hkl[2] / c])
    g_cry_np = np.asarray(g_cry, dtype=np.float64)        # (N, 3)
    q_pred = (U @ g_cry_np.T).T                            # (N, 3)
    return q_pred, hkls


# ---------------------------------------------------------------------------
# Differentiable single-patch fit
# ---------------------------------------------------------------------------

def _sigma_from_cholesky(L_params: torch.Tensor) -> torch.Tensor:
    """Build a positive-definite Σ from 6 Cholesky params.

    `L_params` is (..., 6) carrying (L11, L21, L22, L31, L32, L33).
    Returns Σ = (L L.T)^-1  -- but we actually fit Σ⁻¹ = L L.T to keep
    the math diff-friendly (Σ⁻¹ ≻ 0 ⟺ Σ ≻ 0).

    Diagonals are softplus-ed to stay positive.
    """
    softplus = torch.nn.functional.softplus
    L = torch.zeros(*L_params.shape[:-1], 3, 3,
                    dtype=L_params.dtype, device=L_params.device)
    L[..., 0, 0] = softplus(L_params[..., 0])
    L[..., 1, 0] = L_params[..., 1]
    L[..., 1, 1] = softplus(L_params[..., 2])
    L[..., 2, 0] = L_params[..., 3]
    L[..., 2, 1] = L_params[..., 4]
    L[..., 2, 2] = softplus(L_params[..., 5])
    Sigma_inv = L @ L.transpose(-1, -2)
    return Sigma_inv


def fit_single_patch(
    q_patch: torch.Tensor,             # (M, 3) voxel q-vectors
    I_patch: torch.Tensor,             # (M,)
    q_init: torch.Tensor,              # (3,)
    *,
    sigma_init: float = 0.05,          # initial 1-σ of Gaussian (1/Å)
    n_steps: int = 200,
    lr: float = 1e-2,
    loss_kind: str = "lsq",            # "lsq" | "sqrt_w" | "poisson"
    return_history: bool = False,
) -> dict:
    """Fit a 3-D anisotropic Gaussian + baseline to a patch.

    All tensors must be on the same device. Returns a dict with the fitted
    parameters and convergence info.
    """
    if q_patch.shape[0] < 6:
        raise ValueError(f"need at least 6 voxels to fit; got {q_patch.shape[0]}")

    dtype  = q_patch.dtype
    device = q_patch.device

    # parameters
    q0 = q_init.detach().clone().to(dtype=dtype, device=device).requires_grad_(True)
    log_A = torch.log(I_patch.max().clamp_min(1.0)).detach().clone().requires_grad_(True)
    log_baseline = torch.log(
        (I_patch.median() + 1.0).clamp_min(1.0)
    ).detach().clone().requires_grad_(True)
    # initial Σ⁻¹ = (1/sigma_init²) I  →  L11 = L22 = L33 = 1/sigma_init, off-diag=0
    diag_init = 1.0 / float(sigma_init)
    # softplus_inv: x = log(exp(x) - 1) for stable initialization
    softplus_inv = math.log(math.exp(diag_init) - 1.0)
    L_params = torch.tensor(
        [softplus_inv, 0.0, softplus_inv, 0.0, 0.0, softplus_inv],
        dtype=dtype, device=device,
    ).requires_grad_(True)

    opt = torch.optim.Adam([q0, log_A, log_baseline, L_params], lr=lr)
    history = []
    for step in range(n_steps):
        opt.zero_grad()
        Sigma_inv = _sigma_from_cholesky(L_params)
        delta = q_patch - q0
        quad = (delta @ Sigma_inv * delta).sum(dim=-1)
        pred = torch.exp(log_A) * torch.exp(-0.5 * quad) + torch.exp(log_baseline)
        if loss_kind == "lsq":
            loss = ((I_patch - pred) ** 2).sum()
        elif loss_kind == "sqrt_w":
            # Poisson-style weighting via sqrt(I) → relative residuals;
            # increases the influence of the diffuse wings on the fit.
            w = 1.0 / torch.sqrt(I_patch.clamp_min(1.0))
            loss = (((I_patch - pred) * w) ** 2).sum()
        elif loss_kind == "poisson":
            # Negative log Poisson likelihood (drop constant log-factorial).
            # pred is guaranteed > 0 because baseline = exp(log_baseline) > 0.
            loss = (pred - I_patch * torch.log(pred.clamp_min(1e-30))).sum()
        else:
            raise ValueError(f"unknown loss_kind: {loss_kind!r}")
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu()))

    Sigma_inv_final = _sigma_from_cholesky(L_params).detach()
    Sigma_final = torch.linalg.inv(Sigma_inv_final)
    # eigendecomposition for principal axes — move to CPU because
    # torch.linalg.eigh is not implemented on MPS. This is post-fit
    # diagnostic only (no gradient flow needed here).
    Sigma_cpu = Sigma_final.detach().cpu()
    eigvals, eigvecs = torch.linalg.eigh(Sigma_cpu)
    sigma_eig = torch.sqrt(eigvals.clamp_min(0.0))

    converged = len(history) > 2 and history[-1] <= history[0]

    return dict(
        q_fit=q0.detach().cpu().numpy(),
        amplitude=float(torch.exp(log_A).detach().cpu()),
        baseline=float(torch.exp(log_baseline).detach().cpu()),
        sigma_eig=sigma_eig.cpu().numpy(),
        sigma_axes=eigvecs.cpu().numpy(),
        Sigma=Sigma_final.cpu().numpy(),
        final_loss=history[-1],
        converged=bool(converged),
        history=history if return_history else None,
    )


# ---------------------------------------------------------------------------
# Pipeline: patches for every predicted hkl
# ---------------------------------------------------------------------------

def fit_asterism_patches(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, intensity: np.ndarray,
    *,
    U: np.ndarray, a: float, c: float,
    crystal=None,
    q_max_inv_A: float = 10.0,
    crop_halfwidth: float = 0.10,         # 1/Å base box half-width
    crop_q_scale: float = 0.03,           # plus a fraction-of-|q| term
    min_voxels: int = 20,
    sigma_init: float = 0.05,
    n_steps: int = 200,
    lr: float = 1e-2,
    loss_kind: str = "lsq",
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> List[AsterismFit]:
    """Fit a 3-D Gaussian asterism at every predicted hkl position.

    Returns a list of `AsterismFit` (one per hkl with enough voxels in the crop).
    """
    device_ = resolve_device(device)
    dtype_  = resolve_dtype(device_, dtype)

    q_pred_all, hkls_all = predict_hkl_positions(
        U=U, a=a, c=c, q_max_inv_A=q_max_inv_A, crystal=crystal,
    )

    q_all = np.stack([qx, qy, qz], axis=1)       # (N, 3)
    I_all = np.asarray(intensity, dtype=np.float64)

    out: List[AsterismFit] = []
    for q0, hkl in zip(q_pred_all, hkls_all):
        # crop box: half-width grows with |q0|
        half = crop_halfwidth + crop_q_scale * float(np.linalg.norm(q0))
        in_box = np.all(np.abs(q_all - q0[None, :]) < half, axis=1)
        n_in = int(in_box.sum())
        if n_in < min_voxels:
            continue

        q_patch_t = torch.as_tensor(q_all[in_box], dtype=dtype_, device=device_)
        I_patch_t = torch.as_tensor(I_all[in_box], dtype=dtype_, device=device_)
        q_init_t  = torch.as_tensor(q0, dtype=dtype_, device=device_)

        try:
            fit = fit_single_patch(
                q_patch_t, I_patch_t, q_init_t,
                sigma_init=sigma_init, n_steps=n_steps, lr=lr,
                loss_kind=loss_kind,
            )
        except Exception:
            continue

        out.append(AsterismFit(
            hkl=hkl,
            q_pred=q0.astype(np.float64),
            q_fit=fit["q_fit"].astype(np.float64),
            amplitude=fit["amplitude"],
            baseline=fit["baseline"],
            sigma_eig=fit["sigma_eig"].astype(np.float64),
            sigma_axes=fit["sigma_axes"].astype(np.float64),
            integrated_intensity=float(I_all[in_box].sum()),
            n_voxels=n_in,
            final_loss=fit["final_loss"],
            converged=fit["converged"],
        ))
    return out


def strain_tensor_from_centroids(
    fits: Sequence["AsterismFit"], *,
    weight_by_intensity: bool = True,
) -> dict:
    """Fit a 3-D strain tensor `ε` that best explains `q_fit − q_pred` across all hkls.

    Model: small-strain linearization in reciprocal space,

        q_obs = (I + ε) · q_pred   ⇒   q_obs − q_pred = ε · q_pred

    Stack one (q_pred, q_obs − q_pred) per hkl and solve the
    weighted-least-squares for the 9 entries of ε. The symmetric part is
    the true lattice-strain tensor; the antisymmetric part is residual
    rotation (should be small if the seed-orientation refinement is good).

    Returns dict with
      `epsilon` (3, 3): full strain tensor
      `epsilon_sym` (3, 3): symmetric part (true lattice strain)
      `epsilon_antisym` (3, 3): antisymmetric (residual rotation, ideally ~0)
      `residual_norm`: ‖q_obs − (I+ε)·q_pred‖₂ after fit
      `n_hkls`: number of fits used
      `principal_strains`: 3-vector of eigenvalues of `epsilon_sym`
      `principal_axes`: 3×3 matrix whose columns are eigenvectors
    """
    if len(fits) < 4:
        raise ValueError(
            f"need at least 4 fits to solve 9 strain entries; got {len(fits)}"
        )
    q_pred = np.array([f.q_pred for f in fits])     # (N, 3)
    q_obs  = np.array([f.q_fit  for f in fits])      # (N, 3)
    delta  = q_obs - q_pred                          # (N, 3)
    if weight_by_intensity:
        w = np.array([f.integrated_intensity for f in fits])
        w = w / w.sum()
    else:
        w = np.ones(len(fits)) / len(fits)

    # Per row of ε (say ε_i*): solve for the 3 entries from the i-th column
    # of delta. Stacked: delta_i = q_pred @ ε_i*.T  ⇒  ε_i*.T = lstsq(q_pred, delta_i)
    epsilon = np.zeros((3, 3))
    for i in range(3):
        A_mat = q_pred * w[:, None]                 # weight rows
        b_vec = delta[:, i] * w
        sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        epsilon[i, :] = sol                          # ε[i, j] = sol[j]

    pred_delta = q_pred @ epsilon.T
    resid_norm = float(np.linalg.norm(delta - pred_delta))
    eps_sym = 0.5 * (epsilon + epsilon.T)
    eps_anti = 0.5 * (epsilon - epsilon.T)
    eigvals, eigvecs = np.linalg.eigh(eps_sym)
    return dict(
        epsilon=epsilon,
        epsilon_sym=eps_sym,
        epsilon_antisym=eps_anti,
        residual_norm=resid_norm,
        n_hkls=len(fits),
        principal_strains=eigvals,
        principal_axes=eigvecs,
        volumetric_strain=float(np.trace(eps_sym)),
    )


def build_bragg_residual_intensity(
    qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, intensity: np.ndarray,
    fits: Sequence[AsterismFit],
    *,
    clip_negative: bool = True,
) -> np.ndarray:
    """Subtract the fitted Bragg model from the measured intensity.

    Returns a copy of `intensity` with each fit's 3-D Gaussian (without the
    baseline) subtracted. The residual is the diffuse-rod component plus any
    asterism wings the Gaussian model couldn't capture.

    Parameters
    ----------
    clip_negative : bool, default True
        If True, clip residuals below zero. Set False to keep the signed
        residual (useful for diagnostic plots).
    """
    q_all = np.stack([qx, qy, qz], axis=1)
    residual = intensity.astype(np.float64).copy()
    for f in fits:
        Sigma = f.sigma_axes @ np.diag(f.sigma_eig ** 2) @ f.sigma_axes.T
        Sigma_inv = np.linalg.inv(Sigma)
        delta = q_all - f.q_fit[None, :]
        quad = np.einsum("ni,ij,nj->n", delta, Sigma_inv, delta)
        bragg = f.amplitude * np.exp(-0.5 * quad)
        residual -= bragg
    if clip_negative:
        residual = np.clip(residual, 0.0, None)
    return residual
