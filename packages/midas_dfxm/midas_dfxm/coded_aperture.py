"""Coded-aperture (structured-illumination) depth-resolved DFXM.

Implements the APS 6-ID-C structured-illumination DFXM of Dark-Field X-ray
Microscopy with Structured Illumination for Three-Dimensional Imaging
(Comm. Phys. 2025, arXiv:2405.12799): a binary **de Bruijn** coded aperture is
scanned across the incident beam, casting a shift-coded silhouette so that depth
along the diffraction axis is recovered **without sample rotation**. For each
detector pixel the measurement series is a Toeplitz mixing of the depth profile,

    d = A s ,   A[m, n] = code[m + n]

with ``d`` (M aperture positions), ``s`` (N depth bins) and ``A`` the ``(M, N)``
coding matrix. The published reconstruction is a per-pixel **non-negative least
squares** solve with no coupling (:func:`decode_nnls`).

This module adds the regularized / compressive inverse the paper names as future
work (:func:`decode_regularized`): a differentiable, non-negative global solve
with smoothness priors **along depth** and **across neighbouring pixels**, which
is exactly where a per-pixel NNLS leaves information on the table (low aperture
counts, weak signal, beam-intensity drift). Everything torch-differentiable and
device/dtype-portable; the NNLS baseline delegates to SciPy.

Units: aperture positions / depth bins are index units; :func:`depth_axis`
converts a depth-bin index to a physical coordinate (micrometers) along the
scattering axis given the aperture step and Bragg angle.
"""
from __future__ import annotations

import numpy as np
import torch


def de_bruijn(order: int = 8, k: int = 2) -> np.ndarray:
    """Binary de Bruijn sequence ``B(k, order)`` (length ``k**order``).

    Every length-``order`` window of the cyclic sequence is unique, which is what
    makes the coded-aperture silhouette locally decodable. Default ``B(2, 8)`` is
    the length-256 order-8 sequence used at 6-ID-C. Returns a ``(k**order,)``
    int array of code bits (1 = absorbing Au bar, 0 = open membrane).
    """
    a = [0] * (k * order)
    seq: list[int] = []

    def db(t: int, p: int) -> None:
        if t > order:
            if order % p == 0:
                seq.extend(a[1 : p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return np.asarray(seq, dtype=np.int64)


def coding_matrix(
    code,
    n_positions: int,
    n_depth: int,
    *,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Toeplitz coding matrix ``A`` ``(n_positions, n_depth)`` from a binary ``code``.

    ``A[m, n] = code[(m + n) mod L]`` — the silhouette element seen by depth bin
    ``n`` at aperture position ``m`` (cyclic in the code length ``L`` so any scan
    range is admissible). The forward model is ``d = A s``. Differentiable in
    nothing (fixed instrument) but kept as a tensor for a torch forward.
    """
    c = torch.as_tensor(np.asarray(code), device=device, dtype=dtype)
    L = c.shape[0]
    m = torch.arange(n_positions, device=device).view(-1, 1)
    n = torch.arange(n_depth, device=device).view(1, -1)
    idx = (m + n) % L
    return c[idx]


def coded_forward(depth_signal: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Coded-aperture forward ``d = A s``.

    ``depth_signal`` is ``(N,)`` for a single pixel or ``(N, P)`` batched over
    ``P`` pixels; returns the measurement series ``(M,)`` or ``(M, P)``.
    Differentiable in ``depth_signal`` — the forward operator for the inverse.
    """
    return A @ depth_signal


def coded_identifiability(A: torch.Tensor) -> dict:
    """Conditioning of a coding matrix for depth recovery.

    Returns ``{'rank', 'cond', 'singular_values', 'recoverable'}``; ``recoverable``
    is ``True`` iff the depth profile is fully constrained (rank ``= n_depth``).
    Use to choose the aperture code / scan range before beamtime, analogous to
    :func:`midas_dfxm.strain_identifiability` for the strain inverse.
    """
    s = torch.linalg.svdvals(A)
    n_depth = A.shape[1]
    tol = s.max() * 1e-10
    rank = int((s > tol).sum())
    cond = float(s.max() / s[rank - 1]) if rank == n_depth else float("inf")
    return {"rank": rank, "cond": cond, "singular_values": s, "recoverable": rank == n_depth}


def decode_nnls(d, A) -> torch.Tensor:
    """Per-pixel non-negative least squares depth decode — the published baseline.

    Solves ``min_s ||A s - d||^2  s.t.  s >= 0`` independently per pixel via
    ``scipy.optimize.nnls`` (Comm. Phys. 2025). ``d`` is ``(M,)`` or ``(M, P)``;
    returns ``(N,)`` or ``(N, P)``. No cross-pixel or depth coupling — noise and
    beam drift pass straight through, which is the opening for
    :func:`decode_regularized`.
    """
    from scipy.optimize import nnls

    An = A.detach().cpu().numpy()
    dt = torch.as_tensor(d)
    single = dt.ndim == 1
    D = dt.reshape(dt.shape[0], -1).detach().cpu().numpy()
    out = np.stack([nnls(An, D[:, p])[0] for p in range(D.shape[1])], axis=1)
    res = torch.as_tensor(out, dtype=A.dtype, device=A.device)
    return res[:, 0] if single else res


def _depth_smoothness(s: torch.Tensor) -> torch.Tensor:
    """Curvature penalty along the depth axis (axis 0) of ``s`` ``(N, ...)``."""
    if s.shape[0] < 3:
        return torch.zeros((), device=s.device, dtype=s.dtype)
    return (s[2:] - 2 * s[1:-1] + s[:-2]).pow(2).mean()


def _pixel_smoothness(s: torch.Tensor, shape) -> torch.Tensor:
    """Curvature penalty across the detector grid for depth profiles ``s`` ``(N, P)``."""
    ny, nx = shape
    vol = s.transpose(0, 1).reshape(ny, nx, s.shape[0])  # (ny, nx, N)
    pen = torch.zeros((), device=s.device, dtype=s.dtype)
    if ny > 2:
        pen = pen + (vol[2:] - 2 * vol[1:-1] + vol[:-2]).pow(2).mean()
    if nx > 2:
        pen = pen + (vol[:, 2:] - 2 * vol[:, 1:-1] + vol[:, :-2]).pow(2).mean()
    return pen


def decode_regularized(
    d: torch.Tensor,
    A: torch.Tensor,
    *,
    lambda_depth: float = 0.0,
    lambda_pixel: float = 0.0,
    shape=None,
    steps: int = 400,
    lr: float = 5e-2,
    init=None,
) -> torch.Tensor:
    """Non-negative, regularized coded-aperture depth decode (differentiable).

    Minimises ``||A s - d||^2 + lambda_depth * curv_depth(s) +
    lambda_pixel * curv_pixel(s)`` over all pixels jointly, with ``s = softplus``
    enforcing non-negativity smoothly (so the solve is differentiable end-to-end,
    unlike the projected NNLS baseline). ``lambda_pixel`` couples neighbouring
    detector pixels (requires ``shape=(ny, nx)``) — the compressive / regularized
    reconstruction the paper defers to future work.

    ``d`` is ``(M,)`` or ``(M, P)``; returns ``(N,)`` or ``(N, P)``. Reuses
    ``midas_invert.optimize.fit`` for the loop.
    """
    from midas_invert.optimize import fit

    single = d.ndim == 1
    D = d.reshape(d.shape[0], -1)                       # (M, P)
    N, P = A.shape[1], D.shape[1]
    if init is None:
        base = decode_nnls(D, A).clamp_min(1e-6)        # warm start from baseline
    else:
        base = torch.as_tensor(init, dtype=A.dtype, device=A.device).reshape(N, P).clamp_min(1e-6)
    # invert softplus for the warm start: raw = log(exp(s) - 1)
    raw = torch.log(torch.expm1(base.clamp_min(1e-6))).clone().requires_grad_(True)

    def loss_fn():
        s = torch.nn.functional.softplus(raw)           # (N, P) >= 0
        data = (A @ s - D).pow(2).mean()
        reg = torch.zeros((), device=A.device, dtype=A.dtype)
        if lambda_depth > 0:
            reg = reg + lambda_depth * _depth_smoothness(s)
        if lambda_pixel > 0 and shape is not None:
            reg = reg + lambda_pixel * _pixel_smoothness(s, shape)
        return data + reg

    fit([raw], loss_fn, steps=steps, lr=lr)
    s = torch.nn.functional.softplus(raw).detach()
    return s[:, 0] if single else s


def depth_axis(
    n_depth: int,
    *,
    aperture_step_um: float = 1.0,
    two_theta_deg: float = 10.0,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Physical depth coordinate (micrometers) for each depth bin along the scattering axis.

    A one-step aperture shift advances the decoded silhouette by one depth bin; the
    projection onto the scattering axis scales the aperture step by ``1/sin(2*theta)``
    (small-angle geometry of the upstream coded mask, 6-ID-C). Returns ``(n_depth,)``
    centred on zero. This is a convenience mapping for plotting/interpretation; the
    inverse itself runs in bin units.
    """
    tt = torch.deg2rad(torch.as_tensor(two_theta_deg, device=device, dtype=dtype))
    pitch = aperture_step_um / torch.sin(tt)
    idx = torch.arange(n_depth, device=device, dtype=dtype)
    return (idx - (n_depth - 1) / 2.0) * pitch
