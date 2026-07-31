"""Box reparameterisation and Tikhonov priors for L-BFGS fitting.

PyTorch L-BFGS is unbounded. To replicate the C-code's hard NLopt boxes
``[x0 - tol, x0 + tol]`` we expose each physical parameter as
``x = x0 + tol * tanh(u)`` where ``u`` is the unbounded variable the
optimiser actually sees. Optionally we also add a Tikhonov term that
softly pulls the parameter back toward its seed; this is layered *on
top of* the tanh, not as a replacement. Hard tanh keeps L-BFGS from
running off; Tikhonov says "even within the box, prefer the seed unless
data strongly disagrees".

The same module also exposes the C-style ΔLsd encoding: the C code
parameterises per-layer Lsd as ``Lsd[i] = Lsd[i-1] + ΔLsd[i]`` for i ≥ 1,
with the same per-layer tolerance applied to the deltas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
#  tanh box
# ---------------------------------------------------------------------------

class TanhBox:
    """Unbounded ↔ bounded reparameterisation for one parameter group.

    Forward: ``x = x0 + tol * tanh(u)``.
    Init:    ``u0 = 0`` so ``x = x0`` at construction. (We never need to
    invert from a non-seed start because L-BFGS always starts at u=0.)

    ``tol`` must be strictly positive, broadcastable against ``x0``.
    """

    def __init__(self, x0: torch.Tensor, tol: torch.Tensor | float):
        self._x0 = x0.detach().clone()
        if not torch.is_tensor(tol):
            tol = torch.full_like(self._x0, float(tol))
        if (tol <= 0).any():
            raise ValueError("tanh-box tol must be strictly positive")
        self._tol = tol.detach().clone()
        self._u = torch.zeros_like(self._x0, requires_grad=True)

    @property
    def u(self) -> torch.Tensor:
        """Unbounded optimiser variable (the leaf)."""
        return self._u

    @property
    def x(self) -> torch.Tensor:
        """Bounded physical variable: ``x0 + tol * tanh(u)``."""
        return self._x0 + self._tol * torch.tanh(self._u)

    def reset_to_seed(self) -> None:
        """Zero the unbounded variable so ``x == x0`` again. Used by the
        multi-start driver when seeding a fresh local search at a
        previously-found best.
        """
        with torch.no_grad():
            self._u.zero_()

    def perturb(self, scale: float, generator: torch.Generator | None = None) -> None:
        """Re-seed the unbounded variable with a Gaussian perturbation
        scaled by ``scale * tol``. ``scale`` is in units of "tolerance
        widths" — 0.3 is a reasonable default for multi-start.
        """
        with torch.no_grad():
            # Draw on the GENERATOR's device, then move. torch.randn refuses a
            # CPU generator with a CUDA target ("Expected a 'cuda' device type
            # for generator but found 'cpu'"), and callers reasonably build one
            # CPU generator for reproducibility regardless of where the
            # optimisation variables live -- fit_multipoint does exactly that,
            # which meant multi-start had never worked on CUDA at all.
            #
            # Drawing on the generator's device rather than switching the
            # generator to CUDA keeps the random stream IDENTICAL across
            # devices, so a CPU and a GPU run take the same multi-start path.
            gen_device = generator.device if generator is not None else self._u.device
            noise = torch.randn(
                self._u.shape, dtype=self._u.dtype, device=gen_device,
                generator=generator,
            )
            self._u.copy_(scale * noise.to(self._u.device))

    def tikhonov(self, sigma: torch.Tensor | float, lam: float = 1.0) -> torch.Tensor:
        """Quadratic penalty on the deviation from seed in σ-units.

        Returns ``lam * sum_p ((x_p - x0_p) / sigma_p)^2``. ``sigma`` may
        be a tensor (per-element) or a scalar.
        """
        if not torch.is_tensor(sigma):
            sigma = torch.full_like(self._x0, float(sigma))
        delta = self.x - self._x0
        return lam * ((delta / sigma) ** 2).sum()


# ---------------------------------------------------------------------------
#  Per-layer ΔLsd encoder (C-compatible)
# ---------------------------------------------------------------------------

@dataclass
class LsdEncoding:
    """Decomposition of per-layer Lsd into ``(Lsd[0], ΔLsd[1:])``.

    Mirrors ``FitOrientationParameters.c`` lines 206-215 exactly:

        x[6]   = Lsd[0]                  ;  ±LsdTol
        x[i]   = Lsd[i] - Lsd[i-1]        ;  ±LsdRelativeTol  (i ≥ 7)

    Forward and inverse round-trip with no information loss.
    """
    Lsd0: torch.Tensor          # scalar
    deltas: torch.Tensor        # (nLayers - 1,)

    @classmethod
    def from_lsds(cls, Lsds: torch.Tensor) -> "LsdEncoding":
        """Encode an ``(nLayers,)`` Lsd tensor."""
        if Lsds.ndim != 1:
            raise ValueError(f"Lsds must be 1D, got shape {Lsds.shape}")
        Lsd0 = Lsds[0:1].clone()
        if Lsds.shape[0] == 1:
            deltas = torch.zeros(0, dtype=Lsds.dtype, device=Lsds.device)
        else:
            deltas = (Lsds[1:] - Lsds[:-1]).clone()
        return cls(Lsd0=Lsd0, deltas=deltas)

    def decode(self) -> torch.Tensor:
        """Invert: cumulative sum of ``Lsd0`` + ``deltas``."""
        if self.deltas.numel() == 0:
            return self.Lsd0
        steps = torch.cat([self.Lsd0, self.deltas], dim=0)
        return torch.cumsum(steps, dim=0)


def normalize_orient_mat(om: np.ndarray) -> np.ndarray:
    """Scale an orientation matrix by ``det^(-1/3)``.

    Direct port of ``NormalizeMat`` in ``NF_HEDM/src/SharedFuncsFit.c:870-880``.
    The C ``OrientMat.bin`` writer accumulates float drift so the rotation
    matrices stored on disk are not exactly orthogonal; the C readers fix
    this by scaling all 9 elements by the cube-root of the determinant.
    Matches the C path bit-for-bit.

    Accepts a single ``(3, 3)`` matrix or a batch of shape ``(..., 3, 3)``.
    """
    if om.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3), got {om.shape}")
    det = np.linalg.det(om)
    # Guard against zero / sign-flipped dets — the C code does not, but
    # we want a sane result rather than a NaN. ``np.cbrt`` handles the
    # negative-determinant branch correctly (real cube root).
    scale = np.where(np.abs(det) > 1e-12, 1.0 / np.cbrt(det), 1.0)
    return om * scale[..., None, None]


# ---------------------------------------------------------------------------
#  Misorientation-uniqueness check (Top-N saves)
# ---------------------------------------------------------------------------
#
# Crystal-symmetry-aware misorientation lives in :mod:`midas_stress.orientation`
# (it has the full quaternion symmetry tables for every space group plus the
# fundamental-region reduction routine). Use those directly rather than
# re-porting the tables here; they are both C-backed (fast) with a pure-Python
# fallback when the C lib isn't available.

def quaternion_from_euler_zxz(eul: torch.Tensor) -> torch.Tensor:
    """Bunge ZXZ Euler angles → unit quaternion ``(w, x, y, z)``.

    Pure tensor implementation — does not allocate Python lists, so it
    composes with :func:`torch.func.vmap` if the caller wants to batch
    it. ``eul`` is in radians; output convention matches MIDAS's
    ``OrientMat2Quat``.
    """
    half = eul * 0.5
    c = torch.cos(half)
    s = torch.sin(half)
    c0, c1, c2 = c[..., 0], c[..., 1], c[..., 2]
    s0, s1, s2 = s[..., 0], s[..., 1], s[..., 2]
    w = c0 * c1 * c2 - s0 * c1 * s2
    x = c0 * s1 * c2 + s0 * s1 * s2
    y = -c0 * s1 * s2 + s0 * s1 * c2
    z = c0 * c1 * s2 + s0 * c1 * c2
    q = torch.stack([w, x, y, z], dim=-1)
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def misorientation_deg_symmetric(
    eul1: "torch.Tensor | np.ndarray | list",
    eul2: "torch.Tensor | np.ndarray | list",
    space_group: int,
) -> float:
    """Crystal-symmetry-aware misorientation in **degrees**.

    Thin wrapper around
    :func:`midas_stress.orientation.misorientation`, which ports the
    C ``GetMisOrientationAngle`` (NF_HEDM/src/GetMisorientation.c:381)
    bit-for-bit (it dlopens the C lib when available and falls back to
    pure Python otherwise).

    Both Eulers are interpreted as Bunge ZXZ in **radians**.
    """
    from midas_stress.orientation import misorientation
    e1 = _as_list(eul1)
    e2 = _as_list(eul2)
    angle_rad, _axis = misorientation(e1, e2, space_group)
    return float(angle_rad) * (180.0 / math.pi)


# ---------------------------------------------------------------------------
#  Vectorised pairwise misorientation (numpy, no per-pair Python loop)
# ---------------------------------------------------------------------------
#
# midas_stress's pure-Python ``misorientation`` is the dominant cost in the
# writeback loop: ~17 s on the full Au grid (3.5k voxels × ~4 winners each
# ≈ 14k offer() calls, each one walking 24 cubic symmetry quaternions in a
# Python loop).  Falling out to a vectorised numpy implementation here drops
# that to ≲0.5 s.  We match midas_stress's ``_misorientation_quat_pair_torch``
# semantics exactly: reduce each quaternion to its fundamental zone
# representative (max-w sym equivalent), then take δq = q1FR_inv ⊗ q2FR, then
# reduce δq to the FZ as well, and the rotation angle of that FZ rep is the
# disorientation.

_SYM_QUAT_CACHE: dict[int, np.ndarray] = {}


def _get_sym_quats(space_group: int) -> np.ndarray:
    """Cached ``(n_sym, 4)`` symmetry quaternion table for ``space_group``."""
    cached = _SYM_QUAT_CACHE.get(space_group)
    if cached is not None:
        return cached
    from midas_stress.orientation import make_symmetries
    n_sym, sym = make_symmetries(space_group)
    arr = np.asarray(sym, dtype=np.float64).reshape(n_sym, 4)
    _SYM_QUAT_CACHE[space_group] = arr
    return arr


def _quat_product_np(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Hamilton product ``q ⊗ r`` over the last axis; arbitrary leading dims."""
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rw, rx, ry, rz = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    w = qw * rw - qx * rx - qy * ry - qz * rz
    x = qw * rx + qx * rw + qy * rz - qz * ry
    y = qw * ry - qx * rz + qy * rw + qz * rx
    z = qw * rz + qx * ry - qy * rx + qz * rw
    return np.stack([w, x, y, z], axis=-1)


def euler_zxz_to_quat_np(euler: np.ndarray) -> np.ndarray:
    """Bunge ZXZ Euler (radians) → quaternion, batched, all-numpy.

    Goes via orient-matrix (using midas_stress's batched numpy path)
    so the result matches the conversion midas_stress's
    :func:`misorientation` performs internally.
    """
    from midas_stress.orientation import euler_to_orient_mat_batch
    eul = np.asarray(euler, dtype=np.float64)
    is_single = (eul.ndim == 1)
    if is_single:
        eul = eul[None, :]
    om_flat = np.asarray(euler_to_orient_mat_batch(eul), dtype=np.float64)
    om_flat = om_flat.reshape(-1, 9)

    m00, m01, m02 = om_flat[:, 0], om_flat[:, 1], om_flat[:, 2]
    m10, m11, m12 = om_flat[:, 3], om_flat[:, 4], om_flat[:, 5]
    m20, m21, m22 = om_flat[:, 6], om_flat[:, 7], om_flat[:, 8]
    trace = m00 + m11 + m22

    N = om_flat.shape[0]
    q = np.zeros((N, 4), dtype=np.float64)
    A = trace > 0
    B = (~A) & (m00 > m11) & (m00 > m22)
    C = (~A) & (~B) & (m11 > m22)
    D = (~A) & (~B) & (~C)

    if A.any():
        s = 0.5 / np.sqrt(trace[A] + 1.0)
        q[A, 0] = 0.25 / s
        q[A, 1] = (m21[A] - m12[A]) * s
        q[A, 2] = (m02[A] - m20[A]) * s
        q[A, 3] = (m10[A] - m01[A]) * s
    if B.any():
        s = 2.0 * np.sqrt(1.0 + m00[B] - m11[B] - m22[B])
        q[B, 0] = (m21[B] - m12[B]) / s
        q[B, 1] = 0.25 * s
        q[B, 2] = (m01[B] + m10[B]) / s
        q[B, 3] = (m02[B] + m20[B]) / s
    if C.any():
        s = 2.0 * np.sqrt(1.0 + m11[C] - m00[C] - m22[C])
        q[C, 0] = (m02[C] - m20[C]) / s
        q[C, 1] = (m01[C] + m10[C]) / s
        q[C, 2] = 0.25 * s
        q[C, 3] = (m12[C] + m21[C]) / s
    if D.any():
        s = 2.0 * np.sqrt(1.0 + m22[D] - m00[D] - m11[D])
        q[D, 0] = (m10[D] - m01[D]) / s
        q[D, 1] = (m02[D] + m20[D]) / s
        q[D, 2] = (m12[D] + m21[D]) / s
        q[D, 3] = 0.25 * s

    neg = q[:, 0] < 0
    q[neg] *= -1.0
    n = np.linalg.norm(q, axis=1, keepdims=True)
    q /= np.clip(n, 1e-12, None)
    return q[0] if is_single else q


def _fz_quat_np(q: np.ndarray, sym: np.ndarray) -> np.ndarray:
    """Reduce quaternion(s) to fundamental-zone representative.

    Mirrors :func:`midas_stress.orientation._fundamental_zone_torch`
    semantics: for each input quat, multiply with every symmetry,
    pick the result with the largest ``w`` component, normalise.
    """
    q_b = q[..., None, :]                                 # (..., 1, 4)
    qts = _quat_product_np(q_b, sym)                      # (..., n_sym, 4)
    idx = np.argmax(qts[..., 0], axis=-1)                 # (...,)
    out = np.take_along_axis(
        qts, idx[..., None, None].repeat(4, axis=-1), axis=-2,
    ).squeeze(-2)                                         # (..., 4)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.clip(norm, 1e-12, None)


def pairwise_miso_deg_vec(
    quat_new: np.ndarray,
    quats_existing: np.ndarray,
    space_group: int,
) -> np.ndarray:
    """Misorientation in **degrees** between one quaternion and a batch.

    Parameters
    ----------
    quat_new : np.ndarray (4,)
    quats_existing : np.ndarray (N, 4)
    space_group : int

    Returns
    -------
    np.ndarray (N,)
        Per-pair disorientation angle in degrees.
    """
    sym = _get_sym_quats(space_group)                     # (n_sym, 4)
    q1FR = _fz_quat_np(np.asarray(quat_new, dtype=np.float64), sym)
    q2FR = _fz_quat_np(np.asarray(quats_existing, dtype=np.float64), sym)
    q1_inv = q1FR.copy()
    q1_inv[1:] *= -1.0
    QP = _quat_product_np(q1_inv[None, :], q2FR)          # (N, 4)
    MisV = _fz_quat_np(QP, sym)                           # (N, 4)
    w = np.clip(MisV[..., 0], -1.0, 1.0)
    return 2.0 * np.arccos(w) * (180.0 / math.pi)


def _as_list(eul) -> list:
    if torch.is_tensor(eul):
        return eul.detach().cpu().tolist()
    if isinstance(eul, np.ndarray):
        return eul.tolist()
    return list(eul)


# ---------------------------------------------------------------------------
#  Convenience: parameter group assembly for L-BFGS
# ---------------------------------------------------------------------------

def gather_leaves(*boxes: TanhBox) -> Tuple[torch.Tensor, ...]:
    """Return the unbounded leaf tensors for one or more :class:`TanhBox`
    groups, in order. Pass directly to :class:`torch.optim.LBFGS`.
    """
    return tuple(b.u for b in boxes)
