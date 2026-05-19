"""Orientation representations, conversions, and misorientation computation.

Supports Euler angles, quaternions, orientation matrices, and Rodrigues vectors.

Two backends:
  - **NumPy** (default): pure-Python/NumPy implementations. No compiled
    dependency — works from a plain ``pip install`` with no MIDAS C build.
  - **PyTorch**: dispatched automatically when any input is a `torch.Tensor`.
    Returns torch tensors on the input's device and dtype. No autograd
    breakage — operations are differentiable end-to-end.

All Euler angles in RADIANS.  All misorientation angles returned in RADIANS.
"""

import math

import numpy as np
import torch
from scipy.linalg import expm

EPS = 1e-12
_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


# ===================================================================
#  Backend dispatch helpers
# ===================================================================

def _is_torch(*args) -> bool:
    """True if any positional arg is a torch.Tensor."""
    return any(isinstance(a, torch.Tensor) for a in args)


def _torch_dtype_device(*args) -> tuple[torch.dtype, torch.device]:
    """Pick (dtype, device) from the first torch.Tensor arg."""
    for a in args:
        if isinstance(a, torch.Tensor):
            return a.dtype, a.device
    return torch.float64, torch.device("cpu")


def _to_torch(x, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Coerce x (scalar/list/ndarray/tensor) to a tensor with target dtype/device."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype, device=device)
    return torch.as_tensor(x, dtype=dtype, device=device)


# ===================================================================
#  C library loading
# ===================================================================





# ===================================================================
#  Internal helpers
# ===================================================================

def _normalize_quat(q):
    return q / np.linalg.norm(q)


# ===================================================================
#  Symmetry tables (Python fallback)
# ===================================================================

_TricSym = [[1, 0, 0, 0]]
_MonoSym = [[1, 0, 0, 0], [0, 0, 1, 0]]
_OrtSym = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
_TetSym = [
    [1, 0, 0, 0], [0.70711, 0, 0, 0.70711], [0, 0, 0, 1], [0.70711, 0, 0, -0.70711],
    [0, 1, 0, 0], [0, 0, 1, 0], [0, 0.70711, 0.70711, 0], [0, -0.70711, 0.70711, 0],
]
_TetSymLow = [[1, 0, 0, 0], [0.70711, 0, 0, 0.70711], [0, 0, 0, 1], [0.70711, 0, 0, -0.70711]]
_TrigSym = [
    [1, 0, 0, 0], [0, 0.86603, -0.5, 0], [0.5, 0, 0, 0.86603],
    [0, 0, 1, 0], [0.5, 0, 0, -0.86603], [0, 0.86603, 0.5, 0],
]
_TrigSym2 = [
    [1, 0, 0, 0], [0.5, 0, 0, 0.86603], [0.5, 0, 0, -0.86603],
    [0, 0.5, -0.86603, 0], [0, 1, 0, 0], [0, 0.5, 0.86603, 0],
]
_TrigSymLow = [[1, 0, 0, 0], [0.5, 0, 0, 0.86603], [0.5, 0, 0, -0.86603]]
_HexSym = [
    [1, 0, 0, 0], [0.86603, 0, 0, 0.5], [0.5, 0, 0, 0.86603], [0, 0, 0, 1],
    [0.5, 0, 0, -0.86603], [0.86603, 0, 0, -0.5], [0, 1, 0, 0], [0, 0.86603, 0.5, 0],
    [0, 0.5, 0.86603, 0], [0, 0, 1, 0], [0, -0.5, 0.86603, 0], [0, -0.86603, 0.5, 0],
]
_HexSymLow = [
    [1, 0, 0, 0], [0.86603, 0, 0, 0.5], [0.5, 0, 0, 0.86603],
    [0, 0, 0, 1], [0.5, 0, 0, -0.86603], [0.86603, 0, 0, -0.5],
]
_CubSym = [
    [1, 0, 0, 0], [0.70711, 0.70711, 0, 0], [0, 1, 0, 0], [0.70711, -0.70711, 0, 0],
    [0.70711, 0, 0.70711, 0], [0, 0, 1, 0], [0.70711, 0, -0.70711, 0],
    [0.70711, 0, 0, 0.70711], [0, 0, 0, 1], [0.70711, 0, 0, -0.70711],
    [0.5, 0.5, 0.5, 0.5], [0.5, -0.5, -0.5, -0.5], [0.5, -0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5, -0.5], [0.5, 0.5, -0.5, 0.5], [0.5, -0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5, 0.5], [0.5, 0.5, 0.5, -0.5], [0, 0.70711, 0.70711, 0],
    [0, -0.70711, 0.70711, 0], [0, 0.70711, 0, 0.70711], [0, 0.70711, 0, -0.70711],
    [0, 0, 0.70711, 0.70711], [0, 0, 0.70711, -0.70711],
]
_CubSymLow = [
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    [0.5, 0.5, 0.5, 0.5], [0.5, -0.5, -0.5, -0.5], [0.5, -0.5, 0.5, 0.5],
    [0.5, 0.5, -0.5, -0.5], [0.5, 0.5, -0.5, 0.5], [0.5, -0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5, 0.5], [0.5, 0.5, 0.5, -0.5],
]
_TrigType2SGs = {149, 151, 153, 157, 159, 162, 163}


# ===================================================================
#  Core functions — C-backed with pure-Python fallback
# ===================================================================

def make_symmetries(space_group: int):
    """Generate symmetry quaternions for a space group.

    Parameters
    ----------
    space_group : int
        Space group number (1-230).

    Returns
    -------
    n_sym : int
        Number of symmetry operations.
    sym : list of list
        Symmetry quaternions, each [w, x, y, z].
    """
    return _make_symmetries_py(space_group)


def euler_to_orient_mat(euler) -> list:
    """Euler angles (radians) -> flat 9-element orientation matrix.

    Parameters
    ----------
    euler : array-like or torch.Tensor, shape (3,)
        Euler angles [psi, phi, theta] in radians.

    Returns
    -------
    list of float (NumPy backend) or torch.Tensor shape (9,) (torch backend).
    """
    if _is_torch(euler):
        return _euler_to_orient_mat_torch(euler)
    return _euler_to_orient_mat_py(euler)


def orient_mat_to_quat(orient_mat) -> np.ndarray:
    """Orientation matrix -> quaternion [w, x, y, z].

    Parameters
    ----------
    orient_mat : array-like or torch.Tensor, length 9 or shape (3, 3) or
                 a leading-batch (..., 9) / (..., 3, 3) tensor.

    Returns
    -------
    ndarray shape (4,) (NumPy backend) or torch.Tensor of matching leading
    batch shape with a trailing (4,) (torch backend).
    """
    if _is_torch(orient_mat):
        return _orient_mat_to_quat_torch(orient_mat)
    return _orient_mat_to_quat_py(orient_mat)


def orient_mat_to_euler(m) -> np.ndarray:
    """Orientation matrix -> Euler angles (radians).

    Parameters
    ----------
    m : ndarray, shape (3, 3) or (9,); or torch.Tensor of same shape.

    Returns
    -------
    ndarray shape (3,) (NumPy backend) or torch.Tensor (..., 3) (torch backend).
    """
    if _is_torch(m):
        return _orient_mat_to_euler_torch(m)
    m = np.asarray(m, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape((3, 3))
    return _orient_mat_to_euler_py(m)


def quaternion_product(q, r) -> np.ndarray:
    """Hamilton product q * r.  Returns normalized with w >= 0.

    Parameters
    ----------
    q, r : array-like or torch.Tensor, shape (4,) or (..., 4) for the torch
           backend (broadcasting allowed).

    Returns
    -------
    ndarray shape (4,) (NumPy backend) or torch.Tensor of the broadcast
    shape with trailing (4,) (torch backend).
    """
    if _is_torch(q, r):
        return _quaternion_product_torch(q, r)
    return _quaternion_product_py(q, r)


def quat_to_orient_mat(q) -> list:
    """Quaternion [w, x, y, z] -> orientation matrix (flat-9).

    Parameters
    ----------
    q : array-like or torch.Tensor, shape (4,) or (..., 4) for batched torch.

    Returns
    -------
    list of float length 9 (NumPy backend) or torch.Tensor of shape
    (..., 9) (torch backend).
    """
    if _is_torch(q):
        return _quat_to_orient_mat_torch(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return [
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ]


def fundamental_zone(quat, space_group: int | None = None, *, sym=None) -> np.ndarray:
    """Reduce quaternion to fundamental region for a space group.

    Parameters
    ----------
    quat : array-like or torch.Tensor, shape (4,) or (..., 4)
    space_group : int, optional
        Required unless `sym` is provided.
    sym : array-like or torch.Tensor, optional
        Pre-computed symmetry table (n_sym, 4). When supplied, skips the
        per-call `make_symmetries(space_group)` lookup; useful when many
        quaternions are reduced under the same space group in a hot loop.
        If both `space_group` and `sym` are passed, `sym` wins.

    Returns
    -------
    ndarray shape (4,) (NumPy backend) or torch.Tensor of the input's
    broadcast shape with trailing (4,) (torch backend).
    """
    if sym is None and space_group is None:
        raise ValueError("fundamental_zone requires `space_group` or `sym`")
    if _is_torch(quat):
        return _fundamental_zone_torch(quat, space_group, sym=sym)
    if sym is None:
        n_sym, sym_use = make_symmetries(space_group)
    else:
        sym_use = np.asarray(sym, dtype=np.float64)
        if sym_use.ndim != 2 or sym_use.shape[1] != 4:
            raise ValueError(f"sym must have shape (n_sym, 4); got {sym_use.shape}")
        n_sym = sym_use.shape[0]
        sym_use = [list(row) for row in sym_use]
    return _fundamental_zone_py(quat, n_sym, sym_use)


def matrix_mult_f33(m, n):
    """3×3 matrix multiplication. Numpy and torch transparent.

    Parameters
    ----------
    m, n : array-like or torch.Tensor, shape (3, 3) — or batched (..., 3, 3) for torch.

    Returns
    -------
    ndarray (3, 3) (NumPy backend) or torch.Tensor (..., 3, 3) (torch backend).
    """
    if _is_torch(m, n):
        dtype, device = _torch_dtype_device(m, n)
        return torch.matmul(
            _to_torch(m, dtype=dtype, device=device),
            _to_torch(n, dtype=dtype, device=device),
        )
    return np.asarray(m, dtype=np.float64) @ np.asarray(n, dtype=np.float64)


def misorientation(euler1, euler2, space_group: int):
    """Misorientation between two Euler-angle orientations.

    Parameters
    ----------
    euler1, euler2 : array-like, shape (3,)
        Euler angles in radians.
    space_group : int
        Space group number (1-230).

    Returns
    -------
    angle : float
        Misorientation angle in radians.
    axis : ndarray, shape (3,)
        Misorientation axis (unit vector, or zero if angle ~ 0).
    """
    om1 = euler_to_orient_mat(euler1)
    om2 = euler_to_orient_mat(euler2)
    return misorientation_om(om1, om2, space_group)


def misorientation_om(om1, om2, space_group: int):
    """Misorientation between two orientation matrices.

    Parameters
    ----------
    om1, om2 : array-like or torch.Tensor — length 9 or shape (3, 3).
    space_group : int

    Returns
    -------
    angle : float (NumPy) or torch.Tensor scalar (torch)
        Misorientation angle in radians.
    axis : ndarray (3,) (NumPy) or torch.Tensor (3,) (torch).
    """
    if _is_torch(om1, om2):
        return _misorientation_om_torch(om1, om2, space_group)
    q1 = orient_mat_to_quat(om1)
    q2 = orient_mat_to_quat(om2)
    n_sym, sym = make_symmetries(space_group)

    q1FR = _fundamental_zone_py(list(q1), n_sym, sym)
    q2FR = _fundamental_zone_py(list(q2), n_sym, sym)
    q1FR[0] = -q1FR[0]
    QP = _quaternion_product_py(q1FR, q2FR)
    MisV = _fundamental_zone_py(QP, n_sym, sym)
    w = min(1.0, float(MisV[0]))
    ang = 2.0 * math.acos(w)

    # Compute axis
    half = ang / 2.0
    s = math.sin(half) if ang > EPS else 0.0
    if abs(s) < EPS:
        return ang, np.array([0.0, 0.0, 0.0])

    # Derive axis from misorientation quaternion
    q1FR = _fundamental_zone_py(list(q1), n_sym, sym)
    q2FR = _fundamental_zone_py(list(q2), n_sym, sym)

    q1FR_inv = np.array([-q1FR[0], q1FR[1], q1FR[2], q1FR[3]])
    QP = quaternion_product(q1FR_inv, q2FR)

    MisV = _fundamental_zone_py(QP, n_sym, sym)

    return ang, np.array(MisV[1:4]) / s


# ===================================================================
#  Batch functions
# ===================================================================

def misorientation_om_batch(oms1, oms2, space_group: int) -> np.ndarray:
    """Batch misorientation for n pairs of orientation matrices.

    Parameters
    ----------
    oms1, oms2 : ndarray or torch.Tensor — (n, 9) or (n, 3, 3).
    space_group : int

    Returns
    -------
    ndarray (n,) (NumPy backend) or torch.Tensor (n,) (torch backend).
    """
    if _is_torch(oms1, oms2):
        return _misorientation_om_batch_torch(oms1, oms2, space_group)
    oms1 = np.ascontiguousarray(oms1, dtype=np.float64)
    oms2 = np.ascontiguousarray(oms2, dtype=np.float64)
    n = oms1.shape[0]
    angles = np.empty(n, dtype=np.float64)
    for i in range(n):
        angles[i], _ = misorientation_om(list(oms1[i]), list(oms2[i]), space_group)
    return angles


def misorientation_quat_batch(quats1, quats2, space_group: int) -> np.ndarray:
    """Batch misorientation for n pairs of quaternions.

    Parameters
    ----------
    quats1, quats2 : ndarray or torch.Tensor (n, 4)
    space_group : int

    Returns
    -------
    ndarray (n,) (NumPy backend) or torch.Tensor (n,) (torch backend).
    """
    if _is_torch(quats1, quats2):
        return _misorientation_quat_batch_torch(quats1, quats2, space_group)
    quats1 = np.ascontiguousarray(quats1, dtype=np.float64)
    quats2 = np.ascontiguousarray(quats2, dtype=np.float64)
    n = quats1.shape[0]
    angles = np.empty(n, dtype=np.float64)
    for i in range(n):
        om1 = quat_to_orient_mat(list(quats1[i]))
        om2 = quat_to_orient_mat(list(quats2[i]))
        angles[i], _ = misorientation_om(om1, om2, space_group)
    return angles


# ===================================================================
#  Vectorized Euler -> orient mat (pure NumPy, no C needed)
# ===================================================================

def euler_to_orient_mat_batch(euler) -> np.ndarray:
    """Vectorized Euler angles (radians) -> flat-9 orientation matrices.

    Parameters
    ----------
    euler : ndarray or torch.Tensor (n, 3)

    Returns
    -------
    ndarray (n, 9) (NumPy backend) or torch.Tensor (n, 9) (torch backend).
    """
    if _is_torch(euler):
        return _euler_to_orient_mat_batch_torch(euler)
    cps = np.cos(euler[:, 0]); cph = np.cos(euler[:, 1]); cth = np.cos(euler[:, 2])
    sps = np.sin(euler[:, 0]); sph = np.sin(euler[:, 1]); sth = np.sin(euler[:, 2])
    m = np.zeros((euler.shape[0], 9))
    m[:, 0] = cth * cps - sth * cph * sps
    m[:, 1] = -cth * cph * sps - sth * cps
    m[:, 2] = sph * sps
    m[:, 3] = cth * sps + sth * cph * cps
    m[:, 4] = cth * cph * cps - sth * sps
    m[:, 5] = -sph * cps
    m[:, 6] = sth * sph
    m[:, 7] = cth * sph
    m[:, 8] = cph
    return m


# ===================================================================
#  Utility functions
# ===================================================================

def axis_angle_to_orient_mat(axis, angle_deg) -> np.ndarray:
    """Axis-angle to 3x3 orientation matrix (Rodrigues formula).

    Parameters
    ----------
    axis : array-like or torch.Tensor, shape (3,) — or (..., 3) for batched torch.
    angle_deg : float, ndarray or torch.Tensor (degrees). Broadcasts with axis.

    Returns
    -------
    ndarray (3, 3) (NumPy backend) or torch.Tensor (..., 3, 3) (torch backend).
    """
    if _is_torch(axis, angle_deg):
        return _axis_angle_to_orient_mat_torch(axis, angle_deg)
    u = np.asarray(axis, dtype=np.float64)
    u = u / np.linalg.norm(u)
    angle = float(angle_deg) * _DEG2RAD
    c = math.cos(angle)
    s = math.sin(angle)
    R = np.zeros((3, 3))
    R[0, 0] = c + u[0]*u[0]*(1-c)
    R[1, 0] = u[2]*s + u[1]*u[0]*(1-c)
    R[2, 0] = -u[1]*s + u[2]*u[0]*(1-c)
    R[0, 1] = -u[2]*s + u[0]*u[1]*(1-c)
    R[1, 1] = c + u[1]*u[1]*(1-c)
    R[2, 1] = u[0]*s + u[2]*u[1]*(1-c)
    R[0, 2] = u[1]*s + u[0]*u[2]*(1-c)
    R[1, 2] = -u[0]*s + u[1]*u[2]*(1-c)
    R[2, 2] = c + u[2]*u[2]*(1-c)
    return R


def rodrigues_to_orient_mat(rod) -> np.ndarray:
    """Rodrigues vector to 3x3 orientation matrix.

    Parameters
    ----------
    rod : array-like or torch.Tensor, shape (3,) — or (..., 3) for batched torch.

    Returns
    -------
    ndarray (3, 3) (NumPy backend) or torch.Tensor (..., 3, 3) (torch backend).
    """
    if _is_torch(rod):
        return _rodrigues_to_orient_mat_torch(rod)
    rod = np.asarray(rod, dtype=np.float64)
    norm = np.linalg.norm(rod)
    cThOver2 = math.cos(math.atan(norm))
    th = 2 * math.atan(norm)
    quat = np.array([cThOver2, rod[0]/cThOver2, rod[1]/cThOver2, rod[2]/cThOver2])
    if th > EPS:
        w = quat[1:] * th / math.sin(th / 2)
    else:
        w = np.array([0.0, 0.0, 0.0])
    wskew = np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0],
    ])
    return expm(wskew)


# ===================================================================
#  Pure-Python fallback implementations
# ===================================================================

def _make_symmetries_py(sg):
    if sg <= 2:    return 1, _TricSym
    elif sg <= 15: return 2, _MonoSym
    elif sg <= 74: return 4, _OrtSym
    elif sg <= 88: return 4, _TetSymLow
    elif sg <= 142: return 8, _TetSym
    elif sg <= 148: return 3, _TrigSymLow
    elif sg <= 167: return 6, (_TrigSym2 if sg in _TrigType2SGs else _TrigSym)
    elif sg <= 176: return 6, _HexSymLow
    elif sg <= 194: return 12, _HexSym
    elif sg <= 206: return 12, _CubSymLow
    elif sg <= 230: return 24, _CubSym
    else: return 0, []


def _quaternion_product_py(q, r):
    Q = [0, 0, 0, 0]
    Q[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3]
    Q[1] = r[1]*q[0] + r[0]*q[1] + r[3]*q[2] - r[2]*q[3]
    Q[2] = r[2]*q[0] + r[0]*q[2] + r[1]*q[3] - r[3]*q[1]
    Q[3] = r[3]*q[0] + r[0]*q[3] + r[2]*q[1] - r[1]*q[2]
    if Q[0] < 0:
        Q = [-Q[0], -Q[1], -Q[2], -Q[3]]
    return _normalize_quat(np.array(Q))


def _fundamental_zone_py(quat_in, n_sym, sym):
    max_cos = -10000.0
    quat_out = np.asarray(quat_in, dtype=np.float64)
    for i in range(n_sym):
        qt = _quaternion_product_py(quat_in, sym[i])
        if max_cos < qt[0]:
            max_cos = qt[0]
            quat_out = qt
    return _normalize_quat(quat_out)


def _orient_mat_to_quat_py(om):
    trace = om[0] + om[4] + om[8]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        q = [0.25/s, (om[7]-om[5])*s, (om[2]-om[6])*s, (om[3]-om[1])*s]
    elif om[0] > om[4] and om[0] > om[8]:
        s = 2.0 * math.sqrt(1.0 + om[0] - om[4] - om[8])
        q = [(om[7]-om[5])/s, 0.25*s, (om[1]+om[3])/s, (om[2]+om[6])/s]
    elif om[4] > om[8]:
        s = 2.0 * math.sqrt(1.0 + om[4] - om[0] - om[8])
        q = [(om[2]-om[6])/s, (om[1]+om[3])/s, 0.25*s, (om[5]+om[7])/s]
    else:
        s = 2.0 * math.sqrt(1.0 + om[8] - om[0] - om[4])
        q = [(om[3]-om[1])/s, (om[2]+om[6])/s, (om[5]+om[7])/s, 0.25*s]
    if q[0] < 0:
        q = [-q[0], -q[1], -q[2], -q[3]]
    return _normalize_quat(np.array(q))


def _euler_to_orient_mat_py(euler):
    psi, phi, theta = euler[0], euler[1], euler[2]
    cps, sps = math.cos(psi), math.sin(psi)
    cph, sph = math.cos(phi), math.sin(phi)
    cth, sth = math.cos(theta), math.sin(theta)
    return [
        cth*cps - sth*cph*sps, -cth*cph*sps - sth*cps, sph*sps,
        cth*sps + sth*cph*cps,  cth*cph*cps - sth*sps, -sph*cps,
        sth*sph, cth*sph, cph,
    ]


def _sin_cos_to_angle(s, c):
    c = max(-1.0, min(1.0, c))
    return math.acos(c) if s >= 0 else 2.0 * math.pi - math.acos(c)


def _orient_mat_to_euler_py(m):
    if m.ndim == 1:
        m = m.reshape((3, 3))
    val = min(1.0, max(-1.0, float(m[2, 2])))
    phi = 0.0 if abs(val - 1.0) < EPS else math.acos(val)
    sph = math.sin(phi)
    if abs(sph) < EPS:
        psi = 0.0
        theta = (_sin_cos_to_angle(m[1, 0], m[0, 0]) if abs(val - 1.0) < EPS
                 else _sin_cos_to_angle(-m[1, 0], m[0, 0]))
    else:
        c_psi = -m[1, 2] / sph
        c_psi = max(-1.0, min(1.0, c_psi))
        psi = _sin_cos_to_angle(m[0, 2] / sph, c_psi)
        c_th = m[2, 1] / sph
        c_th = max(-1.0, min(1.0, c_th))
        theta = _sin_cos_to_angle(m[2, 0] / sph, c_th)
    return np.array([psi, phi, theta])


# ===================================================================
#  PyTorch-backend implementations
#
#  Activated by isinstance(arg, torch.Tensor) dispatch in the public
#  functions above. All implementations:
#    - return torch.Tensor on the input's device and dtype
#    - support batched inputs where the NumPy API is single-instance
#    - are differentiable end-to-end
# ===================================================================


def _euler_to_orient_mat_torch(euler: torch.Tensor) -> torch.Tensor:
    """(3,) Euler [psi, phi, theta] (rad) -> (9,) flat orientation matrix."""
    psi, phi, theta = euler[..., 0], euler[..., 1], euler[..., 2]
    cps, sps = torch.cos(psi), torch.sin(psi)
    cph, sph = torch.cos(phi), torch.sin(phi)
    cth, sth = torch.cos(theta), torch.sin(theta)
    return torch.stack(
        [
            cth * cps - sth * cph * sps,
            -cth * cph * sps - sth * cps,
            sph * sps,
            cth * sps + sth * cph * cps,
            cth * cph * cps - sth * sps,
            -sph * cps,
            sth * sph,
            cth * sph,
            cph,
        ],
        dim=-1,
    )


def _euler_to_orient_mat_batch_torch(euler: torch.Tensor) -> torch.Tensor:
    """(n, 3) Euler -> (n, 9). Same formulas as single-shot."""
    return _euler_to_orient_mat_torch(euler)


def _quaternion_product_torch(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Hamilton product q * r, normalized, w >= 0. Broadcasts over leading dims."""
    dtype, device = _torch_dtype_device(q, r)
    q = _to_torch(q, dtype=dtype, device=device)
    r = _to_torch(r, dtype=dtype, device=device)
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rw, rx, ry, rz = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    # Matches _quaternion_product_py (note operand order in C):
    #   Q[0] = r0*q0 - r1*q1 - r2*q2 - r3*q3
    #   Q[1] = r1*q0 + r0*q1 + r3*q2 - r2*q3
    #   Q[2] = r2*q0 + r0*q2 + r1*q3 - r3*q1
    #   Q[3] = r3*q0 + r0*q3 + r2*q1 - r1*q2
    Q0 = rw * qw - rx * qx - ry * qy - rz * qz
    Q1 = rx * qw + rw * qx + rz * qy - ry * qz
    Q2 = ry * qw + rw * qy + rx * qz - rz * qx
    Q3 = rz * qw + rw * qz + ry * qx - rx * qy
    Q = torch.stack([Q0, Q1, Q2, Q3], dim=-1)
    sign = torch.where(Q[..., :1] < 0, -torch.ones_like(Q[..., :1]), torch.ones_like(Q[..., :1]))
    Q = Q * sign
    norm = torch.linalg.vector_norm(Q, dim=-1, keepdim=True).clamp_min(EPS)
    return Q / norm


def _quat_to_orient_mat_torch(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) -> (..., 9) flat orientation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    out = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return out


def _normalize_om(om: torch.Tensor) -> torch.Tensor:
    """Accept either (..., 9) or (..., 3, 3) and return (..., 9)."""
    if om.shape[-1] == 9 and om.dim() >= 1 and om.shape[-1] != 3:
        return om
    if om.shape[-2:] == (3, 3):
        return om.reshape(*om.shape[:-2], 9)
    if om.shape[-1] == 3 and om.shape[-2:] != (3, 3):
        # leaf is (..., 3) — caller probably passed Euler by mistake
        raise ValueError(
            f"Expected shape (..., 9) or (..., 3, 3); got {tuple(om.shape)}"
        )
    return om


def _orient_mat_to_quat_torch(orient_mat: torch.Tensor) -> torch.Tensor:
    """(..., 9) or (..., 3, 3) -> (..., 4) quaternion [w, x, y, z]."""
    om = _normalize_om(orient_mat)
    m0, m1, m2 = om[..., 0], om[..., 1], om[..., 2]
    m3, m4, m5 = om[..., 3], om[..., 4], om[..., 5]
    m6, m7, m8 = om[..., 6], om[..., 7], om[..., 8]
    trace = m0 + m4 + m8

    # Branch A: trace > 0
    sA = 0.5 / torch.sqrt(torch.clamp(trace + 1.0, min=EPS))
    qA = torch.stack(
        [0.25 / sA, (m7 - m5) * sA, (m2 - m6) * sA, (m3 - m1) * sA],
        dim=-1,
    )

    # Branch B: m0 dominant
    sB = 2.0 * torch.sqrt(torch.clamp(1.0 + m0 - m4 - m8, min=EPS))
    qB = torch.stack([(m7 - m5) / sB, 0.25 * sB, (m1 + m3) / sB, (m2 + m6) / sB], dim=-1)

    # Branch C: m4 dominant
    sC = 2.0 * torch.sqrt(torch.clamp(1.0 + m4 - m0 - m8, min=EPS))
    qC = torch.stack([(m2 - m6) / sC, (m1 + m3) / sC, 0.25 * sC, (m5 + m7) / sC], dim=-1)

    # Branch D: m8 dominant
    sD = 2.0 * torch.sqrt(torch.clamp(1.0 + m8 - m0 - m4, min=EPS))
    qD = torch.stack([(m3 - m1) / sD, (m2 + m6) / sD, (m5 + m7) / sD, 0.25 * sD], dim=-1)

    use_A = (trace > 0).unsqueeze(-1)
    use_B = ((m0 > m4) & (m0 > m8)).unsqueeze(-1)
    use_C = (m4 > m8).unsqueeze(-1)

    q = torch.where(use_A, qA, torch.where(use_B, qB, torch.where(use_C, qC, qD)))

    # Hemisphere flip + normalize.
    sign = torch.where(q[..., :1] < 0, -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    q = q * sign
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(EPS)
    return q / norm


def _orient_mat_to_euler_torch(m: torch.Tensor) -> torch.Tensor:
    """(3,3) or (9,) or (..., 3, 3) -> (..., 3) Euler [psi, phi, theta] (rad).

    Mirrors `_orient_mat_to_euler_py` including the gimbal-lock branch. Single
    matrices supported; batches are processed elementwise via torch.where.
    """
    if m.dim() == 1 and m.shape[-1] == 9:
        m = m.reshape(3, 3)
    elif m.shape[-1] == 9 and m.dim() >= 2:
        m = m.reshape(*m.shape[:-1], 3, 3)
    if m.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (3,3) or (9,) input; got {tuple(m.shape)}")

    val = m[..., 2, 2].clamp(-1.0, 1.0)
    is_gimbal = (1.0 - val).abs() < EPS

    phi = torch.acos(val)
    sph = torch.sin(phi)
    sph_safe = torch.where(is_gimbal, torch.ones_like(sph), sph)

    # Non-gimbal branch
    c_psi = (-m[..., 1, 2] / sph_safe).clamp(-1.0, 1.0)
    s_psi_arg = m[..., 0, 2] / sph_safe
    psi_pi = torch.acos(c_psi)
    psi = torch.where(s_psi_arg >= 0, psi_pi, 2 * math.pi - psi_pi)

    c_th = (m[..., 2, 1] / sph_safe).clamp(-1.0, 1.0)
    s_th_arg = m[..., 2, 0] / sph_safe
    th_pi = torch.acos(c_th)
    theta = torch.where(s_th_arg >= 0, th_pi, 2 * math.pi - th_pi)

    # Gimbal branches:
    psi_g = torch.zeros_like(phi)
    # val ~ +1: phi=0, theta = atan2(m10, m00); val ~ -1: phi=pi, theta = atan2(-m10, m00)
    val_pos = (val - 1.0).abs() < EPS
    th_pos_pi = torch.acos(m[..., 0, 0].clamp(-1.0, 1.0))
    th_pos = torch.where(m[..., 1, 0] >= 0, th_pos_pi, 2 * math.pi - th_pos_pi)
    th_neg_pi = torch.acos(m[..., 0, 0].clamp(-1.0, 1.0))
    th_neg = torch.where(-m[..., 1, 0] >= 0, th_neg_pi, 2 * math.pi - th_neg_pi)
    theta_g = torch.where(val_pos, th_pos, th_neg)
    phi_g = torch.where(val_pos, torch.zeros_like(phi), torch.full_like(phi, math.pi))

    psi = torch.where(is_gimbal, psi_g, psi)
    phi = torch.where(is_gimbal, phi_g, phi)
    theta = torch.where(is_gimbal, theta_g, theta)
    return torch.stack([psi, phi, theta], dim=-1)


def _axis_angle_to_orient_mat_torch(axis, angle_deg) -> torch.Tensor:
    """Rodrigues formula. Supports broadcasting over leading dims."""
    dtype, device = _torch_dtype_device(axis, angle_deg)
    u = _to_torch(axis, dtype=dtype, device=device)
    angle = _to_torch(angle_deg, dtype=dtype, device=device) * (math.pi / 180.0)
    norm = torch.linalg.vector_norm(u, dim=-1, keepdim=True).clamp_min(EPS)
    u = u / norm
    c = torch.cos(angle)
    s = torch.sin(angle)
    omc = 1 - c
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    # Build rows; layout matches the NumPy reference exactly.
    row0 = torch.stack(
        [c + ux * ux * omc, -uz * s + ux * uy * omc, uy * s + ux * uz * omc],
        dim=-1,
    )
    row1 = torch.stack(
        [uz * s + uy * ux * omc, c + uy * uy * omc, -ux * s + uy * uz * omc],
        dim=-1,
    )
    row2 = torch.stack(
        [-uy * s + uz * ux * omc, ux * s + uz * uy * omc, c + uz * uz * omc],
        dim=-1,
    )
    return torch.stack([row0, row1, row2], dim=-2)


def _rodrigues_to_orient_mat_torch(rod: torch.Tensor) -> torch.Tensor:
    """Rodrigues vector -> 3x3 R via skew-symmetric matrix exponential.

    Mirrors the NumPy implementation step-for-step (note: the formula uses
    `quat[1:] * th / sin(th/2)` rather than the textbook `axis * angle`;
    this matches the C library reference and we preserve it for parity).
    """
    dtype, device = _torch_dtype_device(rod)
    r = _to_torch(rod, dtype=dtype, device=device)
    norm = torch.linalg.vector_norm(r, dim=-1, keepdim=False)
    c_th_over_2 = torch.cos(torch.atan(norm))
    th = 2.0 * torch.atan(norm)
    spatial = r / c_th_over_2.unsqueeze(-1)
    sin_half = torch.sin(th / 2.0)
    safe = th > EPS
    sin_half_safe = torch.where(safe, sin_half, torch.ones_like(sin_half))
    scale = torch.where(
        safe, th / sin_half_safe, torch.zeros_like(th)
    )
    w = spatial * scale.unsqueeze(-1)
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    zeros = torch.zeros_like(wx)
    skew = torch.stack(
        [
            torch.stack([zeros, -wz, wy], dim=-1),
            torch.stack([wz, zeros, -wx], dim=-1),
            torch.stack([-wy, wx, zeros], dim=-1),
        ],
        dim=-2,
    )
    return torch.linalg.matrix_exp(skew)


# ----------------- symmetry / fundamental zone in torch -----------------


def _make_symmetries_torch(space_group: int, dtype, device) -> torch.Tensor:
    """Return symmetry quaternions as a (n_sym, 4) tensor on (dtype, device)."""
    n_sym, sym = _make_symmetries_py(space_group)
    return torch.tensor(sym, dtype=dtype, device=device)


def _fundamental_zone_torch(quat: torch.Tensor, space_group: int | None, *, sym=None) -> torch.Tensor:
    """Reduce quaternion(s) to the fundamental zone for `space_group`.

    If `sym` is supplied (shape (n_sym, 4)), skip the per-call symmetry-table
    lookup. Falls back to `_make_symmetries_torch(space_group, ...)` otherwise.
    """
    if sym is not None:
        sym_t = _to_torch(sym, dtype=quat.dtype, device=quat.device)
        if sym_t.ndim != 2 or sym_t.shape[1] != 4:
            raise ValueError(f"sym must have shape (n_sym, 4); got {tuple(sym_t.shape)}")
    else:
        if space_group is None:
            raise ValueError("_fundamental_zone_torch requires space_group or sym")
        sym_t = _make_symmetries_torch(space_group, quat.dtype, quat.device)  # (n_sym, 4)
    # Broadcast: q is (..., 4); sym is (n_sym, 4). Result: (..., n_sym, 4).
    q_b = quat.unsqueeze(-2).expand(*quat.shape[:-1], sym_t.shape[0], 4)
    s_b = sym_t.expand(*quat.shape[:-1], sym_t.shape[0], 4)
    qts = _quaternion_product_torch(q_b, s_b)  # (..., n_sym, 4)
    # Pick the sym op with maximum w (qt[..., 0]).
    _, idx = torch.max(qts[..., 0], dim=-1, keepdim=True)
    idx_b = idx.unsqueeze(-1).expand(*idx.shape, 4)
    out = torch.gather(qts, dim=-2, index=idx_b).squeeze(-2)
    norm = torch.linalg.vector_norm(out, dim=-1, keepdim=True).clamp_min(EPS)
    return out / norm


# ----------------- misorientation -----------------


def _misorientation_quat_pair_torch(
    q1: torch.Tensor, q2: torch.Tensor, sym: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (angle in rad, axis (...,3)) for one pair of quats and a sym table."""
    # Reduce both into FZ
    def _fz(q, sym):
        q_b = q.unsqueeze(-2).expand(*q.shape[:-1], sym.shape[0], 4)
        s_b = sym.expand(*q.shape[:-1], sym.shape[0], 4)
        qts = _quaternion_product_torch(q_b, s_b)
        _, idx = torch.max(qts[..., 0], dim=-1, keepdim=True)
        idx_b = idx.unsqueeze(-1).expand(*idx.shape, 4)
        out = torch.gather(qts, dim=-2, index=idx_b).squeeze(-2)
        return out / torch.linalg.vector_norm(out, dim=-1, keepdim=True).clamp_min(EPS)

    q1FR = _fz(q1, sym)
    q2FR = _fz(q2, sym)
    # angle: 2 * acos(w of q1FR_inv * q2FR -> FZ)
    q1_inv = torch.stack([-q1FR[..., 0], q1FR[..., 1], q1FR[..., 2], q1FR[..., 3]], dim=-1)
    QP = _quaternion_product_torch(q1_inv, q2FR)
    MisV = _fz(QP, sym)
    w = MisV[..., 0].clamp(max=1.0)
    angle = 2.0 * torch.acos(w)
    half = angle / 2.0
    s = torch.sin(half)
    s_safe = torch.where(s.abs() > EPS, s, torch.ones_like(s))
    axis = MisV[..., 1:4] / s_safe.unsqueeze(-1)
    axis = torch.where(
        (s.abs() > EPS).unsqueeze(-1), axis, torch.zeros_like(axis)
    )
    return angle, axis


def _misorientation_om_torch(om1, om2, space_group: int):
    dtype, device = _torch_dtype_device(om1, om2)
    om1 = _to_torch(om1, dtype=dtype, device=device)
    om2 = _to_torch(om2, dtype=dtype, device=device)
    q1 = _orient_mat_to_quat_torch(om1)
    q2 = _orient_mat_to_quat_torch(om2)
    sym = _make_symmetries_torch(space_group, dtype, device)
    return _misorientation_quat_pair_torch(q1, q2, sym)


def _misorientation_om_batch_torch(oms1, oms2, space_group: int) -> torch.Tensor:
    angle, _ = _misorientation_om_torch(oms1, oms2, space_group)
    return angle


def _misorientation_quat_batch_torch(quats1, quats2, space_group: int) -> torch.Tensor:
    dtype, device = _torch_dtype_device(quats1, quats2)
    q1 = _to_torch(quats1, dtype=dtype, device=device)
    q2 = _to_torch(quats2, dtype=dtype, device=device)
    sym = _make_symmetries_torch(space_group, dtype, device)
    angle, _ = _misorientation_quat_pair_torch(q1, q2, sym)
    return angle
