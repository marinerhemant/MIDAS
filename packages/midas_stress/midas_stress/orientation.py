"""Orientation representations, conversions, and misorientation computation.

Supports Euler angles, quaternions, orientation matrices, and Rodrigues vectors.
Uses the MIDAS C library (libmidas_orientation) when available for performance,
with pure-Python fallbacks.

All Euler angles in RADIANS.  All misorientation angles returned in RADIANS.
"""

import ctypes
import math
import os
import platform

import numpy as np
from scipy.linalg import expm

EPS = 1e-12
_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


# ===================================================================
#  C library loading
# ===================================================================

_c_double_p = ctypes.POINTER(ctypes.c_double)
_sym_t = ctypes.c_double * 4 * 24
_d3 = ctypes.c_double * 3


def _find_lib():
    """Search for libmidas_orientation in standard locations."""
    ext = '.dylib' if platform.system() == 'Darwin' else '.so'
    here = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        os.path.join(here, '..', '..', '..', '..', 'build', 'lib'),
        os.path.join(here, '..', '..', '..', '..', 'lib'),
        os.path.join(here, '..', '..', '..', 'build', 'lib'),
        os.path.join(here, '..', '..', '..', 'lib'),
    ]
    # Also check MIDAS_LIB_DIR environment variable
    env_dir = os.environ.get('MIDAS_LIB_DIR')
    if env_dir:
        search_dirs.insert(0, env_dir)
    for d in search_dirs:
        p = os.path.join(d, f'libmidas_orientation{ext}')
        if os.path.exists(p):
            return p
    return None


def _setup_lib(lib):
    """Set argtypes/restype for C functions."""
    lib.MakeSymmetries.argtypes = [ctypes.c_int, _sym_t]
    lib.MakeSymmetries.restype = ctypes.c_int

    lib.Euler2OrientMat9.argtypes = [_c_double_p, _c_double_p]
    lib.Euler2OrientMat9.restype = None

    lib.OrientMat2Quat.argtypes = [_c_double_p, _c_double_p]
    lib.OrientMat2Quat.restype = None

    lib.OrientMat2Euler.argtypes = [ctypes.POINTER(_d3), _c_double_p]
    lib.OrientMat2Euler.restype = None

    lib.QuaternionProduct.argtypes = [_c_double_p, _c_double_p, _c_double_p]
    lib.QuaternionProduct.restype = None

    lib.GetMisOrientationAngle.argtypes = [
        _c_double_p, _c_double_p, ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, _sym_t]
    lib.GetMisOrientationAngle.restype = ctypes.c_double

    lib.BringDownToFundamentalRegionSym.argtypes = [
        _c_double_p, _c_double_p, ctypes.c_int, _sym_t]
    lib.BringDownToFundamentalRegionSym.restype = None

    lib.normalizeQuat.argtypes = [_c_double_p]
    lib.normalizeQuat.restype = None

    lib.GetMisOrientationAngleBatch.argtypes = [
        ctypes.c_int, _c_double_p, _c_double_p, _c_double_p, ctypes.c_int, _sym_t]
    lib.GetMisOrientationAngleBatch.restype = None

    lib.GetMisOrientationAngleOMBatch.argtypes = [
        ctypes.c_int, _c_double_p, _c_double_p, _c_double_p, ctypes.c_int]
    lib.GetMisOrientationAngleOMBatch.restype = None

    lib.Euler2OrientMatBatch.argtypes = [ctypes.c_int, _c_double_p, _c_double_p]
    lib.Euler2OrientMatBatch.restype = None

    lib.OrientMat2QuatBatch.argtypes = [ctypes.c_int, _c_double_p, _c_double_p]
    lib.OrientMat2QuatBatch.restype = None


_lib = None
_lib_path = _find_lib()
if _lib_path is not None:
    try:
        _lib = ctypes.CDLL(_lib_path)
        _setup_lib(_lib)
    except OSError:
        _lib = None

_USE_C = _lib is not None


# ===================================================================
#  Internal helpers
# ===================================================================

def _sym_to_c(NrSym, Sym):
    s = _sym_t()
    for i in range(NrSym):
        for j in range(4):
            s[i][j] = Sym[i][j]
    return s


def _sym_from_c(NrSym, c_sym):
    return [[c_sym[i][j] for j in range(4)] for i in range(NrSym)]


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
    if _USE_C:
        c_sym = _sym_t()
        n = _lib.MakeSymmetries(space_group, c_sym)
        return n, _sym_from_c(n, c_sym)
    return _make_symmetries_py(space_group)


def euler_to_orient_mat(euler) -> list:
    """Euler angles (radians) -> flat 9-element orientation matrix.

    Parameters
    ----------
    euler : array-like, shape (3,)
        Euler angles [psi, phi, theta] in radians.

    Returns
    -------
    list of float, length 9
    """
    if _USE_C:
        e = (ctypes.c_double * 3)(*euler)
        m = (ctypes.c_double * 9)()
        _lib.Euler2OrientMat9(e, m)
        return list(m)
    return _euler_to_orient_mat_py(euler)


def orient_mat_to_quat(orient_mat) -> np.ndarray:
    """Flat 9-element orientation matrix -> quaternion [w, x, y, z].

    Parameters
    ----------
    orient_mat : array-like, length 9

    Returns
    -------
    ndarray, shape (4,)
    """
    if _USE_C:
        om = (ctypes.c_double * 9)(*orient_mat)
        q = (ctypes.c_double * 4)()
        _lib.OrientMat2Quat(om, q)
        return np.array([q[0], q[1], q[2], q[3]])
    return _orient_mat_to_quat_py(orient_mat)


def orient_mat_to_euler(m) -> np.ndarray:
    """3x3 or flat-9 orientation matrix -> Euler angles (radians).

    Parameters
    ----------
    m : ndarray, shape (3, 3) or (9,)

    Returns
    -------
    ndarray, shape (3,)
    """
    m = np.asarray(m, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape((3, 3))
    if _USE_C:
        m33 = (_d3 * 3)()
        for i in range(3):
            for j in range(3):
                m33[i][j] = m[i][j]
        e = (ctypes.c_double * 3)()
        _lib.OrientMat2Euler(m33, e)
        return np.array([e[0], e[1], e[2]])
    return _orient_mat_to_euler_py(m)


def quaternion_product(q, r) -> np.ndarray:
    """Hamilton product q * r.  Returns normalized with w >= 0.

    Parameters
    ----------
    q, r : array-like, shape (4,)
        Quaternions [w, x, y, z].

    Returns
    -------
    ndarray, shape (4,)
    """
    if _USE_C:
        cq = (ctypes.c_double * 4)(*q)
        cr = (ctypes.c_double * 4)(*r)
        cQ = (ctypes.c_double * 4)()
        _lib.QuaternionProduct(cq, cr, cQ)
        return np.array([cQ[0], cQ[1], cQ[2], cQ[3]])
    return _quaternion_product_py(q, r)


def quat_to_orient_mat(q) -> list:
    """Quaternion [w, x, y, z] -> flat 9-element orientation matrix.

    Parameters
    ----------
    q : array-like, shape (4,)

    Returns
    -------
    list of float, length 9
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    return [
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ]


def fundamental_zone(quat, space_group: int) -> np.ndarray:
    """Reduce quaternion to fundamental region for a space group.

    Parameters
    ----------
    quat : array-like, shape (4,)
    space_group : int

    Returns
    -------
    ndarray, shape (4,)
    """
    n_sym, sym = make_symmetries(space_group)
    if _USE_C:
        qin = (ctypes.c_double * 4)(*quat)
        qout = (ctypes.c_double * 4)()
        c_sym = _sym_to_c(n_sym, sym)
        _lib.BringDownToFundamentalRegionSym(qin, qout, n_sym, c_sym)
        return np.array([qout[0], qout[1], qout[2], qout[3]])
    return _fundamental_zone_py(quat, n_sym, sym)


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
    """Misorientation between two orientation matrices (flat-9).

    Parameters
    ----------
    om1, om2 : array-like, length 9
    space_group : int

    Returns
    -------
    angle : float
        Misorientation angle in radians.
    axis : ndarray, shape (3,)
    """
    q1 = orient_mat_to_quat(om1)
    q2 = orient_mat_to_quat(om2)
    n_sym, sym = make_symmetries(space_group)

    if _USE_C:
        c_sym = _sym_to_c(n_sym, sym)
        cq1 = (ctypes.c_double * 4)(*q1)
        cq2 = (ctypes.c_double * 4)(*q2)
        angle = ctypes.c_double()
        _lib.GetMisOrientationAngle(cq1, cq2, ctypes.byref(angle), n_sym, c_sym)
        ang = angle.value
    else:
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
    if _USE_C:
        c_sym_obj = _sym_to_c(n_sym, sym)
        qin1 = (ctypes.c_double * 4)(*q1)
        qout1 = (ctypes.c_double * 4)()
        _lib.BringDownToFundamentalRegionSym(qin1, qout1, n_sym, c_sym_obj)
        q1FR = np.array([qout1[0], qout1[1], qout1[2], qout1[3]])
        qin2 = (ctypes.c_double * 4)(*q2)
        qout2 = (ctypes.c_double * 4)()
        _lib.BringDownToFundamentalRegionSym(qin2, qout2, n_sym, c_sym_obj)
        q2FR = np.array([qout2[0], qout2[1], qout2[2], qout2[3]])
    else:
        q1FR = _fundamental_zone_py(list(q1), n_sym, sym)
        q2FR = _fundamental_zone_py(list(q2), n_sym, sym)

    q1FR_inv = np.array([-q1FR[0], q1FR[1], q1FR[2], q1FR[3]])
    QP = quaternion_product(q1FR_inv, q2FR)

    if _USE_C:
        c_sym_obj2 = _sym_to_c(n_sym, sym)
        qpIn = (ctypes.c_double * 4)(*QP)
        qpOut = (ctypes.c_double * 4)()
        _lib.BringDownToFundamentalRegionSym(qpIn, qpOut, n_sym, c_sym_obj2)
        MisV = np.array([qpOut[0], qpOut[1], qpOut[2], qpOut[3]])
    else:
        MisV = _fundamental_zone_py(QP, n_sym, sym)

    return ang, np.array(MisV[1:4]) / s


# ===================================================================
#  Batch functions
# ===================================================================

def misorientation_om_batch(oms1, oms2, space_group: int) -> np.ndarray:
    """Batch misorientation for n pairs of flat-9 orientation matrices.

    Parameters
    ----------
    oms1, oms2 : ndarray (n, 9)
    space_group : int

    Returns
    -------
    ndarray (n,) misorientation angles in radians.
    """
    oms1 = np.ascontiguousarray(oms1, dtype=np.float64)
    oms2 = np.ascontiguousarray(oms2, dtype=np.float64)
    n = oms1.shape[0]
    angles = np.empty(n, dtype=np.float64)
    if _USE_C:
        _lib.GetMisOrientationAngleOMBatch(
            n, oms1.ctypes.data_as(_c_double_p),
            oms2.ctypes.data_as(_c_double_p),
            angles.ctypes.data_as(_c_double_p), space_group)
        return angles
    for i in range(n):
        angles[i], _ = misorientation_om(list(oms1[i]), list(oms2[i]), space_group)
    return angles


def misorientation_quat_batch(quats1, quats2, space_group: int) -> np.ndarray:
    """Batch misorientation for n pairs of quaternions.

    Parameters
    ----------
    quats1, quats2 : ndarray (n, 4)
    space_group : int

    Returns
    -------
    ndarray (n,) misorientation angles in radians.
    """
    quats1 = np.ascontiguousarray(quats1, dtype=np.float64)
    quats2 = np.ascontiguousarray(quats2, dtype=np.float64)
    n = quats1.shape[0]
    angles = np.empty(n, dtype=np.float64)
    if _USE_C:
        n_sym, sym = make_symmetries(space_group)
        c_sym = _sym_to_c(n_sym, sym)
        _lib.GetMisOrientationAngleBatch(
            n, quats1.ctypes.data_as(_c_double_p),
            quats2.ctypes.data_as(_c_double_p),
            angles.ctypes.data_as(_c_double_p), n_sym, c_sym)
        return angles
    for i in range(n):
        om1 = quat_to_orient_mat(list(quats1[i]))
        om2 = quat_to_orient_mat(list(quats2[i]))
        angles[i], _ = misorientation_om(om1, om2, space_group)
    return angles


# ===================================================================
#  Vectorized Euler -> orient mat (pure NumPy, no C needed)
# ===================================================================

def euler_to_orient_mat_batch(euler: np.ndarray) -> np.ndarray:
    """Vectorized Euler angles (radians) -> flat-9 orientation matrices.

    Parameters
    ----------
    euler : ndarray (n, 3)

    Returns
    -------
    ndarray (n, 9)
    """
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

def axis_angle_to_orient_mat(axis, angle_deg: float) -> np.ndarray:
    """Axis-angle to 3x3 orientation matrix (Rodrigues formula).

    Parameters
    ----------
    axis : array-like, shape (3,)
    angle_deg : float
        Rotation angle in degrees.

    Returns
    -------
    ndarray (3, 3)
    """
    u = np.asarray(axis, dtype=np.float64)
    u = u / np.linalg.norm(u)
    angle = angle_deg * _DEG2RAD
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
    rod : array-like, shape (3,)

    Returns
    -------
    ndarray (3, 3)
    """
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
