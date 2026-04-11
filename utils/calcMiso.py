"""
calcMiso.py — Python interface to MIDAS orientation/misorientation math.

Primary implementation lives in GetMisorientation.c (libmidas_orientation).
All core functions call the C library via ctypes.  Pure-Python fallbacks
are used only when the shared library is not found (e.g. before first build).

All Euler angles in RADIANS.  All misorientation angles returned in RADIANS.
"""

from math import sin, cos, acos, sqrt, fabs, atan
import numpy as np
from scipy.linalg import expm
import ctypes
import os
import platform

import math as _math
rad2deg = 180.0 / _math.pi
deg2rad = _math.pi / 180.0
EPS = 0.000000000001

# ══════════════════════════════════════════════════════════════
#  C library loading
# ══════════════════════════════════════════════════════════════

_c_double_p = ctypes.POINTER(ctypes.c_double)
_sym_t = ctypes.c_double * 4 * 24    # double Sym[24][4]
_d3 = ctypes.c_double * 3

def _find_lib():
    ext = '.dylib' if platform.system() == 'Darwin' else '.so'
    here = os.path.dirname(os.path.abspath(__file__))
    for d in ['build/lib', 'lib']:
        p = os.path.join(here, '..', d, f'libmidas_orientation{ext}')
        if os.path.exists(p):
            return p
    return None

def _setup_lib(lib):
    """Set argtypes/restype for every C function we wrap."""

    lib.MakeSymmetries.argtypes = [ctypes.c_int, _sym_t]
    lib.MakeSymmetries.restype  = ctypes.c_int

    lib.Euler2OrientMat9.argtypes = [_c_double_p, _c_double_p]
    lib.Euler2OrientMat9.restype  = None

    lib.OrientMat2Quat.argtypes = [_c_double_p, _c_double_p]
    lib.OrientMat2Quat.restype  = None

    lib.OrientMat2Euler.argtypes = [ctypes.POINTER(ctypes.c_double * 3), _c_double_p]
    lib.OrientMat2Euler.restype  = None

    lib.QuaternionProduct.argtypes = [_c_double_p, _c_double_p, _c_double_p]
    lib.QuaternionProduct.restype  = None

    lib.GetMisOrientationAngle.argtypes = [
        _c_double_p, _c_double_p, ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, _sym_t]
    lib.GetMisOrientationAngle.restype = ctypes.c_double

    lib.GetMisOrientation.argtypes = [
        _c_double_p, _c_double_p, _c_double_p,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    lib.GetMisOrientation.restype = ctypes.c_double

    lib.BringDownToFundamentalRegionSym.argtypes = [
        _c_double_p, _c_double_p, ctypes.c_int, _sym_t]
    lib.BringDownToFundamentalRegionSym.restype = None

    lib.normalizeQuat.argtypes = [_c_double_p]
    lib.normalizeQuat.restype  = None

    # Batch
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

# Try loading at import time
_lib = None
_lib_path = _find_lib()
if _lib_path is not None:
    try:
        _lib = ctypes.CDLL(_lib_path)
        _setup_lib(_lib)
    except OSError:
        _lib = None

_USE_C = _lib is not None

# ══════════════════════════════════════════════════════════════
#  Helper: convert Python Sym list ↔ ctypes Sym[24][4]
# ══════════════════════════════════════════════════════════════

def _sym_to_c(NrSym, Sym):
    s = _sym_t()
    for i in range(NrSym):
        for j in range(4):
            s[i][j] = Sym[i][j]
    return s

def _sym_from_c(NrSym, c_sym):
    return [[c_sym[i][j] for j in range(4)] for i in range(NrSym)]

# ══════════════════════════════════════════════════════════════
#  Core functions — C-backed with pure-Python fallback
# ══════════════════════════════════════════════════════════════

def normalize(quat):
    return quat / np.linalg.norm(quat)

def MakeSymmetries(SGNr):
    """Returns (NrSymmetries, Sym) where Sym is a list of [w,x,y,z] quaternions."""
    if _USE_C:
        c_sym = _sym_t()
        n = _lib.MakeSymmetries(SGNr, c_sym)
        return n, _sym_from_c(n, c_sym)
    return _MakeSymmetries_py(SGNr)

def Euler2OrientMat(Euler):
    """Euler angles (radians) → flat 9-element orientation matrix (list)."""
    if _USE_C:
        e = (ctypes.c_double * 3)(*Euler)
        m = (ctypes.c_double * 9)()
        _lib.Euler2OrientMat9(e, m)
        return list(m)
    return _Euler2OrientMat_py(Euler)

def OrientMat2Quat(OrientMat):
    """Flat 9-element orientation matrix → quaternion [w,x,y,z] (numpy array)."""
    if _USE_C:
        om = (ctypes.c_double * 9)(*OrientMat)
        q = (ctypes.c_double * 4)()
        _lib.OrientMat2Quat(om, q)
        return np.array([q[0], q[1], q[2], q[3]])
    return _OrientMat2Quat_py(OrientMat)

def OrientMat2Euler(m):
    """3x3 or flat-9 orientation matrix → Euler angles (radians, numpy array)."""
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
    return _OrientMat2Euler_py(m)

def QuaternionProduct(q, r):
    """Hamilton product Q = q * r.  Returns normalized numpy array with Q[0] >= 0."""
    if _USE_C:
        cq = (ctypes.c_double * 4)(*q)
        cr = (ctypes.c_double * 4)(*r)
        cQ = (ctypes.c_double * 4)()
        _lib.QuaternionProduct(cq, cr, cQ)
        return np.array([cQ[0], cQ[1], cQ[2], cQ[3]])
    return _QuaternionProduct_py(q, r)

def BringDownToFundamentalRegionSym(QuatIn, NrSymmetries, Sym):
    """Reduce quaternion to fundamental region.  Returns numpy array."""
    if _USE_C:
        qin = (ctypes.c_double * 4)(*QuatIn)
        qout = (ctypes.c_double * 4)()
        c_sym = _sym_to_c(NrSymmetries, Sym)
        _lib.BringDownToFundamentalRegionSym(qin, qout, NrSymmetries, c_sym)
        return np.array([qout[0], qout[1], qout[2], qout[3]])
    return _BringDownToFundamentalRegionSym_py(QuatIn, NrSymmetries, Sym)

def GetMisOrientationAngle(euler1, euler2, SGNum):
    """Misorientation between two Euler-angle orientations (radians).
    Returns (angle_rad, axis)."""
    om1 = Euler2OrientMat(euler1)
    om2 = Euler2OrientMat(euler2)
    return GetMisOrientationAngleOM(om1, om2, SGNum)

def GetMisOrientationAngleOM(OM1, OM2, SGNum):
    """Misorientation between two flat-9 orientation matrices.
    Returns (angle_rad, axis)."""
    q1 = OrientMat2Quat(OM1)
    q2 = OrientMat2Quat(OM2)
    NrSym, Sym = MakeSymmetries(SGNum)
    if _USE_C:
        c_sym = _sym_to_c(NrSym, Sym)
        cq1 = (ctypes.c_double * 4)(*q1)
        cq2 = (ctypes.c_double * 4)(*q2)
        angle = ctypes.c_double()
        _lib.GetMisOrientationAngle(cq1, cq2, ctypes.byref(angle), NrSym, c_sym)
        ang = angle.value
    else:
        q1FR = _BringDownToFundamentalRegionSym_py(list(q1), NrSym, Sym)
        q2FR = _BringDownToFundamentalRegionSym_py(list(q2), NrSym, Sym)
        q1FR[0] = -q1FR[0]
        QP = _QuaternionProduct_py(q1FR, q2FR)
        MisV = _BringDownToFundamentalRegionSym_py(QP, NrSym, Sym)
        w = min(1.0, float(MisV[0]))
        ang = 2.0 * acos(w)
    # Compute axis from angle
    half = ang / 2.0
    s = sin(half) if ang > EPS else 0.0
    if fabs(s) < EPS:
        return ang, np.array([0.0, 0.0, 0.0])
    # Re-derive MisV quaternion to get axis
    q1FR = BringDownToFundamentalRegionSym(q1, NrSym, Sym)
    q2FR = BringDownToFundamentalRegionSym(q2, NrSym, Sym)
    q1FR_inv = np.array([-q1FR[0], q1FR[1], q1FR[2], q1FR[3]])
    QP = QuaternionProduct(q1FR_inv, q2FR)
    MisV = BringDownToFundamentalRegionSym(QP, NrSym, Sym)
    return ang, np.array(MisV[1:4]) / s

def Quat2OrientMat(q):
    """Convert quaternion (w, x, y, z) to a flat 9-element orientation matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    om = [0]*9
    om[0] = 1 - 2*(y*y + z*z)
    om[1] = 2*(x*y - w*z)
    om[2] = 2*(x*z + w*y)
    om[3] = 2*(x*y + w*z)
    om[4] = 1 - 2*(x*x + z*z)
    om[5] = 2*(y*z - w*x)
    om[6] = 2*(x*z - w*y)
    om[7] = 2*(y*z + w*x)
    om[8] = 1 - 2*(x*x + y*y)
    return om

# ══════════════════════════════════════════════════════════════
#  Batch functions (OpenMP via C library, Python-loop fallback)
# ══════════════════════════════════════════════════════════════

def GetMisOrientationAngleOMBatch(OMs1, OMs2, SGNum):
    """Batch misorientation for n pairs of flat-9 OMs.  Returns (n,) radians."""
    OMs1 = np.ascontiguousarray(OMs1, dtype=np.float64)
    OMs2 = np.ascontiguousarray(OMs2, dtype=np.float64)
    n = OMs1.shape[0]
    angles = np.empty(n, dtype=np.float64)
    if _USE_C:
        _lib.GetMisOrientationAngleOMBatch(
            n, OMs1.ctypes.data_as(_c_double_p),
            OMs2.ctypes.data_as(_c_double_p),
            angles.ctypes.data_as(_c_double_p), SGNum)
        return angles
    for i in range(n):
        angles[i], _ = GetMisOrientationAngleOM(list(OMs1[i]), list(OMs2[i]), SGNum)
    return angles

def GetMisOrientationAngleBatch(quats1, quats2, SGNum):
    """Batch misorientation for n pairs of quaternions.  Returns (n,) radians."""
    quats1 = np.ascontiguousarray(quats1, dtype=np.float64)
    quats2 = np.ascontiguousarray(quats2, dtype=np.float64)
    n = quats1.shape[0]
    angles = np.empty(n, dtype=np.float64)
    if _USE_C:
        NrSym, Sym = MakeSymmetries(SGNum)
        c_sym = _sym_to_c(NrSym, Sym)
        _lib.GetMisOrientationAngleBatch(
            n, quats1.ctypes.data_as(_c_double_p),
            quats2.ctypes.data_as(_c_double_p),
            angles.ctypes.data_as(_c_double_p), NrSym, c_sym)
        return angles
    for i in range(n):
        q1, q2 = list(quats1[i]), list(quats2[i])
        om1, om2 = Quat2OrientMat(q1), Quat2OrientMat(q2)
        angles[i], _ = GetMisOrientationAngleOM(om1, om2, SGNum)
    return angles

# ══════════════════════════════════════════════════════════════
#  Pure-Python utility functions (no C equivalent needed)
# ══════════════════════════════════════════════════════════════

def AxisAngleToOrientMat(axis, angle):
    R = np.zeros((3,3))
    lenInv = 1 / np.linalg.norm(axis)
    u = axis[0] * lenInv
    v = axis[1] * lenInv
    w = axis[2] * lenInv
    angleRad = deg2rad * angle
    rcos = cos(angleRad)
    rsin = sin(angleRad)
    R[0][0] =      rcos + u*u*(1-rcos)
    R[1][0] =  w * rsin + v*u*(1-rcos)
    R[2][0] = -v * rsin + w*u*(1-rcos)
    R[0][1] = -w * rsin + u*v*(1-rcos)
    R[1][1] =      rcos + v*v*(1-rcos)
    R[2][1] =  u * rsin + w*v*(1-rcos)
    R[0][2] =  v * rsin + u*w*(1-rcos)
    R[1][2] = -u * rsin + v*w*(1-rcos)
    R[2][2] =      rcos + w*w*(1-rcos)
    return R

def MatrixMultF33(m, n):
    res = np.zeros((3,3))
    for r in range(3):
        res[r][0] = m[r][0]*n[0][0] + m[r][1]*n[1][0] + m[r][2]*n[2][0]
        res[r][1] = m[r][0]*n[0][1] + m[r][1]*n[1][1] + m[r][2]*n[2][1]
        res[r][2] = m[r][0]*n[0][2] + m[r][1]*n[1][2] + m[r][2]*n[2][2]
    return res

def eul2omMat(euler):
    """Vectorized Euler (radians) → flat-9 OMs.  Pure numpy, no C needed."""
    m_out = np.zeros((euler.shape[0], 9))
    cps = np.cos(euler[:,0]); cph = np.cos(euler[:,1]); cth = np.cos(euler[:,2])
    sps = np.sin(euler[:,0]); sph = np.sin(euler[:,1]); sth = np.sin(euler[:,2])
    m_out[:,0] = cth * cps - sth * cph * sps
    m_out[:,1] = -cth * cph * sps - sth * cps
    m_out[:,2] = sph * sps
    m_out[:,3] = cth * sps + sth * cph * cps
    m_out[:,4] = cth * cph * cps - sth * sps
    m_out[:,5] = -sph * cps
    m_out[:,6] = sth * sph
    m_out[:,7] = cth * sph
    m_out[:,8] = cph
    return m_out

def CalcEtaAngleAll(y, z):
    alpha = rad2deg * np.arccos(z / np.linalg.norm(np.array([y,z]), axis=0))
    alpha[y > 0] *= -1
    return alpha

def rod2om(rod):
    cThOver2 = cos(atan(np.linalg.norm(rod)))
    th = 2 * atan(np.linalg.norm(rod))
    quat = np.array([cThOver2, rod[0]/cThOver2, rod[1]/cThOver2, rod[2]/cThOver2])
    if th > EPS:
        w = quat[1:] * th / sin(th/2)
    else:
        w = np.array([0,0,0])
    wskew = np.array([[    0, -w[2],  w[1]],
                      [ w[2],     0, -w[0]],
                      [-w[1],  w[0],     0]])
    OM = expm(wskew)
    return OM


# ══════════════════════════════════════════════════════════════
#  Pure-Python fallback implementations
#  (used only when libmidas_orientation is not available)
# ══════════════════════════════════════════════════════════════

# Symmetry tables (fallback only — canonical source is GetMisorientation.c)
_TricSym=[[1,0,0,0]]
_MonoSym=[[1,0,0,0],[0,0,1,0]]
_OrtSym=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
_TetSym=[[1,0,0,0],[0.70711,0,0,0.70711],[0,0,0,1],[0.70711,0,0,-0.70711],
         [0,1,0,0],[0,0,1,0],[0,0.70711,0.70711,0],[0,-0.70711,0.70711,0]]
_TetSymLow=[[1,0,0,0],[0.70711,0,0,0.70711],[0,0,0,1],[0.70711,0,0,-0.70711]]
_TrigSym=[[1,0,0,0],[0,0.86603,-0.5,0],[0.5,0,0,0.86603],
          [0,0,1,0],[0.5,0,0,-0.86603],[0,0.86603,0.5,0]]
_TrigSym2=[[1,0,0,0],[0.5,0,0,0.86603],[0.5,0,0,-0.86603],
           [0,0.5,-0.86603,0],[0,1,0,0],[0,0.5,0.86603,0]]
_TrigSymLow=[[1,0,0,0],[0.5,0,0,0.86603],[0.5,0,0,-0.86603]]
_HexSym=[[1,0,0,0],[0.86603,0,0,0.5],[0.5,0,0,0.86603],[0,0,0,1],
         [0.5,0,0,-0.86603],[0.86603,0,0,-0.5],[0,1,0,0],[0,0.86603,0.5,0],
         [0,0.5,0.86603,0],[0,0,1,0],[0,-0.5,0.86603,0],[0,-0.86603,0.5,0]]
_HexSymLow=[[1,0,0,0],[0.86603,0,0,0.5],[0.5,0,0,0.86603],
            [0,0,0,1],[0.5,0,0,-0.86603],[0.86603,0,0,-0.5]]
_CubSym=[[1,0,0,0],[0.70711,0.70711,0,0],[0,1,0,0],[0.70711,-0.70711,0,0],
         [0.70711,0,0.70711,0],[0,0,1,0],[0.70711,0,-0.70711,0],
         [0.70711,0,0,0.70711],[0,0,0,1],[0.70711,0,0,-0.70711],
         [0.5,0.5,0.5,0.5],[0.5,-0.5,-0.5,-0.5],[0.5,-0.5,0.5,0.5],
         [0.5,0.5,-0.5,-0.5],[0.5,0.5,-0.5,0.5],[0.5,-0.5,0.5,-0.5],
         [0.5,-0.5,-0.5,0.5],[0.5,0.5,0.5,-0.5],[0,0.70711,0.70711,0],
         [0,-0.70711,0.70711,0],[0,0.70711,0,0.70711],[0,0.70711,0,-0.70711],
         [0,0,0.70711,0.70711],[0,0,0.70711,-0.70711]]
_CubSymLow=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],
            [0.5,0.5,0.5,0.5],[0.5,-0.5,-0.5,-0.5],[0.5,-0.5,0.5,0.5],
            [0.5,0.5,-0.5,-0.5],[0.5,0.5,-0.5,0.5],[0.5,-0.5,0.5,-0.5],
            [0.5,-0.5,-0.5,0.5],[0.5,0.5,0.5,-0.5]]

_TrigType2SGs = {149, 151, 153, 157, 159, 162, 163}

def _MakeSymmetries_py(SGNr):
    if SGNr<=2:    return 1, _TricSym
    elif SGNr<=15: return 2, _MonoSym
    elif SGNr<=74: return 4, _OrtSym
    elif SGNr<=88: return 4, _TetSymLow
    elif SGNr<=142: return 8, _TetSym
    elif SGNr<=148: return 3, _TrigSymLow
    elif SGNr<=167: return 6, (_TrigSym2 if SGNr in _TrigType2SGs else _TrigSym)
    elif SGNr<=176: return 6, _HexSymLow
    elif SGNr<=194: return 12, _HexSym
    elif SGNr<=206: return 12, _CubSymLow
    elif SGNr<=230: return 24, _CubSym
    else: return 0, []

def _QuaternionProduct_py(q, r):
    Q = [0,0,0,0]
    Q[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3]
    Q[1] = r[1]*q[0] + r[0]*q[1] + r[3]*q[2] - r[2]*q[3]
    Q[2] = r[2]*q[0] + r[0]*q[2] + r[1]*q[3] - r[3]*q[1]
    Q[3] = r[3]*q[0] + r[0]*q[3] + r[2]*q[1] - r[1]*q[2]
    if Q[0] < 0:
        Q[0] = -Q[0]; Q[1] = -Q[1]; Q[2] = -Q[2]; Q[3] = -Q[3]
    return normalize(Q)

def _BringDownToFundamentalRegionSym_py(QuatIn, NrSymmetries, Sym):
    maxCos = -10000.0
    QuatOut = QuatIn
    for i in range(NrSymmetries):
        qt = _QuaternionProduct_py(QuatIn, Sym[i])
        if maxCos < qt[0]:
            maxCos = qt[0]
            QuatOut = qt
    return normalize(QuatOut)

def _OrientMat2Quat_py(OrientMat):
    Quat = [0,0,0,0]
    trace = OrientMat[0] + OrientMat[4] + OrientMat[8]
    if trace > 0:
        s = 0.5/sqrt(trace+1.0)
        Quat[0] = 0.25/s
        Quat[1] = (OrientMat[7]-OrientMat[5])*s
        Quat[2] = (OrientMat[2]-OrientMat[6])*s
        Quat[3] = (OrientMat[3]-OrientMat[1])*s
    else:
        if OrientMat[0]>OrientMat[4] and OrientMat[0]>OrientMat[8]:
            s = 2.0*sqrt(1.0+OrientMat[0]-OrientMat[4]-OrientMat[8])
            Quat[0] = (OrientMat[7]-OrientMat[5])/s
            Quat[1] = 0.25*s
            Quat[2] = (OrientMat[1]+OrientMat[3])/s
            Quat[3] = (OrientMat[2]+OrientMat[6])/s
        elif OrientMat[4] > OrientMat[8]:
            s = 2.0*sqrt(1.0+OrientMat[4]-OrientMat[0]-OrientMat[8])
            Quat[0] = (OrientMat[2]-OrientMat[6])/s
            Quat[1] = (OrientMat[1]+OrientMat[3])/s
            Quat[2] = 0.25*s
            Quat[3] = (OrientMat[5]+OrientMat[7])/s
        else:
            s = 2.0*sqrt(1.0+OrientMat[8]-OrientMat[0]-OrientMat[4])
            Quat[0] = (OrientMat[3]-OrientMat[1])/s
            Quat[1] = (OrientMat[2]+OrientMat[6])/s
            Quat[2] = (OrientMat[5]+OrientMat[7])/s
            Quat[3] = 0.25*s
    if Quat[0] < 0:
        Quat[0] = -Quat[0]; Quat[1] = -Quat[1]
        Quat[2] = -Quat[2]; Quat[3] = -Quat[3]
    return normalize(Quat)

def _Euler2OrientMat_py(Euler):
    psi, phi, theta = Euler[0], Euler[1], Euler[2]
    cps = cos(psi); cph = cos(phi); cth = cos(theta)
    sps = sin(psi); sph = sin(phi); sth = sin(theta)
    return [cth*cps - sth*cph*sps, -cth*cph*sps - sth*cps, sph*sps,
            cth*sps + sth*cph*cps,  cth*cph*cps - sth*sps, -sph*cps,
            sth*sph, cth*sph, cph]

def _sin_cos_to_angle_py(s, c):
    c = max(-1.0, min(1.0, c))
    return acos(c) if s >= 0 else 2.0*np.pi - acos(c)

def _OrientMat2Euler_py(m):
    if m.ndim == 1:
        m = m.reshape((3,3))
    if m[2][2] > 1: m[2][2] = 1
    phi = 0 if fabs(m[2][2]-1.0) < EPS else acos(max(-1.0, min(1.0, m[2][2])))
    sph = sin(phi)
    if fabs(sph) < EPS:
        psi = 0.0
        theta = (_sin_cos_to_angle_py(m[1][0], m[0][0]) if fabs(m[2][2]-1.0)<EPS
                 else _sin_cos_to_angle_py(-m[1][0], m[0][0]))
    else:
        psi = (_sin_cos_to_angle_py(m[0][2]/sph, -m[1][2]/sph) if fabs(-m[1][2]/sph)<=1.0
               else _sin_cos_to_angle_py(m[0][2]/sph, 1))
        theta = (_sin_cos_to_angle_py(m[2][0]/sph, m[2][1]/sph) if fabs(m[2][1]/sph)<=1.0
                 else _sin_cos_to_angle_py(m[2][0]/sph, 1))
    return np.array([psi, phi, theta])
