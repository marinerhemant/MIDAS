"""pyFAI ↔ midas-integrate-v2 coordinate conversion helpers.

Two independent conventions have to be got right when moving a beam centre
between pyFAI and MIDAS. Both fail silently.

1. THE 0.5 px CORNER/CENTRE SHIFT
---------------------------------
- pyFAI uses **pixel-corner** indexing — pixel ``(0, 0)`` is the *corner* of
  the first pixel; the centre of pixel ``(i, j)`` is at ``(i + 0.5, j + 0.5)``.
- MIDAS uses **pixel-centre** indexing — pixel ``(0, 0)`` IS the centre of the
  first pixel.

So the same physical beam-impact point is recorded as different numbers::

    poni = (BC + 0.5) * pixel_size

Skipping the ``+ 0.5`` gives a calibration off by half a pixel — small but
systematic, big enough to shift Bragg peaks at high R and to break apparent
vs. true d-spacings at the per-mille level.

2. THE DETECTOR ORIENTATION
---------------------------
pyFAI's ``Detector_config`` may carry an ``orientation`` flag saying where
array element ``[0, 0]`` sits in the laboratory frame. It controls whether
pyFAI's ``poni1``/``poni2`` axes run *with* or *against* the array's row and
column indices, so it changes which pixel a given PONI names:

    ==  ============  =========  =========
    id  name          row flip   col flip
    ==  ============  =========  =========
     0  Unspecified   no         no
     1  TopLeft       YES        YES
     2  TopRight      YES        no
     3  BottomRight   no         no
     4  BottomLeft    no         YES
    ==  ============  =========  =========

Measured against pyFAI 2026.2.1 itself (``Detector.calc_cartesian_positions``)
rather than taken from documentation; pinned by
``tests/test_poni_orientation.py``, which re-derives the table from the
installed pyFAI and fails if a future release changes it.

Ignoring a flip puts the beam centre ``(N - 1 - 2*BC)`` pixels wrong — hundreds
of pixels, in the wrong half of the detector, with no error raised.

.. warning::

   **Orientation 3 is pyFAI's DEFAULT, and is geometrically identical to 0.**
   A PONI file that declares ``"orientation": 3`` is therefore telling you
   nothing: it is the no-flip case. It is *not* evidence that a flip is needed,
   and it cannot explain one that is.

   A row flip can still be present without the PONI saying so, because the
   flag describes the array *the calibration was performed on*. If the reader
   used at calibration time (``fabio``, a beamline loader, an ImageJ export)
   returns a different row order than the reader used downstream
   (``tifffile``), the centre lands ``N_rows - 1 - row`` off and **no metadata
   records it**. Confirm the beam centre against the data — Friedel pairs or
   ring concentricity — whenever a PONI comes from outside your own pipeline.

3. A PONI IS NOT THE BEAM CENTRE ONCE THE DETECTOR IS TILTED
------------------------------------------------------------
The PONI is the foot of the perpendicular from the sample to the detector
plane. The direct beam meets that plane elsewhere, ``dist * tan(rot)`` away
(about one pixel at one degree for a typical geometry), and ``dist`` is the
perpendicular distance, not MIDAS's ``Lsd`` (the distance along the beam). So
``poni = (BC + 0.5) * pixel_size`` holds for an untilted detector ONLY.

For a tilted one use :func:`midas_to_poni` / :func:`poni_to_midas` /
:func:`poni_file_to_midas`. They take the whole geometry (``Lsd``, ``BC``,
``tx, ty, tz`` against ``dist``, ``poni1, poni2``, ``rot1, rot2, rot3``) and
are pinned per pixel against pyFAI's own 2theta by ``tests/test_poni_tilt.py``.
With ``rot3 = tx = 0`` they reduce to ``Lsd = dist / (cos rot1 cos rot2)``,
``tz = -rot1``, ``ty = +rot2`` and BC = the pixel where pyFAI's 2theta is zero.

4. WHICH BEAM-CENTRE COORDINATE IS THE ROW
------------------------------------------
Images are ``[row, column] = [Z, Y]`` (``NrPixelsY`` is the number of
columns). pyFAI's axis 1 is the row and axis 2 the column, so ``BC_z`` and
``pxZ`` go with ``poni1`` / ``pixel1`` and ``BC_y`` and ``pxY`` with ``poni2`` /
``pixel2``. The primitives :func:`bc_to_poni` / :func:`poni_to_bc` keep the
pyFAI naming (their first coordinate is the ROW); the spec-level functions
(:func:`make_pyfai_integrator`, :func:`midas_to_poni`, :func:`poni_to_midas`,
:func:`poni_file_to_midas`) use MIDAS names. Before 2026-10-14
``make_pyfai_integrator`` passed ``BC_y`` as the row, which transposes any beam
centre with ``BC_y != BC_z`` (60 px-equivalent in ``tests/test_poni_tilt.py``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Orientation id -> (flip along axis 1 / rows, flip along axis 2 / columns).
# Verified against pyFAI 2026.2.1; see test_poni_orientation.py.
_ORIENTATION_FLIPS: Dict[int, Tuple[bool, bool]] = {
    0: (False, False),   # Unspecified — pyFAI treats as BottomRight
    1: (True, True),     # TopLeft
    2: (True, False),    # TopRight
    3: (False, False),   # BottomRight — pyFAI's DEFAULT
    4: (False, True),    # BottomLeft
}

_ORIENTATION_NAMES: Dict[int, str] = {
    0: "Unspecified", 1: "TopLeft", 2: "TopRight",
    3: "BottomRight", 4: "BottomLeft",
}

#: Orientations that imply no flip, so MIDAS's naive conversion is already right.
NO_FLIP_ORIENTATIONS = frozenset({0, 3})


def orientation_flips(orientation: Optional[int]) -> Tuple[bool, bool]:
    """Return ``(flip_rows, flip_cols)`` for a pyFAI orientation id.

    ``None`` is treated as unspecified (no flip), matching pyFAI's default.
    Raises ``NotImplementedError`` for the axis-swapping orientations 5-8
    (Transpose / Rotate270 / Transverse / Rotate90) rather than returning a
    wrong answer for them.
    """
    if orientation is None:
        return (False, False)
    o = int(orientation)
    if o in _ORIENTATION_FLIPS:
        return _ORIENTATION_FLIPS[o]
    if 5 <= o <= 8:
        raise NotImplementedError(
            f"pyFAI orientation {o} transposes the detector axes, which "
            "changes the array shape as well as the indexing. Convert the "
            "image and the PONI to a non-transposing orientation first."
        )
    raise ValueError(f"unknown pyFAI detector orientation {orientation!r}")


def _axis_shape(shape: Optional[Tuple[int, int]], flip_r: bool, flip_c: bool
                ) -> Tuple[Optional[int], Optional[int]]:
    if not (flip_r or flip_c):
        return (None, None)
    if shape is None:
        raise ValueError(
            "this detector orientation flips an axis, so the array shape is "
            "needed to convert; pass shape=(n_rows, n_cols)"
        )
    n1, n2 = int(shape[0]), int(shape[1])
    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"shape must be positive, got {shape!r}")
    return (n1, n2)


def bc_to_poni(
    BC_y_px: float, BC_z_px: float,
    pxY_um: float, pxZ_um: float,
    *,
    orientation: Optional[int] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float]:
    """Convert MIDAS BC (pixel-centre, px) to pyFAI PONI (pixel-corner, m).

    ``BC_y_px`` indexes rows (pyFAI axis 1), ``BC_z_px`` columns (axis 2), as
    the PONI header's "1 refers to the Y axis, 2 to the X axis" states.

    Pass ``orientation`` (and ``shape=(n_rows, n_cols)``) when the target PONI
    declares a flipping detector orientation. Omitting both reproduces the
    historical no-flip behaviour exactly.

    Returns ``(poni1_m, poni2_m)`` in metres.
    """
    flip_r, flip_c = orientation_flips(orientation)
    n1, n2 = _axis_shape(shape, flip_r, flip_c)
    y = (n1 - 1) - BC_y_px if flip_r else BC_y_px
    z = (n2 - 1) - BC_z_px if flip_c else BC_z_px
    return ((y + 0.5) * pxY_um * 1e-6,
            (z + 0.5) * pxZ_um * 1e-6)


def poni_to_bc(
    poni1_m: float, poni2_m: float,
    pxY_um: float, pxZ_um: float,
    *,
    orientation: Optional[int] = None,
    shape: Optional[Tuple[int, int]] = None,
) -> Tuple[float, float]:
    """Convert pyFAI PONI (pixel-corner, m) to MIDAS BC (pixel-centre, px).

    The inverse of :func:`bc_to_poni`; see it for ``orientation``/``shape``.

    Returns ``(BC_y_px, BC_z_px)`` in pixels, indexing the array as it is
    stored — rows then columns.

    .. warning::

       **The names here are pyFAI's, and they are the opposite way round from
       the rest of MIDAS.** In this module ``BC_y`` means ROWS, following the
       PONI header's "1 refers to the Y axis". In
       :class:`midas_defect.geometry.Geometry`, ``bcy_px`` means COLUMNS and
       ``bcz_px`` means rows. Assigning this function's output straight into
       ``Geometry(bcy_px=..., bcz_px=...)`` therefore **transposes the beam
       centre**, which on real data shifted it by ~120 px and made indexing
       fail outright (2026-09-03). Use :func:`poni_file_to_row_col` instead, or
       unpack as ``row, col = poni_file_to_bc(...)`` and pass
       ``bcz_px=row, bcy_px=col``.
    """
    flip_r, flip_c = orientation_flips(orientation)
    n1, n2 = _axis_shape(shape, flip_r, flip_c)
    y = poni1_m / (pxY_um * 1e-6) - 0.5
    z = poni2_m / (pxZ_um * 1e-6) - 0.5
    if flip_r:
        y = (n1 - 1) - y
    if flip_c:
        z = (n2 - 1) - z
    return y, z


def read_poni(path: Union[str, Path]) -> Dict[str, Any]:
    """Parse a pyFAI ``.poni`` file into a plain dict.

    Numeric entries (``Distance``, ``Poni1``, ``Poni2``, ``Rot1``-``Rot3``,
    ``Wavelength``) come back as floats; ``Detector_config`` is JSON-decoded
    into a dict. Two convenience keys are added when the config supplies them:

    - ``orientation`` — the raw pyFAI orientation id, or ``None`` if absent
    - ``shape``       — ``(n_rows, n_cols)`` from ``max_shape``, or ``None``

    Comment lines are ignored. Unknown keys are kept as strings rather than
    dropped, so nothing in the file is silently lost.
    """
    path = Path(path)
    out: Dict[str, Any] = {}
    numeric = {"distance", "poni1", "poni2", "rot1", "rot2", "rot3",
               "wavelength", "pixel1", "pixel2"}
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key == "Detector_config":
                try:
                    out[key] = json.loads(value)
                except json.JSONDecodeError:
                    out[key] = value
            elif key.lower() in numeric:
                try:
                    out[key] = float(value)
                except ValueError:
                    out[key] = value
            else:
                out[key] = value

    cfg = out.get("Detector_config")
    orientation = None
    shape = None
    if isinstance(cfg, dict):
        if cfg.get("orientation") is not None:
            orientation = int(cfg["orientation"])
        ms = cfg.get("max_shape")
        if isinstance(ms, (list, tuple)) and len(ms) == 2:
            shape = (int(ms[0]), int(ms[1]))
    out["orientation"] = orientation
    out["shape"] = shape
    return out


def poni_file_to_bc(path: Union[str, Path]) -> Tuple[float, float]:
    """Read a ``.poni`` file and return its PONI point as MIDAS ``(BC_y, BC_z)``.

    Applies the file's own detector orientation and pixel sizes. Note this is
    the **point of normal incidence**, which equals the beam centre only for an
    untilted detector; with non-zero ``Rot1``/``Rot2`` the direct beam lands
    elsewhere and pyFAI's ``getFit2D()`` is the quantity you want.

    Reading the file does not tell you whether your image reader agrees with
    the calibration's row order — see the module docstring.
    """
    p = read_poni(path)
    cfg = p.get("Detector_config")
    if not isinstance(cfg, dict) or "pixel1" not in cfg or "pixel2" not in cfg:
        raise ValueError(f"{path}: Detector_config lacks pixel1/pixel2")
    return poni_to_bc(
        float(p["Poni1"]), float(p["Poni2"]),
        float(cfg["pixel1"]) * 1e6, float(cfg["pixel2"]) * 1e6,
        orientation=p["orientation"], shape=p["shape"],
    )


def poni_file_to_row_col(path: Union[str, Path]) -> Tuple[float, float]:
    """Read a ``.poni`` and return its PONI point as ``(row_px, col_px)``.

    Same computation as :func:`poni_file_to_bc`, named so it cannot be
    mis-assigned. Prefer this at any boundary with
    :class:`midas_defect.geometry.Geometry`::

        row, col = poni_file_to_row_col(poni)
        Geometry(bcz_px=row, bcy_px=col, ...)   # note: bcz is the ROW

    Remember this is the point of normal incidence, which is the beam centre
    only for an untilted detector, and that the file cannot tell you whether
    your image reader agrees with the calibration's row order.
    """
    return poni_file_to_bc(path)


def describe_orientation(orientation: Optional[int]) -> str:
    """One-line human description, for logs and error messages."""
    if orientation is None:
        return "unspecified (no flip; pyFAI default behaviour)"
    o = int(orientation)
    name = _ORIENTATION_NAMES.get(o, f"id {o}")
    if o in NO_FLIP_ORIENTATIONS:
        extra = "no flip — this is pyFAI's default, so it implies nothing"
    else:
        fr, fc = orientation_flips(o)
        parts = [n for n, f in (("rows", fr), ("columns", fc)) if f]
        extra = "flips " + " and ".join(parts)
    return f"{name} ({o}): {extra}"


# MIDAS lab axes (x = beam, y, z) -> pyFAI axes (1, 2, 3):  x1 = +z, x2 = -y, x3 = x.  The sign choice is the only one of the
# four (with R versus R^T) that reproduces pyFAI's per-pixel 2theta; it is pinned by tests/test_poni_tilt.py.
_AXIS_MAP = np.array([[0.0, 0.0, 1.0],
                      [0.0, -1.0, 0.0],
                      [1.0, 0.0, 0.0]])


def _midas_tilt_matrix(tx_deg: float, ty_deg: float, tz_deg: float) -> np.ndarray:
    """MIDAS ``TRs = Rx(tx) . Ry(ty) . Rz(tz)`` (degrees), as in ``midas_integrate.geometry.build_tilt_matrix``."""
    tx, ty, tz = (math.radians(float(a)) for a in (tx_deg, ty_deg, tz_deg))
    cx, sx, cy, sy, cz, sz = math.cos(tx), math.sin(tx), math.cos(ty), math.sin(ty), math.cos(tz), math.sin(tz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rx @ (Ry @ Rz)


def _pyfai_rotation(rot1: float, rot2: float, rot3: float) -> np.ndarray:
    """pyFAI's ``Geometry.rotation_matrix`` = rot3 . rot2 . rot1 (radians); rot1 and rot2 are left-handed there."""
    c1, s1, c2, s2, c3, s3 = (math.cos(rot1), math.sin(rot1), math.cos(rot2), math.sin(rot2), math.cos(rot3), math.sin(rot3))
    r1 = np.array([[1.0, 0.0, 0.0], [0.0, c1, s1], [0.0, -s1, c1]])
    r2 = np.array([[c2, 0.0, -s2], [0.0, 1.0, 0.0], [s2, 0.0, c2]])
    r3 = np.array([[c3, -s3, 0.0], [s3, c3, 0.0], [0.0, 0.0, 1.0]])
    return r3 @ r2 @ r1


def _pyfai_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """Inverse of :func:`_pyfai_rotation` (|rot2| < 90 deg)."""
    return (math.atan2(-R[2, 1], R[2, 2]), math.asin(max(-1.0, min(1.0, R[2, 0]))), math.atan2(R[1, 0], R[0, 0]))


def midas_to_poni(Lsd_um: float, BC_y: float, BC_z: float, tx: float, ty: float, tz: float,
                  pxY_um: float, pxZ_um: float) -> Dict[str, float]:
    """MIDAS geometry (``Lsd`` in um, ``BC`` in pixels, tilts in degrees) -> the pyFAI geometry that gives the same
    per-pixel 2theta: ``dict(dist, poni1, poni2, rot1, rot2, rot3)`` in metres and radians.

    MIDAS names throughout: ``BC_y`` is the COLUMN and ``BC_z`` the ROW of the image (``image[row=Z, col=Y]``).
    Exact (closed form, no fit); it holds for any tilt, not only small ones. No distortion, no detector flip.
    """
    pxc, pxr, L = float(pxY_um) * 1e-6, float(pxZ_um) * 1e-6, float(Lsd_um) * 1e-6
    TRs = _midas_tilt_matrix(tx, ty, tz)
    rot = _pyfai_angles(_AXIS_MAP @ TRs @ _AXIS_MAP.T)
    R = _pyfai_rotation(*rot)
    p00 = _AXIS_MAP @ (np.array([L, 0.0, 0.0]) + TRs @ np.array([0.0, float(BC_y) * pxc, -float(BC_z) * pxr]))   # lab position of pixel (Y, Z) = (0, 0)
    q = np.linalg.solve(R, p00)
    return dict(dist=float(q[2]), poni1=float(0.5 * pxr - q[0]), poni2=float(0.5 * pxc - q[1]),
                rot1=float(rot[0]), rot2=float(rot[1]), rot3=float(rot[2]))


def poni_to_midas(dist: float, poni1: float, poni2: float, rot1: float, rot2: float, rot3: float,
                  pxY_um: float, pxZ_um: float) -> Dict[str, float]:
    """pyFAI geometry (metres, radians) -> MIDAS geometry: ``dict(Lsd_um, BC_y, BC_z, tx, ty, tz)`` (pixels, degrees).

    Inverse of :func:`midas_to_poni`. ``BC`` is the pixel where the direct beam lands (pyFAI's 2theta = 0), not the PONI;
    ``BC_y`` is the column and ``BC_z`` the row. Assumes pyFAI's no-flip detector orientation.
    """
    pxc, pxr = float(pxY_um) * 1e-6, float(pxZ_um) * 1e-6
    R = _pyfai_rotation(float(rot1), float(rot2), float(rot3))
    T = _AXIS_MAP.T @ R @ _AXIS_MAP
    ab = np.linalg.solve(R[:2, :2], -float(dist) * R[:2, 2])         # detector-plane offsets whose lab x1 = x2 = 0 (the direct beam)
    Lsd = float((R @ np.array([ab[0], ab[1], float(dist)]))[2])
    return dict(Lsd_um=Lsd * 1e6,
                BC_y=float((ab[1] + float(poni2)) / pxc - 0.5), BC_z=float((ab[0] + float(poni1)) / pxr - 0.5),
                tx=math.degrees(math.atan2(-T[1, 2], T[2, 2])),
                ty=math.degrees(math.asin(max(-1.0, min(1.0, T[0, 2])))),
                tz=math.degrees(math.atan2(-T[0, 1], T[0, 0])))


def poni_file_to_midas(path: Union[str, Path]) -> Dict[str, float]:
    """Read a ``.poni`` file and return the MIDAS geometry ``dict(Lsd_um, BC_y, BC_z, tx, ty, tz)`` it describes.

    Unlike :func:`poni_file_to_bc` this is correct for a tilted detector (BC is the direct-beam pixel, not the PONI).
    ``BC_y`` is the column and ``BC_z`` the row. A flipping detector orientation is applied only for an untilted file;
    a tilted file with one raises ``NotImplementedError``.
    """
    p = read_poni(path)
    cfg = p.get("Detector_config")
    if not isinstance(cfg, dict) or "pixel1" not in cfg or "pixel2" not in cfg:
        raise ValueError(f"{path}: Detector_config lacks pixel1/pixel2")
    px_row_um, px_col_um = float(cfg["pixel1"]) * 1e6, float(cfg["pixel2"]) * 1e6
    rots = (float(p["Rot1"]), float(p["Rot2"]), float(p["Rot3"]))
    if not any(rots):
        row, col = poni_to_bc(float(p["Poni1"]), float(p["Poni2"]), px_row_um, px_col_um,
                              orientation=p["orientation"], shape=p["shape"])
        return dict(Lsd_um=float(p["Distance"]) * 1e6, BC_y=col, BC_z=row, tx=0.0, ty=0.0, tz=0.0)
    if any(orientation_flips(p["orientation"])):
        raise NotImplementedError(
            f"{path}: tilted PONI with a flipping detector orientation ({describe_orientation(p['orientation'])}); "
            "flip the image to the no-flip orientation first")
    return poni_to_midas(float(p["Distance"]), float(p["Poni1"]), float(p["Poni2"]), *rots, px_col_um, px_row_um)


def make_pyfai_integrator(spec, *, ImportError_msg: bool = True,
                          orientation: Optional[int] = None,
                          shape: Optional[Tuple[int, int]] = None):
    """Build a pyFAI ``AzimuthalIntegrator`` with the same per-pixel 2theta as ``spec``.

    Applies the half-pixel shift, the row/column assignment (``BC_z``/``pxZ`` are the rows) and the full tilt
    (``tx, ty, tz`` -> ``rot1..3``, ``Lsd`` -> ``dist``, direct-beam pixel -> PONI). Raises ``ImportError`` if pyFAI isn't
    installed, ``NotImplementedError`` for a tilted spec with a flipping ``orientation``. No distortion is carried over.
    """
    try:
        import pyFAI
    except ImportError as e:
        raise ImportError(
            "pyFAI not installed. pip install pyFAI"
        ) from e
    pxy_um, pxz_um = spec.effective_pxYZ()           # pxZ unset (0) means square, as elsewhere in v2
    tx, ty, tz = float(spec.tx), float(spec.ty), float(spec.tz)
    if tx or ty or tz:
        if any(orientation_flips(orientation)):
            raise NotImplementedError("tilted spec with a flipping detector orientation is not supported")
        g = midas_to_poni(float(spec.Lsd), float(spec.BC_y), float(spec.BC_z), tx, ty, tz, pxy_um, pxz_um)
    else:
        poni1_m, poni2_m = bc_to_poni(float(spec.BC_z), float(spec.BC_y), pxz_um, pxy_um,
                                      orientation=orientation, shape=shape)
        g = dict(dist=float(spec.Lsd) * 1e-6, poni1=poni1_m, poni2=poni2_m, rot1=0.0, rot2=0.0, rot3=0.0)
    kwargs = dict(
        dist=g["dist"], poni1=g["poni1"], poni2=g["poni2"], rot1=g["rot1"], rot2=g["rot2"], rot3=g["rot3"],
        pixel1=pxz_um * 1e-6, pixel2=pxy_um * 1e-6,
        wavelength=float(spec.Wavelength) * 1e-10,
    )
    if orientation is not None:
        from pyFAI.detectors import Detector
        kwargs.pop("pixel1"); kwargs.pop("pixel2")
        kwargs["detector"] = Detector(
            pixel1=pxz_um * 1e-6, pixel2=pxy_um * 1e-6,
            max_shape=shape, orientation=int(orientation),
        )
    return pyFAI.AzimuthalIntegrator(**kwargs)


__all__ = [
    "bc_to_poni", "poni_to_bc", "poni_file_to_row_col",
    "make_pyfai_integrator",
    "midas_to_poni", "poni_to_midas", "poni_file_to_midas",
    "read_poni", "poni_file_to_bc",
    "orientation_flips", "describe_orientation",
    "NO_FLIP_ORIENTATIONS",
]
