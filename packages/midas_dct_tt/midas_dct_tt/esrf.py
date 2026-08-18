"""Read ESRF DCT geometry and convert it into MIDAS conventions.

Phase 5 of ``implementation_plan.md``: the first module in this package that
faces *real* data rather than a synthetic phantom.

Why this module is only a converter
-----------------------------------
The ESRF ``dct`` toolbox (MATLAB, ``gtGeoLabDefaultParameters.m`` and friends) is
the de-facto reference implementation of DCT, so its parameter file is the format
real DCT geometry actually arrives in. Happily its **lab frame is identical to
ours** -- the definition is stated verbatim in ``gtGeoLabDefaultParameters.m``::

    par_labgeo.beamdir = [1 0 0];
    deflabX = 'Along the beam direction.'
    deflabY = 'Right-handed from Y=cross(Z,X).'
    deflabZ = 'Along rotation axis. Positive away from sample stage.'

which is exactly :mod:`midas_dfxm.conventions` (lab ``x`` along the beam, ``z``
up, ``y = z x x``, ``v_lab = R @ v_sample``). Their sample frame
(``gtGeoSamDefaultParameters.m``: ``orig=[0 0 0]``, ``dirx/diry/dirz`` = identity,
"the Lab but rotating with the sample") is ours too. So **no rotation is needed**
and this module does not attempt one.

What *does* differ is four things, and every one of them is a silent corruption
-- each produces a perfectly plausible image that is simply wrong:

===========================  ==================  =======================
quantity                     ESRF                MIDAS
===========================  ==================  =======================
length unit                  mm (``labunit``)    micrometres
pixel index origin           1-based MATLAB      0-based Python
stage handedness             ``rotdir`` vector   ``DCT_OMEGA_SIGN``
detector orientation         ``detdiru/detdirv`` normal + ``center_px``
===========================  ==================  =======================

The pixel-origin one is the nastiest, because it is *exactly one pixel* and the
two defaults look unrelated until you write them down: ESRF sets
``detrefu = detsizeu/2 + 0.5``, which in 1-based indexing is the geometric centre
of an ``N``-pixel row; subtracting 1 for Python gives ``(N-1)/2``, which is
precisely :func:`~midas_dct_tt.project.default_center_px`. They agree -- but only
after the conversion. :func:`esrf_center_px` is that subtraction, and
``tests/test_esrf.py`` pins the two defaults against each other so the agreement
cannot rot.

The handedness mapping, and why it is safe
------------------------------------------
ESRF applies a **right-handed** rotation by ``+omega`` about the ``rotdir``
vector and puts the stage's handedness into that vector; we rotate about lab
``+z`` by ``sign * omega`` and put the handedness into a scalar. Verified rather
than assumed: ``gtMathsRotationMatrixComp.m`` builds ``sin`` as the standard
cross-product matrix ``[n]_x``, so ``gtMathsRotationTensor`` is Rodrigues with
``+sin`` -- the same rotation as ``midas_dfxm.conventions.rotation_matrix``.
``tests/test_esrf.py`` checks the two agree numerically to 1e-12 over a sweep of
axes and angles before trusting the scalar mapping ``sign = rotdir_z``.

That mapping then reproduces the known table: ESRF's ``'pmo'`` stage is
``rotdir = [0 0 -1]``, i.e. ``sign = -1``, the same clockwise sense as the 1-ID
aero stage (:data:`~midas_dct_tt.conventions.DCT_OMEGA_SIGN_AERO`), while
``diffrz``/``omega``/``mrsrot``/``srot`` are ``[0 0 +1]``. A sign error here does
not crash and does not degrade any fit -- it **mirrors** the reconstruction
undetectably. Hence the conversion refuses to guess: a ``rotdir`` that is not
along ``+-z`` raises rather than being silently projected.

Scope, stated honestly
----------------------
* The **conversion functions** below are pure and fully tested.
* :func:`load_esrf_parameters` is tested against genuine ``.mat`` files in both
  formats (classic via ``scipy.io.savemat``, and a ``-v7.3`` nested-struct layout
  via ``h5py``), including magic-byte format detection --
  ``tests/test_esrf_loader.py``. It has still **never seen a file written by
  ESRF's own toolbox**, since no DCT dataset is in hand, so a quirk of the real
  writer could still bite. The conversion path is deliberately separable, so a bad
  parse costs you only the parsing: hand-build the dicts and call
  :class:`ESRFGeometry`.
* Motor angles are **not** converted. ESRF's ``gtAlignReflection_miller.m`` solves
  the alignment as goniometer motors (``samrx``, ``samry``, ``diffrz``) while
  :func:`~midas_dct_tt.conventions.tt_alignment` returns a rotation matrix.
  The motor pair is solved by :func:`midas_dct_tt.goniometer.topotomo_tilts`,
  validated against 74 real ID11 scans to a median 0.05 deg; ``diffrz`` remains
  unpinned and this module does not pretend otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .conventions import DCT_OMEGA_SIGN_AERO, DCT_OMEGA_SIGN_CCW
from .forward import PlaneDetector
from .project import default_center_px

__all__ = [
    "ESRF_LENGTH_UNIT_UM",
    "ESRFGeometry",
    "esrf_center_px",
    "esrf_detector",
    "esrf_detector_basis",
    "esrf_omega_sign",
    "load_esrf_parameters",
    "mm_to_um",
    "rodrigues_to_crystal_to_sample",
]

#: ESRF DCT records lengths in millimetres (``par_labgeo.labunit = 'mm'``);
#: MIDAS is micrometres throughout. One number, one place.
ESRF_LENGTH_UNIT_UM = 1000.0

_EPS = 1e-9


def mm_to_um(x):
    """ESRF millimetres -> MIDAS micrometres. Accepts scalars, tuples or tensors."""
    if isinstance(x, torch.Tensor):
        return x * ESRF_LENGTH_UNIT_UM
    if isinstance(x, (list, tuple)):
        return type(x)(float(v) * ESRF_LENGTH_UNIT_UM for v in x)
    return float(x) * ESRF_LENGTH_UNIT_UM


def rodrigues_to_crystal_to_sample(rod):
    """Rodrigues vector from a ``pymicro``-written grain map -> crystal-to-sample.

    DCT grain maps in the ESRF deposits are written by ``pymicro``, which stores
    ``r = -n tan(theta/2)`` (the NEGATED convention, ``Orientation.
    Rodrigues2OrientationMatrix``). Applying the textbook formula

        ``R = I + 2/(1 + r.r) (K + K^2)``,   ``K = skew(r)``

    to the stored vector therefore yields crystal-to-sample directly, with no
    transpose. ``pymicro``'s ``orientation_matrix()`` is passive
    (sample-to-crystal); this returns its transpose.

    Why not :func:`midas_stress.orientation.rodrigues_to_orient_mat`
    ---------------------------------------------------------------
    **Below midas-stress 0.9.0** that function disagreed, and not by a
    convention. It returned a valid rotation (``det = 1``, orthogonal to 1e-16)
    about the **correct axis** but at the **wrong angle**, inflated by
    ``1/cos^2(theta/2)``: exact at zero, +0.2 deg at 5 deg, 60 -> 80 deg, and
    90 -> 180 deg. No choice of sign, transpose or active/passive alters a
    rotation's angle while preserving its axis, so it could not be reconciled by
    adaptation. Substituting it here moved the 74-scan tilt residual from
    0.043 deg to 26.5 deg -- indistinguishable from assigning the scans to
    random grains. See ``dev/real_data/convention_bridge.py``, which tests all
    four sign/transpose combinations.

    **Fixed in midas-stress 0.9.0**, which builds the rotation vector as
    ``n * theta`` and now agrees with this function to 2.7e-15 over random
    Rodrigues vectors. This implementation is kept because midas-dct-tt does not
    otherwise import midas-stress and floors it only transitively; delegating
    would add a hard floor for three lines of arithmetic. Delete it if that
    floor is ever wanted for another reason.
    """
    r = np.asarray(rod, dtype=float).reshape(3)
    K = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
    return np.eye(3) + (2.0 / (1.0 + float(r @ r))) * (K + K @ K)


def esrf_center_px(detrefu, detrefv) -> tuple:
    """ESRF 1-based reference pixel -> MIDAS 0-based ``center_px``.

    MATLAB indexes pixels ``1..N``, so ESRF's default reference
    ``detrefu = detsizeu/2 + 0.5`` is the geometric centre of the row. Python
    indexes ``0..N-1``, where that same centre is ``(N-1)/2``. The two differ by
    exactly 1, which is small enough to survive review and large enough to bias
    every reconstructed position by a pixel.
    """
    return float(detrefu) - 1.0, float(detrefv) - 1.0


def esrf_omega_sign(rotdir, *, atol: float = 1e-6) -> float:
    """ESRF ``rotdir`` vector -> MIDAS scalar omega sign.

    Returns :data:`~midas_dct_tt.conventions.DCT_OMEGA_SIGN_CCW` (+1) or
    :data:`~midas_dct_tt.conventions.DCT_OMEGA_SIGN_AERO` (-1).

    Raises
    ------
    ValueError
        If ``rotdir`` is not parallel to lab ``z``. A tilted rotation axis is a
        perfectly legal ESRF configuration but it is *not* expressible as a
        scalar sign in :func:`~midas_dct_tt.conventions.dct_sample_rotation`, and
        quietly projecting it onto ``z`` would mirror or skew the reconstruction
        with no visible symptom. Build the rotation explicitly instead.
    """
    r = torch.as_tensor(rotdir, dtype=torch.float64).reshape(-1)
    if r.numel() != 3:
        raise ValueError(f"rotdir must have 3 components, got {r.numel()}")
    n = float(torch.linalg.vector_norm(r))
    if n < _EPS:
        raise ValueError("rotdir is the zero vector")
    r = r / n
    if float(torch.abs(r[0])) > atol or float(torch.abs(r[1])) > atol:
        raise ValueError(
            f"rotdir {tuple(float(v) for v in r)} is not parallel to lab z. "
            "MIDAS' DCT scan takes a scalar sign about +z; a tilted axis needs an "
            "explicit rotation, not a projection onto z."
        )
    return DCT_OMEGA_SIGN_CCW if float(r[2]) > 0 else DCT_OMEGA_SIGN_AERO


def esrf_detector_basis(detdiru, detdirv, *, atol: float = 1e-6):
    """``(u_hat, v_hat, normal)`` in the lab frame from ESRF's detector directions.

    ESRF stores the detector as two in-plane unit vectors: ``detdiru`` along
    increasing pixel column and ``detdirv`` along increasing row. The returned
    normal points **from the sample towards the detector** -- the projection
    direction this package's ray-to-plane projector takes as a parameter.

    That outward normal is ``cross(v, u)``, **not** ``cross(u, v)``: the ESRF
    triad ``(detdiru, detdirv, outward)`` is *left*-handed, which is the ordinary
    image convention of u-right/v-down seen from behind the detector. Both
    branches of ``gtGeoDetDefaultParameters.m`` agree on it, and neither is
    ambiguous:

    * inline direct beam -- ``u=+y``, ``v=-z``, ``detrefpos=[dist,0,0]``, so
      ``cross(u,v) = -x`` points back up the beam at the source;
    * vertical no-direct-beam -- ``u=-z``, ``v=+x``, ``detrefpos=[0,dist,0]``,
      so ``cross(u,v) = -y``, again pointing at the sample.

    Taking ``cross(u, v)`` therefore puts the detector *behind* the sample and
    flips the sign of ``distance_um``. :func:`esrf_detector` independently
    re-derives the sense from ``detrefpos`` and raises on disagreement, so this
    convention cannot be wrong silently.

    Both vectors are normalised on the way through, and non-orthogonality is
    rejected rather than silently orthogonalised -- a skewed detector basis in a
    real parameter file means something upstream is wrong, and squaring it up
    here would hide that.
    """
    u = torch.as_tensor(detdiru, dtype=torch.float64).reshape(-1)
    v = torch.as_tensor(detdirv, dtype=torch.float64).reshape(-1)
    if u.numel() != 3 or v.numel() != 3:
        raise ValueError("detdiru and detdirv must each have 3 components")
    nu, nv = float(torch.linalg.vector_norm(u)), float(torch.linalg.vector_norm(v))
    if nu < _EPS or nv < _EPS:
        raise ValueError("detdiru/detdirv must be non-zero")
    u, v = u / nu, v / nv
    dot = float(torch.sum(u * v))
    if abs(dot) > 1e-6:
        raise ValueError(
            f"detdiru and detdirv are not orthogonal (cos = {dot:.3e}). Refusing to "
            "orthogonalise silently -- check the source parameter file."
        )
    n = torch.linalg.cross(v, u)          # left-handed triad -- see docstring
    return u, v, n / torch.linalg.vector_norm(n)


def esrf_detector(detgeo: dict) -> tuple:
    """ESRF ``detgeo`` struct -> ``(PlaneDetector, normal)``, in micrometres.

    Expects the keys written by ``gtGeoDetDefaultParameters.m``: ``pixelsizeu``,
    ``pixelsizev``, ``detsizeu``, ``detsizev``, ``detrefu``, ``detrefv``,
    ``detrefpos``, ``detdiru``, ``detdirv``.

    ``distance_um`` is the reference-pixel position projected onto the detector
    normal, i.e. the perpendicular sample-to-detector distance -- which is what
    :class:`~midas_dct_tt.forward.PlaneDetector` means by it, and is *not* in
    general ``|detrefpos|`` (those agree only when the reference pixel sits on
    the normal through the origin).

    The normal is returned separately because ``PlaneDetector`` carries only a
    distance: this package's projector takes the detector normal as an explicit
    parameter, since TT's detector is perpendicular to ``k_h`` while DCT's is
    perpendicular to the beam.

    Anisotropic pixels are rejected: ``PlaneDetector.pixel_um`` is a single
    number, so a ``pixelsizeu != pixelsizev`` file cannot be represented and
    would otherwise be silently squashed along one axis.
    """
    required = ("pixelsizeu", "pixelsizev", "detsizeu", "detsizev",
                "detrefu", "detrefv", "detrefpos", "detdiru", "detdirv")
    missing = [k for k in required if k not in detgeo]
    if missing:
        raise KeyError(f"detgeo is missing required ESRF fields: {missing}")

    pu, pv = float(detgeo["pixelsizeu"]), float(detgeo["pixelsizev"])
    if abs(pu - pv) > 1e-12 * max(abs(pu), abs(pv), 1.0):
        raise ValueError(
            f"anisotropic pixels (u={pu}, v={pv} mm) cannot be represented by "
            "PlaneDetector, which carries a single pixel_um."
        )

    u, v, n = esrf_detector_basis(detgeo["detdiru"], detgeo["detdirv"])
    refpos = torch.as_tensor(detgeo["detrefpos"], dtype=torch.float64).reshape(-1)
    if refpos.numel() != 3:
        raise ValueError("detrefpos must have 3 components")
    proj = float(torch.sum(refpos * n))
    if proj <= 0.0:
        raise ValueError(
            f"detector normal from cross(detdirv, detdiru) projects onto detrefpos "
            f"as {proj:.6g} mm, i.e. it points away from the detector. Either "
            "detdiru/detdirv are swapped or detrefpos is on the wrong side; "
            "refusing to flip it silently, since that would mirror the "
            "reconstruction with no other symptom."
        )
    distance_um = mm_to_um(proj)

    det = PlaneDetector(
        pixel_um=mm_to_um(pu),
        shape=(int(detgeo["detsizeu"]), int(detgeo["detsizev"])),
        center_px=esrf_center_px(detgeo["detrefu"], detgeo["detrefv"]),
        distance_um=distance_um,
    )
    return det, n


@dataclass
class ESRFGeometry:
    """An ESRF DCT geometry, held as-read, with MIDAS conversions on demand.

    Keeping the raw structs verbatim and converting only through the accessors
    means the file on disk stays the single source of truth: nothing is
    normalised at construction time where it could quietly diverge from what
    ESRF actually wrote.
    """

    labgeo: dict = field(default_factory=dict)
    detgeo: dict = field(default_factory=dict)
    samgeo: dict = field(default_factory=dict)

    def omega_sign(self) -> float:
        """Scalar DCT omega sign from ``labgeo['rotdir']``."""
        if "rotdir" not in self.labgeo:
            raise KeyError("labgeo has no 'rotdir'")
        return esrf_omega_sign(self.labgeo["rotdir"])

    def detector(self) -> tuple:
        """``(PlaneDetector, normal)`` in micrometres."""
        return esrf_detector(self.detgeo)

    def check_frame(self) -> None:
        """Assert the lab frame really is the MIDAS one before anything is trusted.

        The whole no-rotation-needed argument rests on ``beamdir == [1 0 0]``. If
        a file says otherwise, every downstream number is wrong, so fail loudly
        here rather than produce a plausible reconstruction in the wrong frame.
        """
        if "beamdir" in self.labgeo:
            b = torch.as_tensor(self.labgeo["beamdir"], dtype=torch.float64).reshape(-1)
            b = b / torch.linalg.vector_norm(b)
            expect = torch.as_tensor([1.0, 0.0, 0.0], dtype=torch.float64)
            if float(torch.linalg.vector_norm(b - expect)) > 1e-6:
                raise ValueError(
                    f"labgeo beamdir is {tuple(float(x) for x in b)}, not [1,0,0]. "
                    "This package assumes the MIDAS/ESRF shared lab frame; a "
                    "different beam direction needs an explicit frame rotation."
                )


def load_esrf_parameters(path) -> ESRFGeometry:
    """Read an ESRF DCT ``parameters.mat`` into an :class:`ESRFGeometry`.

    Handles both MATLAB ``-v7.3`` (HDF5, via ``h5py``) and older formats (via
    ``scipy.io.loadmat``), picking by inspecting the file's magic bytes rather
    than its extension.

    Tested end-to-end against genuine ``.mat`` files in **both** formats --
    ``scipy.io.savemat`` for the classic one and an ``h5py`` layout matching how
    MATLAB ``-v7.3`` writes a nested struct (see ``tests/test_esrf_loader.py``),
    including format detection from magic bytes rather than the file extension.

    .. note::
       Still not exercised on a file produced by ESRF's own toolbox -- no DCT
       dataset has been available -- so a layout quirk of the real writer could
       still bite. The conversion functions this feeds are independently tested,
       so if the parse goes wrong you can build the ``labgeo``/``detgeo`` dicts by
       hand and call :class:`ESRFGeometry` directly, losing nothing but the parsing.
    """
    with open(path, "rb") as fh:
        magic = fh.read(8)
    is_hdf5 = magic.startswith(b"\x89HDF\r\n\x1a\n")

    if is_hdf5:
        import h5py

        def _deref(f, obj):
            if isinstance(obj, h5py.Group):
                return {k: _deref(f, obj[k]) for k in obj.keys()}
            arr = obj[()]
            if getattr(arr, "dtype", None) is not None and arr.dtype == object:
                return [_deref(f, f[r]) for r in arr.flat]
            # MATLAB writes arrays transposed relative to C order.
            return arr.T if getattr(arr, "ndim", 0) > 1 else arr

        with h5py.File(path, "r") as f:
            root = f["parameters"] if "parameters" in f else f
            raw = {k: _deref(f, root[k]) for k in root.keys()}
    else:
        from scipy.io import loadmat

        m = loadmat(path, struct_as_record=False, squeeze_me=True)
        p = m.get("parameters", m)

        def _tod(o):
            if hasattr(o, "_fieldnames"):
                return {k: _tod(getattr(o, k)) for k in o._fieldnames}
            return o

        raw = _tod(p)

    def _sub(name):
        v = raw.get(name, {})
        return v if isinstance(v, dict) else {}

    geo = ESRFGeometry(labgeo=_sub("labgeo"), detgeo=_sub("detgeo"),
                       samgeo=_sub("samgeo"))
    geo.check_frame()
    return geo


def esrf_default_detgeo(detsizeu: int, detsizev: int, pixelsize_mm: float,
                        dist_mm: float) -> dict:
    """The inline direct-beam ``detgeo`` that ``gtGeoDetDefaultParameters.m`` builds.

    Provided so the conversion path can be exercised -- and regression-tested --
    without an ESRF file in hand. Mirrors the non-flipped
    (``rotation_direction != 'counterclockwise'``) inline branch::

        detrefpos = [dist 0 0];  detdiru = [0 1 0];  detdirv = [0 0 -1];
        detrefu = detsizeu/2 + 0.5;  detrefv = detsizev/2 + 0.5;
    """
    return {
        "pixelsizeu": pixelsize_mm, "pixelsizev": pixelsize_mm,
        "detsizeu": int(detsizeu), "detsizev": int(detsizev),
        "detrefu": detsizeu / 2.0 + 0.5, "detrefv": detsizev / 2.0 + 0.5,
        "detrefpos": (dist_mm, 0.0, 0.0),
        "detdiru": (0.0, 1.0, 0.0), "detdirv": (0.0, 0.0, -1.0),
    }
