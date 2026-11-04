"""Readers for rotation-series frames (CBF with an imgCIF header; needs ``fabio``).

``read_cbf_scan`` returns the stack and a :class:`ScanHeader` with what the headers STATE (wavelength, distance, pixel, gain, scan axis, per-frame angle).
Nothing here decides the lab-frame convention (image orientation, sense of rotation, beam-centre sign): a header does not carry them reliably, and
the workflow resolves them from the indexing itself (:mod:`midas_sx.conventions`). Frames are untrusted data: only numbers are read from the headers,
never executed or followed.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ScanHeader", "read_cbf_scan", "list_cbf_frames", "read_sfrm_scan", "read_sfrm_mask", "list_sfrm_frames", "read_pilatus_cbf_scan", "pilatus_header_fields"]

_NUM = re.compile(rb"(\d+)\D*$")


@dataclass
class ScanHeader:
    n_frames: int
    shape: Tuple[int, int]
    wavelength_A: float
    distance_mm: float
    pixel_mm: float
    gain: float                      # electrons (or photons, per the file) per count, as stated; 1.0 if the header has none
    overload: Optional[float]
    scan_axis: str
    angle_start: float               # of frame 0, degrees
    angle_step: float                # median of the per-frame differences, degrees
    angle_steps_regular: bool        # every per-frame difference within 1e-3 deg of the step
    exposure_s: Optional[float]
    n_undefined: int                 # voxels below zero in the file, set to the header's undefined value 0 (e.g. -1 flags of the ADSC conversion)
    detector: str
    files: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)    # format-specific header values that are not part of the common description (SFRM: kappa-stage angles, centre, zoom ...)


def list_cbf_frames(path_or_glob: str) -> List[str]:
    """Sorted frame paths from a directory or a glob; sorted by the trailing integer of the file stem, not lexically."""
    paths = glob.glob(os.path.join(path_or_glob, "*.cbf")) if os.path.isdir(path_or_glob) else glob.glob(path_or_glob)
    if not paths:
        raise FileNotFoundError(f"no .cbf frames under {path_or_glob!r}")

    def key(p):
        m = re.search(r"(\d+)(?=\.cbf$)", p)
        return (int(m.group(1)) if m else -1, p)
    return sorted(paths, key=key)


def _f(x) -> Optional[float]:
    try:
        return float(x.decode() if isinstance(x, bytes) else x)
    except (TypeError, ValueError):
        return None


def _scan_axis_row(header: Dict, axis_id: str) -> Optional[Dict]:
    """The row of the ``_diffrn_scan_axis`` loop for ``axis_id``. fabio stores a CBF loop as ``header['loop_'] = [[names, [row dicts]], ...]``."""
    loops = header.get("loop_")
    for loop in loops if isinstance(loops, list) else []:
        if not (isinstance(loop, list) and len(loop) == 2 and isinstance(loop[1], list)):
            continue
        for r in loop[1]:
            if isinstance(r, dict) and r.get(b"_diffrn_scan_axis.axis_id", b"").decode(errors="replace").upper() == axis_id.upper():
                return r
    return None


def read_cbf_scan(path_or_glob: str, *, max_frames: Optional[int] = None, scan_axis: str = "GONIOMETER_PHI",
                  dtype=np.uint16) -> Tuple[np.ndarray, ScanHeader]:
    """Read the frames; returns ``(stack (n, rows, cols) of dtype, ScanHeader)``.

    The stack keeps the file's integer counts (no gain applied). Negative voxels are the file's flag for an invalid pixel (the Zenodo 7265997 frames carry
    -1 in some pixels although the detector counts are unsigned): they are set to 0, the header's ``undefined_value``, which the mask builder treats as
    dead, and counted in ``ScanHeader.n_undefined``. Values above the range of ``dtype`` raise instead of wrapping.
    """
    import fabio
    files = list_cbf_frames(path_or_glob)
    if max_frames is not None:
        files = files[:max_frames]
    first = fabio.open(files[0])
    shape = first.data.shape
    stack = np.empty((len(files),) + shape, dtype=dtype)
    starts, notes = [], []
    n_undef = 0
    info = np.iinfo(dtype)
    for i, p in enumerate(files):
        im = first if i == 0 else fabio.open(p)
        d = im.data
        if d.shape != shape:
            raise ValueError(f"{p}: shape {d.shape} differs from {shape}")
        if d.max() > info.max:
            raise ValueError(f"{p}: values {d.min()}..{d.max()} do not fit {np.dtype(dtype)}")
        neg = d < max(info.min, 0)
        if neg.any():
            n_undef += int(neg.sum())
            d = np.where(neg, 0, d)
        elif info.min < 0 and d.min() < info.min:
            raise ValueError(f"{p}: values {d.min()}..{d.max()} do not fit {np.dtype(dtype)}")
        stack[i] = d
        row = _scan_axis_row(im.header, scan_axis)
        starts.append(_f(row.get(b"_diffrn_scan_axis.angle_start")) if row else None)
    h = first.header
    wl = _f(h.get("_diffrn_radiation_wavelength.wavelength"))
    dist = _f(h.get("_diffrn_measurement.sample_detector_distance"))
    px = None
    m = re.search(r"PIXEL_SIZE=\s*([0-9.eE+-]+)", str(h.get("_diffrn_data_frame.details", "")))
    if m:
        px = float(m.group(1))
    if None in (wl, dist, px):
        raise ValueError(f"{files[0]}: header lacks wavelength / distance / pixel size ({wl}, {dist}, {px})")
    if any(s is None for s in starts):
        raise ValueError(f"no {scan_axis} angle_start in the frame headers")
    a = np.asarray(starts, float)
    d = np.diff(a)
    step = float(np.median(d)) if len(d) else 0.0
    regular = bool(len(d) == 0 or np.all(np.abs(d - step) < 1e-3))
    if n_undef:
        notes.append(f"{n_undef} voxels below zero set to 0 (invalid-pixel flags)")
    if not regular:
        notes.append("per-frame angle differences are not constant: the scan is NOT a regular sweep")
    exposure = _f(h.get("_diffrn_scan_frame.integration_time"))
    hdr = ScanHeader(n_frames=len(files), shape=shape, wavelength_A=wl, distance_mm=dist, pixel_mm=px,
                     gain=_f(h.get("_array_intensities.gain")) or 1.0, overload=_f(h.get("_array_intensities.overload")),
                     scan_axis=scan_axis, angle_start=float(a[0]), angle_step=step, angle_steps_regular=regular,
                     exposure_s=exposure, n_undefined=n_undef, detector=str(h.get("_diffrn_detector.type", "")).strip(), files=files, notes=notes)
    return stack, hdr


# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Bruker SFRM
# --------------------------------------------------------------------------------------------------------------------------------------------------------

def list_sfrm_frames(path_or_glob: str, *, prefix: Optional[str] = None) -> List[str]:
    """Sorted ``.sfrm`` frame paths from a directory or a glob, ordered by the trailing integer of the stem. ``prefix`` keeps one RUN of a Bruker series
    (``mo_x_01_`` keeps ``mo_x_01_0001.sfrm`` but not ``mo_x_xa_01_0001.sfrm``)."""
    paths = glob.glob(os.path.join(path_or_glob, "*.sfrm")) if os.path.isdir(path_or_glob) else glob.glob(path_or_glob)
    if prefix is not None:
        paths = [p for p in paths if os.path.basename(p).startswith(prefix)]
    if not paths:
        raise FileNotFoundError(f"no .sfrm frames under {path_or_glob!r} (prefix {prefix!r})")
    paths = [p for p in paths if re.search(r"_\d+\.sfrm$", p) and (prefix is not None and "_xa_" in prefix or not re.search(r"_xa_\d+_\d+\.sfrm$", p))]    # Bruker active-mask frames (_xa_) belong to no run

    def key(p):
        m = re.search(r"(\d+)(?=\.sfrm$)", p)
        return (int(m.group(1)) if m else -1, p)
    return sorted(paths, key=key)


def _floats(x) -> List[float]:
    out = []
    for tok in str(x).replace("\n", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            break
    return out


def read_sfrm_mask(path: str) -> np.ndarray:
    """The ACTIVE MASK frame of a Bruker run (``*_xa_*.sfrm``): returns a bool array, True = DEAD / not active (the midas_defect convention)."""
    import fabio
    im = fabio.open(path)
    if "MASK" not in str(im.header.get("TYPE", "")).upper():
        raise ValueError(f"{path}: TYPE {im.header.get('TYPE')!r} is not an active-mask frame")
    return np.asarray(im.data) == 0


def read_sfrm_scan(path_or_glob: str, *, prefix: Optional[str] = None, max_frames: Optional[int] = None, dtype=np.uint32) -> Tuple[np.ndarray, ScanHeader]:
    """Read one Bruker run (fabio's SFRM reader, overflow table included: the summed counts of a frame equal the header ``NCOUNTS``).

    ``ScanHeader.angle_start`` is the START EDGE of frame 0 and ``angle_step`` the signed ``INCREME`` of the scanned axis (``AXIS`` 2 = omega, 3 = phi);
    the per-frame start edges are unwrapped through 360 deg and checked against the step (``angle_steps_regular``). Wavelength is the first WAVELEN entry (K-alpha
    average). The distance is the SECOND DISTANC value (the first when only one is given) taken as centimetres, the pixel size is 10 / (pixels per cm of DETTYPE) x ZOOM: both are only STARTING values (the geometry
    is fitted in the workflow); ``extra`` carries the kappa-stage angles (2theta of the detector, omega, phi, chi), CENTER, ZOOM and the raw strings.
    Negative voxels (none seen in the Zenodo set) would be set to 0 and counted, as for CBF.
    """
    import fabio
    files = list_sfrm_frames(path_or_glob, prefix=prefix)
    if max_frames is not None:
        files = files[:max_frames]
    first = fabio.open(files[0])
    shape = first.data.shape
    stack = np.empty((len(files),) + shape, dtype=dtype)
    info = np.iinfo(dtype)
    n_undef, starts = 0, []
    for i, p in enumerate(files):
        im = first if i == 0 else fabio.open(p)
        d = np.asarray(im.data)
        if d.shape != shape:
            raise ValueError(f"{p}: shape {d.shape} differs from {shape}")
        if d.max() > info.max:
            raise ValueError(f"{p}: values {d.min()}..{d.max()} do not fit {np.dtype(dtype)}")
        neg = d < 0
        if neg.any():
            n_undef += int(neg.sum())
            d = np.where(neg, 0, d)
        stack[i] = d
        starts.append(_floats(im.header.get("START"))[0])
    h = first.header
    wl = _floats(h.get("WAVELEN"))
    dist = _floats(h.get("DISTANC"))
    det = _floats(str(h.get("DETTYPE", "")).split(None, 1)[1] if len(str(h.get("DETTYPE", "")).split(None, 1)) > 1 else "")
    zoom = _floats(h.get("ZOOM"))
    centre = _floats(h.get("CENTER"))
    angles = _floats(h.get("ANGLES"))
    if not wl or not dist or not det or len(angles) < 4 or len(centre) < 2:
        raise ValueError(f"{files[0]}: header lacks WAVELEN / DISTANC / DETTYPE / ANGLES / CENTER")
    zf = zoom[0] if zoom else 1.0
    pixel_mm = 10.0 / det[0] * zf
    # DISTANC holds two values (2.7961 and 3.8001 in the Zenodo 18910433 set). The SAINT p4p quotes the first; a radial fit of the vendor's reflection list (2theta of the
    # CIF cell against the distance of the observed pixel from the beam, 1.0 deg rms with the right axis order, 8 deg with the first value) needs D / pixel = 285, i.e. the SECOND
    # value (38.0 mm / 0.133 mm). The second value is therefore the sample-detector distance used here when two are given; the workflow refines D / px anyway.
    distance_cm = dist[1] if len(dist) > 1 else dist[0]
    inc = _floats(h.get("INCREME"))
    step = float(inc[0]) if inc else 0.0
    a = np.degrees(np.unwrap(np.radians(np.asarray(starts, float))))
    d_a = np.diff(a)
    regular = bool(len(d_a) == 0 or np.all(np.abs(d_a - step) < 1e-3))
    notes = []
    if n_undef:
        notes.append(f"{n_undef} voxels below zero set to 0")
    if not regular:
        notes.append("per-frame start angles do not follow INCREME: the run is NOT a regular sweep")
    axis = {2: "OMEGA", 3: "PHI"}.get(int(_floats(h.get("AXIS"))[0]) if _floats(h.get("AXIS")) else -1, "UNKNOWN")
    hdr = ScanHeader(n_frames=len(files), shape=shape, wavelength_A=float(wl[0]), distance_mm=float(distance_cm) * 10.0, pixel_mm=float(pixel_mm), gain=1.0,
                     overload=None, scan_axis=axis, angle_start=float(a[0]), angle_step=step, angle_steps_regular=regular,
                     exposure_s=(_floats(h.get("ELAPSDR")) or [None])[0], n_undefined=n_undef, detector=str(h.get("DETTYPE", "")).split()[0] if h.get("DETTYPE") else "",
                     files=files, notes=notes,
                     extra=dict(detector_2theta_deg=angles[0], omega_deg=angles[1], phi_deg=angles[2], chi_deg=angles[3], centre_xy_px=(centre[0], centre[1]), zoom=zf,
                                pixels_per_cm=det[0], angle_starts_unwrapped=a, raw_distance=str(h.get("DISTANC")), raw_dettype=str(h.get("DETTYPE")),
                                model=str(h.get("MODEL", "")), temperature=str(h.get("LOWTEMP", ""))))
    return stack, hdr


# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Dectris Pilatus CBF (header convention PILATUS_1.2: the metadata is a '# Key value' text block in _array_data.header_contents)
# --------------------------------------------------------------------------------------------------------------------------------------------------------

def pilatus_header_fields(text: str) -> Dict[str, str]:
    """``{'Wavelength': '0.68890 A', 'Detector_distance': '0.16000 m', ...}`` from the ``# Key value`` lines of a Pilatus CBF header block."""
    out = {}
    for line in str(text).replace("\r", "").split("\n"):
        line = line.strip()
        if not line.startswith("#"):
            continue
        body = line[1:].strip()
        if not body:
            continue
        key, _, val = body.partition(" ")
        key = key.rstrip(":")
        if "=" in key:
            key = key.split("=")[0]
        out[key] = val.strip().lstrip("=").strip()
    return out


def _num(s, default=None):
    toks = _floats(str(s).replace("(", " ").replace(")", " ").replace(",", " "))
    return toks[0] if toks else default


def read_pilatus_cbf_scan(path_or_glob: str, *, prefix: Optional[str] = None, max_frames: Optional[int] = None, dtype=np.uint32,
                          keep_invalid: bool = False) -> Tuple[np.ndarray, ScanHeader]:
    """One run of Pilatus CBF frames (fabio). Negative pixels (-1 module gap, -2 excluded / hot pixel) are invalid: they are set to 0 in the stack and collected in
    ``ScanHeader.extra['dead_mask']`` (True = invalid in ANY frame). Counts are Poisson photons (gain 1; the camera applies its Tau dead-time correction).
    ``keep_invalid=True`` keeps the -1 / -2 flags in the stack (int32): ``midas_defect.ingest.build_mask`` reads the Pilatus convention "negative = gap / bad pixel" itself.

    ``angle_start`` is the ``Start_angle`` of frame 0, ``angle_step`` the ``Angle_increment``, ``scan_axis`` the ``Oscillation_axis``; ``extra`` carries Beam_xy (x = column,
    y = row, pixels), Detector_2theta, Polarization, the Omega / Phi / Chi / Kappa angles, Count_cutoff, Tau and the detector name. Beam_xy is the point where the beam would
    hit a detector at 2theta = 0, i.e. the foot of the perpendicular when the detector is swung (the workflow resolves that against the data).
    """
    import fabio
    files = [f for f in list_cbf_frames(path_or_glob) if prefix is None or os.path.basename(f).startswith(prefix)]
    if not files:
        raise FileNotFoundError(f"no .cbf frames under {path_or_glob!r} (prefix {prefix!r})")
    if max_frames is not None:
        files = files[:max_frames]
    first = fabio.open(files[0])
    fld = pilatus_header_fields(first.header.get("_array_data.header_contents", ""))
    needed = ("Wavelength", "Detector_distance", "Pixel_size", "Start_angle", "Angle_increment", "Oscillation_axis")
    miss = [k for k in needed if k not in fld]
    if miss:
        raise ValueError(f"{files[0]}: not a Pilatus 1.2 header (missing {miss})")
    shape = first.data.shape
    if keep_invalid:
        dtype = np.int32
    stack = np.zeros((len(files),) + shape, dtype=dtype)
    dead = np.zeros(shape, bool)
    info = np.iinfo(dtype)
    starts, n_neg = [], 0
    for i, p in enumerate(files):
        im = first if i == 0 else fabio.open(p)
        d = np.asarray(im.data)
        if d.shape != shape:
            raise ValueError(f"{p}: shape {d.shape} differs from {shape}")
        if d.max() > info.max:
            raise ValueError(f"{p}: values {d.min()}..{d.max()} do not fit {np.dtype(dtype)}")
        neg = d < 0
        if neg.any():
            dead |= neg
            n_neg += int(neg.sum())
            if not keep_invalid:
                d = np.where(neg, 0, d)
        stack[i] = d
        starts.append(_num(pilatus_header_fields(im.header.get("_array_data.header_contents", "")).get("Start_angle"), np.nan))
    a = np.asarray(starts, float)
    if not np.all(np.isfinite(a)):
        raise ValueError("a frame without Start_angle")
    step = float(_num(fld["Angle_increment"]))
    d_a = np.diff(a)
    regular = bool(len(d_a) == 0 or np.all(np.abs(d_a - step) < 1e-3))
    tok = fld["Pixel_size"].split()
    pixel_mm = float(tok[0]) * {"m": 1e3, "mm": 1.0, "um": 1e-3}.get(tok[1] if len(tok) > 1 else "m", 1e3)       # '172e-6 m x 172e-6 m'
    notes = []
    if not regular:
        notes.append("per-frame Start_angle does not follow Angle_increment: NOT a regular sweep")
    bxy = [float(v) for v in _floats(fld.get("Beam_xy", "").replace("(", " ").replace(")", " ").replace(",", " "))[:2]]
    hdr = ScanHeader(n_frames=len(files), shape=shape, wavelength_A=float(_num(fld["Wavelength"])), distance_mm=float(_num(fld["Detector_distance"])) * 1e3,
                     pixel_mm=pixel_mm, gain=1.0, overload=_num(fld.get("Count_cutoff")), scan_axis=fld["Oscillation_axis"].upper(), angle_start=float(a[0]),
                     angle_step=step, angle_steps_regular=regular, exposure_s=_num(fld.get("Exposure_time")), n_undefined=n_neg,
                     detector=" ".join(first.header.get("_array_data.header_contents", "").split("\n")[0].replace("#", "").replace("Detector:", "").split())[:60], files=files,
                     notes=notes,
                     extra=dict(dead_mask=dead, beam_xy_px=tuple(bxy), detector_2theta_deg=_num(fld.get("Detector_2theta"), 0.0), polarization=_num(fld.get("Polarization"), 0.5),
                                omega_deg=_num(fld.get("Omega")), phi_deg=_num(fld.get("Phi")), chi_deg=_num(fld.get("Chi")), kappa_deg=_num(fld.get("Kappa")),
                                tau_s=_num(fld.get("Tau")), count_cutoff=_num(fld.get("Count_cutoff")), angle_starts=a, header_convention=str(first.header.get("_array_data.header_convention", "")),
                                pixel_size_raw=fld["Pixel_size"]))
    return stack, hdr
