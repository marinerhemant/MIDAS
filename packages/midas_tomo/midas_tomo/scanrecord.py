"""Reading a 1-ID tomography scan's own record — the provenance backbone.

Every tomography scan at 1-ID writes ``<prefix>_TomoFastScan.dat``, and it is
**self-describing**. Its header carries the pixel size, the propagation
distance, the energy, the angular range, the handedness, and the exact number
of white, dark and projection frames. Nothing in ``packages/`` read it before
2026-08-23, which is why every reconstruction in the wild starts from a
hand-written ``prepare_data_<sample>.py`` with frame indices counted by eye::

    startNrDark  = 10938
    startNrWhite = 7317
    startNrData  = 7327

All three are derivable from the header, and this module derives them.

Why it matters more than convenience
------------------------------------
``tomocupy_args.yml`` records ``pixelSize: 1.17`` for both beamtimes surveyed
here. **Both are wrong.** 1.17 µm is the PointGrey value from
``ad_settings.csv``; these scans ran on a FLIR-GH1 at 5X, which is 0.708 µm
(bt_1id_jun25b ``nmc811s5tomo1``) and 0.69 µm (bt_1id_jul26 ``tomo_Ce_ht525_s2``). A
1.65x pixel-size error is a **4.5x error in every volume** computed from the
reconstruction. The scan record is the only place the truth is written down.

The metastr also names the in-plane handedness (``left handed``), which the
reconstruction itself cannot carry. Note what that does and does not settle:
it names a *convention*, not an axis assignment, so it does not by itself pick
one of the eight ``midas_stress.frames.TOMO_IN_PLANE`` permutations. Verify
with the meta-null regardless.

The omega sign
--------------
The metastr records the SPEC angles. On the **aero** stage the sample turns the
other way, and the standing 1-ID rule is to negate every omega. This is not an
inference from the rule alone: the beamline's own reconstruction driver
(``midas_tomo_python_nmc811_s5_tomo1.py``) writes
``thetas = np.arange(180, -180.1, -0.1)`` against a metastr that says
``-180/180/0.100``, which is exactly that negation. :meth:`TomoScan.thetas`
applies it and records that it did.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = ["TomoScan", "FrameBlock", "read_scan_record", "parse_metastr"]


@dataclass(frozen=True)
class FrameBlock:
    """A contiguous run of image sequence numbers with one role."""

    role: str          # front_white | projections | back_white | dark
    start: int
    count: int

    @property
    def stop(self) -> int:
        """One past the last sequence number."""
        return self.start + self.count

    def numbers(self) -> range:
        return range(self.start, self.stop)


_METASTR_KEYS = {
    "scan start/end/step angles": "angles",
    "exp. time": "exp_time",
    "WF#": "n_white",
    "DF#": "n_dark",
    "Proj#": "n_projections",
    "scantype": "scantype",
    "sumimg#": "sum_images",
    "shiftmotor": "shift_motor",
    "sampleshift": "sample_shift",
}


def parse_metastr(metastr: str) -> Dict[str, Any]:
    """Pull the fields out of a ``tomo_metastr`` string.

    The format is three ``---``-separated sections: a free-form optics
    description, a ``key=value`` list, and a trailing file list. The optics
    half is positional and comma-separated::

        D~100.000000mm, FLIR-GH1, 5X, 0.708 um/px, aero axis, left handed

    Parsed by *meaning* rather than by position, because the bt_1id_jul26 scan
    inserts an extra ``imgZE=100.000000`` field into the same section and a
    positional reader would silently shift every value after it.
    """
    out: Dict[str, Any] = {"raw": metastr}
    body = metastr.split("tomo_metastr:", 1)[-1]
    sections = body.split("---")
    optics = sections[0]

    m = re.search(r"D~\s*([-\d.]+)\s*mm", optics)
    if m:
        out["propagation_mm"] = float(m.group(1))
    m = re.search(r"([-\d.]+)\s*um/px", optics)
    if m:
        out["pixel_size_um"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*X\b", optics)
    if m:
        out["magnification"] = m.group(1) + "X"
    m = re.search(r"\b(left|right)\s+handed\b", optics, re.I)
    if m:
        out["handedness"] = m.group(1).lower()
    m = re.search(r"\b(\w+)\s+axis\b", optics)
    if m:
        out["rotation_axis"] = m.group(1).lower()
    # The detector is the comma field that is neither a distance, a
    # magnification, a pixel size, an axis, a handedness nor a key=value.
    for tok in (t.strip() for t in optics.split(",")):
        if (tok and "=" not in tok and "um/px" not in tok
                and not tok.startswith("D~") and "axis" not in tok
                and "handed" not in tok and not re.fullmatch(r"[\d.]+X", tok)):
            out.setdefault("detector", tok)

    kv = sections[1] if len(sections) > 1 else ""
    for key, name in _METASTR_KEYS.items():
        m = re.search(re.escape(key) + r"\s*=\s*([^,]+)", kv)
        if m:
            out[name] = m.group(1).strip()

    if "angles" in out:
        nums = re.findall(r"[-\d.]+", str(out["angles"]))
        if len(nums) >= 3:
            out["omega_start_deg"] = float(nums[0])
            out["omega_end_deg"] = float(nums[1])
            out["omega_step_deg"] = float(nums[2])
    for k in ("n_white", "n_dark", "n_projections"):
        if k in out:
            out[k] = int(re.findall(r"\d+", str(out[k]))[0])
    return out


@dataclass(frozen=True)
class TomoScan:
    """One tomography scan, as its own record describes it.

    Every field is read from ``<prefix>_TomoFastScan.dat``. Nothing is
    defaulted: a scan whose record does not state the pixel size raises rather
    than inheriting one, because the value that would be inherited
    (``tomocupy_args.yml``) is measurably wrong for both scans surveyed.
    """

    source: Path
    path: str
    image_prefix: str
    exposure_s: float
    energy_kev: float
    pixel_size_um: float
    propagation_mm: float
    detector: str
    magnification: str
    rotation_axis: str
    handedness: str
    omega_start_deg: float
    omega_end_deg: float
    omega_step_deg: float
    n_projections: int
    n_white: int
    n_dark: int
    first_image: int
    last_image: int
    n_images: int
    blocks: Tuple[FrameBlock, ...]
    started: str = ""
    shift_motor: Optional[str] = None
    raw_metastr: str = ""

    # ------------------------------------------------------------- geometry

    def fov_um(self, n_columns: int) -> float:
        """Field of view across ``n_columns`` detector columns.

        Takes the width as an argument because the scan record does **not**
        state it — it gives the pixel size and nothing about the sensor or any
        crop that was applied afterwards. The bt_1id_jun25b ``.raw`` files are a
        128x128 crop out of a much larger frame, so a self-computed FOV would
        be wrong by whatever the crop was.
        """
        if n_columns <= 0:
            raise ValueError(f"n_columns must be > 0; got {n_columns}")
        return float(n_columns) * self.pixel_size_um

    @property
    def is_aero(self) -> bool:
        return self.rotation_axis == "aero"

    def thetas(self, *, apply_aero_sign: bool = True) -> np.ndarray:
        """Projection angles in degrees, ready for the engine.

        On the aero stage the recorded SPEC angles run opposite to the sample
        rotation, so they are negated — see the module docstring, where the
        beamline's own driver is shown doing the same thing. Pass
        ``apply_aero_sign=False`` to get the angles exactly as recorded.
        """
        th = self.omega_start_deg + self.omega_step_deg * np.arange(
            self.n_projections, dtype=np.float64
        )
        if apply_aero_sign and self.is_aero:
            return -th
        return th

    def block(self, role: str) -> Optional[FrameBlock]:
        for b in self.blocks:
            if b.role == role:
                return b
        return None

    @property
    def has_back_white(self) -> bool:
        return self.block("back_white") is not None

    def frame_paths(self, role: str, *, root: Union[str, Path, None] = None,
                    ext: str = ".tif", digits: int = 6) -> List[Path]:
        """Absolute paths of the frames in one block.

        ``root`` defaults to the ``Path:`` recorded in the scan file, which is
        the acquisition machine's view and often not mounted where the
        analysis runs — pass the local root instead of editing the record.
        """
        b = self.block(role)
        if b is None:
            raise KeyError(
                f"this scan has no {role!r} block; it has "
                f"{[x.role for x in self.blocks]}"
            )
        base = Path(root) if root is not None else Path(self.path)
        stem = Path(self.image_prefix).name
        sub = Path(self.image_prefix).parent
        return [base / sub / f"{stem}_{n:0{digits}d}{ext}" for n in b.numbers()]

    def provenance(self) -> Dict[str, Any]:
        """Everything a downstream product should cite, in one dict."""
        return {
            "scan_record": str(self.source),
            "image_prefix": self.image_prefix,
            "pixel_size_um": self.pixel_size_um,
            "pixel_size_source": "tomo_metastr (NOT tomocupy_args.yml)",
            "propagation_mm": self.propagation_mm,
            "detector": self.detector,
            "magnification": self.magnification,
            "energy_kev": self.energy_kev,
            "rotation_axis": self.rotation_axis,
            "handedness": self.handedness,
            "handedness_note": "names a convention, not a TOMO_IN_PLANE "
                               "assignment - verify with the meta-null",
            "omega_deg": [self.omega_start_deg, self.omega_end_deg,
                          self.omega_step_deg],
            "omega_sign_applied": "negated (aero)" if self.is_aero else "as recorded",
            "n_projections": self.n_projections,
            "n_white": self.n_white,
            "n_dark": self.n_dark,
            "has_back_white": self.has_back_white,
            "blocks": {b.role: [b.start, b.count] for b in self.blocks},
        }


def _header_value(lines: List[str], label: str) -> Optional[str]:
    pat = re.compile(re.escape(label) + r"\s*:?\s*(.+)", re.I)
    for ln in lines:
        m = pat.match(ln.strip())
        if m:
            return m.group(1).strip()
    return None


def read_scan_record(path: Union[str, Path]) -> TomoScan:
    """Parse ``<prefix>_TomoFastScan.dat`` into a :class:`TomoScan`.

    The frame layout is **derived and then cross-checked**, never assumed. The
    header states the block sizes (``WF#``, ``DF#``, ``Proj#``) and,
    independently, the first and last image sequence numbers and the total
    count. Those two statements have to agree; when they do not, this raises
    rather than picking one, because the failure mode is an off-by-one block
    boundary that silently averages projections into the flat field.
    """
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()

    meta_line = next((l for l in lines if "tomo_metastr" in l), None)
    if meta_line is None:
        raise ValueError(
            f"{path} has no tomo_metastr line. That string is the only record "
            "of the pixel size, the propagation distance and the handedness, "
            "and the fallbacks are wrong (tomocupy_args.yml carries a "
            "different camera's pixel size). Refusing to guess."
        )
    meta = parse_metastr(meta_line)

    for req in ("pixel_size_um", "n_projections", "n_white", "n_dark",
                "omega_start_deg"):
        if req not in meta:
            raise ValueError(
                f"{path}: tomo_metastr does not state {req!r}. Parsed: "
                f"{sorted(k for k in meta if k != 'raw')}"
            )

    wf_start_s = _header_value(lines, "White field image sequence starts at")
    first_s = _header_value(lines, "First image sequence number")
    last_s = _header_value(lines, "Last image sequence number")
    count_s = _header_value(lines, "Number of images taken in this scan")
    if wf_start_s is None:
        raise ValueError(f"{path}: no 'White field image sequence starts at'")

    wf_start = int(float(wf_start_s))
    first = int(float(first_s)) if first_s else wf_start
    n_white = int(meta["n_white"])
    n_dark = int(meta["n_dark"])
    n_proj = int(meta["n_projections"])

    # How many non-projection frames are there? Measured from the totals, so
    # the front/back white question is answered by the file, not assumed.
    if count_s is not None:
        n_images = int(float(count_s))
    elif last_s is not None:
        n_images = int(float(last_s)) - first + 1
    else:
        raise ValueError(
            f"{path}: neither the total image count nor the last sequence "
            "number is recorded, so the frame layout cannot be cross-checked."
        )

    n_extra = n_images - n_proj
    if n_extra == n_white + n_dark:
        has_back = False
    elif n_extra == 2 * n_white + n_dark:
        has_back = True
    else:
        raise ValueError(
            f"{path}: the frame counts do not close. The record says "
            f"{n_images} images total and {n_proj} projections, leaving "
            f"{n_extra} for calibration, but WF#={n_white} and DF#={n_dark} "
            f"account for {n_white + n_dark} (no back white) or "
            f"{2 * n_white + n_dark} (with one). Do not guess a block "
            "boundary - a wrong one averages projections into the flat field."
        )

    cur = wf_start
    blocks = [FrameBlock("front_white", cur, n_white)]
    cur += n_white
    blocks.append(FrameBlock("projections", cur, n_proj))
    cur += n_proj
    if has_back:
        blocks.append(FrameBlock("back_white", cur, n_white))
        cur += n_white
    blocks.append(FrameBlock("dark", cur, n_dark))
    cur += n_dark

    if last_s is not None and cur - 1 != int(float(last_s)):
        raise ValueError(
            f"{path}: the derived layout ends at image {cur - 1} but the "
            f"record says the last image is {last_s}. One of the block sizes "
            "is wrong; refusing to guess which."
        )

    energy_s = _header_value(lines, "Energy (keV)")
    exposure_s = _header_value(lines, "Exposure time (s)")

    return TomoScan(
        source=path,
        path=_header_value(lines, "Path") or "",
        image_prefix=_header_value(lines, "Image prefix") or "",
        exposure_s=float(exposure_s) if exposure_s else float("nan"),
        energy_kev=float(energy_s) if energy_s else float("nan"),
        pixel_size_um=float(meta["pixel_size_um"]),
        propagation_mm=float(meta.get("propagation_mm", float("nan"))),
        detector=str(meta.get("detector", "")),
        magnification=str(meta.get("magnification", "")),
        rotation_axis=str(meta.get("rotation_axis", "")),
        handedness=str(meta.get("handedness", "")),
        omega_start_deg=float(meta["omega_start_deg"]),
        omega_end_deg=float(meta.get("omega_end_deg", float("nan"))),
        omega_step_deg=float(meta.get("omega_step_deg", float("nan"))),
        n_projections=n_proj, n_white=n_white, n_dark=n_dark,
        first_image=first, last_image=cur - 1, n_images=n_images,
        blocks=tuple(blocks),
        started=_header_value(lines, "Beginning of tomography scan at") or "",
        shift_motor=meta.get("shift_motor"),
        raw_metastr=meta_line.strip(),
    )
