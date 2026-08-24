"""Raw TIFF frames -> the reconstruction engine's binary layout.

What this replaces
------------------
Every NMC811 reconstruction in the tree starts from a hand-written
``prepare_data_<sample>.py`` whose frame boundaries were counted by eye::

    nDarks = 10; nWhites = 10; nImages = 3601
    startNrDark = 10938; startNrWhite = 7317
    startNrWhite2 = 10928; startNrData = 7327

One script per sample, four magic numbers each, and a wrong one averages
projections into the flat field without any error. All four come out of
:mod:`midas_tomo.scanrecord`, which reads them from the scan's own record and
cross-checks them against the recorded first/last image numbers.

The layout being written
------------------------
The C engine reads, in order::

    dark    float32   (ny, nx)      mean of the dark block
    white1  float32   (ny, nx)      mean of the front white block
    white2  float32   (ny, nx)      mean of the back white block
    data    uint16    (n, ny, nx)   the projections, unmodified

Two whites are always written because the engine always expects two. A scan
with no back white duplicates the front one, and ``provenance`` records that
it was duplicated rather than measured — otherwise a single-flat scan would
silently claim a flat-field drift measurement it never made.

Arithmetic parity
-----------------
The means are accumulated **sequentially in float32**, matching the hand
scripts. This is deliberate and not merely conservative: ``np.mean`` uses
pairwise summation, which is more accurate but gives different last bits, and
the point of the gate in ``test_ingest_parity.py`` is to reproduce an existing
``.raw`` byte for byte. Where the two disagree the existing file wins, because
reproducing history is the whole reason to have a parity mode.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .scanrecord import TomoScan

__all__ = ["IngestResult", "stage_scan_to_binary", "read_frame", "mean_block"]

log = logging.getLogger(__name__)

Crop = Tuple[int, int, int, int]      # row0, row1, col0, col1


def read_frame(path: Union[str, Path]) -> np.ndarray:
    """One TIFF frame as a 2-D array, in its stored dtype."""
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "reading TIFF frames needs tifffile: pip install tifffile"
        ) from exc
    arr = tifffile.imread(str(path))
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D frame, got shape {arr.shape}")
    return arr


def _crop(arr: np.ndarray, crop: Optional[Crop]) -> np.ndarray:
    if crop is None:
        return arr
    r0, r1, c0, c1 = crop
    if not (0 <= r0 < r1 <= arr.shape[0] and 0 <= c0 < c1 <= arr.shape[1]):
        raise ValueError(
            f"crop {crop} does not fit inside a {arr.shape} frame"
        )
    return arr[r0:r1, c0:c1]


def mean_block(
    paths: Sequence[Union[str, Path]], *, crop: Optional[Crop] = None
) -> np.ndarray:
    """Mean of a block of frames, as float32, summed sequentially.

    Sequential float32 accumulation rather than ``np.mean`` — see the module
    docstring. Cropping happens per frame, which is bit-identical to cropping
    the sum (the same additions in the same order) and much cheaper.
    """
    if not paths:
        raise ValueError("cannot average an empty block of frames")
    acc: Optional[np.ndarray] = None
    for p in paths:
        im = _crop(read_frame(p), crop).astype(np.float32)
        acc = im if acc is None else acc + im
    return (acc / np.float32(len(paths))).astype(np.float32)


@dataclass
class IngestResult:
    """What :func:`stage_scan_to_binary` wrote, and what to check about it."""

    out_bin: Path
    thetas_path: Optional[Path]
    n_projections: int
    ny: int
    nx: int
    crop: Optional[Crop]
    dark_mean: float
    white_mean: float
    n_saturated_white: int
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def contrast(self) -> float:
        """``white - dark``, the signal a flat-field correction divides by."""
        return self.white_mean - self.dark_mean

    def summary(self) -> str:
        return "\n".join([
            f"wrote {self.out_bin}",
            f"  frames     {self.n_projections} x {self.ny} x {self.nx} uint16",
            f"  crop       {self.crop if self.crop else 'none (full frame)'}",
            f"  dark mean  {self.dark_mean:.2f}",
            f"  white mean {self.white_mean:.2f}   (contrast {self.contrast:.2f})",
            f"  saturated white pixels: {self.n_saturated_white}",
        ])


def stage_scan_to_binary(
    scan: TomoScan,
    out_bin: Union[str, Path],
    *,
    root: Union[str, Path],
    crop: Optional[Crop] = None,
    thetas_path: Union[str, Path, None] = None,
    ext: str = ".tif",
    digits: int = 6,
    apply_aero_sign: bool = True,
    saturation: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> IngestResult:
    """Write ``dark, white1, white2, projections`` for the engine.

    ``root`` is the local directory holding the scan's image folder; the
    ``Path:`` inside the record is the acquisition machine's view and is
    usually not mounted where the analysis runs.

    ``crop`` is ``(row0, row1, col0, col1)`` and is **not** inferred. Cropping
    is a choice about which part of the specimen to reconstruct, and guessing
    it would silently change the field of view — the bt_1id_jun25b ``.raw`` files
    are a 128x128 crop of a much larger frame, chosen by hand.

    Also writes the angle file, with the aero sign applied (see
    :meth:`TomoScan.thetas`), unless ``thetas_path`` is False-y.
    """
    out_bin = Path(out_bin)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda m: None)

    def paths(role: str):
        return scan.frame_paths(role, root=root, ext=ext, digits=digits)

    say("averaging dark")
    dark = mean_block(paths("dark"), crop=crop)
    say("averaging front white")
    white1 = mean_block(paths("front_white"), crop=crop)
    if scan.has_back_white:
        say("averaging back white")
        white2 = mean_block(paths("back_white"), crop=crop)
        white2_source = "measured (back_white block)"
    else:
        white2 = white1
        white2_source = ("DUPLICATED from the front white - this scan has no "
                         "back white, so it measures no flat-field drift")

    dark_mean = float(dark.mean())
    white_mean = float(white1.mean())
    if white_mean <= dark_mean:
        raise ValueError(
            f"the white field ({white_mean:.2f}) is not brighter than the dark "
            f"field ({dark_mean:.2f}). The blocks are almost certainly swapped "
            "or mis-indexed; a flat-field correction from this would produce "
            "negative transmission, which is impossible."
        )

    n_sat = 0
    if saturation is not None:
        n_sat = int((white1 >= saturation).sum())
        if scan.has_back_white:
            n_sat += int((white2 >= saturation).sum())
        if n_sat:
            log.warning(
                "%d white-field pixels at or above the saturation level %d; "
                "the flat field is clipped there and those columns will not "
                "normalise", n_sat, saturation,
            )

    ny, nx = dark.shape
    proj_paths = paths("projections")
    with out_bin.open("wb") as f:
        dark.tofile(f)
        white1.tofile(f)
        white2.tofile(f)
        for i, p in enumerate(proj_paths):
            if i % 500 == 0:
                say(f"projection {i + 1}/{len(proj_paths)}")
            im = _crop(read_frame(p), crop).astype(np.uint16)
            if im.shape != (ny, nx):
                raise ValueError(
                    f"{p}: cropped to {im.shape} but the calibration frames "
                    f"are {(ny, nx)}. The frames are not all the same size."
                )
            im.tofile(f)

    tp: Optional[Path] = None
    if thetas_path is not False:
        tp = Path(thetas_path) if thetas_path else out_bin.with_name(
            out_bin.stem + "_thetas.txt"
        )
        th = scan.thetas(apply_aero_sign=apply_aero_sign)
        tp.write_text("\n".join(f"{v}" for v in th) + "\n")
        say(f"wrote {len(th)} angles to {tp.name}")

    prov = dict(scan.provenance())
    prov.update({
        "ingest_root": str(root),
        "crop": list(crop) if crop else None,
        "ny": ny, "nx": nx,
        "white2_source": white2_source,
        "mean_accumulation": "sequential float32 (matches the hand scripts; "
                             "np.mean would differ in the last bits)",
        "layout": "dark f32, white1 f32, white2 f32, projections uint16",
        "thetas_file": str(tp) if tp else None,
    })

    return IngestResult(
        out_bin=out_bin, thetas_path=tp, n_projections=len(proj_paths),
        ny=ny, nx=nx, crop=crop, dark_mean=dark_mean, white_mean=white_mean,
        n_saturated_white=n_sat, provenance=prov,
    )
