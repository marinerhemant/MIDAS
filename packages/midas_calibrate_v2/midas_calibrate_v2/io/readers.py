"""Image readers — TIFF, HDF5, GE binary, CBF.  File format auto-detected
from the extension; ``data_loc`` argument is for HDF5 dataset path.

Bad-pixel sentinels
-------------------
Detectors mark module gaps and dead pixels with an out-of-band value rather
than a flag.  The convention is not consistent between vendors:

- Pilatus writes **negative** values (``-1`` gap, ``-2`` overflow), so the
  familiar ``img[img < 0] = 0`` catches them.
- Dectris EIGER writes the **largest representable unsigned value**
  (``2**32-1`` for uint32).  That is the opposite end of the range, so every
  ``< 0`` guard fails open and a fitter is handed 4.29e9 as a photon count.

:func:`read_image` therefore detects the unsigned dtype-max sentinel by
default, zeroes those pixels, and warns.  Pass ``return_mask=True`` to get the
boolean mask back so it can be written out for ``midas-integrate-v2 --mask``.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np


class BadPixelSentinelWarning(UserWarning):
    """Raised when a frame carries out-of-band bad-pixel values.

    Not an error: the pixels are zeroed and reported.  It fires so that a
    sentinel is never silently averaged into a profile or a ring centroid.
    """


def _split_sentinel(arr: np.ndarray, bad_value):
    """Return ``(arr_with_sentinels_zeroed, mask)``.

    ``mask`` is a 2-D boolean array (``True`` = bad) or ``None`` when nothing
    was flagged.  For a 3-D stack the mask is the union over frames, so a pixel
    bad in any frame is bad in the result — the conservative choice, and it
    keeps the mask meaningful after the frames are averaged.

    Runs on the raw integer data **before** any averaging.  Averaging first
    would blend the sentinel with real counts into a value that no longer
    equals the sentinel and could not be detected at all.
    """
    if bad_value is None:
        return arr, None
    if bad_value == "auto":
        if arr.dtype.kind != "u":
            return arr, None            # signed / float: the < 0 convention
        sentinel = np.iinfo(arr.dtype).max
    else:
        sentinel = bad_value

    hit = arr == sentinel
    if not hit.any():
        return arr, None
    mask = hit.any(axis=0) if hit.ndim == 3 else hit
    arr = np.where(hit, 0, arr)
    return arr, mask


def _reducer(frame_reduce: str):
    """Frame-stack reducer by name.

    Not silently defaulted: a typo'd name would otherwise average when the
    caller asked for a median, and the two differ exactly where it matters --
    on a zinger.
    """
    try:
        return {"mean": np.mean, "median": np.median}[frame_reduce]
    except KeyError:
        raise ValueError(
            f"frame_reduce must be 'mean' or 'median'; got {frame_reduce!r}"
        ) from None


def read_image(
    path: str | Path,
    *,
    data_loc: str = "exchange/data",
    skip_frame: int = 0,
    im_trans: tuple = (),
    data_type: int = 1,
    bad_value="auto",
    return_mask: bool = False,
    frame_reduce: str = "mean",
    frame_shape: tuple[int, int] | None = None,
):
    """Read a 2-D image from any supported format.

    Parameters
    ----------
    path : file path.  Extension determines the reader.
    data_loc : HDF5 dataset path (only for .h5 / .hdf5).
    skip_frame : number of leading frames to skip in multi-frame files.
    im_trans : tuple of MIDAS image-transformation codes
        (1 = flip Y, 2 = flip Z, 3 = transpose).
    data_type : raw-binary numeric type (only for GE-style files):
        1 = uint16, 2 = float64, 3 = float32, 4 = uint32, 5 = int32.
    bad_value : ``"auto"`` (default) flags the largest representable value of
        an unsigned integer frame as a bad-pixel sentinel — the EIGER / Dectris
        convention.  Pass a number to name the sentinel explicitly (e.g. ``-1``
        for a Pilatus gap), or ``None`` to disable the check entirely and get
        the raw values.
    return_mask : also return the boolean bad-pixel mask.
    frame_reduce : how a multi-frame file collapses to one image, ``"mean"``
        (default, and what every caller got before this was configurable) or
        ``"median"``.  Use ``"median"`` for a calibrant exposure: it rejects
        zingers, which a mean smears across the ring they land on.  Ignored
        for single-frame formats.
    frame_shape : ``(nrows, ncols)`` override for GE-style raw binaries only.
        Normally unnecessary — the shape is read from the file's header.  Needed
        only for a header-blanked file whose length fits more than one shape
        (a 4-frame 512² file is byte-for-byte as long as a 1-frame 1024² one);
        that case warns with :class:`GEFrameLayoutWarning` and this is the
        answer to it.

    Returns
    -------
    np.ndarray of shape (nz, ny), float64 — or ``(image, mask)`` when
    ``return_mask`` is set.  ``mask`` is ``True`` at bad pixels, and is all
    ``False`` when nothing was flagged.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".tif", ".tiff"):
        img, mask = _read_tiff(p, bad_value=bad_value)
    elif ext in (".h5", ".hdf5", ".hdf", ".nxs"):
        img, mask = _read_hdf5(p, data_loc=data_loc, skip_frame=skip_frame,
                               bad_value=bad_value, frame_reduce=frame_reduce)
    elif ext == ".cbf":
        img, mask = _read_cbf(p, bad_value=bad_value)
    elif ".ge" in p.name.lower():
        img, mask = _read_ge(p, data_type=data_type, skip_frame=skip_frame,
                             bad_value=bad_value, frame_reduce=frame_reduce,
                             frame_shape=frame_shape)
    else:
        raise ValueError(f"Unrecognised image format: {p}")

    if mask is not None:
        warnings.warn(
            f"{p.name}: {int(mask.sum())} of {mask.size} pixels "
            f"({100.0 * mask.mean():.3f} %) carry the bad-pixel sentinel; they "
            f"have been set to 0. Pass them to the integrator as a mask "
            f"(read_image(..., return_mask=True)) — they are not counts.",
            BadPixelSentinelWarning,
            stacklevel=2,
        )

    # the mask has to ride along through the same flips, or it stops lining up
    for opt in im_trans:
        if opt == 1:
            img = img[:, ::-1]
            mask = None if mask is None else mask[:, ::-1]
        elif opt == 2:
            img = img[::-1, :]
            mask = None if mask is None else mask[::-1, :]
        elif opt == 3:
            img = img.T
            mask = None if mask is None else mask.T

    img = np.ascontiguousarray(img.astype(np.float64))
    if not return_mask:
        return img
    if mask is None:
        mask = np.zeros(img.shape, dtype=bool)
    return img, np.ascontiguousarray(mask)


def read_dark(path: str | Path | None, **kwargs) -> Optional[np.ndarray]:
    """Same as :func:`read_image` but returns ``None`` if path is None or empty."""
    if path is None or str(path) == "":
        return None
    return read_image(path, **kwargs)


# ----------------------------------------------------------- backends

def _read_tiff(path: Path, *, bad_value="auto"):
    try:
        import tifffile
        raw = np.asarray(tifffile.imread(str(path)))
    except ImportError:
        from PIL import Image
        raw = np.asarray(Image.open(str(path)))
    raw, mask = _split_sentinel(raw, bad_value)
    return raw.astype(np.float64), mask


def _read_hdf5(path: Path, *, data_loc: str, skip_frame: int, bad_value="auto",
               frame_reduce: str = "mean"):
    import h5py
    with h5py.File(str(path), "r") as f:
        dset = f[data_loc]
        data = dset[skip_frame:] if dset.ndim >= 3 else dset[...]
    data, mask = _split_sentinel(data, bad_value)
    if data.ndim == 3:
        return _reducer(frame_reduce)(data, axis=0).astype(np.float64), mask
    return data.astype(np.float64), mask


# --------------------------------------------------------- GE frame layout
#
# The frame shape must come from the file, not from a guess.  A GE binary is
# just a header followed by contiguous frames, so the file LENGTH alone cannot
# identify the shape: 2048² is an exact multiple of 1024², which is an exact
# multiple of 512², so *every* length that admits a 2048² frame also admits
# 1024² and 512² frames.  Ranking candidate sides — by size, by frame count,
# by anything — therefore cannot be made correct; the only real fix is to read
# the dimensions the detector wrote down.
#
# Three header flavours actually occur in APS data:
#
#   1. GE "ADEPT" binary header.  10-byte ImageFormat magic, then the field
#      table below (little-endian, byte offsets fixed by the format).  Carries
#      rows, columns, bit depth, frame count and its own length.
#   2. EDF text header.  Files named ``<stem>_NNNNNN.edf.geN`` (the 1-ID
#      norm) start with an ASCII EDF block — ``Dim_1``, ``Dim_2``,
#      ``EDF_HeaderSize``, ``Num_Images`` — NOT a GE binary header.  Reading
#      the GE field offsets out of one yields plausible-looking garbage
#      (26465 x 11877 on the shipped CeO2 example), so the EDF case has to be
#      recognised before the binary offsets are ever applied.
#   3. Blanked header.  APS firmware after ~2018 writes 8192 zero bytes.
#      Nothing is recoverable; the length heuristic is all there is, and it is
#      ambiguous whenever more than one side divides the payload.

_GE_DEFAULT_HEADER_BYTES = 8192

# ADEPT header field offsets, little-endian.  Same table fabio's GEimage.py
# walks; only the fields needed for the layout are listed.
_GE_OFF_IMAGE_FORMAT = (0, 10)     # str
_GE_OFF_STD_HDR_SIZE = 12          # uint32, bytes of standard header
_GE_OFF_USR_HDR_SIZE = 18          # uint32, bytes of user header
_GE_OFF_N_FRAMES = 22              # uint16
_GE_OFF_N_ROWS = 24                # uint16
_GE_OFF_N_COLS = 26                # uint16
_GE_OFF_DEPTH_BITS = 28            # uint16

# Sides tried only when no header survives, largest first.
_GE_FALLBACK_SIDES = (4096, 2048, 1024, 512)

# Sides a real GE panel actually has.  When the payload divides by one of
# these the ambiguity is resolved by the hardware, not by our ranking, so it
# is not reported: an 8 MB/frame file with a blanked header came off a GE
# panel, and every APS dark frame would otherwise warn.  A file that only
# fits the smaller sides did not come off a GE panel, so nothing external
# breaks the tie and the caller has to be told.
_GE_PANEL_SIDES = frozenset({4096, 2048})


class GEFrameLayoutWarning(UserWarning):
    """The GE frame shape could not be read from the file and was guessed.

    Fires only when the guess is genuinely ambiguous — more than one square
    side divides the payload — because that is exactly when it can be wrong
    and nothing downstream would notice: the wrong shape still reads, still
    integrates, and still fits rings, just to a different geometry.
    """


def _ge_header_from_edf(head: bytes):
    """``(header_bytes, nrows, ncols, declared_frames)`` from a leading EDF
    text block, or None.  ``declared_frames`` is None when unstated.

    ``.edf.geN`` is an EDF header stapled to GE frames.  ``Dim_1`` is the fast
    (column) axis and ``Dim_2`` the slow (row) axis, so the frame is
    ``(Dim_2, Dim_1)`` — they happen to be equal on a square GE panel, but the
    order is not free to choose.
    """
    if head.lstrip(b"\r\n \t")[:1] != b"{":
        return None
    text = head.decode("latin-1", errors="replace")

    def _field(name):
        m = re.search(rf"\b{name}\s*=\s*(\d+)", text)
        return int(m.group(1)) if m else None

    dim1, dim2 = _field("Dim_1"), _field("Dim_2")
    if not dim1 or not dim2:
        return None
    nbytes = _field("EDF_HeaderSize")
    if nbytes is None:
        # No self-declared length: an EDF block is terminated by '}' and
        # padded to a multiple of 512, so round the terminator up.
        end = text.find("}")
        if end < 0:
            return None
        nbytes = -(-(end + 1) // 512) * 512
    return nbytes, dim2, dim1, _field("Num_Images")


def _ge_header_from_adept(head: bytes):
    """``(header_bytes, nrows, ncols, declared_frames)`` from a GE binary
    header, or None.

    None means "no usable header": the blanked all-zero header APS writes, a
    truncated file, or fields that do not describe this file's length.
    """
    if len(head) < _GE_OFF_DEPTH_BITS + 2:
        return None
    fmt = head[slice(*_GE_OFF_IMAGE_FORMAT)]
    if not fmt.strip(b"\x00"):
        return None                       # blanked header — nothing to read
    u16 = lambda o: int.from_bytes(head[o:o + 2], "little")
    u32 = lambda o: int.from_bytes(head[o:o + 4], "little")
    nrows, ncols = u16(_GE_OFF_N_ROWS), u16(_GE_OFF_N_COLS)
    depth = u16(_GE_OFF_DEPTH_BITS)
    nbytes = u32(_GE_OFF_STD_HDR_SIZE) + u32(_GE_OFF_USR_HDR_SIZE)
    # Every one of these is a guard against reading the offsets out of a file
    # that is not an ADEPT header at all (see the EDF case): garbage passes
    # the "is it non-zero" test but not this.
    if not (0 < nrows <= 65535 and 0 < ncols <= 65535):
        return None
    if depth not in (8, 16, 32, 64):
        return None
    if not (0 < nbytes <= 1 << 20):
        return None
    return nbytes, nrows, ncols, u16(_GE_OFF_N_FRAMES) or None


def _ge_frame_layout(path: Path, itemsize: int):
    """Resolve ``(offset_bytes, n_frames, nrows, ncols)`` for a GE binary.

    Header first, length heuristic only if there is no header.  Raises
    ValueError when neither can produce a layout that accounts for the file
    exactly — silence there is what let a 4 x 512² file read as 1 x 1024².
    """
    size = path.stat().st_size
    with open(path, "rb") as fh:
        head = fh.read(65536)

    tried = []
    for parse in (_ge_header_from_edf, _ge_header_from_adept):
        got = parse(head)
        if got is None:
            continue
        nbytes, nrows, ncols, declared = got
        frame = nrows * ncols * itemsize
        payload = size - nbytes
        # A header that does not account for the file is a misparse, not a
        # truth: fall through to the next flavour rather than trust it.
        if frame > 0 and payload > 0 and payload % frame == 0:
            n = payload // frame
            # The length is authoritative -- it is what we can actually read --
            # but a disagreement means the acquisition did not finish writing,
            # and silently averaging a short stack is how that goes unnoticed.
            if declared and declared != n:
                warnings.warn(
                    f"{path.name}: header declares {declared} frames but the "
                    f"file holds {n}. Reading {n}; the acquisition was "
                    f"probably truncated.",
                    GEFrameLayoutWarning, stacklevel=4)
            return nbytes, n, nrows, ncols
        tried.append(f"{parse.__name__} gave {nrows}x{ncols} after {nbytes} B, "
                     f"which does not divide {size} B")

    # No header. Enumerate every square side the payload admits, at both the
    # standard header offset and none, and only then decide.
    cands = []
    for offset in (_GE_DEFAULT_HEADER_BYTES, 0):
        payload = size - offset
        if payload <= 0:
            continue
        for side in _GE_FALLBACK_SIDES:
            frame = side * side * itemsize
            if payload >= frame and payload % frame == 0:
                cands.append((offset, payload // frame, side, side))
        if cands:
            break                          # do not mix header/no-header reads

    if not cands:
        raise ValueError(
            f"{path.name}: {size} B carries no GE header and is not a whole "
            f"number of square {itemsize}-byte frames "
            f"(tried sides {_GE_FALLBACK_SIDES})"
            + ("; " + "; ".join(tried) if tried else ""))

    chosen = cands[0]                      # largest side — see module comment
    if len(cands) > 1 and chosen[2] not in _GE_PANEL_SIDES:
        warnings.warn(
            f"{path.name}: no readable GE header, and the file length is "
            f"consistent with more than one frame shape "
            + ", ".join(f"{n}x{r}x{c}" for _, n, r, c in cands)
            + f". Reading it as {chosen[1]}x{chosen[2]}x{chosen[3]}. Pass "
              f"frame_shape=(nrows, ncols) to read_image if that is wrong — "
              f"a wrong shape does not fail, it just produces a different "
              f"detector.",
            GEFrameLayoutWarning, stacklevel=4)
    return chosen


def _read_ge(path: Path, *, data_type: int = 1, skip_frame: int = 0,
             bad_value="auto", frame_reduce: str = "mean",
             frame_shape: tuple[int, int] | None = None):
    """GE binary frame reader.

    Takes the frame shape from the file's own header (GE ADEPT binary, or the
    EDF text block on a ``.edf.geN``) and only guesses from the file length
    when the header is blank — see the module-level note on why the length
    alone cannot decide.  ``frame_shape=(nrows, ncols)`` overrides both, for
    the header-less file whose length is genuinely ambiguous.
    """
    dtype_map = {1: np.uint16, 2: np.float64, 3: np.float32,
                 4: np.uint32, 5: np.int32}
    np_dtype = dtype_map.get(data_type, np.uint16)
    itemsize = np.dtype(np_dtype).itemsize
    # Resolved before any file I/O: a bad reducer name is the caller's error
    # and must not be reported as a frame-layout problem.
    red = _reducer(frame_reduce)

    if frame_shape is not None:
        nrows, ncols = (int(v) for v in frame_shape)
        size = path.stat().st_size
        frame = nrows * ncols * itemsize
        for offset in (_GE_DEFAULT_HEADER_BYTES, 0):
            payload = size - offset
            if payload > 0 and payload % frame == 0:
                break
        else:
            raise ValueError(
                f"{path.name}: {size} B is not a whole number of "
                f"{nrows}x{ncols} {itemsize}-byte frames, with or without an "
                f"{_GE_DEFAULT_HEADER_BYTES} B header")
        n = payload // frame
    else:
        offset, n, nrows, ncols = _ge_frame_layout(path, itemsize)

    if skip_frame >= n:
        raise ValueError(f"skip_frame={skip_frame} ≥ {n} frames")
    arr = np.fromfile(str(path), dtype=np_dtype, offset=offset,
                      count=n * nrows * ncols).reshape(n, nrows, ncols)
    stack, mask = _split_sentinel(arr[skip_frame:], bad_value)
    return red(stack, axis=0).astype(np.float64), mask


def _read_cbf(path: Path, *, bad_value="auto"):
    """CBF reader via MIDAS's own ``midas_zipper`` reader.

    CBF is the Crystallographic Binary File format used by Pilatus / Eiger /
    Varex.  We deliberately do **not** use ``fabio`` here: fabio returns the
    raw frame, whereas the entire MIDAS pipeline (``midas_zipper`` →
    ``midas_peakfit`` → indexer) works in the transposed/double-flipped MIDAS
    convention (``pixels.reshape(nrows, ncols).T[::-1, ::-1]``).  Reading with
    fabio yields a Y↔Z-transposed, both-axes-flipped beam centre and MIRRORED
    tilts/distortion that are invalid for the pipeline.  Using the same reader
    the pipeline uses guarantees the calibration geometry is consistent with
    downstream indexing.
    """
    try:
        from midas_zipper._read_cbf import read_cbf
    except ImportError as exc:  # pragma: no cover - dependency declared in pyproject
        raise RuntimeError(
            "CBF reading requires the `midas-zipper` package (a declared "
            "midas-calibrate-v2 dependency). Install with "
            "`pip install midas-zipper`."
        ) from exc
    _header, data = read_cbf(str(path), check_md5=False)
    data, mask = _split_sentinel(data, bad_value)
    return data.astype(np.float64), mask


__all__ = ["read_image", "read_dark", "BadPixelSentinelWarning",
           "GEFrameLayoutWarning"]
