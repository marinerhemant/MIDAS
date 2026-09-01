"""GE raw-binary frame layout: read it from the header, don't guess it.

Two defects motivate this file.

1. ``_read_ge`` inferred the side length by trying ``(2048, 4096, 1024, 512)``
   and taking the first that divided the payload.  That cannot be right for
   *any* ordering: 2048² is 4x 1024² is 16x 512², so every length that admits a
   2048² frame also admits the smaller ones.  A genuine 4-frame 512² file was
   read as 1 frame of 1024² — right byte count, wrong detector, no error, and
   every ring position downstream silently wrong.  "Most frames" is not a fix
   either; it inverts the bug and misreads every real 2048² panel as 512².
   The only fix is to read the dimensions the detector wrote.

2. ``except (ValueError, Exception)`` around the header attempt swallowed
   everything and re-raised whatever the *no-header* retry said, so the real
   reason a read failed never reached the user.
"""
from __future__ import annotations

import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from midas_calibrate_v2.io.readers import (
    GEFrameLayoutWarning,
    _ge_frame_layout,
    read_image,
)

REPO = Path(__file__).resolve().parents[3]


# ── file builders ────────────────────────────────────────────────────────────

def _adept_header(nrows, ncols, nframes, depth_bits=16,
                  std_bytes=6144, usr_bytes=2048):
    """A GE "ADEPT" binary header, field offsets per the GE format.

    Offsets are the ones fabio's GEimage.py walks and the ones the shipped
    ``gui/GEBad/BadImg.ge3`` actually carries (verified: ADEPT magic,
    std=6144, usr=2048, 2048x2048, 16 bit) — not invented here.
    """
    h = bytearray(std_bytes + usr_bytes)
    h[0:10] = b"ADEPT".ljust(10, b"\x00")
    struct.pack_into("<H", h, 10, 2)            # VersionOfStandardHeader
    struct.pack_into("<L", h, 12, std_bytes)    # StandardHeaderSizeInBytes
    struct.pack_into("<H", h, 16, 3)            # VersionOfUserHeader
    struct.pack_into("<L", h, 18, usr_bytes)    # UserHeaderSizeInBytes
    struct.pack_into("<H", h, 22, nframes)      # NumberOfFrames
    struct.pack_into("<H", h, 24, nrows)        # NumberOfRowsInFrame
    struct.pack_into("<H", h, 26, ncols)        # NumberOfColsInFrame
    struct.pack_into("<H", h, 28, depth_bits)   # ImageDepthInBits
    return bytes(h)


def _edf_header(nrows, ncols, nframes, nbytes=512):
    """The ASCII EDF block that a 1-ID ``<stem>_NNNNNN.edf.geN`` starts with.

    Dim_1 is the fast (column) axis, Dim_2 the slow (row) axis.
    """
    body = (
        "\n{"
        f"EDF_DataBlockID = 1.Image.Psd ; \r\n"
        f"EDF_HeaderSize = {nbytes:10d} ; \r\n"
        "ByteOrder = LowByteFirst ; \r\n"
        f"Num_Images = {nframes} ; \r\n"
        "DataType = UnsignedShort ; \r\n"
        f"Dim_1 = {ncols} ; \r\n"
        f"Dim_2 = {nrows} ; \r\n"
    )
    raw = body.encode("latin-1")
    assert len(raw) < nbytes - 2
    return raw + b" " * (nbytes - len(raw) - 2) + b"}\n"


def _write(path, header: bytes, stack: np.ndarray):
    with open(path, "wb") as f:
        f.write(header)
        np.ascontiguousarray(stack, dtype=np.uint16).tofile(f)
    return path


def _ramp(nframes, side):
    """Frames that differ, so a wrong frame count changes the reduced value."""
    return np.stack([np.full((side, side), 10 * (i + 1), dtype=np.uint16)
                     for i in range(nframes)])


# ── defect 1: the 4 x 512² case ──────────────────────────────────────────────

def test_adept_header_gives_four_512_frames_not_one_1024(tmp_path):
    """The headline regression: 4 x 512² is byte-for-byte 1 x 1024².

    With the header read, there is nothing to guess — 4 frames, 512², and the
    reduced image is the mean of the four distinct frame values.
    """
    stack = _ramp(4, 512)
    p = _write(tmp_path / "four_000001.ge3",
               _adept_header(512, 512, 4), stack)

    offset, nframes, nrows, ncols = _ge_frame_layout(p, 2)
    assert (nframes, nrows, ncols) == (4, 512, 512)
    assert offset == 8192

    img = read_image(p)
    assert img.shape == (512, 512)
    assert img[0, 0] == pytest.approx(np.mean([10, 20, 30, 40]))


def test_adept_header_wins_over_the_length_heuristic(tmp_path):
    """A 2048²-divisible length that the header says is 512².

    16 x 512² has exactly the length of 1 x 2048², so the old code called it
    one 2048² frame.  The header is the authority.
    """
    p = _write(tmp_path / "sixteen_000001.ge3",
               _adept_header(512, 512, 16), _ramp(16, 512))
    assert _ge_frame_layout(p, 2)[1:] == (16, 512, 512)


def test_non_square_header_is_honoured(tmp_path):
    """Nothing requires a GE frame to be square; the old side-length search
    could not express a non-square frame at all."""
    stack = np.stack([np.full((256, 1024), 7, dtype=np.uint16) for _ in range(3)])
    p = _write(tmp_path / "rect_000001.ge3",
               _adept_header(256, 1024, 3), stack)
    assert _ge_frame_layout(p, 2)[1:] == (3, 256, 1024)
    assert read_image(p).shape == (256, 1024)


def test_edf_text_header_is_read_not_guessed(tmp_path):
    """``.edf.geN`` carries an ASCII EDF block, not a GE binary header.

    Reading the GE binary offsets out of one yields plausible garbage
    (26465 x 11877 on the shipped CeO2 example), so the EDF flavour has to be
    recognised first.  Dim_2 x Dim_1 = rows x cols.
    """
    stack = np.stack([np.full((256, 128), 5, dtype=np.uint16) for _ in range(3)])
    p = _write(tmp_path / "cal_000099.edf.ge5",
               _edf_header(256, 128, 3), stack)
    assert _ge_frame_layout(p, 2) == (512, 3, 256, 128)
    assert read_image(p).shape == (256, 128)


def test_headerless_ambiguity_warns_and_frame_shape_settles_it(tmp_path):
    """Blank header + a length two shapes fit: unknowable, so say so.

    APS firmware after ~2018 writes 8192 zero bytes.  Then the length really
    is all there is and it really is ambiguous — the honest move is to warn
    and give the caller a way to answer.
    """
    p = _write(tmp_path / "blank_000001.ge3", b"\0" * 8192, _ramp(4, 512))

    with pytest.warns(GEFrameLayoutWarning, match="more than one frame shape"):
        _ge_frame_layout(p, 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GEFrameLayoutWarning)
        assert read_image(p).shape == (1024, 1024)          # the guess
    assert read_image(p, frame_shape=(512, 512)).shape == (512, 512)
    # and it is really 4 frames, not a reshape of one
    assert read_image(p, frame_shape=(512, 512))[0, 0] == pytest.approx(25.0)


def test_blank_header_2048_panel_does_not_warn(tmp_path):
    """A blanked 2048² panel frame must stay quiet.

    Its length also divides by 1024² and 512², but 2048 is a real GE panel and
    the smaller sides are not, so the hardware breaks the tie.  Warning here
    would fire on every APS dark file and teach users to ignore the warning
    that matters.
    """
    p = _write(tmp_path / "dark_000001.ge3", b"\0" * 8192,
               np.zeros((1, 2048, 2048), dtype=np.uint16))
    with warnings.catch_warnings():
        warnings.simplefilter("error", GEFrameLayoutWarning)
        assert _ge_frame_layout(p, 2)[1:] == (1, 2048, 2048)


def test_headerless_unambiguous_length_is_silent(tmp_path):
    """3 x 512² fits nothing else, so there is nothing to warn about."""
    p = _write(tmp_path / "three_000001.ge3", b"\0" * 8192, _ramp(3, 512))
    with warnings.catch_warnings():
        warnings.simplefilter("error", GEFrameLayoutWarning)
        assert _ge_frame_layout(p, 2)[1:] == (3, 512, 512)


def test_no_header_at_all_still_reads(tmp_path):
    """Early GE data has no header; the offset-0 path must survive."""
    p = _write(tmp_path / "raw_000001.ge3", b"", _ramp(3, 512))
    assert _ge_frame_layout(p, 2) == (0, 3, 512, 512)


def test_header_that_does_not_account_for_the_file_is_rejected(tmp_path):
    """A header claiming a shape the length contradicts is a misparse.

    Falling through to the heuristic is right; trusting it would be worse than
    guessing.
    """
    p = _write(tmp_path / "liar_000001.ge3",
               _adept_header(999, 777, 1), _ramp(3, 512))
    assert _ge_frame_layout(p, 2)[1:] == (3, 512, 512)


# ── defect 2: the swallowed exception ───────────────────────────────────────

def test_skip_frame_error_is_the_error_reported(tmp_path):
    """``except (ValueError, Exception)`` re-raised the offset-0 attempt.

    So asking for skip_frame=99 on a 3-frame file used to surface
    "can't reshape 790528 pixels into a square frame" — the header bytes
    counted as data at offset 0 — and never mentioned skip_frame.
    """
    p = _write(tmp_path / "three_000001.ge3", b"\0" * 8192, _ramp(3, 512))
    with pytest.raises(ValueError, match=r"skip_frame=99"):
        read_image(p, skip_frame=99)


def test_unreadable_length_names_both_attempts(tmp_path):
    """When nothing works, the message has to carry what was tried."""
    p = _write(tmp_path / "odd_000001.ge3", b"\0" * 8192,
               np.zeros(12345, dtype=np.uint16))
    with pytest.raises(ValueError) as exc:
        read_image(p)
    msg = str(exc.value)
    assert "odd_000001.ge3" in msg
    assert "tried sides" in msg


def test_a_header_misparse_is_reported_alongside_the_length_failure(tmp_path):
    """Both the header finding and the length failure, not one hiding the
    other."""
    p = _write(tmp_path / "liar_000001.ge3",
               _adept_header(999, 777, 1), np.zeros(12345, dtype=np.uint16))
    with pytest.raises(ValueError) as exc:
        read_image(p)
    assert "999x777" in str(exc.value)


def test_bad_reducer_is_still_the_callers_error(tmp_path):
    """A typo'd frame_reduce must not be reported as a layout problem."""
    p = _write(tmp_path / "three_000001.ge3", b"\0" * 8192, _ramp(3, 512))
    with pytest.raises(ValueError, match="frame_reduce"):
        read_image(p, frame_reduce="mediann")


def test_frame_shape_override_that_does_not_divide_is_refused(tmp_path):
    p = _write(tmp_path / "three_000001.ge3", b"\0" * 8192, _ramp(3, 512))
    with pytest.raises(ValueError, match="whole number of 300x300"):
        read_image(p, frame_shape=(300, 300))


# ── real files shipped with the repo ─────────────────────────────────────────

@pytest.mark.parametrize("rel,expect", [
    # blanked header, 5 x 2048² uint16
    ("FF_HEDM/Example/Calibration/dark_6s_000010.ge1", (8192, 5, 2048, 2048)),
    # ASCII EDF header: Dim_1=Dim_2=2048, Num_Images=5, EDF_HeaderSize=8192
    ("FF_HEDM/Example/Calibration/CeO2_1s_65pt351keV_1860mm_000007.edf.ge1",
     (8192, 5, 2048, 2048)),
    # genuine ADEPT header: std 6144 + usr 2048, 1 x 2048²
    ("gui/GEBad/BadImg.ge3", (8192, 1, 2048, 2048)),
])
def test_shipped_ge_files_resolve(rel, expect):
    """The three header flavours that actually occur, on real files."""
    p = REPO / rel
    if not p.is_file():
        pytest.skip(f"{rel} not present in this checkout")
    with warnings.catch_warnings():
        warnings.simplefilter("error", GEFrameLayoutWarning)
        assert _ge_frame_layout(p, 2) == expect


def test_declared_frame_count_disagreeing_with_the_length_warns(tmp_path):
    """A truncated acquisition: the header says 4 frames, 3 were written.

    The length is authoritative — it is what can actually be read — but
    averaging a short stack without saying so is how a half-finished exposure
    reaches a calibration as if it were complete.
    """
    p = _write(tmp_path / "short_000001.ge3",
               _adept_header(512, 512, 4), _ramp(3, 512))
    with pytest.warns(GEFrameLayoutWarning, match="declares 4 frames"):
        offset, nframes, nrows, ncols = _ge_frame_layout(p, 2)
    assert (nframes, nrows, ncols) == (3, 512, 512)


def test_matching_declared_frame_count_is_silent(tmp_path):
    p = _write(tmp_path / "ok_000001.ge3",
               _adept_header(512, 512, 3), _ramp(3, 512))
    with warnings.catch_warnings():
        warnings.simplefilter("error", GEFrameLayoutWarning)
        assert _ge_frame_layout(p, 2)[1:] == (3, 512, 512)
