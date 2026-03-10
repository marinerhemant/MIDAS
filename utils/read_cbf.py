#!/usr/bin/env python
"""
Minimal but complete CBF (Crystallographic Binary File) reader.

Reads CBF files that use x-CBF_BYTE_OFFSET compression, such as those
produced by Dectris Pilatus and Varex detectors.

Parses:
  - Full ASCII CIF header (key-value pairs, loop_ blocks)
  - Binary-section metadata (dimensions, dtype, MD5, etc.)
  - Pilatus 1.2 sub-headers (detector, pixel size, wavelength, angles …)

Based on the fabio cbfimage.py reader (silx-kit/fabio, MIT license).

Usage:
    from read_cbf import read_cbf
    header, data = read_cbf("image.cbf")
"""

import re
import hashlib
import base64
import logging
import numpy as np

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────

STARTER = b"\x0c\x1a\x04\xd5"                       # 4-byte magic
BINARY_SECTION_MARKER = b"--CIF-BINARY-FORMAT-SECTION--"

DATA_TYPES = {
    "signed 8-bit integer":    np.int8,
    "signed 16-bit integer":   np.int16,
    "signed 32-bit integer":   np.int32,
    "signed 64-bit integer":   np.int64,
    "unsigned 8-bit integer":  np.uint8,
    "unsigned 16-bit integer": np.uint16,
    "unsigned 32-bit integer": np.uint32,
    "unsigned 64-bit integer": np.uint64,
}

# Minimum required binary-header keys (warn if missing)
MINIMUM_KEYS = [
    "X-Binary-Size-Fastest-Dimension",
    "X-Binary-Size-Second-Dimension",
    "X-Binary-Size",
    "X-Binary-Number-of-Elements",
    "X-Binary-Element-Type",
]

# Pilatus 1.2 header keywords we know how to parse
# key → (value_word_indices, type_converters)
PILATUS_KEYWORDS = {
    "Detector":           (slice(1, None), str),
    "Pixel_size":         ([1, 4],         [float, float]),
    "Exposure_time":      ([1],            [float]),
    "Exposure_period":    ([1],            [float]),
    "Tau":                ([1],            [float]),
    "Count_cutoff":       ([1],            [int]),
    "Threshold_setting":  ([1],            [float]),
    "Gain_setting":       ([1, 2],         [str, str]),
    "N_excluded_pixels":  ([1],            [int]),
    "Excluded_pixels":    ([1],            [str]),
    "Flat_field":         ([1],            [str]),
    "Trim_file":          ([1],            [str]),
    "Image_path":         ([1],            [str]),
    "Wavelength":         ([1],            [float]),
    "Energy_range":       ([1, 2],         [float, float]),
    "Detector_distance":  ([1],            [float]),
    "Detector_Voffset":   ([1],            [float]),
    "Beam_xy":            ([1, 2],         [float, float]),
    "Flux":               ([1],            [float]),
    "Filter_transmission":([1],            [float]),
    "Start_angle":        ([1],            [float]),
    "Angle_increment":    ([1],            [float]),
    "Detector_2theta":    ([1],            [float]),
    "Polarization":       ([1],            [float]),
    "Alpha":              ([1],            [float]),
    "Kappa":              ([1],            [float]),
    "Phi":                ([1],            [float]),
    "Phi_increment":      ([1],            [float]),
    "Chi":                ([1],            [float]),
    "Chi_increment":      ([1],            [float]),
    "Omega":              ([1],            [float]),
    "Omega_increment":    ([1],            [float]),
    "Oscillation_axis":   ([1],            [str]),
    "N_oscillations":     ([1],            [int]),
    "Start_position":     ([1],            [float]),
    "Position_increment": ([1],            [float]),
    "Shutter_time":       ([1],            [float]),
}


# ─── Byte-offset decoder (Numba-accelerated) ─────────────────────────────

def _make_decoder():
    """Build the fastest available byte-offset decoder."""

    def _core_decode(raw_u8, out):
        """Decode x-CBF_BYTE_OFFSET into a pre-allocated output array.

        Writing directly into ``out`` (which can be any integer dtype)
        avoids a separate int64→target-dtype copy after decoding.

        Byte-offset encoding (little-endian):
          1. Read 1 byte as signed int8 diff.
          2. If -128  → read next 2 bytes as int16.
          3. If -32768 → read next 4 bytes as int32.
          4. If -2147483648 → read next 8 bytes as int64.
          5. Accumulate diffs → absolute pixel value.
        """
        n_elements = out.shape[0]
        pos = 0
        val = np.int64(0)

        for i in range(n_elements):
            b = raw_u8[pos]
            diff = np.int64(b if b < 128 else b - 256)
            pos += 1

            if diff == -128:
                lo = np.int64(raw_u8[pos])
                hi = np.int64(raw_u8[pos + 1])
                diff = lo | (hi << 8)
                if diff >= 32768:
                    diff -= 65536
                pos += 2

                if diff == -32768:
                    b0 = np.int64(raw_u8[pos])
                    b1 = np.int64(raw_u8[pos + 1])
                    b2 = np.int64(raw_u8[pos + 2])
                    b3 = np.int64(raw_u8[pos + 3])
                    diff = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                    if diff >= 2147483648:
                        diff -= 4294967296
                    pos += 4

                    if diff == -2147483648:
                        d = np.int64(0)
                        for shift in range(8):
                            d |= np.int64(raw_u8[pos + shift]) << np.int64(shift * 8)
                        diff = d
                        pos += 8

            val += diff
            out[i] = val

        return out

    if HAS_NUMBA:
        return numba.njit(cache=True)(_core_decode)
    return _core_decode


_decode_impl = _make_decoder()


def _decode_byte_offset(raw: bytes, n_elements: int,
                        dtype: np.dtype = np.int64) -> np.ndarray:
    """Decode x-CBF_BYTE_OFFSET compressed data.

    Returns a 1-D array of the requested *dtype*, decoded in a single pass
    (no intermediate int64 buffer when the target dtype is narrower).
    """
    out = np.empty(n_elements, dtype=dtype)
    return _decode_impl(np.frombuffer(raw, dtype=np.uint8), out)


# ─── CIF header parsing ──────────────────────────────────────────────────

def _parse_cif_header(header_bytes: bytes) -> dict:
    """Parse the ASCII CIF portion of the file into a dict.

    Handles:
      - Simple ``_key  value`` pairs
      - Multi-line values delimited by semicolons (``;`` … ``;``)
      - ``loop_`` blocks (stored under the ``"loop_"`` key)
      - Comments (lines starting with ``#``, stripped)
    """
    cif = {}
    text = header_bytes.decode("ascii", errors="replace")
    lines = text.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip blanks and comments
        if not line or line.startswith("#"):
            i += 1
            continue

        # data_ block header — skip the line itself
        if line.lower().startswith("data_"):
            i += 1
            continue

        # loop_ block
        if line.lower().startswith("loop_"):
            i += 1
            keys = []
            while i < len(lines) and lines[i].strip().startswith("_"):
                keys.append(lines[i].strip())
                i += 1
            loop_data = []
            while i < len(lines):
                l = lines[i].strip()
                if not l or l.startswith("_") or l.lower().startswith("loop_") or l.lower().startswith("data_"):
                    break
                loop_data.append(l)
                i += 1
            cif.setdefault("loop_", []).append((keys, loop_data))
            continue

        # _key value pair
        if line.startswith("_"):
            parts = line.split(None, 1)
            key = parts[0]
            if len(parts) == 2:
                val = parts[1].strip().strip("'\"")
                cif[key] = val
            else:
                # Value may be on the next line, or a semicolon-delimited block
                i += 1
                if i < len(lines):
                    nxt = lines[i].strip()
                    if nxt == ";":
                        # Multi-line value between ; ... ;
                        i += 1
                        val_lines = []
                        while i < len(lines) and lines[i].strip() != ";":
                            val_lines.append(lines[i])
                            i += 1
                        cif[key] = "\n".join(val_lines)
                        i += 1  # skip closing ;
                    else:
                        cif[key] = nxt.strip("'\"")
                continue

            i += 1
            continue

        # Semicolon-delimited value (orphan — shouldn't normally happen)
        if line == ";":
            i += 1
            val_lines = []
            while i < len(lines) and lines[i].strip() != ";":
                val_lines.append(lines[i])
                i += 1
            i += 1  # skip closing ;
            continue

        i += 1

    return cif


def _parse_pilatus_header(content: str) -> dict:
    """Parse Pilatus 1.2 sub-header lines (``# Key value …``)."""
    result = {}
    # Clean up special chars that act as whitespace in Pilatus headers
    for ch in "()#:=,":
        content = content.replace(ch, " ")

    for line in content.splitlines():
        words = line.split()
        if not words:
            continue
        for keyword, (indices, types) in PILATUS_KEYWORDS.items():
            if words[0] == keyword:
                try:
                    if isinstance(indices, slice):
                        result[keyword] = " ".join(str(types(w)) for w in words[indices])
                    elif isinstance(types, list):
                        if len(indices) == 1:
                            result[keyword] = types[0](words[indices[0]])
                        else:
                            result[keyword] = tuple(
                                t(words[j]) for t, j in zip(types, indices)
                            )
                    else:
                        result[keyword] = types(words[indices[0]])
                except (IndexError, ValueError):
                    pass
                break
    return result


# ─── Binary-section header parsing ───────────────────────────────────────

def _parse_binary_header(header_bytes: bytes) -> dict:
    """Parse key: value lines between the binary-section marker and the
    4-byte starter magic.  Handles continuation lines (e.g. Content-Type
    spanning two lines with conversions= on the next)."""
    info = {}
    last_key = None
    for line in header_bytes.split(b"\n"):
        line = line.strip()
        if not line or len(line) < 3:
            continue

        if b":" in line:
            key, _, val = line.partition(b":")
            key_str = key.strip().decode("ascii")
            val_str = val.strip(b' "\t\r\n').decode("ascii")
            info[key_str] = val_str
            last_key = key_str
        elif b"=" in line:
            key, _, val = line.partition(b"=")
            info[key.strip().decode("ascii")] = val.strip(b' "\t\r\n').decode("ascii")
        else:
            # Continuation of previous line (e.g. 'conversions="..."')
            stripped = line.strip()
            if b"=" in stripped:
                key, _, val = stripped.partition(b"=")
                info[key.strip().decode("ascii")] = val.strip(b' "\t\r\n').decode("ascii")
            elif last_key is not None:
                # Append to previous value
                info[last_key] += " " + stripped.decode("ascii", errors="replace")

    # Validate minimum keys
    missing = [k for k in MINIMUM_KEYS if k not in info]
    if missing:
        logger.warning("Mandatory keys missing in CBF binary header: %s",
                        ", ".join(missing))

    return info


# ─── Public API ───────────────────────────────────────────────────────────

def read_cbf(filename: str, check_md5: bool = True):
    """Read a CBF file and return (header_dict, 2d_numpy_array).

    Parameters
    ----------
    filename : str
        Path to the .cbf file.
    check_md5 : bool
        If True, verify the MD5 checksum of the binary payload (if present).

    Returns
    -------
    header : dict
        Merged CIF + binary-section metadata.  CIF keys start with ``_``,
        binary-section keys start with ``X-Binary-`` or ``Content-``.
        If the file uses Pilatus 1.2 convention, a ``"pilatus"`` sub-dict
        is also included.
    data : np.ndarray
        2-D image array with shape (rows, cols).
    """
    with open(filename, "rb") as f:
        raw_file = f.read()

    # ── Locate the binary section ──
    marker_pos = raw_file.find(BINARY_SECTION_MARKER)
    if marker_pos < 0:
        raise ValueError("No CIF binary section marker found in file")

    starter_pos = raw_file.find(STARTER, marker_pos)
    if starter_pos < 0:
        raise ValueError("No binary data starter magic found in file")

    # ── Parse the ASCII CIF header (everything before the marker) ──
    # Walk backward from marker to find the enclosing semicolon delimiter
    cif_end = raw_file.rfind(b";", 0, marker_pos)
    if cif_end < 0:
        cif_end = marker_pos
    cif_header = _parse_cif_header(raw_file[:cif_end])

    # ── Parse the binary-section sub-header ──
    bin_header_bytes = raw_file[marker_pos + len(BINARY_SECTION_MARKER): starter_pos]
    bin_header = _parse_binary_header(bin_header_bytes)

    # ── Merge headers (CIF first, binary-section on top) ──
    header = {}
    for k, v in cif_header.items():
        if k == "loop_":
            continue  # keep loops separate if needed
        header[k] = v
    header.update(bin_header)

    # ── Parse Pilatus sub-header if present ──
    if cif_header.get("_array_data.header_convention", "").strip("'\" ") == "PILATUS_1.2":
        contents = cif_header.get("_array_data.header_contents", "")
        if contents:
            header["pilatus"] = _parse_pilatus_header(contents)

    # ── Extract image parameters ──
    ncols = int(header["X-Binary-Size-Fastest-Dimension"])
    nrows = int(header["X-Binary-Size-Second-Dimension"])
    n_elements = int(header["X-Binary-Number-of-Elements"])
    binary_size = int(header["X-Binary-Size"])

    type_str = header.get("X-Binary-Element-Type", "signed 32-bit integer")
    dtype = DATA_TYPES.get(type_str, np.int32)

    if n_elements != nrows * ncols:
        raise ValueError(
            f"Element count mismatch: {n_elements} != {nrows}*{ncols}"
        )

    # ── Extract the compressed payload ──
    data_start = starter_pos + len(STARTER)
    compressed = raw_file[data_start: data_start + binary_size]

    # ── Optional MD5 check ──
    if check_md5 and "Content-MD5" in header:
        expected = header["Content-MD5"]
        computed = base64.b64encode(hashlib.md5(compressed).digest()).decode("ascii")
        if expected != computed:
            logger.warning("MD5 mismatch — expected %s, got %s", expected, computed)

    # ── Decompress & reshape (single-pass, no intermediate copy) ──
    pixels = _decode_byte_offset(compressed, n_elements, dtype=dtype)
    data = pixels.reshape(nrows, ncols)

    return header, data


def warmup():
    """Pre-compile the Numba JIT decoder with a tiny synthetic buffer.

    Call this once before batch operations to ensure the first real
    ``read_cbf()`` call doesn't pay the 1-2 s JIT compilation cost.
    If the cache already exists on disk, this returns almost instantly.
    """
    tiny = np.array([0, 1, 2, 3], dtype=np.uint8)
    out = np.empty(4, dtype=np.int64)
    _decode_impl(tiny, out)


def read_cbf_metadata(filename: str) -> dict:
    """Read only the header from a CBF file (no pixel decompression).

    This is much faster than ``read_cbf()`` (~1 ms vs ~44 ms) because it
    skips the byte-offset decode entirely.  Useful for probing image
    dimensions, pixel size, and data type before committing to a full read.

    Returns
    -------
    header : dict
        Same merged CIF + binary-section header as ``read_cbf()``.
    """
    with open(filename, "rb") as f:
        raw_file = f.read()

    marker_pos = raw_file.find(BINARY_SECTION_MARKER)
    if marker_pos < 0:
        raise ValueError("No CIF binary section marker found in file")

    starter_pos = raw_file.find(STARTER, marker_pos)
    if starter_pos < 0:
        raise ValueError("No binary data starter magic found in file")

    cif_end = raw_file.rfind(b";", 0, marker_pos)
    if cif_end < 0:
        cif_end = marker_pos
    cif_header = _parse_cif_header(raw_file[:cif_end])

    bin_header_bytes = raw_file[marker_pos + len(BINARY_SECTION_MARKER): starter_pos]
    bin_header = _parse_binary_header(bin_header_bytes)

    header = {}
    for k, v in cif_header.items():
        if k == "loop_":
            continue
        header[k] = v
    header.update(bin_header)

    if cif_header.get("_array_data.header_convention", "").strip("'\" ") == "PILATUS_1.2":
        contents = cif_header.get("_array_data.header_contents", "")
        if contents:
            header["pilatus"] = _parse_pilatus_header(contents)

    return header


# ─── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.cbf>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    fname = sys.argv[1]
    hdr, img = read_cbf(fname)

    print("=== Header ===")
    for k, v in hdr.items():
        if k == "pilatus":
            print("  [Pilatus sub-header]")
            for pk, pv in v.items():
                print(f"    {pk}: {pv}")
        else:
            print(f"  {k}: {v}")

    print(f"\n=== Image ===")
    print(f"  Shape : {img.shape}")
    print(f"  Dtype : {img.dtype}")
    print(f"  Min   : {img.min()}")
    print(f"  Max   : {img.max()}")
    print(f"  Mean  : {img.mean():.2f}")
