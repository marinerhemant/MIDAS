"""
map_header.py — Python reader for Map.bin / nMap.bin parameter-hash headers.

The MapHeader is a 64-byte header prepended to Map.bin and nMap.bin by
DetectorMapper / DetectorMapperZarr.  It contains a SHA-256 hash of the
geometry parameters so that downstream consumers can detect stale files.

Header layout (64 bytes total):
    magic        uint32  4   0x3050414D ("MAP0")
    version      uint32  4   1
    param_hash   bytes  32   SHA-256
    reserved     bytes  24   zeros

Usage
-----
    from map_header import read_map_header, compute_param_hash
    hdr = read_map_header(Path("Map.bin"))
    if hdr is not None:
        expected = compute_param_hash(Lsd=..., yCen=..., ...)
        if hdr["hash"] != expected:
            print("WARNING: Map.bin is stale — parameters changed")
"""

from pathlib import Path
import hashlib
import struct
from typing import Optional

MAP_HEADER_MAGIC = 0x3050414D  # "MAP0"
MAP_HEADER_SIZE = 64


def read_map_header(path: Path) -> Optional[dict]:
    """Read the 64-byte header from a Map.bin or nMap.bin file.

    Returns dict with keys 'magic', 'version', 'hash' (hex string),
    or None if the file has no header (legacy format).
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size < MAP_HEADER_SIZE:
        return None

    with open(path, 'rb') as f:
        data = f.read(MAP_HEADER_SIZE)

    if len(data) < MAP_HEADER_SIZE:
        return None

    magic, version = struct.unpack_from('<II', data, 0)
    if magic != MAP_HEADER_MAGIC:
        return None

    param_hash = data[8:40]
    return {
        'magic': magic,
        'version': version,
        'hash': param_hash.hex(),
    }


def compute_param_hash(
    Lsd: float, yCen: float, zCen: float,
    pxY: float, pxZ: float,
    tx: float, ty: float, tz: float,
    p0: float, p1: float, p2: float, p3: float, p4: float,
    RhoD: float,
    RBinSize: float, EtaBinSize: float,
    RMin: float, RMax: float, EtaMin: float, EtaMax: float,
    NrPixelsY: int, NrPixelsZ: int,
) -> str:
    """Compute the same SHA-256 hash as MapHeader.h's map_header_compute().

    Returns the hash as a lowercase hex string.
    """
    # Must match the canonical string in MapHeader.h exactly:
    canonical = (
        f"BC={yCen:.6f},{zCen:.6f}|EtaBinSize={EtaBinSize:.6f}|"
        f"EtaMax={EtaMax:.6f}|EtaMin={EtaMin:.6f}|"
        f"Lsd={Lsd:.6f}|NrPixelsY={NrPixelsY}|NrPixelsZ={NrPixelsZ}|"
        f"RBinSize={RBinSize:.6f}|RMax={RMax:.6f}|RMin={RMin:.6f}|"
        f"RhoD={RhoD:.6f}|"
        f"p0={p0:.6f}|p1={p1:.6f}|p2={p2:.6f}|p3={p3:.6f}|p4={p4:.6f}|"
        f"pxY={pxY:.6f}|pxZ={pxZ:.6f}|"
        f"tx={tx:.6f}|ty={ty:.6f}|tz={tz:.6f}"
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def check_map_header(path: Path, label: str = "Map.bin") -> bool:
    """Quick check: does the file have a valid parameter header?

    Prints info/warning and returns True if header present, False otherwise.
    """
    hdr = read_map_header(path)
    if hdr is None:
        print(f"  WARNING: {label} has no parameter header (legacy format).")
        print(f"    Consider regenerating with latest DetectorMapper(Zarr).")
        return False
    else:
        short_hash = hdr['hash'][:16]
        print(f"  {label}: header ok (v{hdr['version']}, hash={short_hash}...)")
        return True
