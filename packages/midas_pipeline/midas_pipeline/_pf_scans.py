"""Per-scan iteration helper for PF mode upstream stages.

pf_MIDAS.py runs ``zip_convert``, ``hkl``, ``peakfit``, ``merge_overlaps``,
``calc_radius``, and ``transforms`` PER SCAN POSITION (one invocation per
scan, in parallel via parsl). The current midas-pipeline orchestrator
calls each stage once per layer, so each PF upstream stage needs to
iterate the scans itself.

This module centralises:

* parsing the upstream Parameters.txt (``RawFolder``, ``FileStem``,
  ``StartFileNrFirstLayer``, ``NrFilesPerSweep``, ``Padding``, ``Ext``);
* deriving the absolute scan number per (layer_nr, scan_nr_within_layer);
* discovering / creating the per-scan working directory and locating the
  ``.MIDAS.zip`` archive inside it.

The contract a scan-iterating stage needs is one ``PFScanInfo`` per
scan: where to read the zip, where to write outputs, what Y position
this scan sits at.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameters.txt parsing
# ---------------------------------------------------------------------------


def _strip_comment(line: str) -> str:
    """Drop ``#...`` trailing comments + final ``;`` punctuation."""
    line = line.split("#", 1)[0].rstrip()
    if line.endswith(";"):
        line = line[:-1].rstrip()
    return line


def parse_params_kv(path: str | Path) -> dict[str, str]:
    """Read a MIDAS-style Parameters.txt and return ``{key: value_string}``.

    For repeated keys (``RingNumbers``, ``RingRadii``, etc.) only the last
    occurrence wins. Values are kept as the raw whitespace-joined remainder
    of the line so callers can split as needed (single int, single float,
    space-separated triple, etc.).

    Tolerates the C param-file's trailing ``;`` and ``#`` comments.
    """
    out: dict[str, str] = {}
    p = Path(path)
    for line in p.read_text().splitlines():
        line = _strip_comment(line)
        if not line:
            continue
        toks = line.split(None, 1)
        if not toks:
            continue
        key = toks[0]
        val = toks[1] if len(toks) > 1 else ""
        out[key] = val.strip()
    return out


def _int_field(kv: dict[str, str], key: str, default: Optional[int] = None) -> Optional[int]:
    if key not in kv:
        return default
    digits = re.findall(r"-?\d+", kv[key])
    return int(digits[0]) if digits else default


def _float_field(kv: dict[str, str], key: str, default: Optional[float] = None) -> Optional[float]:
    if key not in kv:
        return default
    try:
        return float(kv[key].split()[0])
    except (ValueError, IndexError):
        return default


def _str_field(kv: dict[str, str], key: str, default: str = "") -> str:
    if key not in kv:
        return default
    return kv[key].split()[0] if kv[key] else default


# ---------------------------------------------------------------------------
# Per-scan dataclass + iterator
# ---------------------------------------------------------------------------


@dataclass
class PFScanInfo:
    """All the per-scan paths and numbers an upstream PF stage needs."""

    layer_nr: int                # 1-based layer index
    scan_nr: int                 # 1-based scan index within layer (1..n_scans)
    abs_scan_nr: int             # absolute file numbering (e.g. 561274)
    scan_dir: Path               # per-scan working dir
    zip_path: Path               # ``.MIDAS.zip`` archive path (may not exist yet)
    y_position_um: float         # Y position from positions.csv

    @property
    def temp_dir(self) -> Path:
        return self.scan_dir / "Temp"

    @property
    def allpeaks_ps_bin(self) -> Path:
        return self.temp_dir / "AllPeaks_PS.bin"

    @property
    def allpeaks_px_bin(self) -> Path:
        return self.temp_dir / "AllPeaks_PX.bin"

    @property
    def hkls_csv(self) -> Path:
        return self.scan_dir / "hkls.csv"

    @property
    def input_all_extra_csv(self) -> Path:
        return self.scan_dir / "InputAllExtraInfoFittingAll.csv"


def _positions_for_layer(layer_dir: Path, n_scans_hint: Optional[int] = None) -> np.ndarray:
    """Load 1-D Y positions (µm) for the current layer's scans, ascending.

    Reads ``positions.csv`` if present; otherwise raises since callers
    can't iterate scans without knowing how many there are.
    """
    pcsv = layer_dir / "positions.csv"
    if not pcsv.exists():
        raise FileNotFoundError(
            f"_pf_scans: missing {pcsv}. Need positions.csv to iterate "
            "scans for PF mode."
        )
    arr = np.atleast_1d(np.loadtxt(pcsv, dtype=np.float64))
    arr = np.sort(arr)
    if n_scans_hint is not None and arr.size != n_scans_hint:
        LOG.warning(
            "positions.csv has %d entries but n_scans hint is %d; using "
            "positions.csv length.", arr.size, n_scans_hint,
        )
    return arr


def iter_pf_scans(
    *,
    params_file: str | Path,
    layer_dir: Path,
    layer_nr: int,
    raw_dir: Optional[str | Path] = None,
    n_scans_hint: Optional[int] = None,
) -> List[PFScanInfo]:
    """Materialise the per-scan list for the given layer.

    Parameters
    ----------
    params_file
        Path to the upstream Parameters.txt. Must define ``FileStem``,
        ``StartFileNrFirstLayer``, ``NrFilesPerSweep`` (the offset used
        to step from layer N to N+1; also reused as the scan-to-scan
        offset within a layer per pf_MIDAS.py:646), and ideally
        ``Padding`` (default 7), ``Ext`` (default ``.tif``).
    layer_dir
        ``<result_dir>/LayerNr_{N}`` — where ``positions.csv`` lives.
    layer_nr
        1-based layer index.
    raw_dir
        Directory containing per-scan subdirs. If None, falls back to
        ``RawFolder`` from params, then ``layer_dir``. Each per-scan
        subdir name is the absolute scan number (zero-padded or not).
    n_scans_hint
        Cross-check ``positions.csv`` row count.

    Returns
    -------
    A list of ``PFScanInfo``, one per scan in this layer, ordered by
    ascending Y position (matches the indexer / refiner's voxel grid
    convention, see midas_index.run_scanning).
    """
    layer_dir = Path(layer_dir)
    kv = parse_params_kv(params_file)
    file_stem = _str_field(kv, "FileStem")
    start_nr_first_layer = _int_field(kv, "StartFileNrFirstLayer")
    nr_files_per_sweep = _int_field(kv, "NrFilesPerSweep") or _int_field(kv, "numFilesPerScan")
    padding = _int_field(kv, "Padding", default=7)
    ext = _str_field(kv, "Ext", default=".tif")

    if start_nr_first_layer is None:
        raise ValueError(
            "_pf_scans: Parameters.txt missing StartFileNrFirstLayer."
        )
    if nr_files_per_sweep is None:
        raise ValueError(
            "_pf_scans: Parameters.txt missing NrFilesPerSweep / numFilesPerScan."
        )

    positions = _positions_for_layer(layer_dir, n_scans_hint=n_scans_hint)
    n_scans = positions.size

    raw_root: Path
    if raw_dir is not None:
        raw_root = Path(raw_dir)
    elif "RawFolder" in kv:
        raw_root = Path(kv["RawFolder"].split()[0])
    else:
        raw_root = layer_dir

    out: List[PFScanInfo] = []
    layer_base = start_nr_first_layer + (layer_nr - 1) * nr_files_per_sweep
    for s in range(1, n_scans + 1):
        abs_scan_nr = layer_base + (s - 1) * nr_files_per_sweep
        # Try several scan-dir layouts: unpadded number, padded, layer-dir nested.
        candidates = [
            raw_root / str(abs_scan_nr),
            raw_root / f"{abs_scan_nr:0{padding}d}",
            layer_dir / str(abs_scan_nr),
        ]
        scan_dir = next((c for c in candidates if c.exists()), candidates[0])
        zip_name = f"{file_stem}_{abs_scan_nr:0{padding}d}.MIDAS.zip"
        zip_path = scan_dir / zip_name
        out.append(PFScanInfo(
            layer_nr=layer_nr,
            scan_nr=s,
            abs_scan_nr=abs_scan_nr,
            scan_dir=scan_dir,
            zip_path=zip_path,
            y_position_um=float(positions[s - 1]),
        ))
    return out


def n_scans_for_layer(layer_dir: Path) -> int:
    """Convenience: count of scans in this layer (from positions.csv)."""
    return _positions_for_layer(Path(layer_dir)).size


__all__ = [
    "PFScanInfo",
    "iter_pf_scans",
    "n_scans_for_layer",
    "parse_params_kv",
]
