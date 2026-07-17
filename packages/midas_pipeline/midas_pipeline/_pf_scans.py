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
import os
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

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
    """Load 1-D Y positions (µm) for the current layer's scans, in FILE ORDER.

    Convention (unified with ``stages/indexing.py`` and the scanning
    indexer): **file order == acquisition order** — row *k* is the Y
    position of the *k*-th acquired scan (``abs_scan_nr`` ordering), which
    may be descending for a negative ``--scan-step``. The voxel grid is
    sorted downstream by the indexer (``midas_index.run_scanning``:
    acquisition order for the beam-position filter, sorted for the grid).
    An earlier version ``np.sort``-ed here, silently reversing the
    scan↔Y pairing for descending acquisitions.

    Reads ``positions.csv`` if present; otherwise raises since callers
    can't iterate scans without knowing how many there are.
    """
    pcsv = layer_dir / "positions.csv"
    if not pcsv.exists():
        raise FileNotFoundError(
            f"_pf_scans: missing {pcsv}. Need positions.csv to iterate "
            "scans for PF mode. (The pipeline materializes it at layer "
            "setup from the scan geometry; if you are driving stages "
            "manually, pre-seed <layer_dir>/positions.csv — one Y per "
            "line, acquisition order.)"
        )
    arr = np.atleast_1d(np.loadtxt(pcsv, dtype=np.float64))
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
    work_dir: Optional[str | Path] = None,
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
    work_dir
        N11: writable per-scan work root. When set, ``scan_dir`` (where
        zip_convert mkdirs, Temp/ and per-scan CSVs live) is
        ``<work_dir>/<abs_scan_nr>`` instead of the raw scan dir — needed
        when RawFolder is read-only collaborator data. A pre-built
        ``.MIDAS.zip`` already sitting in the raw scan dir is still
        honoured.

    Returns
    -------
    A list of ``PFScanInfo``, one per scan in this layer, in acquisition
    order (== positions.csv file order): scan ``s`` pairs with row ``s-1``
    of positions.csv. The indexer sorts positions for its voxel grid
    itself (midas_index.run_scanning).
    """
    layer_dir = Path(layer_dir)
    kv = parse_params_kv(params_file)
    file_stem = _str_field(kv, "FileStem")
    start_nr_first_layer = _int_field(kv, "StartFileNrFirstLayer")
    nr_files_per_sweep = _int_field(kv, "NrFilesPerSweep") or _int_field(kv, "numFilesPerScan")
    padding = _int_field(kv, "Padding", default=7)
    ext = _str_field(kv, "Ext", default=".tif")

    # Check positions FIRST: a missing positions.csv is the
    # silent-corruption case (P0-2, hard error at the callers), and must
    # not be masked by an incomplete Parameters.txt (soft skip).
    positions = _positions_for_layer(layer_dir, n_scans_hint=n_scans_hint)
    n_scans = positions.size

    if start_nr_first_layer is None:
        raise ValueError(
            "_pf_scans: Parameters.txt missing StartFileNrFirstLayer."
        )
    if nr_files_per_sweep is None:
        raise ValueError(
            "_pf_scans: Parameters.txt missing NrFilesPerSweep / numFilesPerScan."
        )

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
        if work_dir is not None:
            # N11: raw dir stays read-only; all outputs go to the work
            # root. Prefer a pre-built zip in the raw location.
            raw_zip = zip_path
            scan_dir = Path(work_dir) / str(abs_scan_nr)
            zip_path = raw_zip if raw_zip.exists() else scan_dir / zip_name
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
    "claim_scan",
    "fan_out_scans",
    "iter_pf_scans",
    "n_scans_for_layer",
    "parse_params_kv",
    "release_scan",
]


# ---------------------------------------------------------------------------
# Per-scan claims + fan-out (N5/N6)
# ---------------------------------------------------------------------------
#
# Two independent runners racing over the same scan list both pick the
# lowest-undone scan (observed on the Ni Layer-3 run with an external
# 2-worker helper: outputs identical, so benign — but still wrong). A
# per-scan CLAIM makes each scan single-owner: an atomic O_EXCL file
# under ``<layer_dir>/midas_log/claims/`` (the layer dir, NOT the scan
# dir — raw data may be read-only, see N11). Claims from a dead process
# on the same host are broken automatically; claims from another host are
# honoured (no cross-host liveness probe).


def _claims_dir(layer_dir: Path) -> Path:
    d = Path(layer_dir) / "midas_log" / "claims"
    d.mkdir(parents=True, exist_ok=True)
    return d


def claim_scan(layer_dir: Path, stage: str, scan_nr: int) -> bool:
    """Atomically claim ``(stage, scan_nr)``. True = we own it."""
    path = _claims_dir(layer_dir) / f"{stage}.scan{scan_nr}.claim"
    payload = f"{socket.gethostname()} {os.getpid()}\n"
    for _attempt in (0, 1):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            return True
        except FileExistsError:
            try:
                host, pid = path.read_text().split()
                pid = int(pid)
            except (ValueError, OSError):
                return False                    # unreadable → honour it
            if host != socket.gethostname():
                return False                    # other host → honour it
            try:
                os.kill(pid, 0)
                return False                    # live process → honour it
            except (ProcessLookupError, PermissionError) as e:
                if isinstance(e, PermissionError):
                    return False                # live (not ours) → honour
            # Stale claim from a dead local process: break it and retry.
            LOG.warning("breaking stale %s claim for scan %d (dead pid %d)",
                        stage, scan_nr, pid)
            path.unlink(missing_ok=True)
    return False


def release_scan(layer_dir: Path, stage: str, scan_nr: int) -> None:
    (_claims_dir(layer_dir) / f"{stage}.scan{scan_nr}.claim").unlink(
        missing_ok=True)


def fan_out_scans(
    scans: List[PFScanInfo],
    worker: Callable[[PFScanInfo], object],
    *,
    layer_dir: Path,
    stage: str,
    n_workers: int = 1,
) -> List[Tuple[PFScanInfo, object]]:
    """Run ``worker(scan)`` over all scans with per-scan claims.

    ``n_workers == 1`` reproduces the serial loop (claims still taken, so
    two independent runners cannot double-process a scan). Workers are
    THREADS: every current per-scan worker either shells out (zip_convert)
    or spends its time in numpy/torch/child-process code that releases
    the GIL (peakfit's own frame pool, transforms' tensor ops).

    Returns ``[(scan, result), ...]`` in scan order. A worker exception
    is captured as the result (callers decide fail/skip semantics); a
    scan claimed by another runner yields the sentinel string
    ``"claimed-elsewhere"``.
    """
    def _one(s: PFScanInfo):
        if not claim_scan(layer_dir, stage, s.scan_nr):
            LOG.info("%s(PF): scan %d claimed by another runner; skipping.",
                     stage, s.scan_nr)
            return "claimed-elsewhere"
        try:
            return worker(s)
        except Exception as e:          # noqa: BLE001 — caller decides
            return e
        finally:
            release_scan(layer_dir, stage, s.scan_nr)

    n = max(1, int(n_workers))
    if n == 1:
        return [(s, _one(s)) for s in scans]
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(_one, scans))
    return list(zip(scans, results))
