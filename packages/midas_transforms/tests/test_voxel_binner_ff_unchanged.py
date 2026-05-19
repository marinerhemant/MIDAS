"""FF regression gate for the voxel_binner: ``bin_data_unified`` with
``scan_positions=None`` must produce a byte-identical ``Spots.bin`` to
today's ``bin_data``.

This is the **single most important test** for stream B: extending
``midas-transforms`` with PF semantics must not change FF behaviour.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from midas_transforms import bin_data, bin_data_unified
from midas_transforms.io import binary as bio


def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def test_unified_ff_mode_byte_identical(tmp_inputall_dir: Path, tmp_path: Path):
    """``bin_data_unified(scan_positions=None)`` writes identical bytes to
    ``bin_data``."""
    # Baseline run (FF bin_data).
    ff_dir = tmp_path / "ff"
    ff_dir.mkdir()
    import shutil
    for fn in ("InputAll.csv", "InputAllExtraInfoFittingAll.csv", "paramstest.txt"):
        shutil.copy(tmp_inputall_dir / fn, ff_dir / fn)
    bin_data(result_folder=ff_dir)

    # Unified run with scan_positions=None.
    unified_dir = tmp_path / "unified"
    unified_dir.mkdir()
    for fn in ("InputAll.csv", "InputAllExtraInfoFittingAll.csv", "paramstest.txt"):
        shutil.copy(tmp_inputall_dir / fn, unified_dir / fn)
    bin_data_unified(result_folder=unified_dir, scan_positions=None)

    # Every output file must be byte-identical.
    for fn in ("Spots.bin", "ExtraInfo.bin", "Data.bin", "nData.bin"):
        assert (ff_dir / fn).exists(), f"baseline missing {fn}"
        assert (unified_dir / fn).exists(), f"unified missing {fn}"
        assert _hash_file(ff_dir / fn) == _hash_file(unified_dir / fn), (
            f"{fn} differs between baseline ``bin_data`` and "
            f"``bin_data_unified(scan_positions=None)``"
        )


def test_unified_ff_mode_no_voxel_sidecar(tmp_inputall_dir: Path):
    """FF-mode call must not emit a PF ``voxel_scan_pos.bin``.

    Since f962e7b2 (unified 10-col binary format), FF mode *does* emit a
    single-row ``positions.csv`` (``"0.000000\\n"``) so the unified
    ``midas_indexer`` can detect FF vs PF from the row count without a
    separate code path. The negative assertion is therefore only on the
    PF-specific ``voxel_scan_pos.bin`` sidecar.
    """
    for fn in ("voxel_scan_pos.bin", "positions.csv"):
        (tmp_inputall_dir / fn).unlink(missing_ok=True)
    bin_data_unified(result_folder=tmp_inputall_dir, scan_positions=None)
    assert not (tmp_inputall_dir / "voxel_scan_pos.bin").exists(), (
        "FF mode must not emit voxel_scan_pos.bin"
    )
    positions_csv = tmp_inputall_dir / "positions.csv"
    assert positions_csv.exists(), (
        "FF mode must emit a 1-line positions.csv sentinel for midas_indexer"
    )
    rows = [r for r in positions_csv.read_text().splitlines() if r.strip()]
    assert len(rows) == 1, (
        f"FF positions.csv must contain exactly one row; got {len(rows)}"
    )
    assert float(rows[0]) == 0.0, (
        f"FF positions.csv row must be 0.0; got {rows[0]!r}"
    )
