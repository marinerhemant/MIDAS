"""Add or replace arrays inside an existing ``.MIDAS.zip``, and prove they took.

The same mechanism :func:`midas_zipper.param_refresh.refresh_analysis_params`
uses for analysis parameters, for arbitrary keys: a per-frame ``omegaCenter``, a
detector mask, a scan's ``dty``. Arrays are staged in a zarr directory store and
merged with ONE ``zip -u`` call, and the archive is re-read afterwards, because
``zip -u`` has a silent no-op mode (exit 12, or an unchanged entry when the
staged file is not newer than the stored one) and a key that reads back stale
is exactly the failure this has to rule out.

Group metadata (``.zgroup``) for every parent of a new key is staged too, so a
key under a group the archive does not have yet is readable afterwards.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .param_refresh import ParamRefreshError, _archive_mtime_floor, _chunk_entries

__all__ = ["write_arrays"]


def write_arrays(
    zip_path: os.PathLike | str,
    arrays: Mapping[str, np.ndarray],
    *,
    chunks: Optional[Mapping[str, Sequence[int]]] = None,
    compressor=None,
) -> None:
    """Write ``{key: array}`` into ``zip_path`` (added if absent, replaced if present).

    ``chunks`` optionally gives a chunk shape per key (default: the whole array
    as one chunk). ``compressor`` is passed to zarr (default: zarr's). Raises
    :class:`ParamRefreshError` if ``zip`` is missing, fails, or any key reads
    back different from what was asked.
    """
    import zarr

    zip_path = Path(zip_path).resolve()
    if not arrays:
        return
    if shutil.which("zip") is None:
        raise ParamRefreshError("the 'zip' executable is required to write into an existing .MIDAS.zip")
    chunks = dict(chunks or {})
    values = {k.strip("/"): np.asarray(v) for k, v in arrays.items()}

    with tempfile.TemporaryDirectory(prefix="midas_zipwrite_") as tmp:
        stage_dir = Path(tmp) / "stage"
        staged = zarr.open(str(stage_dir), "w")
        for key, value in values.items():
            kw = {} if compressor is None else {"compressor": compressor}
            ds = staged.create_dataset(key, shape=value.shape, dtype=value.dtype,
                                       chunks=tuple(chunks.get(key, value.shape or (1,))),
                                       write_empty_chunks=True, **kw)
            ds[...] = value

        entries = []
        for key in values:
            entries.extend(_chunk_entries(stage_dir, key))
            parts = key.split("/")[:-1]                 # parent groups, so a new group is readable
            for i in range(len(parts) + 1):
                zg = Path(*parts[:i], ".zgroup") if i else Path(".zgroup")
                if (stage_dir / zg).is_file():
                    entries.append(str(zg))
        entries = sorted(set(entries))
        if not entries:
            raise ParamRefreshError("staging produced no files to write")

        stamp = max(time.time(), _archive_mtime_floor(zip_path, entries) + 2.0)
        for rel in entries:
            os.utime(stage_dir / rel, (stamp, stamp))
        proc = subprocess.run(["zip", "-u", str(zip_path), *entries], cwd=str(stage_dir),
                              capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise ParamRefreshError(f"zip -u failed on {zip_path} (exit {proc.returncode}): "
                                    f"{(proc.stderr or proc.stdout or '').strip()[:400]}")

    root = zarr.open(str(zip_path), "r")
    stale = []
    for key, value in values.items():
        try:
            got = np.asarray(root[key][...])
        except Exception:                                # noqa: BLE001 - absent key
            got = None
        if got is None or got.shape != value.shape or not np.array_equal(got, value):
            stale.append(key)
    if stale:
        raise ParamRefreshError(f"{zip_path.name}: write did not take for {stale}")
