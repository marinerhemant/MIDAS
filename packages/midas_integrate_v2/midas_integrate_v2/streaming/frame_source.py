"""Frame iterators for sweep-mode integration.

A :class:`FrameSource` is anything you can iterate over to get
``(frame_id: str, image: numpy.ndarray)`` tuples without loading every
frame at once. Three concrete sources cover the common HEDM input
formats:

- :class:`TIFFGlobSource` — directory of ``.tif`` files.
- :class:`HDF5FrameSource` — single HDF5 file with frames stacked
  along axis 0 of a 3-D dataset.
- :class:`ZarrFrameSource` — same shape but Zarr-backed.
- :class:`NumpyArraySource` — for tests / small in-memory stacks.

All sources expose a uniform iterator + ``__len__`` + ``shape`` API
so :func:`integrate_stream` works against any of them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np


class FrameSource:
    """Abstract iterator over (frame_id, image) tuples."""

    @property
    def n_frames(self) -> int:
        raise NotImplementedError

    @property
    def frame_shape(self) -> Tuple[int, int]:
        """Returns ``(NrPixelsZ, NrPixelsY)``."""
        raise NotImplementedError

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        raise NotImplementedError

    def get(self, idx: int) -> Tuple[str, np.ndarray]:
        """Random-access; default implementation iterates."""
        for i, (fid, img) in enumerate(self):
            if i == idx:
                return fid, img
        raise IndexError(f"frame {idx} out of range ({self.n_frames})")


class NumpyArraySource(FrameSource):
    """Wrap an in-memory ``(N, NrPixelsZ, NrPixelsY)`` numpy array."""

    def __init__(self, frames: np.ndarray, *,
                  ids: Optional[List[str]] = None):
        if frames.ndim != 3:
            raise ValueError(
                f"frames must be 3-D (N, NZ, NY), got shape {frames.shape}"
            )
        self._frames = frames
        self._ids = (ids if ids is not None
                      else [f"frame_{i:05d}" for i in range(frames.shape[0])])
        if len(self._ids) != frames.shape[0]:
            raise ValueError("ids length must match frames axis 0")

    @property
    def n_frames(self) -> int:
        return self._frames.shape[0]

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return self._frames.shape[1], self._frames.shape[2]

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        for fid, img in zip(self._ids, self._frames):
            yield fid, img.astype(np.float64)

    def get(self, idx: int) -> Tuple[str, np.ndarray]:
        return self._ids[idx], self._frames[idx].astype(np.float64)


class TIFFGlobSource(FrameSource):
    """Iterate over TIFF files matched by a glob pattern.

    Files are loaded one at a time, in sorted-name order. Frame ID is
    the filename stem (without extension). Requires ``tifffile``.
    """

    def __init__(self, glob_pattern: str | Path, *,
                  shape_check: bool = True):
        try:
            import tifffile  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "TIFFGlobSource requires tifffile; pip install tifffile"
            ) from e
        self._tifffile = __import__("tifffile")
        glob_pattern = str(glob_pattern)
        if "*" in glob_pattern or "?" in glob_pattern:
            from glob import glob as _glob
            paths = sorted(_glob(glob_pattern))
        else:
            # Treat as a single file or a directory
            p = Path(glob_pattern)
            if p.is_dir():
                paths = sorted(str(x) for x in p.glob("*.tif"))
                paths += sorted(str(x) for x in p.glob("*.tiff"))
            elif p.is_file():
                paths = [str(p)]
            else:
                paths = []
        if not paths:
            raise FileNotFoundError(
                f"no TIFF files matched glob pattern {glob_pattern!r}"
            )
        self._paths = [Path(p) for p in paths]
        # Determine frame shape from first file
        first = self._tifffile.imread(self._paths[0])
        if first.ndim != 2:
            raise ValueError(
                f"TIFF {self._paths[0]} is not 2-D (got shape {first.shape})"
            )
        self._shape = first.shape
        self._shape_check = shape_check

    @property
    def n_frames(self) -> int:
        return len(self._paths)

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return self._shape

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        for path in self._paths:
            img = self._tifffile.imread(str(path)).astype(np.float64)
            if self._shape_check and img.shape != self._shape:
                raise ValueError(
                    f"TIFF {path} shape {img.shape} != expected {self._shape}"
                )
            yield path.stem, img

    def get(self, idx: int) -> Tuple[str, np.ndarray]:
        path = self._paths[idx]
        return path.stem, self._tifffile.imread(str(path)).astype(np.float64)


class HDF5FrameSource(FrameSource):
    """Iterate over frames in a single HDF5 dataset.

    Expected layout: a 3-D dataset of shape ``(N, NrPixelsZ, NrPixelsY)``.
    Reads each frame on demand without loading the whole dataset.
    Requires ``h5py``.
    """

    def __init__(self, path: str | Path, *, dataset: str = "frames",
                  ids_dataset: Optional[str] = None,
                  chunk_size: int = 1):
        try:
            import h5py  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "HDF5FrameSource requires h5py; pip install h5py"
            ) from e
        self._h5py = __import__("h5py")
        self._path = str(path)
        self._dataset_name = dataset
        with self._h5py.File(self._path, "r") as f:
            if dataset not in f:
                raise KeyError(
                    f"HDF5 file {self._path} has no dataset {dataset!r}"
                )
            arr = f[dataset]
            if arr.ndim != 3:
                raise ValueError(
                    f"dataset {dataset!r} must be 3-D, got shape {arr.shape}"
                )
            self._shape = (int(arr.shape[1]), int(arr.shape[2]))
            self._n = int(arr.shape[0])
            if ids_dataset is not None:
                ids = f[ids_dataset][:]
                self._ids = [
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in ids
                ]
            else:
                self._ids = None
        self._chunk_size = max(1, int(chunk_size))

    @property
    def n_frames(self) -> int:
        return self._n

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return self._shape

    def _make_id(self, i: int) -> str:
        if self._ids is not None:
            return self._ids[i]
        return f"frame_{i:05d}"

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        with self._h5py.File(self._path, "r") as f:
            arr = f[self._dataset_name]
            for start in range(0, self._n, self._chunk_size):
                end = min(self._n, start + self._chunk_size)
                chunk = arr[start:end].astype(np.float64)
                for k in range(end - start):
                    yield self._make_id(start + k), chunk[k]

    def get(self, idx: int) -> Tuple[str, np.ndarray]:
        with self._h5py.File(self._path, "r") as f:
            arr = f[self._dataset_name]
            return self._make_id(idx), arr[idx].astype(np.float64)


class GEBinaryFrameSource(FrameSource):
    """GE-detector raw binary file (header-detected; uint16 frames).

    GE files come in two flavours: an 8192-byte header (newer firmware)
    or no header (early data). We detect by checking that the file size
    minus 8192 is divisible by ``side * side * 2``; if so, it has the
    header. Frames are read lazily via :func:`numpy.memmap` so a 10 GB
    file does not load into RAM.

    The convention follows ``midas_calibrate_v2.io.readers._read_ge``:
    little-endian uint16 → float64 on read.

    Parameters
    ----------
    path :
        Path to the ``.ge3`` / ``.geN`` file (any GE-format binary).
    side :
        Side length of the (square) detector. Most APS GE detectors are
        2048×2048; pass explicitly when the file lacks a header. If
        None (default), we try 2048 first then 4096.
    skip_frame :
        Number of frames at the start to ignore (e.g., dark / shuttered
        first frame).
    frame_id_prefix :
        Prefix for generated frame ids (``ge_00000`` etc.).
    """

    _GE_HEADER_BYTES = 8192

    def __init__(
        self,
        path: str | Path,
        *,
        side: Optional[int] = None,
        skip_frame: int = 0,
        frame_id_prefix: str = "ge",
    ):
        self._path = Path(path)
        if not self._path.is_file():
            raise FileNotFoundError(self._path)
        size = self._path.stat().st_size
        # Try detecting header. Prefer caller-supplied side over autodetect.
        candidate_sides = (
            [int(side)] if side is not None else [2048, 4096, 1024, 1536]
        )
        detected = None
        for s in candidate_sides:
            frame_bytes = s * s * 2
            if frame_bytes <= 0:
                continue
            for header in (self._GE_HEADER_BYTES, 0):
                payload = size - header
                if payload > 0 and payload % frame_bytes == 0:
                    detected = (s, header, payload // frame_bytes)
                    break
            if detected is not None:
                break
        if detected is None:
            raise ValueError(
                f"could not deduce GE frame layout for {self._path} "
                f"(size {size} B, tried sides {candidate_sides})"
            )
        self._side, self._header, n_frames = detected
        self._skip = max(0, int(skip_frame))
        self._n_frames = max(0, n_frames - self._skip)
        self._prefix = str(frame_id_prefix)
        # Lazy memmap (re-opened per iter so multiple iters work).

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return self._side, self._side

    def _open_memmap(self) -> "np.memmap":
        # Total frames recorded on disk including skipped
        frames_on_disk = self._n_frames + self._skip
        return np.memmap(
            self._path, dtype="<u2", mode="r",
            offset=self._header,
            shape=(frames_on_disk, self._side, self._side),
        )

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        mm = self._open_memmap()
        for k in range(self._n_frames):
            j = self._skip + k
            fid = f"{self._prefix}_{k:05d}"
            yield fid, np.asarray(mm[j], dtype=np.float64)

    def get(self, idx: int) -> Tuple[str, np.ndarray]:
        if not (0 <= idx < self._n_frames):
            raise IndexError(f"frame {idx} out of range")
        mm = self._open_memmap()
        j = self._skip + idx
        return f"{self._prefix}_{idx:05d}", np.asarray(mm[j], dtype=np.float64)


class EDFFrameSource(FrameSource):
    """EDF (ESRF Data Format) frame source via ``fabio``.

    Accepts a single multi-frame EDF or a list/glob of single-frame EDFs.
    Requires ``pip install fabio`` (declared as the ``edf`` extra of
    midas-integrate-v2).
    """

    def __init__(
        self,
        path: str | Path | List[str | Path],
        *,
        frame_id_prefix: str = "edf",
    ):
        try:
            import fabio  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "EDFFrameSource requires fabio; pip install fabio"
            ) from e
        self._fabio = __import__("fabio")
        if isinstance(path, (list, tuple)):
            paths = [Path(p) for p in path]
        else:
            p = Path(path)
            if "*" in str(p) or "?" in str(p):
                from glob import glob as _glob
                paths = [Path(x) for x in sorted(_glob(str(p)))]
            elif p.is_dir():
                paths = sorted(list(p.glob("*.edf")) + list(p.glob("*.edf.gz")))
            else:
                paths = [p]
        if not paths:
            raise FileNotFoundError(f"no EDF files found at {path!r}")
        self._paths = paths
        # Determine total frame count by opening each file's first image.
        first = self._fabio.open(str(paths[0]))
        n_per = getattr(first, "nframes", 1)
        self._shape = (int(first.shape[0]), int(first.shape[1]))
        if len(paths) == 1:
            self._n_frames = int(n_per)
        else:
            # Multi-file: assume each file is single-frame (common ESRF).
            self._n_frames = len(paths)
        first.close()
        self._prefix = str(frame_id_prefix)

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return self._shape

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        if len(self._paths) == 1:
            edf = self._fabio.open(str(self._paths[0]))
            try:
                for k in range(edf.nframes):
                    frame = edf.getframe(k) if edf.nframes > 1 else edf
                    yield (f"{self._prefix}_{k:05d}",
                            np.asarray(frame.data, dtype=np.float64))
            finally:
                edf.close()
        else:
            for k, p in enumerate(self._paths):
                edf = self._fabio.open(str(p))
                try:
                    yield (p.stem,
                            np.asarray(edf.data, dtype=np.float64))
                finally:
                    edf.close()

    def get(self, idx: int) -> Tuple[str, np.ndarray]:
        if not (0 <= idx < self._n_frames):
            raise IndexError(f"frame {idx} out of range")
        if len(self._paths) == 1:
            edf = self._fabio.open(str(self._paths[0]))
            try:
                frame = edf.getframe(idx) if edf.nframes > 1 else edf
                return (f"{self._prefix}_{idx:05d}",
                        np.asarray(frame.data, dtype=np.float64))
            finally:
                edf.close()
        else:
            p = self._paths[idx]
            edf = self._fabio.open(str(p))
            try:
                return p.stem, np.asarray(edf.data, dtype=np.float64)
            finally:
                edf.close()


class ZarrFrameSource(FrameSource):
    """Iterate over frames in a Zarr 3-D array (chunked HDF5 alternative)."""

    def __init__(self, store: str | Path, *,
                  group_path: Optional[str] = None,
                  chunk_size: int = 1):
        try:
            import zarr  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ZarrFrameSource requires zarr; pip install 'zarr<3'"
            ) from e
        self._zarr = __import__("zarr")
        self._store_path = str(store)
        self._group_path = group_path
        z = self._zarr.open(self._store_path, mode="r")
        if group_path is not None:
            for piece in group_path.strip("/").split("/"):
                if piece:
                    z = z[piece]
        if z.ndim != 3:
            raise ValueError(
                f"Zarr array at {store}/{group_path} must be 3-D, got "
                f"shape {z.shape}"
            )
        self._shape = (int(z.shape[1]), int(z.shape[2]))
        self._n = int(z.shape[0])
        self._chunk_size = max(1, int(chunk_size))

    @property
    def n_frames(self) -> int:
        return self._n

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return self._shape

    def _open(self):
        z = self._zarr.open(self._store_path, mode="r")
        if self._group_path is not None:
            for piece in self._group_path.strip("/").split("/"):
                if piece:
                    z = z[piece]
        return z

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        z = self._open()
        for start in range(0, self._n, self._chunk_size):
            end = min(self._n, start + self._chunk_size)
            chunk = z[start:end][:].astype(np.float64)
            for k in range(end - start):
                yield f"frame_{start + k:05d}", chunk[k]

    def get(self, idx: int) -> Tuple[str, np.ndarray]:
        z = self._open()
        return f"frame_{idx:05d}", z[idx][:].astype(np.float64)
