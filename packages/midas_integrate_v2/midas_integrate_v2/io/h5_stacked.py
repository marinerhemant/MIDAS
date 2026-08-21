"""Write the stacked HDF5 the streaming GPU integrator produced.

``IntegratorFitPeaksGPUStream`` emits flat binaries which
``integrator_stream_process_h5_stacked.py`` folded into one HDF5 of
consolidated arrays — as opposed to ``integrator_stream_process_h5.py``, which
writes one dataset *per frame*, named after the source file. For a scan of a
few thousand frames the per-frame form costs an HDF5 link per frame and is
slow to open; the stacked form is a handful of large datasets::

    /OmegaSumFrame          (n_groups, n_r, n_eta)   f64
    /frame_names            (n_frames,)              utf-8
    /lineouts               (n_frames, n_r)          f64  intensity only
    /lineouts_simple_mean   (n_frames, n_r)          f64
    /fit                    (n_frames, n_peaks, 7)   f64
    /geometry_maps/{R,TTh,Eta,Area,Q}_map  (n_r, n_eta)  f64

The R column is dropped from the lineouts: it is identical for every frame and
is recoverable from ``geometry_maps/R_map`` (or ``/r_axis_px``, which this
writer adds because carrying the axis explicitly costs ``n_r`` floats and
saves every reader a slice).

As in :mod:`.zarr_gsas`, v2's ``(n_eta, n_r)`` cakes are transposed to the
C's ``(n_r, n_eta)`` on the way out.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .zarr_gsas import _as_numpy, reta_map

__all__ = ["StackedH5Writer", "write_stacked_h5"]

#: Row order of ``/REtaMap``, split into named ``geometry_maps`` datasets.
_GEOM_ROWS = (
    ("R_map",   "R-center for each bin",                "pixels"),
    ("TTh_map", "TwoTheta-center for each bin",         "degrees"),
    ("Eta_map", "Eta-center for each bin",              "degrees"),
    ("Area_map", "Effective pixel area for each bin",   "fractional pixels"),
    ("Q_map",   "Q-center for each bin",                "inv_Angstrom"),
)


def _require_h5py():
    try:
        import h5py
    except ImportError as e:                       # pragma: no cover
        raise ImportError(
            "writing a stacked HDF5 requires h5py; pip install h5py"
        ) from e
    return h5py


class StackedH5Writer:
    """Incremental writer for the stacked HDF5.

    Datasets are pre-allocated from ``n_frames``, so each frame is written
    straight to its slice and nothing accumulates in memory except the running
    ``OmegaSumFrame`` chunk. Use as a context manager, or call :meth:`close`.

    Parameters
    ----------
    path :
        Output ``.h5``.
    spec :
        :class:`~midas_integrate_v2.spec.IntegrationSpec` the cakes came from.
    n_frames :
        Total frames to be written. Required — HDF5 datasets are allocated
        up front rather than resized per frame.
    bin_area :
        Per-bin area weight for ``geometry_maps/Area_map``; see
        :func:`~midas_integrate_v2.io.zarr_gsas.reta_map`.
    omega_sum_frames :
        Frames per ``OmegaSumFrame`` group. ``0`` disables the dataset;
        ``-1`` sums every frame into one group.
    n_peaks :
        Peaks per frame; allocates ``/fit`` as ``(n_frames, n_peaks, 7)``.
        ``0`` omits it.
    write_lineouts / write_simple_mean :
        Allocate the 1-D profile stacks.
    compression :
        Passed to ``create_dataset``; ``None`` for none. The 2-D stack is the
        bulk of the file and compresses well.
    metadata :
        Optional :class:`~midas_integrate_v2.io.writers.ProfileMetadata`,
        stored as a JSON root attribute for provenance.
    """

    def __init__(self, path: Path | str, *, spec, n_frames: int,
                 bin_area: Optional[Any] = None,
                 omega_sum_frames: int = 1,
                 n_peaks: int = 0,
                 write_lineouts: bool = True,
                 write_simple_mean: bool = False,
                 compression: Optional[str] = "gzip",
                 metadata: Optional[Any] = None):
        h5py = _require_h5py()
        self._h5py = h5py
        if n_frames <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.n_r = spec.n_r_bins
        self.n_eta = spec.n_eta_bins
        self.n_frames = int(n_frames)
        self.n_peaks = int(n_peaks)
        self.omega_sum_frames = int(omega_sum_frames)

        if self.omega_sum_frames == -1:
            self.n_groups = 1
        elif self.omega_sum_frames > 0:
            self.n_groups = -(-self.n_frames // self.omega_sum_frames)  # ceil
        else:
            self.n_groups = 0

        self._f = h5py.File(self.path, "w")
        f = self._f
        f.attrs["creation_date"] = datetime.datetime.now(
            datetime.timezone.utc).isoformat()
        f.attrs["num_frames"] = self.n_frames
        f.attrs["layout"] = "midas_integrate_v2 stacked"
        if metadata is not None:
            from dataclasses import asdict, is_dataclass
            f.attrs["metadata"] = json.dumps(
                asdict(metadata) if is_dataclass(metadata) else metadata)

        # -- geometry maps, split out of the REtaMap rows ------------------
        geom = f.create_group("geometry_maps")
        rmap = reta_map(spec, bin_area)                    # (5, n_r, n_eta)
        for row, (name, desc, units) in enumerate(_GEOM_ROWS):
            ds = geom.create_dataset(name, data=rmap[row], track_times=False)
            ds.attrs["description"] = desc
            ds.attrs["units"] = units
        # The shared radial axis, so a reader need not slice R_map.
        ds = f.create_dataset("r_axis_px", data=rmap[0, :, 0],
                              track_times=False)
        ds.attrs["units"] = "pixels"

        def _alloc(name, shape, chunks):
            # maxshape leaves axis 0 growable so close() can truncate to the
            # frames that actually arrived; a zero-filled tail is
            # indistinguishable from genuinely empty frames.
            return f.create_dataset(name, shape=shape, dtype=np.float64,
                                    maxshape=(None,) + tuple(shape[1:]),
                                    chunks=chunks, compression=compression,
                                    track_times=False)

        self._lineouts = (_alloc("lineouts", (self.n_frames, self.n_r),
                                 (1, self.n_r)) if write_lineouts else None)
        self._lineouts_sm = (_alloc("lineouts_simple_mean",
                                    (self.n_frames, self.n_r), (1, self.n_r))
                             if write_simple_mean else None)
        self._fit = (_alloc("fit", (self.n_frames, self.n_peaks, 7),
                            (1, self.n_peaks, 7)) if self.n_peaks > 0 else None)
        self._osf = (_alloc("OmegaSumFrame",
                            (self.n_groups, self.n_r, self.n_eta),
                            (1, self.n_r, self.n_eta))
                     if self.n_groups > 0 else None)
        if self._osf is not None:
            self._osf.attrs["omega_sum_frames"] = self.omega_sum_frames

        self._names: list[str] = []
        self._omegas: list[float] = []
        self._i = 0
        self._group_i = 0
        self._chunk: Optional[np.ndarray] = None
        self._chunk_n = 0
        self._closed = False

    def __enter__(self) -> "StackedH5Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def add_frame(self, cake=None, *, name: Optional[str] = None,
                  omega: Optional[float] = None,
                  lineout: Optional[Any] = None,
                  lineout_simple_mean: Optional[Any] = None,
                  fit: Optional[Any] = None) -> None:
        """Add one frame.

        ``cake`` is the ``(n_eta, n_r)`` 2-D result and feeds
        ``OmegaSumFrame``; ``lineout`` / ``lineout_simple_mean`` are ``(n_r,)``
        profiles. Any of them may be omitted if the corresponding dataset was
        not allocated.
        """
        if self._closed:
            raise RuntimeError("writer is closed")
        if self._i >= self.n_frames:
            raise ValueError(
                f"more frames added than the {self.n_frames} allocated"
            )
        i = self._i
        self._names.append(name if name is not None else str(i))
        self._omegas.append(float(i) if omega is None else float(omega))

        if lineout is not None and self._lineouts is not None:
            self._lineouts[i] = self._check_1d(lineout, "lineout")
        if lineout_simple_mean is not None and self._lineouts_sm is not None:
            self._lineouts_sm[i] = self._check_1d(lineout_simple_mean,
                                                  "lineout_simple_mean")
        if fit is not None and self._fit is not None:
            arr = _as_numpy(fit)
            if arr.shape != (self.n_peaks, 7):
                raise ValueError(
                    f"fit shape {arr.shape} != (n_peaks, 7) {(self.n_peaks, 7)}"
                )
            self._fit[i] = arr

        if cake is not None and self._osf is not None:
            arr = _as_numpy(cake)
            if arr.shape == (self.n_eta, self.n_r):
                arr = arr.T
            elif arr.shape != (self.n_r, self.n_eta):
                raise ValueError(
                    f"cake shape {arr.shape} is neither (n_eta, n_r) "
                    f"{(self.n_eta, self.n_r)} nor (n_r, n_eta) "
                    f"{(self.n_r, self.n_eta)}"
                )
            if self._chunk is None:
                self._chunk = np.zeros((self.n_r, self.n_eta), dtype=np.float64)
                self._chunk_n = 0
            np.add(self._chunk, np.nan_to_num(arr, nan=0.0), out=self._chunk)
            self._chunk_n += 1
            if (self.omega_sum_frames > 0
                    and self._chunk_n == self.omega_sum_frames):
                self._flush_chunk()

        self._i += 1

    def _check_1d(self, prof, what: str) -> np.ndarray:
        arr = _as_numpy(prof)
        if arr.shape != (self.n_r,):
            raise ValueError(
                f"{what} shape {arr.shape} != (n_r,) {(self.n_r,)}"
            )
        return arr

    def _flush_chunk(self) -> None:
        if self._chunk is None or self._chunk_n == 0:
            return
        if self._group_i >= self.n_groups:
            raise ValueError(
                f"more OmegaSumFrame groups than the {self.n_groups} allocated"
            )
        self._osf[self._group_i] = self._chunk
        self._group_i += 1
        self._chunk = None
        self._chunk_n = 0

    def close(self) -> Path:
        """Flush the trailing chunk, write frame names + omegas, close."""
        if self._closed:
            return self.path
        if self._chunk_n > 0:
            self._flush_chunk()
        if self._i != self.n_frames:
            # Fewer frames arrived than allocated: truncate rather than leave
            # zero-filled tail slices a reader cannot tell from empty frames.
            for ds in (self._lineouts, self._lineouts_sm, self._fit):
                if ds is not None:
                    ds.resize(self._i, axis=0)
            self._f.attrs["num_frames"] = self._i
            self._f.attrs["n_frames_allocated"] = self.n_frames
        if self._osf is not None and self._group_i != self.n_groups:
            self._osf.resize(self._group_i, axis=0)
        self._f.create_dataset(
            "frame_names",
            data=np.array(self._names, dtype=object),
            dtype=self._h5py.string_dtype(), track_times=False)
        ds = self._f.create_dataset(
            "Omegas", data=np.asarray(self._omegas, dtype=np.float64),
            track_times=False)
        ds.attrs["units"] = "degrees"
        self._f.close()
        self._closed = True
        return self.path


def write_stacked_h5(
    path: Path | str,
    cakes: Iterable[Any],
    *,
    spec,
    n_frames: int,
    frame_names: Optional[Sequence[str]] = None,
    omegas: Optional[Sequence[float]] = None,
    lineouts: Optional[Sequence[Any]] = None,
    lineouts_simple_mean: Optional[Sequence[Any]] = None,
    fits: Optional[Sequence[Any]] = None,
    bin_area: Optional[Any] = None,
    omega_sum_frames: int = 1,
    n_peaks: int = 0,
    compression: Optional[str] = "gzip",
    metadata: Optional[Any] = None,
) -> Path:
    """Write a stacked HDF5 from a sequence of integrated cakes.

    ``cakes`` is consumed lazily, so a generator keeps peak memory to one frame
    plus one ``OmegaSumFrame`` chunk. Returns the path written.
    """
    def _at(seq, i):
        return None if seq is None else seq[i]

    with StackedH5Writer(path, spec=spec, n_frames=n_frames,
                         bin_area=bin_area,
                         omega_sum_frames=omega_sum_frames,
                         n_peaks=n_peaks,
                         write_lineouts=lineouts is not None,
                         write_simple_mean=lineouts_simple_mean is not None,
                         compression=compression,
                         metadata=metadata) as w:
        for i, cake in enumerate(cakes):
            w.add_frame(cake,
                        name=_at(frame_names, i),
                        omega=_at(omegas, i),
                        lineout=_at(lineouts, i),
                        lineout_simple_mean=_at(lineouts_simple_mean, i),
                        fit=_at(fits, i))
    return Path(path)
