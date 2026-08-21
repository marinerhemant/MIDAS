"""Write the MIDAS ``.zarr.zip`` that the C integrator used to produce.

``IntegratorZarrOMP`` wrote a ``.caked.hdf`` which ``integrator.py`` then
converted to ``.zarr.zip`` via ``midas2zip.Hdf5ToZarr``. That zip is what
GSAS-II's ``G2pwd_MIDAS.py`` reader ("MIDAS zarr") opens, and what
``midas_zipper`` / ``midas_pipeline`` consume. v2 had no way to emit it.

This module writes the zip directly, reproducing the full C layout
(``IntegratorZarrOMP.c:2074-2168``)::

    /REtaMap                              (5, n_r, n_eta)   f64
    /IntegrationResult/FrameNr_<i>        (n_r, n_eta)      f64
    /OmegaSumFrame/LastFrameNumber_<i>    (n_r, n_eta)      f64
    /Omegas                               (n_frames,)       f64
    /SumFrames                            (n_r, n_eta)      f64
    /InstrumentParameters/<name>          (1,)              f64

Two conventions have to be bridged, and both are easy to get wrong:

* v2's ``integrate_*`` return **(n_eta, n_r)**; the C writes **(n_r, n_eta)**.
  Every array is transposed on the way out, as in :mod:`.v1_outputs`.
* Attributes must be written in a *single* ``attrs.update()`` per node. A
  zarr ``ZipStore`` appends rather than replaces, so a second write leaves two
  ``.zattrs`` members for one node and readers may pick up the stale one.

zarr 2 only, deliberately: ``midas_zipper`` pins ``zarr>=2.13,<3`` and uses
``zarr.hierarchy`` / ``zarr.copy_all``, both removed in zarr 3, as do
``midas_ff_pipeline``, ``midas_peakfit``, ``midas_pipeline`` and
``midas_transforms``. A zarr-3-format zip would be unreadable by every
downstream MIDAS consumer and by the GSAS-II reader.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "DEFAULT_INSTRUMENT_PARAMS",
    "INSTRUMENT_PARAM_NAMES",
    "reta_map",
    "instrument_params_from_spec",
    "GSASZarrWriter",
    "write_gsas_zarr_zip",
]

#: Order matters only for readability; the C writes each as its own dataset.
INSTRUMENT_PARAM_NAMES = ("Polariz", "Lam", "SH_L", "U", "V", "W",
                          "X", "Y", "Z", "Distance")

#: GSAS-II profile-coefficient defaults, matching ``G2pwd_MIDAS.py`` and the
#: reference ``create_zarr_zip`` in ``integrator_stream_process_h5.py``. Only
#: ``Lam``/``Distance``/``Polariz`` are knowable from an IntegrationSpec; the
#: rest are refinement starting values that the C also defaulted.
DEFAULT_INSTRUMENT_PARAMS: Dict[str, float] = {
    "Lam": 0.413263,
    "Polariz": 0.99,
    "SH_L": 0.002,
    "U": 1.163,
    "V": -0.126,
    "W": 0.063,
    "X": 0.0,
    "Y": 0.0,
    "Z": 0.0,
    "Distance": 1000000.0,
}


def _require_zarr():
    try:
        import zarr
    except ImportError as e:                       # pragma: no cover
        raise ImportError(
            "writing a MIDAS .zarr.zip requires zarr 2.x; "
            "pip install 'zarr>=2.16,<3'"
        ) from e
    major = int(zarr.__version__.split(".")[0])
    if major >= 3:
        raise RuntimeError(
            f"zarr {zarr.__version__} is installed, but the MIDAS zarr layout "
            "is zarr-format 2. Every MIDAS consumer (midas_zipper, "
            "midas_pipeline, midas_ff_pipeline, midas_peakfit, "
            "midas_transforms) and the GSAS-II reader pin zarr<3. "
            "Install 'zarr>=2.16,<3'."
        )
    return zarr


def _as_numpy(a) -> np.ndarray:
    """Accept a torch tensor or anything array-like; return float64 ndarray."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def _float(x) -> float:
    return float(x.detach()) if hasattr(x, "detach") else float(x)


def _r_bin_edges(spec) -> tuple[np.ndarray, np.ndarray]:
    """Low/high R edges in pixels, matching ``IntegratorZarrOMP.c:1478-1494``.

    R-mode bins are uniform in R. Q-mode bins are uniform in Q and mapped to R
    by ``R(Q) = (Lsd/px) * tan(2*asin(Q*Lam/(4*pi)))``.
    """
    n_r = spec.n_r_bins
    i = np.arange(n_r, dtype=np.float64)
    if spec.q_mode_active:
        px = spec.effective_pxYZ()[0]
        lsd = _float(spec.Lsd)
        lam = _float(spec.Wavelength)
        q_lo = spec.QMin + spec.QBinSize * i
        q_hi = spec.QMin + spec.QBinSize * (i + 1.0)
        scale = lsd / px
        return (scale * np.tan(2.0 * np.arcsin(q_lo * lam / (4.0 * math.pi))),
                scale * np.tan(2.0 * np.arcsin(q_hi * lam / (4.0 * math.pi))))
    return (spec.RMin + spec.RBinSize * i,
            spec.RMin + spec.RBinSize * (i + 1.0))


def reta_map(spec, bin_area: Optional[Any] = None) -> np.ndarray:
    """The ``/REtaMap`` array, shape ``(5, n_r, n_eta)``.

    Rows are ``Radius, 2Theta, Eta, BinArea, Q`` — pixels, degrees, degrees,
    pixels, inverse angstrom — exactly as
    ``IntegratorZarrOMP.c:1755-1762`` fills them.

    ``bin_area`` is the per-bin summed area weight (the C's ``totArea``). It is
    a property of the geometry alone, so pass
    :func:`midas_integrate_v2.io.v1_outputs.bin_weights` output, shaped
    ``(n_eta, n_r)`` like everything else in v2. When omitted the row is left
    at zero and a warning is issued, because a consumer cannot tell an
    unpopulated area row from a genuinely empty one.
    """
    n_r, n_eta = spec.n_r_bins, spec.n_eta_bins
    r_lo, r_hi = _r_bin_edges(spec)
    r_mean = 0.5 * (r_lo + r_hi)                                   # (n_r,)
    eta_i = np.arange(n_eta, dtype=np.float64)
    eta_mean = spec.EtaMin + spec.EtaBinSize * (eta_i + 0.5)       # (n_eta,)

    px = spec.effective_pxYZ()[0]
    lsd = _float(spec.Lsd)
    lam = _float(spec.Wavelength)

    two_theta_rad = np.arctan(r_mean * px / lsd)                   # (n_r,)
    q = ((4.0 * math.pi / lam) * np.sin(two_theta_rad / 2.0)
         if lam > 0 else np.zeros_like(two_theta_rad))

    out = np.zeros((5, n_r, n_eta), dtype=np.float64)
    out[0] = r_mean[:, None]
    out[1] = np.degrees(two_theta_rad)[:, None]
    out[2] = eta_mean[None, :]
    if bin_area is None:
        warnings.warn(
            "reta_map: no bin_area supplied; /REtaMap row 3 (BinArea) will be "
            "zero. Pass bin_weights(geom, integrate_fn) to populate it.",
            stacklevel=2,
        )
    else:
        area = _as_numpy(bin_area)
        if area.shape == (n_eta, n_r):
            area = area.T
        if area.shape != (n_r, n_eta):
            raise ValueError(
                f"bin_area shape {area.shape} is neither (n_eta, n_r) "
                f"{(n_eta, n_r)} nor (n_r, n_eta) {(n_r, n_eta)}"
            )
        out[3] = area
    out[4] = q[:, None]
    return out


def instrument_params_from_spec(
    spec, overrides: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Instrument parameters for ``/InstrumentParameters``.

    ``Lam``, ``Distance`` and ``Polariz`` come from the spec; the GSAS-II
    profile coefficients fall back to :data:`DEFAULT_INSTRUMENT_PARAMS` unless
    overridden. ``Distance`` stays in micrometres, as the C wrote it.
    """
    params = dict(DEFAULT_INSTRUMENT_PARAMS)
    lam = _float(spec.Wavelength)
    if lam > 0:
        params["Lam"] = lam
    params["Distance"] = _float(spec.Lsd)
    params["Polariz"] = float(spec.PolarizationFraction)
    if overrides:
        unknown = set(overrides) - set(INSTRUMENT_PARAM_NAMES)
        if unknown:
            raise ValueError(
                f"unknown instrument parameter(s) {sorted(unknown)}; "
                f"valid: {list(INSTRUMENT_PARAM_NAMES)}"
            )
        params.update({k: float(v) for k, v in overrides.items()})
    return params


class GSASZarrWriter:
    """Incremental writer for the MIDAS ``.zarr.zip``.

    Frames are added one at a time and the running ``OmegaSumFrame`` chunk is
    flushed every ``omega_sum_frames``, so a long scan never has to be held in
    memory. Use as a context manager, or call :meth:`close` yourself.

    Parameters
    ----------
    path :
        Output ``.zarr.zip``.
    spec :
        :class:`~midas_integrate_v2.spec.IntegrationSpec` the cakes came from.
    bin_area :
        Per-bin area weight for ``/REtaMap`` row 3; see :func:`reta_map`.
    instrument_params :
        Overrides for ``/InstrumentParameters``.
    omega_sum_frames :
        Frames per ``OmegaSumFrame`` chunk, the C's ``chunkFiles``. ``0``
        disables the group; ``-1`` sums every frame into one.
    individual_save :
        Write ``/IntegrationResult/FrameNr_<i>`` per frame (the C's
        ``individualSave``). Off by default — it doubles the file size and
        GSAS-II does not read it.
    sum_images :
        Write ``/SumFrames``, the sum over every frame (the C's ``sumImages``).
    """

    def __init__(self, path: Path | str, *, spec,
                 bin_area: Optional[Any] = None,
                 instrument_params: Optional[Mapping[str, float]] = None,
                 omega_sum_frames: int = 1,
                 individual_save: bool = False,
                 sum_images: bool = True):
        zarr = _require_zarr()
        self._zarr = zarr
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.n_r = spec.n_r_bins
        self.n_eta = spec.n_eta_bins
        self.omega_sum_frames = int(omega_sum_frames)
        self.individual_save = bool(individual_save)
        self.sum_images = bool(sum_images)
        # Validated up front: a bad override should fail before the zip exists.
        self._instrument_params = instrument_params_from_spec(
            spec, instrument_params)

        self._store = zarr.ZipStore(str(self.path), mode="w")
        self._root = zarr.group(store=self._store, overwrite=True)

        # Geometry is frame-independent; write it up front like the C does on
        # its first frame.
        self._root.array("REtaMap", data=reta_map(spec, bin_area),
                         dtype="float64", chunks=False)
        self._root["REtaMap"].attrs.update({
            "nRBins": int(self.n_r),
            "nEtaBins": int(self.n_eta),
            "Header": "Radius,2Theta,Eta,BinArea,Q",
            "Units": "Pixels,Degrees,Degrees,Pixels,InvAngstrom",
        })

        self._res_grp = (self._root.create_group("IntegrationResult")
                         if self.individual_save else None)
        self._osf_grp = (self._root.create_group("OmegaSumFrame")
                         if self.omega_sum_frames != 0 else None)

        self._n_frames = 0
        self._omegas: list[float] = []
        self._sum: Optional[np.ndarray] = None       # /SumFrames accumulator
        self._chunk: Optional[np.ndarray] = None     # OmegaSumFrame accumulator
        self._chunk_n = 0
        self._chunk_first_ome: Optional[float] = None
        self._chunk_meta: Dict[str, list] = {"Temperature": [], "Pressure": [],
                                             "I": [], "I0": []}
        self._closed = False

    # -- context manager -------------------------------------------------
    def __enter__(self) -> "GSASZarrWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writing ---------------------------------------------------------
    def add_frame(self, cake, *, omega: Optional[float] = None,
                  temperature: Optional[float] = None,
                  pressure: Optional[float] = None,
                  current: Optional[float] = None,
                  current_i0: Optional[float] = None) -> None:
        """Add one integrated cake, shaped ``(n_eta, n_r)`` as v2 returns it.

        ``(n_r, n_eta)`` is also accepted when the two differ, so a caller that
        has already transposed is not silently mis-stored. Omega defaults to
        the frame index, matching the reference converter's behaviour when the
        parameter file carries no omega.
        """
        if self._closed:
            raise RuntimeError("writer is closed")
        arr = _as_numpy(cake)
        if arr.shape == (self.n_eta, self.n_r):
            arr = arr.T
        elif arr.shape != (self.n_r, self.n_eta):
            raise ValueError(
                f"cake shape {arr.shape} is neither (n_eta, n_r) "
                f"{(self.n_eta, self.n_r)} nor (n_r, n_eta) "
                f"{(self.n_r, self.n_eta)}"
            )
        arr = np.ascontiguousarray(arr)

        idx = self._n_frames
        ome = float(idx) if omega is None else float(omega)
        self._omegas.append(ome)

        if self._res_grp is not None:
            self._res_grp.array(f"FrameNr_{idx}", data=arr,
                                dtype="float64", chunks=False)
            self._res_grp[f"FrameNr_{idx}"].attrs.update({
                "omega": ome,
                "Header": "Radius,Eta",
                "Units": "Pixels,Degrees",
            })

        if self.sum_images:
            # nan-safe: a masked bin is NAN in the C, and one masked frame
            # must not wipe the summed image everywhere.
            if self._sum is None:
                self._sum = np.zeros_like(arr)
            np.add(self._sum, np.nan_to_num(arr, nan=0.0), out=self._sum)

        if self._osf_grp is not None:
            if self._chunk is None:
                self._chunk = np.zeros_like(arr)
                self._chunk_first_ome = ome
                self._chunk_n = 0
            np.add(self._chunk, np.nan_to_num(arr, nan=0.0), out=self._chunk)
            self._chunk_n += 1
            for key, val in (("Temperature", temperature), ("Pressure", pressure),
                             ("I", current), ("I0", current_i0)):
                if val is not None:
                    self._chunk_meta[key].append(float(val))
            if (self.omega_sum_frames > 0
                    and self._chunk_n == self.omega_sum_frames):
                self._flush_chunk(last_frame=idx, last_ome=ome)

        self._n_frames += 1

    def _flush_chunk(self, *, last_frame: int, last_ome: float) -> None:
        """Write one ``/OmegaSumFrame/LastFrameNumber_<i>`` dataset."""
        if self._chunk is None or self._chunk_n == 0:
            return
        name = f"LastFrameNumber_{last_frame}"
        self._osf_grp.array(name, data=self._chunk, dtype="float64",
                            chunks=False)
        attrs: Dict[str, Any] = {
            "LastFrameNumber": int(last_frame),
            "Number Of Frames Summed": int(self._chunk_n),
            "FirstOme": float(self._chunk_first_ome),
            "LastOme": float(last_ome),
        }
        # The C averages these scalars over the frames in the chunk.
        for key in ("Temperature", "Pressure", "I", "I0"):
            vals = self._chunk_meta[key]
            if vals:
                attrs[key] = float(np.mean(vals))
        self._osf_grp[name].attrs.update(attrs)      # single write: see module docstring
        self._chunk = None
        self._chunk_n = 0
        self._chunk_first_ome = None
        self._chunk_meta = {k: [] for k in self._chunk_meta}

    def close(self) -> Path:
        """Flush the trailing chunk, write ``/Omegas`` + ``/SumFrames``, close."""
        if self._closed:
            return self.path
        if self._osf_grp is not None and self._chunk_n > 0:
            # Trailing partial chunk, and the omega_sum_frames == -1 case where
            # every frame accumulates into one dataset.
            self._flush_chunk(last_frame=self._n_frames - 1,
                              last_ome=self._omegas[-1])
        if self._omegas:
            self._root.array("Omegas",
                             data=np.asarray(self._omegas, dtype=np.float64),
                             dtype="float64", chunks=False)
            self._root["Omegas"].attrs.update({"Units": "Degrees"})
        if self.sum_images and self._sum is not None:
            self._root.array("SumFrames", data=self._sum,
                             dtype="float64", chunks=False)
            self._root["SumFrames"].attrs.update({
                "Header": "Radius,Eta",
                "Units": "Pixels,Degrees",
                "nFrames": int(self._n_frames),
            })
        ip = self._root.create_group("InstrumentParameters")
        for key, val in self._instrument_params.items():
            ip.array(key, data=np.array([val], dtype=np.float64),
                     dtype="float64", chunks=False)
        self._store.close()
        self._closed = True
        return self.path


def write_gsas_zarr_zip(
    path: Path | str,
    cakes: Iterable[Any],
    *,
    spec,
    omegas: Optional[Sequence[float]] = None,
    bin_area: Optional[Any] = None,
    instrument_params: Optional[Mapping[str, float]] = None,
    omega_sum_frames: int = 1,
    individual_save: bool = False,
    sum_images: bool = True,
    temperatures: Optional[Sequence[float]] = None,
    pressures: Optional[Sequence[float]] = None,
    currents: Optional[Sequence[float]] = None,
    currents_i0: Optional[Sequence[float]] = None,
) -> Path:
    """Write a MIDAS ``.zarr.zip`` from a sequence of integrated cakes.

    ``cakes`` may be any iterable of ``(n_eta, r)`` arrays or torch tensors —
    it is consumed lazily, so a generator keeps peak memory to one frame plus
    one ``OmegaSumFrame`` chunk. Returns the path written.
    """
    def _at(seq, i):
        return None if seq is None else seq[i]

    with GSASZarrWriter(path, spec=spec, bin_area=bin_area,
                        instrument_params=instrument_params,
                        omega_sum_frames=omega_sum_frames,
                        individual_save=individual_save,
                        sum_images=sum_images) as w:
        for i, cake in enumerate(cakes):
            w.add_frame(cake,
                        omega=_at(omegas, i),
                        temperature=_at(temperatures, i),
                        pressure=_at(pressures, i),
                        current=_at(currents, i),
                        current_i0=_at(currents_i0, i))
    return Path(path)
