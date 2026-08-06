"""Read externally-produced BCDI arrays into the :mod:`midas_2d.bcdi` chain.

The point of entry for data you did not simulate here: a 3-D array someone
else's code produced, in whatever container they had to hand. The awkward part
is never the file format, it is that "my FFT data" can mean three different
things, which need three different amounts of processing:

    ``"object"``     a real-space complex object  psi(r)      -> FFT, then |.|^2
    ``"amplitude"``  a far-field complex amplitude A(q)       -> |.|^2
    ``"intensity"``  |A(q)|^2 already, or measured counts     -> use as is

:func:`load_bcdi` reads the container; :meth:`BCDIData.to_intensity` applies
whichever of those is right. The kind is *inferred conservatively* -- a complex
array is ambiguous between "object" and "amplitude" and you will be asked to
say which, because guessing wrong silently produces a plausible, wrong answer.

Two other conventions have to be stated rather than guessed, so both are
explicit arguments with loud defaults:

``centered``
    Is q = 0 at the array centre (fftshift-ed, the usual way people store and
    look at BCDI data) or at index 0 (raw FFT order)? Default True.
``axis order``
    :mod:`midas_2d.bcdi` uses ``(detector column, detector row, rocking step)``.
    If the file is stored rocking-first -- common, since it is the acquisition
    order -- pass ``transpose=(2, 0, 1)`` or use :meth:`BCDIData.permute`.

Container support is by lazy import, so none of these are hard dependencies:

    .npy .npz            numpy                (always available)
    .h5 .hdf5 .cxi .nxs  h5py                 (``pip install h5py``)
    .mat                 h5py (v7.3) or scipy (``pip install scipy``)
    .tif .tiff           tifffile             (``pip install tifffile``)
    .bin .raw            numpy, needs explicit ``dtype`` and ``shape``

CXI is the coherent-imaging community standard; for those files the default
dataset path ``entry_1/data_1/data`` is tried first.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "BCDIData",
    "load_bcdi",
    "list_datasets",
]

_KINDS = ("object", "amplitude", "intensity")

# Dataset paths tried, in order, when none is given.
_DEFAULT_PATHS = (
    "entry_1/data_1/data",          # CXI standard
    "entry/data/data",              # NeXus
    "data", "intensity", "amplitude", "object", "psi", "diff", "diffraction",
)


@dataclass
class BCDIData:
    """A loaded BCDI array plus the conventions needed to interpret it."""

    array: Any                                  # torch tensor, 3-D
    kind: str                                   # one of _KINDS
    centered: bool = True
    source: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        import torch

        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}, got {self.kind!r}")
        self.array = torch.as_tensor(self.array)
        if self.array.dim() != 3:
            raise ValueError(
                f"expected a 3-D array, got shape {tuple(self.array.shape)}. "
                "BCDI needs the full rocking stack; for a single detector frame "
                "add a length-1 axis.")
        if self.kind == "intensity" and self.array.is_complex():
            raise ValueError("kind='intensity' but the array is complex")

    # ------------------------------------------------------------- reshaping
    def permute(self, order: Sequence[int]) -> "BCDIData":
        """Reorder axes into (detector column, detector row, rocking step)."""
        if sorted(order) != [0, 1, 2]:
            raise ValueError(f"order must be a permutation of (0,1,2), got {order}")
        return BCDIData(self.array.permute(*order).contiguous(), self.kind,
                        self.centered, self.source, dict(self.meta))

    def recenter(self, centered: bool = True) -> "BCDIData":
        """Move q = 0 between array centre and index 0."""
        import torch

        if centered == self.centered:
            return self
        a = torch.fft.fftshift(self.array) if centered else torch.fft.ifftshift(self.array)
        return BCDIData(a, self.kind, centered, self.source, dict(self.meta))

    # -------------------------------------------------------------- the chain
    def to_intensity(self):
        """``|A(q)|^2`` on the detector, whatever kind was loaded.

        Differentiable. For ``kind="object"`` this runs the forward transform
        with the :mod:`midas_2d.bcdi` sign convention
        (``A(q) = sum psi exp(-i q.r)``), so a displacement field must have been
        encoded as ``psi = s exp(-i G.u)``.
        """
        import torch

        from .bcdi import object_to_amplitude

        if self.kind == "intensity":
            return self.array
        A = object_to_amplitude(self.array, centered=self.centered) \
            if self.kind == "object" else self.array
        return A.real * A.real + A.imag * A.imag

    def summary(self) -> str:
        import torch

        a = self.array
        finite = torch.isfinite(a.abs() if a.is_complex() else a)
        mag = (a.abs() if a.is_complex() else a)[finite]
        pos = mag[mag > 0]
        dr = float(mag.max() / pos.min()) if pos.numel() else float("nan")
        lines = [
            f"  source        {self.source or '<in memory>'}",
            f"  kind          {self.kind}"
            + ("   (complex)" if a.is_complex() else "   (real)"),
            f"  shape         {tuple(a.shape)}   dtype {a.dtype}",
            f"  centered      {self.centered}"
            "   (q = 0 at the array centre)" if self.centered
            else f"  centered      {self.centered}   (q = 0 at index 0)",
            f"  magnitude     min {float(mag.min()):.4g}  max {float(mag.max()):.4g}"
            f"  dynamic range {dr:.3e}",
        ]
        if not bool(finite.all()):
            lines.append(f"  WARNING       {int((~finite).sum())} non-finite elements")
        return "\n".join(lines)


# ------------------------------------------------------------------- loading
def _require(mod: str, why: str):
    try:
        return __import__(mod)
    except ImportError as exc:                            # pragma: no cover
        raise ImportError(
            f"reading {why} needs {mod!r}, which midas-2d does not require by "
            f"default. Install it with: pip install {mod}") from exc


def list_datasets(path) -> list[str]:
    """Every 3-D dataset path inside an HDF5-family file, with shapes.

    Use this when :func:`load_bcdi` cannot find the array on its own -- it tells
    you what to pass as ``dataset``.
    """
    h5py = _require("h5py", "HDF5/CXI/NeXus files")
    found: list[str] = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
            found.append(f"{name}  {obj.shape}  {obj.dtype}")

    with h5py.File(str(path), "r") as f:
        f.visititems(visit)
    return found


def _load_array(path: Path, *, dataset, dtype, shape):
    """Container -> numpy array. Format dispatch only; no interpretation."""
    suffix = path.suffix.lower()
    numpy = __import__("numpy")

    if suffix == ".npy":
        return numpy.load(path)

    if suffix == ".npz":
        with numpy.load(path) as z:
            keys = list(z.keys())
            if dataset is not None:
                if dataset not in z:
                    raise KeyError(f"{dataset!r} not in {path.name}; has {keys}")
                return z[dataset]
            cand = [k for k in keys if z[k].ndim == 3]
            if len(cand) != 1:
                raise ValueError(
                    f"{path.name} has {len(cand)} 3-D arrays {cand}; "
                    "pass dataset= to choose one")
            return z[cand[0]]

    if suffix in (".h5", ".hdf5", ".cxi", ".nxs"):
        h5py = _require("h5py", "HDF5/CXI/NeXus files")
        with h5py.File(str(path), "r") as f:
            if dataset is not None:
                if dataset not in f:
                    raise KeyError(
                        f"{dataset!r} not in {path.name}. Available 3-D datasets:\n  "
                        + "\n  ".join(list_datasets(path)))
                return f[dataset][()]
            for cand in _DEFAULT_PATHS:
                if cand in f and getattr(f[cand], "ndim", 0) == 3:
                    return f[cand][()]
            avail = list_datasets(path)
            raise ValueError(
                f"could not find a 3-D dataset in {path.name} at any default "
                f"path. Pass dataset=<name>. Available:\n  " + "\n  ".join(avail)
                if avail else f"no 3-D dataset in {path.name}")

    if suffix == ".mat":
        # v7.3 .mat IS HDF5; older ones are not.
        try:
            h5py = __import__("h5py")
            with h5py.File(str(path), "r") as f:
                if dataset is not None:
                    return f[dataset][()]
                for k in f:
                    if getattr(f[k], "ndim", 0) == 3:
                        return f[k][()]
        except (ImportError, OSError):
            pass
        scipy_io = _require("scipy", "MATLAB v7 files").io
        md = scipy_io.loadmat(str(path))
        cand = {k: v for k, v in md.items()
                if not k.startswith("__") and getattr(v, "ndim", 0) == 3}
        if dataset is not None:
            return md[dataset]
        if len(cand) != 1:
            raise ValueError(f"{path.name} has 3-D variables {list(cand)}; "
                             "pass dataset= to choose one")
        return next(iter(cand.values()))

    if suffix in (".tif", ".tiff"):
        tifffile = _require("tifffile", "TIFF stacks")
        return tifffile.imread(str(path))

    if suffix in (".bin", ".raw", ""):
        if dtype is None or shape is None:
            raise ValueError(
                f"{path.name} is headerless: pass dtype= and shape= "
                "(e.g. dtype='float32', shape=(128,128,128))")
        a = numpy.fromfile(path, dtype=numpy.dtype(dtype))
        want = int(numpy.prod(shape))
        if a.size != want:
            raise ValueError(
                f"{path.name} holds {a.size} values of {dtype} but shape={tuple(shape)} "
                f"needs {want}. Wrong dtype or wrong shape.")
        return a.reshape(shape)

    raise ValueError(f"unsupported container {suffix!r} for {path.name}")


def load_bcdi(path, *, kind=None, dataset=None, centered: bool = True,
              transpose=None, dtype=None, shape=None) -> BCDIData:
    """Read a 3-D BCDI array from disk.

    Parameters
    ----------
    path : str or Path
    kind : {"object", "amplitude", "intensity"}, optional
        What the numbers mean. A **real** array is taken as ``"intensity"``.
        A **complex** array is genuinely ambiguous between a real-space object
        and a far-field amplitude, so it must be declared -- inferring it would
        silently apply an extra Fourier transform, or fail to apply a needed
        one, and either way the result looks plausible.
    dataset : str, optional
        Dataset name/path inside .npz/.h5/.cxi/.mat. Defaults try the CXI and
        NeXus standard locations; :func:`list_datasets` shows what is there.
    centered : bool
        True (default) if q = 0 is at the array centre.
    transpose : sequence of 3 ints, optional
        Axis permutation applied on load, to reach
        (detector column, detector row, rocking step). Rocking-first files want
        ``(1, 2, 0)``.
    dtype, shape :
        Required for headerless .bin/.raw only.

    Returns
    -------
    BCDIData
    """
    import numpy
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = numpy.ascontiguousarray(_load_array(path, dataset=dataset, dtype=dtype,
                                              shape=shape))
    if raw.ndim != 3:
        raise ValueError(f"{path.name} holds a {raw.ndim}-D array {raw.shape}; "
                         "BCDI needs 3-D (two detector axes + rocking)")

    is_complex = numpy.iscomplexobj(raw)
    if kind is None:
        if is_complex:
            raise ValueError(
                f"{path.name} is complex, which is ambiguous: it could be a "
                "real-space object (needs an FFT) or a far-field amplitude "
                "(does not). Pass kind='object' or kind='amplitude'. Guessing "
                "would produce a plausible but wrong pattern.")
        kind = "intensity"
    if kind == "intensity" and is_complex:
        raise ValueError(f"{path.name} is complex but kind='intensity' was given")

    # float32/complex64 keeps big arrays affordable; float64 stays float64.
    t = torch.from_numpy(raw if raw.dtype.kind in "fc"
                         else raw.astype(numpy.float32))
    data = BCDIData(t, kind=kind, centered=centered, source=str(path),
                    meta={"container": path.suffix.lower(), "dataset": dataset})
    if transpose is not None:
        data = data.permute(transpose)
    return data
