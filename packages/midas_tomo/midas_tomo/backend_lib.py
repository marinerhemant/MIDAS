"""In-process access to the gridrec engine via ctypes.

``midas_tomo_run()`` is the same entry point the CLI binary calls, so this
path and the subprocess path run identical code compiled from identical
sources with identical flags.

They still do not agree to the *bit*. Measured: ~3e-7 relative difference,
consistently, because plan selection differs between a fresh process and one
that has already loaded numpy and h5py. Identical code is not sufficient for
identical floating-point output when an autotuning FFT planner sits in the
middle. ``tests/test_library.py`` asserts agreement to float32 rounding.

Why this exists: the subprocess route stages every input to disk and parses
the output shape back out of a filename. On a 128x180 sinogram roughly 73% of
the wall time was spawn plus staging rather than reconstruction, and midas-dt
makes that worse by reconstructing many (Q, eta) channels.

Two things to know before relying on it:

* **No process boundary.** A segfault in the engine takes the interpreter with
  it, where the subprocess path would have returned a non-zero exit code. The
  three ``exit(2)`` sites are gone (they now return error codes), but this is
  still C. ``midas_tomo.api`` keeps the subprocess route available.
* **OpenMP.** The library links its own OpenMP runtime. Loading it into a
  process that already has one -- torch and numpy both ship one on macOS --
  can abort at load time. :func:`available` reports the failure rather than
  letting it kill the caller, and ``KMP_DUPLICATE_LIB_OK=TRUE`` is the usual
  workaround.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
from pathlib import Path

__all__ = [
    "LIBRARY_STEMS",
    "fft_engine_code",
    "FFT_ENGINES",
    "run_param_file_with_sinos",
    "run_arrays",
    "TomoLibraryError",
    "available",
    "library_path",
    "load",
    "run_param_file",
    "why_unavailable",
]

log = logging.getLogger(__name__)

#: Platform-specific shared-library file names, in search order.
LIBRARY_STEMS = ("libmidastomo.so", "libmidastomo.dylib", "midastomo.dll")

_lib = None
_load_error: str | None = None
_lock = threading.Lock()


class TomoLibraryError(RuntimeError):
    """The shared library is missing, unloadable, or reported a failure."""


def _candidates() -> list[Path]:
    out: list[Path] = []
    try:
        import importlib.resources

        base = Path(str(importlib.resources.files("midas_tomo") / "bin"))
        out += [base / n for n in LIBRARY_STEMS]
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for prefix in {sys.prefix, sys.exec_prefix}:
        base = Path(prefix) / "lib" / pyver / "site-packages" / "midas_tomo" / "bin"
        out += [base / n for n in LIBRARY_STEMS]
    env = os.environ.get("MIDAS_TOMO_LIBRARY")
    if env:
        out.insert(0, Path(env))
    return out


def library_path() -> Path | None:
    """Path to the shared library, or ``None`` if it was not built."""
    for c in _candidates():
        if c.is_file():
            return c
    return None


def load(*, force: bool = False):
    """Load and return the ``ctypes.CDLL``, caching the result.

    Raises :class:`TomoLibraryError` if it cannot be loaded — including the
    OpenMP-collision case, which would otherwise abort the process during
    ``CDLL()``.
    """
    global _lib, _load_error
    with _lock:
        if _lib is not None and not force:
            return _lib
        path = library_path()
        if path is None:
            _load_error = (
                "libmidastomo was not built. It needs the same toolchain as the "
                "binary (C compiler, FFTW3f, OpenMP). Searched: "
                + ", ".join(str(c) for c in _candidates()[:3])
            )
            raise TomoLibraryError(_load_error)
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            _load_error = (
                f"could not load {path}: {exc}. If this mentions an OpenMP "
                f"runtime, the library's OpenMP is colliding with the one torch "
                f"or numpy already loaded; try KMP_DUPLICATE_LIB_OK=TRUE, or use "
                f"the subprocess backend."
            )
            raise TomoLibraryError(_load_error) from exc

        lib.midas_tomo_run.argtypes = [
            ctypes.c_char_p,  # paramFileName
            ctypes.c_int,     # requestedProcs
            ctypes.c_int,     # useGPU
            ctypes.c_int,     # useFftwBridge
            ctypes.c_int,     # useDeterministic
        ]
        lib.midas_tomo_run.restype = ctypes.c_int
        lib.midas_tomo_run_sinos.argtypes = [
            ctypes.c_char_p,                  # paramFileName
            ctypes.c_int,                     # requestedProcs
            ctypes.c_int,                     # useGPU
            ctypes.c_int,                     # useFftwBridge
            ctypes.c_int,                     # useDeterministic
            ctypes.POINTER(ctypes.c_float),   # sinos (may be NULL)
            ctypes.c_size_t,                  # sinoBytes
        ]
        lib.midas_tomo_run_sinos.restype = ctypes.c_int
        lib.midas_tomo_run_arrays.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,   # sinos in
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,   # recon out
        ]
        lib.midas_tomo_run_arrays.restype = ctypes.c_int
        # The superset entry point: in-memory I/O *and* an explicit FFT
        # backend. Bound because it is the only way to reach --fft-engine from
        # Python -- without it the CLI advertises a flag the package's own API
        # cannot set, and "use fftw to reproduce historical runs bit-for-bit"
        # is unreachable for every caller that is not shelling out.
        lib.midas_tomo_run_full.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,   # sinos in
            ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,   # recon out
            ctypes.c_int,                                      # fftEngine
        ]
        lib.midas_tomo_run_full.restype = ctypes.c_int
        lib.midas_tomo_error_message.argtypes = [ctypes.c_int]
        lib.midas_tomo_error_message.restype = ctypes.c_char_p

        _lib, _load_error = lib, None
        log.info("loaded %s", path)
        return lib


def available() -> bool:
    """``True`` if the shared library is present and loadable."""
    try:
        load()
        return True
    except TomoLibraryError:
        return False


def why_unavailable() -> str:
    """Why :func:`available` returned ``False``, or ``""`` if it did not."""
    try:
        load()
        return ""
    except TomoLibraryError as exc:
        return str(exc)


def error_message(code: int) -> str:
    """Translate an engine error code into a sentence."""
    try:
        lib = load()
    except TomoLibraryError:
        return f"engine error code {code}"
    raw = lib.midas_tomo_error_message(int(code))
    return raw.decode() if raw else f"engine error code {code}"


#: ``--fft-engine`` names, matching ``MIDAS_FFT_*`` in ``c_src/midas_fft.h``.
FFT_ENGINES = {"fftw": 0, "pocketfft": 1}

#: What the C picks when nothing is requested. Keep in step with tomo_init.c.
DEFAULT_FFT_ENGINE = "pocketfft"


def fft_engine_code(name: str | None) -> int:
    """Map an ``--fft-engine`` name to its C enum value.

    ``None`` means the engine's own default, which is pocketfft. Passing an
    unknown name raises rather than silently falling back -- a typo'd engine
    name that quietly selected the default would make a reproducibility run
    report success while using the wrong transform.
    """
    if name is None:
        name = DEFAULT_FFT_ENGINE
    key = str(name).strip().lower()
    if key not in FFT_ENGINES:
        raise ValueError(
            f"unknown FFT engine {name!r}; expected one of "
            f"{sorted(FFT_ENGINES)}. Default is {DEFAULT_FFT_ENGINE!r}."
        )
    return FFT_ENGINES[key]


def _refuse_gpu(gpu: bool) -> None:
    """Reject ``gpu=True`` on the library path instead of quietly using CPU.

    The shared library is built from the CPU sources and never links CUDA --
    the CUDA target is a separate executable, ``MIDAS_TOMO_GPU``. Passing
    ``useGPU=1`` into this library makes the C print a warning to stdout and
    reconstruct on the CPU, returning 0. Every caller that checks the return
    value sees success and gets a CPU answer labelled as a GPU one.

    That is not hypothetical: it made ``scripts/verify_gpu.py`` report a
    GPU-vs-CPU difference of exactly zero on a machine with two working A6000s.
    """
    if gpu:
        raise TomoLibraryError(
            "the midas-tomo shared library has no CUDA path (the CUDA target "
            "is the separate MIDAS_TOMO_GPU executable). Passing gpu=True here "
            "would silently reconstruct on the CPU and report success. Use "
            "midas_tomo.backend_c.run_binary(..., gpu=True), or the api-level "
            "backend='subprocess'."
        )


def _abs_param_file(param_file) -> str:
    """Absolute path to the parameter file, resolved in the *caller's* cwd.

    Every runner below chdir's into ``cwd`` before handing this string to the
    C, and the C just ``fopen``s it. A relative ``param_file`` -- which is what
    the api layer produces from a relative ``--out`` (``out/midastomo.par``) --
    stops resolving the moment the chdir lands, ``fopen`` returns NULL at
    ``tomo_utils.c:255``, and that is the one failure path in the C that prints
    no specific message: the user gets "Parameter file could not be read.
    Exiting." after a nonsense "Sinograms are not a power of 2" line. Resolve
    before the chdir, not after. Paths *inside* the file are already absolute
    (``TomoConfig.to_lines``).
    """
    return str(Path(param_file).expanduser().resolve())


def run_param_file(
    param_file: str | os.PathLike,
    n_cpus: int,
    *,
    gpu: bool = False,
    fftw_bridge: bool = False,
    deterministic: bool = False,
    fft_engine: str | None = None,
    cwd: str | os.PathLike | None = None,
) -> int:
    """Call the engine in-process.

    Signature mirrors :func:`midas_tomo.backend_c.run_binary` so the two
    backends are interchangeable.

    ``cwd`` is honoured by changing directory around the call, because the
    engine still resolves its wisdom-file cache relative to the *process*
    working directory. That makes this call **not thread-safe** against other
    code that depends on the cwd; a lock is held for the duration.
    ``param_file`` may be relative -- it is resolved before that chdir, see
    :func:`_abs_param_file`.

    ``gpu=True`` is refused -- see :func:`_refuse_gpu`.
    """
    _refuse_gpu(gpu)
    lib = load()
    param_file = _abs_param_file(param_file)
    prev = None
    try:
        if cwd is not None:
            _lock.acquire()
            prev = os.getcwd()
            os.chdir(str(cwd))
        # midas_tomo_run_full, not midas_tomo_run: the short entry point
        # hard-codes the FFT backend, so it cannot honour fft_engine.
        rc = lib.midas_tomo_run_full(
            param_file.encode(), int(n_cpus),
            1 if gpu else 0, 1 if fftw_bridge else 0, 1 if deterministic else 0,
            None, 0, None, 0, fft_engine_code(fft_engine),
        )
    finally:
        if prev is not None:
            os.chdir(prev)
            _lock.release()
    if rc != 0:
        raise TomoLibraryError(
            f"midas_tomo_run_full failed (code {rc}): {error_message(rc)}"
        )
    return rc


def run_param_file_with_sinos(
    param_file: str | os.PathLike,
    sinos,
    n_cpus: int,
    *,
    gpu: bool = False,
    fftw_bridge: bool = False,
    deterministic: bool = False,
    fft_engine: str | None = None,
    cwd: str | os.PathLike | None = None,
) -> int:
    """Run the engine reading sinograms straight from *sinos* in memory.

    ``sinos`` is a C-contiguous float32 array shaped ``(n_slices, n_thetas,
    det_xdim)``. It is passed by pointer -- not copied, not staged to disk --
    so the parameter file needs no ``dataFileName`` at all.

    The engine still writes its reconstruction to the file named by
    ``reconFileName``; only the input side is in memory. That is the half that
    matters for XRD-CT, where the sinogram stack is the large object.

    The array must stay alive for the duration of the call, which it does
    here because a reference is held on the stack.

    ``gpu=True`` is refused -- see :func:`_refuse_gpu`.
    """
    import numpy as np

    _refuse_gpu(gpu)
    lib = load()
    arr = np.ascontiguousarray(sinos, dtype=np.float32)
    ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    nbytes = ctypes.c_size_t(arr.nbytes)

    param_file = _abs_param_file(param_file)
    prev = None
    try:
        if cwd is not None:
            _lock.acquire()
            prev = os.getcwd()
            os.chdir(str(cwd))
        rc = lib.midas_tomo_run_full(
            param_file.encode(), int(n_cpus),
            1 if gpu else 0, 1 if fftw_bridge else 0, 1 if deterministic else 0,
            ptr, nbytes, None, 0, fft_engine_code(fft_engine),
        )
    finally:
        if prev is not None:
            os.chdir(prev)
            _lock.release()
    if rc != 0:
        raise TomoLibraryError(
            f"midas_tomo_run_full failed (code {rc}): {error_message(rc)}"
        )
    return rc


def run_arrays(
    param_file: str | os.PathLike,
    sinos,
    out_shape: tuple,
    n_cpus: int,
    *,
    fftw_bridge: bool = False,
    deterministic: bool = False,
    fft_engine: str | None = None,
    cwd: str | os.PathLike | None = None,
):
    """Fully in-memory reconstruction: array in, array out, no data files.

    ``sinos`` is ``(n_slices, n_thetas, det_xdim)`` float32. ``out_shape`` is
    the expected cube shape, normally ``(n_shifts, n_slices, X, X)`` — take it
    from ``TomoConfig.n_shifts`` and ``.recon_xdim`` rather than guessing.

    The parameter file is still read, for geometry and options; only the bulk
    data avoids the filesystem. Returns the allocated output array.

    Not available on the GPU path (its writer goes through an mmap'd file) or
    with ``saveReconSeparate``; the engine rejects both explicitly.
    """
    import numpy as np

    lib = load()
    arr = np.ascontiguousarray(sinos, dtype=np.float32)
    out = np.empty(out_shape, dtype=np.float32)

    param_file = _abs_param_file(param_file)
    prev = None
    try:
        if cwd is not None:
            _lock.acquire()
            prev = os.getcwd()
            os.chdir(str(cwd))
        rc = lib.midas_tomo_run_full(
            param_file.encode(), int(n_cpus), 0,
            1 if fftw_bridge else 0, 1 if deterministic else 0,
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(arr.nbytes),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(out.nbytes),
            fft_engine_code(fft_engine),
        )
    finally:
        if prev is not None:
            os.chdir(prev)
            _lock.release()
    if rc != 0:
        raise TomoLibraryError(
            f"midas_tomo_run_full failed (code {rc}): {error_message(rc)}"
        )
    return out
