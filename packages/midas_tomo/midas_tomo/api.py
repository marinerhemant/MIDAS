"""High-level reconstruction entry points.

``run_tomo``
    From raw projections plus dark and white references.
``run_tomo_from_sinos``
    From sinograms that are already assembled (``areSinos=1``).

Both are thin wrappers: stage the inputs into a working directory, write a
:class:`~midas_tomo.config.TomoConfig`, run the binary, read the cube back.
The parameter-file writing that the two legacy functions each open-coded now
lives in one place, which is what stopped them drifting apart.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from . import backend_c, backend_lib
from .config import TomoConfig, parse_shift_arg

__all__ = [
    "run_tomo",
    "run_tomo_from_sinos",
    "write_thetas",
    "read_recon_cube",
    "run_engine",
]

log = logging.getLogger(__name__)


def write_thetas(thetas: Sequence[float], path: str | os.PathLike) -> Path:
    """Write rotation angles, one per line, in degrees."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in np.asarray(thetas, dtype=np.float64).ravel():
            f.write(f"{t}\n")
    return path


def read_recon_cube(
    cfg: TomoConfig, n_slices: int, *, n_cleanup: int | None = None
) -> tuple[np.ndarray, Path]:
    """Read the float32 cube the engine wrote, shaped from *cfg*.

    Returns ``(cube, path)``. Shape is ``(n_shifts, n_slices, X, X)``, or
    ``(n_cleanup, n_shifts, n_slices, X, X)`` in sweep mode.
    """
    path = cfg.output_path(n_slices, n_cleanup=n_cleanup)
    x = cfg.recon_xdim
    shape = (cfg.n_shifts, n_slices, x, x)
    if n_cleanup is not None:
        shape = (n_cleanup, *shape)
    count = int(np.prod(shape))

    if not path.is_file():
        raise FileNotFoundError(
            f"MIDAS_TOMO did not produce {path}. It exited successfully, so "
            f"the most likely cause is a shape mismatch between the config "
            f"and the staged input."
        )
    # Guard the reshape explicitly: np.fromfile silently returns a short array
    # when the file is smaller than requested, and the reshape error that
    # follows does not say which of the two is wrong.
    actual = path.stat().st_size // 4
    if actual != count:
        raise ValueError(
            f"{path.name} holds {actual} float32 values but the config implies "
            f"{count} ({'x'.join(str(s) for s in shape)}). Check det_xdim / "
            f"det_ydim / extra_pad / shift_values against the staged data."
        )
    cube = np.fromfile(path, dtype=np.float32, count=count).reshape(shape)
    return cube, path


def run_engine(
    param_file,
    n_cpus: int,
    *,
    backend: str = "auto",
    gpu: bool = False,
    fftw_bridge: bool = False,
    deterministic: bool = False,
    fft_engine: str | None = None,
    cwd=None,
) -> str:
    """Run the engine on *param_file*; return the backend actually used.

    ``backend``
        ``"library"`` calls ``midas_tomo_run`` in-process via ctypes: no
        process spawn, and no round-trip through the filesystem beyond the
        parameter file itself.
        ``"subprocess"`` spawns the CLI binary. Slower, but a crash in the C
        is contained by the process boundary instead of taking the
        interpreter down with it.
        ``"auto"`` uses the library when it loads and falls back to the
        subprocess otherwise -- notably when the library's OpenMP runtime
        collides with the one torch or numpy already loaded.

    The default is ``"auto"``. The library is ~2.6x faster on a small
    reconstruction and avoids staging large intermediates to disk, and the
    subprocess path remains the fallback whenever the shared object cannot be
    loaded -- which is the OpenMP-collision case, not a rare one.

    The two backends agree to float32 rounding, **not** to the bit; see
    ``tests/test_library.py`` for the measured numbers and why.
    """
    if backend not in {"auto", "library", "subprocess"}:
        raise ValueError(
            f"backend must be 'auto', 'library' or 'subprocess'; got {backend!r}"
        )

    # The shared library is CPU-only *by construction*: CMake builds
    # ``midastomo`` from the CPU sources, and the CUDA target is a separate
    # EXECUTABLE (``MIDAS_TOMO_GPU``), never a second .so. So a gpu=True
    # request routed through the library reconstructs on the CPU while
    # reporting success -- the C prints "built without CUDA" to stdout and
    # returns a CPU answer, which no caller checking a return value can tell
    # from a real GPU run.
    #
    # Measured on copland (2 x A6000, CUDA 12.8): scripts/verify_gpu.py
    # reported a GPU-vs-CPU difference of exactly 0.000e+00 and passed, because
    # both "backends" were the same CPU code. A GPU check that cannot fail is
    # worse than no GPU check.
    if gpu:
        if backend == "library":
            raise ValueError(
                "backend='library' cannot use the GPU: libmidastomo is built "
                "without CUDA (the CUDA target is the separate MIDAS_TOMO_GPU "
                "executable). Use backend='auto' or backend='subprocess'."
            )
        log.info("gpu=True: using the subprocess backend "
                 "(the shared library has no CUDA path)")

    if backend in {"auto", "library"} and not gpu:
        if backend_lib.available():
            backend_lib.run_param_file(
                param_file, n_cpus, gpu=gpu, fftw_bridge=fftw_bridge,
                deterministic=deterministic, fft_engine=fft_engine, cwd=cwd,
            )
            return "library"
        if backend == "library":
            raise backend_lib.TomoLibraryError(backend_lib.why_unavailable())
        log.info("library backend unavailable (%s); using subprocess",
                 backend_lib.why_unavailable())

    backend_c.run_binary(
        param_file, n_cpus, gpu=gpu, fftw_bridge=fftw_bridge,
        deterministic=deterministic, fft_engine=fft_engine, cwd=cwd,
    )
    return "subprocess"


def _cleanup_files(paths, workdir, cfg: TomoConfig) -> None:
    for p in list(paths) + cfg.wisdom_paths(workdir):
        try:
            Path(p).unlink()
        except (FileNotFoundError, IsADirectoryError):
            pass


def _pad_to_even(arr: np.ndarray, axis: int) -> tuple[np.ndarray, int]:
    """Duplicate the last entry along *axis* if its length is odd.

    The engine reconstructs slices in pairs, so an odd count would drop or
    misread the last one. Returns the array and the original length.
    """
    n = arr.shape[axis]
    if n % 2 == 0:
        return arr, n
    tail = np.take(arr, [n - 1], axis=axis)
    return np.concatenate([arr, tail], axis=axis), n


def run_tomo(
    data,
    dark,
    whites,
    workingdir,
    thetas,
    *,
    shifts=0.0,
    filter_nr: int = 2,
    do_log: bool = True,
    extra_pad: bool = False,
    auto_centering: bool = True,
    n_cpus: int = 40,
    do_cleanup: bool = True,
    ring_removal: float = 0.0,
    do_stripe_removal: bool = False,
    stripe_snr: float = 3.0,
    stripe_la_size: int = 61,
    stripe_sm_size: int = 21,
    use_gpu: bool = False,
    fftw_bridge: bool = False,
    deterministic: bool = False,
    fft_engine: str | None = None,
    backend: str = "auto",
) -> np.ndarray:
    """Reconstruct from raw projections.

    Parameters
    ----------
    data : ndarray, shape (n_thetas + 2, n_slices, xdim)
        Raw projections. The engine skips the first two frames (it subtracts
        2 from the theta count internally to account for the dark/white
        preamble), so pass the stack with those two leading frames present.
    dark : ndarray, shape (n_slices, xdim) or (xdim,)
        Dark-field reference.
    whites : ndarray, shape (2, n_slices, xdim) or (2, xdim)
        Two white-field references; the engine interpolates between them.
    workingdir : path
        Scratch and output directory; created if missing.
    thetas : 1-D array
        Rotation angles in degrees.
    shifts : float or (start, end, step)
        Rotation-axis shift, or a range to sweep.
    filter_nr : int
        0 none, 1 Shepp-Logan, 2 Hann (default), 3 Hamming, 4 ramp.
    do_log : bool
        Take -log of transmission. Set False when the input is already an
        intensity you want back-projected directly (the XRD-CT case).
    extra_pad : bool
        Pad to 2x the next power of two instead of 1x.
    n_cpus : int
        OpenMP thread count.
    use_gpu, fftw_bridge, deterministic
        See :mod:`midas_tomo.backend_c`.

    Returns
    -------
    ndarray, shape (n_shifts, n_slices, X, X)
        ``X`` is ``next_power_of_2(xdim)``, doubled when *extra_pad*.
    """
    t0 = time.time()
    workingdir = Path(workingdir)
    workingdir.mkdir(parents=True, exist_ok=True)

    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"data must be 3-D (theta, slice, x); got shape {data.shape}")
    n_thetas, n_slices, xdim = data.shape

    data, original_n_slices = _pad_to_even(np.asarray(data, dtype=np.float32), axis=1)
    n_slices = data.shape[1]

    # Staged layout the engine expects: dark (float32), two whites (float32),
    # then the projections as uint16.
    infn = workingdir / "input.bin"
    with infn.open("wb") as f:
        np.asarray(dark, dtype=np.float32).tofile(f)
        np.asarray(whites, dtype=np.float32).tofile(f)
        data.astype(np.uint16).tofile(f)

    thetas_fn = write_thetas(thetas, workingdir / "midastomo_thetas.txt")
    start, end, step, _ = parse_shift_arg(shifts)

    cfg = TomoConfig(
        data_file=infn,
        recon_file=workingdir / "output",
        are_sinos=False,
        det_xdim=xdim,
        det_ydim=n_slices,
        theta_file=thetas_fn,
        filter_nr=filter_nr,
        shift_values=(start, end, step),
        do_log=do_log,
        extra_pad=extra_pad,
        auto_centering=auto_centering,
        ring_removal_coeff=ring_removal or None,
        do_stripe_removal=do_stripe_removal,
        stripe_snr=stripe_snr,
        stripe_la_size=stripe_la_size,
        stripe_sm_size=stripe_sm_size,
        deterministic=deterministic,
    )
    par_fn = cfg.to_param_file(workingdir / "midastomo.par")
    log.info("staged inputs in %.3fs", time.time() - t0)

    run_engine(
        par_fn, n_cpus, backend=backend, gpu=use_gpu, fftw_bridge=fftw_bridge,
        deterministic=deterministic, fft_engine=fft_engine, cwd=workingdir,
    )

    t1 = time.time()
    cube, outfn = read_recon_cube(cfg, n_slices)
    cube = cube[:, :original_n_slices]

    if do_cleanup:
        _cleanup_files([outfn, par_fn, thetas_fn, infn], workingdir, cfg)
    log.info("read result in %.3fs", time.time() - t1)
    return cube


def run_tomo_from_sinos(
    sinograms,
    workingdir,
    thetas,
    *,
    shifts=0.0,
    filter_nr: int = 2,
    do_log: bool = False,
    extra_pad: bool = False,
    auto_centering: bool = True,
    n_cpus: int = 1,
    do_cleanup: bool = True,
    ring_removal: float = 0.0,
    do_stripe_removal: bool = False,
    stripe_snr: float = 3.0,
    stripe_la_size: int = 61,
    stripe_sm_size: int = 21,
    use_gpu: bool = False,
    fftw_bridge: bool = False,
    deterministic: bool = False,
    fft_engine: str | None = None,
    backend: str = "auto",
) -> np.ndarray:
    """Reconstruct from pre-assembled sinograms.

    This is the entry point XRD-CT uses: one sinogram per (Q, eta) channel,
    already normalised, with ``do_log=False`` because the values are
    diffracted intensity rather than transmission.

    Parameters
    ----------
    sinograms : ndarray
        ``(n_thetas, xdim)`` for a single slice, or
        ``(n_slices, n_thetas, xdim)`` for a stack.
    thetas : 1-D array
        Rotation angles in degrees, length ``n_thetas``.

    Returns
    -------
    ndarray, shape (n_shifts, n_slices, X, X)
    """
    t0 = time.time()
    workingdir = Path(workingdir)
    workingdir.mkdir(parents=True, exist_ok=True)

    sinograms = np.asarray(sinograms, dtype=np.float32)
    if sinograms.ndim == 2:
        sinograms = sinograms[np.newaxis]
    if sinograms.ndim != 3:
        raise ValueError(
            f"sinograms must be 2-D or 3-D (slice, theta, x); got {sinograms.shape}"
        )

    n_slices, n_thetas, xdim = sinograms.shape
    if len(np.asarray(thetas).ravel()) != n_thetas:
        raise ValueError(
            f"thetas has {len(np.asarray(thetas).ravel())} entries but the "
            f"sinograms have {n_thetas} projection rows"
        )

    sinograms, original_n_slices = _pad_to_even(sinograms, axis=0)
    n_slices = sinograms.shape[0]

    infn = workingdir / "input_sino.bin"
    sinograms.tofile(infn)

    thetas_fn = write_thetas(thetas, workingdir / "midastomo_thetas.txt")
    start, end, step, _ = parse_shift_arg(shifts)

    cfg = TomoConfig(
        data_file=infn,
        recon_file=workingdir / "output",
        are_sinos=True,
        det_xdim=xdim,
        det_ydim=n_slices,
        theta_file=thetas_fn,
        filter_nr=filter_nr,
        shift_values=(start, end, step),
        do_log=do_log,
        extra_pad=extra_pad,
        auto_centering=auto_centering,
        ring_removal_coeff=ring_removal or None,
        do_stripe_removal=do_stripe_removal,
        stripe_snr=stripe_snr,
        stripe_la_size=stripe_la_size,
        stripe_sm_size=stripe_sm_size,
        deterministic=deterministic,
    )
    par_fn = cfg.to_param_file(workingdir / "midastomo.par")
    log.info("staged inputs in %.3fs", time.time() - t0)

    run_engine(
        par_fn, n_cpus, backend=backend, gpu=use_gpu, fftw_bridge=fftw_bridge,
        deterministic=deterministic, fft_engine=fft_engine, cwd=workingdir,
    )

    t1 = time.time()
    cube, outfn = read_recon_cube(cfg, n_slices)
    cube = cube[:, :original_n_slices]

    if do_cleanup:
        _cleanup_files([outfn, par_fn, thetas_fn, infn], workingdir, cfg)
    log.info("read result in %.3fs", time.time() - t1)
    return cube
