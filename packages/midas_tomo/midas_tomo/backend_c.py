"""Locating and running the bundled gridrec binaries.

Two binaries may be present under ``<site-packages>/midas_tomo/bin/``:

``MIDAS_TOMO``
    The CPU gridrec engine. Built whenever a C compiler, FFTW3f, HDF5 and
    OpenMP were all available at ``pip install`` time.
``MIDAS_TOMO_GPU``
    The CUDA variant. Built only where ``nvcc`` and the CUDA toolkit exist.

Both are optional. ``pip install midas-tomo`` never fails on a missing
dependency — :func:`available` reports the situation and
:func:`why_unavailable` says what to install.

CLI contract (see ``c_src/tomo_init.c``)::

    MIDAS_TOMO ParamsFile.txt numberOfParallelJobs [--gpu] [--fftw-bridge]

``--fftw-bridge`` makes the GPU path use CPU FFTW for its transforms, which
is how the CPU and GPU results are checked against each other.
"""

from __future__ import annotations

import importlib.resources
import os
import subprocess
import sys
from pathlib import Path

__all__ = [
    "CPU_BINARY",
    "GPU_BINARY",
    "TomoBackendUnavailableError",
    "available",
    "binary_path",
    "run_binary",
    "supports_deterministic",
    "why_unavailable",
]

CPU_BINARY = "MIDAS_TOMO"
GPU_BINARY = "MIDAS_TOMO_GPU"


class TomoBackendUnavailableError(RuntimeError):
    """Raised when a reconstruction is requested but no binary is present."""


def _candidates(name: str) -> list[Path]:
    """Every place the binary could plausibly live, best guess first.

    1. ``importlib.resources`` — the standard scikit-build-core wheel layout.
    2/3. ``sys.prefix`` / ``sys.exec_prefix`` site-packages — covers the
       scikit-build-core *editable* install, where ``importlib.resources``
       resolves to the source tree (which has no ``bin/``) while the binary
       actually sits in site-packages.
    """
    out: list[Path] = []
    try:
        res = importlib.resources.files("midas_tomo") / "bin" / name
        out.append(Path(str(res)))
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for prefix in {sys.prefix, sys.exec_prefix}:
        out.append(Path(prefix) / "lib" / pyver / "site-packages" / "midas_tomo" / "bin" / name)
    return out


def binary_path(*, gpu: bool = False) -> Path:
    """Path to the requested binary.

    The file may not exist — use :func:`available` to test. When nothing is
    found the first candidate is returned so error messages can name a
    concrete path.
    """
    name = GPU_BINARY if gpu else CPU_BINARY
    cands = _candidates(name)
    for c in cands:
        if c.is_file():
            return c
    return cands[0] if cands else Path(name)


def available(*, gpu: bool = False) -> bool:
    """``True`` if the requested binary is present and executable."""
    p = binary_path(gpu=gpu)
    return p.is_file() and os.access(p, os.X_OK)


def why_unavailable(*, gpu: bool = False) -> str:
    """A sentence explaining why the binary is missing, and what to do.

    We cannot see the install-time CMake log from here, so this reports the
    likely causes rather than the actual one. The real reason was printed as
    a ``message(WARNING)`` during ``pip install``.
    """
    if available(gpu=gpu):
        return ""
    if gpu:
        if available(gpu=False):
            return (
                f"{GPU_BINARY} was not built: no CUDA toolkit was found at install "
                f"time. The CPU engine ({CPU_BINARY}) is available and will be used "
                f"instead."
            )
        return f"Neither {GPU_BINARY} nor {CPU_BINARY} was built. {why_unavailable()}"
    return (
        f"{CPU_BINARY} was not built at install time. It needs a C compiler, "
        f"FFTW3f, HDF5 and OpenMP. Install the missing pieces — e.g. "
        f"`conda install -c conda-forge fftw hdf5` (plus `brew install libomp` "
        f"on macOS) — then `pip install --force-reinstall --no-binary :all: "
        f"midas-tomo`. Expected location: {binary_path()}"
    )


def _usage_text(*, gpu: bool = False) -> str:
    """The binary's own usage banner, or ``""`` if it cannot be run."""
    if not available(gpu=gpu):
        return ""
    try:
        # Invoked with no arguments the binary prints usage and exits 1.
        proc = subprocess.run(
            [str(binary_path(gpu=gpu))],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return (proc.stdout or "") + (proc.stderr or "")


def supports_deterministic(*, gpu: bool = False) -> bool:
    """``True`` if this binary understands ``--deterministic``.

    Worth probing rather than assuming: ``tomo_init.c`` parses argv with a
    chain of ``strcmp`` and **silently ignores anything it does not
    recognise**. Passing the flag to an older binary would therefore appear to
    work while still planning with ``FFTW_MEASURE`` — a wrong answer with no
    error, which is exactly the failure mode worth spending a subprocess call
    to avoid.
    """
    return "--deterministic" in _usage_text(gpu=gpu)


def run_binary(
    param_file: str | os.PathLike,
    n_cpus: int,
    *,
    gpu: bool = False,
    fftw_bridge: bool = False,
    deterministic: bool = False,
    cwd: str | os.PathLike | None = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """Run a gridrec binary on *param_file*.

    Parameters
    ----------
    param_file
        Path to the MIDAS_TOMO parameter file. Paths *inside* it must be
        absolute — the binary does not resolve them relative to its own cwd.
    n_cpus
        OpenMP thread count.
    gpu
        Use ``MIDAS_TOMO_GPU``. Falls back to the CPU binary with a warning
        if the GPU one is absent but the CPU one is present.
    fftw_bridge
        GPU only: compute the FFTs with CPU FFTW so the result is bitwise
        comparable to the CPU path. Slower; used for parity testing.
    deterministic
        Plan with ``FFTW_ESTIMATE`` instead of ``FFTW_MEASURE``: reproducible
        across runs and machines, and no wisdom file written. Raises if the
        installed binary does not advertise the flag, rather than letting it
        be silently ignored.
    cwd
        Working directory for the subprocess. Worth setting: the engine writes
        its ``fftwf_wisdom_*.txt`` planner cache into the *process* working
        directory, so leaving this as None scatters wisdom files wherever the
        caller happened to be.
    check
        Raise ``CalledProcessError`` on a non-zero exit status.

    Raises
    ------
    TomoBackendUnavailableError
        If no usable binary is installed.
    """
    import warnings

    use_gpu = gpu
    if gpu and not available(gpu=True):
        if available(gpu=False):
            warnings.warn(
                f"{GPU_BINARY} not installed; falling back to {CPU_BINARY}. "
                f"{why_unavailable(gpu=True)}",
                RuntimeWarning,
                stacklevel=2,
            )
            use_gpu = False
        else:
            raise TomoBackendUnavailableError(why_unavailable(gpu=True))

    if not available(gpu=use_gpu):
        raise TomoBackendUnavailableError(why_unavailable(gpu=use_gpu))

    cmd = [str(binary_path(gpu=use_gpu)), str(param_file), str(n_cpus)]
    if use_gpu:
        cmd.append("--gpu")
        if fftw_bridge:
            cmd.append("--fftw-bridge")
    elif fftw_bridge:
        raise ValueError("fftw_bridge only applies to the GPU path (gpu=True)")

    if deterministic:
        if not supports_deterministic(gpu=use_gpu):
            raise TomoBackendUnavailableError(
                "deterministic=True was requested but the installed "
                f"{GPU_BINARY if use_gpu else CPU_BINARY} does not support "
                "--deterministic. The binary would ignore the flag silently and "
                "still plan with FFTW_MEASURE, so this is refused rather than "
                "reported as success. Reinstall midas-tomo >= 0.1.0 built from "
                "the current c_src/."
            )
        cmd.append("--deterministic")

    return subprocess.run(
        cmd, cwd=None if cwd is None else str(cwd),
        check=check, capture_output=capture_output, text=True,
    )
