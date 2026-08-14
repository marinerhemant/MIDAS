"""Packaging parity: does the packaged build still produce the fork's numbers?

This is the gate that makes "midas-tomo is a repackage, not a rewrite" a
checkable claim rather than an assertion.

Set ``MIDAS_TOMO_REFERENCE_BIN`` to a MIDAS_TOMO built from the pristine
pre-fork ``TOMO/src`` (see ``scripts/build_reference_binary.sh``) and the tests
run; otherwise they skip.

Why not a committed golden file: the default engine plans with FFTW_MEASURE
and caches wisdom, so its output depends on the machine, the FFTW build and
whether a wisdom file was present. Bytes frozen on one machine mean nothing on
another. Building both binaries on the *same* machine with identical flags
isolates the one variable under test -- the packaging change.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from midas_tomo import TomoConfig, backend_c
from midas_tomo.api import read_recon_cube, write_thetas

from .phantom import make_sino_dataset

REFERENCE_BIN = os.environ.get("MIDAS_TOMO_REFERENCE_BIN", "")

pytestmark = pytest.mark.needs_binary

if not backend_c.available():
    pytest.skip(f"MIDAS_TOMO not built: {backend_c.why_unavailable()}",
                allow_module_level=True)
if not REFERENCE_BIN or not Path(REFERENCE_BIN).is_file():
    pytest.skip(
        "set MIDAS_TOMO_REFERENCE_BIN to a pre-fork build "
        "(scripts/build_reference_binary.sh) to run packaging-parity tests",
        allow_module_level=True,
    )


def _usage(binary: str) -> str:
    p = subprocess.run([str(binary)], capture_output=True, text=True)
    return (p.stdout or "") + (p.stderr or "")


def _resolved_libfftw(binary: str) -> str | None:
    """Which libfftw3f the dynamic loader will actually hand this binary.

    ``None`` if it cannot be determined (no ``ldd``, e.g. macOS) -- callers
    treat that as "cannot check" rather than "they match".
    """
    if not shutil.which("ldd"):
        return None
    try:
        out = subprocess.run(["ldd", str(binary)], capture_output=True,
                             text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "libfftw3f" in line and "=>" in line:
            return line.split("=>", 1)[1].strip().split(" ")[0]
    return None


def _digest(path: str) -> str | None:
    """sha256 of a file, or ``None`` if it cannot be read."""
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _require_same_fftw() -> None:
    """Fail with the RIGHT diagnosis when the two binaries load different FFTWs.

    This cost an hour once, so it is worth a check. ``build_reference_binary.sh``
    links with a bare ``-lfftw3f`` and embeds no RPATH, so the reference falls
    back to ``/lib64/libfftw3f.so.3``. The CMake-built package embeds an RPATH
    to whatever FFTW it was configured against. With no ``LD_LIBRARY_PATH``,
    the two therefore run *different builds of FFTW* and disagree by ~2 float32
    ULP (2.8e-07 relative, deterministic and reproducible within each binary).

    Left unchecked, the comparison then reports "the packaged build no longer
    reproduces the pre-fork binary" -- pointing at c_src, the compile flags and
    FORK.txt, none of which is wrong. Parity itself is fine: exporting
    LD_LIBRARY_PATH to either copy of the custom FFTW makes both binaries load
    the same library and the reconstructions become bitwise identical.
    """
    pkg = _resolved_libfftw(str(backend_c.binary_path()))
    ref = _resolved_libfftw(REFERENCE_BIN)
    if pkg is None or ref is None:
        return                       # cannot check; do not pretend otherwise

    # Compare CONTENT, not the path. On this filesystem the same FFTW is
    # reachable as both /home/beams/... and /home/beams12/..., and the packaged
    # binary carries a DT_RPATH (which beats LD_LIBRARY_PATH) while the
    # reference has none. The two therefore report different paths to the same
    # bytes, and a path comparison would fail a perfectly good environment.
    pkg_h, ref_h = _digest(pkg), _digest(ref)
    if pkg_h is None or ref_h is None or pkg_h == ref_h:
        return
    pytest.fail(
        "ENVIRONMENT, not a parity regression: the two binaries load "
        "different builds of FFTW, which disagree by ~2 float32 ULP.\n"
        f"  packaged  -> {pkg}  (sha256 {pkg_h[:12]})\n"
        f"  reference -> {ref}  (sha256 {ref_h[:12]})\n"
        "Point both at one FFTW and re-run, e.g.\n"
        f"  export LD_LIBRARY_PATH={Path(pkg).parent}:$LD_LIBRARY_PATH\n"
        "Only if they still differ after that is c_src/ or FORK.txt worth "
        "looking at."
    )


def _stage(workdir: Path, sino: np.ndarray, angles: np.ndarray) -> TomoConfig:
    """Write the sinogram, angles and parameter file into *workdir*."""
    workdir.mkdir(parents=True, exist_ok=True)
    sino3 = np.ascontiguousarray(
        np.stack([sino, sino]).astype(np.float32)     # even slice count
    )
    infn = workdir / "input_sino.bin"
    sino3.tofile(infn)
    cfg = TomoConfig(
        data_file=infn,
        recon_file=workdir / "output",
        are_sinos=True,
        det_xdim=sino.shape[1],
        det_ydim=sino3.shape[0],
        theta_file=write_thetas(angles, workdir / "thetas.txt"),
        filter_nr=2,
        do_log=False,
    )
    cfg.to_param_file(workdir / "p.par")
    return cfg


def _run(binary: str, workdir: Path, extra: list[str] | None = None,
         *, engine: str | None = "fftw") -> np.ndarray:
    """Run *binary*; by default force the FFTW engine.

    The package default is pocketfft now, but the pre-fork reference predates
    the flag and can only use FFTW. Comparing the two therefore means asking
    the packaged binary for FFTW explicitly -- otherwise this test would be
    measuring the engine difference (~3e-7) rather than the packaging change.
    The reference binary rejects the flag, so it is only passed when the
    binary advertises it.
    """
    args = list(extra or [])
    if engine and "--fft-engine" in _usage(binary):
        args.append(f"--fft-engine={engine}")
    proc = subprocess.run(
        [str(binary), str(workdir / "p.par"), "2", *args],
        cwd=workdir, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"{binary} failed:\n{proc.stdout}\n{proc.stderr}"
    cfg = TomoConfig.from_param_file(workdir / "p.par")
    cube, _ = read_recon_cube(cfg, 2)
    return cube


@pytest.fixture(scope="module")
def dataset():
    return make_sino_dataset(n=128, n_angles=180)


def test_packaged_build_matches_the_fork_bitwise(dataset, tmp_path):
    """Same wisdom, same machine, same flags -> identical bytes.

    Both runs share one working directory so they see the same wisdom file:
    the reference runs first and creates it, the packaged build then reads it.
    Without that, one would plan with FFTW_MEASURE and the other with
    FFTW_WISDOM_ONLY, and any difference would be the planner rather than the
    code.
    """
    _require_same_fftw()
    _, sino, angles = dataset

    ref_dir = tmp_path / "shared"
    _stage(ref_dir, sino, angles)
    ref = _run(REFERENCE_BIN, ref_dir)

    wisdom = sorted(ref_dir.glob("fftwf_wisdom_*"))
    assert wisdom, "reference run produced no wisdom file; the pin is not established"

    pkg_dir = tmp_path / "packaged"
    _stage(pkg_dir, sino, angles)
    for w in wisdom:                       # carry the pin across
        shutil.copy2(w, pkg_dir / w.name)
    pkg = _run(str(backend_c.binary_path()), pkg_dir)

    np.testing.assert_array_equal(
        ref, pkg,
        err_msg=(
            "the packaged build no longer reproduces the pre-fork binary. "
            "Something in c_src/, the compile flags, or the parameter file "
            "changed the numbers -- check c_src/FORK.txt for an undeclared "
            "divergence."
        ),
    )


def test_deterministic_flag_does_not_perturb_the_default_path(dataset, tmp_path):
    """Adding --deterministic must not have changed the DEFAULT behaviour.

    The divergence recorded in FORK.txt is opt-in. This runs the packaged
    build with no flags and requires it to match the reference exactly, which
    is what "opt-in" has to mean if the golden-file gate is to survive.
    """
    _require_same_fftw()
    _, sino, angles = dataset

    shared = tmp_path / "shared"
    _stage(shared, sino, angles)
    ref = _run(REFERENCE_BIN, shared)
    pkg = _run(str(backend_c.binary_path()), shared)   # same dir, same wisdom

    np.testing.assert_array_equal(
        ref, pkg,
        err_msg="the --deterministic patch leaked into the default code path",
    )
