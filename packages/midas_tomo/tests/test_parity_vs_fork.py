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
    _, sino, angles = dataset

    shared = tmp_path / "shared"
    _stage(shared, sino, angles)
    ref = _run(REFERENCE_BIN, shared)
    pkg = _run(str(backend_c.binary_path()), shared)   # same dir, same wisdom

    np.testing.assert_array_equal(
        ref, pkg,
        err_msg="the --deterministic patch leaked into the default code path",
    )
