"""The pocketfft backend: correctness, agreement with FFTW, reproducibility.

Measured on chiltepin (128x180 phantom, gcc 11.5, FFTW 3.3.3):

    FFTW-free build reconstructs         corr +0.9278 (same as the FFTW build)
    pocketfft vs FFTW                    2.9e-07 relative, not bitwise
    pocketfft, two fresh runs            BITWISE IDENTICAL, no flag needed
    FFTW, two fresh runs                 NOT identical
    speed                                pocketfft 0.86x of FFTW (faster)
"""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from midas_tomo import TomoConfig, backend_c
from midas_tomo.api import read_recon_cube, run_tomo_from_sinos

from .phantom import make_sino_dataset

pytestmark = pytest.mark.needs_binary

if not backend_c.available():
    pytest.skip(f"engine not built: {backend_c.why_unavailable()}",
                allow_module_level=True)


def _usage() -> str:
    p = subprocess.run([str(backend_c.binary_path())], capture_output=True, text=True)
    return (p.stdout or "") + (p.stderr or "")


HAS_ENGINE_FLAG = "--fft-engine" in _usage()
pytestmark = [pytestmark, pytest.mark.skipif(
    not HAS_ENGINE_FLAG, reason="binary predates --fft-engine")]


@pytest.fixture(scope="module")
def dataset():
    return make_sino_dataset(n=128, n_angles=180)


def _run(wd, engine, sino, angles, extra=()):
    wd.mkdir(parents=True, exist_ok=True)
    run_tomo_from_sinos(sino, wd, angles, n_cpus=2, backend="subprocess",
                        do_cleanup=False)
    p = subprocess.run(
        [str(backend_c.binary_path()), str(wd / "midastomo.par"), "2",
         f"--fft-engine={engine}", *extra],
        cwd=str(wd), capture_output=True, text=True,
    )
    assert p.returncode == 0, p.stderr[-400:]
    assert f"FFT engine: {engine}" in p.stdout, "the engine flag was ignored"
    return read_recon_cube(TomoConfig.from_param_file(wd / "midastomo.par"), 2)[0]


def test_pocketfft_agrees_with_fftw_to_float_precision(dataset, tmp_path):
    """Same transform, different rounding. NOT bitwise -- asserting that would
    be asserting something false."""
    _, sino, angles = dataset
    f = _run(tmp_path / "f", "fftw", sino, angles)
    p = _run(tmp_path / "p", "pocketfft", sino, angles)
    scale = float(np.abs(f).max())
    diff = float(np.abs(f - p).max())
    assert diff < 1e-5 * scale, (
        f"engines differ by {diff:.3e} (rel {diff / scale:.2e}) -- far beyond "
        f"float32 rounding, so they are not computing the same transform"
    )


def test_pocketfft_reconstructs_the_phantom(dataset, tmp_path):
    phantom, sino, angles = dataset
    rec = _run(tmp_path / "p", "pocketfft", sino, angles)
    x, h = rec.shape[-1], 32
    crop = rec[0, 0][x // 2 - h:x // 2 + h, x // 2 - h:x // 2 + h]
    corr = np.corrcoef(crop.ravel(),
                       phantom[64 - h:64 + h, 64 - h:64 + h].ravel())[0, 1]
    assert corr > 0.85, f"pocketfft reconstruction correlates only {corr:.3f}"


def test_pocketfft_is_reproducible_without_any_flag(dataset, tmp_path):
    """The property FFTW does not have.

    pocketfft has no timing-based planner and no wisdom cache, so two fresh
    runs agree to the bit with no --deterministic needed. Measured: FFTW in
    the same situation does NOT.
    """
    _, sino, angles = dataset
    a = _run(tmp_path / "a", "pocketfft", sino, angles)
    b = _run(tmp_path / "b", "pocketfft", sino, angles)
    np.testing.assert_array_equal(
        a, b, err_msg="pocketfft was expected to be reproducible by construction"
    )


def test_deterministic_is_accepted_but_redundant_for_pocketfft(dataset, tmp_path):
    """--deterministic is an FFTW planner setting; with pocketfft it is a
    no-op rather than an error, and says so."""
    _, sino, angles = dataset
    wd = tmp_path / "d"
    wd.mkdir()
    run_tomo_from_sinos(sino, wd, angles, n_cpus=2, backend="subprocess",
                        do_cleanup=False)
    p = subprocess.run(
        [str(backend_c.binary_path()), str(wd / "midastomo.par"), "2",
         "--fft-engine=pocketfft", "--deterministic"],
        cwd=str(wd), capture_output=True, text=True,
    )
    assert p.returncode == 0
    assert "redundant" in p.stdout


def test_unknown_engine_is_rejected(dataset, tmp_path):
    _, sino, angles = dataset
    wd = tmp_path / "u"
    wd.mkdir()
    run_tomo_from_sinos(sino, wd, angles, n_cpus=2, backend="subprocess",
                        do_cleanup=False)
    p = subprocess.run(
        [str(backend_c.binary_path()), str(wd / "midastomo.par"), "2",
         "--fft-engine=fftw3"],
        cwd=str(wd), capture_output=True, text=True,
    )
    assert p.returncode != 0
    assert "unknown --fft-engine" in p.stderr
