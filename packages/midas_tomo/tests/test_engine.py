"""Tests that need the compiled MIDAS_TOMO binary.

Skipped wholesale where the engine was not built (no FFTW, no compiler, ...),
which is the normal state on a laptop.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_tomo import backend_c, run_tomo_from_sinos
from midas_tomo.center import find_center

from .phantom import make_sino_dataset

pytestmark = pytest.mark.needs_binary

if not backend_c.available():
    pytest.skip(
        f"MIDAS_TOMO not built: {backend_c.why_unavailable()}",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def dataset():
    # 128 / 180 rather than 64 / 90: the reference projector below is a
    # linear-splat Radon transform, and on a 64-grid its own error caps the
    # achievable correlation around 0.84 regardless of the engine.
    return make_sino_dataset(n=128, n_angles=180)


def test_reconstruction_recovers_phantom_structure(dataset, tmp_path):
    """The reconstruction must correlate strongly with the phantom.

    A correlation threshold, not an RMSE one: gridrec's output carries an
    arbitrary scale and offset relative to the input, so absolute agreement
    would be testing the normalisation rather than the reconstruction.
    """
    phantom, sino, angles = dataset
    cube = run_tomo_from_sinos(sino, tmp_path, angles, n_cpus=2)

    assert cube.shape[0] == 1                     # one shift
    assert cube.shape[1] == 1                     # one slice
    x = cube.shape[-1]
    assert x == 128                               # next_power_of_2(128)

    recon = cube[0, 0]
    # Crop both to the central region: the engine pads, so the outer ring is
    # empty and would dominate a whole-image correlation.
    n = phantom.shape[0]
    h = n // 4
    r = recon[x // 2 - h:x // 2 + h, x // 2 - h:x // 2 + h]
    p = phantom[n // 2 - h:n // 2 + h, n // 2 - h:n // 2 + h]
    corr = np.corrcoef(r.ravel(), p.ravel())[0, 1]
    # 0.85, not 0.99. The ceiling here is the reference projector's own
    # accuracy, not the engine's -- measured at ~0.93 for this phantom. This
    # asserts "the reconstruction is recognisably the object, in the right
    # orientation", which is what catches a convention or wiring regression.
    # It is NOT an accuracy claim about gridrec.
    assert corr > 0.85, f"reconstruction correlates only {corr:.3f} with the phantom"


def test_multiple_slices_are_independent(dataset, tmp_path):
    """Two different sinograms in one call must not bleed into each other."""
    phantom, sino, angles = dataset
    flipped = sino[:, ::-1].copy()
    stack = np.stack([sino, flipped])
    cube = run_tomo_from_sinos(stack, tmp_path, angles, n_cpus=2)
    assert cube.shape[1] == 2
    assert not np.allclose(cube[0, 0], cube[0, 1])


def test_shift_sweep_returns_one_slice_per_shift(dataset, tmp_path):
    _, sino, angles = dataset
    cube = run_tomo_from_sinos(sino, tmp_path, angles, shifts=[-2, 3, 1], n_cpus=2)
    # 6 shifts: the engine reconstructs shifts in pairs and rejects odd counts.
    assert cube.shape[0] == 6
    res = find_center(cube, (-2.0, 3.0, 1.0))
    assert res["best_shift"] in {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0}


def test_centred_data_prefers_a_small_shift(dataset, tmp_path):
    """A perfectly centred phantom should not want a large axis correction."""
    _, sino, angles = dataset
    cube = run_tomo_from_sinos(sino, tmp_path, angles, shifts=[-3, 4, 1], n_cpus=2)   # 8 shifts (even)
    res = find_center(cube, (-3.0, 4.0, 1.0))
    assert abs(res["best_shift"]) <= 1.0, (
        f"centred phantom picked shift {res['best_shift']}, which suggests a "
        f"convention mismatch rather than a centring error"
    )


@pytest.mark.skipif(
    not backend_c.supports_deterministic(),
    reason="this binary predates --deterministic",
)
def test_deterministic_is_bitwise_reproducible(dataset, tmp_path):
    """Two runs in *different fresh directories* must agree to the bit.

    This is the property the default path does not have: with FFTW_MEASURE the
    plan is chosen by timing, and the wisdom cache makes a cold run and a warm
    run take different paths.
    """
    _, sino, angles = dataset
    a = run_tomo_from_sinos(sino, tmp_path / "a", angles, n_cpus=2, deterministic=True)
    b = run_tomo_from_sinos(sino, tmp_path / "b", angles, n_cpus=2, deterministic=True)
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(
    not backend_c.supports_deterministic(),
    reason="this binary predates --deterministic",
)
def test_deterministic_writes_no_wisdom(dataset, tmp_path):
    """The FFTW_ESTIMATE path must not drop a planner cache in the cwd."""
    _, sino, angles = dataset
    wd = tmp_path / "clean"
    run_tomo_from_sinos(sino, wd, angles, n_cpus=2, deterministic=True,
                        do_cleanup=False)
    assert not list(wd.glob("fftwf_wisdom_*")), (
        "deterministic mode still wrote a wisdom file"
    )


@pytest.mark.skipif(
    not backend_c.supports_deterministic(),
    reason="this binary predates --deterministic",
)
def test_deterministic_agrees_with_default_to_float_precision(dataset, tmp_path):
    """Different plan, same transform: agreement to float32 rounding.

    Deliberately NOT asserting bitwise equality -- a different plan means a
    different order of floating-point operations, so the low-order bits are
    expected to differ. Asserting equality here is the mistake this test
    exists to prevent.
    """
    _, sino, angles = dataset
    est = run_tomo_from_sinos(sino, tmp_path / "est", angles, n_cpus=2,
                              deterministic=True)
    mea = run_tomo_from_sinos(sino, tmp_path / "mea", angles, n_cpus=2)
    scale = float(np.abs(mea).max())
    max_diff = float(np.abs(est - mea).max())
    assert max_diff < 1e-4 * scale, (
        f"FFTW_ESTIMATE and FFTW_MEASURE differ by {max_diff:.3e} "
        f"(scale {scale:.3e}) -- far more than rounding, so the two paths are "
        f"not computing the same transform"
    )


@pytest.mark.parametrize("backend", ["library", "subprocess"])
def test_relative_working_directory_reconstructs(dataset, tmp_path,
                                                 monkeypatch, backend):
    """A relative output directory must work, on both backends.

    ``midas-tomo-reconstruct --out out`` leaves ``out`` relative all the way
    down to ``workingdir / "midastomo.par"``. Both runners then chdir (or start
    the child) in ``workingdir`` before handing that string to the engine, so
    ``out/midastomo.par`` no longer resolved, ``fopen`` returned NULL, and the
    only report was "Parameter file could not be read. Exiting." -- preceded by
    a nonsense "Sinograms are not a power of 2. They will be increased to 1",
    which sends the reader after the wrong problem entirely.
    """
    from pathlib import Path

    from midas_tomo import backend_lib

    if backend == "library" and not backend_lib.available():
        pytest.skip(backend_lib.why_unavailable())

    phantom, sino, angles = dataset
    monkeypatch.chdir(tmp_path)
    cube = run_tomo_from_sinos(sino, Path("relative_out"), angles,
                               n_cpus=2, backend=backend)
    assert cube.shape[0] == 1
    assert cube.shape[1] == 1
    assert np.isfinite(cube).all()
    assert float(np.abs(cube).max()) > 0.0
