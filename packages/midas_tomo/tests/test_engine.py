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
    # 6 shifts (even).
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


_needs_odd = pytest.mark.skipif(
    not (backend_c.supports_odd_shifts() and backend_c.supports_deterministic()),
    reason="this binary predates odd shift counts (issue #7) or --deterministic",
)


def test_scalar_shift_gives_one_shift(dataset, tmp_path):
    _, sino, angles = dataset
    cube = run_tomo_from_sinos(sino, tmp_path, angles, shifts=1.5, n_cpus=2)
    assert cube.shape[:2] == (1, 1)
    assert np.isfinite(cube).all() and np.any(cube)


@_needs_odd
@pytest.mark.parametrize("shifts, n", [([-2, 2, 1], 5), ([-1, 1, 1], 3),
                                       ([-10, 10, 1], 21)])
def test_odd_shift_count_reconstructs_every_shift(dataset, tmp_path, shifts, n):
    """Issue #7: an odd count must run and fill every (shift, slice) plane."""
    _, sino, angles = dataset
    stack = np.stack([sino, sino[:, ::-1], sino[::-1]])          # 3 slices
    cube = run_tomo_from_sinos(stack, tmp_path, angles, shifts=shifts,
                               n_cpus=4, deterministic=True)
    assert cube.shape[:2] == (n, 3)
    for i in range(n):
        for j in range(3):
            assert np.any(cube[i, j]), f"shift {i}, slice {j} is empty"


@_needs_odd
def test_odd_shift_count_matches_even_neighbour(dataset, tmp_path):
    """Odd and even sweeps share their pairs, so they share their bits.

    [-2, 2, 1] (5) and [-2, 3, 1] (6) pair shifts 0/1 and 2/3 identically in
    the dual-slot gridrec call, so those four planes must agree bitwise. The
    fifth is reconstructed alone in the odd sweep and alongside +3 in the even
    one; the two slots of a gridrec call are independent up to float32
    rounding, so that plane agrees to rounding, not to the bit.
    """
    _, sino, angles = dataset
    stack = np.stack([sino, sino[:, ::-1], sino[::-1]])
    odd = run_tomo_from_sinos(stack, tmp_path / "odd", angles, shifts=[-2, 2, 1],
                              n_cpus=4, deterministic=True)
    even = run_tomo_from_sinos(stack, tmp_path / "even", angles, shifts=[-2, 3, 1],
                               n_cpus=4, deterministic=True)
    np.testing.assert_array_equal(odd[:4], even[:4])
    scale = np.abs(even[4]).max()
    np.testing.assert_allclose(odd[4], even[4], rtol=0, atol=1e-5 * scale)


@_needs_odd
def test_odd_sweep_planes_match_single_shift_runs(dataset, tmp_path):
    """Each plane of an odd sweep is the reconstruction at THAT shift.

    Catches the old failure mode, where an odd count paired shifts across a
    slice boundary and wrote planes against the wrong (slice, shift).
    """
    _, sino, angles = dataset
    stack = np.stack([sino, sino[:, ::-1], sino[::-1]])
    sweep = run_tomo_from_sinos(stack, tmp_path / "sweep", angles,
                                shifts=[-1, 1, 1], n_cpus=4, deterministic=True)
    for i, s in enumerate((-1.0, 0.0, 1.0)):
        one = run_tomo_from_sinos(stack, tmp_path / f"s{i}", angles, shifts=s,
                                  n_cpus=4, deterministic=True)
        scale = np.abs(one).max()
        np.testing.assert_allclose(sweep[i], one[0], rtol=0, atol=1e-5 * scale)


# Ranges whose count the old C and the old Python disagreed on, or where the
# float arithmetic is inexact. The old engine's abs() was the tomo_heads.h
# macro, so the span was not truncated; the disagreements were at exact
# halves (C round() is half-away, Python round() half-even, and the C works in
# float32) and a negative step, where the old count went negative and the
# engine segfaulted.
_COUNT_RANGES = [
    (-2, 2.25, 0.25),         # 18
    (-10, 10, 1),             # 21
    (-1, 1, 0.1),             # 21
    (0.37 - 2, 0.37 + 2, 0.25),  # 17: workflow's fine sweep, float span
    (0, 2.5, 1),              # 4: old Python said 3
    (-10, -9.85, 0.1),        # 3: old C said 2
    (-10, -9.35, 0.1),        # 8: old C said 7
    (2, -2, -0.5),            # 9: old C crashed
]


@_needs_odd
@pytest.mark.parametrize("backend", ["library", "subprocess"])
@pytest.mark.parametrize("shifts", _COUNT_RANGES)
def test_engine_shift_count_matches_python(tmp_path, shifts, backend):
    """Issue #14: the engine must reconstruct exactly parse_shift_arg's count.

    read_recon_cube opens the file named with Python's count and checks its
    size, so an engine that counts differently fails here, as it did in use.
    """
    from midas_tomo import backend_lib
    from midas_tomo.config import parse_shift_arg

    if backend == "library" and not backend_lib.available():
        pytest.skip(backend_lib.why_unavailable())
    rng = np.random.default_rng(0)
    sino = rng.random((2, 36, 32)).astype(np.float32)
    angles = np.linspace(0.0, 175.0, 36)
    n = parse_shift_arg(list(shifts))[3]
    cube = run_tomo_from_sinos(sino, tmp_path, angles, shifts=list(shifts),
                               n_cpus=2, do_log=False, deterministic=True,
                               backend=backend, do_cleanup=False)
    assert cube.shape[:2] == (n, 2)
    assert list(tmp_path.glob(f"output_NrShifts_{n:03d}_*.bin"))
    for i in range(n):
        assert np.any(cube[i]), f"shift plane {i} is empty"


def _cleanup_sweep_from_sinos(stack, angles, wd, shifts, grid):
    """Drive the engine's stripeConfigFile sweep directly on sinograms."""
    from midas_tomo.api import read_recon_cube, run_engine, write_thetas
    from midas_tomo.config import TomoConfig

    wd.mkdir(parents=True, exist_ok=True)
    stack = np.ascontiguousarray(stack, dtype=np.float32)
    stack.tofile(wd / "in.bin")
    grid_fn = wd / "grid.txt"
    grid_fn.write_text("# snr la sm\n" + "".join(f"{s} {la} {sm}\n" for s, la, sm in grid))
    cfg = TomoConfig(
        data_file=wd / "in.bin", recon_file=wd / "out", are_sinos=True,
        det_xdim=stack.shape[2], det_ydim=stack.shape[0],
        theta_file=write_thetas(angles, wd / "th.txt"),
        shift_values=shifts, do_log=False, stripe_config_file=grid_fn,
        deterministic=True,
    )
    run_engine(cfg.to_param_file(wd / "p.par"), 4, deterministic=True, cwd=wd)
    return read_recon_cube(cfg, stack.shape[0], n_cleanup=len(grid))[0]


@_needs_odd
def test_cleanup_sweep_with_one_shift(dataset, tmp_path):
    """Issue #15: a stripeConfigFile sweep with ONE shift, and an odd slice count.

    Both used to be refused ("sweep requires n_shifts >= 2", then "Number of
    slices must be even"): the old paired loop reconstructed only every other
    slice at one shift. With shifts paired within a slice, one shift is one job
    per slice. Each plane must equal (to the float32 rounding between gridrec's
    two slots) the first shift of the two-shift sweep, and the plain
    no-sweep reconstruction with the same cleanup settings.
    """
    _, sino, angles = dataset
    stack = np.stack([sino, sino[:, ::-1], sino[::-1]]).astype(np.float32)
    stack[:, :, 40] *= 1.5                               # a stripe to clean up
    stack[:, :, 70] *= 0.6
    grid = [(0.0, 0, 0), (1.5, 31, 11)]                  # baseline, cleaned
    one = _cleanup_sweep_from_sinos(stack, angles, tmp_path / "one",
                                    (0.5, 0.5, 1.0), grid)
    two = _cleanup_sweep_from_sinos(stack, angles, tmp_path / "two",
                                    (0.5, 1.5, 1.0), grid)
    assert one.shape[:3] == (2, 1, 3)
    for c in range(2):
        for j in range(3):
            assert np.any(one[c, 0, j]), f"config {c}, slice {j} is empty"
            scale = np.abs(two[c, 0, j]).max()
            np.testing.assert_allclose(one[c, 0, j], two[c, 0, j], rtol=0,
                                       atol=1e-5 * scale)
    # the cleanup actually changed something, so the comparison has teeth
    assert not np.allclose(one[0], one[1])

    for c, (snr, la, sm) in enumerate(grid):
        plain = run_tomo_from_sinos(stack, tmp_path / f"plain{c}", angles,
                                    shifts=0.5, n_cpus=4, do_log=False,
                                    deterministic=True,
                                    do_stripe_removal=snr > 0,
                                    stripe_snr=snr or 3.0,
                                    stripe_la_size=la or 61,
                                    stripe_sm_size=sm or 21)
        scale = np.abs(plain).max()
        np.testing.assert_allclose(one[c, 0], plain[0], rtol=0, atol=1e-5 * scale)


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
