"""The in-process (ctypes) backend vs the subprocess one.

Both call ``midas_tomo_run`` compiled from identical sources with identical
flags. That makes them agree to float32 rounding -- but *not* to the bit, and
the difference between those two claims is the point of this file.

Measured on chiltepin, three trials:

===================  ==========  ==========  ==========
planner              lib==lib    sub==sub    lib==sub
===================  ==========  ==========  ==========
FFTW_MEASURE (deflt) always      2 of 3      2 of 3
FFTW_ESTIMATE        always      always      never
===================  ==========  ==========  ==========

So the default planner is not reproducible run to run, and --deterministic
gives reproducibility only *within* a backend. Differences are ~3e-7 relative
throughout, i.e. a few ulps of float32.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_tomo import backend_c, backend_lib, run_tomo_from_sinos
from midas_tomo.api import run_engine

from .phantom import make_sino_dataset

pytestmark = pytest.mark.needs_binary

if not backend_lib.available():
    pytest.skip(
        f"libmidastomo not loadable: {backend_lib.why_unavailable()}",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def dataset():
    return make_sino_dataset(n=128, n_angles=180)


def test_library_and_subprocess_agree_to_float_precision(dataset, tmp_path):
    """The two backends agree to float32 rounding, NOT to the bit.

    Measured on chiltepin over three trials, with `deterministic=True` so each
    backend is individually reproducible: library and subprocess differ by
    ~2.9e-7 relative, *consistently* -- the same difference every trial, so it
    is systematic (plan selection differing between a fresh process and one
    that has already loaded numpy/h5py) rather than timing noise.

    An earlier version of this test asserted bitwise equality on the grounds
    that both backends compile from the same sources. They do; that is not
    sufficient. Identical code can still pick a different FFTW plan, and a
    different plan means a different order of floating-point operations.
    """
    _, sino, angles = dataset
    lib = run_tomo_from_sinos(sino, tmp_path / "lib", angles, n_cpus=2,
                              backend="library", deterministic=True, do_cleanup=False)
    sub = run_tomo_from_sinos(sino, tmp_path / "sub", angles, n_cpus=2,
                              backend="subprocess", deterministic=True, do_cleanup=False)
    scale = float(np.abs(sub).max())
    diff = float(np.abs(lib - sub).max())
    assert diff < 1e-5 * scale, (
        f"backends differ by {diff:.3e} (scale {scale:.3e}, rel {diff/scale:.2e}); "
        f"that is far beyond float32 rounding, so they are not computing the "
        f"same thing"
    )


def test_each_backend_is_self_consistent_under_deterministic(dataset, tmp_path):
    """Within one backend, --deterministic really is reproducible.

    This is the property --deterministic actually buys, and the one the
    default FFTW_MEASURE path does NOT have: see
    test_default_planner_is_not_reproducible_run_to_run.
    """
    _, sino, angles = dataset
    for backend in ("library", "subprocess"):
        runs = [
            run_tomo_from_sinos(sino, tmp_path / f"{backend}{k}", angles, n_cpus=2,
                                backend=backend, deterministic=True, do_cleanup=False)
            for k in range(2)
        ]
        np.testing.assert_array_equal(
            runs[0], runs[1],
            err_msg=f"{backend} backend was not reproducible under deterministic=True",
        )


def test_library_is_reusable_across_calls(dataset, tmp_path):
    """Repeated in-process calls must not accumulate state.

    The engine was written to run once per process. Anything it leaves behind
    -- FFTW plans, wisdom strings, static buffers -- now persists between
    calls, so the second call has to match the first exactly.
    """
    _, sino, angles = dataset
    a = run_tomo_from_sinos(sino, tmp_path / "a", angles, n_cpus=2,
                            backend="library", do_cleanup=False)
    b = run_tomo_from_sinos(sino, tmp_path / "b", angles, n_cpus=2,
                            backend="library", do_cleanup=False)
    c = run_tomo_from_sinos(sino, tmp_path / "c", angles, n_cpus=2,
                            backend="library", do_cleanup=False)
    np.testing.assert_array_equal(a, b, err_msg="2nd in-process call differed from the 1st")
    np.testing.assert_array_equal(a, c, err_msg="3rd in-process call differed from the 1st")


def test_library_survives_a_bad_parameter_file(tmp_path):
    """A malformed input must raise, not kill the interpreter.

    This is the whole reason the three exit(2) sites became error codes: in a
    subprocess a fatal error is an exit status, in-process it would take the
    caller's session with it. If this test crashes pytest rather than failing,
    that regression is back.
    """
    bad = tmp_path / "bad.par"
    bad.write_text("dataFileName /nonexistent/nope.bin\nreconFileName /tmp/x\n")
    with pytest.raises(Exception):
        backend_lib.run_param_file(bad, 1, cwd=tmp_path)


def test_auto_backend_prefers_the_library(dataset, tmp_path):
    """`auto` still picks the library when asked for explicitly.

    Note the package DEFAULT is "subprocess", not "auto" -- see run_engine's
    docstring for the state bug that keeps the library opt-in.
    """
    _, sino, angles = dataset
    par = tmp_path / "p"
    par.mkdir()
    run_tomo_from_sinos(sino, par, angles, n_cpus=2, do_cleanup=False)
    # run_engine reports which backend it used; "auto" should pick the library
    # whenever it loads.
    used = run_engine(par / "midastomo.par", 2, backend="auto", cwd=par)
    assert used == "library"


def test_explicit_subprocess_backend_is_honoured(dataset, tmp_path):
    _, sino, angles = dataset
    par = tmp_path / "p"
    par.mkdir()
    run_tomo_from_sinos(sino, par, angles, n_cpus=2, do_cleanup=False)
    used = run_engine(par / "midastomo.par", 2, backend="subprocess", cwd=par)
    assert used == "subprocess"


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="must be 'auto'"):
        run_engine("x.par", 1, backend="ctypes")


@pytest.mark.skipif(not backend_c.available(), reason="needs the CLI binary too")
def test_library_is_faster_than_subprocess(dataset, tmp_path, capsys):
    """Report the speedup. Informational -- asserts only that it is not slower.

    A hard threshold would be flaky on a loaded machine, and the point of the
    in-process path is avoiding disk staging, which this small phantom barely
    exercises.
    """
    import time

    _, sino, angles = dataset
    for be in ("library", "subprocess"):   # warm both paths
        run_tomo_from_sinos(sino, tmp_path / f"warm_{be}", angles, n_cpus=2,
                            backend=be, do_cleanup=False)

    timings = {}
    for be in ("library", "subprocess"):
        wd = tmp_path / f"bench_{be}"
        t0 = time.perf_counter()
        for _ in range(5):
            run_tomo_from_sinos(sino, wd, angles, n_cpus=2, backend=be,
                                do_cleanup=False)
        timings[be] = (time.perf_counter() - t0) / 5

    with capsys.disabled():
        print(f"\n  library    {timings['library']*1000:7.2f} ms/call")
        print(f"  subprocess {timings['subprocess']*1000:7.2f} ms/call")
        print(f"  speedup    {timings['subprocess']/timings['library']:7.2f}x")
    assert timings["library"] <= timings["subprocess"] * 1.5


def test_default_planner_is_not_reproducible_run_to_run(dataset, tmp_path):
    """Document the default planner's non-determinism as a measured fact.

    FFTW_MEASURE picks a plan by timing candidates, so two identical runs in
    two fresh directories can land on different plans and produce different
    low-order bits. Measured on chiltepin: two subprocess runs disagreed in 2
    of 3 trials.

    Asserted as a *tolerance*, not as inequality -- demanding that they differ
    would be just as wrong, since they agree whenever the planner happens to
    make the same choice. What must always hold is that any difference stays
    at rounding level. If this ever fails loudly, the planner is not the
    explanation and something real has broken.
    """
    _, sino, angles = dataset
    runs = [
        run_tomo_from_sinos(sino, tmp_path / f"m{k}", angles, n_cpus=2,
                            backend="subprocess", do_cleanup=False)
        for k in range(2)
    ]
    scale = float(np.abs(runs[0]).max())
    diff = float(np.abs(runs[0] - runs[1]).max())
    assert diff < 1e-5 * scale, (
        f"two default-planner runs differ by {diff:.3e} (rel {diff/scale:.2e}), "
        f"which is more than plan-selection rounding can explain"
    )


def test_varying_shift_counts_in_one_process(dataset, tmp_path):
    """Regression: reusing the engine in-process with different shift counts.

    This used to segfault. `initFFTMemoryStructures` nulled the FFT buffers but
    left the two `fftwf_plan` handles uninitialised, and
    `destroyFFTMemoryStructures` destroyed them unconditionally -- so a worker
    that never ran a transform of a given rank passed garbage to
    `fftwf_destroy_plan`. As a standalone binary the crash landed at teardown
    after the output was already written, so nothing noticed; in-process it
    killed the interpreter.

    The sequence matters: it takes a call with a DIFFERENT shift count to
    leave a worker with an unused plan handle. Identical repeated calls never
    triggered it, which is why the first reproduction attempt failed to.
    """
    _, sino, angles = dataset
    stack = np.stack([sino, sino[:, ::-1].copy()])
    for k, kw in enumerate([
        dict(shifts=0.0),
        dict(shifts=[-2, 3, 1]),      # 6 shifts
        dict(shifts=[-3, 4, 1]),      # 8 shifts
        dict(shifts=0.0, deterministic=True),
    ]):
        data = stack if k == 1 else sino
        run_tomo_from_sinos(data, tmp_path / f"s{k}", angles, n_cpus=2,
                            backend="library", do_cleanup=False, **kw)


def test_in_memory_sinograms_match_the_staged_file(dataset, tmp_path):
    """Passing sinograms by pointer must equal staging them to disk, bitwise.

    Same engine, same parameter file, same planner -- the only difference is
    where the sinogram bytes came from. Anything other than bitwise equality
    means the pointer path is reading the array differently (stride, dtype,
    slice offset), which is exactly the failure worth catching early.
    """
    _, sino, angles = dataset
    staged_dir = tmp_path / "staged"
    staged = run_tomo_from_sinos(sino, staged_dir, angles, n_cpus=2,
                                 backend="library", deterministic=True,
                                 do_cleanup=False)

    # Reuse the parameter file the staged run wrote, but feed the array in
    # memory. _pad_to_even duplicated the single slice, so mirror that.
    arr = np.ascontiguousarray(np.stack([sino, sino]), dtype=np.float32)
    mem_dir = tmp_path / "mem"
    mem_dir.mkdir()
    par = (staged_dir / "midastomo.par").read_text()
    par = par.replace(str(staged_dir / "output"), str(mem_dir / "output"))
    (mem_dir / "midastomo.par").write_text(par)

    backend_lib.run_param_file_with_sinos(
        mem_dir / "midastomo.par", arr, 2, deterministic=True, cwd=mem_dir,
    )

    from midas_tomo.api import read_recon_cube
    from midas_tomo.config import TomoConfig
    cfg = TomoConfig.from_param_file(mem_dir / "midastomo.par")
    in_memory, _ = read_recon_cube(cfg, 2)

    np.testing.assert_array_equal(
        staged[:, :1], in_memory[:, :1],
        err_msg="in-memory sinogram input differs from the staged-file path",
    )


def test_in_memory_rejects_an_undersized_buffer(dataset, tmp_path):
    """A buffer too small for the declared geometry must fail, not over-read."""
    _, sino, angles = dataset
    wd = tmp_path / "w"
    run_tomo_from_sinos(sino, wd, angles, n_cpus=2, backend="library",
                        do_cleanup=False)
    truncated = np.ascontiguousarray(sino[:, : sino.shape[1] // 2], dtype=np.float32)
    with pytest.raises(backend_lib.TomoLibraryError):
        backend_lib.run_param_file_with_sinos(
            wd / "midastomo.par", truncated, 2, cwd=wd,
        )


def test_fully_in_memory_matches_the_file_path(dataset, tmp_path):
    """Array in, array out, no data files -- and bitwise equal to via-disk.

    The last of the round-trips: no input.bin, no output cube on disk, and no
    parsing the result's shape back out of a filename.
    """
    from midas_tomo.config import TomoConfig

    _, sino, angles = dataset
    staged_dir = tmp_path / "staged"
    staged = run_tomo_from_sinos(sino, staged_dir, angles, n_cpus=2,
                                 backend="library", deterministic=True,
                                 do_cleanup=False)

    cfg = TomoConfig.from_param_file(staged_dir / "midastomo.par")
    arr = np.ascontiguousarray(np.stack([sino, sino]), dtype=np.float32)
    x = cfg.recon_xdim
    out = backend_lib.run_arrays(
        staged_dir / "midastomo.par", arr, (cfg.n_shifts, 2, x, x), 2,
        deterministic=True, cwd=tmp_path,
    )
    np.testing.assert_array_equal(
        staged[:, :1], out[:, :1],
        err_msg="fully in-memory result differs from the via-disk result",
    )


def test_in_memory_output_rejects_a_small_buffer(dataset, tmp_path):
    from midas_tomo.config import TomoConfig

    _, sino, angles = dataset
    wd = tmp_path / "w"
    run_tomo_from_sinos(sino, wd, angles, n_cpus=2, backend="library",
                        do_cleanup=False)
    cfg = TomoConfig.from_param_file(wd / "midastomo.par")
    arr = np.ascontiguousarray(np.stack([sino, sino]), dtype=np.float32)
    with pytest.raises(backend_lib.TomoLibraryError):
        backend_lib.run_arrays(
            wd / "midastomo.par", arr,
            (cfg.n_shifts, 1, cfg.recon_xdim, cfg.recon_xdim),  # 1 slice, not 2
            2, cwd=wd,
        )
