"""make_seed must not reach for diplib by default.

diplib's MedianFilter segfaults on some real images on macOS (OpenMP runtime
conflict, documented in seed/__init__.py). A segfault is SIGSEGV, not a Python
exception, so the `try/except Exception` around the call cannot catch it: the
process exits 0 with no traceback and the caller just sees the run stop after
"auto-seeder launched". That happened on a real Pilatus 2M CdTe CeO2 frame.

seed/from_image.py already defaulted to False; auto_seed.py did not, and it is
the one make_seed exposes. Every caller inside the package already passes False
explicitly, so this pins the default rather than changing any behaviour.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest

from midas_calibrate_v2.seed import auto_seed
from midas_calibrate_v2.seed.auto_seed import make_seed


def test_make_seed_does_not_default_to_diplib():
    assert inspect.signature(make_seed).parameters["use_diplib"].default is False


def test_smooth_background_does_not_default_to_diplib():
    fn = getattr(auto_seed, "_smooth_background", None)
    if fn is None:                      # private helper, name may change
        cands = [f for n, f in vars(auto_seed).items()
                 if callable(f) and "use_diplib" in
                 getattr(inspect.signature(f), "parameters", {})]
        assert cands, "no helper takes use_diplib"
        fn = cands[0]
    assert inspect.signature(fn).parameters["use_diplib"].default is False


def test_the_two_seed_modules_agree_on_the_default():
    """They disagreed for a while, and the unsafe one was the public path."""
    from midas_calibrate_v2.seed import from_image
    a = inspect.signature(make_seed).parameters["use_diplib"].default
    cands = [f for n, f in vars(from_image).items()
             if callable(f) and "use_diplib" in
             getattr(inspect.signature(f), "parameters", {})]
    assert cands, "from_image exposes no use_diplib function"
    for f in cands:
        assert inspect.signature(f).parameters["use_diplib"].default == a, (
            f"{f.__name__} disagrees with make_seed on the use_diplib default")


def test_default_call_never_touches_diplib(monkeypatch):
    """The discriminating test: a default make_seed must not call into diplib.

    Replaces the module's diplib handle with an object that fails loudly if
    anything touches it. Under the old default this fires; the real bug killed
    the process instead, which no test can catch.
    """
    # RECORD the touch, do not raise. Raising is useless here and the reason is
    # the bug itself: the diplib call sits inside `try/except Exception`, so an
    # AssertionError from a probe gets SWALLOWED and the code quietly falls back
    # to scipy -- the test then passes under the OLD default too. (Measured: it
    # did.) The real failure is SIGSEGV, which that except cannot catch either.
    touched = []

    class Landmine:
        def __getattr__(self, name):
            touched.append(name)
            raise RuntimeError("probe")     # steer execution to the scipy path

    if getattr(auto_seed, "_HAS_DIPLIB", False):
        monkeypatch.setattr(auto_seed, "_DIPLIB", Landmine(), raising=False)
    else:
        monkeypatch.setattr(auto_seed, "_HAS_DIPLIB", True, raising=False)
        monkeypatch.setattr(auto_seed, "_DIPLIB", Landmine(), raising=False)

    rng = np.random.default_rng(0)
    img = rng.random((1200, 1100)) * 10.0
    yy, xx = np.mgrid[0:1200, 0:1100]
    r = np.hypot(yy - 600, xx - 550)
    for rad in (180.0, 260.0, 340.0):
        img += 400.0 * np.exp(-((r - rad) ** 2) / 8.0)
    make_seed(img.astype(np.float32), wavelength_A=0.42459, px_um=172.0,
              calibrant="CeO2")          # no use_diplib -> must use the default
    assert not touched, (
        f"make_seed reached diplib{touched} with default arguments")
