"""Threshold calibration: BlanketSigma, the float threshold, and the warning.

The reduction catalog on sample A (10 um grid, loop 0, identical seeds) is what
these encode: every good configuration sat near 3.5 sigma of the post-denoise
residual, while the SAME absolute ``BlanketSubtraction 2`` meant

    7.5 sigma unsummed   ->  75 distinct orientations recovered
    3.6 sigma on a 3-sum -> 412

Nothing in a parameter file distinguished those two, which is what the warning
exists to surface.
"""
import numpy as np
import pytest
import torch

from midas_nf_preprocess.process_images.params import ProcessParams
from midas_nf_preprocess.process_images.pipeline import ProcessImagesPipeline


def _params(**over):
    p = ProcessParams.__new__(ProcessParams)
    for f in p.__dataclass_fields__.values():           # dataclass defaults
        setattr(p, f.name, f.default if f.default is not None else None)
    p.nr_pixels_y = p.nr_pixels_z = 64
    p.nr_files_per_distance = 8
    p.n_distances = 1
    p.blanket_subtraction = 2.0
    p.blanket_sigma = 0.0
    p.nlm_denoise = 0
    p.do_log_filter = 0
    p.mean_filt_radius = 0
    for k, v in over.items():
        setattr(p, k, v)
    return p


def _noisy(sigma, n=8, size=64, seed=0):
    """Stack whose median-subtracted residual has a known sigma_MAD."""
    g = torch.Generator().manual_seed(seed)
    base = torch.zeros(size, size)
    stack = base + sigma * torch.randn(n, size, size, generator=g)
    return stack, torch.zeros(size, size)


def test_blanket_subtraction_parses_as_float():
    """The int cast was a sensitivity FLOOR, not a real constraint.

    sigma_MAD of an unsummed NLM-denoised residual is ~0.27 counts, so the
    smallest legal int threshold (1) is already 3.7 sigma -- nothing below that
    could be expressed at all.
    """
    assert ProcessParams.__dataclass_fields__["blanket_subtraction"].type in (
        "float", float)
    p = _params(blanket_subtraction=0.5)
    assert p.blanket_subtraction == pytest.approx(0.5)


def test_measured_sigma_matches_known_input():
    pipe = ProcessImagesPipeline(_params(), device="cpu")
    stack, median = _noisy(0.25)
    got = pipe.measure_threshold_sigma(stack, median)
    assert got == pytest.approx(0.25, rel=0.15)


def test_blanket_sigma_sets_threshold_from_noise():
    """BlanketSigma 3.5 on sigma 0.2 must give 0.7, not an integer."""
    pipe = ProcessImagesPipeline(
        _params(blanket_sigma=3.5, blanket_subtraction=999.0), device="cpu")
    stack, median = _noisy(0.2)
    thr = pipe._resolve_threshold(stack, median, 1)
    assert thr == pytest.approx(3.5 * 0.2, rel=0.2)
    assert thr != 999.0, "BlanketSigma must override the absolute value"


def test_warns_when_threshold_is_far_above_the_noise():
    """The 7.5-sigma case: 'BlanketSubtraction 2' on an unsummed residual."""
    pipe = ProcessImagesPipeline(
        _params(blanket_subtraction=2.0), device="cpu")
    stack, median = _noisy(0.268)                     # measured on sample A
    with pytest.warns(RuntimeWarning, match=r"sigma of this layer"):
        thr = pipe._resolve_threshold(stack, median, 1)
    assert thr == 2.0, "warning must not change the reduction"


def test_no_warning_in_the_good_band():
    """The 3.6-sigma case: same key, on a 3-frame sum. Must stay quiet."""
    pipe = ProcessImagesPipeline(
        _params(blanket_subtraction=2.0), device="cpu")
    stack, median = _noisy(0.556)                     # measured on sum3
    with warnings_as_errors():
        assert pipe._resolve_threshold(stack, median, 1) == 2.0


def test_no_warning_when_threshold_disabled():
    """BlanketSubtraction 0 is a legitimate config, not a 0-sigma threshold."""
    pipe = ProcessImagesPipeline(
        _params(blanket_subtraction=0.0), device="cpu")
    stack, median = _noisy(0.2)
    with warnings_as_errors():
        assert pipe._resolve_threshold(stack, median, 1) == 0.0


def test_degenerate_sigma_falls_back_to_absolute():
    """Photon-starved data: an all-zero residual must not divide by ~0."""
    pipe = ProcessImagesPipeline(
        _params(blanket_subtraction=3.0), device="cpu")
    stack = torch.zeros(8, 64, 64)
    with warnings_as_errors():
        assert pipe._resolve_threshold(stack, torch.zeros(64, 64), 1) == 3.0


def test_blanket_sigma_warns_and_falls_back_on_degenerate_data():
    pipe = ProcessImagesPipeline(
        _params(blanket_sigma=3.5, blanket_subtraction=3.0), device="cpu")
    stack = torch.zeros(8, 64, 64)
    with pytest.warns(RuntimeWarning, match="essentially all"):
        assert pipe._resolve_threshold(stack, torch.zeros(64, 64), 1) == 3.0


class warnings_as_errors:
    """Assert no RuntimeWarning escapes the block."""

    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error", RuntimeWarning)
        return self

    def __exit__(self, *a):
        return self._ctx.__exit__(*a)
