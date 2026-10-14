"""Bad pixels must reach the E-step, and a screen that eats the data must say so.

Two failures, one root. `calibrate()` found the -1/-2 gap sentinels in the RAW
frame, but only to mask the seeder; `autocalibrate_pv` took an optional mask and,
given none, did nothing, and the paper runners clamped every pixel below 1 up to
1 (`img[img < 1] = 1`) before calling it. A module gap then became a block of
valid pixels (1, or 2-3 after the raw dark is subtracted), so the cake's coverage map counted it as covered and
the clipped-window screen could not see it. On the Pilatus the wide capture
window accepted fits centred in the 7 and 17 px gaps, and the fit ended near
190 uE instead of 30.

The silent part: the screen can also remove nearly everything (a dark exceeding
its image left 50 of ~7500 candidate fits on GE) and the run still converged and
reported a strain.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from midas_calibrate.params import CalibrationParams as V1

from midas_calibrate_v2.pipelines import single_pv
from midas_calibrate_v2.pipelines.diagnostics import (
    fit_retention_gate, run_all_gates, worst_severity,
)
from midas_calibrate_v2.pipelines.single_pv import (
    FitRetentionWarning, _note_retention, _screen_retention, autocalibrate_pv,
)
from midas_calibrate_v2.seed.mask import apply_sentinel_mask, sentinel_mask_from_raw


# ------------------------------------------------------------- the helper
def test_signed_int_gap_and_dead_markers_are_masked():
    im = np.full((6, 8), 100, dtype=np.int32)
    im[2, 3] = -1
    im[4, 5] = -2
    m = sentinel_mask_from_raw(im)
    assert m is not None and m.dtype == bool
    assert m.sum() == 2 and m[2, 3] and m[4, 5]


def test_float_frames_use_the_same_markers():
    im = np.full((5, 5), 7.0)
    im[0, 0] = -1.0
    assert sentinel_mask_from_raw(im)[0, 0]


@pytest.mark.parametrize("dt", [np.uint16, np.uint32])
def test_unsigned_dtype_max_is_the_dead_pixel_marker(dt):
    im = np.full((4, 4), 10, dtype=dt)
    im[1, 1] = np.iinfo(dt).max
    m = sentinel_mask_from_raw(im)
    assert m.sum() == 1 and m[1, 1]


def test_a_clean_frame_gives_no_mask():
    assert sentinel_mask_from_raw(np.full((4, 4), 5, dtype=np.int32)) is None


def test_a_legitimate_zero_count_pixel_is_data_not_a_sentinel():
    im = np.full((4, 4), 5, dtype=np.int32)
    im[2, 2] = 0
    assert sentinel_mask_from_raw(im) is None


def test_clamping_erases_the_markers_which_is_why_the_runners_failed():
    """The harness bug, as a property: after `img[img < 1] = 1` nothing is left
    to find. The gap is now indistinguishable from valid pixels."""
    im = np.full((6, 8), 100, dtype=np.int32)
    im[2:4, 3:5] = -1
    assert sentinel_mask_from_raw(im) is not None
    clamped = im.astype(np.float64)
    clamped[clamped < 1] = 1
    assert sentinel_mask_from_raw(clamped) is None


# --------------------------------------------- the wiring into autocalibrate_pv
class _Stop(Exception):
    pass


def _v1():
    return V1(NrPixelsY=64, NrPixelsZ=48, pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
              BC_y=32.0, BC_z=24.0, Wavelength=0.17, SpaceGroup=225,
              LatticeConstant=(5.41,) * 3 + (90.0,) * 3, MaxRingRad=3000.0)


def _capture_mask(monkeypatch, image, **kw):
    seen = {}

    def fake_cake(params, img, rt, mask=None, **k):
        seen["mask"] = mask
        seen["img"] = img
        raise _Stop

    monkeypatch.setattr(single_pv, "integrate_cake", fake_cake)
    with pytest.raises(_Stop):
        autocalibrate_pv(_v1(), image, n_iter=1, auto_max_ring=False,
                         capture_window_px=0.0, verbose=False, **kw)
    return seen


def _gap_image(dtype=np.int32):
    im = np.full((48, 64), 100, dtype=dtype)
    im[20:23, :] = -1                                   # a module gap
    return im


def test_autocalibrate_pv_masks_the_sentinels_it_is_given(monkeypatch):
    seen = _capture_mask(monkeypatch, _gap_image())
    m = np.asarray(seen["mask"])
    assert m is not None and m.shape == (48, 64)
    assert m[20:23, :].all() and not m[:20, :].any() and not m[23:, :].any()
    assert (np.asarray(seen["img"])[20:23, :] == 0).all(), \
        "sentinels must not enter the cake as intensities"


def test_a_user_mask_is_kept_and_the_gaps_are_added_to_it(monkeypatch):
    """Union, not replacement: a user who masked a pixel keeps it masked, and
    the gaps are masked on top. Gaps must never get in because someone passed
    a mask."""
    user = np.zeros((48, 64), dtype=bool)
    user[0, 0] = True
    seen = _capture_mask(monkeypatch, _gap_image(), mask=user)
    m = np.asarray(seen["mask"])
    assert m[0, 0], "the caller's own mask was dropped"
    assert m[20:23, :].all(), "gaps were let in because a mask was supplied"


def test_a_user_mask_on_a_clean_frame_is_returned_unchanged(monkeypatch):
    user = np.zeros((48, 64), dtype=bool)
    user[5, 5] = True
    seen = _capture_mask(monkeypatch, np.full((48, 64), 100, dtype=np.int32),
                         mask=user)
    assert np.array_equal(np.asarray(seen["mask"]), user)


def test_a_frame_without_sentinels_is_untouched(monkeypatch):
    clean = np.full((48, 64), 100, dtype=np.int32)
    seen = _capture_mask(monkeypatch, clean)
    assert seen["mask"] is None


# ------------------------------------------------- retention accounting + warning
class _BF:
    def __init__(self, snr):
        self.snr = torch.as_tensor(snr, dtype=torch.float64)


def test_retention_counts_only_snr_passing_fits():
    """Off-detector cells produce a fit each with SNR ~0; counting them would
    make a screen that ate every real fit look healthy."""
    bf = _BF([10.0, 10.0, 10.0, 10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    keep = torch.tensor([True, False, False, False] + [True] * 6)
    n_pass, n_kept = _screen_retention(bf, keep, snr_min=3.0)
    assert (n_pass, n_kept) == (4, 1)


def test_warns_when_the_screen_ate_most_of_the_data():
    log = []
    with pytest.warns(FitRetentionWarning):
        _note_retention(log, "capture", 0, 1391, 35, 0.5)
    assert log == [dict(phase="capture", iteration=0, n_snr_pass=1391, n_kept=35)]


def test_silent_when_retention_is_healthy():
    log = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _note_retention(log, "fine", 2, 1000, 940, 0.5)
    assert log[0]["n_kept"] == 940


# ------------------------------------------------------------------- the gate
def _rec(phase, n_pass, n_kept):
    return dict(phase=phase, iteration=0, n_snr_pass=n_pass, n_kept=n_kept)


def test_gate_fails_on_the_ge_case():
    g = fit_retention_gate([_rec("capture", 1391, 35)])
    assert g.severity == "fail" and g.metrics["n_kept"] == 35


def test_gate_warns_between_the_thresholds():
    assert fit_retention_gate([_rec("fine", 1000, 300)]).severity == "warn"


def test_gate_passes_a_healthy_run_and_the_worst_phase_decides():
    assert fit_retention_gate([_rec("capture", 1000, 900),
                               _rec("fine", 1000, 950)]).severity == "ok"
    g = fit_retention_gate([_rec("capture", 1000, 900), _rec("fine", 1000, 50)])
    assert g.severity == "fail" and g.metrics["phase"] == "fine"


def test_gate_with_no_record_is_ok_but_says_so():
    g = fit_retention_gate([])
    assert g.severity == "ok" and "no clipped-window" in g.message


def test_run_all_gates_includes_it_only_when_given_a_record():
    from midas_calibrate_v2.pipelines.single import IterRecord
    kw = dict(v1_init=_v1(), unpacked={"Lsd": torch.tensor(1e6), "BC_y": torch.tensor(32.0),
                        "BC_z": torch.tensor(24.0)},
              history=[IterRecord(iteration=0, n_fitted=10, cost=0.0, rc=1,
                                  mean_strain_uE=5.0, Lsd=1e6, BC_y=32.0, BC_z=24.0,
                                  ty=0.0, tz=0.0)])
    names = [r.name for r in run_all_gates(**kw)]
    assert "fit_retention" not in names
    out = run_all_gates(retention=[_rec("capture", 1391, 35)], **kw)
    assert "fit_retention" in [r.name for r in out]
    assert worst_severity(out) == "fail"


# ----------------------------------------------------- apply_sentinel_mask itself
def test_apply_sentinel_mask_unions_zeroes_and_counts():
    im = np.full((6, 8), 100, dtype=np.int32); im[2, 3] = -1
    dk = np.full((6, 8), 3, dtype=np.int32);   dk[4, 5] = -2     # dark-only marker
    user = np.zeros((6, 8), dtype=bool);       user[0, 0] = True
    i2, d2, m2, n = apply_sentinel_mask(im, dk, user)
    assert m2[2, 3] and m2[4, 5] and m2[0, 0] and n == 3
    assert i2[2, 3] == 0 and d2[4, 5] == 0 and i2[0, 0] == 0
    assert i2[1, 1] == 100 and d2[1, 1] == 3
    assert im[2, 3] == -1, "the caller's array must not be modified"


def test_apply_sentinel_mask_leaves_a_clean_frame_alone():
    im = np.full((4, 4), 5, dtype=np.int32)
    i2, d2, m2, n = apply_sentinel_mask(im, None, None)
    assert i2 is im and m2 is None and n == 0


def test_apply_sentinel_mask_does_not_guess_on_a_mismatched_mask():
    im = np.full((4, 4), 5, dtype=np.int32); im[0, 0] = -1
    bad = np.zeros((3, 3), dtype=bool)
    i2, _, m2, n = apply_sentinel_mask(im, None, bad)
    assert m2 is bad and n == 0           # downstream raises its own clear error


def test_a_subtracted_frame_is_NOT_a_valid_input():
    """Why detection happens on the raw frame: integer subtraction produces
    exact -1/-2 values legitimately, which would be mistaken for gaps."""
    img = np.full((4, 4), 100, dtype=np.int32)
    dark = np.full((4, 4), 101, dtype=np.int32)
    assert sentinel_mask_from_raw(img - dark) is not None        # a false positive
    assert sentinel_mask_from_raw(img) is None and sentinel_mask_from_raw(dark) is None


# ------------------------------------- the centroid-path chokepoint (calibrate())
def test_run_estep_v1_masks_gaps_for_every_centroid_path_pipeline(monkeypatch):
    """single, joint_cake, multi, nn_residual, bayesian* and calibrate() all build
    their cake through run_estep_v1. Before, none of them masked the gaps."""
    from midas_calibrate_v2.pipelines import _common
    seen = {}

    def fake_estep(params, image, rt, dark=None, mask=None, **k):
        seen["mask"], seen["image"], seen["dark"] = mask, image, dark
        return None, []                           # -> RuntimeError, caught below

    monkeypatch.setattr(_common, "run_estep", fake_estep)
    im = _gap_image()
    dk = np.full(im.shape, 2, dtype=np.int32); dk[20:23, :] = -1
    with pytest.raises(RuntimeError):
        _common.run_estep_v1(_v1(), im, dark=dk)
    m = np.asarray(seen["mask"])
    assert m[20:23, :].all() and not m[:20, :].any()
    assert (np.asarray(seen["image"])[20:23, :] == 0).all()
    assert (np.asarray(seen["dark"])[20:23, :] == 0).all()


# ----------------------------------------------------------------- the tripwire
def test_every_cake_call_site_goes_through_the_sentinel_step():
    """Source-level guard. A cake is built only in a small set of places; each
    must apply the sentinel mask first. A NEW call site fails this test until it
    is added here on purpose, which is the point."""
    import pathlib
    import re
    root = pathlib.Path(single_pv.__file__).parent
    sanctioned = {"_common.py", "single_pv.py", "single_pv_2d.py"}
    callers = set()
    for f in sorted(root.glob("*.py")):
        txt = f.read_text()
        # real calls, not the definition, an import, a docstring or a comment
        code = "\n".join(l for l in txt.splitlines()
                         if not l.lstrip().startswith(("#", "def ", "from ", "import ")))
        if re.search(r"(?<![\w.])(integrate_cake|run_estep)\(", code):
            callers.add(f.name)
    assert callers <= sanctioned, (
        f"new cake-building call site(s) {sorted(callers - sanctioned)}: they must "
        f"call seed.mask.apply_sentinel_mask on the raw frame first, then be added "
        f"to the sanctioned set in this test")
    for name in sanctioned:
        assert "apply_sentinel_mask(" in (root / name).read_text(), \
            f"{name} builds a cake but never applies the sentinel mask"
