"""The ring overlay must be drawn in the frame the geometry was fitted in.

``calibrate()`` applies ``im_trans`` internally, so ``BC_y``/``BC_z`` index the
columns/rows of the TRANSFORMED frame. ``write_ring_overlay`` used to plot the
raw frame under those rings: on a 20-ID-D Varex (``ImTransOpt 2``) the measured
rings sat at Z = 2879 - BC_z while the predicted ones sat at BC_z, on a fit
that was correct (median radial offset 0.021 px once integrated in the right
frame). These tests put a ring at a known centre in the transformed frame,
un-apply the transform to make the raw frame, and check that the array the
overlay plots has its ring centred on BC -- and that the raw frame does NOT,
so the check can fail.
"""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from midas_calibrate_v2.pipelines.ff_calibrate import (
    overlay_frame,
    write_ring_overlay,
)

# Off-centre in both axes, on a non-square frame, so any flip or transpose
# moves the ring centre by many pixels.
NZ, NY = 128, 160
BC_Y, BC_Z = 70.0, 50.0
R = 30.0


def _ring_in_fit_frame() -> np.ndarray:
    z, y = np.mgrid[0:NZ, 0:NY].astype(float)
    r = np.hypot(y - BC_Y, z - BC_Z)
    return np.exp(-0.5 * ((r - R) / 1.5) ** 2)


def _centroid(img):
    """(y, z) intensity centroid; a full ring's centroid is its centre."""
    z, y = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    w = img.sum()
    return float((img * y).sum() / w), float((img * z).sum() / w)


def _result(im_trans):
    return SimpleNamespace(BC_y=BC_Y, BC_z=BC_Z, Lsd=1_000_000.0, pxY=200.0,
                           im_trans=im_trans)


def test_flip_z_overlay_frame_puts_ring_on_bc():
    fit = _ring_in_fit_frame()
    raw = fit[::-1, :]  # un-apply ImTransOpt 2 (a flip is its own inverse)

    # The raw frame is mirrored: its ring is NOT at BC. Without this the test
    # below could pass with the transform doing nothing.
    ry, rz = _centroid(raw)
    assert abs(rz - (NZ - 1 - BC_Z)) < 1.0 and abs(rz - BC_Z) > 20

    shown = overlay_frame(raw, _result((2,)))
    np.testing.assert_array_equal(shown, fit)
    y, z = _centroid(shown)
    assert abs(y - BC_Y) < 1.0 and abs(z - BC_Z) < 1.0


def test_flip_y_then_transpose_overlay_frame():
    fit = _ring_in_fit_frame()
    # im_trans (1, 3): flip Y, then transpose. Invert in reverse order.
    raw = fit.T[:, ::-1]
    shown = overlay_frame(raw, _result((1, 3)))
    np.testing.assert_array_equal(shown, fit)


def test_no_im_trans_is_unchanged():
    fit = _ring_in_fit_frame()
    np.testing.assert_array_equal(overlay_frame(fit, _result(())), fit)
    # A result without the attribute at all behaves the same.
    no_attr = SimpleNamespace(BC_y=BC_Y, BC_z=BC_Z, Lsd=1e6, pxY=200.0)
    np.testing.assert_array_equal(overlay_frame(fit, no_attr), fit)


@pytest.mark.parametrize("im_trans, make_raw", [
    ((2,), lambda a: a[::-1, :]),
    ((), lambda a: a),
])
def test_write_ring_overlay_plots_the_fit_frame(tmp_path, monkeypatch,
                                                im_trans, make_raw):
    pytest.importorskip("midas_hkls")
    from matplotlib.axes import Axes

    shown = {}
    real_imshow = Axes.imshow

    def spy(self, X, *a, **k):
        shown["X"] = np.asarray(X)
        shown["origin"] = k.get("origin")
        return real_imshow(self, X, *a, **k)

    monkeypatch.setattr(Axes, "imshow", spy)
    fit = _ring_in_fit_frame()
    out = tmp_path / "overlay.png"
    write_ring_overlay(make_raw(fit), _result(im_trans), 0.173, "CeO2", out)

    assert out.exists()
    # origin="lower" keeps array row index == plotted Z, so circles drawn at
    # (BC_y, BC_z) land on the ring in the plotted array.
    assert shown["origin"] == "lower"
    np.testing.assert_array_equal(shown["X"], fit)
    y, z = _centroid(shown["X"])
    assert abs(y - BC_Y) < 1.0 and abs(z - BC_Z) < 1.0
