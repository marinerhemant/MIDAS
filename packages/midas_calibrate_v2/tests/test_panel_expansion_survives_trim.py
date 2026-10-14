"""The panel-expansion gauge must reach the LM after the pre-LM trim.

`autocalibrate_pv` built its residual function twice: once over the full fit
set, and again over the trimmed set the LM actually minimises. The second copy
re-added the zero-sum and Gaussian-prior rows but not the expansion row
(c2daa4d1), so `add_panel_no_expansion_constraint` was silently inert under
every `trim_mode` except "off". `tests/test_panel_expansion_gauge.py` only
tested the row in isolation, so nothing noticed.

Both sets now come from one builder, `_make_residual_fn`. Test the property
that failed: the constraint row is present, and identical, for a full and a
trimmed fit set.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from midas_calibrate_v2.forward.panels import PanelLayout
from midas_calibrate_v2.loss.panel_gauge import expansion_mode
from midas_calibrate_v2.pipelines.single_pv import _make_residual_fn

BC = 8.0
LAMBDA = 1e6


def _layout_and_points(n_y=2, n_z=2, size=8):
    layout = PanelLayout.regular(n_y, n_z, size, size, gap_y=0, gap_z=0)
    H, W = layout.panel_index_mask.shape
    ys, zs = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64), indexing="ij")
    Y, Z = ys.reshape(-1), zs.reshape(-1)
    idx = layout.panel_index_mask.reshape(-1)
    keep = idx >= 0
    return layout, Y[keep], Z[keep], idx[keep]


def _ds(Y, Z, idx):
    return SimpleNamespace(
        Y_pix=Y, Z_pix=Z, panel_idx=idx,
        ring_two_theta_deg=torch.full((Y.numel(),), 5.0, dtype=torch.float64),
        rho_d=torch.tensor(1000.0, dtype=torch.float64),
        weights=None, ring_idx=None, ring_d_spacing_A=None,
    )


def _spec(no_expansion):
    return SimpleNamespace(no_panel_expansion=no_expansion,
                           panel_expansion_lambda=LAMBDA,
                           panel_expansion_row_scale=1.0,      # this test is about the row's presence, not its scale
                           fix_panel_id=0, parameters={})


def _unpacked(layout, amp):
    n = layout.n_panels()
    cy = layout.panel_centers_y.to(torch.float64)
    cz = layout.panel_centers_z.to(torch.float64)
    m = expansion_mode(cy, cz, BC, BC)
    f64 = torch.float64
    return {
        "Lsd": torch.tensor(1e6, dtype=f64),
        "BC_y": torch.tensor(BC, dtype=f64), "BC_z": torch.tensor(BC, dtype=f64),
        "ty": torch.tensor(0.0, dtype=f64), "tz": torch.tensor(0.0, dtype=f64),
        "pxY": torch.tensor(200.0, dtype=f64),
        "panel_delta_yz": amp * m,
        "panel_delta_theta": torch.zeros(n, dtype=f64),
    }


def _build(ds, layout, no_expansion):
    return _make_residual_fn(
        ds, spec=_spec(no_expansion), panel_layout=layout,
        v1_params=SimpleNamespace(BC_y=BC, BC_z=BC))


def test_expansion_row_present_on_full_and_on_trimmed_set():
    layout, Y, Z, idx = _layout_and_points()
    full = _ds(Y, Z, idx)
    keep = torch.arange(Y.numel()) % 3 != 0            # what a trim would leave
    trimmed = _ds(Y[keep], Z[keep], idx[keep])
    u = _unpacked(layout, amp=0.5)

    r_full = _build(full, layout, True)(u)
    r_trim = _build(trimmed, layout, True)(u)

    # data rows + exactly one expansion row, whichever set the LM sees
    assert r_full.numel() == Y.numel() + 1
    assert r_trim.numel() == int(keep.sum()) + 1, (
        "the trimmed residual lost the expansion row: no_panel_expansion is "
        "inert after the pre-LM trim"
    )
    # and the row is the same constraint: amplitude x lambda
    assert abs(float(r_full[-1]) - LAMBDA * 0.5) < 1e-3
    assert abs(float(r_trim[-1]) - LAMBDA * 0.5) < 1e-3


def test_no_expansion_row_when_the_gauge_is_off():
    """Control: the extra row appears because of the flag, not unconditionally."""
    layout, Y, Z, idx = _layout_and_points()
    u = _unpacked(layout, amp=0.5)
    r = _build(_ds(Y, Z, idx), layout, False)(u)
    assert r.numel() == Y.numel()
