"""Gauge rows must be weighted against the data rows, not against nothing.

The Sigma panel = 0 and the panel-expansion rows are sqrt(lambda) * (a sum of panel parameters), with
lambda = 1e6, next to data rows that are strain (~1e-5). Unscaled, the gauge curvature (1e6 per panel
parameter) dwarfs the data curvature in the Marquardt diagonal and the LM takes no steps along the free
panel directions: on the Pilatus frame autocalibrate_pv with spec.zero_sum_panels left the 240-DOF panel
block at its v1 seed (kept strain 38.8 uE against 29.3-30.6 once the rows are scaled), and the
panel-expansion constraint diverged (ty -4.9 deg, 989 uE). `spec.zero_sum_row_scale` /
`spec.panel_expansion_row_scale` (a number, or "auto" = rms of the data residual at the first
evaluation) set lambda_eff = lambda * s**2; 1.0 is the unscaled legacy row.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from midas_calibrate_v2.forward.panels import PanelLayout
from midas_calibrate_v2.loss.panel_gauge import expansion_mode
from midas_calibrate_v2.pipelines.single_pv import _make_residual_fn, _row_scale

BC = 8.0


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


def _spec(**kw):
    base = dict(no_panel_expansion=False, panel_expansion_lambda=1e6, fix_panel_id=0, parameters={},
                zero_sum_panels=False, zero_sum_lambda=1e6)
    base.update(kw)
    return SimpleNamespace(**base)


def _rows(spec, amp=0.5):
    layout, Y, Z, idx = _layout_and_points()
    u = _unpacked(layout, amp=amp)
    u["panel_delta_yz"] = u["panel_delta_yz"] + 0.1          # a nonzero common shift: a nonzero zero-sum row
    ds = _ds(Y, Z, idx)
    fn = _make_residual_fn(ds, spec=spec, panel_layout=layout, v1_params=SimpleNamespace(BC_y=BC, BC_z=BC))
    r = fn(u)
    return r[Y.numel():], r[:Y.numel()]


def test_row_scale_number_and_auto_and_bad_string():
    assert _row_scale(1e-5, {}, torch.ones(3), "x") == 1e-5
    c = {}
    s = _row_scale("auto", c, torch.full((4,), 2e-5, dtype=torch.float64), "x")
    assert abs(s - 2e-5) < 1e-12
    assert _row_scale("auto", c, torch.full((4,), 9.0, dtype=torch.float64), "x") == s     # frozen after the first call
    with pytest.raises(ValueError):
        _row_scale("hard", {}, torch.ones(3), "x")


def test_zero_sum_rows_scale_linearly_with_the_row_scale():
    legacy, _ = _rows(_spec(zero_sum_panels=True, zero_sum_row_scale=1.0))
    scaled, _ = _rows(_spec(zero_sum_panels=True, zero_sum_row_scale=1e-5))
    assert legacy.abs().max() > 0
    assert torch.allclose(scaled, legacy * 1e-5, rtol=1e-9)


def test_auto_scale_is_the_rms_of_the_data_residual():
    legacy, data = _rows(_spec(zero_sum_panels=True, zero_sum_row_scale=1.0))
    auto, data2 = _rows(_spec(zero_sum_panels=True, zero_sum_row_scale="auto"))
    s = float(data2.pow(2).mean().sqrt())
    assert s > 0
    assert torch.allclose(auto, legacy * s, rtol=1e-9)


def test_expansion_row_scales_with_its_own_attribute():
    legacy, _ = _rows(_spec(no_panel_expansion=True, panel_expansion_row_scale=1.0))
    scaled, _ = _rows(_spec(no_panel_expansion=True, panel_expansion_row_scale=1e-4))
    assert legacy.abs().max() > 0
    # the expansion row is lambda * amp (it does not carry a square root), so its row weight scales with s
    assert torch.allclose(scaled, legacy * 1e-4, rtol=1e-9)


def test_default_is_auto_not_the_unscaled_legacy_rows():
    default, data = _rows(_spec(zero_sum_panels=True))
    legacy, _ = _rows(_spec(zero_sum_panels=True, zero_sum_row_scale=1.0))
    s = float(data.pow(2).mean().sqrt())
    assert torch.allclose(default, legacy * s, rtol=1e-9)


def test_block_units_divide_each_block_by_its_bounds_half_width():
    from midas_calibrate_v2.parameters.parameter import Parameter
    layout, Y, Z, idx = _layout_and_points()
    n = layout.n_panels()
    u = _unpacked(layout, amp=0.5)
    u["panel_delta_lsd"] = torch.full((n,), 100.0, dtype=torch.float64)    # a nonzero common dLsd, um
    ds = _ds(Y, Z, idx)
    params = {"panel_delta_lsd": Parameter(name="panel_delta_lsd", init=[0.0] * n, bounds=(-2000.0, 2000.0))}
    base = dict(zero_sum_panels=True, zero_sum_row_scale=1.0, parameters=params)
    def rows(**kw):
        spec = _spec(**{**base, **kw})
        fn = _make_residual_fn(ds, spec=spec, panel_layout=layout, v1_params=SimpleNamespace(BC_y=BC, BC_z=BC))
        return fn(u)[Y.numel():]
    raw = rows(zero_sum_block_units=None)
    bounded = rows(zero_sum_block_units="bounds")
    default = rows()
    assert torch.allclose(default, bounded)
    # the dLsd entry (index 3 of the concatenated rows: yz(2), theta, lsd) is divided by 2000
    assert abs(float(bounded[3]) * 2000.0 - float(raw[3])) < 1e-6 * abs(float(raw[3]))
    # blocks without bounds are untouched
    assert torch.allclose(bounded[:3], raw[:3])
