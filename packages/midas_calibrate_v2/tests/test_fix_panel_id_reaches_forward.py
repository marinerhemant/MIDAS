"""The anchored panel must actually be anchored in the forward model.

`FixPanelID` had a three-stage journey and died on the last one:

    parameter file  ->  CalibrationParams.FixedPanelID   (fixed 2026-08-19)
                    ->  spec.fix_panel_id                (already worked)
                    ->  pixel_to_REta(fix_panel_id=...)  (never passed)

`pseudo_strain_residual` forwarded every other panel argument and omitted this
one, so the model always anchored panel 0 whatever the file said. Found by a
context-free run on 2026-08-19: its calibration of a file specifying
`FixPanelID 28` left panel 0 at exactly zero and panel 28 free.

Testing the plumbing (`spec.fix_panel_id == 28`) would NOT have caught it —
that assertion passed throughout. So test the property the gauge exists to
provide: the residual is invariant to the anchored panel's shift, because that
panel's contribution is masked out.
"""
from __future__ import annotations

import pytest
import torch

from midas_calibrate_v2.forward.panels import PanelLayout
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual


def _setup(n_y=2, n_z=2, size=8):
    layout = PanelLayout.regular(n_y, n_z, size, size, gap_y=0, gap_z=0)
    n = layout.n_panels()
    H, W = layout.panel_index_mask.shape
    ys, zs = torch.meshgrid(torch.arange(H, dtype=torch.float64),
                            torch.arange(W, dtype=torch.float64), indexing="ij")
    Y, Z = ys.reshape(-1), zs.reshape(-1)
    idx = layout.panel_index_mask.reshape(-1)
    keep = idx >= 0
    return layout, n, Y[keep], Z[keep], idx[keep]


def _params(n, dtype=torch.float64):
    return {
        "Lsd": torch.tensor(1e6, dtype=dtype),
        "BC_y": torch.tensor(8.0, dtype=dtype),
        "BC_z": torch.tensor(8.0, dtype=dtype),
        "ty": torch.tensor(0.0, dtype=dtype),
        "tz": torch.tensor(0.0, dtype=dtype),
        "pxY": torch.tensor(200.0, dtype=dtype),
        "panel_delta_yz": torch.zeros(n, 2, dtype=dtype),
        "panel_delta_theta": torch.zeros(n, dtype=dtype),
    }


def _resid(p, layout, Y, Z, idx, fix_id, tth):
    return pseudo_strain_residual(
        Y, Z, tth, p,
        rho_d=torch.tensor(1000.0, dtype=torch.float64),
        panel_layout=layout, panel_idx=idx, fix_panel_id=fix_id,
    )


@pytest.mark.parametrize("fix_id", [0, 1, 3])
def test_anchored_panel_shift_has_no_effect(fix_id):
    """Moving the anchored panel must not change the residual at all."""
    layout, n, Y, Z, idx = _setup()
    tth = torch.full((Y.numel(),), 5.0, dtype=torch.float64)

    p0 = _params(n)
    base = _resid(p0, layout, Y, Z, idx, fix_id, tth)

    moved = _params(n)
    moved["panel_delta_yz"] = moved["panel_delta_yz"].clone()
    moved["panel_delta_yz"][fix_id, 0] = 0.7      # shove the anchored panel
    moved["panel_delta_yz"][fix_id, 1] = -0.4
    after = _resid(moved, layout, Y, Z, idx, fix_id, tth)

    assert torch.allclose(base, after), (
        f"panel {fix_id} was supposed to be anchored, but moving it changed "
        f"the residual by up to {(base - after).abs().max():.3e}"
    )


def test_a_non_anchored_panel_does_move_the_residual():
    """The control: without it, a residual that ignores ALL panels would pass."""
    layout, n, Y, Z, idx = _setup()
    tth = torch.full((Y.numel(),), 5.0, dtype=torch.float64)

    p0 = _params(n)
    base = _resid(p0, layout, Y, Z, idx, 0, tth)

    moved = _params(n)
    moved["panel_delta_yz"] = moved["panel_delta_yz"].clone()
    moved["panel_delta_yz"][2, 0] = 0.7           # panel 2, not the anchor
    after = _resid(moved, layout, Y, Z, idx, 0, tth)

    assert not torch.allclose(base, after), \
        "a free panel's shift had no effect — the panel model is not connected"


def test_the_anchor_actually_follows_fix_panel_id():
    """Anchor 3, then move panel 0: it must move. This is the regression."""
    layout, n, Y, Z, idx = _setup()
    tth = torch.full((Y.numel(),), 5.0, dtype=torch.float64)

    p0 = _params(n)
    base = _resid(p0, layout, Y, Z, idx, 3, tth)

    moved = _params(n)
    moved["panel_delta_yz"] = moved["panel_delta_yz"].clone()
    moved["panel_delta_yz"][0, 0] = 0.7
    after = _resid(moved, layout, Y, Z, idx, 3, tth)

    assert not torch.allclose(base, after), (
        "panel 0 behaved as the anchor even though fix_panel_id=3 — the gauge "
        "is not reaching the forward model"
    )
