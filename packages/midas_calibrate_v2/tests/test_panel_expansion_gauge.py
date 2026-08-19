"""The panel field has a second nullspace, and the translation gauge misses it.

Pushing every module outward in proportion to its radius shifts all ring radii
exactly the way an Lsd error does. `fix_panel_id` and `Σ panel = 0` both remove
the *translation* nullspace and leave this one alone; on a 48-panel Pilatus
11 % of the fitted field sat in it (2026-08-19).
"""
from __future__ import annotations

import torch

from midas_calibrate_v2.forward.panels import PanelLayout
from midas_calibrate_v2.loss.panel_gauge import (
    expansion_mode, panel_expansion_residual,
)

BC_Y, BC_Z = 737.9, 842.4


def _layout():
    return PanelLayout.regular(6, 8, 243, 195, gap_y=(1, 7, 1, 7, 1), gap_z=(17,)*7)


def _mode(lay):
    return expansion_mode(lay.panel_centers_y.to(torch.float64),
                          lay.panel_centers_z.to(torch.float64), BC_Y, BC_Z)


def test_mode_is_unit_norm_and_shaped_per_panel():
    m = _mode(_layout())
    assert m.shape == (48, 2)
    assert abs(float(torch.linalg.vector_norm(m)) - 1.0) < 1e-12


def test_mode_points_outward_from_the_beam_centre():
    lay = _layout()
    m = _mode(lay)
    cy = lay.panel_centers_y.reshape(-1).to(torch.float64)
    cz = lay.panel_centers_z.reshape(-1).to(torch.float64)
    radial = torch.stack([cy - BC_Y, cz - BC_Z], dim=1)
    # every module's displacement is parallel to its own outward radius
    cos = (m * radial).sum(1) / (
        torch.linalg.vector_norm(m, dim=1) * torch.linalg.vector_norm(radial, dim=1))
    assert torch.all(cos > 0.999)


def test_pure_expansion_field_is_penalised():
    lay = _layout()
    m = _mode(lay)
    r = panel_expansion_residual({"panel_delta_yz": 0.5 * m},
                                 panel_layout=lay, bc_y=BC_Y, bc_z=BC_Z,
                                 lambda_ex=1e6)
    assert r.numel() == 1
    assert abs(float(r[0]) - 1e6 * 0.5) < 1e-3          # amplitude x lambda


def test_pure_translation_is_NOT_penalised_by_this_gauge():
    """Translation is the other gauge's job; this one must leave it alone."""
    lay = _layout()
    dyz = torch.zeros(48, 2, dtype=torch.float64)
    dyz[:, 0] = 0.7                                      # uniform shift in y
    r = panel_expansion_residual({"panel_delta_yz": dyz}, panel_layout=lay,
                                 bc_y=BC_Y, bc_z=BC_Z, lambda_ex=1e6)
    # a uniform shift has only a small projection on the expansion mode; it is
    # not what this constraint targets
    amp = abs(float(r[0])) / 1e6
    assert amp < 0.25 * 0.7


def test_tangential_field_is_not_penalised():
    lay = _layout()
    m = _mode(lay)
    tang = torch.stack([-m[:, 1], m[:, 0]], dim=1)       # rotate 90 deg
    r = panel_expansion_residual({"panel_delta_yz": tang}, panel_layout=lay,
                                 bc_y=BC_Y, bc_z=BC_Z, lambda_ex=1e6)
    assert abs(float(r[0])) < 1e-6


def test_scales_with_lambda():
    lay = _layout(); m = _mode(lay)
    a = panel_expansion_residual({"panel_delta_yz": m}, panel_layout=lay,
                                 bc_y=BC_Y, bc_z=BC_Z, lambda_ex=1.0)
    b = panel_expansion_residual({"panel_delta_yz": m}, panel_layout=lay,
                                 bc_y=BC_Y, bc_z=BC_Z, lambda_ex=1e3)
    assert abs(float(b[0]) - 1e3 * float(a[0])) < 1e-6


def test_absent_panels_give_an_empty_row():
    r = panel_expansion_residual({"Lsd": torch.tensor(650000.0)},
                                 panel_layout=_layout(), bc_y=BC_Y, bc_z=BC_Z)
    assert r.numel() == 0


def test_is_differentiable():
    lay = _layout(); m = _mode(lay)
    dyz = (0.3 * m).clone().requires_grad_(True)
    panel_expansion_residual({"panel_delta_yz": dyz}, panel_layout=lay,
                             bc_y=BC_Y, bc_z=BC_Z, lambda_ex=1.0).sum().backward()
    assert dyz.grad is not None and torch.isfinite(dyz.grad).all()


def test_registrar_sets_the_spec_flags():
    from midas_calibrate_v2.compat.from_v1 import add_panel_no_expansion_constraint

    class _Spec:  # minimal stand-in; the registrar only sets attributes
        pass
    s = _Spec()
    add_panel_no_expansion_constraint(s, lambda_ex=2e5)
    assert s.no_panel_expansion is True
    assert s.panel_expansion_lambda == 2e5
