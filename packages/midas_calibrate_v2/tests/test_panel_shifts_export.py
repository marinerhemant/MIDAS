"""A multi-panel calibration must emit its panel shifts.

Before this, ``write_v1_paramstest`` deliberately skipped the ``panel_delta_*``
entries ("panel data goes to a separate file") and nothing ever wrote that
file. The refined panel geometry stayed inside the result object, and every
downstream integrator ran with zero panel shifts — silently, because a missing
``PanelShiftsFile`` is indistinguishable from a single-panel detector.
"""
from __future__ import annotations

import torch

from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.compat.to_v1 import write_v1_paramstest


def _template(**over) -> V1Params:
    # CalibrationParams carries no panel fields; the panel layout lives in the
    # v2 spec. Only PanelShiftsFile is consulted here, as an optional override
    # of the sidecar name.
    p = V1Params()
    for k, v in over.items():
        setattr(p, k, v)
    return p


def _unpacked(n=4) -> dict:
    return {
        "panel_delta_yz": torch.tensor(
            [[0.10, -0.20], [0.30, 0.40], [-0.50, 0.60], [0.70, -0.80]][:n]),
        "panel_delta_theta": torch.tensor([1e-3, -2e-3, 3e-3, -4e-3][:n]),
        "panel_delta_lsd": torch.tensor([1.5, -2.5, 3.5, -4.5][:n]),
        "panel_delta_p2": torch.tensor([1e-5, -2e-5, 3e-5, -4e-5][:n]),
    }


def test_sidecar_is_written_and_referenced(tmp_path):
    out = tmp_path / "paramstest_v2.txt"
    shifts = write_v1_paramstest(_unpacked(), _template(), out)

    assert shifts is not None and shifts.exists()
    assert shifts.parent == out.parent          # sits beside the paramstest
    # the paramstest must point at it, or nothing will load it
    assert f"PanelShiftsFile {shifts.name}" in out.read_text()


def test_sidecar_roundtrips_through_the_v1_loader(tmp_path):
    """The file must be readable by the loader the integrator actually uses."""
    from midas_integrate.panel import generate_panels, load_panel_shifts

    out = tmp_path / "paramstest_v2.txt"
    shifts = write_v1_paramstest(_unpacked(), _template(), out)

    panels = generate_panels(n_panels_y=2, n_panels_z=2,
                             panel_size_y=10, panel_size_z=10,
                             gaps_y=[0], gaps_z=[0])
    load_panel_shifts(shifts, panels)

    assert panels[0].dY == 0.10 and panels[0].dZ == -0.20
    assert panels[3].dY == 0.70 and panels[3].dZ == -0.80
    assert abs(panels[1].dTheta - (-2e-3)) < 1e-9
    assert abs(panels[2].dLsd - 3.5) < 1e-6


def test_default_sidecar_name_derives_from_the_paramstest(tmp_path):
    out = tmp_path / "paramstest_v2.txt"
    shifts = write_v1_paramstest(_unpacked(), _template(), out)
    assert shifts.name == "paramstest_v2_panelshifts.txt"


def test_explicit_name_overrides_the_template(tmp_path):
    out = tmp_path / "paramstest_v2.txt"
    shifts = write_v1_paramstest(_unpacked(), _template(), out,
                                 panel_shifts_name="my_shifts.txt")
    assert shifts.name == "my_shifts.txt"


def test_single_panel_calibration_writes_no_sidecar(tmp_path):
    out = tmp_path / "paramstest_v2.txt"
    shifts = write_v1_paramstest({"Lsd": torch.tensor(1e6)}, _template(), out)
    assert shifts is None
    assert not list(tmp_path.glob("*panelshifts*"))
