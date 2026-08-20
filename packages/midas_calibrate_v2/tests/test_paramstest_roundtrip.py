"""A written paramstest must be usable as-is, and must not inherit stale panels.

Both defects here were found by handing the calibrate-integrate doc set to a
model with no project context and watching it reduce a real dataset
(2026-08-19). It hit both and had to hand-patch the file to continue.

1. ``CalibrationParams.to_text`` emitted no ``RBinSize``/``EtaBinSize`` even
   though ``from_file`` parses them, so a freshly written paramstest could not
   be handed to an integrator without editing.

2. The template's ``PanelShiftsFile`` rides through ``CalibrationParams.extra``
   into ``to_text``. The old guard only appended ours when no such line was
   present, so a template that already named one left the NEW geometry pointing
   at the PREVIOUS calibration's shifts — silently, and those are exactly the
   ones just superseded.
"""
from __future__ import annotations

import torch

from midas_calibrate.params import CalibrationParams as V1Params
from midas_calibrate_v2.compat.to_v1 import write_v1_paramstest


def _unpacked(n=4) -> dict:
    return {
        "panel_delta_yz": torch.zeros(n, 2),
        "panel_delta_theta": torch.zeros(n),
    }


def _template(**over) -> V1Params:
    p = V1Params(NrPixelsY=16, NrPixelsZ=16, pxY=172.0, Lsd=1e6,
                 BC_y=8.0, BC_z=8.0, Wavelength=0.19, RhoD=1000.0,
                 SpaceGroup=225, MaxRingRad=100.0)
    for k, v in over.items():
        setattr(p, k, v)
    return p


# ── 1. binning keys survive the round trip ───────────────────────────────
def test_binning_keys_are_written(tmp_path):
    t = _template(RBinSize=0.5, EtaBinSize=1.0)
    out = tmp_path / "paramstest_v2.txt"
    write_v1_paramstest({"Lsd": torch.tensor(1e6)}, t, out)
    text = out.read_text()
    assert "RBinSize 0.5" in text
    assert "EtaBinSize 1" in text


def test_binning_keys_round_trip_through_from_file(tmp_path):
    t = _template(RBinSize=0.5, EtaBinSize=1.0)
    out = tmp_path / "paramstest_v2.txt"
    write_v1_paramstest({"Lsd": torch.tensor(1e6)}, t, out)
    back = V1Params.from_file(out)
    assert back.RBinSize == 0.5
    assert back.EtaBinSize == 1.0


# ── 2. a stale PanelShiftsFile must never survive ────────────────────────
def test_template_panelshifts_is_not_inherited(tmp_path):
    """The dangerous case: template names the PREVIOUS calibration's file."""
    t = _template()
    t.extra["PanelShiftsFile"] = "panel_shifts.txt"     # the old, superseded one
    out = tmp_path / "paramstest_v2.txt"
    shifts = write_v1_paramstest(_unpacked(), t, out)

    text = out.read_text()
    lines = [l for l in text.splitlines() if l.startswith("PanelShiftsFile")]
    assert len(lines) == 1, f"expected exactly one PanelShiftsFile, got {lines}"
    assert lines[0] == f"PanelShiftsFile {shifts.name}"
    assert "panel_shifts.txt" not in text


def test_written_file_points_at_a_sidecar_that_exists(tmp_path):
    t = _template()
    t.extra["PanelShiftsFile"] = "panel_shifts.txt"
    out = tmp_path / "paramstest_v2.txt"
    write_v1_paramstest(_unpacked(), t, out)

    named = [l.split(None, 1)[1] for l in out.read_text().splitlines()
             if l.startswith("PanelShiftsFile")][0]
    assert (out.parent / named).exists(), "PanelShiftsFile names a missing file"


def test_single_panel_leaves_no_panelshifts_line(tmp_path):
    """No panels refined -> no sidecar, and no inherited line either."""
    t = _template()
    t.extra["PanelShiftsFile"] = "panel_shifts.txt"
    out = tmp_path / "paramstest_v2.txt"
    assert write_v1_paramstest({"Lsd": torch.tensor(1e6)}, t, out) is None
    # the inherited line is still there — nothing overrode it — so a caller
    # reusing this template must know it refers to the template's own file.
    # Documented rather than silently dropped: dropping it would discard a
    # legitimate hand-maintained shifts file on a single-panel rerun.
    assert "panel_shifts.txt" in out.read_text()
