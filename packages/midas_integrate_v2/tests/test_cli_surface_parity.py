"""What the library can do, the CLI must expose.

Three defects in this stack were the same shape: the capability existed one
layer down and simply had no way in from the layer users touch.

  * per-panel shifts — the shared geometry kernel took `panel_layout`,
    `panel_idx`, `delta_yz`... and `pixel_to_REta_from_spec` passed none of
    them, so every v2 binning path dropped the panel calibration.
  * `--mask` — the geometry classes all take `mask=`; the one-shot CLI had no
    flag, so 202 550 masked pixels went into a profile as raw values.
  * `--device` — `spec_from_v1_paramstest` took `device=`; the CLI never passed
    it, so v2 was CPU-only and a documented GPU path did not exist.

None of these is a logic error, so no numerical test finds them. They are
*surface* gaps, and the check is structural: for each capability the library
accepts, assert the command line offers a way to reach it.
"""
from __future__ import annotations

import inspect

import pytest

from midas_integrate_v2 import cli
from midas_integrate_v2.binning.hard import HardBinGeometry
from midas_integrate_v2.binning.polygon import PolygonBinGeometry
from midas_integrate_v2.binning.subpixel import SubpixelBinGeometry
from midas_integrate_v2.compat.from_v1 import spec_from_v1_paramstest


def _flags(main_fn) -> str:
    """The source of a CLI entry point, where its add_argument calls live."""
    return inspect.getsource(main_fn)


# ── the geometry classes accept a mask -> the CLI must too ───────────────
@pytest.mark.parametrize("cls", [HardBinGeometry, SubpixelBinGeometry,
                                 PolygonBinGeometry])
def test_geometry_classes_accept_a_mask(cls):
    """Guard the premise of the next test."""
    assert "mask" in inspect.signature(cls.from_spec).parameters


def test_one_shot_cli_exposes_mask():
    assert '"--mask"' in _flags(cli.integrate_main), (
        "every binning geometry takes mask=, but midas-integrate-v2 has no "
        "--mask; masked pixels enter the profile as raw values"
    )


# ── the spec loader accepts a device -> the CLI must too ─────────────────
def test_spec_loader_accepts_a_device():
    assert "device" in inspect.signature(spec_from_v1_paramstest).parameters


@pytest.mark.parametrize("entry", ["integrate_main", "server_main"])
def test_cli_exposes_device(entry):
    assert '"--device"' in _flags(getattr(cli, entry)), (
        f"{entry} does not expose --device, so it is pinned to whatever "
        f"device the spec was built on"
    )


# ── every binning mode the library implements should be reachable ────────
def test_every_binning_mode_is_reachable_from_the_cli():
    src = _flags(cli.integrate_main)
    for mode in ("hard", "subpixel", "polygon", "soft"):
        assert f'"{mode}"' in src, (
            f"binning mode {mode!r} exists in the library but --mode does not "
            f"offer it"
        )


# ── the v1 binary writers exist -> the CLI must be able to emit them ─────
def test_cli_can_write_the_v1_binaries():
    from midas_integrate_v2.io import v1_outputs
    assert hasattr(v1_outputs, "write_v1_outputs")
    assert '"--v1-out"' in _flags(cli.integrate_main), (
        "io.v1_outputs can write lineout.bin/Int2D.bin but the one-shot CLI "
        "offers no way to ask for them"
    )


# ── the panel path: capability must reach the forward model ──────────────
def test_forward_model_receives_the_panel_arguments():
    """The original defect: the kernel took panel arguments and the v2 shim
    passed none of them."""
    from midas_integrate_v2.forward import pixels
    src = inspect.getsource(pixels)
    for kw in ("panel_layout", "panel_idx", "delta_yz", "delta_theta",
               "delta_lsd_panel", "delta_p2_panel", "fix_panel_id"):
        assert kw in src, (
            f"pixel_to_REta accepts {kw} but the spec shim never passes it — "
            f"this is how the panel calibration was silently discarded"
        )


def test_the_parity_check_can_fail():
    """Control: a capability the CLI genuinely does not expose must be
    reported as absent, or these assertions prove nothing."""
    assert '"--definitely-not-a-real-flag"' not in _flags(cli.integrate_main)
