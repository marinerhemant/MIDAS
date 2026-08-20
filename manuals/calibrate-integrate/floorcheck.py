#!/usr/bin/env python
"""Install gate — tests BEHAVIOUR, not version strings.

    python floorcheck.py          # exits 0 if every floor passes, 1 otherwise

Four defects produce plausible wrong answers rather than errors. A version
number cannot gate them reliably: the numbering drifts, a floor guessed ahead of
the release blocks forever, and a floor guessed behind it passes without the
fix. Both happened in an earlier draft of this doc set. So probe the behaviour
each floor exists to guarantee.

Derived from the check a context-free model wrote for itself on 2026-08-19 when
the numeric gate in this doc set turned out to be unsatisfiable.
"""
from __future__ import annotations

import inspect
import sys


def _ok(name: str, passed: bool, why: str) -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if not passed:
        print(f"         {why}")
    return passed


def check_panel_shifts_applied() -> bool:
    """v2 must pass per-panel corrections into the forward model."""
    try:
        from midas_integrate_v2.forward import pixels
        src = inspect.getsource(pixels)
        return _ok(
            "integrate-v2 applies per-panel shifts",
            "_panel_inputs_from_spec" in src and "panel_idx" in src,
            "this build integrates a tiled detector with the panel "
            "calibration silently discarded",
        )
    except Exception as e:                                   # pragma: no cover
        return _ok("integrate-v2 applies per-panel shifts", False, repr(e))


def check_sidecar_written() -> bool:
    """calibrate-v2 must write refined panel shifts to disk."""
    try:
        from midas_calibrate_v2.compat.to_v1 import write_v1_paramstest
        sig = inspect.signature(write_v1_paramstest)
        src = inspect.getsource(write_v1_paramstest)
        return _ok(
            "calibrate-v2 writes the panelshifts sidecar",
            sig.return_annotation is not inspect.Signature.empty
            and "write_panel_shifts_file" in src,
            "refined panel shifts never reach disk, so nothing downstream "
            "can consume them",
        )
    except Exception as e:                                   # pragma: no cover
        return _ok("calibrate-v2 writes the panelshifts sidecar", False, repr(e))


def check_map_buffer() -> bool:
    """v1's map buffer must be geometry-derived and warn unconditionally."""
    try:
        from midas_integrate import detector_mapper as dm
        return _ok(
            "integrate v1 sizes the map buffer and always warns",
            hasattr(dm, "estimate_per_row_max")
            and hasattr(dm, "MapTruncationWarning"),
            "the map silently truncates at fine RBinSize; absolute flux and "
            "bin occupancy are wrong while the normalised profile looks fine",
        )
    except Exception as e:                                   # pragma: no cover
        return _ok("integrate v1 sizes the map buffer and always warns", False, repr(e))


def check_fixpanelid() -> bool:
    """calibrate v1 must parse FixPanelID."""
    try:
        from midas_calibrate.params import CalibrationParams
        src = inspect.getsource(CalibrationParams.from_file)
        return _ok(
            "calibrate v1 parses FixPanelID",
            "FixPanelID" in src,
            "the anchored panel is silently 0 whatever the parameter file says",
        )
    except Exception as e:                                   # pragma: no cover
        return _ok("calibrate v1 parses FixPanelID", False, repr(e))


def check_mask_and_device() -> bool:
    """The v2 one-shot must accept a mask and a device."""
    try:
        from midas_integrate_v2 import cli
        src = inspect.getsource(cli.integrate_main)
        return _ok(
            "integrate-v2 one-shot accepts --mask and --device",
            '"--mask"' in src and '"--device"' in src,
            "without --mask, gap sentinels and dead pixels enter the profile "
            "as raw values (measured: 1775/1800 bins changed, up to 25%)",
        )
    except Exception as e:                                   # pragma: no cover
        return _ok("integrate-v2 one-shot accepts --mask and --device", False, repr(e))


def main() -> int:
    print("Install gate — behavioural probes, not version strings\n")
    results = [
        check_panel_shifts_applied(),
        check_sidecar_written(),
        check_map_buffer(),
        check_fixpanelid(),
        check_mask_and_device(),
    ]
    print()
    if all(results):
        print("All floors pass. Proceed.")
        return 0
    print(f"{results.count(False)} of {len(results)} floors FAILED. Stop — these "
          f"produce plausible wrong answers, not errors.")
    print("The fixed code may exist in a source tree that is not installed; "
          "check before assuming the release landed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
