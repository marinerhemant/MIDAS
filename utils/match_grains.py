#!/usr/bin/env python
"""DEPRECATED shim -- match_grains now lives in ``midas_process_grains``.

Moved 2026-09-01. The script version hardcoded a positional Grains.csv layout
that matched neither the 47- nor the 53-column real file (it put ``eFab11`` at
19, ``Confidence`` at 38 and ``Radius`` at 42, where both real widths carry
``DiffPos`` at 19, ``GrainRadius`` at 22 and ``Confidence`` at 23), so
``--size-filter`` filtered on ``RMSErrorStrain``. The package version resolves
columns by name through ``midas_process_grains.io.read`` and takes
misorientation from ``midas_stress``.

Use:  midas-match-grains match|stitch ...
  or: from midas_process_grains.matching import match_grains, stitch_layers

This shim forwards argv so existing invocations keep working.
"""
import sys
import warnings

warnings.warn(
    "utils/match_grains.py is deprecated; use `midas-match-grains` or "
    "midas_process_grains.matching. This shim forwards to the package.",
    DeprecationWarning, stacklevel=2,
)

try:
    from midas_process_grains.matching import *          # noqa: F401,F403
    from midas_process_grains.matching import main
except ImportError as exc:                                # pragma: no cover
    sys.exit(
        "match_grains has moved into midas_process_grains, which is not "
        f"importable here ({exc}). Install it with:\n"
        "    pip install -e packages/midas_process_grains"
    )

if __name__ == "__main__":
    sys.exit(main())
