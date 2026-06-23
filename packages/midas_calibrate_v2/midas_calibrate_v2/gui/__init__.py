"""GUI helpers for midas-calibrate-v2.

Lightweight Tk + matplotlib tools for cases where the fully-automated
auto-seed cannot work:

  * Off-detector beam (high-energy / large-Lsd geometries).
  * Limited azimuthal arc coverage (panel gaps, beamstop arms).
  * Tiled detectors where chord-bisector picks the wrong rings.

The ringpicker produces a seed (BC, Lsd) consumable verbatim by
``midas_calibrate_v2.pipelines.first_time.first_time_calibrate`` with
``auto_seed=False``.
"""

from ._circle_fit import (
    kasa_circle_fit,
    geometric_lm_refine,
    joint_bc_lsd_fit,
)

__all__ = [
    "kasa_circle_fit",
    "geometric_lm_refine",
    "joint_bc_lsd_fit",
]
