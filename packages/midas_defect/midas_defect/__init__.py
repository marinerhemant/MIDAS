"""midas_defect — phase-agnostic diffuse-scattering defect metrology for FF-HEDM.

Quantifies the diffuse field around and between the Bragg peaks of an indexed
FF-HEDM dataset (asterism, fault rods, the full intensity budget, defect /
selection-rule tests) to produce a per-grain defect inventory. Driven by a
`midas_hkls.Crystal` + a `Geometry`; nothing is hard-wired to a phase.
"""

import os

# `geometry.pixel_to_qlab` reuses `midas_transforms.apply_tilt_distortion`, which
# links a second OpenMP runtime alongside torch's. Allow the duplicate at import
# time (before either runtime initializes) so callers don't hit a hard crash.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.1"
