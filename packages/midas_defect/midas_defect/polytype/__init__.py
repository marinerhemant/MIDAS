"""Polytype satellite-peak diagnostics: activated axis, enhancement, lamella L.

Geometry-honest (use these): ``satellite_radial_excess`` (texture-safe excess +
discrete-vs-relrod verdict); ``find_satellite_axis`` / ``aggregate_lamella_thickness``
(one aggregate L along the measured satellite direction — NOT per-grain/per-variant).
Deprecated (raise by default): ``polytype_satellite_enhancement`` (voxel-count
inflation), ``per_grain_lamella_thickness`` (projection artifact). See AUDIT_2026-06-23.md.
"""

from .activated_axis import detect_activated_111_axis
from .aggregate_thickness import aggregate_lamella_thickness, find_satellite_axis
from .cell_index import (
    NINE_R_SEQUENCE,
    PolytypeCell,
    close_packed_basis,
    index_reflections,
    nine_r_from_fcc,
    polytype_reflections,
    structure_factor_intensity,
)
from .doublet_survey import DoubletSurvey, survey_doublets
from .fault_balance import periodic_aperiodic_balance
from .finite_stack import (
    FiniteStack,
    build_close_packed_slab,
    g_111,
    on_axis_ladder,
    slab_intensity,
)
from .ladder import SatelliteLadder, build_satellite_ladder, decontaminate_ladder
from .lamella_thickness import per_grain_lamella_thickness
from .modulation_tilt import fit_modulation_tilt, sigma3_landing_residual
from .modulation_type import ModulationFit, classify_modulation
from .satellite_doublet import SatelliteDoublet, resolve_satellite_doublet
from .satellite_excess import satellite_radial_excess
from .satellite_intensity import polytype_satellite_enhancement

__all__ = [
    "NINE_R_SEQUENCE",
    "PolytypeCell",
    "SatelliteDoublet",
    "SatelliteLadder",
    "close_packed_basis",
    "structure_factor_intensity",
    "aggregate_lamella_thickness",
    "build_satellite_ladder",
    "decontaminate_ladder",
    "index_reflections",
    "nine_r_from_fcc",
    "polytype_reflections",
    "detect_activated_111_axis",
    "find_satellite_axis",
    "fit_modulation_tilt",
    "periodic_aperiodic_balance",
    "sigma3_landing_residual",
    "per_grain_lamella_thickness",      # deprecated
    "polytype_satellite_enhancement",   # deprecated
    "resolve_satellite_doublet",
    "satellite_radial_excess",
    # finite-stack 3-D simulator (on-axis extinction + modulation proof)
    "FiniteStack",
    "build_close_packed_slab",
    "slab_intensity",
    "on_axis_ladder",
    "g_111",
    # sample-wide doublet survey
    "DoubletSurvey",
    "survey_doublets",
    # modulation-type classifier
    "ModulationFit",
    "classify_modulation",
]
