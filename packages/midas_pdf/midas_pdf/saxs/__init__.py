"""Small-angle X-ray scattering (SAXS) forward model + joint SAXS+PDF refinement.

Rev 8 additions:

  * :mod:`midas_pdf.saxs.form_factors` — sphere / ellipsoid / cylinder
    |F(Q)|² and Percus-Yevick hard-sphere structure factor. All torch-
    differentiable.
  * :mod:`midas_pdf.saxs.model` — :class:`SAXSModel` with optional
    lognormal polydispersity + hard-sphere interparticle S(Q).
  * :mod:`midas_pdf.saxs.joint` — :func:`joint_refine` for simultaneous
    SAXS + PDF fitting.

See ``dev/SAXS_PDF_PLAN.md`` for the multi-week plan; Rev 8 executes
Days 1-3 of that plan (form factors + model + synthetic validation).
"""

from .form_factors import (
    sphere_form_factor_squared,
    ellipsoid_form_factor_squared,
    cylinder_form_factor_squared,
    percus_yevick_S,
)
from .model import SAXSModel, lognormal_quadrature_nodes
from .joint import (
    sphere_characteristic_function, joint_refine, JointRefineResult,
)
from .joint_bayesian import (
    JointBayesianResult, joint_refine_svi, joint_refine_nuts,
)
from .core_shell import (
    core_shell_sphere_form_factor_squared,
    multi_shell_sphere_form_factor_squared,
)
from .wide_band import (
    GuinierFit, guinier_fit, PorodFit, porod_fit,
    porod_invariant, kratky_plot,
    worm_like_chain_form_factor_squared,
)
from .joint_three_way import (
    ThreeWayJointResult, joint_refine_three_way,
)
from .joint_three_way_bayesian import (
    ThreeWayBayesianResult, joint_three_way_refine_svi,
    joint_three_way_refine_nuts,
)

__all__ = [
    "sphere_form_factor_squared",
    "ellipsoid_form_factor_squared",
    "cylinder_form_factor_squared",
    "percus_yevick_S",
    "SAXSModel", "lognormal_quadrature_nodes",
    "sphere_characteristic_function",
    "joint_refine", "JointRefineResult",
    "joint_refine_svi", "joint_refine_nuts", "JointBayesianResult",
    "core_shell_sphere_form_factor_squared",
    "multi_shell_sphere_form_factor_squared",
    "GuinierFit", "guinier_fit", "PorodFit", "porod_fit",
    "porod_invariant", "kratky_plot",
    "worm_like_chain_form_factor_squared",
    "ThreeWayJointResult", "joint_refine_three_way",
    "ThreeWayBayesianResult", "joint_three_way_refine_svi",
    "joint_three_way_refine_nuts",
]
