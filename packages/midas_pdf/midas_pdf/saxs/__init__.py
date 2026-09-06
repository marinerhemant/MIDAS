"""SAXS layer for midas-pdf: joint SAXS + PDF refinement.

**The SAXS primitives moved to** :mod:`midas_saxs`. ``form_factors``, ``model``,
``wide_band`` and ``core_shell`` now live there, because a SAXS package should
not have to depend on a total-scattering/PDF package to get a sphere form
factor, and because MIDAS gained a second, much larger SAXS consumer (2-D
detector-image simulation from dislocation networks). They are re-exported below
unchanged, so every historical import path keeps working::

    from midas_pdf.saxs import SAXSModel, sphere_form_factor_squared   # still fine

New code should import them from :mod:`midas_saxs` directly.

What stays here is the genuinely PDF-coupled part -- simultaneous refinement of a
SAXS curve and a PDF against one structural model:

  * :mod:`midas_pdf.saxs.joint` -- :func:`joint_refine`, and
    :func:`sphere_characteristic_function` (the real-space companion that damps
    ``G(r)`` for a finite particle; a PDF quantity, not a SAXS one).
  * :mod:`midas_pdf.saxs.joint_bayesian` -- SVI / NUTS posteriors.
  * :mod:`midas_pdf.saxs.joint_three_way` and its Bayesian counterpart --
    SAXS + PDF + Bragg.
"""

# Re-exported from midas-saxs. Do NOT re-port these; one definition only.
from midas_saxs import (  # noqa: F401
    SAXSModel,
    core_shell_sphere_form_factor_squared,
    cylinder_form_factor_squared,
    ellipsoid_form_factor_squared,
    GuinierFit,
    guinier_fit,
    kratky_plot,
    lognormal_quadrature_nodes,
    multi_shell_sphere_form_factor_squared,
    percus_yevick_S,
    PorodFit,
    porod_fit,
    porod_invariant,
    sphere_form_factor_squared,
    worm_like_chain_form_factor_squared,
)

from .joint import (
    sphere_characteristic_function, joint_refine, JointRefineResult,
)
from .joint_bayesian import (
    JointBayesianResult, joint_refine_svi, joint_refine_nuts,
)
from .joint_three_way import (
    ThreeWayJointResult, joint_refine_three_way,
)
from .joint_three_way_bayesian import (
    ThreeWayBayesianResult, joint_three_way_refine_svi,
    joint_three_way_refine_nuts,
)

__all__ = [
    # --- re-exported from midas_saxs ---
    "sphere_form_factor_squared",
    "ellipsoid_form_factor_squared",
    "cylinder_form_factor_squared",
    "percus_yevick_S",
    "SAXSModel", "lognormal_quadrature_nodes",
    "core_shell_sphere_form_factor_squared",
    "multi_shell_sphere_form_factor_squared",
    "GuinierFit", "guinier_fit", "PorodFit", "porod_fit",
    "porod_invariant", "kratky_plot",
    "worm_like_chain_form_factor_squared",
    # --- joint SAXS + PDF, which stays here ---
    "sphere_characteristic_function",
    "joint_refine", "JointRefineResult",
    "joint_refine_svi", "joint_refine_nuts", "JointBayesianResult",
    "ThreeWayJointResult", "joint_refine_three_way",
    "ThreeWayBayesianResult", "joint_three_way_refine_svi",
    "joint_three_way_refine_nuts",
]
