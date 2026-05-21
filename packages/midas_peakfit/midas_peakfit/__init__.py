"""midas_peakfit: differentiable PyTorch peak-fitting for FF-HEDM Zarr archives.

Drop-in replacement for the C tool ``PeaksFittingOMPZarrRefactor``. Replaces
NLopt Nelder-Mead with batched Levenberg-Marquardt on CPU/CUDA. Output binary
files (``AllPeaks_PS.bin``, ``AllPeaks_PX.bin``) match the C tool's format
(see ``FF_HEDM/src/PeaksFittingConsolidatedIO.h``).

Entry points:
    peakfit_torch CLI         — see ``midas_peakfit.cli``
    midas_peakfit.run(...)    — programmatic; see ``midas_peakfit.pipeline_main``
"""

__version__ = "0.4.1"

from midas_peakfit.lm import LMConfig, lm_solve  # noqa: E402,F401
from midas_peakfit.lm_generic import (  # noqa: E402,F401
    GenericLMConfig,
    lm_solve_arrowhead,
    lm_solve_generic,
)
from midas_peakfit.params import ZarrParams  # noqa: E402,F401
from midas_peakfit.reparam import u_to_x, x_to_u  # noqa: E402,F401

# Promoted spec / parameter / pack / inference / spline / loss substrate.
# These were originally in midas_calibrate_v2; promoted to midas_peakfit so
# midas_calibrate_v2, midas_fit_grain, and midas_joint_ff_calibrate share a
# single differentiable-inverse-problem stack.
from midas_peakfit.parameter import (  # noqa: E402,F401
    GaussianPrior,
    HalfCauchyPrior,
    Parameter,
    Prior,
    UniformPrior,
)
from midas_peakfit.transforms import (  # noqa: E402,F401
    Identity,
    Log,
    Logit,
    Scaled,
    Transform,
)
from midas_peakfit.spec import ParameterSpec  # noqa: E402,F401
from midas_peakfit.pack import (  # noqa: E402,F401
    MultiPackInfo,
    PackInfo,
    pack_multi,
    pack_spec,
    refined_bounds,
    refined_indices,
    refined_subset,
    unpack_spec,
    write_refined_back,
)
from midas_peakfit.lm_spec import lm_minimise  # noqa: E402,F401
from midas_peakfit.laplace import (  # noqa: E402,F401
    LaplaceResult,
    fisher_at_map,
    laplace_at_map,
    report_laplace,
)
from midas_peakfit.spline import (  # noqa: E402,F401
    TPSpline,
    fit_tps,
    fit_tps_refinable,
)
from midas_peakfit.constraints import (  # noqa: E402,F401
    gaussian_prior_residual,
    zero_sum_residual,
)
from midas_peakfit.prior import sum_log_prior  # noqa: E402,F401

__all__ = [
    # Existing peak-fitting API.
    "GenericLMConfig",
    "LMConfig",
    "ZarrParams",
    "__version__",
    "lm_solve",
    "lm_solve_arrowhead",
    "lm_solve_generic",
    "u_to_x",
    "x_to_u",
    # Spec / parameter / pack substrate.
    "Parameter",
    "Prior",
    "GaussianPrior",
    "HalfCauchyPrior",
    "UniformPrior",
    "Transform",
    "Identity",
    "Log",
    "Logit",
    "Scaled",
    "ParameterSpec",
    "PackInfo",
    "MultiPackInfo",
    "pack_spec",
    "unpack_spec",
    "pack_multi",
    "refined_indices",
    "refined_bounds",
    "refined_subset",
    "write_refined_back",
    # Inference.
    "lm_minimise",
    "LaplaceResult",
    "laplace_at_map",
    "fisher_at_map",
    "report_laplace",
    # Spline.
    "TPSpline",
    "fit_tps",
    "fit_tps_refinable",
    # Loss / constraints / priors.
    "zero_sum_residual",
    "gaussian_prior_residual",
    "sum_log_prior",
]
