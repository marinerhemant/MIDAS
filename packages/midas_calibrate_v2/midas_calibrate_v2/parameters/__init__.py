"""Parameter framework — every input is a Parameter with metadata.

The unifying abstraction across all v2 capabilities (single, multi-image,
Bayesian, NN-residual): every quantity that *could* be refined is declared
as a :class:`Parameter` with ``(init, refined?, prior?, bounds?, transform?)``
metadata; the forward model takes a single packed tensor; refined components
carry autograd and fixed components don't.
"""
from .parameter import Parameter, Prior, GaussianPrior, HalfCauchyPrior, UniformPrior
from .spec import CalibrationSpec, MultiImageSpec
from .pack import pack_spec, unpack_spec, refined_indices, refined_bounds, refined_subset
from .transforms import Transform, Identity, Log, Logit, Scaled

__all__ = [
    "Parameter", "Prior", "GaussianPrior", "HalfCauchyPrior", "UniformPrior",
    "CalibrationSpec", "MultiImageSpec",
    "pack_spec", "unpack_spec", "refined_indices", "refined_bounds", "refined_subset",
    "Transform", "Identity", "Log", "Logit", "Scaled",
]
