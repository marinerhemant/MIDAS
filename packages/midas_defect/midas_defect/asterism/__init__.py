"""Per-grain asterism (peak-shape) analysis.

Geometry-honest (use this): ``per_grain_asterism_local`` — per-spot-local radial(strain)
/ tangential(rotation) split. The global second-moment tensor + single-mean_q
radial/azimuthal split (``per_grain_asterism_tensor`` + ``asterism_anisotropy_per_grain``
radial/azimuthal fields, ``edge_fraction_per_grain``) mixes a grain's multi-directional
spots and mislabels strain vs rotation (AUDIT_2026-06-23.md); the eigenvalue *magnitude*
is still usable, the radial/azimuthal split is not.
"""

from .direction import edge_fraction_per_grain
from .eigenvalue_spectrum import asterism_anisotropy_per_grain
from .family_asterism import family_asterism_arc, reflection_directions
from .local_decomposition import per_grain_asterism_local
from .second_moment import per_grain_asterism_tensor

__all__ = [
    "asterism_anisotropy_per_grain",
    "edge_fraction_per_grain",
    "family_asterism_arc",
    "reflection_directions",
    "per_grain_asterism_local",
    "per_grain_asterism_tensor",
]
