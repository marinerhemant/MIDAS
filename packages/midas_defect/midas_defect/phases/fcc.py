"""Slip and twin systems for face-centred-cubic crystals.

References
----------
Hirth & Lothe, *Theory of Dislocations*, 2nd ed., 1982 (Thompson tetrahedron,
Shockley partials).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._systems import cubic_systems

# {111}<110> -- 12 perfect slip systems (Thompson tetrahedron)
FCC_SLIP_111_110: NDArray[np.floating] = cubic_systems((1, 1, 1), (1, 1, 0))

# {111}<112> -- 12 twin / Shockley-partial systems
FCC_TWIN_111_112: NDArray[np.floating] = cubic_systems((1, 1, 1), (1, 1, 2))

# Partial slip uses the same geometry as the twin family; Burgers magnitude
# differs (b_partial = a/6 <112> = b_perfect/sqrt(3)) but the unit (n, b)
# direction pair is shared.
FCC_PARTIAL_111_112: NDArray[np.floating] = FCC_TWIN_111_112

# Textbook FCC twinning shear: gamma = 1/sqrt(2)
GAMMA_TWIN_FCC: float = 1.0 / np.sqrt(2.0)


__all__ = [
    "FCC_PARTIAL_111_112",
    "FCC_SLIP_111_110",
    "FCC_TWIN_111_112",
    "GAMMA_TWIN_FCC",
]
