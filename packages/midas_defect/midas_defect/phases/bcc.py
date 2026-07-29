"""Slip and twin systems for body-centred-cubic crystals.

References
----------
Christian, *The Theory of Transformations in Metals and Alloys*, Pergamon, 2002
(Sec. 21.2: BCC twinning, {112}<111> shear).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._systems import cubic_systems

# {110}<111> -- 12 primary slip systems
BCC_SLIP_110_111: NDArray[np.floating] = cubic_systems((1, 1, 0), (1, 1, 1))

# {112}<111> -- 12 secondary slip systems
BCC_SLIP_112_111: NDArray[np.floating] = cubic_systems((1, 1, 2), (1, 1, 1))

# {123}<111> -- tertiary slip (often omitted in CP simulations)
BCC_SLIP_123_111: NDArray[np.floating] = cubic_systems((1, 2, 3), (1, 1, 1))

# {112}<111> -- 12 twin systems; geometrically identical to {112}<111> slip
# but the shear sense is fixed (twin has a unique sign).
BCC_TWIN_112_111: NDArray[np.floating] = BCC_SLIP_112_111

# Textbook BCC twinning shear: gamma = 1/sqrt(2)
GAMMA_TWIN_BCC: float = 1.0 / np.sqrt(2.0)


__all__ = [
    "BCC_SLIP_110_111",
    "BCC_SLIP_112_111",
    "BCC_SLIP_123_111",
    "BCC_TWIN_112_111",
    "GAMMA_TWIN_BCC",
]
