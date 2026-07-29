"""PRISMS-Plasticity I/O stub.

Mirrors :mod:`damask_io`. Implementations are deferred; signatures fixed so
collaborator code can target them today and the body lands later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_NOT_IMPLEMENTED_MSG = (
    "{name} is a midas_defect.cpfem stub. The signature is reserved for future "
    "PRISMS-Plasticity integration; please open an issue if you need a real "
    "implementation."
)


def write_prisms_initial_microstructure(
    OM: NDArray[np.floating],
    pos: NDArray[np.floating],
    grain_radii: NDArray[np.floating],
    output_path: str = "microstructure.txt",
) -> None:
    """Emit a PRISMS-Plasticity-compatible initial-microstructure file."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(name="write_prisms_initial_microstructure"))


def read_prisms_grain_output(prisms_output_path: str) -> dict[str, Any]:
    """Parse PRISMS-Plasticity per-grain output."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(name="read_prisms_grain_output"))


__all__ = ["read_prisms_grain_output", "write_prisms_initial_microstructure"]
