"""FEPX crystal-plasticity finite-element I/O stub.

Mirrors :mod:`damask_io`. Implementations are deferred; signatures fixed so
collaborator code can target them today and the body lands later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_NOT_IMPLEMENTED_MSG = (
    "{name} is a midas_defect.cpfem stub. The signature is reserved for future "
    "FEPX integration; please open an issue if you need a real implementation."
)


def write_fepx_initial_microstructure(
    OM: NDArray[np.floating],
    pos: NDArray[np.floating],
    grain_radii: NDArray[np.floating],
    output_path: str = "microstructure.mesh",
) -> None:
    """Emit a FEPX-compatible mesh / orientation file pair."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(name="write_fepx_initial_microstructure"))


def read_fepx_grain_output(fepx_output_path: str) -> dict[str, Any]:
    """Parse FEPX per-grain output back to AnalysisResult-compatible arrays."""
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(name="read_fepx_grain_output"))


__all__ = ["read_fepx_grain_output", "write_fepx_initial_microstructure"]
