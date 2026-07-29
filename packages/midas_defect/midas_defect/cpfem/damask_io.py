"""DAMASK crystal-plasticity finite-element I/O stub.

Reserves the canonical I/O signatures so external collaborators have a stable
import surface to target. Implementations are deferred until a real DAMASK
roundtrip can be exercised against a representative simulation.

If you need this now, open an issue describing the DAMASK version and the
microstructure schema you're emitting -- the implementation effort is small
but format-version dependent.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_NOT_IMPLEMENTED_MSG = (
    "{name} is a midas_defect.cpfem stub. The signature is reserved for future "
    "DAMASK integration; please open an issue if you need a real implementation."
)


def write_damask_initial_microstructure(
    OM: NDArray[np.floating],
    pos: NDArray[np.floating],
    grain_radii: NDArray[np.floating],
    rho_per_grain: NDArray[np.floating] | None = None,
    output_path: str = "microstructure.hdf5",
) -> None:
    """Write a DAMASK-compatible HDF5 microstructure file.

    Planned columns
    ---------------
    grain_id : int
    orientation : (3, 3) per grain
    centroid : (3,) per grain (micrometres)
    radius   : float per grain (micrometres)
    rho_initial : float per grain (m^-2), optional

    Implementation pending; raises :class:`NotImplementedError`.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(name="write_damask_initial_microstructure"))


def read_damask_grain_output(damask_output_path: str) -> dict[str, Any]:
    """Read DAMASK per-grain stress / strain output.

    Returns
    -------
    dict with planned keys
        grain_ids       (n_grains,) int
        sigma_per_grain (n_grains, 3, 3) Pa
        eps_per_grain   (n_grains, 3, 3)
        rho_per_grain   (n_grains,) m^-2

    Implementation pending; raises :class:`NotImplementedError`.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG.format(name="read_damask_grain_output"))


__all__ = ["read_damask_grain_output", "write_damask_initial_microstructure"]
