"""Slip and twin system tables, phase-aware.

FCC and BCC are cubic and generated from Miller-index families via
:func:`_systems.cubic_systems`. HCP is parameterized by ``c/a`` and exposed
through :func:`hcp.hcp_systems`.
"""

from . import bcc, fcc, hcp
from .bcc import (
    BCC_SLIP_110_111,
    BCC_SLIP_112_111,
    BCC_SLIP_123_111,
    BCC_TWIN_112_111,
    GAMMA_TWIN_BCC,
)
from .fcc import (
    FCC_PARTIAL_111_112,
    FCC_SLIP_111_110,
    FCC_TWIN_111_112,
    GAMMA_TWIN_FCC,
)
from .hcp import (
    bravais_to_miller,
    gamma_twin_hcp_compressive,
    gamma_twin_hcp_tensile,
    hcp_systems,
    miller_to_bravais,
)

from ..types import CrystalPhase


def n_systems(phase: CrystalPhase, kind: str = "slip") -> int:
    """Number of slip or twin systems for the canonical family of each phase.

    ``kind`` is ``"slip"`` (primary slip family) or ``"twin"`` (primary twin
    family). HCP uses basal slip and tensile twin as the primary defaults.
    """
    if kind not in ("slip", "twin"):
        raise ValueError(f"kind must be 'slip' or 'twin'; got {kind!r}")
    table = {
        (CrystalPhase.FCC, "slip"): FCC_SLIP_111_110.shape[0],
        (CrystalPhase.FCC, "twin"): FCC_TWIN_111_112.shape[0],
        (CrystalPhase.BCC, "slip"): BCC_SLIP_110_111.shape[0],
        (CrystalPhase.BCC, "twin"): BCC_TWIN_112_111.shape[0],
        (CrystalPhase.HCP, "slip"): len(hcp.BASAL_SLIP_4INDEX),
        (CrystalPhase.HCP, "twin"): len(hcp.TWIN_TENSILE_4INDEX),
    }
    return table[(phase, kind)]


__all__ = [
    "BCC_SLIP_110_111",
    "BCC_SLIP_112_111",
    "BCC_SLIP_123_111",
    "BCC_TWIN_112_111",
    "CrystalPhase",
    "FCC_PARTIAL_111_112",
    "FCC_SLIP_111_110",
    "FCC_TWIN_111_112",
    "GAMMA_TWIN_BCC",
    "GAMMA_TWIN_FCC",
    "bcc",
    "bravais_to_miller",
    "fcc",
    "gamma_twin_hcp_compressive",
    "gamma_twin_hcp_tensile",
    "hcp",
    "hcp_systems",
    "miller_to_bravais",
    "n_systems",
]
