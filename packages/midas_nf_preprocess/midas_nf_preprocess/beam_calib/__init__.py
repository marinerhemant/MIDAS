"""Beam-centre and detector-distance calibration for NF-HEDM.

These are the measurements handbook §6 describes as procedures but which had
no implementation anywhere in ``packages/`` -- every beamtime re-wrote them in
a scratch directory, so the same mistakes recurred. The handbook even cited
reference implementations living outside the repository.

Contents
--------
``triangulate``
    Absolute sample-to-detector distance from spots seen at two or more
    detector positions (§6i / §6i-bis). Runs its own nulls and gates and
    **refuses to return a distance when they fail**.

Design rule for this subpackage
-------------------------------
**Controls belong inside the function that reports the number.** Every one of
these quantities has, at least once, been reported wrongly from a hand-rolled
script whose null was a no-op or whose gate was skipped -- including by
someone who had just read the warning against it. A caller who cannot forget
a control, and cannot misimplement it, is worth more than a documented
procedure.
"""

from .triangulate import (
    PairSolve,
    TriangulationResult,
    accidental_match_rate,
    triangulate,
)

__all__ = [
    "triangulate",
    "TriangulationResult",
    "PairSolve",
    "accidental_match_rate",
]
