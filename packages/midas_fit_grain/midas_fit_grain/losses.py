"""Canonical residual-loss names — the single source of truth for callers.

``midas-fit-grain`` owns which losses exist and what they mean. Every pipeline
that shells out to it must take its ``--loss`` choices from here rather than
re-declaring them, because a hardcoded copy silently rots:

    midas_ff_pipeline  --loss  {pixel, angular, internal_angle}   default pixel
    midas_fit_grain    --loss  {full3d, angular, internal_angle}  default full3d

``pixel`` was retired (it was 2-D in (y, z), omitted omega, and left the
crystal free to rotate in ω — see dev/REFINEMENT_DRIFT_FIX.md), but the caller
kept offering it AND defaulting to it. Every single-detector
``midas-ff-pipeline run`` with default flags therefore died in argparse inside
the refiner. Multi-detector runs survived only because a separate branch
rewrote ``pixel`` to ``angular`` before dispatch.

Deliberately free of heavy imports (no torch) so a CLI can import it at
parse time.
"""

from __future__ import annotations

#: Losses ``midas-fit-grain`` accepts, in the order its ``--loss`` lists them.
LOSS_CHOICES: tuple[str, ...] = ("full3d", "angular", "internal_angle")

#: What ``midas-fit-grain`` uses when ``--loss`` is not given.
DEFAULT_LOSS: str = "full3d"

#: Retired names, mapped to their replacement. Callers should accept these so
#: existing scripts keep running, and say loudly that they are substituting.
DEPRECATED_LOSSES: dict[str, str] = {"pixel": "full3d"}

#: Losses evaluated in DETECTOR PIXELS, and therefore meaningless when the
#: refiner sees spots from several panels at once — each panel carries its own
#: beam centre and Lsd, so one global pixel residual mixes incompatible frames.
#:
#: ``full3d`` belongs here as much as the retired ``pixel`` did: it stacks
#: ``y_pixel``, ``z_pixel`` and ``Δω · r_px`` (residuals.py:152-168), all of
#: which are computed from ``y_BC``/``z_BC``/``px``. ``angular`` compares
#: (2θ, η, ω) and is the geometry-independent choice for a multi-panel run.
PANEL_DEPENDENT_LOSSES: frozenset[str] = frozenset({"pixel", "full3d"})

#: What to substitute on a multi-panel run.
MULTIDET_LOSS: str = "angular"


def resolve(loss: str) -> tuple[str, str | None]:
    """Map *loss* onto a name the refiner accepts.

    Returns ``(resolved, note)``; *note* is a human-readable explanation when a
    substitution happened and ``None`` when *loss* passed through untouched.
    Unknown names are returned unchanged so the refiner's own argparse produces
    the error, rather than this helper inventing a different one.
    """
    if loss in DEPRECATED_LOSSES:
        new = DEPRECATED_LOSSES[loss]
        return new, (f"loss {loss!r} was retired (2-D in y,z; omitted omega, so "
                     f"orientation drifted freely) — using {new!r}")
    return loss, None
