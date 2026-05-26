"""Mode dispatcher: ``legacy`` / ``paper_claim`` / ``spot_aware`` / ``adaptive``.

Each mode is just a parameter override over the common pipeline. The actual
algorithm lives in ``pipeline.py`` and ``compute/*``; this module only knows
how to substitute mode-appropriate defaults into a :class:`ProcessGrainsParams`
instance.

``adaptive`` is the data-driven mode (no user-set MisoriTol). It runs the
spot_aware pipeline but with the misori threshold *derived at run-time from
the antimode of the pairwise-misorientation histogram* of the alive candidates.
See ``compute.adaptive.derive_misori_tol`` for the antimode finder. Empirical
result on a Ni FF dataset (~57 k candidates → ~11 k grains): θ* = 0.011°,
matching the §3.6 paper specification and contradicting the legacy 0.4° rule.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional

from .params import ProcessGrainsParams


VALID_MODES = ("legacy", "paper_claim", "spot_aware", "adaptive")


def apply_mode_defaults(
    params: ProcessGrainsParams,
    mode: str,
) -> ProcessGrainsParams:
    """Substitute per-mode defaults into ``params`` for fields the user
    didn't set explicitly.

    The user's explicit values always win. Sentinels:
      * ``MisoriTol is None`` means "not set in paramstest"; substitute the
        per-mode default.

    Returns a *copy* so the caller's params object isn't mutated.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}; got {mode!r}")
    p = replace(params)

    if mode == "legacy":
        # Reproduce current C ProcessGrains.c behaviour bit-for-bit.
        # Phase 1 misori cutoff is 0.4° (FF_HEDM/src/ProcessGrains.c:195).
        if p.MisoriTol is None:
            p.MisoriTol = 0.4
        # Legacy strain: the C uses NLOPT-bounded Kenesei. We use the
        # bounded SciPy Kenesei which matches the same algorithm.
        p.StrainMethod = "kenesei"
    elif mode == "paper_claim":
        # The §3.6 spec the C code does not actually enforce.
        if p.MisoriTol is None:
            p.MisoriTol = 0.01
        p.JaccardTol = 0.9
        # Paper §3.7 names two methods; "lattice parameter refinement" is
        # described as recommended for the average strain state. Use it
        # here, but the user can override.
        p.StrainMethod = "fable_beaudoin"
    elif mode == "spot_aware":
        # The new default — what we recommend.
        if p.MisoriTol is None:
            # 0.5° empirically gathers all C-cluster equivalents (intra-cluster
            # pair misori reaches up to ~0.5° in real data; tighter values
            # split real grains). Phase 2 spot-aware sub-clustering then
            # splits sub-grains based on hkl-spot overlap.
            p.MisoriTol = 0.5
        # StrainMethod stays whatever the user picked.
        # The ParamsTest default is "kenesei" (bounded Kenesei lstsq,
        # matches the C reference numerically). Use --strain-method
        # fable_beaudoin for the lattice-parameter route, or
        # --strain-method both to emit both.
    elif mode == "adaptive":
        # The misori threshold is DERIVED from the data at run-time. Leaving
        # MisoriTol as None here is a sentinel that ``pipeline.run`` honours:
        # it calls ``compute.adaptive.derive_misori_tol(quats_alive,
        # space_group)`` to find the antimode of the pairwise-misori
        # histogram. The user can still override MisoriTol explicitly in
        # paramstest; that bypasses the antimode derivation.
        #
        # Defaults inherited from spot_aware: Phase 2 spot-aware machinery
        # disambiguates sub-grains by spot overlap, Pass A merges adjacent
        # clusters by pooled-spot Jaccard, etc. The adaptive mode only
        # changes WHERE Phase 1's misori cutoff lives.
        pass   # MisoriTol stays None; pipeline.run resolves it
    return p.validated()


def misori_tol_rad(params: ProcessGrainsParams) -> float:
    """Return the misorientation cutoff as radians; raises if unresolved."""
    if params.MisoriTol is None:
        raise ValueError(
            "MisoriTol unresolved. Call apply_mode_defaults() before running, "
            "or for ``adaptive`` mode let ``pipeline.run`` derive it from the "
            "antimode (compute/adaptive.derive_misori_tol)."
        )
    return float(params.MisoriTol) * math.pi / 180.0


def needs_adaptive_misori(params: ProcessGrainsParams, mode: str) -> bool:
    """True when ``pipeline.run`` should derive MisoriTol from the data
    (adaptive mode with no user override)."""
    return mode == "adaptive" and params.MisoriTol is None
