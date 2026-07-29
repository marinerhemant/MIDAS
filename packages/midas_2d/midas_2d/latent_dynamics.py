"""Equation-of-motion discovery -- now provided by the shared ``midas_invert``
leaf (``midas_invert.sindy``).  Thin re-export so existing ``midas_2d`` imports
are unchanged; ``discover_eom`` is the historical name for ``discover_dynamics``.
"""
from midas_invert.sindy import (
    discover_dynamics as discover_eom,
    integrate_latent_ode,
    library_terms,
)

__all__ = ["library_terms", "integrate_latent_ode", "discover_eom"]
