"""Forwarding shim: this module moved to :mod:`midas_saxs.wide_band`.

Kept so the deep import path ``midas_pdf.saxs.wide_band`` keeps resolving for
code written before the move. There is no implementation here and there must
never be one -- import from :mod:`midas_saxs` in new code.
"""
from midas_saxs.wide_band import *          # noqa: F401,F403
from midas_saxs.wide_band import __all__    # noqa: F401
