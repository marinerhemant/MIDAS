"""Forwarding shim: this module moved to :mod:`midas_saxs.form_factors`.

Kept so the deep import path ``midas_pdf.saxs.form_factors`` keeps resolving for
code written before the move. There is no implementation here and there must
never be one -- import from :mod:`midas_saxs` in new code.
"""
from midas_saxs.form_factors import *          # noqa: F401,F403
from midas_saxs.form_factors import __all__    # noqa: F401
