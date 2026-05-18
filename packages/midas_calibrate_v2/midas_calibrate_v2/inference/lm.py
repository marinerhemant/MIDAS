"""Re-export shim — spec-aware LM wrapper now lives in
``midas_peakfit.lm_spec``.  ``GenericLMConfig`` is also re-exported for
runners that imported it from this module historically.
"""
from midas_peakfit import GenericLMConfig
from midas_peakfit.lm_spec import lm_minimise

__all__ = ["lm_minimise", "GenericLMConfig"]
