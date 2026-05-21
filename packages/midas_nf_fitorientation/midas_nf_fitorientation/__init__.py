"""midas-nf-fitorientation: differentiable NF-HEDM orientation + calibration fitter.

Drop-in replacement for the three NF-HEDM C executables
:program:`FitOrientationOMP`, :program:`FitOrientationParameters`, and
:program:`FitOrientationParametersMultiPoint`. Built on the
:mod:`midas_diffract` autograd forward model with PyTorch L-BFGS instead
of NLopt Nelder-Mead, a Gaussian-splat differentiable surrogate for
:func:`CalcFracOverlap`, and tanh-reparameterised box constraints with
optional Tikhonov regularisation on calibration parameters.

Quick start
-----------
The CLI entry points mirror the C executables::

    midas-nf-fit-orientation params.txt blockNr nBlocks nCPUs
    midas-nf-fit-parameters  params.txt rowNr [nCPUs]
    midas-nf-fit-multipoint  params.txt [nCPUs]

The Python API exposes the same three drivers::

    from midas_nf_fitorientation import (
        fit_orientation_run, fit_parameters_run, fit_multipoint_run,
    )
"""

__version__ = "0.3.2"

from .params import FitParams, parse_paramfile
from .fit_orientation import fit_orientation_run
from .fit_parameters import fit_parameters_run
from .fit_multipoint import fit_multipoint_run

__all__ = [
    "FitParams",
    "parse_paramfile",
    "fit_orientation_run",
    "fit_parameters_run",
    "fit_multipoint_run",
    "__version__",
]
