"""Inference backends.

The same forward model + loss is consumed by:

- :mod:`lm`      Levenberg-Marquardt (extends midas_peakfit.lm_solve_generic)
- :mod:`lbfgs`   torch.optim.LBFGS for general smooth losses
- :mod:`adam`    Adam (used to train NN-residual models)
- :mod:`laplace` Hessian-at-MAP → Gaussian posterior (cheap UQ)
- :mod:`vi`      Mean-field Gaussian VI via pyro (full posterior, opt-in)
- :mod:`hmc`     NUTS via pyro (gold-standard posterior, opt-in)
"""
from . import lbfgs, adam, laplace
try:
    from . import vi
except ImportError:
    vi = None
try:
    from . import hmc
except ImportError:
    hmc = None
from . import lm

__all__ = ["lm", "lbfgs", "adam", "laplace", "vi", "hmc"]
