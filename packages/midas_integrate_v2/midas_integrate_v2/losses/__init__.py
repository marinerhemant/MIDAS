"""Curated loss functions for joint integration + calibration refinement.

Each loss is a plain :class:`torch.nn.Module` that takes the integrated
2D ``(n_eta, n_r)`` array (or a 1D profile derived from it) and returns
a scalar tensor with autograd hooked up to the upstream
:class:`IntegrationSpec` parameters.

Three families:

- **Profile-target** (:class:`ProfileMSELoss`, :class:`ProfileWeightedMSELoss`):
  match an observed 1D profile to a reference. Useful for instrument
  matching, transfer-function learning.
- **Geometry-driven** (:class:`EtaUniformityLoss`, :class:`PeakPositionLoss`):
  exploit calibrant-specific physical constraints (Debye-Scherrer
  rings should be uniform along η; peak positions should land on
  predicted R values). The most common refinement targets.
- **Bayesian-prior** (:class:`GaussianPriorLoss`): wrap any spec
  parameter with a Gaussian regulariser around a prior mean. Composes
  with any data loss above.
"""
from .profile import ProfileMSELoss, ProfileWeightedMSELoss
from .geometry import EtaUniformityLoss, PeakPositionLoss
from .bayesian import GaussianPriorLoss
from .multi import MultiImageLoss, BatchedSpecLoss
from .quasi_2d import EtaSliceLoss, WedgeLoss, RingMaskedLoss

__all__ = [
    "ProfileMSELoss",
    "ProfileWeightedMSELoss",
    "EtaUniformityLoss",
    "PeakPositionLoss",
    "GaussianPriorLoss",
    "MultiImageLoss",
    "BatchedSpecLoss",
    "EtaSliceLoss",
    "WedgeLoss",
    "RingMaskedLoss",
]
