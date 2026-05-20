"""Geometry primitives shared by V-map / soft-attribution / forward modeling.

* :class:`SampleGrid` — voxel positions, grain map, sample mask packaged as
  torch tensors on a single device.
* :class:`BeamProfile` (+ :class:`TopHat`, :class:`Gaussian`) —
  differentiable beam-fraction-over-voxel kernels, ``torch.nn.Module``-based
  so refinable beam parameters integrate with :mod:`torch.optim`.

These live in :mod:`midas_transforms` (rather than :mod:`midas_pipeline`) so
that the forward model in :mod:`midas_transforms.radius.forward_model` can
consume them without a circular dependency on the orchestrator package.
"""
from .absorption import absorption_factor, path_length_in_sample
from .beam import BeamProfile, Gaussian, TopHat
from .sample import SampleGrid

__all__ = [
    "BeamProfile",
    "Gaussian",
    "SampleGrid",
    "TopHat",
    "absorption_factor",
    "path_length_in_sample",
]
