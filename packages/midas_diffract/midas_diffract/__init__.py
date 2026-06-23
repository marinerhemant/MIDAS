"""midas-diffract: End-to-end differentiable forward model for HEDM.

A PyTorch-based differentiable forward simulator for far-field, near-field,
and point-focused High-Energy Diffraction Microscopy, with pixel-exact
agreement against the canonical C reference simulators in MIDAS.

See the companion paper for methodology:
    Sharma, Zhang, Andrejevic, Cherukara, "An End-to-End Differentiable Forward
    Model for High-Energy Diffraction Microscopy," IUCrJ (in prep, 2026).

Quick start
-----------
    import torch
    import midas_diffract as md

    # Construct the forward model from geometry + reflection list
    geom = md.HEDMGeometry(Lsd=..., y_BC=..., z_BC=..., px=..., ...)
    model = md.HEDMForwardModel(hkls=hkls_cart, thetas=thetas,
                                geometry=geom, hkls_int=hkls_int)

    # Forward: grain state -> predicted spots
    euler = torch.tensor([[phi1, Phi, phi2]], requires_grad=True)
    pos   = torch.tensor([[x, y, z]], requires_grad=True)
    spots = model(euler, pos, lattice_params=latc)

    # Scalar loss -> gradient to all inputs via autograd
    loss = (spots.omega * spots.valid).pow(2).sum()
    loss.backward()
"""

__version__ = "0.2.0"

from .forward import (
    HEDMForwardModel,
    HEDMGeometry,
    ScanConfig,
    TriVoxelConfig,
    SpotDescriptors,
)
from .hkls import hkls_for_forward_model
from .losses import SpotMatchingLoss
from .optimize import optimize_single_grain, evaluate_recovery

__all__ = [
    "HEDMForwardModel",
    "HEDMGeometry",
    "ScanConfig",
    "TriVoxelConfig",
    "SpotDescriptors",
    "SpotMatchingLoss",
    "hkls_for_forward_model",
    "optimize_single_grain",
    "evaluate_recovery",
    "__version__",
]
