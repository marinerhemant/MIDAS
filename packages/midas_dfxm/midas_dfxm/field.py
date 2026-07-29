"""Deformation-gradient field container and the kinematic deform operator.

Phase 0 of ``implementation_plan.md``.

A DFXM sample is described by a voxelised field of deformation gradients
``F(r)`` (3x3 per voxel) relative to a reference lattice + reference orientation.
The observable in reciprocal space is the *local* scattering vector

    Q(r) = F(r)^{-T} @ G0

where ``G0`` is the reference reflection's reciprocal vector in the sample frame
(reciprocal vectors transform by the inverse transpose of ``F``). Polar
decomposition ``F = R @ U`` separates the local **rotation** ``R`` (mosaicity)
from the right **stretch** ``U`` (elastic strain), the two quantities DFXM
mosaicity- and strain-scans respectively probe.

Everything here is torch-differentiable (autograd flows to ``F``, the reference
orientation, and the lattice) and device/dtype-preserving. Reference reciprocal
geometry reuses ``midas_stress.tensor.lattice_params_to_A_matrix``; no lattice
math is re-derived.

Units: voxel positions in micrometers; lattice in Angstrom; ``|Q|`` in 1/Angstrom
(times ``2*pi``, i.e. ``|Q| = 2*pi/d``).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from midas_stress.tensor import lattice_params_to_A_matrix

_TWO_PI = 2.0 * torch.pi


def reciprocal_basis(lattice_params: torch.Tensor) -> torch.Tensor:
    """Reciprocal basis ``B`` (with the ``2*pi`` convention) from lattice params.

    ``lattice_params`` is ``(..., 6) = [a, b, c, alpha, beta, gamma]`` (Angstrom,
    degrees). Returns ``(..., 3, 3)`` with ``G_cartesian = B @ hkl`` and
    ``|G| = 2*pi/d``. Differentiable in the lattice parameters.
    """
    A = lattice_params_to_A_matrix(lattice_params)  # frac -> cartesian, columns = a,b,c
    return _TWO_PI * torch.linalg.inv(A).transpose(-1, -2)


def deform_reflection(F: torch.Tensor, G0: torch.Tensor) -> torch.Tensor:
    """Local scattering vector ``Q = F^{-T} @ G0`` for a field of ``F``.

    ``F`` is ``(..., 3, 3)`` and ``G0`` is ``(3,)`` or ``(..., 3)``. Solved as
    ``F^T Q = G0`` (no explicit inverse) for numerical stability and clean
    gradients. Returns ``(..., 3)``.
    """
    Ft = F.transpose(-1, -2)
    rhs = G0.expand(*F.shape[:-2], 3).unsqueeze(-1)
    return torch.linalg.solve(Ft, rhs).squeeze(-1)


def polar_decomposition(F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Right polar decomposition ``F = R @ U`` (R rotation, U SPD stretch).

    Uses the symmetric eigendecomposition of ``C = F^T F`` (differentiable for
    distinct eigenvalues, the physical regime for deformation gradients). Returns
    ``(R, U)`` each ``(..., 3, 3)``.
    """
    C = F.transpose(-1, -2) @ F
    w, V = torch.linalg.eigh(C)  # C = V diag(w) V^T, w > 0
    sqrt_w = torch.sqrt(w.clamp_min(0.0))
    U = V @ torch.diag_embed(sqrt_w) @ V.transpose(-1, -2)
    U_inv = V @ torch.diag_embed(1.0 / sqrt_w) @ V.transpose(-1, -2)
    R = F @ U_inv
    return R, U


def small_strain_from_F(F: torch.Tensor) -> torch.Tensor:
    """Infinitesimal (symmetric) strain tensor ``eps = sym(F) - I``.

    ``eps = 0.5 (F + F^T) - I``. Adequate for the small elastic strains DFXM
    resolves (~1e-4); for large deformation use the Green-Lagrange strain from
    :func:`polar_decomposition` (``U - I``). Returns ``(..., 3, 3)``.
    """
    eye = torch.eye(3, device=F.device, dtype=F.dtype)
    return 0.5 * (F + F.transpose(-1, -2)) - eye


@dataclass
class DeformationField:
    """Voxelised deformation-gradient field for a single grain.

    Attributes
    ----------
    positions : (N, 3) tensor
        Voxel centre positions in micrometers (sample frame).
    F : (N, 3, 3) tensor
        Per-voxel deformation gradient relative to the reference lattice.
    reference_orientation : (3, 3) tensor
        ``OM0`` mapping crystal -> sample for the *reference* (undeformed) lattice.
    lattice_params : (6,) tensor
        ``[a, b, c, alpha, beta, gamma]`` (Angstrom, degrees) of the reference cell.
    shape : tuple[int, int, int] | None
        Optional ``(nx, ny, nz)`` grid shape if the voxels form a regular grid
        (enables image reshaping and finite-difference curvature).
    """

    positions: torch.Tensor
    F: torch.Tensor
    reference_orientation: torch.Tensor
    lattice_params: torch.Tensor
    shape: tuple[int, int, int] | None = None

    @property
    def n_voxels(self) -> int:
        return self.positions.shape[0]

    def reference_G(self, hkl) -> torch.Tensor:
        """Reference reflection's reciprocal vector ``G0`` in the sample frame.

        ``G0 = OM0 @ B @ hkl``. ``hkl`` is a length-3 tuple/tensor. Returns ``(3,)``.
        """
        hkl_t = torch.as_tensor(
            hkl, device=self.F.device, dtype=self.F.dtype
        )
        B = reciprocal_basis(self.lattice_params)
        G_crystal = B @ hkl_t
        return self.reference_orientation @ G_crystal

    def local_Q(self, hkl) -> torch.Tensor:
        """Per-voxel local scattering vector ``Q(r)`` for reflection ``hkl``.

        Returns ``(N, 3)`` in the sample frame. Differentiable in ``F``, the
        reference orientation, and the lattice.
        """
        G0 = self.reference_G(hkl)
        return deform_reflection(self.F, G0)

    def local_rotation_stretch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-voxel ``(R, U)`` from polar decomposition of ``F``. ``(N,3,3)`` each."""
        return polar_decomposition(self.F)

    def to(self, *, device=None, dtype=None) -> "DeformationField":
        """Return a copy on the given device/dtype (tensors moved together)."""
        kw = {}
        if device is not None:
            kw["device"] = device
        if dtype is not None:
            kw["dtype"] = dtype
        return DeformationField(
            positions=self.positions.to(**kw),
            F=self.F.to(**kw),
            reference_orientation=self.reference_orientation.to(**kw),
            lattice_params=self.lattice_params.to(**kw),
            shape=self.shape,
        )
