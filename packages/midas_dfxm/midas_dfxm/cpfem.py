"""Crystal-plasticity FEM <-> DFXM interchange and validation harness.

Connects an external crystal-plasticity solver (JAX-CPFEM / ``jax_fem``, or any code
that outputs a per-voxel deformation-gradient field) to the DFXM forward+inverse
stack. Two levels of coupling:

1. **Black-box (this module, runs anywhere).** The CP solver produces ``F(r)`` on a
   voxel grid; :func:`load_cpfem_field` wraps it as a :class:`DeformationField`, and
   :func:`validate_dfxm_on_cpfem` runs the DFXM forward then the strain inverse and
   compares the recovered strain to the CP ground truth. This validates the whole
   DFXM stack on a *physically realistic* field — a much stronger anchor than
   analytic slip fields, and available before any beamline data.

2. **End-to-end differentiable (see ``examples/jax_dfxm_bridge.py``).** For gradient
   flow from CP parameters (CRSS, hardening) through the CPFEM *and* the DFXM forward
   in one graph, the DFXM observable is reimplemented in JAX (parity-checked against
   this torch forward) so it composes with ``jax_fem`` on the GPU host. That is the
   flagship coupling; this module is the framework-agnostic half.

I/O uses plain ``.npz`` (numpy) so no framework is imposed on the interchange.
"""
from __future__ import annotations

import numpy as np
import torch

from .field import DeformationField, small_strain_from_F
from .generators import field_from_deformation_gradient
from .inverse import normal_strain, recover_strain_direct, strain_design_matrix


def save_cpfem_field(path: str, F: np.ndarray, positions: np.ndarray, *,
                     lattice_params=(3.6356, 3.6356, 3.6356, 90.0, 90.0, 90.0),
                     shape=None) -> None:
    """Write a CPFEM field to ``.npz`` (the interchange format).

    ``F`` is ``(N, 3, 3)``, ``positions`` ``(N, 3)`` (micrometers). ``shape`` is the
    optional ``(nx, ny, nz)`` grid. This is the format a ``jax_fem`` export writes and
    :func:`load_cpfem_field` reads.
    """
    np.savez(path, F=np.asarray(F), positions=np.asarray(positions),
             lattice_params=np.asarray(lattice_params),
             shape=np.asarray(shape if shape is not None else [-1]))


def load_cpfem_field(path_or_arrays, *, device=None, dtype=torch.float64,
                     orientation=None) -> DeformationField:
    """Load a CPFEM ``F(r)`` field (``.npz`` path or ``(F, positions[, latc, shape])``).

    Returns a :class:`DeformationField` ready for the DFXM forward. The canonical
    entry point for JAX-CPFEM / external / any deformation-gradient field.
    """
    if isinstance(path_or_arrays, str):
        d = np.load(path_or_arrays, allow_pickle=False)
        F, positions = d["F"], d["positions"]
        latc = d["lattice_params"] if "lattice_params" in d else (3.6356,) * 3 + (90.0,) * 3
        sh = d["shape"] if "shape" in d else None
        shape = tuple(int(x) for x in sh) if sh is not None and int(sh[0]) > 0 else None
    else:
        F, positions, *rest = path_or_arrays
        latc = rest[0] if rest else (3.6356,) * 3 + (90.0,) * 3
        shape = rest[1] if len(rest) > 1 else None
    Ft = torch.as_tensor(np.asarray(F), device=device, dtype=dtype)
    pos = torch.as_tensor(np.asarray(positions), device=device, dtype=dtype)
    return field_from_deformation_gradient(
        Ft, pos, orientation=orientation,
        lattice_params=tuple(float(x) for x in np.asarray(latc).ravel()[:6]), shape=shape)


def cpfem_true_strain(field: DeformationField) -> torch.Tensor:
    """Ground-truth per-voxel strain (Voigt-6) from the CP deformation gradient.

    ``eps = sym(F) - I`` -> ``[e11, e22, e33, e23, e13, e12]``. The reference the DFXM
    strain inverse is scored against.
    """
    eps = small_strain_from_F(field.F)  # (N,3,3)
    return torch.stack([eps[:, 0, 0], eps[:, 1, 1], eps[:, 2, 2],
                        eps[:, 1, 2], eps[:, 0, 2], eps[:, 0, 1]], dim=-1)


def validate_dfxm_on_cpfem(
    field: DeformationField,
    reflections,
    *,
    noise_std: float = 0.0,
    seed: int = 0,
) -> dict:
    """Forward-model DFXM normal strains from a CP field, then invert and score.

    Simulates the per-voxel normal strain in each reflection (optionally with noise),
    recovers the full strain tensor (:func:`recover_strain_direct`), and compares to
    the CP ground truth :func:`cpfem_true_strain`. Returns
    ``{'recovered', 'truth', 'rms_error', 'max_error'}``. The end-to-end check that the
    DFXM inverse reproduces a realistic crystal-plasticity strain field.
    """
    truth = cpfem_true_strain(field)                       # (N, 6)
    meas = torch.stack([normal_strain(field, hkl) for hkl in reflections], dim=0)  # (K, N)
    if noise_std > 0:
        g = torch.Generator(device=meas.device).manual_seed(int(seed))
        meas = meas + noise_std * torch.randn(meas.shape, generator=g,
                                              device=meas.device, dtype=meas.dtype)
    recovered = recover_strain_direct(meas, reflections)   # (N, 6)
    err = recovered - truth
    return {
        "recovered": recovered,
        "truth": truth,
        "rms_error": float((err ** 2).mean().sqrt()),
        "max_error": float(err.abs().max()),
    }
