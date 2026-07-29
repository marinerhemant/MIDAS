"""Independent validation oracle for the DFXM forward.

Item #2 of the post-Phase-5 roadmap — the credibility anchor.

The single biggest risk for a synthetic-first package is that "the inverse recovers
its own forward" — an implementation bug in the forward would be invisible because
both directions share it. To guard against that, this module reimplements the core
DFXM forward observable in **plain numpy**, sharing no code with the torch path
(``forward.py`` / ``resolution.py`` / ``field.py``). Bit-level agreement between the
two independent implementations rules out implementation error in the error-prone
deform + frame + resolution math. This mirrors ``midas_2d``'s ``debye_reference_numpy``.

Scope note (honest): comparison against the **published DTU toolboxes** (`darling`,
`dfxm-geo`) is a separate, stronger check that requires installing external code
(not available offline here); it is a real-data-era deliverable. The numpy oracle
below is the self-contained check we can run now — it validates the *code*, while the
analytic-limit tests (rocking FWHM, forbidden reflections, g.b extinction) validate
the *physics*.
"""
from __future__ import annotations

import numpy as np
import torch

from .conventions import GoniometerSetting
from .field import DeformationField
from .resolution import ResolutionFunction


def _orthonormal_frame_np(q_nom: np.ndarray):
    """Independent numpy build of the (e_par, e_t1, e_t2) frame aligned to q_nom."""
    e_par = q_nom / np.linalg.norm(q_nom)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(e_par @ seed) > 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    e_t1 = seed - (seed @ e_par) * e_par
    e_t1 = e_t1 / np.linalg.norm(e_t1)
    e_t2 = np.cross(e_par, e_t1)
    return e_par, e_t1, e_t2


def voxel_intensity_numpy(
    F: np.ndarray,
    G0: np.ndarray,
    G_gonio: np.ndarray,
    q_nom: np.ndarray,
    *,
    sigma_par: float,
    sigma_perp: float,
    sf2: float = 1.0,
) -> np.ndarray:
    """Independent numpy reimplementation of :func:`midas_dfxm.forward.voxel_intensity`.

    ``F`` is ``(N, 3, 3)``, ``G0``/``q_nom`` are ``(3,)``, ``G_gonio`` the ``(3, 3)``
    sample->lab rotation. Computes, per voxel: ``Q = F^-T G0`` (solve, not inverse),
    ``Q_lab = G_gonio @ Q``, then the Gaussian acceptance centred on ``q_nom``. Returns
    ``(N,)``. Pure numpy — no torch, no shared helpers.
    """
    N = F.shape[0]
    # Q = F^-T G0  <=>  F^T Q = G0
    Q = np.stack([np.linalg.solve(F[i].T, G0) for i in range(N)], axis=0)  # (N, 3)
    Q_lab = Q @ G_gonio.T                                                  # rows: G@Q
    e_par, e_t1, e_t2 = _orthonormal_frame_np(q_nom)
    d = Q_lab - q_nom
    d_par = d @ e_par
    d_t1 = d @ e_t1
    d_t2 = d @ e_t2
    chi2 = (d_par / sigma_par) ** 2 + (d_t1 / sigma_perp) ** 2 + (d_t2 / sigma_perp) ** 2
    return sf2 * np.exp(-0.5 * chi2)


def cross_check_voxel_intensity(
    field: DeformationField,
    hkl,
    goniometer: GoniometerSetting,
    resolution: ResolutionFunction,
) -> dict:
    """Run both the torch forward and the numpy oracle; return max abs discrepancy.

    Returns ``{'torch', 'numpy', 'max_abs_diff'}``. A tiny ``max_abs_diff`` (~1e-12)
    certifies the torch deform+resolution implementation against an independent path.
    """
    from .forward import voxel_intensity  # local import to keep this module light

    torch_val = voxel_intensity(field, hkl, goniometer, resolution).detach()

    G0 = field.reference_G(hkl).detach().cpu().numpy()
    G_gonio = goniometer.sample_rotation(dtype=field.F.dtype).detach().cpu().numpy()
    q_nom = resolution.q_nom.detach().cpu().numpy()
    F = field.F.detach().cpu().numpy()
    np_val = voxel_intensity_numpy(
        F, G0, G_gonio, q_nom,
        sigma_par=float(resolution.sigma_par), sigma_perp=float(resolution.sigma_perp),
    )
    diff = float(np.max(np.abs(torch_val.cpu().numpy() - np_val)))
    return {"torch": torch_val, "numpy": np_val, "max_abs_diff": diff}
