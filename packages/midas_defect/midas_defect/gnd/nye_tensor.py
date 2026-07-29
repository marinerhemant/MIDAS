"""Per-grain Nye tensor via least-squares orientation-gradient fit.

For each grain :math:`g` with :math:`k_{NN}` nearest neighbours, fit the
linear orientation gradient

    Omega(x) = Omega_0 + L (x - x_g)

where :math:`Omega = (R - I) - 0.5 (R - I)^T` is the antisymmetric
small-rotation tensor extracted from the relative rotation between grain
:math:`g` and a neighbour, and :math:`L` is a (3, 3, 3) third-order tensor
of gradients :math:`L_{ijk} = \\partial Omega_{ij} / \\partial x_k`. The
Kröner (1959) Nye tensor is then

    alpha_{ij} = epsilon_{jkl} L_{ilk}                (eq. 3.34, Nye 1953)

and the scalar GND density follows from the Frobenius norm

    rho_GND = || alpha ||_F / b

with :math:`b` the Burgers magnitude (m).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _relative_small_rotation(R_a: NDArray[np.floating], R_b: NDArray[np.floating]) -> NDArray[np.floating]:
    """Antisymmetric small-rotation tensor Omega such that R_b ~= R_a (I + Omega)."""
    Rrel = R_b @ R_a.T
    # Skew part: Omega = 0.5 (Rrel - Rrel^T)
    return 0.5 * (Rrel - Rrel.T)


def _omega_to_vec(Omega: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert antisymmetric Omega to its 3-vector dual omega s.t. Omega v = omega x v."""
    return np.array([Omega[2, 1], Omega[0, 2], Omega[1, 0]])


def per_grain_nye_tensor(
    OM: NDArray[np.floating],
    pos: NDArray[np.floating],
    burgers_length: float,
    variant_labels: NDArray[np.intp] | None = None,
    k_NN: int = 8,
    pos_to_metres: float = 1.0e-6,
    z_reliable: bool = True,
) -> dict:
    """Per-grain Nye tensor and scalar GND density.

    Parameters
    ----------
    OM, pos : (n_grains, 3, 3) and (n_grains, 3)
        Orientation matrices and grain positions.
    burgers_length : m
    variant_labels : optional; restricts NN search to same-variant grains.
    k_NN : number of nearest in-variant neighbours per grain.
    pos_to_metres : conversion factor from ``pos`` units to metres
        (default 1e-6 = pos in um).
    z_reliable : if False (REQUIRED for FF-HEDM, where refined Z is +/-210 um;
        AUDIT_2026-06-23.md), the z-component of the neighbour offsets is zeroed so the
        noisy out-of-layer gradient is not injected into the Nye tensor. The result is
        an in-plane (layer-z) GND estimate (z-gradient components are 0). Pass the layer
        index (not refined Z) as ``pos[:,2]`` and keep ``z_reliable=False`` for FF data.

    Returns
    -------
    dict with
        alpha_per_grain  (n_grains, 3, 3)  Nye tensor (m^-1); NaN if insufficient NN
        rho_GND_per_grain (n_grains,)      m^-2;  || alpha ||_F / b
    """
    from scipy.spatial import cKDTree

    OM = np.asarray(OM, dtype=float)
    pos = np.asarray(pos, dtype=float)
    n = OM.shape[0]
    if pos.shape[0] != n:
        raise ValueError(f"length mismatch: OM {n} vs pos {pos.shape[0]}")

    var = (
        np.zeros(n, dtype=int) if variant_labels is None else np.asarray(variant_labels, dtype=int)
    )

    alpha = np.full((n, 3, 3), np.nan, dtype=float)

    for lbl in sorted(set(int(x) for x in np.unique(var))):
        idx = np.where(var == lbl)[0]
        if idx.size < 5:  # need at least k_NN >= 4 NN for the 3D LS fit; require 4 NN + self
            continue
        # neighbour search must also ignore unreliable z, else z-noise changes WHICH
        # neighbours are picked (not just the gradient).
        psp = pos[idx][:, :2] if not z_reliable else pos[idx]
        tree = cKDTree(psp)
        k_eff = min(k_NN + 1, idx.size)
        _, nn = tree.query(psp, k=k_eff)
        if k_eff < 5:
            continue
        nn = nn[:, 1:]  # drop self

        for li, g in enumerate(idx):
            nbrs = idx[nn[li]]
            # Build (k, 3) position differences in metres, and (k, 3) omega-vectors
            # (axial dual of the small relative rotation Omega).
            dx = (pos[nbrs] - pos[g]) * pos_to_metres
            if not z_reliable:
                dx[:, 2] = 0.0      # drop unreliable out-of-layer (refined-Z) gradient
            omega_vecs = np.stack(
                [_omega_to_vec(_relative_small_rotation(OM[g], OM[t])) for t in nbrs],
                axis=0,
            )
            # Least-squares fit: omega_i = G_ij dx_j  (3, 3) gradient tensor.
            G, *_ = np.linalg.lstsq(dx, omega_vecs, rcond=None)  # shape (3, 3) -- G[j, i] is d omega_i / d x_j
            # Convert axial-vector gradient to Nye tensor:
            #   alpha_ij = - d omega_i / d x_j  (Kroner 1959, sign depending on convention)
            # We use the convention alpha = - G^T so that pure tilt about z gives
            # diagonal alpha_zz > 0.
            alpha[g] = -G.T

    rho = np.linalg.norm(alpha, axis=(1, 2)) / burgers_length
    return {
        "alpha_per_grain": alpha,
        "rho_GND_per_grain": rho,
    }


__all__ = ["per_grain_nye_tensor"]
