"""MTEX-friendly export of cake data and pole figures.

MTEX (Bachmann et al., 2010) is the de-facto MATLAB texture-analysis
package. Its native ``XPC`` and ``EPF`` ASCII formats are simple
3-column / grid layouts; we emit them straight from a ``(η, R)`` cake
or a stereographic-projected pole-figure intensity grid.

XPC (cake)::

    # MTEX cake export | n_eta n_R
    # eta_deg R_units intensity
    eta_0 R_0 I[0,0]
    eta_0 R_1 I[0,1]
    ...

EPF (pole figure on (alpha, beta))::

    # MTEX pole figure | hkl=H K L
    # alpha_deg beta_deg intensity
    alpha_0 beta_0 I[0,0]
    ...

Both formats keep the *raw* counts; downstream MTEX scripts apply their
own ODF / E-WIMV normalisation.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def write_mtex_xpc(
    path: str | Path,
    int2d: np.ndarray,
    eta_axis_deg: np.ndarray,
    R_axis: np.ndarray,
    *,
    hkl_rings: Optional[List[Tuple[int, int, int]]] = None,
    R_units: str = "px",
) -> Path:
    """Write a 2-D cake to an MTEX-friendly XPC text file.

    Parameters
    ----------
    int2d :
        ``(n_eta, n_R)`` intensity array.
    eta_axis_deg :
        ``(n_eta,)`` η-axis in degrees.
    R_axis :
        ``(n_R,)`` R axis (pixels, 2θ-degrees, or Q-Å⁻¹).
    hkl_rings :
        Optional list of ``(h, k, l)`` annotated on each ring index.
        Recorded in the header for downstream MTEX scripts.
    R_units :
        ``"px"``, ``"2theta_deg"``, or ``"Q_invA"``. Recorded in the
        header.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    int2d = np.asarray(int2d, dtype=np.float64)
    eta_axis_deg = np.asarray(eta_axis_deg, dtype=np.float64)
    R_axis = np.asarray(R_axis, dtype=np.float64)
    if int2d.shape != (eta_axis_deg.shape[0], R_axis.shape[0]):
        raise ValueError(
            f"int2d shape {int2d.shape} != "
            f"(n_eta={eta_axis_deg.shape[0]}, n_R={R_axis.shape[0]})"
        )
    n_eta, n_r = int2d.shape
    with open(path, "w") as f:
        f.write(f"# MTEX cake export | n_eta {n_eta} n_R {n_r}\n")
        f.write(f"# R_units: {R_units}\n")
        if hkl_rings is not None:
            hkl_str = "; ".join(f"({h},{k},{l})" for (h, k, l) in hkl_rings)
            f.write(f"# rings_hkl: {hkl_str}\n")
        f.write(f"# eta_deg {R_units} intensity\n")
        for i in range(n_eta):
            for j in range(n_r):
                f.write(
                    f"{eta_axis_deg[i]:.5f} {R_axis[j]:.5f} {int2d[i, j]:.6e}\n"
                )
    return path


def write_mtex_epf(
    path: str | Path,
    pole_figure: np.ndarray,
    alpha_grid_deg: np.ndarray,
    beta_grid_deg: np.ndarray,
    *,
    hkl: Tuple[int, int, int],
) -> Path:
    """Write a single-hkl pole figure to MTEX-friendly EPF text format.

    ``pole_figure`` is shape ``(n_alpha, n_beta)`` with the polar angle
    α (declination) and azimuthal β as the two axes. The header records
    the ``hkl`` for downstream MTEX scripts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pole_figure = np.asarray(pole_figure, dtype=np.float64)
    alpha_grid_deg = np.asarray(alpha_grid_deg, dtype=np.float64)
    beta_grid_deg = np.asarray(beta_grid_deg, dtype=np.float64)
    if pole_figure.shape != (alpha_grid_deg.shape[0], beta_grid_deg.shape[0]):
        raise ValueError(
            f"pole_figure shape {pole_figure.shape} != "
            f"(n_alpha={alpha_grid_deg.shape[0]}, "
            f"n_beta={beta_grid_deg.shape[0]})"
        )
    with open(path, "w") as f:
        f.write(f"# MTEX pole figure | hkl={hkl[0]} {hkl[1]} {hkl[2]}\n")
        f.write("# alpha_deg beta_deg intensity\n")
        for i, alpha in enumerate(alpha_grid_deg):
            for j, beta in enumerate(beta_grid_deg):
                f.write(f"{alpha:.5f} {beta:.5f} {pole_figure[i, j]:.6e}\n")
    return path


__all__ = ["write_mtex_xpc", "write_mtex_epf"]
