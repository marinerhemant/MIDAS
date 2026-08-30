"""Pole figure construction from cake (η, R) data.

Given a 2D integrated cake at one sample orientation, the visible η
range carries one stripe of a pole figure for each ring. To build the
full pole figure the user must rotate the sample (χ, φ) and stack
slices; here we provide the mapping for one slice.

For a Bragg ring at fixed 2θ on a flat detector, the η-coordinate of each pixel
along the ring maps to the sample-frame azimuth β = η + φ, at the fixed
declination **α = 90° − θ** set by the ring's Bragg angle alone. We emit
``(α, β, intensity)`` triples on a regular stereographic grid.

Note α is NOT set by the sample tilt — that was the pre-2026-08-29 behaviour and
it put every ring at the centre of the figure; see the warning on
:func:`cake_to_pole_figure`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def cake_to_pole_figure(
    int2d: np.ndarray,
    eta_axis_deg: np.ndarray,
    R_axis: np.ndarray,
    *,
    hkl_R_px: float,
    two_theta_deg: Optional[float] = None,
    capture_radius_px: float = 3.0,
    sample_rotation_chi_deg: float = 0.0,
    sample_rotation_phi_deg: float = 0.0,
    output_grid: Tuple[int, int] = (181, 91),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project one ring's η stripe onto a stereographic pole-figure grid.

    The pole declination is set by the ring's BRAGG ANGLE, derived here rather
    than recalled. With incident beam ``k_i = (0, 0, 1)`` and diffracted
    ``k_f = (sin2θ cosη, sin2θ sinη, cos2θ)``::

        Q ∝ k_f − k_i = 2 sinθ · (cosθ cosη, cosθ sinη, −sinθ)

    so the unit pole direction is ``(cosθ cosη, cosθ sinη, −sinθ)``, whose angle
    from the back-along-beam axis has cosine ``sinθ``. Therefore::

        α (declination) = 90° − θ        β (azimuth) = η + φ

    Every pole from ONE ring sits at a CONSTANT declination — but at
    ``90° − θ``, near the rim for a typical powder ring, not at the centre.

    .. warning::

       **Before 2026-08-29 this function ignored the Bragg angle entirely.** It
       set ``α = sample_rotation_chi_deg % 90`` and used ``hkl_R_px`` only to
       select the η stripe, so with the default ``chi = 0`` every ring — at any
       2θ — was dumped into the α = 0 bin at the centre of the pole figure.
       Measured error, four rings at a 1000 px sample-detector distance:

       ============  ==============  ================  =========
       ring R (px)   correct α       α produced        error
       ============  ==============  ================  =========
       25            89.284°         0.000°            −89.284°
       50            88.569°         0.000°            −88.569°
       100           87.145°         0.000°            −87.145°
       175           85.037°         0.000°            −85.037°
       ============  ==============  ================  =========

       The ``% 90`` also silently wrapped ``chi = 90°`` to 0. Output went
       straight into :func:`write_popla_pol`, i.e. a real POPLA ``.pol`` file
       that texture software would read as a genuine pole figure.

    Parameters
    ----------
    int2d :
        ``(n_eta, n_R)`` cake intensity.
    eta_axis_deg :
        ``(n_eta,)`` η axis (degrees).
    R_axis :
        ``(n_R,)`` R axis.
    hkl_R_px :
        Radius of the ring of interest in same units as R_axis. Selects the
        η stripe; it does NOT set the declination (R units are arbitrary here,
        so 2θ cannot be recovered from it without the geometry).
    two_theta_deg :
        Scattering angle 2θ of this ring, degrees. **Required** — the pole
        declination is ``90° − θ`` and there is no way to get θ from
        ``hkl_R_px`` alone. Omitting it raises rather than silently producing
        the old, wrong, centre-of-the-figure result.
    capture_radius_px :
        ±window around the ring (R-axis units).
    sample_rotation_phi_deg :
        Rotation about the beam axis, degrees. This is an unambiguous
        azimuthal offset: it simply adds to β.
    sample_rotation_chi_deg :
        Sample tilt, degrees. **Not implemented** — a tilt rotates the pole
        cone out of the beam-axis frame, and the result depends on which
        stage axis χ turns about. Adding χ to α is only a first-order
        approximation valid at one azimuth, not around the ring, so a nonzero
        value raises instead of guessing a stage convention.
    output_grid :
        ``(n_alpha, n_beta)`` grid resolution for the output.

    Returns
    -------
    alpha_grid_deg, beta_grid_deg, intensity :
        - ``alpha_grid_deg`` shape ``(n_alpha,)``, range [0, 90).
        - ``beta_grid_deg`` shape ``(n_beta,)``, range [0, 360).
        - ``intensity`` shape ``(n_beta, n_alpha)`` (β rows × α cols).
    """
    int2d = np.asarray(int2d, dtype=np.float64)
    eta = np.asarray(eta_axis_deg, dtype=np.float64)
    R = np.asarray(R_axis, dtype=np.float64)
    if int2d.shape != (eta.shape[0], R.shape[0]):
        raise ValueError(
            f"int2d shape {int2d.shape} != "
            f"(n_eta={eta.shape[0]}, n_R={R.shape[0]})"
        )
    n_alpha, n_beta = output_grid
    # Capture stripe at the ring
    in_ring = np.abs(R - hkl_R_px) <= capture_radius_px
    if not in_ring.any():
        raise ValueError(
            f"no R bins within {capture_radius_px} of ring at R={hkl_R_px}"
        )
    stripe = int2d[:, in_ring].sum(axis=1)              # (n_eta,)

    if two_theta_deg is None:
        raise ValueError(
            "cake_to_pole_figure needs two_theta_deg: the pole declination is "
            "90 - theta, and theta cannot be recovered from hkl_R_px alone "
            "(R_axis units are arbitrary here). Before 2026-08-29 this "
            "argument did not exist and the declination was taken from "
            "sample_rotation_chi_deg instead, which put every ring at the "
            "CENTRE of the pole figure regardless of its 2theta - up to 89 deg "
            "wrong. Pass the ring's 2theta."
        )
    if float(sample_rotation_chi_deg) != 0.0:
        raise NotImplementedError(
            "sample_rotation_chi_deg != 0 is not supported. A sample tilt "
            "rotates the pole cone out of the beam-axis frame, and the result "
            "depends on which stage axis chi turns about; adding chi to alpha "
            "(what this function used to do) is a first-order approximation "
            "valid at one azimuth only, not around the ring. Specify the stage "
            "convention and this can be implemented properly."
        )

    # alpha = 90 - theta, constant around the ring (see the derivation in the
    # docstring). beta = eta + phi.
    theta_deg = 0.5 * float(two_theta_deg)
    alpha_value = 90.0 - theta_deg
    if not (0.0 <= alpha_value < 90.0):
        raise ValueError(
            f"two_theta_deg={two_theta_deg!r} gives a declination "
            f"{alpha_value:.3f} deg outside [0, 90); 2theta must be in "
            f"(0, 180]."
        )
    alpha_grid_deg = np.linspace(0.0, 90.0, n_alpha, endpoint=False)
    beta_grid_deg = np.linspace(0.0, 360.0, n_beta, endpoint=False)
    # Resample η stripe onto β grid via linear interpolation (wrap)
    eta_unwrapped = (eta + sample_rotation_phi_deg) % 360.0
    sort_idx = np.argsort(eta_unwrapped)
    beta_intensity = np.interp(
        beta_grid_deg, eta_unwrapped[sort_idx], stripe[sort_idx],
        period=360.0,
    )
    # Place the stripe at the alpha bin closest to alpha_value
    alpha_idx = int(np.argmin(np.abs(alpha_grid_deg - alpha_value)))
    intensity = np.zeros((n_beta, n_alpha), dtype=np.float64)
    intensity[:, alpha_idx] = beta_intensity
    return alpha_grid_deg, beta_grid_deg, intensity


def write_popla_pol(
    path: str | Path,
    alpha_grid_deg: np.ndarray,
    beta_grid_deg: np.ndarray,
    intensity: np.ndarray,
    *,
    hkl: Tuple[int, int, int],
) -> Path:
    """Write a POPLA-format pole-figure file.

    POPLA expects a fixed-format ASCII grid: 4-space-separated columns
    of intensity, with an HKL header line. We use a simplified form
    that POPLA's reader accepts: header + one float per α-β bin.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    intensity = np.asarray(intensity, dtype=np.float64)
    if intensity.shape != (beta_grid_deg.shape[0], alpha_grid_deg.shape[0]):
        raise ValueError(
            f"intensity shape {intensity.shape} != "
            f"(n_beta={beta_grid_deg.shape[0]}, "
            f"n_alpha={alpha_grid_deg.shape[0]})"
        )
    with open(path, "w") as f:
        f.write(f"# POPLA pole figure | hkl={hkl[0]} {hkl[1]} {hkl[2]}\n")
        f.write(
            f"# alpha_step={alpha_grid_deg[1] - alpha_grid_deg[0]:.4f} "
            f"beta_step={beta_grid_deg[1] - beta_grid_deg[0]:.4f}\n"
        )
        f.write(f"# n_alpha={alpha_grid_deg.shape[0]} "
                f"n_beta={beta_grid_deg.shape[0]}\n")
        for i in range(intensity.shape[0]):
            row = " ".join(f"{v:.6e}" for v in intensity[i])
            f.write(row + "\n")
    return path


__all__ = ["cake_to_pole_figure", "write_popla_pol"]
