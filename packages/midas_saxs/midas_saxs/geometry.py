"""Transmission-SAXS detector geometry: pixel -> q.

Reuses the canonical MIDAS detector model -- tilt ``R_z R_y R_x`` plus the
15-coefficient radial distortion -- through
:func:`midas_transforms.fit_setup.transform.apply_tilt_distortion`. That matters
beyond tidiness: a SAXS geometry and an FF-HEDM geometry calibrated from the
same detector must agree about where q sits, and they only do if both go through
one implementation of the distortion. Nothing here re-ports it.

The one genuine difference from the HEDM case is that transmission SAXS has **no
omega rotation** -- the sample does not turn, so lab q *is* sample q. That is why
this is a separate small module rather than a call into the HEDM path with
``omega=0``: there is no rotation to pass.

Units -- read this before wiring anything up
--------------------------------------------
* ``q`` from this module is in **inverse angstroms**, the SAXS convention, and
  matches ``midas_defect.geometry.pixel_to_qlab``.
* :mod:`midas_ddd`'s Fourier kernel wants **inverse micrometers**, because its
  node positions are micrometers.

The factor between them is 1e4 and it is easy to lose. :func:`inv_A_to_inv_um`
and :func:`inv_um_to_inv_A` exist so the conversion is always a named call, and
:mod:`midas_saxs.strain_source` uses them rather than a bare literal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

__all__ = [
    "SAXSGeometry",
    "inv_A_to_inv_um",
    "inv_um_to_inv_A",
    "pixel_to_q",
    "q_magnitude_grid",
    "two_theta_to_q",
]

#: 1 inverse angstrom = 1e4 inverse micrometers.
_INV_A_PER_INV_UM = 1.0e-4
_UM_PER_A = 1.0e-4


def inv_A_to_inv_um(q_inv_A):
    """q in 1/A -> q in 1/um (multiply by 1e4)."""
    return q_inv_A * 1.0e4


def inv_um_to_inv_A(q_inv_um):
    """q in 1/um -> q in 1/A (multiply by 1e-4)."""
    return q_inv_um * 1.0e-4


@dataclass
class SAXSGeometry:
    """Flat-panel transmission-SAXS geometry.

    All distances micrometers, angles degrees, wavelength angstroms -- the MIDAS
    convention throughout.

    Attributes
    ----------
    lsd_um
        Sample-to-detector distance.
    bcy_px, bcz_px
        Direct-beam position on the detector: ``bcy`` is the column (horizontal),
        ``bcz`` the row (vertical). For SAXS this is the **beam centre proper**,
        i.e. where the undiffracted beam would land, usually behind the beamstop.
    px_um
        Pixel pitch.
    wavelength_A
        X-ray wavelength.
    n_pix_y, n_pix_z
        Detector extent in pixels (columns, rows).
    tx_deg, ty_deg, tz_deg
        Detector tilts about lab X, Y, Z.
    p_coeffs, rho_d_um
        MIDAS 15-coefficient distortion model and its normalisation radius.
        All-zero ``p_coeffs`` means no distortion.
    beamstop_radius_px
        Pixels within this radius of the beam centre are masked. A SAXS image
        without a beamstop mask is not a SAXS image; the direct beam is many
        orders of magnitude above the signal.
    """

    lsd_um: float
    bcy_px: float
    bcz_px: float
    px_um: float
    wavelength_A: float
    n_pix_y: int
    n_pix_z: int
    tx_deg: float = 0.0
    ty_deg: float = 0.0
    tz_deg: float = 0.0
    p_coeffs: Sequence[float] = (0.0,) * 15
    rho_d_um: float = 200000.0
    beamstop_radius_px: float = 0.0
    label: str = ""

    # --------------------------------------------------------------- helpers

    @property
    def photon_energy_keV(self) -> float:
        return 12.398419843320026 / self.wavelength_A

    def q_at_pixel_radius(self, r_px: float) -> float:
        """|q| (1/A) at a radius of ``r_px`` pixels from the beam centre, no tilt.

        The quick sanity check on whether a geometry covers the q range you
        think it does.
        """
        two_theta = math.atan2(r_px * self.px_um, self.lsd_um)
        return 4.0 * math.pi * math.sin(0.5 * two_theta) / self.wavelength_A

    @property
    def q_range_inv_A(self) -> Tuple[float, float]:
        """``(q_min, q_max)`` spanned by the panel, from the beamstop to the
        farthest corner."""
        q_min = self.q_at_pixel_radius(max(self.beamstop_radius_px, 1.0))
        corners = [
            (0.0, 0.0), (self.n_pix_y, 0.0),
            (0.0, self.n_pix_z), (self.n_pix_y, self.n_pix_z),
        ]
        r_max = max(math.hypot(cy - self.bcy_px, cz - self.bcz_px) for cy, cz in corners)
        return q_min, self.q_at_pixel_radius(r_max)

    def _tensors(self, dtype, device):
        t = lambda v: torch.as_tensor(v, dtype=dtype, device=device)
        return dict(
            lsd=t(self.lsd_um), bcy=t(self.bcy_px), bcz=t(self.bcz_px),
            px=t(self.px_um), lamb=t(self.wavelength_A),
            tx=t(self.tx_deg), ty=t(self.ty_deg), tz=t(self.tz_deg),
            p=torch.as_tensor(np.asarray(self.p_coeffs, dtype=float),
                              dtype=dtype, device=device),
            rho=t(self.rho_d_um),
        )

    # -------------------------------------------------------------- the maps

    def pixel_grid(self, *, dtype=torch.float64, device=None):
        """``(rows, cols)`` index grids covering the whole panel, each ``(Z, Y)``."""
        rows = torch.arange(self.n_pix_z, dtype=dtype, device=device)
        cols = torch.arange(self.n_pix_y, dtype=dtype, device=device)
        return torch.meshgrid(rows, cols, indexing="ij")

    def beamstop_mask(self, *, dtype=torch.float64, device=None) -> torch.Tensor:
        """``True`` where a pixel is USABLE (outside the beamstop). ``(Z, Y)`` bool."""
        rows, cols = self.pixel_grid(dtype=dtype, device=device)
        r = torch.hypot(cols - self.bcy_px, rows - self.bcz_px)
        return r > self.beamstop_radius_px


def pixel_to_q(
    rows: Union[torch.Tensor, np.ndarray, Sequence[float]],
    cols: Union[torch.Tensor, np.ndarray, Sequence[float]],
    geom: SAXSGeometry,
    *,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    """Detector pixel ``(row, col)`` -> lab-frame q, **inverse angstroms**.

    ``q = (2 pi / lambda) (k_f - k_i)`` with ``k_i = +X``. Returns ``(..., 3)``.

    Differentiable through every geometry scalar (Lsd, beam centre, pixel pitch,
    wavelength, tilts); the pixel indices themselves are not differentiable.

    There is no omega: transmission SAXS does not rotate the sample, so this is
    both the lab-frame and the sample-frame q.
    """
    from midas_transforms.fit_setup.transform import apply_tilt_distortion

    rows_t = torch.as_tensor(rows, dtype=dtype, device=device)      # Z_pix
    cols_t = torch.as_tensor(cols, dtype=dtype, device=device)      # Y_pix
    g = geom._tensors(dtype, device)

    Yl, Zl = apply_tilt_distortion(
        cols_t, rows_t, Lsd=g["lsd"], BC_y=g["bcy"], BC_z=g["bcz"],
        tx=g["tx"], ty=g["ty"], tz=g["tz"], p_coeffs=g["p"], px=g["px"],
        rho_d=g["rho"],
    )
    x_um = g["lsd"].expand_as(Yl) if Yl.ndim > 0 else g["lsd"]
    p_lab = torch.stack([x_um, Yl, Zl], dim=-1)

    k_f = p_lab / torch.linalg.vector_norm(p_lab, dim=-1, keepdim=True)
    k_i = torch.zeros_like(k_f)
    k_i[..., 0] = 1.0
    k0 = 2.0 * math.pi / g["lamb"]
    return k0 * (k_f - k_i)


def q_magnitude_grid(geom: SAXSGeometry, *, dtype=torch.float64, device=None):
    """``|q|`` (1/A) for every pixel on the panel. ``(Z, Y)``."""
    rows, cols = geom.pixel_grid(dtype=dtype, device=device)
    q = pixel_to_q(rows, cols, geom, dtype=dtype, device=device)
    return torch.linalg.vector_norm(q, dim=-1)


def two_theta_to_q(two_theta_deg, wavelength_A: float):
    """``q = 4 pi sin(theta) / lambda``, inverse angstroms."""
    tt = torch.as_tensor(two_theta_deg, dtype=torch.float64)
    return 4.0 * math.pi * torch.sin(0.5 * torch.deg2rad(tt)) / wavelength_A
