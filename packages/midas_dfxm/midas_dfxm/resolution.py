"""Reciprocal-space resolution function of the DFXM microscope.

Phase 1 of ``implementation_plan.md`` — the NEW instrument core.

The objective (CRL / MLL) plus incident-beam conditioning act as a bandpass in
reciprocal space: a sample voxel with local scattering vector ``Q(r)`` (sample
frame) contributes intensity to the image only insofar as its lab-frame vector
``Q_lab = G(s) @ Q(r)`` (for goniometer setting ``s``) overlaps the instrument's
acceptance, which is centred on the nominal accepted vector ``q_nom``.

We model the acceptance as an anisotropic Gaussian aligned to ``q_nom``:

    Res(Q_lab) = exp(-1/2 [ (d_par/sigma_par)^2 + (d_t1/sigma_t)^2 + (d_t2/sigma_t)^2 ])

with ``d = Q_lab - q_nom`` decomposed into a **longitudinal** component
``d_par`` along ``q_nom`` (probed by an energy / 2-theta *strain scan*, width set
by the incident energy bandwidth) and two **transverse** components (probed by a
*mosaicity scan* rocking the sample, width set by beam divergence + objective NA).

This is the geometrical-optics (kinematic) resolution model. It is differentiable
in the widths and in ``q_nom`` (hence in the reference orientation / lattice /
goniometer), and device/dtype-portable. A wave-optics acceptance is a Phase-5+
contingency (``waveoptics.py``).

Units: reciprocal vectors in 1/Angstrom (times ``2*pi``); widths ``sigma_*`` in the
same units.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def _orthonormal_frame(q_nom: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-handed frame ``(e_par, e_t1, e_t2)`` with ``e_par`` along ``q_nom``."""
    e_par = q_nom / torch.linalg.vector_norm(q_nom)
    # Pick a seed not parallel to e_par.
    seed = torch.tensor([0.0, 0.0, 1.0], device=q_nom.device, dtype=q_nom.dtype)
    if torch.abs(torch.dot(e_par, seed)) > 0.9:
        seed = torch.tensor([1.0, 0.0, 0.0], device=q_nom.device, dtype=q_nom.dtype)
    e_t1 = seed - torch.dot(seed, e_par) * e_par
    e_t1 = e_t1 / torch.linalg.vector_norm(e_t1)
    e_t2 = torch.linalg.cross(e_par, e_t1)
    return e_par, e_t1, e_t2


@dataclass
class ResolutionFunction:
    """Gaussian reciprocal-space acceptance centred on ``q_nom`` (lab frame).

    Attributes
    ----------
    q_nom : (3,) tensor
        Nominal accepted scattering vector (lab frame), i.e. the reference
        reflection placed at the centre of the acceptance.
    sigma_par : float | tensor
        Longitudinal (radial, along ``q_nom``) width [1/Angstrom] — energy
        bandwidth. Probed by the strain scan.
    sigma_perp : float | tensor
        Transverse width [1/Angstrom] — divergence + objective NA. Probed by the
        mosaicity scan.
    """

    q_nom: torch.Tensor
    sigma_par: float = 5e-3
    sigma_perp: float = 5e-3
    sigma_perp2: "float | None" = None   # roll (2nd transverse) width; None -> isotropic transverse (== sigma_perp)
    rock_dir: "torch.Tensor | None" = None  # in-scattering-plane direction orienting e_t1 (rock); None -> arbitrary transverse frame

    def weight(self, Q_lab: torch.Tensor) -> torch.Tensor:
        """Acceptance weight in ``[0, 1]`` for lab-frame vectors ``Q_lab`` ``(..., 3)``.

        Differentiable in ``Q_lab``, ``q_nom``, and the widths. Anisotropic transverse
        when ``sigma_perp2`` is set (``sigma_perp`` = rock/``e_t1``, ``sigma_perp2`` =
        roll/``e_t2``); if ``rock_dir`` is given, ``e_t1`` is aligned to its component
        perpendicular to ``q_nom`` (the scattering-plane / rock direction).
        """
        if self.rock_dir is not None:
            e_par = self.q_nom / torch.linalg.vector_norm(self.q_nom)
            rd = torch.as_tensor(self.rock_dir, device=Q_lab.device, dtype=Q_lab.dtype)
            e_t1 = rd - (rd @ e_par) * e_par
            e_t1 = e_t1 / torch.linalg.vector_norm(e_t1)
            e_t2 = torch.linalg.cross(e_par, e_t1)
        else:
            e_par, e_t1, e_t2 = _orthonormal_frame(self.q_nom)
        d = Q_lab - self.q_nom
        d_par = d @ e_par
        d_t1 = d @ e_t1
        d_t2 = d @ e_t2
        s_par = torch.as_tensor(self.sigma_par, device=Q_lab.device, dtype=Q_lab.dtype)
        s_perp = torch.as_tensor(self.sigma_perp, device=Q_lab.device, dtype=Q_lab.dtype)
        s_perp2 = torch.as_tensor(
            self.sigma_perp if self.sigma_perp2 is None else self.sigma_perp2,
            device=Q_lab.device, dtype=Q_lab.dtype)
        chi2 = (d_par / s_par) ** 2 + (d_t1 / s_perp) ** 2 + (d_t2 / s_perp2) ** 2
        return torch.exp(-0.5 * chi2)

    @property
    def fwhm_transverse(self) -> float:
        """Transverse FWHM in reciprocal-space units (``2*sqrt(2 ln2) * sigma_perp``)."""
        return 2.3548200450309493 * float(self.sigma_perp)

    def rocking_fwhm_deg(self, axis=(1.0, 0.0, 0.0)) -> float:
        """Analytic mosaicity-scan (rocking) FWHM in **degrees** for a perfect crystal.

        A small rock ``delta`` (rad) about ``axis`` displaces ``Q_lab`` transversely
        at rate ``|axis_hat x q_nom|`` (only the ``q`` component perpendicular to the
        rocking axis moves). The Gaussian then has angular FWHM
        ``2*sqrt(2 ln2) * sigma_perp / |axis_hat x q_nom|``. This is the Phase-1
        analytic-limit check; note it depends on which axis is rocked.
        """
        ax = torch.as_tensor(axis, device=self.q_nom.device, dtype=self.q_nom.dtype)
        ax = ax / torch.linalg.vector_norm(ax)
        rate = float(torch.linalg.vector_norm(torch.linalg.cross(ax, self.q_nom)))
        return torch.rad2deg(torch.tensor(self.fwhm_transverse / rate)).item()


def poulsen_resolution_widths(
    q_mag,
    *,
    two_theta_deg: float,
    div_v: float = 0.53e-3,
    div_h: float = 0.53e-3,
    na: float = 0.731e-3,
    eps: float = 1.4e-4,
) -> dict:
    """Principal reciprocal-space resolution widths (Poulsen 2017, Eqs. 58-63).

    The rigorous DFXM resolution element is anisotropic with three distinct principal
    widths in the ``(rock, roll, par)`` frame::

        sigma_rock^2 = (|Q0|^2/4) (div_v^2 + na^2)
        sigma_roll^2 = (|Q0|^2/4 sin^2 theta) (div_h^2 + na^2)
        sigma_par^2  = (|Q0|^2/4) (4 eps^2 + cot^2 theta (div_v^2 + na^2))

    with ``theta = two_theta/2``, ``div_v/div_h`` the incident vertical/horizontal
    divergence (r.m.s.), ``na`` the objective angular acceptance (r.m.s.), ``eps`` the
    r.m.s. energy spread. Returns ``{'sigma_rock', 'sigma_roll', 'sigma_par'}`` in the
    same 1/Angstrom units as :class:`ResolutionFunction`. This is the closed-form
    ground truth against which the (isotropic-``perp``, Gaussian) approximation of
    :class:`ResolutionFunction` should be benchmarked (rock << roll ~ par: a thin plate).
    """
    q = float(q_mag)
    theta = 0.5 * two_theta_deg * torch.pi / 180.0
    s, c = torch.sin(torch.tensor(theta)).item(), torch.cos(torch.tensor(theta)).item()
    ang = div_v ** 2 + na ** 2
    sigma_rock = 0.5 * q * (ang ** 0.5)
    sigma_roll = 0.5 * q / s * ((div_h ** 2 + na ** 2) ** 0.5)
    sigma_par = 0.5 * q * ((4 * eps ** 2 + (c / s) ** 2 * ang) ** 0.5)
    return {"sigma_rock": sigma_rock, "sigma_roll": sigma_roll, "sigma_par": sigma_par}


def aligned_resolution(
    q_nom: torch.Tensor,
    *,
    sigma_par: float = 5e-3,
    sigma_perp: float = 5e-3,
    sigma_perp2: "float | None" = None,
    rock_dir: "torch.Tensor | None" = None,
) -> ResolutionFunction:
    """Build a :class:`ResolutionFunction` centred on ``q_nom`` (lab frame).

    ``q_nom`` is typically ``G_ref @ G0_sample`` — the reference reflection placed
    at the aligned goniometer setting (see :func:`midas_dfxm.scan.reference_q_nom`).
    Pass ``sigma_perp2`` (and ``rock_dir`` to orient the axes) for a fully anisotropic
    transverse resolution; omit them for the isotropic-``perp`` default.
    """
    return ResolutionFunction(q_nom=q_nom, sigma_par=sigma_par, sigma_perp=sigma_perp,
                              sigma_perp2=sigma_perp2, rock_dir=rock_dir)


def poulsen_aligned_resolution(
    q_nom: torch.Tensor,
    *,
    two_theta_deg: float,
    rock_dir: "torch.Tensor | None" = None,
    **poulsen_kwargs,
) -> ResolutionFunction:
    """Fully anisotropic DFXM resolution in one call: the Poulsen widths
    (``sigma_rock``, ``sigma_roll``, ``sigma_par``) wired into an anisotropic
    :class:`ResolutionFunction` (``sigma_perp = rock`` on ``e_t1``, ``sigma_perp2 =
    roll`` on ``e_t2``). ``rock_dir`` (the in-scattering-plane direction) orients the
    axes so ``rock`` and ``roll`` land on the physical directions. Extra kwargs
    (``div_v``, ``div_h``, ``na``, ``eps``) pass to :func:`poulsen_resolution_widths`.
    """
    q_mag = float(torch.linalg.vector_norm(q_nom))
    w = poulsen_resolution_widths(q_mag, two_theta_deg=two_theta_deg, **poulsen_kwargs)
    return ResolutionFunction(q_nom=q_nom, sigma_par=w["sigma_par"],
                              sigma_perp=w["sigma_rock"], sigma_perp2=w["sigma_roll"],
                              rock_dir=rock_dir)
