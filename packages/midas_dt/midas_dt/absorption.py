"""Self-absorption: the correction, and where it can honestly be applied.

The beam is attenuated on the way in to a voxel and the diffracted beam is
attenuated on the way out. A voxel at the far side of the sample therefore
contributes less signal than an identical voxel at the near side, and an
uncorrected map reads as though the sample were richer at its surface. This is
the first entry in the known-limits ledger, and it is the one that most often
turns a phase-fraction map from qualitative into wrong.

The measured projection is

    P(omega, t, R) = sum over voxels v on the ray of  I_v(R) * A(v, omega)

with ``A = exp(-(mu-path in) - (mu-path out))``. The attenuation sits **inside**
the sum, weighting each voxel differently, so absorption is not a property of
the ray -- which is precisely why it biases along the ray and why no rescaling
of the sinogram can undo it.

That gives two honest options and one dishonest one:

**Exact (branch C).** Fold ``A`` into the forward operator and solve with it in
place: :func:`attenuated_projection_matrix`, passed to
``run_direct(attenuation=...)``. The model then describes the measurement, so
nothing needs undoing. This is the real payoff of having a forward model.

**Approximate (branches A and B).** Reconstruct as usual, then divide each
voxel by the attenuation it suffered *on average over the rotation*:
:func:`correct_reconstruction`. This removes the bulk near/far bias. It does
NOT undo the distortion absorption already caused in the reconstruction
itself, because the omega-dependence of ``A`` violates the Radon model the
reconstruction assumed. Results are marked approximate and the limits ledger
keeps saying so.

**Dishonest.** Dividing the sinogram by a per-ray factor. It looks like a
correction and cannot be one: a single number per ray cannot invert a
position-dependent weighting inside the sum. Not provided.

Everything here needs ``mu`` at the beam energy. The usual source is the
transmission channel recorded alongside the diffraction -- an absorption CT
reconstruction, in units of inverse pixels. :func:`mu_from_transmission` does
that through ``midas-tomo``. A uniform ``mu`` over a sample mask
(:func:`uniform_mu`) is a cruder alternative that still captures most of the
near/far bias for a roughly homogeneous sample.
"""

from __future__ import annotations

import logging

import numpy as np

from .recon import Reconstruction

__all__ = [
    "attenuation_factors",
    "attenuated_projection_matrix",
    "correct_reconstruction",
    "mu_from_transmission",
    "uniform_mu",
    "in_plane_deflection_deg",
]

log = logging.getLogger(__name__)


def uniform_mu(mask, mu_per_pixel: float) -> np.ndarray:
    """A constant attenuation coefficient inside *mask*, zero outside.

    ``mu_per_pixel`` is the linear attenuation coefficient in **inverse
    pixels** of the reconstruction grid, not inverse cm: everything downstream
    integrates over voxel counts. Convert with ``mu_cm * pixel_size_cm``.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"mask must be a square 2-D grid, got {m.shape}")
    if mu_per_pixel < 0:
        raise ValueError(f"mu must be >= 0, got {mu_per_pixel}")
    return np.where(m, float(mu_per_pixel), 0.0)


def mu_from_transmission(
    transmission_sinogram,
    omega_deg,
    *,
    pixel_size: float = 1.0,
    **recon_kw,
) -> np.ndarray:
    """Reconstruct ``mu`` from a transmission sinogram, via ``midas-tomo``.

    *transmission_sinogram* is ``(n_omega, n_translations)`` of
    ``I / I_0`` -- already normalised by the incident flux. The log is taken
    here, so pass transmittance and not its logarithm.

    Returns ``mu`` in inverse pixels on the reconstruction grid, ready for
    :func:`attenuation_factors`. Negative values (noise around zero
    attenuation in air) are clipped to zero: a negative mu is unphysical and
    would *amplify* rather than attenuate downstream.
    """
    from midas_tomo import run_tomo_from_sinos

    t = np.asarray(transmission_sinogram, dtype=np.float64)
    if t.ndim != 2:
        raise ValueError(f"transmission must be (n_omega, n_trans), got {t.shape}")
    if np.nanmax(t) > 10:
        raise ValueError(
            "transmission values exceed 10, which suggests raw counts rather "
            "than I/I_0. Normalise by the incident flux first."
        )
    # -log(I/I0) is the projection of mu. Floor the transmittance: a zero or
    # negative reading (a dead pixel, or a beamstop) would give an infinite
    # path integral and poison the whole reconstruction.
    proj = -np.log(np.clip(t, 1e-6, None))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        cube = run_tomo_from_sinos(
            proj[None, :, :].astype(np.float32), td, np.asarray(omega_deg),
            do_log=False, **recon_kw)
    mu = np.asarray(cube[0, 0], dtype=np.float64) / max(float(pixel_size), 1e-12)
    n_neg = int((mu < 0).sum())
    if n_neg:
        log.info("clipped %d negative mu voxels (%.1f%%) to zero",
                 n_neg, 100.0 * n_neg / mu.size)
    return np.clip(mu, 0.0, None)


def in_plane_deflection_deg(two_theta_deg: float, eta_deg: float) -> float:
    """In-plane component of the scattering direction, in degrees.

    A slice reconstruction is 2-D, so only the in-plane part of the diffracted
    ray can be modelled. ``eta = 0`` or ``180`` puts the whole of ``2theta`` in
    the plane; ``eta = +-90`` sends the ray out of the slice entirely, where a
    2-D attenuation model has nothing to say.
    """
    return float(np.degrees(np.arctan2(
        np.sin(np.deg2rad(two_theta_deg)) * np.cos(np.deg2rad(eta_deg)),
        np.cos(np.deg2rad(two_theta_deg)),
    )))


def _path_integral(mu: np.ndarray, angle_deg: float) -> np.ndarray:
    """Cumulative ``mu`` upstream of every voxel for a beam at *angle_deg*.

    Rotate so the beam runs along +row, cumulative-sum, rotate back. The
    cumulative sum is exclusive -- a voxel is not attenuated by itself -- and
    the half-voxel of self-attenuation that a rigorous treatment would include
    is below the accuracy of everything else here.
    """
    from scipy.ndimage import rotate

    rot = rotate(mu, angle_deg, reshape=False, order=1, mode="constant", cval=0.0)
    acc = np.cumsum(rot, axis=0) - rot                       # exclusive
    back = rotate(acc, -angle_deg, reshape=False, order=1, mode="constant",
                  cval=0.0)
    return np.clip(back, 0.0, None)


def attenuation_factors(
    mu: np.ndarray,
    omega_deg,
    *,
    two_theta_deg: float = 0.0,
    eta_deg: float = 0.0,
    small_angle: bool | None = None,
) -> np.ndarray:
    """``A(voxel, omega)`` in ``(0, 1]``, shape ``(n_omega, size, size)``.

    The product of the incoming and outgoing attenuation for every voxel at
    every rotation.

    Parameters
    ----------
    mu : (size, size) array
        Linear attenuation coefficient in inverse pixels.
    two_theta_deg, eta_deg : float
        Scattering geometry of the channel being corrected. The outgoing ray
        leaves the voxel deflected by the in-plane part of ``2theta`` (see
        :func:`in_plane_deflection_deg`).
    small_angle : bool, optional
        Treat the outgoing ray as parallel to the incoming one. Defaults to
        True when the in-plane deflection is under 5 degrees, which is the
        normal case at synchrotron energies -- at 90 keV with rings at
        100-500 px and a metre of sample-detector distance, ``2theta`` is
        roughly 0.5-2.7 degrees and the approximation costs almost nothing.
        It is NOT safe at low energy or for high-angle rings, so it is
        computed rather than assumed, and the exact path is used when the
        deflection is large.

    Notes
    -----
    Returns 1.0 everywhere if ``mu`` is all zeros, which makes "no absorption
    information" a no-op rather than an error.
    """
    mu = np.asarray(mu, dtype=np.float64)
    if mu.ndim != 2 or mu.shape[0] != mu.shape[1]:
        raise ValueError(f"mu must be a square 2-D grid, got {mu.shape}")
    if np.any(mu < 0):
        raise ValueError("mu has negative entries; attenuation cannot amplify")

    omega = np.asarray(omega_deg, dtype=np.float64).ravel()
    if not mu.any():
        return np.ones((omega.size, *mu.shape))

    defl = in_plane_deflection_deg(two_theta_deg, eta_deg)
    if small_angle is None:
        small_angle = abs(defl) < 5.0
    if abs(np.cos(np.deg2rad(eta_deg))) < 0.2 and abs(two_theta_deg) > 1e-9:
        log.warning(
            "eta=%.1f deg puts the diffracted ray nearly out of the slice; a "
            "2-D attenuation model cannot represent its exit path and the "
            "outgoing term will be underestimated", eta_deg)

    out = np.empty((omega.size, *mu.shape))
    for i, om in enumerate(omega):
        # In: what the incident beam already crossed to reach this voxel.
        # Out: what the diffracted beam still has to cross to escape -- the
        # DOWNSTREAM integral, not the upstream one. Using upstream for both
        # would say a voxel at the entry face escapes for free, which is
        # backwards and would bias the correction the wrong way.
        l_in = _path_integral(mu, om)
        l_out = _downstream(mu, om if small_angle else om + defl)
        out[i] = np.exp(-(l_in + l_out))
    np.clip(out, 1e-12, 1.0, out=out)
    return out


def _downstream(mu: np.ndarray, angle_deg: float) -> np.ndarray:
    """Cumulative ``mu`` DOWNSTREAM of every voxel along *angle_deg*.

    The outgoing leg: what the diffracted beam still has to cross to escape.
    Computed as the reversed exclusive cumulative sum, not as
    ``total - upstream``, so it stays consistent with the same interpolation.
    """
    from scipy.ndimage import rotate

    rot = rotate(mu, angle_deg, reshape=False, order=1, mode="constant", cval=0.0)
    acc = np.cumsum(rot[::-1], axis=0)[::-1] - rot           # exclusive
    back = rotate(acc, -angle_deg, reshape=False, order=1, mode="constant",
                  cval=0.0)
    return np.clip(back, 0.0, None)


def attenuated_projection_matrix(
    size: int,
    angles_deg,
    n_det: int,
    factors: np.ndarray,
    *,
    dtype=None,
    device=None,
):
    """The Radon operator with per-(voxel, omega) attenuation folded in.

    This is the exact route: the operator now describes what the instrument
    actually measures, so ``run_direct`` solves the right inverse problem
    rather than a corrected version of the wrong one.

    *factors* is ``(n_omega, size, size)`` from :func:`attenuation_factors`.
    """
    import torch

    from .direct import projection_matrix

    A = projection_matrix(size, angles_deg, n_det, dtype=dtype, device=device)
    f = np.asarray(factors, dtype=np.float64)
    n_ang = len(np.asarray(angles_deg).ravel())
    if f.shape != (n_ang, size, size):
        raise ValueError(
            f"factors must be (n_omega, size, size) = "
            f"({n_ang}, {size}, {size}); got {f.shape}")

    idx = A.indices()
    vals = A.values()
    # Row index encodes (angle, detector); scale each entry by the attenuation
    # of ITS voxel at ITS angle. Doing it on the COO values keeps the operator
    # sparse and keeps the adjoint exact -- torch transposes it for us.
    ang_of_row = torch.div(idx[0], n_det, rounding_mode="floor")
    flat = torch.as_tensor(f.reshape(n_ang, -1), dtype=vals.dtype,
                           device=vals.device)
    scaled = vals * flat[ang_of_row, idx[1]]
    return torch.sparse_coo_tensor(idx, scaled, A.shape, dtype=vals.dtype,
                                   device=vals.device).coalesce()


def correct_reconstruction(
    recon: Reconstruction,
    factors: np.ndarray,
    *,
    floor: float = 1e-3,
) -> Reconstruction:
    """Divide each voxel by its rotation-averaged attenuation. APPROXIMATE.

    For branches A and B, which reconstruct before they know about absorption.
    It removes the bulk near/far bias and does not pretend to more: the
    reconstruction it corrects was produced by an algorithm that assumed the
    Radon model, which absorption violates, so the residual distortion stays.
    Branch C with :func:`attenuated_projection_matrix` is the exact route.

    *floor* bounds the amplification. A voxel deep inside a strongly absorbing
    sample can have a mean attenuation near zero, and dividing by it turns
    noise into a bright artefact that looks like a real feature; anything below
    the floor is left as NaN instead, which propagates visibly.
    """
    f = np.asarray(factors, dtype=np.float64)
    if f.ndim != 3:
        raise ValueError(f"factors must be (n_omega, size, size); got {f.shape}")
    mean_a = f.mean(axis=0)
    if mean_a.shape != recon.intensity.shape[-2:]:
        raise ValueError(
            f"factors grid {mean_a.shape} does not match the reconstruction "
            f"{recon.intensity.shape[-2:]}")

    safe = np.where(mean_a >= floor, mean_a, np.nan)
    n_dropped = int(np.isnan(safe).sum())
    if n_dropped:
        log.warning(
            "%d of %d voxels have mean attenuation below %.3g and were set to "
            "NaN rather than amplified by more than %.0fx",
            n_dropped, safe.size, floor, 1.0 / floor)

    limits = recon.limits
    for setter in ("with_note", "note", "add_note"):
        if hasattr(limits, setter):
            try:
                limits = getattr(limits, setter)(
                    "self-absorption corrected by the rotation-averaged factor "
                    "(approximate: the reconstruction itself assumed the Radon "
                    "model, which absorption violates)")
            except Exception:                        # pragma: no cover
                pass
            break

    return Reconstruction(
        intensity=recon.intensity / safe[None, :, :],
        variance=(None if recon.variance is None
                  else recon.variance / (safe[None, :, :] ** 2)),
        bin_shape=recon.bin_shape, channel=recon.channel, limits=limits,
        sign_applied=recon.sign_applied,
    )
