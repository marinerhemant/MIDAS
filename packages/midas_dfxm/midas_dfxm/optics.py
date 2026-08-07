"""Magnifying objective optics: sample voxel -> detector pixel projection.

Phase 1 of ``implementation_plan.md`` — the NEW real-space imaging geometry.

The objective images the illuminated sample volume along the diffracted-beam
optical axis ``k_out`` with magnification ``M`` onto a detector perpendicular to
``k_out``. A sample voxel at lab position ``r`` maps to detector coordinates by
projecting ``r`` onto the plane transverse to ``k_out`` (an orthonormal detector
basis ``(u, v)``), scaling by ``M``, and converting to pixels. The imaging is an
*inclined* projection because the sample plane is tilted relative to the optical
axis — captured here by the choice of ``k_out`` (from ``2*theta``) rather than by
an ad-hoc tilt.

Differentiable in positions, magnification, and ``k_out`` direction; device/dtype
portable. Detector distortion, when needed, composes via ``midas_distortion``
(single source of truth) — omitted from the ideal Phase-1 projection.

Units: positions in micrometers; pixel size in micrometers; pixels dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def diffracted_beam_direction(two_theta_deg: float, *, device=None, dtype=torch.float64) -> torch.Tensor:
    """Unit ``k_out`` for a reflection at ``2*theta`` in the vertical diffraction plane.

    Incident beam along lab ``x``; diffraction plane is ``x``-``z`` (vertical).
    ``k_out = (cos 2theta) xhat + (sin 2theta) zhat``.

    This assumes the diffraction plane is vertical, which is the DFXM case (the
    objective is placed there). Techniques that image *one chosen grain's*
    reflection -- topotomography, DCT -- get whatever direction crystallography
    gives them, and must pass an explicit ``k_out`` to
    :class:`ObjectiveOptics` instead. See :meth:`ObjectiveOptics.from_k_out`.
    """
    tt = torch.deg2rad(torch.as_tensor(two_theta_deg, device=device, dtype=dtype))
    return torch.stack([torch.cos(tt), torch.zeros_like(tt), torch.sin(tt)])


def two_theta_from_k_out(k_out) -> float:
    """Scattering angle ``2*theta`` (degrees) of an arbitrary ``k_out``.

    ``2*theta`` is by definition the angle between the incident beam (lab ``x``)
    and ``k_out``, so this inverts :func:`diffracted_beam_direction` for any
    direction, not only vertical-plane ones.
    """
    k = torch.as_tensor(k_out)
    if not torch.is_floating_point(k):
        k = k.to(torch.float64)
    n = torch.linalg.vector_norm(k.detach())
    if float(n) == 0.0:
        raise ValueError("k_out must be a non-zero vector")
    c = (k.detach()[0] / n).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(c)))


def detector_basis(k_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Orthonormal detector axes ``(u, v)`` spanning the plane transverse to ``k_out``.

    ``u`` is horizontal (in the lab ``x``-``y`` sense), ``v`` completes a
    right-handed set. Returns two ``(3,)`` unit vectors.
    """
    k = k_out / torch.linalg.vector_norm(k_out)
    up = torch.tensor([0.0, 0.0, 1.0], device=k.device, dtype=k.dtype)
    if torch.abs(torch.dot(k, up)) > 0.9:
        up = torch.tensor([0.0, 1.0, 0.0], device=k.device, dtype=k.dtype)
    u = torch.linalg.cross(up, k)
    u = u / torch.linalg.vector_norm(u)
    v = torch.linalg.cross(k, u)
    return u, v


@dataclass
class ObjectiveOptics:
    """Magnifying objective imaging model.

    Attributes
    ----------
    two_theta_deg : float
        Bragg ``2*theta`` of the imaged reflection (sets the optical axis ``k_out``).
    magnification : float
        Transverse magnification ``M`` of the objective.
    pixel_um : float
        Detector pixel size in micrometers.
    detector_shape : tuple[int, int]
        ``(n_u, n_v)`` detector dimensions in pixels.
    center_px : tuple[float, float]
        Pixel coordinate of the optical axis (defaults to detector centre).
    k_out : tensor (3,), optional
        Explicit optical axis in the lab frame. When ``None`` (the default) the
        axis is derived from ``two_theta_deg`` and lies in the vertical
        ``x``-``z`` diffraction plane, which is the DFXM geometry. Supply it when
        the imaged reflection is *not* in that plane -- topotomography and DCT
        image one chosen grain's ``G``, which points wherever crystallography
        puts it. Need not be normalised. Gradients flow through it.

        It must agree with ``two_theta_deg`` (checked at construction, since
        ``two_theta_deg`` remains the scattering angle other code reads);
        :meth:`from_k_out` derives the pair consistently for you.

    Notes
    -----
    Setting ``M = 1`` with an explicit ``k_out`` gives the unmagnified parallel
    projection along an arbitrary diffracted beam -- i.e. the TT/DCT topograph
    geometry -- with no other change to this class.
    """

    two_theta_deg: float
    magnification: float = 10.0
    pixel_um: float = 1.0
    detector_shape: tuple[int, int] = (256, 256)
    center_px: tuple[float, float] | None = None
    k_out: torch.Tensor | None = None
    NA: float | None = None
    wavelength_A: float | None = None

    def __post_init__(self):
        if self.k_out is None:
            return
        # An inconsistent (k_out, two_theta_deg) pair is a silent physics bug:
        # the projection would use one direction while anything reading
        # two_theta_deg (resolution widths, Lorentz factors) uses another.
        # Check once, here, rather than on every project() call.
        tt = two_theta_from_k_out(self.k_out)
        if abs(tt - float(self.two_theta_deg)) > 1e-6:
            raise ValueError(
                f"k_out is at 2*theta = {tt:.9g} deg but two_theta_deg = "
                f"{float(self.two_theta_deg):.9g} deg. They must agree -- "
                "use ObjectiveOptics.from_k_out(k_out, ...) to derive both."
            )

    @classmethod
    def from_k_out(cls, k_out, **kwargs) -> "ObjectiveOptics":
        """Build from an explicit optical axis, deriving ``two_theta_deg`` from it.

        The TT/DCT entry point: pass ``k_out = k_in + G`` for the grain and
        reflection being imaged, plus ``magnification=1.0`` for an unmagnified
        parallel projection. Remaining keyword arguments are as for the
        constructor.

        ``two_theta_deg`` is stored as a plain float, so gradients w.r.t. the
        scattering angle do not flow through *it* -- but the projection uses
        ``k_out`` directly, so gradients w.r.t. the beam direction do.
        """
        if "two_theta_deg" in kwargs:
            raise TypeError(
                "from_k_out derives two_theta_deg from k_out; do not pass it as well"
            )
        return cls(two_theta_deg=two_theta_from_k_out(k_out), k_out=k_out, **kwargs)

    def optical_axis(self, *, device=None, dtype=torch.float64) -> torch.Tensor:
        """Unit optical axis ``(3,)``: explicit ``k_out`` if set, else from ``2*theta``."""
        if self.k_out is None:
            return diffracted_beam_direction(self.two_theta_deg, device=device, dtype=dtype)
        k = self.k_out
        if not isinstance(k, torch.Tensor):
            k = torch.as_tensor(k, device=device, dtype=dtype)
        else:
            k = k.to(device=device, dtype=dtype)
        return k / torch.linalg.vector_norm(k)

    def project(self, positions_lab: torch.Tensor) -> torch.Tensor:
        """Project lab-frame voxel positions ``(N, 3)`` to pixel coords ``(N, 2)``.

        Returns fractional ``(u_px, v_px)``. Differentiable in positions, ``M``,
        and (when supplied) ``k_out``.
        """
        k_out = self.optical_axis(device=positions_lab.device, dtype=positions_lab.dtype)
        u, v = detector_basis(k_out)
        cu, cv = self._center(positions_lab)
        pu = self.magnification * (positions_lab @ u) / self.pixel_um + cu
        pv = self.magnification * (positions_lab @ v) / self.pixel_um + cv
        return torch.stack([pu, pv], dim=-1)

    def _center(self, ref: torch.Tensor) -> tuple[float, float]:
        if self.center_px is not None:
            return self.center_px
        nu, nv = self.detector_shape
        return (nu - 1) / 2.0, (nv - 1) / 2.0

    def psf(self, coeffs=None, *, defocus=0.0, grid_size=128, extent=1.4,
            apodization=1.5, device=None, dtype=torch.float64) -> torch.Tensor:
        """Objective point-spread function sampled at the DETECTOR pixel grid.

        Builds the wave-optics pupil PSF (``aberration.aberrated_psf`` --
        ``|FFT(pupil)|^2`` for a Zernike-aberrated, Gaussian-apodized aperture),
        then resamples it from its native diffraction scale onto this detector's
        object-space pixel so it can be convolved with a rendered image.

        Sampling: the pupil grid spans the unit-disk aperture, so the native PSF
        pixel is ``lambda / (2 * extent * NA)`` in object space; one detector pixel
        images ``pixel_um / M`` of object space. The PSF is scaled by their ratio.
        A diffraction spot finer than a detector pixel collapses to a delta (no
        blur), which is physically correct.

        ``coeffs`` is a dict/tensor of Zernike amplitudes (rad); ``None`` gives the
        unaberrated diffraction-limited PSF. Requires ``NA`` and ``wavelength_A``
        to be set. Differentiable in ``coeffs`` and ``defocus``.
        """
        if self.NA is None or self.wavelength_A is None:
            raise ValueError("ObjectiveOptics.psf requires NA and wavelength_A to be set")
        from .aberration import aberrated_psf
        psf = aberrated_psf(coeffs if coeffs is not None else {}, defocus=defocus,
                            grid_size=grid_size, extent=extent, apodization=apodization,
                            device=device, dtype=dtype)
        lam_um = float(self.wavelength_A) * 1e-4
        dx_native = lam_um / (2.0 * extent * float(self.NA))   # object-space um/PSF-pixel
        dx_image = float(self.pixel_um) / float(self.magnification)  # object-space um/img-pixel
        scale = dx_native / dx_image
        n = max(1, int(round(grid_size * scale)))
        if n != grid_size:
            psf = torch.nn.functional.interpolate(
                psf[None, None], size=(n, n), mode="bilinear", align_corners=False
            )[0, 0].clamp_min(0.0)
        # keep the PSF no larger than the detector (a concentrated PSF's cropped
        # tails carry negligible mass); convolve_psf embeds a smaller PSF centred.
        cap = min(self.detector_shape)
        if psf.shape[-1] > cap:
            off = (psf.shape[-1] - cap) // 2
            psf = psf[off:off + cap, off:off + cap]
        return psf / psf.sum()

    def _psf_scale(self, extent, grid_size):
        """Interp size to map the native pupil-PSF pixel onto the detector pixel."""
        lam_um = float(self.wavelength_A) * 1e-4
        dx_native = lam_um / (2.0 * extent * float(self.NA))
        dx_image = float(self.pixel_um) / float(self.magnification)
        return max(1, int(round(grid_size * (dx_native / dx_image))))

    def amplitude_psf(self, coeffs=None, *, defocus=0.0, grid_size=128, extent=1.4,
                      apodization=1.5, device=None, dtype=torch.float64) -> torch.Tensor:
        """Complex amplitude PSF ``h`` (for coherent imaging), detector-sampled.

        Like :meth:`psf` but returns the field-amplitude PSF from
        ``coherence.coherent_psf`` (normalized ``sum |h|^2 = 1``) instead of the
        intensity PSF, for use with :func:`midas_dfxm.dfxm_image_wave`.
        """
        if self.NA is None or self.wavelength_A is None:
            raise ValueError("ObjectiveOptics.amplitude_psf requires NA and wavelength_A")
        from .coherence import coherent_psf
        h = coherent_psf(coeffs if coeffs is not None else {}, defocus=defocus,
                         grid_size=grid_size, extent=extent, apodization=apodization,
                         device=device, dtype=dtype)
        n = self._psf_scale(extent, grid_size)
        if n != grid_size:
            hr = torch.nn.functional.interpolate(h.real[None, None], size=(n, n),
                                                 mode="bilinear", align_corners=False)[0, 0]
            hi = torch.nn.functional.interpolate(h.imag[None, None], size=(n, n),
                                                 mode="bilinear", align_corners=False)[0, 0]
            h = torch.complex(hr, hi)
        cap = min(self.detector_shape)
        if h.shape[-1] > cap:
            off = (h.shape[-1] - cap) // 2
            h = h[off:off + cap, off:off + cap]
        return h / torch.sqrt((h.abs() ** 2).sum())

    def render(self, positions_lab: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
        """Accumulate per-voxel ``intensity`` onto the detector via bilinear splat.

        Returns an image ``(n_u, n_v)``. Differentiable in ``intensity`` and
        positions (bilinear weights carry gradients). Voxels off the detector are
        dropped.
        """
        px = self.project(positions_lab)
        nu, nv = self.detector_shape
        img = torch.zeros(nu, nv, device=positions_lab.device, dtype=intensity.dtype)
        u = px[:, 0]
        v = px[:, 1]
        u0 = torch.floor(u).long()
        v0 = torch.floor(v).long()
        du = u - u0
        dv = v - v0
        for iu, ju, wu in ((u0, 0, 1 - du), (u0 + 1, 1, du)):
            for iv, jv, wv in ((v0, 0, 1 - dv), (v0 + 1, 1, dv)):
                inb = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
                if not bool(inb.any()):
                    continue
                w = (wu * wv * intensity)[inb]
                flat = (iu[inb] * nv + iv[inb])
                img.view(-1).index_add_(0, flat, w)
        return img
